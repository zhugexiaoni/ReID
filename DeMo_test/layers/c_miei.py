from typing import Dict, Tuple

import torch
import torch.nn as nn


def _kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL( softmax(p) || softmax(q) ) computed per-sample.

    Args:
        logits_p: [B, C]
        logits_q: [B, C]

    Returns:
        kl: [B]
    """
    log_p = torch.log_softmax(logits_p, dim=1)
    log_q = torch.log_softmax(logits_q, dim=1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=1)


def _ted_margin_from_logits(logits_full: torch.Tensor, logits_drop: torch.Tensor) -> torch.Tensor:
    """Target Evidence Drop (margin version) computed per-sample.

    Uses the *predicted* class from logits_full as target y.

    TED_margin = ReLU( margin_full(y) - margin_drop(y) )
    where margin(y) = logit_y - max_{c!=y} logit_c.

    Args:
        logits_full: [B, C]
        logits_drop: [B, C]

    Returns:
        ted: [B]
    """
    assert logits_full.ndim == 2 and logits_drop.ndim == 2
    B, C = logits_full.shape

    y = logits_full.argmax(dim=1)  # [B]
    y_col = y.view(B, 1)

    # s = logit_y
    s_full = logits_full.gather(1, y_col).squeeze(1)
    s_drop = logits_drop.gather(1, y_col).squeeze(1)

    # s2 = max_{c!=y} logit_c
    one_hot = torch.zeros((B, C), device=logits_full.device, dtype=torch.bool)
    one_hot.scatter_(1, y_col, True)

    # NOTE(fp16): use a finite, dtype-safe negative fill value to avoid fp16 overflow.
    # -1e9 cannot be represented in float16.
    neg_fill = torch.finfo(logits_full.dtype).min
    s2_full = logits_full.masked_fill(one_hot, neg_fill).max(dim=1).values
    s2_drop = logits_drop.masked_fill(one_hot, neg_fill).max(dim=1).values

    margin_full = s_full - s2_full
    margin_drop = s_drop - s2_drop
    return torch.relu(margin_full - margin_drop)


class CounterfactualSubstitutePlugin(nn.Module):
    """C-MIEI: Counterfactual influence -> feature substitution intervention.

    Supports:
    - batch-level intervention (previous behavior)
    - sample-level intervention with cap p_max and warmup_epochs

    This is *feature-level* intervention (not gradient scaling):
    - Estimate per-modality counterfactual influence CI via KL divergence between
      normal fused logits and logits when dropping one modality feature.
    - If one modality dominates, substitute that modality feature with
      batch-prototype + small noise.

    Implementation notes:
    - No extra modality classifier head required.
    - Only needs access to fused head callable fused_logits_fn(fr, fn, ft)->logits.
    """

    def __init__(
        self,
        k: int = 3,
        sigma: float = 0.05,
        abs_thr: float = 0.03,
        rel_thr: float = 1.25,
        sample_level: bool = False,
        p_max: float = 0.5,
        warmup_epochs: int = 5,
        drop_mode: str = 'zero',
        eps: float = 1e-12,
    ):
        super().__init__()
        assert k >= 1
        assert sigma >= 0
        assert drop_mode in {'zero', 'mean'}
        assert 0.0 < p_max <= 1.0
        assert warmup_epochs >= 0

        self.k = int(k)
        self.sigma = float(sigma)
        self.abs_thr = float(abs_thr)
        self.rel_thr = float(rel_thr)
        self.sample_level = bool(sample_level)
        self.p_max = float(p_max)
        self.warmup_epochs = int(warmup_epochs)
        self.drop_mode = drop_mode
        self.eps = float(eps)

        self._step = 0
        self._epoch = 0

        # last cached stats (for logging when not estimating)
        self._last_ci_mean = {'r': 0.0, 'n': 0.0, 't': 0.0}
        self._last_ratio = 0.0
        self._last_chosen = 'none'
        self._last_hist = {'r': 0.0, 'n': 0.0, 't': 0.0}

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    @staticmethod
    def _substitute(feat: torch.Tensor, sigma: float) -> torch.Tensor:
        """Substitute by batch prototype + noise."""
        mu = feat.mean(dim=0, keepdim=True)
        if sigma <= 0:
            return mu.expand_as(feat)
        return mu.expand_as(feat) + torch.randn_like(feat) * sigma

    def _drop(self, feat: torch.Tensor) -> torch.Tensor:
        if self.drop_mode == 'zero':
            return torch.zeros_like(feat)
        return feat.mean(dim=0, keepdim=True).expand_as(feat)

    @torch.no_grad()
    def _estimate_ci_per_sample(
        self,
        fr: torch.Tensor,
        fn: torch.Tensor,
        ft: torch.Tensor,
        fused_logits_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-sample CI-like scores for each modality: (r,n,t) each [B].

        NOTE:
        - Historically we used KL( softmax(z) || softmax(z_drop) ).
        - For sample-level triggering KL often becomes tiny due to softmax saturation.
        - We now use TED_margin: ReLU( (z_y - z_2) - (z_drop_y - z_drop_2) ),
          where y is argmax class of full logits z.
        """
        # IMPORTANT: counterfactual passes must NOT update BN running stats.
        # We pass bn_update=False and rely on the model-side implementation.
        z = fused_logits_fn(fr, fn, ft, bn_update=False)
        z_r = fused_logits_fn(self._drop(fr), fn, ft, bn_update=False)
        z_n = fused_logits_fn(fr, self._drop(fn), ft, bn_update=False)
        z_t = fused_logits_fn(fr, fn, self._drop(ft), bn_update=False)
        return _ted_margin_from_logits(z, z_r), _ted_margin_from_logits(z, z_n), _ted_margin_from_logits(z, z_t)

    @torch.no_grad()
    def _choose_and_mask(self, ci_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Choose dominant modality per sample and return chosen_idx and mask.

        Args:
            ci_stack: [B, 3] (r,n,t)

        Returns:
            chosen_idx: [B] in {0,1,2}
            intervene_mask: [B] bool
        """
        # top2 per sample
        top2_vals, top2_idx = torch.topk(ci_stack, k=2, dim=1, largest=True, sorted=True)
        v1 = top2_vals[:, 0]
        v2 = top2_vals[:, 1]
        chosen_idx = top2_idx[:, 0]

        cond = (v1 > self.abs_thr) & (v1 / (v2 + self.eps) > self.rel_thr)
        return chosen_idx, cond

    def forward(
        self,
        fr: torch.Tensor,
        fn: torch.Tensor,
        ft: torch.Tensor,
        fused_logits_fn,
        enable: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        self._step += 1

        # default: reuse cached numbers
        ci_r_mean, ci_n_mean, ci_t_mean = (
            self._last_ci_mean['r'],
            self._last_ci_mean['n'],
            self._last_ci_mean['t'],
        )
        ratio = self._last_ratio
        chosen = self._last_chosen
        hist = dict(self._last_hist)

        # only estimate every k steps, and skip estimation entirely during warmup
        # so warmup training is as close as possible to baseline behavior.
        do_estimate = enable and (self._epoch >= self.warmup_epochs) and (self._step % self.k == 0)

        if do_estimate:
            ci_r, ci_n, ci_t = self._estimate_ci_per_sample(fr, fn, ft, fused_logits_fn)
            ci_stack = torch.stack([ci_r, ci_n, ci_t], dim=1)  # [B,3]

            ci_r_mean = float(ci_r.mean().item())
            ci_n_mean = float(ci_n.mean().item())
            ci_t_mean = float(ci_t.mean().item())
            self._last_ci_mean = {'r': ci_r_mean, 'n': ci_n_mean, 't': ci_t_mean}

            if not self.sample_level:
                # batch-level choose
                cis = {'r': ci_r_mean, 'n': ci_n_mean, 't': ci_t_mean}
                sorted_items = sorted(cis.items(), key=lambda kv: kv[1], reverse=True)
                (m1, v1), (m2, v2) = sorted_items[0], sorted_items[1]
                chosen = 'none'
                if (v1 > self.abs_thr) and (v1 / (v2 + self.eps) > self.rel_thr):
                    chosen = m1

                intervened = False
                if enable and (self._epoch >= self.warmup_epochs) and chosen in {'r', 'n', 't'}:
                    intervened = True
                    if chosen == 'r':
                        fr = self._substitute(fr, self.sigma)
                    elif chosen == 'n':
                        fn = self._substitute(fn, self.sigma)
                    else:
                        ft = self._substitute(ft, self.sigma)

                ratio = 1.0 if intervened else 0.0
                hist = {'r': 1.0 if chosen == 'r' else 0.0,
                        'n': 1.0 if chosen == 'n' else 0.0,
                        't': 1.0 if chosen == 't' else 0.0}

            else:
                # sample-level choose + cap
                chosen_idx, mask = self._choose_and_mask(ci_stack)

                # cap to top p_max by dominant CI value
                B = ci_stack.size(0)
                cap = max(1, int(round(self.p_max * B)))
                dom_val = ci_stack.gather(1, chosen_idx.view(-1, 1)).squeeze(1)  # [B]

                # keep only masked samples, then take top-k by dom_val
                valid_idx = torch.nonzero(mask, as_tuple=False).view(-1)
                if valid_idx.numel() > cap:
                    # select top cap among valid
                    topk = torch.topk(dom_val[valid_idx], k=cap, largest=True).indices
                    keep_idx = valid_idx[topk]
                    new_mask = torch.zeros_like(mask)
                    new_mask[keep_idx] = True
                    mask = new_mask

                # apply only after warmup
                if enable and (self._epoch >= self.warmup_epochs) and mask.any():
                    # modality masks
                    mask_r = mask & (chosen_idx == 0)
                    mask_n = mask & (chosen_idx == 1)
                    mask_t = mask & (chosen_idx == 2)

                    if mask_r.any():
                        fr = fr.clone()
                        fr[mask_r] = self._substitute(fr[mask_r], self.sigma)
                    if mask_n.any():
                        fn = fn.clone()
                        fn[mask_n] = self._substitute(fn[mask_n], self.sigma)
                    if mask_t.any():
                        ft = ft.clone()
                        ft[mask_t] = self._substitute(ft[mask_t], self.sigma)

                ratio = float(mask.float().mean().item())

                # chosen histogram (over all samples, not only masked)
                hist = {
                    'r': float((chosen_idx == 0).float().mean().item()),
                    'n': float((chosen_idx == 1).float().mean().item()),
                    't': float((chosen_idx == 2).float().mean().item()),
                }

                # for logging: the most frequent chosen
                chosen = ['r', 'n', 't'][int(torch.bincount(chosen_idx, minlength=3).argmax().item())]

            self._last_ratio = ratio
            self._last_chosen = chosen
            self._last_hist = dict(hist)

        stats = {
            'cmiei_step': int(self._step),
            'cmiei_epoch': int(self._epoch),
            'cmiei_k': int(self.k),
            'cmiei_sigma': float(self.sigma),
            'cmiei_abs_thr': float(self.abs_thr),
            'cmiei_rel_thr': float(self.rel_thr),
            'cmiei_sample_level': bool(self.sample_level),
            'cmiei_p_max': float(self.p_max),
            'cmiei_warmup_epochs': int(self.warmup_epochs),

            # main stats
            'cmiei_ratio': float(ratio),
            'cmiei_chosen': str(chosen),
            'cmiei_ted_r': float(ci_r_mean),
            'cmiei_ted_n': float(ci_n_mean),
            'cmiei_ted_t': float(ci_t_mean),

            # histogram over chosen modality (sample-level; still filled for batch-level)
            'cmiei_hist_r': float(hist.get('r', 0.0)),
            'cmiei_hist_n': float(hist.get('n', 0.0)),
            'cmiei_hist_t': float(hist.get('t', 0.0)),
        }
        return fr, fn, ft, stats
