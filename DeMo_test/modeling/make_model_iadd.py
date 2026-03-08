"""modeling/make_model_iadd.py

DeMo + IADD adaptation

Goal:
- Do NOT modify the original DeMo model codepaths.
- Provide a new make_model entry that builds the native DeMo architecture
  (same as default modeling.make_model used by this repo) but adapts its
  forward outputs to be compatible with engine/processor_iadd.py.

Key contract for IADD training loop (engine/processor_iadd.py):
- model(img, ...) should return a dict in training mode containing:
    - fused logits/feat used by base loss (ori_* or moe_* etc)
    - logits_dict: per-modality logits for RGB/NI/TI
    - feats_dict: per-modality features for RGB/NI/TI
  so the IADDPlugin can compute MCD + dynamic distillation.

Extra support added for your new requirement:
- When HDM/ATM are disabled AND DIRECT=0:
    * Native DeMo only returns per-modality (score, feat) pairs.
    * We create an "ori" feature by concatenating [RGB_global, NI_global, TI_global]
      and add an auxiliary classifier head to produce ori_score.
    * This allows training with base ID loss on ori (triplet part can be disabled,
      because IADD already adds hybrid triplet).
- Evaluation behavior remains identical to native DeMo (adapter returns base outputs
  as-is in eval mode).

This keeps DeMo behavior unchanged; we only standardize outputs and add an aux head
in the wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn


def _as_dict_from_tuple(
    out: Union[Tuple[Any, ...], Sequence[Any]],
    direct: bool,
    hdm_or_atm: bool,
) -> Dict[str, Any]:
    """Convert legacy tuple outputs to the dict schema used by processor_iadd."""

    out = tuple(out)

    if direct:
        if hdm_or_atm:
            moe_score, moe_feat, ori_score, ori_feat, loss_moe = out
            return {
                "moe_score": moe_score,
                "moe_feat": moe_feat,
                "ori_score": ori_score,
                "ori_feat": ori_feat,
                "loss_moe": loss_moe,
            }
        else:
            ori_score, ori_feat = out
            return {"ori_score": ori_score, "ori_feat": ori_feat}

    # non-direct
    if hdm_or_atm:
        (
            moe_score,
            moe_feat,
            rgb_score,
            rgb_feat,
            ni_score,
            ni_feat,
            ti_score,
            ti_feat,
            loss_moe,
        ) = out
        return {
            "moe_score": moe_score,
            "moe_feat": moe_feat,
            "loss_moe": loss_moe,
            "logits_dict": {"RGB": rgb_score, "NI": ni_score, "TI": ti_score},
            "feats_dict": {"RGB": rgb_feat, "NI": ni_feat, "TI": ti_feat},
        }

    rgb_score, rgb_feat, ni_score, ni_feat, ti_score, ti_feat = out
    return {
        "logits_dict": {"RGB": rgb_score, "NI": ni_score, "TI": ti_score},
        "feats_dict": {"RGB": rgb_feat, "NI": ni_feat, "TI": ti_feat},
    }


class DeMoIADDAdapter(nn.Module):
    """Adapter that standardizes DeMo outputs for IADD training.

    Also optionally adds an auxiliary ori-concat classifier head when:
    - DIRECT=0
    - HDM=False and ATM=False

    In this case, DeMo has no fused head in training outputs; we create:
    - ori_feat = concat([RGB_feat, NI_feat, TI_feat])
    - ori_score = aux_classifier(ori_feat)

    so base loss can be computed on ori (typically ID-only).
    """

    def __init__(self, base_model: nn.Module, cfg):
        super().__init__()
        self.base = base_model
        self.cfg = cfg

        self.use_aux_ori_head = (
            (not bool(getattr(cfg.MODEL, "DIRECT", False)))
            and (not bool(getattr(cfg.MODEL, "HDM", False)))
            and (not bool(getattr(cfg.MODEL, "ATM", False)))
        )

        if self.use_aux_ori_head:
            feat_dim = int(getattr(base_model, "feat_dim", 512))
            num_classes = int(getattr(base_model, "num_classes", 0))
            self.aux_classifier = nn.Linear(3 * feat_dim, num_classes, bias=False)
            # init same style as DeMo classifiers
            try:
                from modeling.meta_arch import weights_init_classifier

                self.aux_classifier.apply(weights_init_classifier)
            except Exception:
                nn.init.normal_(self.aux_classifier.weight, std=0.001)

    def forward(self, x, label=None, cam_label=None, view_label=None, **kwargs):
        out = self.base(
            x,
            label=label,
            cam_label=cam_label,
            view_label=view_label,
            **kwargs,
        )

        # eval mode: keep *exactly* what base model returns
        if not self.training:
            return out

        # training: standardize to dict
        if isinstance(out, dict):
            return out

        if isinstance(out, (tuple, list)):
            direct = bool(getattr(self.cfg.MODEL, "DIRECT", False))
            hdm_or_atm = bool(getattr(self.cfg.MODEL, "HDM", False) or getattr(self.cfg.MODEL, "ATM", False))
            out_dict = _as_dict_from_tuple(out, direct=direct, hdm_or_atm=hdm_or_atm)

            # If needed, create ori_score/ori_feat for base loss
            if self.use_aux_ori_head and ("feats_dict" in out_dict) and ("logits_dict" in out_dict):
                rgb_feat = out_dict["feats_dict"]["RGB"]
                ni_feat = out_dict["feats_dict"]["NI"]
                ti_feat = out_dict["feats_dict"]["TI"]
                ori_feat = torch.cat([rgb_feat, ni_feat, ti_feat], dim=-1)
                ori_score = self.aux_classifier(ori_feat)
                out_dict["ori_feat"] = ori_feat
                out_dict["ori_score"] = ori_score

            return out_dict

        return out

    def load_param(self, trained_path: str):
        if hasattr(self.base, "load_param"):
            return self.base.load_param(trained_path)
        state_dict = torch.load(trained_path, map_location="cpu")
        incompatible = self.base.load_state_dict(state_dict, strict=False)
        print(incompatible)


def make_model_iadd(cfg, num_class, camera_num, view_num=0):
    """Build native DeMo and wrap it for IADD."""

    from .make_model import make_model as make_demo

    base = make_demo(cfg, num_class, camera_num, view_num)
    model = DeMoIADDAdapter(base, cfg)
    print("===========Building DeMo + IADD Adapter===========")
    return model
