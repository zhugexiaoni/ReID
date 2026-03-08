import torch
import torch.nn as nn

from fvcore.nn import flop_count

from modeling.backbones.basic_cnn_params.flops import give_supported_ops
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from modeling.backbones.vit_pytorch import (
    deit_small_patch16_224,
    vit_base_patch16_224,
    vit_small_patch16_224,
)
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from modeling.moe.AttnMOE import GeneralFusion, QuickGELU

import copy


class DeMo(nn.Module):
    """DeMo + C-MIEI plugin version.

    IMPORTANT:
    - This file is a copy-derived variant of make_model_origin.py.
    - Keep baseline behavior identical when C_MIEI is disabled.
    - Keep original return signature (tuple/list), NOT dict.
    """

    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(DeMo, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512

        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD

        if self.GLOBAL_LOCAL:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.rgb_reduce = nn.Sequential(
                nn.LayerNorm(2 * self.feat_dim),
                nn.Linear(2 * self.feat_dim, self.feat_dim),
                QuickGELU(),
            )
            self.nir_reduce = nn.Sequential(
                nn.LayerNorm(2 * self.feat_dim),
                nn.Linear(2 * self.feat_dim, self.feat_dim),
                QuickGELU(),
            )
            self.tir_reduce = nn.Sequential(
                nn.LayerNorm(2 * self.feat_dim),
                nn.Linear(2 * self.feat_dim, self.feat_dim),
                QuickGELU(),
            )

        if self.HDM or self.ATM:
            self.generalFusion = GeneralFusion(feat_dim=self.feat_dim, num_experts=7, head=self.head, reg_weight=0, cfg=cfg)
            self.classifier_moe = nn.Linear(7 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_moe.apply(weights_init_classifier)
            self.bottleneck_moe = nn.BatchNorm1d(7 * self.feat_dim)
            self.bottleneck_moe.bias.requires_grad_(False)
            self.bottleneck_moe.apply(weights_init_kaiming)

        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

            # NOTE: for parity with other bottlenecks
            self.bottleneck_t.apply(weights_init_kaiming)

            # C-MIEI auxiliary fused head (DIRECT=0 only)
            # Used ONLY for counterfactual influence estimation (TED) and substitution decision.
            # Does NOT change original outputs / loss unless you explicitly add it.
            self.cmiei_classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.cmiei_classifier.apply(weights_init_classifier)
            self.cmiei_bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.cmiei_bottleneck.bias.requires_grad_(False)
            self.cmiei_bottleneck.apply(weights_init_kaiming)

        # C-MIEI plugin
        self._cmiei = None
        self._cmiei_stats = None

        # cache for counterfactual fused logits (to support DIRECT=0 using MOE logits)
        self._cmiei_cache = {}

        # auxiliary fuse head for [ori, moe] feature (used for TED estimation when requested)
        # dim(ori)=3D, dim(moe)=7D => 10D
        self.cmiei_fuse_bottleneck = nn.BatchNorm1d(10 * self.feat_dim)
        self.cmiei_fuse_bottleneck.bias.requires_grad_(False)
        self.cmiei_fuse_bottleneck.apply(weights_init_kaiming)
        self.cmiei_fuse_classifier = nn.Linear(10 * self.feat_dim, self.num_classes, bias=False)
        self.cmiei_fuse_classifier.apply(weights_init_classifier)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def _fused_logits(self, fr, fn, ft, bn_update: bool = True):
        """Compute *fused* logits for C-MIEI counterfactual estimation.

        This helper is used by the C-MIEI plugin as `fused_logits_fn`.

        Requirements:
        - Must be available for BOTH DIRECT=1 and DIRECT=0 training modes.
        - Must avoid polluting BatchNorm running stats during counterfactual passes.

        Behavior:
        - If DIRECT=1: use the model's original DIRECT head (bottleneck + classifier).
        - If DIRECT=0: use a lightweight auxiliary fused head (cmiei_bottleneck + cmiei_classifier)
          that is ONLY used for counterfactual influence estimation and feature substitution.
          It does NOT affect the original training loss unless you explicitly add it.
        """

        # Choose fused logits source for TED estimation.
        fused_source = 'ori'
        try:
            if hasattr(self.cfg.MODEL, 'C_MIEI'):
                fused_source = str(getattr(self.cfg.MODEL.C_MIEI, 'FUSED_SOURCE', 'ori')).lower()
        except Exception:
            fused_source = 'ori'

        # DIRECT=0: if user does not override, default to 'fuse' (since no ori head exists)
        if (not getattr(self, 'direct', 0)) and fused_source == 'ori':
            fused_source = 'fuse'

        if fused_source == 'fuse':
            # Re-compute moe_feat under (fr,fn,ft) counterfactual by dropping corresponding cached token features.
            RGB_cash = self._cmiei_cache.get('RGB_cash', None)
            NI_cash = self._cmiei_cache.get('NI_cash', None)
            TI_cash = self._cmiei_cache.get('TI_cash', None)
            if (RGB_cash is None) or (NI_cash is None) or (TI_cash is None):
                raise RuntimeError('C-MIEI fused_source=fuse requires RGB_cash/NI_cash/TI_cash cached before calling the plugin.')

            # detect which modality is counterfactually dropped by comparing to cached globals
            fr0 = self._cmiei_cache.get('RGB_global', None)
            fn0 = self._cmiei_cache.get('NI_global', None)
            ft0 = self._cmiei_cache.get('TI_global', None)
            if (fr0 is None) or (fn0 is None) or (ft0 is None):
                raise RuntimeError('C-MIEI fused_source=fuse requires RGB_global/NI_global/TI_global cached.')

            def _same(a, b):
                # use exact equality; drop tensors are either zeros or expanded means, so they differ from originals.
                return (a is b) or (a.data_ptr() == b.data_ptr())

            drop_r = not _same(fr, fr0)
            drop_n = not _same(fn, fn0)
            drop_t = not _same(ft, ft0)

            # drop cached token features in the same way as plugin drop_mode
            def _drop_tokens(tok: torch.Tensor):
                if self._cmiei_cache.get('drop_mode', 'zero') == 'zero':
                    return torch.zeros_like(tok)
                # mean over batch dimension (tok is [B, N, C] from backbone forward)
                mu = tok.mean(dim=0, keepdim=True)
                return mu.expand_as(tok)

            RGB_cash_cf = _drop_tokens(RGB_cash) if drop_r else RGB_cash
            NI_cash_cf = _drop_tokens(NI_cash) if drop_n else NI_cash
            TI_cash_cf = _drop_tokens(TI_cash) if drop_t else TI_cash

            if self.training:
                moe_feat, _ = self.generalFusion(RGB_cash_cf, NI_cash_cf, TI_cash_cf, fr, fn, ft)
            else:
                moe_feat = self.generalFusion(RGB_cash_cf, NI_cash_cf, TI_cash_cf, fr, fn, ft)

            ori = torch.cat([fr, fn, ft], dim=-1)
            fuse_feat = torch.cat([ori, moe_feat], dim=-1)  # [B, 10D]
            bn = self.cmiei_fuse_bottleneck
            clf = self.cmiei_fuse_classifier

            if bn_update:
                fuse_bn = bn(fuse_feat)
            else:
                was_training = bn.training
                try:
                    bn.eval()
                    fuse_bn = bn(fuse_feat)
                finally:
                    bn.train(was_training)

            return clf(fuse_bn)

        if fused_source == 'moe':
            z = self._cmiei_cache.get('moe_logits', None)
            if z is None:
                raise RuntimeError('C-MIEI fused_source=moe requires moe_logits cached before calling the plugin.')
            return z

        # DIRECT=1: use the original ori head (BN + classifier)
        ori = torch.cat([fr, fn, ft], dim=-1)
        bn = self.bottleneck
        clf = self.classifier

        if bn_update:
            ori_global = bn(ori)
        else:
            was_training = bn.training
            try:
                bn.eval()
                ori_global = bn(ori)
            finally:
                bn.train(was_training)

        return clf(ori_global)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        self._cmiei_stats = None
        self._cmiei_cache = {}

        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

            # cache token/global features for counterfactual re-computation under fused_source='fuse'
            self._cmiei_cache['RGB_cash'] = RGB_cash
            self._cmiei_cache['NI_cash'] = NI_cash
            self._cmiei_cache['TI_cash'] = TI_cash
            self._cmiei_cache['RGB_global'] = RGB_global
            self._cmiei_cache['NI_global'] = NI_global
            self._cmiei_cache['TI_global'] = TI_global

            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

            if self.HDM or self.ATM:
                moe_feat, loss_moe = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                moe_score = self.classifier_moe(self.bottleneck_moe(moe_feat))
                # cache moe logits for C-MIEI TED estimation under DIRECT=0
                self._cmiei_cache['moe_logits'] = moe_score
                self._cmiei_cache['moe_feat'] = moe_feat

            # --- C-MIEI intervention (now supports BOTH DIRECT=1 and DIRECT=0) ---
            enable = False
            if hasattr(self.cfg.MODEL, 'C_MIEI'):
                enable = bool(getattr(self.cfg.MODEL.C_MIEI, 'ENABLED', False))

            if enable:
                # Decide fused logits source (ori vs moe). If requested moe, require MOE enabled.
                fused_source = 'ori'
                try:
                    fused_source = str(getattr(self.cfg.MODEL.C_MIEI, 'FUSED_SOURCE', 'ori')).lower()
                except Exception:
                    fused_source = 'ori'

                if (not self.direct) and (fused_source == 'ori'):
                    fused_source = 'fuse'

                if (not self.direct) or (fused_source in {'moe', 'fuse'}):
                    if not (self.HDM or self.ATM):
                        raise RuntimeError('C-MIEI fused_source in {moe,fuse} (or DIRECT=0) requires HDM or ATM enabled to provide MOE outputs.')

                from layers.c_miei import CounterfactualSubstitutePlugin
                if self._cmiei is None:
                    k = int(getattr(self.cfg.MODEL.C_MIEI, 'K', 3))
                    sigma = float(getattr(self.cfg.MODEL.C_MIEI, 'SIGMA', 0.05))
                    abs_thr = float(getattr(self.cfg.MODEL.C_MIEI, 'ABS_THR', 0.03))
                    rel_thr = float(getattr(self.cfg.MODEL.C_MIEI, 'REL_THR', 1.25))
                    sample_level = bool(getattr(self.cfg.MODEL.C_MIEI, 'SAMPLE_LEVEL', False))
                    p_max = float(getattr(self.cfg.MODEL.C_MIEI, 'P_MAX', 0.5))
                    warmup_epochs = int(getattr(self.cfg.MODEL.C_MIEI, 'WARMUP_EPOCHS', 5))
                    self._cmiei = CounterfactualSubstitutePlugin(
                        k=k,
                        sigma=sigma,
                        abs_thr=abs_thr,
                        rel_thr=rel_thr,
                        sample_level=sample_level,
                        p_max=p_max,
                        warmup_epochs=warmup_epochs,
                    )

                # cache plugin drop_mode for token-drop consistency
                try:
                    self._cmiei_cache['drop_mode'] = str(getattr(self._cmiei, 'drop_mode', 'zero'))
                except Exception:
                    self._cmiei_cache['drop_mode'] = 'zero'

                RGB_global, NI_global, TI_global, stats = self._cmiei(
                    RGB_global,
                    NI_global,
                    TI_global,
                    fused_logits_fn=self._fused_logits,
                    enable=True,
                )
                self._cmiei_stats = stats

            # original heads
            if self.direct:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
            else:
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))

            # keep original return signature
            if self.direct:
                if self.HDM or self.ATM:
                    # attach stats as an extra last item (NO effect on original loss loop if you ignore it)
                    # but to keep strict compatibility, we do NOT append it here.
                    return moe_score, moe_feat, ori_score, ori, loss_moe
                return ori_score, ori
            else:
                if self.HDM or self.ATM:
                    return moe_score, moe_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, loss_moe
                return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global

        # ---------------- eval (unchanged) ----------------
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']
        if self.miss_type == 'r':
            RGB = torch.zeros_like(RGB)
        elif self.miss_type == 'n':
            NI = torch.zeros_like(NI)
        elif self.miss_type == 't':
            TI = torch.zeros_like(TI)
        elif self.miss_type == 'rn':
            RGB = torch.zeros_like(RGB)
            NI = torch.zeros_like(NI)
        elif self.miss_type == 'rt':
            RGB = torch.zeros_like(RGB)
            TI = torch.zeros_like(TI)
        elif self.miss_type == 'nt':
            NI = torch.zeros_like(NI)
            TI = torch.zeros_like(TI)

        if 'cam_label' in x:
            cam_label = x['cam_label']

        RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

        if self.GLOBAL_LOCAL:
            RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
            NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
            TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
            RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
            NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
            TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

        ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
        if self.HDM or self.ATM:
            moe_feat = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
            if return_pattern == 1:
                return ori
            elif return_pattern == 2:
                return moe_feat
            elif return_pattern == 3:
                return torch.cat([ori, moe_feat], dim=-1)
        return ori


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = DeMo(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building DeMo (C-MIEI)===========')
    return model
