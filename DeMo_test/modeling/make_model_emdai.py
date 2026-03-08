import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from modeling.moe.AttnMOE import GeneralFusion, QuickGELU
import torch


class DeMo(nn.Module):
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
                QuickGELU()
            )
            self.nir_reduce = nn.Sequential(
                nn.LayerNorm(2 * self.feat_dim),
                nn.Linear(2 * self.feat_dim, self.feat_dim),
                QuickGELU()
            )
            self.tir_reduce = nn.Sequential(
                nn.LayerNorm(2 * self.feat_dim),
                nn.Linear(2 * self.feat_dim, self.feat_dim),
                QuickGELU()
            )

        if self.HDM or self.ATM:
            self.generalFusion = GeneralFusion(feat_dim=self.feat_dim, num_experts=7, head=self.head, reg_weight=0,
                                               cfg=cfg)
            self.classifier_moe = nn.Linear(7 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_moe.apply(weights_init_classifier)
            self.bottleneck_moe = nn.BatchNorm1d(7 * self.feat_dim)
            self.bottleneck_moe.bias.requires_grad_(False)
            self.bottleneck_moe.apply(weights_init_kaiming)

        # heads
        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

            # per-modality independent logits (for EMDAI gating / analysis)
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
            self.bottleneck_t.apply(weights_init_kaiming)
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

        # EMDAI plugin (lazy init in forward)
        self._emdai_plugin = None
        self.emdai_stats = None

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

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

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

            if self.direct:
                # independent logits first (for entropy gating)
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))

                # EMDAI intervention at global features (before fusion concat)
                self.emdai_stats = None
                if hasattr(self.cfg.MODEL, 'EMDAI') and getattr(self.cfg.MODEL.EMDAI, 'ENABLED', False):
                    from layers.emdai import EMDAIPlugin
                    if self._emdai_plugin is None:
                        self._emdai_plugin = EMDAIPlugin(
                            threshold=float(self.cfg.MODEL.EMDAI.THRESHOLD),
                            num_classes=int(self.num_classes)
                        )
                    NI_global, ni_stats = self._emdai_plugin(NI_global, RGB_global, RGB_ori_score)
                    TI_global, ti_stats = self._emdai_plugin(TI_global, RGB_global, RGB_ori_score)

                    # ---- Diagnosis Path 1: stats of RGB head + features (to understand max-entropy issue) ----
                    # We keep original emdai_stats keys, but also add detailed scalars.
                    with torch.no_grad():
                        rgb_logits_det = RGB_ori_score.detach()
                        rgb_row_max = rgb_logits_det.max(dim=1).values
                        rgb_row_min = rgb_logits_det.min(dim=1).values
                        rgb_row_std = rgb_logits_det.std(dim=1)

                        rgb_feat_l2 = RGB_global.detach().norm(p=2, dim=1)

                        # entropy (unnormalized) and normalized entropy = H/log(C)
                        max_ent = torch.log(torch.tensor(float(self.num_classes), device=rgb_logits_det.device))
                        rgb_probs = torch.softmax(rgb_logits_det, dim=1)
                        rgb_ent = (-rgb_probs * torch.log(rgb_probs + 1e-8)).sum(dim=1)
                        rgb_ent_norm = rgb_ent / (max_ent + 1e-8)

                    # override emdai_stats with richer stats
                    self.emdai_stats = {
                        # intervention
                        'interv_ratio': float((ni_stats.get('intervention_ratio', 0.0) + ti_stats.get('intervention_ratio', 0.0)) / 2.0),
                        'interv_ratio_ni': float(ni_stats.get('intervention_ratio', 0.0)),
                        'interv_ratio_ti': float(ti_stats.get('intervention_ratio', 0.0)),

                        # entropy
                        'avg_entropy': float((ni_stats.get('avg_entropy', 0.0) + ti_stats.get('avg_entropy', 0.0)) / 2.0),
                        'rgb_ent_mean': float(rgb_ent.mean().item()),
                        'rgb_ent_min': float(rgb_ent.min().item()),
                        'rgb_ent_max': float(rgb_ent.max().item()),
                        'rgb_ent_norm_mean': float(rgb_ent_norm.mean().item()),
                        'rgb_ent_norm_min': float(rgb_ent_norm.min().item()),
                        'rgb_ent_norm_max': float(rgb_ent_norm.max().item()),

                        # logits health
                        'rgb_logits_mean': float(rgb_logits_det.mean().item()),
                        'rgb_logits_std': float(rgb_logits_det.std().item()),
                        'rgb_logits_abs_mean': float(rgb_logits_det.abs().mean().item()),
                        'rgb_logits_row_max_mean': float(rgb_row_max.mean().item()),
                        'rgb_logits_row_min_mean': float(rgb_row_min.mean().item()),
                        'rgb_logits_row_std_mean': float(rgb_row_std.mean().item()),

                        # feature health
                        'rgb_feat_l2_mean': float(rgb_feat_l2.mean().item()),
                        'rgb_feat_l2_min': float(rgb_feat_l2.min().item()),
                        'rgb_feat_l2_max': float(rgb_feat_l2.max().item()),

                        # config
                        'threshold': float(self.cfg.MODEL.EMDAI.THRESHOLD),
                        'num_classes': int(self.num_classes),
                    }

                    # NOTE: removed the legacy minimal overwrite to preserve diagnostics above.

                # fused (direct concat)
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)

                if self.HDM or self.ATM:
                    moe_feat, loss_moe = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                    moe_score = self.classifier_moe(self.bottleneck_moe(moe_feat))
                    return {
                        'moe_score': moe_score,
                        'moe_feat': moe_feat,
                        'ori_score': ori_score,
                        'ori_feat': ori,
                        'loss_moe': loss_moe,
                        'feats_dict': {'RGB': RGB_global, 'NI': NI_global, 'TI': TI_global},
                        'logits_dict': {'RGB': RGB_ori_score, 'NI': NI_ori_score, 'TI': TI_ori_score},
                        'emdai_stats': self.emdai_stats,
                    }

                return {
                    'ori_score': ori_score,
                    'ori_feat': ori,
                    'feats_dict': {'RGB': RGB_global, 'NI': NI_global, 'TI': TI_global},
                    'logits_dict': {'RGB': RGB_ori_score, 'NI': NI_ori_score, 'TI': TI_ori_score},
                    'emdai_stats': self.emdai_stats,
                }

            else:
                # non-direct: keep original behavior (no EMDAI here for now)
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))

                if self.HDM or self.ATM:
                    moe_feat, loss_moe = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                    moe_score = self.classifier_moe(self.bottleneck_moe(moe_feat))
                    return {
                        'moe_score': moe_score, 'moe_feat': moe_feat, 'loss_moe': loss_moe,
                        'logits_dict': {'RGB': RGB_ori_score, 'NI': NI_ori_score, 'TI': TI_ori_score},
                        'feats_dict': {'RGB': RGB_global, 'NI': NI_global, 'TI': TI_global}
                    }
                return {
                    'logits_dict': {'RGB': RGB_ori_score, 'NI': NI_ori_score, 'TI': TI_ori_score},
                    'feats_dict': {'RGB': RGB_global, 'NI': NI_global, 'TI': TI_global}
                }

        else:
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
    print('===========Building DeMo===========')
    return model
