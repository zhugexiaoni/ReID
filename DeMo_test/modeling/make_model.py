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
            self.rgb_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim),QuickGELU())
            self.nir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
            self.tir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())

        if self.HDM or self.ATM:
            self.generalFusion = GeneralFusion(feat_dim=self.feat_dim, num_experts=7, head=self.head, reg_weight=0,
                                               cfg=cfg)
            self.classifier_moe = nn.Linear(7 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_moe.apply(weights_init_classifier)
            self.bottleneck_moe = nn.BatchNorm1d(7 * self.feat_dim)
            self.bottleneck_moe.bias.requires_grad_(False)
            self.bottleneck_moe.apply(weights_init_kaiming)
            
        # 始终初始化单模态分类头 (Invasive change to support IADD in all modes)
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

        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

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
                
            # 计算各模态 Logits (始终计算，供 IADD 使用)
            RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
            NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
            TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))

            # MOE 分支
            moe_s, moe_f, l_moe = None, None, 0.0
            if self.HDM or self.ATM:
                moe_f, l_moe = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                moe_s = self.classifier_moe(self.bottleneck_moe(moe_f))
                
            # ORI 分支 (拼接)
            ori_s, ori_f = None, None
            if self.direct:
                ori_f = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_s = self.classifier(self.bottleneck(ori_f))

            # 返回统一元组 (兼容适配器解析)
            # 约定顺序: moe_s, moe_f, l_moe, ori_s, ori_f, R_s, R_f, N_s, N_f, T_s, T_f
            return (moe_s, moe_f, l_moe, ori_s, ori_f, 
                    RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global)

        else:
            RGB = x['RGB']; NI = x['NI']; TI = x['TI']
            if self.miss_type == 'r': RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n': NI = torch.zeros_like(NI)
            elif self.miss_type == 't': TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn': RGB = torch.zeros_like(RGB); NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt': RGB = torch.zeros_like(RGB); TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt': NI = torch.zeros_like(NI); TI = torch.zeros_like(TI)

            if 'cam_label' in x: cam_label = x['cam_label']
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
                if return_pattern == 1: return ori
                elif return_pattern == 2: return moe_feat
                elif return_pattern == 3: return torch.cat([ori, moe_feat], dim=-1)
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
    print('===========Building DeMo (IADD Ready)===========')
    return model
