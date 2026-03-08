"""
DeMo + DMCG Integration

集成了动态模态协调门控(DMCG)的DeMo模型
用于非平衡多模态学习
"""

import torch
import torch.nn as nn
from DeMo_test.modeling.make_model_emdai import DeMo
from modeling.dmcg_module import DMCGModule, gate_regularization_loss, balance_promotion_loss
from modeling.miei_calculator import MEIECalculator
from modeling.meta_arch import weights_init_classifier, weights_init_kaiming
import torch.nn.functional as F


class DeMo_DMCG(nn.Module):
    """
    DeMo + DMCG 集成模型

    在DeMo的基础上添加DMCG模块来解决模态不平衡问题
    """

    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super().__init__()

        # 初始化基础DeMo模型
        self.demo = DeMo(num_classes, cfg, camera_num, view_num, factory)

        # 获取特征维度
        self.feat_dim = self.demo.feat_dim
        self.num_classes = num_classes
        self.cfg = cfg

        # DMCG相关参数
        self.use_dmcg = cfg.MODEL.DMCG.ENABLED
        self.dmcg_warmup_epochs = cfg.MODEL.DMCG.WARMUP_EPOCHS
        self.lambda_gate = cfg.MODEL.DMCG.LAMBDA_GATE
        self.lambda_balance = cfg.MODEL.DMCG.LAMBDA_BALANCE

        if self.use_dmcg:
            # DMCG模块
            self.dmcg = DMCGModule(
                feat_dim=self.feat_dim,
                n_modalities=3,  # RGB, NI, TI
                hidden_dim=cfg.MODEL.DMCG.HIDDEN_DIM
            )

            # MIEI计算器
            self.miei_calculator = MEIECalculator(
                n_modalities=3,
                n_classes=num_classes,
                beta=cfg.MODEL.DMCG.BETA,
                alpha=cfg.MODEL.DMCG.ALPHA
            )

            # 协调特征的分类器
            if cfg.MODEL.DIRECT:
                # DMCG融合特征的分类器（3模态拼接）
                self.classifier_dmcg = nn.Linear(3 * self.feat_dim, num_classes, bias=False)
                self.classifier_dmcg.apply(weights_init_classifier)
                self.bottleneck_dmcg = nn.BatchNorm1d(3 * self.feat_dim)
                self.bottleneck_dmcg.bias.requires_grad_(False)
                self.bottleneck_dmcg.apply(weights_init_kaiming)

                # 为了计算MIEI，需要单模态的bottleneck和classifier
                self.bottleneck_single = nn.BatchNorm1d(self.feat_dim)
                self.bottleneck_single.bias.requires_grad_(False)
                self.bottleneck_single.apply(weights_init_kaiming)
                self.classifier_single = nn.Linear(self.feat_dim, num_classes, bias=False)
                self.classifier_single.apply(weights_init_classifier)
            else:
                # 每个模态独立的分类器（用于计算平衡损失）
                self.classifier_dmcg_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
                self.classifier_dmcg_ni = nn.Linear(self.feat_dim, num_classes, bias=False)
                self.classifier_dmcg_ti = nn.Linear(self.feat_dim, num_classes, bias=False)
                self.classifier_dmcg_rgb.apply(weights_init_classifier)
                self.classifier_dmcg_ni.apply(weights_init_classifier)
                self.classifier_dmcg_ti.apply(weights_init_classifier)

        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前epoch（用于warmup控制）"""
        self.current_epoch = epoch

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        """
        前向传播

        训练时返回: 各种logits和特征用于损失计算
        测试时返回: 融合特征用于检索
        """
        if self.training:
            return self._forward_train(x, label, cam_label, view_label)
        else:
            return self._forward_test(x, cam_label, view_label, return_pattern)

    def _forward_train(self, x, label, cam_label, view_label):
        """训练时的前向传播"""
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']

        # 1. 提取原始特征（使用DeMo的backbone）
        RGB_cash, RGB_global = self.demo.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global = self.demo.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global = self.demo.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

        # 处理Global+Local特征
        if self.demo.GLOBAL_LOCAL:
            RGB_local = self.demo.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
            NI_local = self.demo.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
            TI_local = self.demo.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
            RGB_global = self.demo.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
            NI_global = self.demo.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
            TI_global = self.demo.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

        # 2. 原始DeMo的融合和分类
        if self.demo.HDM or self.demo.ATM:
            moe_feat, loss_moe = self.demo.generalFusion(RGB_cash, NI_cash, TI_cash,
                                                          RGB_global, NI_global, TI_global)
            moe_score = self.demo.classifier_moe(self.demo.bottleneck_moe(moe_feat))
        else:
            loss_moe = torch.tensor(0.0).to(RGB.device)

        if self.demo.direct:
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.demo.bottleneck(ori)
            ori_score = self.demo.classifier(ori_global)
        else:
            RGB_ori_score = self.demo.classifier_r(self.demo.bottleneck_r(RGB_global))
            NI_ori_score = self.demo.classifier_n(self.demo.bottleneck_n(NI_global))
            TI_ori_score = self.demo.classifier_t(self.demo.bottleneck_t(TI_global))

        # 3. DMCG处理（如果启用且过了warmup期）
        dmcg_enabled = self.use_dmcg and self.current_epoch >= self.dmcg_warmup_epochs

        if dmcg_enabled:
            # 构建特征字典
            features_dict = {
                'RGB': RGB_global,
                'NI': NI_global,
                'TI': TI_global
            }

            # 构建logits字典（用于MIEI计算）
            if self.demo.direct:
                # 如果是direct模式，使用单模态的bottleneck和classifier
                logits_dict = {
                    'RGB': self.classifier_single(self.bottleneck_single(RGB_global)),
                    'NI': self.classifier_single(self.bottleneck_single(NI_global)),
                    'TI': self.classifier_single(self.bottleneck_single(TI_global))
                }
            else:
                logits_dict = {
                    'RGB': RGB_ori_score,
                    'NI': NI_ori_score,
                    'TI': TI_ori_score
                }

            # 计算MIEI
            features_list = [features_dict[m] for m in ['RGB', 'NI', 'TI']]
            logits_list = [logits_dict[m] for m in ['RGB', 'NI', 'TI']]

            miei_sample = self.miei_calculator.compute_miei_sample(
                features_list, logits_list, label
            )

            # 构建MIEI字典
            miei_dict = {}
            modalities = ['RGB', 'NI', 'TI']
            for i, mod in enumerate(modalities):
                decomp = self.miei_calculator.get_decomposition(
                    features_list, logits_list, label
                )[f'modality_{i}']

                miei_dict[mod] = {
                    'stats': torch.stack([
                        torch.tensor(decomp['feature_entropy'], device=RGB.device),
                        torch.tensor(decomp['information_gain'], device=RGB.device),
                        torch.tensor(decomp['correctness'], device=RGB.device),
                        torch.tensor(decomp['uniqueness'], device=RGB.device)
                    ], dim=-1).float(),  # (batch, 4)
                    'score': miei_sample[i].unsqueeze(-1)  # (batch, 1)
                }

            # DMCG协调特征生成
            coord_features, gates = self.dmcg(features_dict, miei_dict)

            # 融合协调后的特征
            if self.demo.direct:
                dmcg_feat = torch.cat([coord_features[m] for m in modalities], dim=-1)
                dmcg_feat_bn = self.bottleneck_dmcg(dmcg_feat)
                dmcg_score = self.classifier_dmcg(dmcg_feat_bn)
            else:
                dmcg_feat_rgb = coord_features['RGB']
                dmcg_feat_ni = coord_features['NI']
                dmcg_feat_ti = coord_features['TI']
                dmcg_score_rgb = self.classifier_dmcg_rgb(dmcg_feat_rgb)
                dmcg_score_ni = self.classifier_dmcg_ni(dmcg_feat_ni)
                dmcg_score_ti = self.classifier_dmcg_ti(dmcg_feat_ti)
                dmcg_feat = torch.cat([dmcg_feat_rgb, dmcg_feat_ni, dmcg_feat_ti], dim=-1)
                dmcg_score = (dmcg_score_rgb + dmcg_score_ni + dmcg_score_ti) / 3

            # 计算DMCG相关损失
            loss_gate = gate_regularization_loss(gates, miei_dict)

            # 平衡损失需要对每个模态单独计算，因此都使用单模态分类器
            if self.demo.direct:
                # DIRECT模式下，使用单模态分类器（每个模态512维）
                classifier_dict = {
                    'RGB': self.classifier_single,
                    'NI': self.classifier_single,
                    'TI': self.classifier_single
                }
            else:
                classifier_dict = {
                    'RGB': self.classifier_dmcg_rgb,
                    'NI': self.classifier_dmcg_ni,
                    'TI': self.classifier_dmcg_ti
                }
            loss_balance = balance_promotion_loss(coord_features, classifier_dict, label)
        else:
            dmcg_score = None
            dmcg_feat = None
            loss_gate = torch.tensor(0.0).to(RGB.device)
            loss_balance = torch.tensor(0.0).to(RGB.device)
            gates = None
            miei_dict = None

        # 4. 返回所有需要的输出
        if self.demo.direct:
            if self.demo.HDM or self.demo.ATM:
                if dmcg_enabled:
                    return {
                        'moe_score': moe_score,
                        'moe_feat': moe_feat,
                        'ori_score': ori_score,
                        'ori_feat': ori,
                        'dmcg_score': dmcg_score,
                        'dmcg_feat': dmcg_feat,
                        'loss_moe': loss_moe,
                        'loss_gate': loss_gate,
                        'loss_balance': loss_balance,
                        'gates': gates,
                        'miei_dict': miei_dict
                    }
                return moe_score, moe_feat, ori_score, ori, loss_moe
            else:
                if dmcg_enabled:
                    return {
                        'ori_score': ori_score,
                        'ori_feat': ori,
                        'dmcg_score': dmcg_score,
                        'dmcg_feat': dmcg_feat,
                        'loss_gate': loss_gate,
                        'loss_balance': loss_balance,
                        'gates': gates,
                        'miei_dict': miei_dict
                    }
                return ori_score, ori
        else:
            if self.demo.HDM or self.demo.ATM:
                if dmcg_enabled:
                    return {
                        'moe_score': moe_score,
                        'moe_feat': moe_feat,
                        'RGB_ori_score': RGB_ori_score,
                        'RGB_global': RGB_global,
                        'NI_ori_score': NI_ori_score,
                        'NI_global': NI_global,
                        'TI_ori_score': TI_ori_score,
                        'TI_global': TI_global,
                        'dmcg_score': dmcg_score,
                        'dmcg_feat': dmcg_feat,
                        'loss_moe': loss_moe,
                        'loss_gate': loss_gate,
                        'loss_balance': loss_balance,
                        'gates': gates,
                        'miei_dict': miei_dict
                    }
                return moe_score, moe_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, loss_moe
            else:
                if dmcg_enabled:
                    return {
                        'RGB_ori_score': RGB_ori_score,
                        'RGB_global': RGB_global,
                        'NI_ori_score': NI_ori_score,
                        'NI_global': NI_global,
                        'TI_ori_score': TI_ori_score,
                        'TI_global': TI_global,
                        'dmcg_score': dmcg_score,
                        'dmcg_feat': dmcg_feat,
                        'loss_gate': loss_gate,
                        'loss_balance': loss_balance,
                        'gates': gates,
                        'miei_dict': miei_dict
                    }
                return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global

    def _forward_test(self, x, cam_label, view_label, return_pattern):
        """测试时的前向传播"""
        # 测试时使用原始DeMo的逻辑
        return self.demo.forward(x, cam_label=cam_label, view_label=view_label, return_pattern=return_pattern)

    def load_param(self, trained_path):
        """加载预训练参数"""
        self.demo.load_param(trained_path)


def make_model_dmcg(cfg, num_class, camera_num, view_num=0):
    """构建DeMo+DMCG模型的工厂函数"""
    from DeMo_test.modeling.make_model_emdai import __factory_T_type
    model = DeMo_DMCG(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building DeMo + DMCG===========')
    print(f'DMCG Enabled: {cfg.MODEL.DMCG.ENABLED}')
    if cfg.MODEL.DMCG.ENABLED:
        print(f'Warmup Epochs: {cfg.MODEL.DMCG.WARMUP_EPOCHS}')
        print(f'Lambda Gate: {cfg.MODEL.DMCG.LAMBDA_GATE}')
        print(f'Lambda Balance: {cfg.MODEL.DMCG.LAMBDA_BALANCE}')
    return model
