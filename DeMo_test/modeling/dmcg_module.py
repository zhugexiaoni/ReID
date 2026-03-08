"""
DMCG (Dynamic Modality Coordination Gating) Module

动态模态协调门控模块 - 用于解决模态级和样本级的不平衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DMCGModule(nn.Module):
    """
    动态模态协调门控模块

    输入：多模态特征字典 + MIEI统计信息
    输出：协调后的多模态特征字典 + 门控参数
    """
    def __init__(self, feat_dim=768, n_modalities=3, hidden_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_modalities = n_modalities

        # 跨模态注意力
        self.query_proj = nn.Linear(feat_dim, hidden_dim)
        self.key_proj = nn.Linear(feat_dim, hidden_dim)
        self.value_proj = nn.Linear(feat_dim, feat_dim)

        # 门控网络 - 每个模态一个
        gate_input_dim = feat_dim + 4 + feat_dim  # 特征 + MIEI(4维) + 注意力上下文
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 3),  # [g_self, g_inter, g_comp]
                nn.Sigmoid()
            ) for _ in range(n_modalities)
        ])

    def cross_modal_attention(self, query, keys):
        """
        计算跨模态注意力

        query: (batch, feat_dim)
        keys: list of (batch, feat_dim)

        返回: attention_context (batch, feat_dim), attention_weights
        """
        Q = self.query_proj(query)  # (batch, hidden_dim)
        K = torch.stack([self.key_proj(k) for k in keys], dim=1)  # (batch, n-1, hidden_dim)
        V = torch.stack([self.value_proj(k) for k in keys], dim=1)  # (batch, n-1, feat_dim)

        # 计算注意力分数
        scores = torch.bmm(K, Q.unsqueeze(-1)).squeeze(-1)  # (batch, n-1)
        scores = scores / (self.query_proj.out_features ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n-1)

        # 加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # (batch, feat_dim)

        return context, attn_weights

    def extract_complementary_feature(self, phi_i, phi_others):
        """
        提取模态i相对于其他模态的互补特征（正交分量）

        phi_i: (batch, feat_dim)
        phi_others: list of (batch, feat_dim)
        """
        # 计算其他模态的平均特征
        phi_avg = torch.stack(phi_others, dim=0).mean(dim=0)  # (batch, feat_dim)

        # 计算phi_i在phi_avg上的投影系数
        numerator = (phi_i * phi_avg).sum(dim=-1, keepdim=True)  # (batch, 1)
        denominator = (phi_avg * phi_avg).sum(dim=-1, keepdim=True) + 1e-8  # (batch, 1)
        projection_coeff = numerator / denominator

        # 正交分量（互补特征）
        phi_comp = phi_i - projection_coeff * phi_avg

        # L2归一化
        phi_comp = F.normalize(phi_comp, p=2, dim=-1)

        return phi_comp

    def forward(self, features_dict, miei_dict):
        """
        前向传播

        features_dict: {mod: (batch, feat_dim) for mod in modalities}
        miei_dict: {mod: {'stats': (batch, 4), 'score': (batch, 1)} for mod in modalities}

        返回:
        - coordinated_features: {mod: (batch, feat_dim) for mod in modalities}
        - gates: {mod: (batch, 3) for mod in modalities}
        """
        modalities = list(features_dict.keys())
        batch_size = features_dict[modalities[0]].shape[0]

        coordinated_features = {}
        gates = {}

        for i, mod_i in enumerate(modalities):
            phi_i = features_dict[mod_i]
            phi_others = [features_dict[m] for m in modalities if m != mod_i]

            # 1. 计算跨模态注意力上下文
            attn_context, attn_weights = self.cross_modal_attention(phi_i, phi_others)

            # 2. 生成门控参数
            miei_stats = miei_dict[mod_i]['stats']  # (batch, 4)
            gate_input = torch.cat([phi_i, miei_stats, attn_context], dim=-1)
            g = self.gate_networks[i](gate_input)  # (batch, 3)
            g_self, g_inter, g_comp = g[:, 0:1], g[:, 1:2], g[:, 2:3]
            gates[mod_i] = g

            # 3. 生成协调特征
            # 3.1 自身特征保留
            feat_self = g_self * phi_i

            # 3.2 跨模态信息融合（attention加权）
            feat_inter = g_inter * attn_context

            # 3.3 互补信息强化
            phi_comp = self.extract_complementary_feature(phi_i, phi_others)
            feat_comp = g_comp * phi_comp

            # 3.4 组合协调特征
            phi_coordinated = feat_self + feat_inter + feat_comp

            # L2归一化
            phi_coordinated = F.normalize(phi_coordinated, p=2, dim=-1)

            coordinated_features[mod_i] = phi_coordinated

        return coordinated_features, gates


def gate_regularization_loss(gates_dict, miei_dict, diversity_weight=0.1):
    """
    门控正则化损失

    gates_dict: {'rgb': (batch, 3), 'ni': (batch, 3), 'ti': (batch, 3)}
    miei_dict: {'rgb': {'score': (batch, 1)}, ...}
    """
    loss_total = 0.0

    for mod in gates_dict.keys():
        g = gates_dict[mod]  # (batch, 3)
        g_self, g_inter, g_comp = g[:, 0], g[:, 1], g[:, 2]

        miei_score = miei_dict[mod]['score'].squeeze(-1)  # (batch,)

        # 1. 多样性正则（鼓励门控值接近0.5，避免极端）
        loss_diversity = ((g_self - 0.5)**2 + (g_inter - 0.5)**2 + (g_comp - 0.5)**2).mean()

        # 2. 一致性正则（低MIEI应该有更高的g_inter）
        target_g_inter = 1.0 - miei_score
        loss_consistency = ((g_inter - target_g_inter)**2).mean()

        # 3. 互补性鼓励
        target_g_comp = 1.0 - miei_score
        loss_complement = ((g_comp - target_g_comp)**2).mean()

        loss_total += diversity_weight * loss_diversity + loss_consistency + loss_complement

    return loss_total / len(gates_dict)


def balance_promotion_loss(coord_features_dict, classifier, labels):
    """
    平衡性促进损失

    coord_features_dict: {'rgb': (batch, feat_dim), ...}
    classifier: nn.Module 或 dict of nn.Module
    labels: (batch,)
    """
    # 计算各模态的预测熵
    entropies = []
    for mod in coord_features_dict.keys():
        # 如果classifier是字典，使用对应的分类器
        if isinstance(classifier, dict):
            logits = classifier[mod](coord_features_dict[mod])
        else:
            logits = classifier(coord_features_dict[mod])

        probs = F.softmax(logits, dim=-1)
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropies.append(H)

    # 熵的方差（越小越平衡）
    entropy_tensor = torch.stack(entropies)
    loss = torch.var(entropy_tensor)

    return loss
