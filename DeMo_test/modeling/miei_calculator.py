"""
MIEI (Modality Information Entropy Imbalance) Calculator

模态信息熵失衡度计算器 - 用于衡量多模态学习的不平衡程度
"""

import torch
import torch.nn.functional as F
import numpy as np


class MEIECalculator:
    """模态信息熵失衡度计算器"""

    def __init__(
        self,
        n_modalities: int,
        n_classes: int,
        beta: list = None,
        alpha: float = 1.0
    ):
        """
        参数:
            n_modalities: 模态数量
            n_classes: 类别数量
            beta: 四个权重参数 [β₁, β₂, β₃, β₄]
            alpha: 错误预测惩罚系数
        """
        self.n_modalities = n_modalities
        self.n_classes = n_classes
        self.alpha = alpha

        # 默认均衡权重
        if beta is None:
            self.beta = [0.25, 0.25, 0.25, 0.25]
        else:
            assert len(beta) == 4 and abs(sum(beta) - 1.0) < 1e-6
            self.beta = beta

        self.eps = 1e-8  # 数值稳定性
        self.log_C = np.log(n_classes)  # 先验熵

    def compute_feature_entropy(self, features):
        """
        计算特征熵 H̄_feature

        参数:
            features: shape (batch, dim)
        返回:
            normalized_entropy: shape (batch,)
        """
        # 归一化为概率分布
        p = F.softmax(torch.abs(features), dim=-1)

        # Shannon熵
        entropy = -torch.sum(p * torch.log(p + self.eps), dim=-1)

        # 归一化
        log_d = np.log(features.shape[-1])
        normalized_entropy = entropy / log_d

        return normalized_entropy

    def compute_information_gain(self, logits):
        """
        计算信息增益 IḠ

        参数:
            logits: shape (batch, n_classes)
        返回:
            normalized_ig: shape (batch,)
        """
        # 预测分布
        q = F.softmax(logits, dim=-1)

        # 预测熵
        H_pred = -torch.sum(q * torch.log(q + self.eps), dim=-1)

        # 信息增益
        IG = self.log_C - H_pred

        # 归一化
        normalized_ig = IG / self.log_C

        return normalized_ig

    def compute_correctness(self, logits, labels, pred_entropy):
        """
        计算预测正确性 C

        参数:
            logits: shape (batch, n_classes)
            labels: shape (batch,)
            pred_entropy: shape (batch,)
        返回:
            correctness: shape (batch,)
        """
        # 预测类别
        preds = torch.argmax(logits, dim=-1)

        # 正确性
        correct = (preds == labels).float()

        # 错误时的惩罚（熵越低越自信，惩罚越大）
        wrong_penalty = -torch.exp(-self.alpha * pred_entropy)

        # C = 1 if correct, else wrong_penalty
        correctness = correct + (1 - correct) * wrong_penalty

        return correctness

    def compute_redundancy(self, features_list):
        """
        计算模态间冗余度 R

        参数:
            features_list: list of tensors, each shape (batch, dim)
        返回:
            redundancy: shape (n_modalities, batch)
        """
        batch_size = features_list[0].shape[0]
        redundancy = torch.zeros(self.n_modalities, batch_size, device=features_list[0].device)

        for i in range(self.n_modalities):
            similarities = []
            for j in range(self.n_modalities):
                if i != j:
                    # Cosine相似度
                    sim = F.cosine_similarity(
                        features_list[i],
                        features_list[j],
                        dim=-1
                    )
                    similarities.append(torch.abs(sim))

            # 平均冗余度
            redundancy[i] = torch.stack(similarities).mean(dim=0)

        return redundancy

    def compute_miei_sample(
        self,
        features_list,
        logits_list,
        labels
    ):
        """
        计算样本级MIEI

        参数:
            features_list: list of tensors, each shape (batch, dim)
            logits_list: list of tensors, each shape (batch, n_classes)
            labels: shape (batch,)
        返回:
            miei: shape (n_modalities, batch)
        """
        batch_size = features_list[0].shape[0]
        miei = torch.zeros(self.n_modalities, batch_size, device=features_list[0].device)

        # 计算各组件
        for i in range(self.n_modalities):
            # 1. 特征熵
            h_feature = self.compute_feature_entropy(features_list[i])

            # 2. 信息增益
            ig = self.compute_information_gain(logits_list[i])

            # 3. 预测熵（用于正确性计算）
            q = F.softmax(logits_list[i], dim=-1)
            h_pred = -torch.sum(q * torch.log(q + self.eps), dim=-1)

            # 4. 正确性
            c = self.compute_correctness(logits_list[i], labels, h_pred)

            # 5. 独特性（1 - 冗余度）
            r = self.compute_redundancy(features_list)[i]
            uniqueness = 1 - r

            # 综合MIEI（使用指数加权）
            miei[i] = (
                torch.pow(h_feature + self.eps, self.beta[0]) *
                torch.pow(ig + self.eps, self.beta[1]) *
                torch.pow(torch.abs(c) + self.eps, self.beta[2]) *
                torch.sign(c) *  # 保留正负号
                torch.pow(uniqueness + self.eps, self.beta[3])
            )

        return miei

    def compute_miei_modality(self, miei_sample):
        """
        计算模态级MIEI（样本级的平均）

        参数:
            miei_sample: shape (n_modalities, batch)
        返回:
            miei_modality: shape (n_modalities,)
        """
        return miei_sample.mean(dim=1)

    def diagnose_imbalance(
        self,
        miei_sample,
        tau_sample=0.1,
        tau_modality=0.05
    ):
        """
        诊断不平衡类型

        参数:
            miei_sample: shape (n_modalities, batch)
            tau_sample: 样本级方差阈值
            tau_modality: 模态级方差阈值
        返回:
            diagnosis: dict
        """
        # 模态级MIEI
        miei_modality = self.compute_miei_modality(miei_sample)

        # 模态级方差
        var_modality = torch.var(miei_modality).item()

        # 样本级方差
        var_sample = torch.var(miei_sample, dim=0)  # shape (batch,)
        mean_var_sample = var_sample.mean().item()

        # 诊断
        modality_balanced = var_modality < tau_modality
        sample_balanced = mean_var_sample < tau_sample

        # 识别主导和被抑制模态
        dominant_modality = torch.argmax(miei_modality).item()
        suppressed_modality = torch.argmin(miei_modality).item()

        # 场景分类
        if modality_balanced and sample_balanced:
            scenario = "场景1: 模态级平衡 + 样本级平衡"
            strategy = "无需干预"
        elif not modality_balanced and sample_balanced:
            scenario = "场景2: 模态级不平衡 + 样本级平衡"
            strategy = "使用模态级方法（OGM, 模态级重采样）"
        elif modality_balanced and not sample_balanced:
            scenario = "场景3: 模态级平衡 + 样本级不平衡"
            strategy = "使用样本级方法（OPM, 样本级重采样）"
        else:
            scenario = "场景4: 模态级不平衡 + 样本级不平衡"
            strategy = "使用混合策略（DMCG推荐）"

        return {
            'scenario': scenario,
            'strategy': strategy,
            'var_modality': var_modality,
            'mean_var_sample': mean_var_sample,
            'modality_balanced': modality_balanced,
            'sample_balanced': sample_balanced,
            'dominant_modality': dominant_modality,
            'suppressed_modality': suppressed_modality,
            'miei_modality': miei_modality.detach().cpu().numpy(),
            'miei_sample': miei_sample.detach().cpu().numpy()
        }

    def get_decomposition(self, features_list, logits_list, labels):
        """
        获取MIEI的各组件分解（用于可解释性分析）

        返回:
            decomposition: dict of tensors
        """
        decomposition = {}

        for i in range(self.n_modalities):
            # 各组件
            h_feature = self.compute_feature_entropy(features_list[i])
            ig = self.compute_information_gain(logits_list[i])

            q = F.softmax(logits_list[i], dim=-1)
            h_pred = -torch.sum(q * torch.log(q + self.eps), dim=-1)
            c = self.compute_correctness(logits_list[i], labels, h_pred)

            r = self.compute_redundancy(features_list)[i]
            uniqueness = 1 - r

            decomposition[f'modality_{i}'] = {
                'feature_entropy': h_feature.detach().cpu().numpy(),
                'information_gain': ig.detach().cpu().numpy(),
                'correctness': c.detach().cpu().numpy(),
                'uniqueness': uniqueness.detach().cpu().numpy()
            }

        return decomposition
