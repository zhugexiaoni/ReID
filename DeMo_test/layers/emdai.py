import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalInterventionFunction(torch.autograd.Function):
    """
    正交梯度干预算子 (Orthogonal Gradient Intervention Operator)
    
    Forward: 透传特征，记录上下文。
    Backward: 将梯度的投影分量剔除，实现正交化。
    """
    @staticmethod
    def forward(ctx, feat_weak, feat_strong, dominance_mask):
        ctx.save_for_backward(feat_strong, dominance_mask)
        return feat_weak.clone()

    @staticmethod
    def backward(ctx, grad_output):
        feat_strong, dominance_mask = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 如果没有样本需要干预，直接返回
        if not dominance_mask.any():
            return grad_input, None, None

        # 为了高效，我们只处理 mask 为 True 的索引
        indices = torch.nonzero(dominance_mask).squeeze(1)
        
        if len(indices) > 0:
            # 提取对应的特征和梯度
            # s: Strong modality features [N, D]
            s = feat_strong[indices]
            # g: Gradient of weak modality [N, D]
            g = grad_input[indices]

            # 计算 s 的模平方，避免除零
            s_norm_sq = torch.sum(s * s, dim=1, keepdim=True) + 1e-8
            
            # 计算投影系数: (g . s) / |s|^2
            dot_product = torch.sum(g * s, dim=1, keepdim=True)
            proj_coeff = dot_product / s_norm_sq
            
            # 计算投影向量
            projection = proj_coeff * s
            
            # 执行正交化：梯度减去投影
            grad_input[indices] = g - projection
        
        return grad_input, None, None


class EMDAIPlugin(nn.Module):
    """
    Entropy-Guided Modality Decoupling & Alignment Intervention (E-MDAI) Plugin
    即插即用的正交干预模块。
    """
    def __init__(self, threshold=0.4, num_classes=None):
        super(EMDAIPlugin, self).__init__()
        self.threshold = threshold
        # 用于熵归一化的最大熵常数
        if num_classes is not None:
            self.max_entropy = torch.log(torch.tensor(float(num_classes)))
        else:
            self.max_entropy = None # 如果未提供，将在 forward 中动态处理或不归一化

    def compute_entropy(self, logits):
        """计算预测分布的香农熵"""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy

    def forward(self, weak_feats, strong_feats, strong_logits):
        """
        Args:
            weak_feats: 需要被干预（引导）的弱势模态特征 (NI/TI)
            strong_feats: 作为参考基准的强势模态特征 (RGB)
            strong_logits: 强势模态的 logits，用于计算熵
        Returns:
            intervened_weak_feats: 经过梯度钩子处理后的特征
            intervention_stats: 干预统计信息（如触发比例）
        """
        # 1. 计算强势模态的熵
        entropy = self.compute_entropy(strong_logits)
        
        # 2. 生成主导性 Mask (Dominance Mask)
        # 归一化熵
        if self.max_entropy is not None:
            norm_entropy = entropy / (self.max_entropy.to(entropy.device) + 1e-8)
        else:
            # 简单的自适应归一化：除以当前 batch 的最大值
            norm_entropy = entropy / (entropy.max() + 1e-8)

        # 熵越低 -> 越自主/越强 -> 需要干预弱势模态使其正交
        dominance_mask = norm_entropy < self.threshold
        
        # 3. 应用正交梯度干预
        intervened_feats = OrthogonalInterventionFunction.apply(weak_feats, strong_feats, dominance_mask)
        
        stats = {
            'intervention_ratio': dominance_mask.float().mean().item(),
            'avg_entropy': entropy.mean().item()
        }
        
        return intervened_feats, stats
