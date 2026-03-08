"""
Instance-Aware Dynamic Distillation (IADD) Plugin

实现非平衡多模态学习中的样本级动态蒸馏和难样本挖掘
(Robust & Vectorized Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IADDPlugin(nn.Module):
    """
    Instance-Aware Dynamic Distillation (IADD) Plugin
    """

    def __init__(self, temperature=2.0, hard_neg_k=10, lambda_distill=1.0, lambda_hybrid=0.5):
        super(IADDPlugin, self).__init__()
        self.T = temperature
        self.k = hard_neg_k
        self.lambda_distill = lambda_distill
        self.lambda_hybrid = lambda_hybrid
        
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def compute_mcd_vectorized(self, features, labels):
        """
        全向量化计算 MCD (修复版)
        """
        B = features.size(0)
        
        # --- DEBUG SECTION START ---
        # if torch.rand(1).item() < 0.1: # Sample 10% of batches to print
        #     with torch.no_grad():
        #         print(f"\n[IADD DEBUG] Batch Size: {B}")
        #         print(f"[IADD DEBUG] Features: Mean={features.mean().item():.4f}, Std={features.std().item():.4f}, Norm={features.norm(dim=1).mean().item():.4f}")
        #         print(f"[IADD DEBUG] Labels (first 10): {labels[:10].tolist()}")
        # --- DEBUG SECTION END ---

        # 1. 相似度矩阵 (B, B) -> 范围 [-1, 1]
        sim_matrix = torch.mm(features, features.t())

        # 2. 构建掩码
        labels = labels.view(B, 1)
        is_pos = torch.eq(labels, labels.t())
        
        # 排除对角线
        eye = torch.eye(B, device=features.device, dtype=torch.bool)
        is_pos = is_pos & ~eye
        is_neg = ~is_pos & ~eye

        # --- DEBUG SECTION START ---
        # if torch.rand(1).item() < 0.1:
        #     pos_pairs = is_pos.sum().item()
        #     neg_pairs = is_neg.sum().item()
        #     print(f"[IADD DEBUG] Positive Pairs: {pos_pairs}, Negative Pairs: {neg_pairs}")
        #     if pos_pairs == 0:
        #         print("[IADD WARNING] No positive pairs found! Check if PK Sampler is working.")
        # --- DEBUG SECTION END ---

        # --- Part A: 类内紧凑度 ---
        # 必须处理 pos_counts 为 0 的情况 (尽管很少见)
        pos_counts = is_pos.sum(dim=1).float()
        pos_sum = (sim_matrix * is_pos.float()).sum(dim=1)
        
        # 避免除以 0
        pos_scores = torch.zeros_like(pos_sum)
        mask_valid_pos = pos_counts > 0
        pos_scores[mask_valid_pos] = pos_sum[mask_valid_pos] / pos_counts[mask_valid_pos]

        # --- Part B: 类间分离度 ---
        # 使用 -2.0 作为填充值 (因为 Cosine Sim 最小是 -1.0)
        neg_matrix_for_topk = sim_matrix.clone()
        neg_matrix_for_topk.masked_fill_(~is_neg, -2.0)

        # 动态调整 K (防止 Batch Size 小于 K)
        actual_k = min(self.k, B - 1)
        if actual_k > 0:
            # Check if we have enough negatives
            neg_counts = is_neg.sum(dim=1)
            # If any sample has fewer negatives than K (very rare in ReID), cap K for that sample?
            # Torch topk requires constant K. So we use global min or fallback.
            # Usually ReID batch size (64) >> K (10) and P*K=4*16. Negatives are plenty.
            
            # Robustness: ensure we don't crash if negs < k
            if neg_counts.min() < actual_k:
                 actual_k = neg_counts.min().item()

            if actual_k > 0:
                hard_neg_sims, _ = torch.topk(neg_matrix_for_topk, k=actual_k, dim=1)
                neg_scores = hard_neg_sims.mean(dim=1)
            else:
                neg_scores = torch.zeros_like(pos_scores)
        else:
            neg_scores = torch.zeros_like(pos_scores)

        # --- Part C: 差分 ---
        mcd_scores = pos_scores - neg_scores
        
        # --- DEBUG SECTION START ---
        # if torch.rand(1).item() < 0.1:
        #     print(f"[IADD DEBUG] PosScores Mean: {pos_scores.mean().item():.4f}")
        #     print(f"[IADD DEBUG] NegScores Mean: {neg_scores.mean().item():.4f}")
        #     print(f"[IADD DEBUG] MCD Mean: {mcd_scores.mean().item():.4f}")
        # --- DEBUG SECTION END ---

        return mcd_scores

    def get_hybrid_dist_matrix(self, dist_m1, dist_m2, weight_m1):
        W = weight_m1.view(-1, 1).expand_as(dist_m1)
        dist_hybrid = W * dist_m1 + (1 - W) * dist_m2
        return dist_hybrid

    @staticmethod
    def cross_modal_dist(feat_a, feat_b):
        """Compute Euclidean distance matrix (sqrt L2), aligned with layers/triplet_loss.py::euclidean_dist.

        Args:
            feat_a: (B, D)
            feat_b: (B, D)
        Returns:
            dist: (B, B) with dist[i,j] = ||a_i - b_j||_2

        Notes:
            - We clamp to avoid sqrt(0) / numerical issues, matching DeMo triplet implementation.
            - Do NOT square distances here; original TripletLoss uses sqrt-L2.
        """
        dist = torch.cdist(feat_a, feat_b, p=2)
        return dist.clamp(min=1e-12)

    def teacher2students(self, teacher_logits, students_logits, teacher_feats, students_feats, labels,
                         temperature=None, lambda_distill=None, lambda_hybrid=None):
        """Teacher teaches two students (no fusion), and build cross-modal hybrid distance.

        Distill loss:
            KL(student1 || teacher) + KL(student2 || teacher)
        Hybrid distance (per pair i,j):
            For each student k:
              w_{i,k} * ||T_i - S_{k,j}|| + (1-w_{i,k}) * ||S_{k,i} - T_j||
            Final D = (D_k1 + D_k2) / 2

        Direction weights w_{i,k} are computed from per-sample MCD gaps:
            w_{i,k} = sigmoid((mcd_T(i) - mcd_Sk(i)) * T2S_T)
        """
        assert len(students_logits) == 2 and len(students_feats) == 2, "Expect exactly 2 students"

        T2S_T = float(temperature) if temperature is not None else float(self.T)
        lam_d = float(lambda_distill) if lambda_distill is not None else float(self.lambda_distill)
        lam_h = float(lambda_hybrid) if lambda_hybrid is not None else float(self.lambda_hybrid)

        # normalize features
        t_feat = F.normalize(teacher_feats, p=2, dim=1)
        s_feat1 = F.normalize(students_feats[0], p=2, dim=1)
        s_feat2 = F.normalize(students_feats[1], p=2, dim=1)

        # per-sample MCD for direction weights
        mcd_t = self.compute_mcd_vectorized(t_feat, labels)              # (B,)
        mcd_s1 = self.compute_mcd_vectorized(s_feat1, labels)            # (B,)
        mcd_s2 = self.compute_mcd_vectorized(s_feat2, labels)            # (B,)
        w1 = self.sigmoid((mcd_t - mcd_s1) * T2S_T)                      # (B,)
        w2 = self.sigmoid((mcd_t - mcd_s2) * T2S_T)                      # (B,)

        # distillation: teacher -> each student (teacher detached)
        prob_t = F.softmax(teacher_logits, dim=1).detach()
        log_prob_s1 = F.log_softmax(students_logits[0], dim=1)
        log_prob_s2 = F.log_softmax(students_logits[1], dim=1)
        loss_s1 = self.kl_loss(log_prob_s1, prob_t).sum(dim=1).mean()
        loss_s2 = self.kl_loss(log_prob_s2, prob_t).sum(dim=1).mean()
        loss_distill = (loss_s1 + loss_s2) * lam_d

        # cross-modal hybrid dist (sqrt-L2, aligned with DeMo triplet)
        d_t_s1 = self.cross_modal_dist(t_feat, s_feat1)                  # (B,B)
        d_s1_t = self.cross_modal_dist(s_feat1, t_feat)                  # (B,B)
        d_t_s2 = self.cross_modal_dist(t_feat, s_feat2)                  # (B,B)
        d_s2_t = self.cross_modal_dist(s_feat2, t_feat)                  # (B,B)

        W1 = w1.view(-1, 1).expand_as(d_t_s1)
        W2 = w2.view(-1, 1).expand_as(d_t_s2)
        hybrid1 = W1 * d_t_s1 + (1.0 - W1) * d_s1_t
        hybrid2 = W2 * d_t_s2 + (1.0 - W2) * d_s2_t
        hybrid_dist = (hybrid1 + hybrid2) * 0.5

        return {
            "loss_distill": loss_distill,
            "hybrid_dist": hybrid_dist,
            "mcd_teacher": mcd_t.mean().item(),
            "mcd_student1": mcd_s1.mean().item(),
            "mcd_student2": mcd_s2.mean().item(),
            "w_t2s1": w1.mean().item(),
            "w_t2s2": w2.mean().item(),
            "lambda_hybrid": lam_h,
        }

    def teacher2students_instancewise(self, logits_rgb, logits_ni, logits_ti,
                                      feats_rgb, feats_ni, feats_ti, labels,
                                      temperature=None, lambda_distill=None):
        """Instance-wise teacher selection for 3 modalities (RGB/NI/TI).

        For each sample i, choose teacher modality as argmax of per-sample MCD among {RGB, NI, TI}.
        The remaining two modalities are students. Teacher distills to both students.

        Hybrid distance D(i,j) (sqrt-L2) is cross-modal and bi-directional for each student and
        averaged over two students:

            D(i,j)=0.5 * sum_{k in students(i)} [ w_{i,k} * ||T_i - S_{k,j}|| + (1-w_{i,k}) * ||S_{k,i} - T_j|| ]

        where w_{i,k} = sigmoid((mcd_T(i) - mcd_Sk(i)) * T2S_T)
        """
        T2S_T = float(temperature) if temperature is not None else float(self.T)
        lam_d = float(lambda_distill) if lambda_distill is not None else float(self.lambda_distill)

        # normalize
        fr = F.normalize(feats_rgb, p=2, dim=1)
        fn = F.normalize(feats_ni, p=2, dim=1)
        ft = F.normalize(feats_ti, p=2, dim=1)

        # per-sample MCD
        mcd_r = self.compute_mcd_vectorized(fr, labels)  # (B,)
        mcd_n = self.compute_mcd_vectorized(fn, labels)
        mcd_t = self.compute_mcd_vectorized(ft, labels)
        mcd_stack = torch.stack([mcd_r, mcd_n, mcd_t], dim=1)  # (B,3)
        teacher_idx = torch.argmax(mcd_stack, dim=1)  # (B,)

        # distillation loss (teacher -> each student), computed per sample then averaged
        # probs/log-probs
        prob_r = F.softmax(logits_rgb, dim=1)
        prob_n = F.softmax(logits_ni, dim=1)
        prob_ti = F.softmax(logits_ti, dim=1)
        logprob_r = F.log_softmax(logits_rgb, dim=1)
        logprob_n = F.log_softmax(logits_ni, dim=1)
        logprob_t = F.log_softmax(logits_ti, dim=1)

        # KLDivLoss reduction='none' returns (B,C); sum over C -> (B,)
        kl_n_from_r = self.kl_loss(logprob_n, prob_r.detach()).sum(dim=1)
        kl_t_from_r = self.kl_loss(logprob_t, prob_r.detach()).sum(dim=1)
        kl_r_from_n = self.kl_loss(logprob_r, prob_n.detach()).sum(dim=1)
        kl_t_from_n = self.kl_loss(logprob_t, prob_n.detach()).sum(dim=1)
        kl_r_from_t = self.kl_loss(logprob_r, prob_ti.detach()).sum(dim=1)
        kl_n_from_t = self.kl_loss(logprob_n, prob_ti.detach()).sum(dim=1)

        # select per sample
        loss_distill_vec = torch.zeros_like(teacher_idx, dtype=fr.dtype)
        # teacher=RGB -> students NI, TI
        mask_r = teacher_idx == 0
        loss_distill_vec[mask_r] = kl_n_from_r[mask_r] + kl_t_from_r[mask_r]
        # teacher=NI -> students RGB, TI
        mask_n = teacher_idx == 1
        loss_distill_vec[mask_n] = kl_r_from_n[mask_n] + kl_t_from_n[mask_n]
        # teacher=TI -> students RGB, NI
        mask_t = teacher_idx == 2
        loss_distill_vec[mask_t] = kl_r_from_t[mask_t] + kl_n_from_t[mask_t]

        loss_distill = loss_distill_vec.mean() * lam_d

        # cross-modal distances (sqrt-L2, aligned with DeMo triplet)
        d_rr = self.cross_modal_dist(fr, fr)
        d_nn = self.cross_modal_dist(fn, fn)
        d_tt = self.cross_modal_dist(ft, ft)
        d_rn = self.cross_modal_dist(fr, fn)
        d_nr = self.cross_modal_dist(fn, fr)
        d_rt = self.cross_modal_dist(fr, ft)
        d_tr = self.cross_modal_dist(ft, fr)
        d_nt = self.cross_modal_dist(fn, ft)
        d_tn = self.cross_modal_dist(ft, fn)

        # direction weights per student (per sample)
        # We need w for (teacher vs student) pairs for each i depending on teacher choice.
        w = torch.zeros((fr.size(0), 2), device=fr.device, dtype=fr.dtype)  # (B,2)
        # store which students indices for logging: sidx[i,0/1] in {0,1,2}
        sidx = torch.empty((fr.size(0), 2), device=fr.device, dtype=torch.long)

        # teacher RGB: students NI, TI
        sidx[mask_r, 0] = 1
        sidx[mask_r, 1] = 2
        w[mask_r, 0] = self.sigmoid((mcd_r - mcd_n)[mask_r] * T2S_T)
        w[mask_r, 1] = self.sigmoid((mcd_r - mcd_t)[mask_r] * T2S_T)
        # teacher NI: students RGB, TI
        sidx[mask_n, 0] = 0
        sidx[mask_n, 1] = 2
        w[mask_n, 0] = self.sigmoid((mcd_n - mcd_r)[mask_n] * T2S_T)
        w[mask_n, 1] = self.sigmoid((mcd_n - mcd_t)[mask_n] * T2S_T)
        # teacher TI: students RGB, NI
        sidx[mask_t, 0] = 0
        sidx[mask_t, 1] = 1
        w[mask_t, 0] = self.sigmoid((mcd_t - mcd_r)[mask_t] * T2S_T)
        w[mask_t, 1] = self.sigmoid((mcd_t - mcd_n)[mask_t] * T2S_T)

        # Build hybrid dist row-wise (each row i uses its teacher/students & weights)
        B = fr.size(0)
        hybrid = torch.zeros((B, B), device=fr.device, dtype=fr.dtype)

        # helper to pick distance matrices by modality indices
        dist_map = {
            (0, 0): d_rr,
            (1, 1): d_nn,
            (2, 2): d_tt,
            (0, 1): d_rn,
            (1, 0): d_nr,
            (0, 2): d_rt,
            (2, 0): d_tr,
            (1, 2): d_nt,
            (2, 1): d_tn,
        }

        # process three teacher groups to avoid python loops over i
        def add_group(mask, teacher_mod):
            if not torch.any(mask):
                return
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            # students for this group are fixed by teacher_mod
            if teacher_mod == 0:
                st = [1, 2]
                wloc = torch.stack([
                    self.sigmoid((mcd_r - mcd_n)[mask] * T2S_T),
                    self.sigmoid((mcd_r - mcd_t)[mask] * T2S_T)
                ], dim=1)
            elif teacher_mod == 1:
                st = [0, 2]
                wloc = torch.stack([
                    self.sigmoid((mcd_n - mcd_r)[mask] * T2S_T),
                    self.sigmoid((mcd_n - mcd_t)[mask] * T2S_T)
                ], dim=1)
            else:
                st = [0, 1]
                wloc = torch.stack([
                    self.sigmoid((mcd_t - mcd_r)[mask] * T2S_T),
                    self.sigmoid((mcd_t - mcd_n)[mask] * T2S_T)
                ], dim=1)

            # distances: teacher_i - student_j and student_i - teacher_j
            d_t_s1 = dist_map[(teacher_mod, st[0])][mask]  # (n,B)
            d_s1_t = dist_map[(st[0], teacher_mod)][mask]
            d_t_s2 = dist_map[(teacher_mod, st[1])][mask]
            d_s2_t = dist_map[(st[1], teacher_mod)][mask]

            W1 = wloc[:, 0].view(-1, 1)
            W2 = wloc[:, 1].view(-1, 1)
            h1 = W1 * d_t_s1 + (1.0 - W1) * d_s1_t
            h2 = W2 * d_t_s2 + (1.0 - W2) * d_s2_t
            hybrid[mask] = (h1 + h2) * 0.5

        add_group(mask_r, 0)
        add_group(mask_n, 1)
        add_group(mask_t, 2)

        return {
            "loss_distill": loss_distill,
            "hybrid_dist": hybrid,
            "teacher_idx": teacher_idx,
            "mcd_rgb": mcd_r.mean().item(),
            "mcd_ni": mcd_n.mean().item(),
            "mcd_ti": mcd_t.mean().item(),
            "w_mean": w.mean().item(),
        }

    def forward(self, logits_m1, logits_m2, feats_m1, feats_m2, labels):
        # 0. 特征归一化 (关键！如果不归一化，点积不是相似度)
        feats_m1_norm = F.normalize(feats_m1, p=2, dim=1)
        feats_m2_norm = F.normalize(feats_m2, p=2, dim=1)

        # 1. 计算 MCD
        mcd_m1 = self.compute_mcd_vectorized(feats_m1_norm, labels)
        mcd_m2 = self.compute_mcd_vectorized(feats_m2_norm, labels)
        
        # DEBUG: Ensure MCD is not 0
        if mcd_m1.abs().sum() < 1e-6:
             pass # Placeholder for breakpoint

        # 2. 动态权重
        # 此时 mcd 应该在 [-2, 2] 之间，通常在 [0, 1] 附近
        gap = mcd_m1 - mcd_m2
        alpha = self.sigmoid(gap * self.T)

        # 3. 蒸馏 Loss
        log_prob_m1 = F.log_softmax(logits_m1, dim=1)
        log_prob_m2 = F.log_softmax(logits_m2, dim=1)
        prob_m1 = F.softmax(logits_m1, dim=1)
        prob_m2 = F.softmax(logits_m2, dim=1)

        loss_m1_teach_m2 = self.kl_loss(log_prob_m2, prob_m1.detach()).sum(dim=1) 
        loss_m2_teach_m1 = self.kl_loss(log_prob_m1, prob_m2.detach()).sum(dim=1)

        loss_distill = (alpha * loss_m1_teach_m2 + (1 - alpha) * loss_m2_teach_m1).mean()

        # 4. 混合距离 (sqrt-L2, aligned with DeMo triplet)
        dist_m1 = self.cross_modal_dist(feats_m1_norm, feats_m1_norm)
        dist_m2 = self.cross_modal_dist(feats_m2_norm, feats_m2_norm)
        
        hybrid_dist = self.get_hybrid_dist_matrix(dist_m1, dist_m2, alpha)

        return {
            "loss_distill": loss_distill * self.lambda_distill,
            "hybrid_dist": hybrid_dist,
            "mcd_m1": mcd_m1.mean().item(),
            "mcd_m2": mcd_m2.mean().item(),
            "alpha_mean": alpha.mean().item()
        }

class HybridTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(HybridTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, dist_mat, labels):
        n = dist_mat.size(0)
        labels = labels.view(n, 1)
        is_pos = torch.eq(labels, labels.t())
        
        # Hard Positive
        dist_ap, _ = torch.max(dist_mat * is_pos.float(), dim=1)
        
        # Hard Negative
        dist_mat_neg = dist_mat.clone()
        dist_mat_neg.masked_fill_(is_pos, float('inf'))
        dist_an, _ = torch.min(dist_mat_neg, dim=1)
        
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
