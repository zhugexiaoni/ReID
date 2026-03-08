# C‑MIEI：Counterfactual Modality Influence Equalization (Feature Intervention)

## 1. 背景与动机
在非平衡多模态 ReID 中，融合分支往往会长期依赖“强模态”（例如 RGB），导致弱模态（NI/TI）在训练过程中逐渐被忽视：

- 融合预测对强模态特征高度敏感
- 弱模态信息难以进入最终判别表示
- 最终出现“单模态主导”的退化

**C‑MIEI** 的目标是：在训练过程中**动态测量融合输出对各模态的依赖程度**，当发现某个模态对当前样本/批次的融合预测具有过强“因果影响”时，执行一个**反事实式（counterfactual）特征干预**，迫使模型在该 step 更多利用其它模态信息。

C‑MIEI 属于 **plugin / intervention（特征级干预）**，不是单纯增加 loss、也不依赖额外训练分类头。

---

## 2. 核心思想：反事实影响力（Counterfactual Influence, CI）
设三模态全局特征为 `f_r, f_n, f_t`（RGB/NI/TI），融合分类头输出 logits：

- 正常融合 logits：

\[
z = Head([f_r, f_n, f_t])
\]

对每个模态做一次“反事实剥离”（drop）——只替换该模态输入，其余不变，且**不重跑 backbone**：

- 去掉 RGB 的反事实 logits：

\[
z^{(-r)} = Head([Drop(f_r), f_n, f_t])
\]

类似可得 `z^{(-n)}`、`z^{(-t)}`。

### 2.1 CI 定义（KL 版本）
对每个样本 i、每个模态 m，定义：

\[
CI_{i,m} = KL(softmax(z_i) \;\|\; softmax(z_i^{(-m)}))
\]

直觉：如果去掉某模态后预测分布变化很大，则说明融合预测对该模态“依赖强”。

---

## 3. 干预机制：Substitution（特征替换）
当 CI 显示某模态对当前样本依赖过强时，对该模态特征执行“信息削弱式替换”，而不是调节梯度。

### 3.1 substitute()（推荐默认）
使用 batch prototype + 小噪声替换，使其保留尺度但去除判别性：

\[
Sub(f) = \mu_{batch}(f) + \sigma \cdot \epsilon
\]

- \(\mu_{batch}(f)\)：batch 维均值，broadcast 到每个样本
- \(\epsilon\)：标准高斯噪声
- \(\sigma\)：噪声强度（默认 0.05）

**作用**：被替换模态仍存在数值输入，但身份判别信息被显著削弱，融合 head 为了降低损失会被迫利用其它模态。

---

## 4. Sample-level 版本（已实现）
C‑MIEI 支持样本级非平衡评估与干预。

### 4.1 per-sample 触发规则
对样本 i：

- `m1 = argmax_m CI_{i,m}`，`m2` 为第二大
- 触发条件：
  - `CI_{i,m1} > ABS_THR`
  - `CI_{i,m1} / (CI_{i,m2} + eps) > REL_THR`

触发后对该样本的 dominant 模态执行 substitution。

### 4.2 干预比例上限（p_max）
为避免过猛导致训练不稳定，每次估计时最多干预 `p_max` 比例的样本：

- 仅对触发样本里 `dominant CI` 最大的 top‑k 执行 substitution
- 默认 `p_max = 0.5`

### 4.3 warmup
前 `WARMUP_EPOCHS` 个 epoch **只统计 CI，不做干预**，默认 `WARMUP_EPOCHS=5`。

---

## 5. 计算频率（K=3）
为降低额外前向代价，CI 估计每 K step 执行一次（默认 K=3）：

- 只在估计 step 做：1 次正常 logits + 3 次 drop logits（仅 head 前向，不重跑 backbone）
- 非估计 step 使用缓存统计值用于日志

---

## 6. 日志与可观测指标
训练日志会打印：

- `C_MIEI_Ratio`：sample-level 干预样本比例（平均）
- `CI_r / CI_n / CI_t`：三模态 CI 的 batch mean
- `Hist[r,n,t]`：每个样本 dominant 模态选择的比例（反映“谁更常主导”）
- 额外行：epoch / step / sample_level / p_max / warmup 等配置确认

---

## 7. 配置项（默认）
在 `config/defaults.py` 中注册：

```yaml
MODEL:
  C_MIEI:
    ENABLED: False
    K: 3
    SIGMA: 0.05
    ABS_THR: 0.03
    REL_THR: 1.25
    SAMPLE_LEVEL: True/False
    P_MAX: 0.5
    WARMUP_EPOCHS: 5
```

示例配置文件：

- `configs/RGBNT201/DeMo_C_MIEI.yml`

---

## 8. 代码入口
- 训练入口：`train_cmiei.py`
- 模型：`modeling/make_model_cmiei.py`
- 插件：`layers/c_miei.py`
- processor：`engine/processor_cmiei.py`

---

## 9. 建议的验证流程（sanity + 对照）
1. **Baseline 对照**：`MODEL.C_MIEI.ENABLED False`，确认与 DeMo 原始训练轨迹一致或接近
2. **开启 C‑MIEI**：检查
   - warmup 前 `C_MIEI_Ratio≈0`
   - warmup 后 ratio > 0 且 Hist 分布合理
3. 超参敏感性：
   - `SIGMA: 0.02 / 0.05 / 0.1`
   - `P_MAX: 0.2 / 0.5`
   - `ABS_THR` 调高/调低影响触发率
