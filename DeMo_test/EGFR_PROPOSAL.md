# Entropy-Guided Feature Rectification (EGFR)
## —— 针对非平衡多模态 ReID 的特征重校准方法

## 1. 动机 (Motivation)

在 RGB-NI-TI 三模态行人重识别任务中，数据呈现高度的非平衡性（Unbalanced）：
1.  **模态级非平衡**：白天 RGB 信息量大，夜晚 TI/NI 信息量大。
2.  **样本级非平衡**：某些特定样本在所有模态下可能都存在遮挡或噪声。

传统的 Teacher-Student 蒸馏（如 IADD）依赖 Loss 约束，强制弱模态逼近强模态。但在极端非平衡下，这种软约束可能失效，或者训练初期“盲人领路”。

**我们提出一个新的角度**：不修改 Loss 函数，而是设计一个**前向特征重校准模块 (Feature Rectification Module)**。利用**预测信息熵 (Prediction Entropy)** 作为不确定性的度量指标，在特征融合前动态地“清洗”和“增强”各模态特征。

---

## 2. 选定指标：预测信息熵 (Prediction Entropy)

基于 MIEI 理论，我们选择 **香农熵 (Shannon Entropy)** 作为衡量模态置信度（Confidence）的核心指标。

对于模态 $m \in \{RGB, NI, TI\}$，假设其独立分类头（我们刚在 `make_model.py` 里加的那些）输出的概率分布为 $P^m \in \mathbb{R}^C$（$C$ 为类别数）：

$$
H(P^m) = - \sum_{k=1}^{C} p_k^m \log(p_k^m)
$$

*   **低熵 (Low Entropy)** $\rightarrow$ 分布尖锐 $\rightarrow$ 模型对该模态下的样本非常**自信**。
*   **高熵 (High Entropy)** $\rightarrow$ 分布平坦 $\rightarrow$ 模型对该模态下的样本感到**困惑/不确定**。

---

## 3. 方法架构：EGFR (Entropy-Guided Feature Rectification)

EGFR 是一个插在 Backbone 之后、Classifier 之前的即插即用模块。

### 3.1 总体流程

1.  **独立感知 (Sensing)**：利用轻量级分类头（Direct Heads）获取 $RGB, NI, TI$ 的初步 Logits，计算各自的熵值 $H_{rgb}, H_{ni}, H_{ti}$。
2.  **熵归一化 (Entropy Normalization)**：将熵值转化为模态间的相对可靠性权重。
3.  **特征重校准 (Rectification)**：利用可靠性权重，对原始特征进行**通道级加权 (Channel-wise Reweighting)** 和 **跨模态补偿 (Cross-Modality Compensation)**。

### 3.2 详细公式推导

#### A. 熵权生成 (Entropy Weight Generation)

首先，计算每个样本 $i$ 在三个模态下的熵向量 $\mathbf{h}_i = [H_i^{RGB}, H_i^{NI}, H_i^{TI}]$。
为了得到通过概率（权重），我们需要对熵取反并归一化（熵越小，权重越大）：

$$
w_i^m = \frac{e^{-H_i^m / \tau}}{\sum_{k \in \{RGB, NI, TI\}} e^{-H_i^k / \tau}}
$$

其中 $\tau$ 是温度系数。$w_i^m$ 反映了模态 $m$ 相对于其他模态的**“相对置信度”**。

#### B. 动态特征门控 (Dynamic Feature Gating)

传统的特征融合通常是简单的 `Concat` 或 `Add`，忽略了特征本身的质量。EGFR 引入门控机制。
设原始特征为 $F_i^{RGB}, F_i^{NI}, F_i^{TI} \in \mathbb{R}^D$。

我们不直接使用 $w_i^m$ 标量乘法，因为特征的不同通道可能承载不同语义。我们利用 $w_i^m$ 生成一个**通道注意力向量**：

$$
\mathbf{A}_i^m = \sigma(\text{MLP}(F_i^m) \cdot w_i^m)
$$

这里 $\mathbf{A}_i^m \in \mathbb{R}^D$。将此注意力应用回特征：

$$
\tilde{F}_i^m = F_i^m \odot (1 + \mathbf{A}_i^m)
$$

**物理意义**：如果某模态熵很低（$w_i^m$ 大），则 MLP 激活值变大，显著增强该模态的特征响应；如果熵很高（不确定），则 $w_i^m \to 0$，特征保持原状或被相对抑制（取决于后续融合）。

#### C. 跨模态互补 (Cross-Modality Complementation) - 创新点

这是解决“非平衡”的关键。如果 $NI$ 模态置信度极低（高熵），单纯抑制它会导致信息丢失。我们利用“高置信度模态”来修补“低置信度模态”。

定义全局加权上下文向量 (Global Context Vector) $G_i$：

$$
G_i = \sum_{m \in \{RGB, NI, TI\}} w_i^m \cdot \tilde{F}_i^m
$$

$G_i$ 代表了当前样本在所有模态中**最可信特征的加权共识**。

然后，对每个模态进行残差修补：

$$
F_{i, final}^m = \tilde{F}_i^m + \gamma \cdot \text{Transform}(G_i)
$$

**效果**：
- 对于强模态（$w$ 大）：$G_i$ 主要由它自己贡献，相当于自我增强。
- 对于弱模态（$w$ 小）：$G_i$ 主要来自其他强模态，相当于**引入了强模态的信息来填补自己的噪声**。

---

## 4. 为什么这个设计优于 IADD？

1.  **前向干预 vs 后向约束**：
    IADD 依赖梯度回传（Loss）来优化参数，这在训练初期不稳定。EGFR 直接在 Forward 阶段根据当前输入质量动态调整特征，推理阶段依然有效（Inference-Aware）。

2.  **处理“全员恶人”的情况**：
    如果三个模态熵都高（$H$ 都大），$w$ 会趋于均匀分布（Softmax性质），EGFR 退化为平均融合，避免了 IADD 中“强制一个弱者去教另一个弱者”的风险。

3.  **无需复杂的 Loss 调参**：
    不需要调整 $\lambda_{distill}$ 或 $\lambda_{hybrid}$，只需要关注模块内部的简单参数。

---

## 5. 论文撰写切入点 (Storytelling)

*   **Problem**: Unbalanced Multimodal Learning usually treats features statically.
*   **Insight**: The "Uncertainty" (Entropy) of a modality varies instance-by-instance.
*   **Proposal**: **Entropy-Guided Feature Rectification (EGFR)**.
*   **Mechanism**: "Listen to the confident modality, silence the confused one, and use the global consensus to repair the broken one."
