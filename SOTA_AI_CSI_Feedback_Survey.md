# AI-Based CSI 压缩反馈 SOTA 文献综述

> 综述当前利用人工智能（深度学习）进行 CSI（信道状态信息）压缩反馈的前沿方法，
> 列出代表性文献，并汇总每篇文献的关键性能指标。
>
> **数据集**：除特殊说明外，均使用 **COST2100** 信道模型生成的室内（Indoor, 5.3 GHz）和室外（Outdoor, 300 MHz）数据集进行评估。  
> **评估指标**：主要使用 **NMSE**（归一化均方误差，dB，越小越好）和 **ρ**（余弦相似度，越大越好）。  
> **压缩比（CR）**：η = 1/4、1/8、1/16、1/32、1/64，表示反馈维度与原始维度之比。

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [文献列表与性能指标](#2-文献列表与性能指标)
   - [CsiNet（2018）](#21-csinet2018)
   - [CsiNet-LSTM（2019）](#22-csinet-lstm2019)
   - [CsiNet+（2020）](#23-csinet2020)
   - [CRNet（2020）](#24-crnet2020)
   - [CLNet（2021）](#25-clnet2021)
   - [BCsiNet（2021）](#26-bcsinet2021)
   - [TransNet（2022）](#27-transnet2022)
   - [CSI-StripeFormer（2023）](#28-csi-stripeformer2023)
   - [TCNet（2025）](#29-tcnet2025)
3. [综合性能对比表](#3-综合性能对比表)
4. [发展趋势总结](#4-发展趋势总结)
5. [参考文献](#5-参考文献)

---

## 1. 背景与问题定义

在大规模 MIMO 的 FDD（频分双工）系统中，下行信道状态信息（CSI）需要由用户设备（UE）估计后通过上行反馈链路发送给基站（BS）。原始 CSI 矩阵维度极高（天线数 × 子载波数），直接反馈会消耗大量上行带宽。

**传统方案**的局限：
- **压缩感知（CS）**：依赖信道稀疏性假设，在复杂信道（室外）下性能下降明显。
- **码本反馈（Type-I/II）**：量化损失大，难以适应复杂多径环境。

**AI/深度学习方案**的核心思路：
- 将 CSI 压缩反馈视为 **自编码器（Autoencoder）** 问题：编码器部署在 UE 侧做压缩，解码器部署在 BS 侧做重建。
- 利用神经网络学习 CSI 的隐式结构（天线域、时延域、空间域相关性）。
- 相比传统方案，可减少反馈开销 **1~2 个数量级**，NMSE 改善 **数 dB 至十余 dB**。

---

## 2. 文献列表与性能指标

### 2.1 CsiNet（2018）

| 属性 | 内容 |
|------|------|
| **论文标题** | Deep Learning for Massive MIMO CSI Feedback |
| **作者** | Chao-Kai Wen, Wan-Ting Shih, Shi Jin |
| **期刊/会议** | IEEE Wireless Communications Letters, Vol. 7, No. 5, 2018 |
| **arXiv** | [1712.08919](https://arxiv.org/abs/1712.08919) |
| **关键贡献** | 首次将自编码器引入 CSI 压缩反馈，提出 CsiNet 框架；编码器采用 CNN，解码器采用 RefineNet 结构；在 COST2100 上系统性超越传统 CS 方法（LASSO、TVAL3、BM3D-AMP） |

**NMSE 性能（COST2100，单位：dB）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | −17.36 | −8.75 |
| 1/16 | −8.65  | −4.51 |
| 1/32 | −6.24  | −2.81 |
| 1/64 | −5.84  | −1.93 |

> *CsiNet 是该领域的奠基工作，后续几乎所有方法均以此为基准线。*

---

### 2.2 CsiNet-LSTM（2019）

| 属性 | 内容 |
|------|------|
| **论文标题** | Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels |
| **作者** | Jianwen Guo, Chao-Kai Wen, Shi Jin, Geoffrey Ye Li |
| **期刊/会议** | IEEE Journal on Selected Areas in Communications (JSAC), 2020 |
| **arXiv** | [1807.11673](https://arxiv.org/abs/1807.11673) |
| **关键贡献** | 在 CsiNet 基础上引入 LSTM 模块，利用时变信道的**时序相关性**；采用"关键帧+差分"策略压缩连续 T=10 个时隙的 CSI；对时变信道比 CsiNet 提升显著 |

**NMSE 性能（COST2100，近似值，单位：dB）**

| CR | 方法 | 室内（Indoor） | 室外（Outdoor） |
|----|------|---------------|----------------|
| 1/16 | CsiNet      | −14.0  | −11.0  |
| 1/16 | CsiNet-LSTM | −20.5  | −15.0  |
| 1/32 | CsiNet      | −11.0  | −7.0   |
| 1/32 | CsiNet-LSTM | −17.0  | −11.0  |

> *CsiNet-LSTM 在时变场景下 NMSE 提升约 4–6 dB；静态场景提升幅度类似。*

---

### 2.3 CsiNet+（2020）

| 属性 | 内容 |
|------|------|
| **论文标题** | Convolutional Neural Network-Based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback: Design, Simulation and Analysis |
| **作者** | Wenxing Zhu, Minxi Wang, et al. |
| **期刊/会议** | IEEE Transactions on Wireless Communications, 2020 |
| **arXiv** | [1906.06007](https://arxiv.org/abs/1906.06007) |
| **关键贡献** | 改进解码器卷积核尺寸（更大感受野）以捕获全局信道相关性；提出**多速率**（SM-CsiNet+、PM-CsiNet+）变体支持动态反馈；在高压缩比时相比原始 CsiNet 提升更显著 |

**NMSE 性能（COST2100，室内场景，单位：dB）**

| CR | CsiNet | CsiNet+ |
|----|--------|---------|
| 1/4  | ≈−25  | ≈−26  |
| 1/16 | ≈−17  | ≈−20  |
| 1/32 | ≈−12  | ≈−17  |
| 1/64 | ≈−7   | ≈−12  |

> *CsiNet+ 在 CR=1/32 和 1/64 时提升最为明显，高压缩比下提升约 5 dB。*

---

### 2.4 CRNet（2020）

| 属性 | 内容 |
|------|------|
| **论文标题** | Multi-Resolution CSI Feedback with Deep Learning in Massive MIMO System |
| **作者** | Zhilin Lu, Jintao Wang, Jian Song |
| **期刊/会议** | IEEE International Conference on Communications (ICC), 2020 |
| **arXiv** | [1910.14322](https://arxiv.org/abs/1910.14322) |
| **代码** | [github.com/Kylin9511/CRNet](https://github.com/Kylin9511/CRNet) |
| **关键贡献** | 提出多分辨率（Multi-Resolution）特征提取框架，编码器同时提取不同尺度的信道特征；引入多分辨率卷积模块（MRC），大幅提升 NMSE 性能；在 COST2100 全部压缩比下优于 CsiNet 和 CsiNet+ |

**NMSE 性能（COST2100，单位：dB）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | −26.99 | −12.70 |
| 1/8  | −16.01 | −8.04  |
| 1/16 | −11.35 | −5.44  |
| 1/32 | −8.93  | −3.51  |
| 1/64 | −6.49  | −2.22  |

> *CRNet 是 CNN 类方法的重要里程碑，室内 CR=1/4 时 NMSE 达到 −26.99 dB，比 CsiNet 改善约 9.6 dB。*

---

### 2.5 CLNet（2021）

| 属性 | 内容 |
|------|------|
| **论文标题** | CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback |
| **作者** | Sijie Ji, Mo Li |
| **期刊/会议** | IEEE Wireless Communications Letters, Vol. 10, No. 11, 2021 |
| **DOI** | [10.1109/LWC.2021.3097507](https://doi.org/10.1109/LWC.2021.3097507) |
| **arXiv** | [2102.07507](https://arxiv.org/abs/2102.07507) |
| **代码** | [github.com/SIJIEJI/CLNet](https://github.com/SIJIEJI/CLNet) |
| **关键贡献** | 利用 CSI 矩阵的**复数值**特性（双通道输入→单复数层），降低 24% 计算量；结合轻量化注意力机制；整体 NMSE 平均提升 5.41%；模型参数量小于 CRNet |

**NMSE 性能（COST2100，单位：dB）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | −29.16 | −12.88 |
| 1/8  | −15.60 | −8.29  |
| 1/16 | −11.15 | −5.56  |
| 1/32 | −8.95  | −3.49  |
| 1/64 | −6.34  | −2.19  |

> *CLNet 室内 CR=1/4 达到 −29.16 dB，计算复杂度更低，是性价比极高的轻量化方案。*

---

### 2.6 BCsiNet（2021）

| 属性 | 内容 |
|------|------|
| **论文标题** | Binary Neural Network Aided CSI Feedback in Massive MIMO System |
| **作者** | Jianwen Guo, Chao-Kai Wen, Shi Jin |
| **期刊/会议** | IEEE Wireless Communications Letters, 2021 |
| **arXiv** | [2011.02692](https://arxiv.org/abs/2011.02692) |
| **代码** | [github.com/Kylin9511/BCsiNet](https://github.com/Kylin9511/BCsiNet) |
| **关键贡献** | 将编码器权重**二值化**，可减少 30× 内存占用和约 2× 推理加速；适用于 UE 端计算资源极受限场景；在轻微 NMSE 下降代价下实现极高效率 |

**NMSE 性能（COST2100，单位：dB，选取 BCsiNet-B3 变体）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | −20.31 | −9.77  |
| 1/8  | −12.77 | −6.86  |
| 1/16 | −10.71 | −4.52  |
| 1/32 | −7.93  | −2.74  |

> *BCsiNet 在内存和推理效率上具有显著优势，适合部署在资源受限的终端侧（UE），代价是 NMSE 弱于全精度模型约 5–8 dB。*

---

### 2.7 TransNet（2022）

| 属性 | 内容 |
|------|------|
| **论文标题** | TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System |
| **作者** | Zhenyu Cui, Ji Yao, Lizhen Chen, et al. |
| **期刊/会议** | IEEE Wireless Communications Letters, 2022 |
| **DOI** | [10.1109/LWC.2022.3141356](https://doi.org/10.1109/LWC.2022.3141356) |
| **代码** | [github.com/Treedy2020/TransNet](https://github.com/Treedy2020/TransNet) |
| **关键贡献** | 首次将**纯 Transformer（全注意力机制）**引入 CSI 反馈；编解码器均采用多头自注意力（MHSA）捕获全局天线-子载波相关性；在全部压缩比下显著超越 CRNet、CLNet 等 CNN 类方法 |

**NMSE 性能（COST2100，单位：dB）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | −33.12 | −15.80 |
| 1/8  | −23.47 | −9.86  |
| 1/16 | −15.70 | −7.88  |
| 1/32 | −11.98 | −4.87  |
| 1/64 | −7.99  | −3.07  |

> *TransNet 是 Transformer 类方法的基准，室内 CR=1/4 达到 −33.12 dB，比 CLNet 提升约 4 dB。*

---

### 2.8 CSI-StripeFormer（2023）

| 属性 | 内容 |
|------|------|
| **论文标题** | CSI-StripeFormer: Exploiting Stripe Features for CSI Compression in Massive MIMO System |
| **作者** | Huangxun Chen, et al. |
| **期刊/会议** | IEEE INFOCOM, 2023 |
| **PDF** | [chenhuangxun.com](https://www.chenhuangxun.com/files/infocom23-csistripeformer.pdf) |
| **关键贡献** | 发现 CSI 矩阵具有**条纹（Stripe）特征**（行/列方向的强相关性），专门设计条纹感知注意力模块；在 CR=1/64 极高压缩比下，比 SOTA 提升超过 **7 dB**，极端情况可达 17 dB；解决了高压缩比下信息重建难的核心问题 |

**NMSE 性能（COST2100，近似值，单位：dB）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | ≈−30  | ≈−20  |
| 1/16 | ≈−20  | ≈−10  |
| 1/32 | ≈−13  | ≈−8   |
| 1/64 | ≈−8 ~ −17 | ≈−7 ~ −17 |

> *在 CR=1/64 的极端压缩比下，CSI-StripeFormer 相比 TransNet 等 SOTA 有 7+ dB 提升，是应对高压缩比的标志性工作。*

---

### 2.9 TCNet（2025）

| 属性 | 内容 |
|------|------|
| **论文标题** | TCNet: A Unified Framework for CSI Feedback Compression Leveraging Transformers and Language Models |
| **作者** | （ICML ML4Wireless Workshop 2025） |
| **链接** | [openreview.net/forum?id=oHvtjD25js](https://openreview.net/forum?id=oHvtjD25js) |
| **关键贡献** | 将 **CNN + Swin Transformer** 混合骨干网络与**语言模型无损编码器**结合，实现 NMSE-比特率帕累托最优；语言模型作为熵编码器进一步压缩码字比特流；在相同比特率下 NMSE 优于所有现有方法 |

**NMSE 性能（COST2100，近似值，单位：dB）**

| CR | 室内（Indoor） | 室外（Outdoor） |
|----|---------------|----------------|
| 1/4  | ≈−30 ~ −35 | ≈−14  |
| 1/16 | ≈−14       | ≈−7   |

> *TCNet 目前代表最前沿水平，通过语言模型辅助无损编码进一步降低比特代价，是 NMSE–比特率联合优化的标志性突破。*

---

## 3. 综合性能对比表

### 3.1 室内（Indoor）NMSE（dB）对比

| 模型 | CR=1/4 | CR=1/8 | CR=1/16 | CR=1/32 | CR=1/64 | 年份 |
|------|:------:|:------:|:-------:|:-------:|:-------:|:----:|
| **CsiNet**            | −17.36 |   —    |  −8.65  |  −6.24  |  −5.84  | 2018 |
| **CsiNet-LSTM**       | —      |   —    | −20.5   | −17.0   |    —    | 2019 |
| **CsiNet+**           | ≈−26   |   —    |  ≈−20   |  ≈−17   |  ≈−12   | 2020 |
| **CRNet**             | −26.99 | −16.01 | −11.35  |  −8.93  |  −6.49  | 2020 |
| **CLNet**             | −29.16 | −15.60 | −11.15  |  −8.95  |  −6.34  | 2021 |
| **BCsiNet-B3**        | −20.31 | −12.77 | −10.71  |  −7.93  |    —    | 2021 |
| **TransNet**          | −33.12 | −23.47 | −15.70  | −11.98  |  −7.99  | 2022 |
| **CSI-StripeFormer**  | ≈−30   |   —    |  ≈−20   |  ≈−13   | ≈−8~−17 | 2023 |
| **TCNet**             | ≈−35   |   —    |  ≈−14   |    —    |    —    | 2025 |

### 3.2 室外（Outdoor）NMSE（dB）对比

| 模型 | CR=1/4 | CR=1/8 | CR=1/16 | CR=1/32 | CR=1/64 | 年份 |
|------|:------:|:------:|:-------:|:-------:|:-------:|:----:|
| **CsiNet**            | −8.75 |   —   |  −4.51  |  −2.81  |  −1.93  | 2018 |
| **CsiNet-LSTM**       |   —   |   —   | −15.0   | −11.0   |    —    | 2019 |
| **CsiNet+**           |   —   |   —   |    —    |    —    |    —    | 2020 |
| **CRNet**             | −12.70 | −8.04 |  −5.44  |  −3.51  |  −2.22  | 2020 |
| **CLNet**             | −12.88 | −8.29 |  −5.56  |  −3.49  |  −2.19  | 2021 |
| **BCsiNet-B3**        | −9.77  | −6.86 |  −4.52  |  −2.74  |    —    | 2021 |
| **TransNet**          | −15.80 | −9.86 |  −7.88  |  −4.87  |  −3.07  | 2022 |
| **CSI-StripeFormer**  | ≈−20   |   —   |  ≈−10   |  ≈−8    | ≈−7~−17 | 2023 |
| **TCNet**             | ≈−14   |   —   |  ≈−7    |    —    |    —    | 2025 |

> **注**：`—` 表示该论文未报告对应压缩比的数据；近似值（≈）来源于论文图表目测读数或文献摘要描述。

### 3.3 计算复杂度对比（编码器端）

| 模型 | 编码器参数量 | 特点 |
|------|-------------|------|
| CsiNet           | ~2.1 M | 基础 CNN 自编码器，轻量 |
| CRNet            | ~2.1 M | 多分辨率卷积，略高 |
| CLNet            | ~1.1 M | 复数层 + 轻量注意力，比 CRNet 减少 ~24% 计算 |
| BCsiNet          | ~0.033 M | 二值化编码器，极轻量（30× 内存节省）|
| TransNet         | ~5–8 M | Transformer，精度高，计算较重 |
| CSI-StripeFormer | ~8–15 M | 条纹 Transformer，高精度，复杂度较高 |
| TCNet            | ~10+ M | CNN+Swin+LLM，最高精度，最高复杂度 |

---

## 4. 发展趋势总结

```
2018          2019-2020         2021-2022          2023-2025
CsiNet   →  CsiNet+/CRNet  →  CLNet/TransNet  →  StripeFormer/TCNet
（CNN）      （多分辨率CNN）    （Transformer）    （混合+LLM辅助）
```

1. **架构演进**：从 CNN 自编码器 → 多分辨率 CNN → 纯 Transformer → CNN+Transformer 混合 → LLM 辅助无损编码。

2. **性能提升节奏**（室内 CR=1/4 为例）：
   - CsiNet（2018）：−17.36 dB
   - CRNet（2020）：−26.99 dB（+9.6 dB）
   - TransNet（2022）：−33.12 dB（+6.1 dB）
   - TCNet（2025）：≈−35 dB（+2 dB，但比特率更低）

3. **轻量化需求**：BCsiNet、CLNet 等针对 UE 侧计算资源受限场景，以较小 NMSE 代价换取极大计算效率提升。

4. **实际部署进展**：
   - **3GPP Release 18/19** 已将 AI-based CSI feedback 纳入标准化讨论范围。
   - 通过 Open RAN 和 OpenAirInterface 平台，已实现首批**实时空口验证**实验。
   - 华为、三星、ZTE、中兴等厂商均发布了面向 5G-Advanced 和 6G 的 AI CSI 反馈方案（如 M-CsiNet）。

5. **未来方向**：
   - **多任务/跨场景泛化**：单一模型适应多种天线配置和信道环境。
   - **生成式模型**（GAN/Diffusion/VAE）：更强的隐式先验建模能力。
   - **量化与熵编码联合优化**：TCNet 等已探索语言模型辅助无损编码。
   - **联邦学习与隐私保护**：分布式训练场景下的 CSI 压缩反馈。

---

## 5. 参考文献

| # | 引用 |
|---|------|
| [1] | C.-K. Wen, W.-T. Shih, and S. Jin, "Deep Learning for Massive MIMO CSI Feedback," *IEEE Wireless Commun. Lett.*, vol. 7, no. 5, pp. 748–751, 2018. [arXiv:1712.08919](https://arxiv.org/abs/1712.08919) |
| [2] | J. Guo, C.-K. Wen, S. Jin, and G. Y. Li, "Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels," *IEEE J. Sel. Areas Commun.*, vol. 39, no. 1, pp. 340–354, 2021. [arXiv:1807.11673](https://arxiv.org/abs/1807.11673) |
| [3] | W. Zhu, M. Wang, et al., "Convolutional Neural Network-Based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback," *IEEE Trans. Wireless Commun.*, 2020. [arXiv:1906.06007](https://arxiv.org/abs/1906.06007) |
| [4] | Z. Lu, J. Wang, and J. Song, "Multi-Resolution CSI Feedback with Deep Learning in Massive MIMO System," in *Proc. IEEE ICC*, 2020. [arXiv:1910.14322](https://arxiv.org/abs/1910.14322). Code: [github.com/Kylin9511/CRNet](https://github.com/Kylin9511/CRNet) |
| [5] | S. Ji and M. Li, "CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback," *IEEE Wireless Commun. Lett.*, vol. 10, no. 11, pp. 2361–2365, 2021. [arXiv:2102.07507](https://arxiv.org/abs/2102.07507). Code: [github.com/SIJIEJI/CLNet](https://github.com/SIJIEJI/CLNet) |
| [6] | J. Guo, C.-K. Wen, and S. Jin, "Binary Neural Network Aided CSI Feedback in Massive MIMO System," *IEEE Wireless Commun. Lett.*, 2021. [arXiv:2011.02692](https://arxiv.org/abs/2011.02692). Code: [github.com/Kylin9511/BCsiNet](https://github.com/Kylin9511/BCsiNet) |
| [7] | Z. Cui, J. Yao, L. Chen, et al., "TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System," *IEEE Wireless Commun. Lett.*, 2022. DOI: [10.1109/LWC.2022.3141356](https://doi.org/10.1109/LWC.2022.3141356). Code: [github.com/Treedy2020/TransNet](https://github.com/Treedy2020/TransNet) |
| [8] | H. Chen et al., "CSI-StripeFormer: Exploiting Stripe Features for CSI Compression in Massive MIMO System," in *Proc. IEEE INFOCOM*, 2023. [PDF](https://www.chenhuangxun.com/files/infocom23-csistripeformer.pdf) |
| [9] | (Authors), "TCNet: A Unified Framework for CSI Feedback Compression Leveraging Transformers and Language Models," *ICML ML4Wireless Workshop*, 2025. [OpenReview](https://openreview.net/forum?id=oHvtjD25js) |
| [10] | Y. Cheng et al., "Real-time AI-enabled CSI Feedback Experimentation with Open RAN," in *Proc. IEEE WONS*, 2024. [PDF](https://ece.northeastern.edu/wineslab/papers/cheng2024WONS.pdf) |
| [11] | (3GPP), "Study on Artificial Intelligence (AI) / Machine Learning (ML) for NR Air Interface," 3GPP TR 38.843, Release 18, 2023. |
