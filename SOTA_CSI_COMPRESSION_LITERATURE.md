# 基于AI的CSI压缩反馈SOTA文献综述

> **说明**：本文档汇总当前利用人工智能（深度学习）进行 CSI（信道状态信息）压缩反馈的主流与最新方法，列出相关文献并给出各方法在标准数据集（COST2100）上的关键性能指标。所有 NMSE 值以 dB 为单位，数值越小（越负）表示重建质量越好。

---

## 背景

在 FDD（频分双工）大规模 MIMO 系统中，用户设备（UE）需要将下行 CSI 经过有损压缩后通过上行信道反馈给基站（BS）。传统压缩感知（CS）方法（LASSO、BM3D-AMP、TVAL3 等）在高压缩比下性能受限，深度学习方法通过端到端自编码器结构，能够学习 CSI 的内在统计特性，显著提升压缩效率和重建精度。

---

## 一、奠基性工作

### 1. CsiNet（2018）

| 项目 | 内容 |
|------|------|
| **标题** | Deep Learning for Massive MIMO CSI Feedback |
| **作者** | Chao-Kai Wen, Wan-Ting Shih, Shi Jin |
| **发表** | IEEE Wireless Communications Letters, 2018 |
| **链接** | [IEEE Xplore](https://ieeexplore.ieee.org/document/8322184) \| [arXiv:1712.08919](https://arxiv.org/abs/1712.08919) |
| **核心思路** | 首次将卷积自编码器应用于 CSI 压缩反馈，编码器将 CSI 矩阵压缩为低维码字，解码器在基站端重建 CSI |

**NMSE 性能（COST2100 数据集）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | -17.36        | -8.75         |
| 1/16   | -8.65         | -4.51         |
| 1/32   | -6.24         | -2.81         |
| 1/64   | -5.84         | -1.93         |

---

## 二、CNN增强方法

### 2. CRNet（2020）

| 项目 | 内容 |
|------|------|
| **标题** | Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System |
| **作者** | Zhilin Lu, Xiaoming He, Caili Guo 等 |
| **发表** | IEEE ICC 2020 |
| **链接** | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9149229) \| [arXiv:1910.14322](https://arxiv.org/abs/1910.14322) \| [GitHub](https://github.com/Kylin9511/CRNet) |
| **核心思路** | 多分辨率特征提取（Multi-Resolution），在同一层并行使用不同卷积核尺寸，同时捕获 CSI 的局部与全局特征，配合余弦退火训练策略 |

**NMSE 性能（COST2100 数据集）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | -26.99        | -12.70        |
| 1/8    | -16.01        | -8.04         |
| 1/16   | -11.35        | -5.44         |
| 1/32   | -8.93         | -3.51         |
| 1/64   | -6.49         | -2.22         |

---

### 3. CLNet（2021）

| 项目 | 内容 |
|------|------|
| **标题** | CLNet: Complex Input Lightweight Neural Network for Massive MIMO CSI Feedback |
| **作者** | Sijie Ji, Mo Li 等 |
| **发表** | IEEE Wireless Communications Letters, 2021 |
| **链接** | [PDF](https://sijieji.github.io/pdf/CLNet_Complex_Input_Lightweight_Neural_Network_Designed_for_Massive_MIMO_CSI_Feedback.pdf) \| [GitHub](https://github.com/SIJIEJI/CLNet) |
| **核心思路** | 复数输入层直接处理 CSI 的实部与虚部；空间注意力机制聚焦关键特征；参数量与计算量约减少 24% |

**NMSE 性能（COST2100 数据集）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | -29.16        | -12.88        |
| 1/8    | -15.60        | —             |
| 1/16   | -11.15        | —             |
| 1/32   | -8.95         | —             |

---

### 4. ACRNet（2021）

| 项目 | 内容 |
|------|------|
| **标题** | Aggregated Network for Massive MIMO CSI Feedback |
| **作者** | Zhilin Lu 等 |
| **发表** | arXiv:2101.06618, 2021 |
| **链接** | [arXiv:2101.06618](https://arxiv.org/abs/2101.06618) \| [GitHub](https://github.com/Kylin9511/ACRNet) |
| **核心思路** | 聚合多条并行分支，引入分组卷积与深度可分离卷积，在更低计算量下实现与 CRNet 相当乃至更优的 NMSE |

**NMSE 性能（COST2100 数据集，室内）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | -27.16 ~ -32  | -10.71 ~ -14  |
| 1/8    | -15.34 ~ -20  | -7.85 ~ -9.7  |
| 1/16   | -10.36 ~ -15  | -5.19 ~ -6.5  |
| 1/32   | -8.60         | —             |

---

### 5. AnciNet（2020）

| 项目 | 内容 |
|------|------|
| **标题** | AnciNet: An Efficient Deep Learning Approach for Feedback Compression of Estimated CSI in Massive MIMO Systems |
| **作者** | Guo Jiajia 等 |
| **发表** | IEEE Wireless Communications Letters, 2020 |
| **链接** | [DeepAI](https://deepai.org/publication/ancinet-an-efficient-deep-learning-approach-for-feedback-compression-of-estimated-csi-in-massive-mimo-systems) |
| **核心思路** | 针对含噪 CSI 估计场景，提取去噪特征后再进行压缩，提升在不完美信道估计下的鲁棒性 |
| **性能** | 在含噪 CSI 估计条件下优于 CsiNet、CRNet 等基线方法，具体 NMSE 随噪声水平和场景而异 |

---

## 三、Transformer 方法

### 6. TransNet（2022）

| 项目 | 内容 |
|------|------|
| **标题** | TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System |
| **作者** | Xuewen Liao 等 |
| **发表** | IEEE Wireless Communications Letters, 2022 |
| **链接** | [IEEE Xplore](https://ieeexplore.ieee.org/document/9705497) \| [GitHub](https://github.com/Treedy2020/TransNet) |
| **核心思路** | 首次将全注意力（Full Attention）Transformer 引入 CSI 压缩，建模天线维与子载波维的全局相关性，编解码器均采用多头自注意力结构 |

**NMSE 性能（COST2100 数据集）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | ≈ -31         | ≈ -15         |
| 1/16   | ≈ -15         | —             |
| 1/32   | ≈ -10         | —             |

---

### 7. TransNet+（2023）

| 项目 | 内容 |
|------|------|
| **标题** | TransNet+: Enhanced Transformer-based CSI Feedback |
| **链接** | [GitHub](https://github.com/serenachen6/TransNet-plus) |
| **核心思路** | 在 TransNet 基础上改进多头注意力策略与训练方案，进一步提升重建精度 |

**NMSE 性能（COST2100 数据集）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | -33.12        | -15.80        |
| 1/8    | -23.47        | -9.86         |
| 1/16   | -15.70        | -7.88         |
| 1/32   | -11.98        | -4.87         |
| 1/64   | -7.99         | -3.07         |

---

### 8. CsiFormer（2022）

| 项目 | 内容 |
|------|------|
| **标题** | CsiFormer: Transformer-Based Massive MIMO CSI Feedback |
| **发表** | 2022 |
| **核心思路** | 基于 Transformer 自编码器，充分捕获 CSI 长距离依赖；在 NMSE 和 BER（误码率）、吞吐量等链路级指标上均优于 CsiNet |

**NMSE 性能（COST2100 数据集，约估）**

| 压缩比 | 室内 NMSE (dB) | 室外 NMSE (dB) |
|--------|---------------|---------------|
| 1/4    | ≈ -33 ~ -34   | ≈ -17         |

---

### 9. SwinCFNet（2024）

| 项目 | 内容 |
|------|------|
| **标题** | Swin Transformer-Based CSI Feedback for Massive MIMO |
| **发表** | IEEE Wireless Communications Letters, 2024（arXiv:2401.06435）|
| **链接** | [arXiv:2401.06435](https://arxiv.org/abs/2401.06435) \| [IEEE Xplore](https://ieeexplore.ieee.org/document/10419637) |
| **核心思路** | 使用 Swin Transformer（分层窗口自注意力）构建自编码器，以类似 CNN 的局部-全局特征提取能力实现 SOTA 性能，参数量较小（~4.87M） |
| **典型 NMSE** | 1/4 压缩比下室外 NMSE ≈ -15.4 dB（4.87M 参数，20.18M FLOPs），在同等模型规模下优于标准 Transformer 方法 |

---

### 10. ST-TNet（2024）

| 项目 | 内容 |
|------|------|
| **标题** | ST-TNet: An Spatio-Temporal Joint Transformer Network for CSI Feedback in FDD Massive MIMO |
| **发表** | Digital Communications and Networks, 2024 |
| **链接** | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S187449072400288X) |
| **核心思路** | 联合时空域 Transformer，同时建模 CSI 的空间相关性与时间变化，支持动态场景下的高精度反馈 |

---

## 四、轻量级 / 增强训练方法

### 11. JPTS-CsiNet / JPTS-CRNet（2023）

| 项目 | 内容 |
|------|------|
| **标题** | Enhancing Deep Learning Performance of Massive MIMO CSI Feedback（JPTS）|
| **链接** | [GitHub](https://github.com/Arway-web/CSI_JPTS) |
| **核心思路** | 联合预训练策略（Joint Pre-Training Strategy），通过改进训练流程显著提升 CsiNet 和 CRNet 的 NMSE，无需修改网络结构 |

**NMSE 性能（COST2100 数据集）**

| 模型          | 场景   | 压缩比 | NMSE (dB) |
|---------------|--------|--------|-----------|
| JPTS-CsiNet   | 室内   | 1/4    | -24.19    |
| JPTS-CsiNet   | 室内   | 1/16   | -10.65    |
| JPTS-CRNet    | 室内   | 1/4    | -26.84    |
| JPTS-CRNet    | 室内   | 1/16   | -11.55    |
| JPTS-CSINet   | 室外   | 1/4    | -12.20    |
| JPTS-CRNet    | 室外   | 1/4    | -12.72    |

---

## 五、新型表示方法

### 12. CSI-INR（2023）

| 项目 | 内容 |
|------|------|
| **标题** | Implicit Neural Representation for CSI Feedback（CSI-INR）|
| **链接** | [项目主页](https://eedavidwu.github.io/CSI-INR/) |
| **核心思路** | 将 CSI 矩阵建模为隐式神经表示（INR）函数，结合元学习（Meta-Learning）生成低比特调制码字反馈，支持灵活压缩比，无需固定码本设计 |
| **性能** | 配合量化与熵编码，在多个压缩比下达到或超越基于特征提取的 DL 方法，性能可与同期 SOTA 媲美 |

---

### 13. CSI-PPPNet（2024）

| 项目 | 内容 |
|------|------|
| **标题** | Deep Learning for CSI Feedback: One-Sided Model and Joint Multi-Module Learning Perspectives |
| **发表** | arXiv:2405.05522, 2024 |
| **链接** | [arXiv:2405.05522](https://arxiv.org/html/2405.05522v1) \| [Samsung Research Blog](https://research.samsung.com/blog/Deep-Learning-for-CSI-Feedback-One-Sided-Model-and-Joint-Multi-Module-Learning-Perspectives) |
| **核心思路** | 单侧（One-Sided）反馈模型，仅在基站侧部署 DL 模型，UE 侧无需 DL 推断；同时支持任意压缩比，易于实际部署与多厂商互操作 |

---

## 六、面向5G/6G标准的评估

### 14. M-CsiNet（5G NR 合规评估，2025）

| 项目 | 内容 |
|------|------|
| **标题** | Performance Evaluation of AI-based CSI Feedback Schemes Compliant with 5G NR |
| **发表** | Digital Communications and Networks, 2025 |
| **链接** | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S187449072500206X) \| [TechRxiv](https://www.techrxiv.org/users/842079/articles/1231932-performance-evaluation-of-ai-based-csi-feedback-techniques-for-5g-advance-and-6g-networks) |
| **核心思路** | 在 3GPP CDL-C 信道模型下评估 AI-based CSI 反馈方案与 5G NR Type-II/Enhanced Type-II 标准的对比 |
| **性能** | 与 Type-II/Enhanced Type-II 相比：反馈开销降低约 2 个数量级，解码复杂度降低约 1 个数量级；在低 10–15 dB SNR 下即可达到最优 BLER 和吞吐量 |

---

## 七、综合性能对比

以下为主要方法在 **COST2100 室内场景、压缩比 1/4** 下的 NMSE 横向对比：

| 方法          | 年份 | NMSE (dB) | 模型特点 |
|---------------|------|-----------|---------|
| CsiNet        | 2018 | -17.36    | 基础卷积自编码器 |
| CRNet         | 2020 | -26.99    | 多分辨率特征提取 |
| AnciNet       | 2020 | > CRNet   | 含噪估计下鲁棒 |
| CLNet         | 2021 | -29.16    | 复数输入 + 注意力 |
| ACRNet        | 2021 | -27 ~ -32 | 聚合多分支 + 轻量化 |
| TransNet      | 2022 | ≈ -31     | 全注意力 Transformer |
| CsiFormer     | 2022 | ≈ -33~34  | Transformer 自编码器 |
| TransNet+     | 2023 | -33.12    | 增强多头注意力 |
| SwinCFNet     | 2024 | SOTA      | Swin Transformer |

---

## 八、性能指标说明

| 指标 | 说明 |
|------|------|
| **NMSE** | 归一化均方误差（dB），主要质量指标，越低越好 |
| **rho (ρ)** | 余弦相似度 / Pearson 相关系数，越接近 1 越好 |
| **压缩比 γ** | 码字维度 / 原始 CSI 维度，如 1/4、1/16、1/32、1/64 |
| **参数量** | 模型参数数量（影响存储和部署成本）|
| **FLOPs** | 浮点运算量（衡量推断复杂度）|
| **BLER/吞吐量** | 链路级指标，用于 5G NR 标准合规评估 |

---

## 参考文献

1. Wen C K, Shih W T, Jin S. Deep learning for massive MIMO CSI feedback[J]. *IEEE Wireless Communications Letters*, 2018, 7(5): 748-751.
2. Lu Z, He X, Guo C, et al. Multi-resolution CSI feedback with deep learning in massive MIMO system[C]. *IEEE ICC*, 2020.
3. Ji S, Li M. CLNet: Complex input lightweight neural network for massive MIMO CSI feedback[J]. *IEEE Wireless Communications Letters*, 2021, 10(10): 2318-2322.
4. Lu Z, et al. Aggregated network for massive MIMO CSI feedback[J]. arXiv:2101.06618, 2021.
5. Guo J, et al. AnciNet: An efficient deep learning approach for feedback compression of estimated CSI in massive MIMO systems[J]. *IEEE Wireless Communications Letters*, 2020.
6. Liao X, et al. TransNet: Full attention network for CSI feedback in FDD massive MIMO system[J]. *IEEE Wireless Communications Letters*, 2022.
7. Wu D, et al. CSI-INR: Implicit neural representation for CSI feedback[EB/OL]. https://eedavidwu.github.io/CSI-INR/, 2023.
8. Gao Z, et al. (JPTS) Enhancing deep learning performance of massive MIMO CSI feedback[EB/OL]. GitHub, 2023.
9. Cao G, et al. Swin transformer-based CSI feedback for massive MIMO[J]. *IEEE Wireless Communications Letters*, 2024. arXiv:2401.06435.
10. (ST-TNet) ST-TNet: An spatio-temporal joint transformer network for CSI feedback in FDD massive MIMO[J]. *Digital Communications and Networks*, 2024.
11. (CSI-PPPNet / One-Sided) Deep learning for CSI feedback: One-sided model and joint multi-module learning perspectives[J]. arXiv:2405.05522, 2024.
12. (M-CsiNet) Performance evaluation of AI-based CSI feedback schemes compliant with 5G NR[J]. *Digital Communications and Networks*, 2025.
