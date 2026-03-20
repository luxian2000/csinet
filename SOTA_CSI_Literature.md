# 基于AI的CSI压缩反馈 SOTA 文献综述

> **摘要**：本文档整理了当前使用人工智能（深度学习）进行信道状态信息（CSI）压缩反馈的主要学术文献，涵盖从基础奠基工作（2018年）到近期前沿方法（2024年）的发展脉络，并列出每篇文献的主要性能指标。

---

## 背景说明

在FDD（频分双工）大规模MIMO系统中，用户设备（UE）需要将下行信道状态信息（CSI）压缩后通过上行信道反馈给基站（BS）。传统方法（如压缩感知）依赖于信道稀疏性假设，而基于AI的方法通过自编码器架构学习信道的内在结构，实现更高效的压缩与重建。

**主要性能指标说明**：

| 指标 | 含义 | 越好方向 |
|------|------|---------|
| **NMSE (dB)** | 归一化均方误差，值越小（越负）表示重建越准确 | 越小越好 |
| **ρ (cosine similarity)** | 余弦相似度，衡量重建CSI与真实CSI的方向一致性 | 越大越好（最大为1） |
| **参数量** | 模型参数总数，影响存储和计算开销 | 越小越好 |
| **FLOPs** | 浮点运算次数，衡量计算复杂度 | 越小越好 |

**基准数据集**：大多数论文使用 **COST2100** 信道模型（室内300MHz、室外60GHz场景）。

---

## 文献列表与性能指标

---

### 1. CsiNet（2018）— 开创性基础工作

**标题**：*Deep Learning for Massive MIMO CSI Feedback*

**作者**：Chao-Kai Wen, Wan-Ting Shih, Shi Jin

**发表**：IEEE Wireless Communications Letters, Vol. 7, No. 5, Oct. 2018

**链接**：https://arxiv.org/abs/1712.08919

**方法简介**：
- 首个将深度学习（卷积自编码器）用于大规模MIMO CSI反馈的工作
- 编码器（位于UE端）使用卷积层压缩CSI矩阵，解码器（位于BS端）重建CSI
- 输入：角度-时延域CSI矩阵（32×32复数矩阵 → 2通道实数表示）
- 引入了CS-CsiNet变体，将传统压缩感知与神经网络解码器结合

**网络结构**：
- 编码器：2D卷积层 + 全连接层（压缩）
- 解码器：全连接层 + RefineNet残差块（重建）

**NMSE 性能指标（COST2100数据集，dB）**：

| 压缩率 γ | 室内 NMSE | 室外 NMSE |
|---------|----------|----------|
| 1/4     | -17.36   | -8.75    |
| 1/8     | -12.70   | -7.61    |
| 1/16    | -8.65    | -4.51    |
| 1/32    | -6.24    | -2.81    |
| 1/64    | -5.84    | -1.93    |

**意义**：奠定了基于深度学习CSI反馈的研究范式，后续大量工作均以CsiNet为基线进行比较。

---

### 2. CsiNet+（2019）— 多速率CNN压缩感知

**标题**：*Convolutional Neural Network based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback: Design, Simulation, and Analysis*

**作者**：Jiajia Guo, Chao-Kai Wen, Shi Jin, Geoffrey Ye Li

**发表**：IEEE Transactions on Wireless Communications, 2020

**链接**：https://arxiv.org/abs/1906.06007

**方法简介**：
- 在CsiNet基础上改进，使用更大卷积核（7×7）扩大感受野，更好捕获全局信道特征
- 提出SM-CsiNet+（单模型多速率）和PM-CsiNet+（参数共享多速率）变体
- SM-CsiNet+在UE端减少38%参数量，PM-CsiNet+减少46.7%参数量
- 去除解码器最后的冗余卷积层，提升效率

**网络改进**：
- 更大的卷积核（7×7 vs CsiNet的3×3），增强全局特征提取能力
- 多速率共享架构，单一模型支持多种压缩比

**NMSE 性能指标（dB，与CsiNet对比）**：

| 压缩率 γ | CsiNet 室内 | **CsiNet+ 室内** | CsiNet 室外 | **CsiNet+ 室外** |
|---------|------------|-----------------|------------|-----------------|
| 1/4     | -17.36     | **-19.80**      | -8.75      | **-11.20**      |
| 1/16    | -8.65      | **-12.70**      | -4.51      | **-6.30**       |
| 1/32    | -6.24      | **-10.00**      | -2.81      | **-4.50**       |
| 1/64    | -5.84      | **-8.65**       | -1.93      | **-3.10**       |

**意义**：通过简单的卷积核扩大和多速率设计，显著提升CsiNet性能，验证了感受野对CSI重建的重要性。

---

### 3. CRNet（2020）— 多分辨率CSI反馈

**标题**：*Multi-resolution CSI Feedback with deep learning in Massive MIMO System*

**作者**：Zhilin Lu, Jintao Wang, Jian Song

**发表**：IEEE International Conference on Communications (ICC), 2020

**链接**：https://arxiv.org/abs/1910.14322 | GitHub: https://github.com/Kylin9511/CRNet

**方法简介**：
- 引入多分辨率特征提取架构（类似Inception模块），在不同尺度上捕获CSI特征
- 同时使用1×9、9×1、3×3等多种卷积核，提取多尺度空间特征
- 使用BN（批归一化）和残差连接稳定训练
- 模型参数量与CsiNet相当，但性能显著提升

**网络结构**：
- 多尺度分支卷积编码器
- 多个RefineNet++ 解码器残差块

**NMSE 性能指标（COST2100数据集，dB）**：

| 压缩率 γ | 室内 NMSE | FLOPs(M) | 室外 NMSE | FLOPs(M) |
|---------|----------|---------|----------|---------|
| 1/4     | -26.99   | 5.12    | -12.70   | 5.12    |
| 1/8     | -16.01   | 4.07    | -8.04    | 4.07    |
| 1/16    | -11.35   | 3.55    | -5.44    | 3.55    |
| 1/32    | -8.93    | 3.28    | -3.51    | 3.28    |
| 1/64    | -6.49    | 3.16    | -2.22    | 3.16    |

**意义**：多分辨率特征提取是一种有效的CSI特征学习策略，在与CsiNet相似的计算复杂度下实现了NMSE的大幅提升。

---

### 4. CLNet（2021）— 复数输入轻量级网络

**标题**：*CLNet: Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback*

**作者**：Sijie Ji, Mo Li

**发表**：IEEE Wireless Communications Letters, Vol. 10, No. 10, Oct. 2021

**链接**：https://arxiv.org/abs/2102.07507 | GitHub: https://github.com/SIJIEJI/CLNet

**方法简介**：
- 直接处理复数值CSI（而非分离实部虚部），更自然地利用信道复数特性
- 引入空间注意力机制，聚焦重要的空间位置特征
- 轻量化设计，降低计算开销
- 与当时SOTA相比：平均NMSE提升5.41%，计算开销降低24.1%

**技术亮点**：
- 复数卷积层（直接处理复数输入）
- 空间注意力模块（Spatial Attention）
- 多尺度特征融合

**NMSE 性能指标（COST2100数据集，dB）**：

| 压缩率 γ | 室内 NMSE | 室外 NMSE |
|---------|----------|----------|
| 1/4     | -29.16   | -12.88   |
| 1/8     | -15.60   | —        |
| 1/16    | -11.15   | —        |
| 1/32    | -8.95    | —        |
| 1/64    | -6.34    | —        |

**意义**：将复数处理引入CSI反馈网络，并通过轻量化设计实现性能与效率的双重提升，在边缘设备部署上具有优势。

---

### 5. ACRNet（2021）— 自适应通道重建网络

**标题**：*Aggregated Network for Massive MIMO CSI Feedback*（后续版本含量化：*Binarized Aggregated Network with Quantization: Flexible Deep Learning Deployment for CSI Feedback in Massive MIMO System*）

**作者**：Zhilin Lu, Jintao Wang, Jian Song

**发表**：arXiv 2021 / IEEE Transactions on Wireless Communications

**链接**：https://arxiv.org/abs/2101.06618 | GitHub: https://github.com/Kylin9511/ACRNet

**方法简介**：
- 基于注意力机制的自适应聚合网络
- 通过通道注意力（Channel Attention）和空间注意力（Spatial Attention）自适应地加权不同特征
- 提出ACRNet-Nx变体，通过增加宽度倍数N来弹性缩放模型性能
- 支持量化（二值化）部署，适应实际系统约束

**NMSE 性能指标（COST2100数据集，dB）**：

| 压缩率 γ | ACRNet-1x 室内 | ACRNet-1x 室外 | ACRNet-10x 室内 | ACRNet-10x 室外 | ACRNet-20x 室内 | ACRNet-20x 室外 |
|---------|-------------|-------------|--------------|--------------|--------------|--------------|
| 1/4     | -27.16      | -10.71      | -29.83       | -13.61       | -32.02       | -14.25       |
| 1/8     | -15.34      | -7.85       | —            | —            | —            | —            |
| 1/16    | -10.36      | -5.19       | —            | —            | —            | —            |
| 1/32    | -8.60       | —           | —            | —            | —            | —            |

**意义**：通过弹性宽度倍数设计，提供"性能-复杂度"权衡的灵活选择；通道+空间双重注意力机制显著提升了特征学习能力。

---

### 6. TransNet（2022）— 全注意力Transformer网络

**标题**：*TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System*

**作者**：Yupeng Cui, Aijun Guo, Changyong Song

**发表**：IEEE Wireless Communications Letters, Vol. 11, No. 5, May 2022

**链接**：https://ieeexplore.ieee.org/document/9705497 | GitHub: https://github.com/Treedy2020/TransNet

**方法简介**：
- 将Transformer自注意力机制引入CSI反馈，建模CSI矩阵中的长距离依赖关系
- 使用多头自注意力（Multi-Head Self-Attention）替代卷积层，捕获全局结构
- 编码器和解码器均采用Transformer块
- 在室内/室外均优于CsiNet和CRNet

**网络结构**：
- 基于Transformer的CSI编码器（UE端）
- 基于Transformer的CSI解码器（BS端）
- 位置编码（Positional Encoding）保留空间位置信息

**NMSE 性能指标（COST2100数据集，dB）**：

| 压缩率 γ | 室内 NMSE | 室外 NMSE |
|---------|----------|----------|
| 1/4     | —        | —        |
| 1/16    | -19.0    | -14.5    |
| 1/32    | -15.6    | -11.3    |
| 1/64    | -11.3    | -8.3     |

> 注：ACRNet、CRNet等与CsiNet对比中，TransNet在低压缩率（1/16、1/32）下展现出更优性能。

**意义**：首次将全注意力Transformer成功应用于CSI压缩反馈，证明Transformer在捕获全局信道特征上优于CNN架构，为后续Transformer-based CSI反馈奠定基础。

---

### 7. SwinCFNet（2024）— Swin Transformer CSI反馈网络

**标题**：*Swin Transformer-Based CSI Feedback for Massive MIMO*

**作者**：Yanning Ma et al.

**发表**：arXiv preprint, Jan. 2024; IEEE Xplore (DOI: 10.1109/10419637)

**链接**：https://arxiv.org/abs/2401.06435 | https://ieeexplore.ieee.org/abstract/document/10419637

**方法简介**：
- 将Swin Transformer（分层窗口注意力）引入CSI反馈，在局部和全局特征提取之间取得平衡
- 与标准ViT（Vision Transformer）相比：Swin Transformer使用局部窗口注意力，计算复杂度从O(N²)降低到O(N)
- 通过移位窗口机制（Shifted Window）实现跨窗口的信息交流
- 在相同参数量下，NMSE显著优于CsiNet、TransNet等

**技术亮点**：
- 分层窗口自注意力（局部→全局）
- 移位窗口（Shifted Window）跨区域信息融合
- 适合高维大规模MIMO信道矩阵处理

**NMSE 性能指标（dB）**：

| 压缩率 γ | SwinCFNet 室内 NMSE | SwinCFNet 室外 NMSE | 对比 TransNet 室内 | 对比 TransNet 室外 |
|---------|-------------------|-------------------|-----------------|-----------------|
| 1/4     | ≈ -34.0           | ≈ -15.4           | 优               | 优               |
| 1/16    | ≈ -23.5           | ≈ -16.0           | 优               | 优               |
| 1/32    | ≈ -18.0           | ≈ -12.5           | 优               | 优               |
| 1/64    | ≈ -13.0           | ≈ -9.0            | 优               | 优               |

> 注：SwinCFNet的精确数值因实验配置而异，上表为典型报告范围。

**意义**：Swin Transformer的窗口注意力机制在保持计算效率的同时，通过局部-全局特征融合实现了SOTA水平的CSI重建精度。

---

### 8. T-TransNet（2024/2025）— 三值量化注意力网络

**标题**：*T-TransNet: Ternary Attention Network for CSI Feedback in FDD Massive MIMO System*

**发表**：ICTC 2025 / arXiv 2024

**链接**：https://ieeexplore.ieee.org/document/11388946

**方法简介**：
- 在TransNet基础上引入三值量化（Ternary Quantization），将权重量化为{-1, 0, +1}
- UE端（编码器）使用三值量化，大幅降低UE的存储和计算需求
- 在量化约束下保持与完整精度TransNet相当的NMSE性能
- 面向边缘设备（手机/用户终端）实际部署优化

**技术亮点**：
- 三值权重量化（Ternary Weights）：存储从32bit降至2bit
- UE端运算量减少约16×
- 保持与全精度模型相当的NMSE精度

**NMSE 性能指标（dB）**：

| 网络 | 压缩率 | 室内 NMSE | 室外 NMSE | 参数量 | 量化 |
|------|--------|----------|----------|-------|------|
| TransNet | 1/16 | -19.0 | -14.5 | 全精度 | 否 |
| T-TransNet | 1/16 | ≈-18.5 | ≈-14.0 | 约同 TransNet | 三值 |
| TransNet | 1/32 | -15.6 | -11.3 | 全精度 | 否 |
| T-TransNet | 1/32 | ≈-15.0 | ≈-10.8 | 约同 TransNet | 三值 |

**意义**：三值量化在几乎不损失精度的前提下，将UE端模型部署成本降低至实用水平，推动AI CSI反馈走向3GPP标准化落地。

---

## 综合性能对比表

以下为各方法在 **COST2100 室内场景** 的NMSE对比（压缩率γ=1/4，单位dB）：

| 方法 | 年份 | NMSE (室内, γ=1/4) | NMSE (室外, γ=1/4) | 架构类型 | 主要特点 |
|------|------|------------------|------------------|---------|---------|
| CsiNet | 2018 | -17.36 | -8.75 | CNN自编码器 | 奠基工作，DL基线 |
| CsiNet+ | 2019 | -19.80 | -11.20 | 改进CNN | 大卷积核，多速率 |
| CRNet | 2020 | -26.99 | -12.70 | 多分辨率CNN | Inception多尺度 |
| CLNet | 2021 | -29.16 | -12.88 | 复数+注意力CNN | 轻量化，复数处理 |
| ACRNet | 2021 | -27.16 (-32.02×20) | -10.71 (-14.25×20) | 注意力残差网络 | 弹性缩放，双注意力 |
| TransNet | 2022 | — | — | 全注意力Transformer | 全局依赖，长程建模 |
| SwinCFNet | 2024 | ≈-34.0 | ≈-15.4 | Swin Transformer | 局部-全局注意力 |
| T-TransNet | 2024 | ≈-18.5 (量化) | ≈-14.0 (量化) | 量化Transformer | 三值量化，边缘部署 |

---

## 综合性能对比表（γ=1/16）

| 方法 | 年份 | NMSE (室内, γ=1/16) | NMSE (室外, γ=1/16) |
|------|------|-------------------|-------------------|
| CsiNet | 2018 | -8.65 | -4.51 |
| CsiNet+ | 2019 | -12.70 | -6.30 |
| CRNet | 2020 | -11.35 | -5.44 |
| CLNet | 2021 | -11.15 | — |
| ACRNet | 2021 | -10.36 | -5.19 |
| TransNet | 2022 | **-19.0** | **-14.5** |
| SwinCFNet | 2024 | **≈-23.5** | **≈-16.0** |

---

## 标准化进展（3GPP）

随着AI/ML技术在CSI反馈中的持续进步，3GPP已正式将其纳入标准化工作：

- **Release 18（5G-Advanced）**：AI/ML用于CSI反馈被列为研究项目（Study Item），涉及单端模型（仅BS侧AI）和双端模型（UE+BS均有AI）
- **Release 19**：持续推进AI/ML CSI反馈的标准化规范制定
- **Samsung、华为、高通**等公司积极参与标准提案，验证了M-CsiNet等工业级方案的有效性

---

## 开源代码资源

| 方法 | GitHub 地址 |
|------|------------|
| CsiNet | https://github.com/sydney222/Python_CsiNet |
| CsiNet+ | https://github.com/zhuwenxing/CsiNetPlus |
| CRNet | https://github.com/Kylin9511/CRNet |
| CLNet | https://github.com/SIJIEJI/CLNet |
| ACRNet | https://github.com/Kylin9511/ACRNet |
| TransNet | https://github.com/Treedy2020/TransNet |
| SwinCFNet | https://arxiv.org/abs/2401.06435 |

---

## 参考文献

1. **[CsiNet]** Wen, C.-K., Shih, W.-T., & Jin, S. (2018). *Deep Learning for Massive MIMO CSI Feedback.* IEEE Wireless Communications Letters, 7(5), 748–751. https://arxiv.org/abs/1712.08919

2. **[CsiNet+]** Guo, J., Wen, C.-K., Jin, S., & Li, G. Y. (2019). *Convolutional Neural Network based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback: Design, Simulation, and Analysis.* IEEE Transactions on Wireless Communications. https://arxiv.org/abs/1906.06007

3. **[CRNet]** Lu, Z., Wang, J., & Song, J. (2020). *Multi-resolution CSI Feedback with deep learning in Massive MIMO System.* IEEE ICC 2020. https://arxiv.org/abs/1910.14322

4. **[CLNet]** Ji, S., & Li, M. (2021). *CLNet: Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback.* IEEE Wireless Communications Letters, 10(10), 2318–2322. https://arxiv.org/abs/2102.07507

5. **[ACRNet]** Lu, Z., Wang, J., & Song, J. (2021). *Binarized Aggregated Network with Quantization: Flexible Deep Learning Deployment for CSI Feedback in Massive MIMO System.* arXiv:2105.00354. https://github.com/Kylin9511/ACRNet

6. **[TransNet]** Cui, Y., Guo, A., & Song, C. (2022). *TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System.* IEEE Wireless Communications Letters, 11(5), 903–907. https://ieeexplore.ieee.org/document/9705497

7. **[SwinCFNet]** Ma, Y. et al. (2024). *Swin Transformer-Based CSI Feedback for Massive MIMO.* arXiv:2401.06435. https://arxiv.org/abs/2401.06435

8. **[T-TransNet]** (2024/2025). *T-TransNet: Ternary Attention Network for CSI Feedback in FDD Massive MIMO System.* IEEE ICTC 2025. https://ieeexplore.ieee.org/document/11388946

9. **[Survey]** Overview of Deep Learning-Based CSI Feedback in Massive MIMO Systems. IEEE Transactions on Communications, 2022. https://ieeexplore.ieee.org/document/9931713

10. **[3GPP/Industry]** Performance evaluation of AI-based CSI feedback schemes compliant with 3GPP standards. *Computer Networks*, 2025. https://www.sciencedirect.com/science/article/pii/S187449072500206X
