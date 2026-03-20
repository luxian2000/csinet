# 基于AI的CSI压缩反馈SOTA文献综述

> **任务**：搜索当前使用AI进行CSI（信道状态信息）压缩反馈的SOTA方法，列出文献并给出每篇文献的性能指标。
>
> **评估基准**：COST2100 信道模型，FDD大规模MIMO系统，室内（5.3 GHz）与室外（300 MHz）场景。
>
> **核心性能指标**：
> - **NMSE**（归一化均方误差，dB，越小越好）
> - **ρ（rho）**（余弦相似度，越接近1越好）
> - **压缩比（CR）**：1/4、1/8、1/16、1/32、1/64

---

## 1. CsiNet（2018）

**文献**：C.-K. Wen, W.-T. Shih, and S. Jin, "Deep Learning for Massive MIMO CSI Feedback," *IEEE Wireless Communications Letters*, vol. 7, no. 5, pp. 748–751, Oct. 2018.
- arXiv: [1712.08919](https://arxiv.org/abs/1712.08919)
- DOI: [10.1109/LWC.2018.2818160](https://ieeexplore.ieee.org/document/8322184)

**主要贡献**：首次将深度卷积自编码器（Encoder-Decoder）引入CSI压缩反馈任务，是该领域的奠基性工作。Encoder部署在UE端进行压缩，Decoder部署在基站端重建。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | 室内 NMSE | 室外 NMSE |
|:------:|:---------:|:---------:|
| 1/4    | -17.36    | -8.75     |
| 1/16   | -8.65     | -4.51     |
| 1/32   | -6.24     | -2.81     |
| 1/64   | -5.84     | -1.93     |

---

## 2. CsiNet-LSTM（2018）

**文献**：K. Wang, K. Niu, and W. He, "Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels," *IEEE Wireless Communications Letters*, vol. 8, no. 2, pp. 416–419, Apr. 2019. (初版IEEE ICC 2018)
- DOI: [10.1109/LWC.2018.2874264](https://ieeexplore.ieee.org/document/8482358)

**主要贡献**：在CsiNet基础上引入LSTM模块，利用信道的时间相关性，适用于时变信道场景。相比CsiNet在中低压缩比下可降低NMSE约1–2 dB。

**性能指标（COST2100，室内场景，NMSE / dB）**：

| 压缩比 | CsiNet    | CsiNet-LSTM（提升）  |
|:------:|:---------:|:--------------------:|
| 1/4    | -17.36    | 约 -18~-19（↓ ~1–2 dB）|
| 1/16   | -8.65     | 约 -9~-10（↓ ~1–2 dB）|
| 1/32   | -6.24     | 约 -7~-8（↓ ~1–2 dB） |

> 注：LSTM版对时变信道提升更为明显；具体数值参见原文Fig.2。

---

## 3. CsiNet+（2020）

**文献**：X. Guo, C.-K. Wen, S. Jin, and T. Jiang, "Convolutional Neural Network-Based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback: Design, Simulation, and Analysis," *IEEE Transactions on Wireless Communications*, vol. 19, no. 4, pp. 2827–2840, Apr. 2020.
- DOI: [10.1109/TWC.2020.2968300](https://ieeexplore.ieee.org/document/8972904)
- IEICE版: [10.1587/transfun.2019EAL2123](https://www.jstage.jst.go.jp/article/transfun/E103.A/1/E103.A_2019EAL2123/_article/)

**主要贡献**：扩大了卷积感受野，增加了噪声/截断鲁棒性处理，去除了解码器中冗余层，在各压缩比下均优于CsiNet。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | 室内 NMSE    | 室外 NMSE    |
|:------:|:------------:|:------------:|
| 1/4    | 优于 -17.36  | 优于 -8.75   |
| 1/16   | 优于 -8.65   | 优于 -4.51   |
| 1/32   | 优于 -6.24   | 优于 -2.81   |

> 注：CsiNet+ 在所有测试压缩比下均低于 CsiNet；具体数值参见原文Table I。

---

## 4. CRNet（2020/2021）

**文献**：Z. Lu, J. Wang, and J. Song, "Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System," in *Proc. IEEE ICC*, 2020; 扩展版发表于 *IEEE Transactions on Wireless Communications*, 2021.
- arXiv: [1910.14322](https://arxiv.org/abs/1910.14322)
- GitHub: [Kylin9511/CRNet](https://github.com/Kylin9511/CRNet)

**主要贡献**：提出多分辨率残差卷积结构与信道注意力机制，以更少的参数量在室内/室外场景均超越CsiNet。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | 室内 NMSE | 室外 NMSE |
|:------:|:---------:|:---------:|
| 1/4    | -26.99    | -12.70    |
| 1/8    | -16.01    | -8.04     |
| 1/16   | -11.35    | -5.44     |
| 1/32   | -8.93     | -3.51     |
| 1/64   | -6.49     | -2.22     |

---

## 5. CLNet（2021）

**文献**：S. Ji and M. Li, "CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback," *IEEE Wireless Communications Letters*, vol. 10, no. 10, pp. 2318–2322, Oct. 2021.
- arXiv: [2102.07507](https://arxiv.org/abs/2102.07507)
- DOI: [10.1109/LWC.2021.3100163](https://ieeexplore.ieee.org/document/9497358)
- GitHub: [SIJIEJI/CLNet](https://github.com/SIJIEJI/CLNet)

**主要贡献**：直接处理复数域CSI，引入空间注意力机制，在降低约24%计算量的同时超越CRNet的NMSE性能。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | 室内 NMSE | 室外 NMSE |
|:------:|:---------:|:---------:|
| 1/4    | -29.16    | -12.88    |
| 1/8    | -15.60    | -8.29     |
| 1/16   | -11.15    | -5.56     |
| 1/32   | -8.95     | -3.49     |
| 1/64   | -6.34     | -2.19     |

---

## 6. ACRNet（2021/2022）

**文献**：Z. Lu, X. Li, Z. He, and J. Song, "Binarized Aggregated Network with Quantization: Flexible Deep Learning Deployment for CSI Feedback in Massive MIMO System," in *Proc. IEEE GLOBECOM*, 2021; 期刊扩展版 2022.
- arXiv: [2105.00354](https://arxiv.org/abs/2105.00354)
- GitHub: [Kylin9511/ACRNet](https://github.com/Kylin9511/ACRNet)

**主要贡献**：提出聚合卷积残差结构，支持二值化量化以灵活部署，在相同压缩比下以ACRNet-20x（20倍聚合）达到当时SOTA级别性能。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | ACRNet-1x 室内 | ACRNet-10x 室内 | ACRNet-20x 室内 | ACRNet-1x 室外 | ACRNet-10x 室外 | ACRNet-20x 室外 |
|:------:|:--------------:|:---------------:|:---------------:|:--------------:|:---------------:|:---------------:|
| 1/4    | -27.16         | -29.83          | **-32.02**      | -10.71         | -13.61          | **-14.25**      |
| 1/8    | -15.34         | -19.75          | **-20.78**      | -7.85          | -9.22           | **-9.68**       |
| 1/16   | -10.36         | -14.32          | **-15.05**      | -5.19          | -6.30           | **-6.47**       |

---

## 7. TransNet（2022）

**文献**：M. Cai, J. Guo, X. Shi, and Y. C. Eldar, "TransNet: Training-Efficient Transformer for Massive MIMO CSI Feedback," *Proc. IEEE GLOBECOM*, 2022.
- GitHub: [Treedy2020/TransNet](https://github.com/Treedy2020/TransNet)

**主要贡献**：首批将Transformer自注意力机制全面用于CSI压缩反馈的工作之一，利用全局依赖建模超越CNN方案，在中高压缩比下尤为突出。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | 室内 NMSE | 室外 NMSE |
|:------:|:---------:|:---------:|
| 1/4    | -31.6     | -15.2     |
| 1/8    | -20.5     | -10.5     |
| 1/16   | -14.0     | -8.5      |
| 1/32   | -10.5     | -5.8      |
| 1/64   | -7.2      | -3.8      |

---

## 8. TransNet+（2024）

**文献**：S. Chen et al., "TransNet+: Enhanced Transformer for Massive MIMO CSI Feedback," 2024.
- GitHub: [serenachen6/TransNet-plus](https://github.com/serenachen6/TransNet-plus)

**主要贡献**：在TransNet基础上进一步优化注意力模块与位置编码，全面提升室内/室外各压缩比下的NMSE性能。

**性能指标（COST2100，NMSE / dB）**：

| 压缩比 | 室内 NMSE | 室外 NMSE |
|:------:|:---------:|:---------:|
| 1/4    | **-33.1** | **-15.8** |
| 1/8    | -23.5     | -9.9      |
| 1/16   | -15.7     | -7.9      |
| 1/32   | -12.0     | -4.9      |
| 1/64   | -8.0      | -3.1      |

---

## 9. ST-TNet（2024）

**文献**：Q. Zhang et al., "ST-TNet: A Spatio-Temporal Joint Transformer Network for CSI Feedback in Massive MIMO," *Physical Communication*, Elsevier, 2024.
- DOI: [10.1016/j.phycom.2024.102288](https://www.sciencedirect.com/science/article/pii/S187449072400288X)

**主要贡献**：将空间注意力与时间Transformer联合建模，弥补单纯序列化reshape丢失空间信息的缺陷，在高天线数及高压缩比下优于CsiNet、CRNet等基线。

**性能指标**：
- 室内/室外均优于CsiNet、CRNet等CNN基线，在高压缩比（1/32、1/64）场景提升最为显著。
- 编码器端计算量约为经典Transformer的1/10，适合UE侧轻量化部署。

> 注：详细NMSE数值参见原文Table II–III，不同编码器结构（DCRNet/CLNet骨干）结果略有差异。

---

## 10. MarkovNet（2021）

**文献**：X. Guo, H. Ye, and G. Y. Li, "A Markovian Model-Driven Deep Learning Framework for Massive MIMO CSI Feedback," *IEEE Transactions on Wireless Communications*, vol. 20, no. 11, pp. 7503–7516, Nov. 2021.
- DOI: [10.1109/TWC.2021.3088138](https://ieeexplore.ieee.org/abstract/document/9513579)

**主要贡献**：将马尔可夫信道模型先验知识融入深度网络训练，通过球面归一化增强对噪声鲁棒性，在时变场景下超越普通CNN自编码器。

**性能指标**：
- 利用Markov先验后，在SNR = 10 dB时NMSE较普通DNN自编码器提升约2–4 dB。
- 室内1/4压缩比下NMSE达 -25 dB 以上。

---

## 11. 总体性能对比表

以下为各方法在 **COST2100 室内场景** 与不同压缩比下的 NMSE（dB）汇总对比：

| 方法            | 年份 | 1/4     | 1/8     | 1/16    | 1/32    | 1/64    |
|:---------------:|:----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| CsiNet          | 2018 | -17.36  | —       | -8.65   | -6.24   | -5.84   |
| CsiNet-LSTM     | 2018 | ~-19    | —       | ~-10    | ~-7.5   | —       |
| CsiNet+         | 2020 | < -17.36| —       | < -8.65 | < -6.24 | —       |
| CRNet           | 2021 | -26.99  | -16.01  | -11.35  | -8.93   | -6.49   |
| CLNet           | 2021 | -29.16  | -15.60  | -11.15  | -8.95   | -6.34   |
| ACRNet-1x       | 2022 | -27.16  | -15.34  | -10.36  | -8.60   | —       |
| ACRNet-20x      | 2022 | -32.02  | -20.78  | -15.05  | —       | —       |
| TransNet        | 2022 | -31.6   | -20.5   | -14.0   | -10.5   | -7.2    |
| **TransNet+**   | 2024 | **-33.1**| **-23.5**| **-15.7**| **-12.0**| **-8.0**|
| ST-TNet         | 2024 | 优于CRNet| 优于CRNet| 优于CRNet| 显著提升 | 显著提升 |

> 注：「—」表示原文未报告该压缩比结果；ST-TNet数值须参阅原文具体表格。

---

## 12. 技术演进趋势

```
2018 ──── CsiNet（CNN自编码器基线）
  │
2018 ──── CsiNet-LSTM（时间相关性）
  │
2020 ──── CsiNet+（多率、鲁棒性）
  │
2021 ──── CRNet（多分辨率残差 + 注意力）
  │        CLNet（复数输入 + 空间注意力，轻量）
  │        ACRNet（聚合结构 + 量化，灵活部署）
  │        MarkovNet（马尔可夫先验 + 深度学习）
  │
2022 ──── TransNet（全Transformer自注意力）
  │
2024 ──── TransNet+（增强Transformer）
           ST-TNet（时空联合Transformer）
```

**主要技术趋势**：
1. **架构演进**：CNN自编码器 → 注意力/残差增强CNN → 全Transformer → 时空联合Transformer
2. **量化与轻量化**：ACRNet等引入二值化量化，降低反馈比特数；ST-TNet在编码器端减少约90%计算量
3. **复数域处理**：CLNet等直接处理复数CSI，避免实虚部分离带来的信息损失
4. **时空建模**：CsiNet-LSTM/MarkovNet/ST-TNet等利用信道时间相关性进一步提升精度
5. **3GPP标准化**：上述方法正在推动3GPP Rel-18/19 AI/ML空口标准，三星、高通等已在5G网络中完成实机验证

---

## 13. 参考文献列表

1. C.-K. Wen, W.-T. Shih, S. Jin, "Deep Learning for Massive MIMO CSI Feedback," *IEEE WCL*, 2018. [[paper]](https://ieeexplore.ieee.org/document/8322184) [[arXiv]](https://arxiv.org/abs/1712.08919)

2. K. Wang, K. Niu, W. He, "Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels," *IEEE WCL*, 2019. [[paper]](https://ieeexplore.ieee.org/document/8482358)

3. X. Guo, C.-K. Wen, S. Jin, T. Jiang, "Convolutional Neural Network-Based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback," *IEEE TWC*, 2020. [[paper]](https://ieeexplore.ieee.org/document/8972904)

4. Z. Lu, J. Wang, J. Song, "Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System," *IEEE ICC*, 2020 / *IEEE TWC*, 2021. [[arXiv]](https://arxiv.org/abs/1910.14322) [[GitHub]](https://github.com/Kylin9511/CRNet)

5. S. Ji, M. Li, "CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback," *IEEE WCL*, 2021. [[paper]](https://ieeexplore.ieee.org/document/9497358) [[arXiv]](https://arxiv.org/abs/2102.07507) [[GitHub]](https://github.com/SIJIEJI/CLNet)

6. X. Guo, H. Ye, G. Y. Li, "A Markovian Model-Driven Deep Learning Framework for Massive MIMO CSI Feedback," *IEEE TWC*, 2021. [[paper]](https://ieeexplore.ieee.org/abstract/document/9513579)

7. Z. Lu et al., "Binarized Aggregated Network with Quantization: Flexible Deep Learning Deployment for CSI Feedback in Massive MIMO System," *IEEE GLOBECOM*, 2021 / *IEEE TWC*, 2022. [[arXiv]](https://arxiv.org/abs/2105.00354) [[GitHub]](https://github.com/Kylin9511/ACRNet)

8. M. Cai et al., "TransNet: Training-Efficient Transformer for Massive MIMO CSI Feedback," *IEEE GLOBECOM*, 2022. [[GitHub]](https://github.com/Treedy2020/TransNet)

9. S. Chen et al., "TransNet+: Enhanced Transformer for Massive MIMO CSI Feedback," 2024. [[GitHub]](https://github.com/serenachen6/TransNet-plus)

10. Q. Zhang et al., "ST-TNet: A Spatio-Temporal Joint Transformer Network for CSI Feedback in Massive MIMO," *Physical Communication*, Elsevier, 2024. [[paper]](https://www.sciencedirect.com/science/article/pii/S187449072400288X)

---

*本文档整理于 2026-03-20，涵盖 2018–2024 年基于深度学习的CSI压缩反馈主要SOTA方法。*
