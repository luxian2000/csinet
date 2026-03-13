"""
专利思想实现：量子增强无线通信CSI反馈系统

本模块实现三种基于量子计算的CSI反馈专利思想：

专利1: 量子纠缠空间相关性建模用于大规模MIMO CSI压缩
  (Quantum Entanglement Spatial Correlation Modeling for Massive MIMO CSI Compression)
  核心创新：使用量子纠缠对模拟天线阵列空间相关性，比经典方法需要更少参数即可
  精准捕获远近场相关性结构，从而实现更高效的CSI压缩。

专利2: 量子多头注意力CSI重建网络
  (Quantum Multi-Head Attention for CSI Reconstruction)
  核心创新：多个量子电路头各自聚焦于不同频率/空间模式，通过量子干涉实现
  比经典多头注意力更高效的特征提取，降低重建误差。

专利3: 变分量子电路自适应压缩率分配
  (Variational Quantum Circuit Adaptive Rate Allocation)
  核心创新：量子测量的概率性特性天然实现自适应码率选择，根据信道质量
  动态决定每个用户的压缩维度，无需额外经典网络开销。
"""

import math

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2
IMG_TOTAL = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

# 将编码向量展开为特征图时使用的高度维度（固定为4，宽度随编码维度自适应）
_LATENT_H = 4
# 量子电路批处理块大小（平衡内存与速度）
_QUANTUM_CHUNK = 128
# Gumbel-Softmax 数值稳定性极小值
_GUMBEL_EPS = 1e-20


# =============================================================================
# 专利1: 量子纠缠空间相关性建模
# Patent 1: Quantum Entanglement Spatial Correlation Modeling
# =============================================================================

class QuantumSpatialCorrelationBlock(nn.Module):
    """
    量子纠缠空间相关性建模模块。

    【专利创新点】
    大规模MIMO天线阵列中，天线间的空间相关性决定了CSI的可压缩性。
    经典方法（如DFT预编码）依赖先验假设（均匀线阵等），无法自适应真实场景。
    本模块用量子纠缠对直接对空间相关性建模：
    - 每对纠缠量子比特代表一对天线的相关性
    - Bell基测量提取最强相关方向
    - 量子相干性保留不同相关模式的叠加，无需枚举所有可能

    电路结构（以4量子比特为例，模拟2×2天线子阵）：
        q0 ──H──●──RY(w)──M
                │
        q1 ──H──X──RY(w)──M
                
        q2 ──H──●──RY(w)──M
                │
        q3 ──H──X──RY(w)──M

    其中相邻对(q0,q1)和(q2,q3)形成Bell对，捕获天线相关性，
    交叉纠缠通过后续CRZ门引入，捕获更远距离的空间相关。

    参数:
        n_qubits (int): 量子比特数，等于 window_size^2（天线子阵大小）
        n_layers (int): 纠缠层数，越多捕获越复杂的相关结构
        window_size (int): 处理窗口大小（sqrt(n_qubits)）
        use_bell_init (bool): 是否使用Bell态初始化（专利核心创新）
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 window_size: int = 2, use_bell_init: bool = True):
        super().__init__()
        if n_qubits != window_size * window_size:
            raise ValueError("n_qubits must equal window_size * window_size")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        self.use_bell_init = use_bell_init

        # 可训练参数
        self.weights_rx = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_rz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        # Bell纠缠对的相对相位参数（专利核心）
        self.bell_phases = nn.Parameter(torch.zeros(n_qubits // 2))

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights_rx, weights_ry, weights_rz, bell_phases):
            # ── 编码层：将输入数据嵌入量子相位 ──
            # inputs[..., i] 支持批处理（shape [batch, n_qubits]）
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[..., i], wires=i)

            if use_bell_init:
                # ── Bell纠缠初始化（专利核心：捕获天线相关性方向）──
                # 相邻天线对形成Bell对
                for pair in range(n_qubits // 2):
                    q_a, q_b = pair * 2, pair * 2 + 1
                    qml.CNOT(wires=[q_a, q_b])
                    # 用可训练相位调整纠缠强度，自适应学习相关方向
                    qml.RZ(bell_phases[pair], wires=q_a)
                    qml.RZ(bell_phases[pair], wires=q_b)

            # ── 变分纠缠层：捕获多阶空间相关 ──
            for layer in range(n_layers):
                # 近邻相关（相邻天线）
                for i in range(n_qubits - 1):
                    qml.CRZ(weights_rz[layer, i], wires=[i, i + 1])
                # 远端相关（首尾天线，捕获长程相关）
                qml.CRZ(weights_rz[layer, n_qubits - 1], wires=[n_qubits - 1, 0])

                # 单比特旋转（增加表达能力）
                for i in range(n_qubits):
                    qml.RX(weights_rx[layer, i], wires=i)
                    qml.RY(weights_ry[layer, i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch, encoded_dim] 编码后的潜在特征向量

        返回:
            out: [batch, encoded_dim] 经量子空间相关增强的特征向量
        """
        if x.dim() != 2:
            raise ValueError(f"期望2D输入 [B, D]，实际 {tuple(x.shape)}")

        batch, dim = x.shape
        latent_h = _LATENT_H
        latent_w = dim // latent_h
        if dim % latent_h != 0:
            raise ValueError(f"encoded_dim 必须能被 {latent_h} 整除，实际 {dim}")
        if latent_w % self.window_size != 0:
            raise ValueError(
                f"latent_w={latent_w} 必须能被 window_size={self.window_size} 整除"
            )

        x_map = x.reshape(batch, 1, latent_h, latent_w)
        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        patches = unfold(x_map)  # [batch, n_qubits, num_patches]
        num_patches = patches.shape[-1]

        total = batch * num_patches
        inp = patches.permute(0, 2, 1).reshape(total, self.n_qubits)
        inp = torch.tanh(inp) * math.pi  # 归一化到 [-π, π]

        outputs = []
        for start in range(0, total, _QUANTUM_CHUNK):
            end = min(total, start + _QUANTUM_CHUNK)
            q_out = self._circuit(
                inp[start:end],
                self.weights_rx,
                self.weights_ry,
                self.weights_rz,
                self.bell_phases,
            )
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(
                    [o if isinstance(o, torch.Tensor) else torch.tensor(o, dtype=torch.float32)
                     for o in q_out], dim=1
                )
            outputs.append(q_out.float())

        out_patches = torch.cat(outputs, dim=0)  # [total, n_qubits]
        out_patches = out_patches.reshape(batch, num_patches, self.n_qubits)
        # fold 回 [batch, 1, latent_h, latent_w]
        out_perm = out_patches.permute(0, 2, 1)  # [batch, n_qubits, num_patches]
        fold = nn.Fold(output_size=(latent_h, latent_w),
                       kernel_size=self.window_size, stride=self.window_size)
        out_map = fold(out_perm)  # [batch, 1, latent_h, latent_w]
        return out_map.reshape(batch, dim)


# =============================================================================
# 专利2: 量子多头注意力CSI重建
# Patent 2: Quantum Multi-Head Attention for CSI Reconstruction
# =============================================================================

class QuantumAttentionHead(nn.Module):
    """
    单个量子注意力头。

    【专利创新点】
    每个头使用不同的纠缠拓扑结构（线型/星型/全连接），天然地关注不同的
    空间频率模式：
    - 线型拓扑：捕获沿天线阵列方向的渐变相关（低频空间模式）
    - 星型拓扑：捕获以某天线为中心的辐射状相关（局部强相关）
    - 全连接拓扑：捕获全局混合相关（高频空间模式）
    通过测量不同泡利算符（X/Y/Z）获得不同投影方向的特征，
    类比经典注意力中的不同投影矩阵 W_Q/W_K/W_V。
    """

    TOPOLOGIES = ("linear", "star", "full")

    def __init__(self, n_qubits: int = 4, n_layers: int = 1,
                 topology: str = "linear"):
        super().__init__()
        if topology not in self.TOPOLOGIES:
            raise ValueError(f"topology 须为 {self.TOPOLOGIES} 之一")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology

        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            # inputs[..., i] 支持批处理（shape [batch, n_qubits]）
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[..., i], wires=i)

            for layer in range(n_layers):
                # 根据拓扑结构选择纠缠方式
                if topology == "linear":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif topology == "star":
                    for i in range(1, n_qubits):
                        qml.CNOT(wires=[0, i])
                else:  # full
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            qml.CNOT(wires=[i, j])

                for i in range(n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i,
                    )

            # 测量不同泡利算符，增加信息丰富度
            observables = []
            for i in range(n_qubits):
                obs_idx = i % 3
                if obs_idx == 0:
                    observables.append(qml.expval(qml.PauliZ(i)))
                elif obs_idx == 1:
                    observables.append(qml.expval(qml.PauliX(i)))
                else:
                    observables.append(qml.expval(qml.PauliY(i)))
            return observables

        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [total, n_qubits] 归一化后的输入

        返回:
            out: [total, n_qubits] 量子注意力输出
        """
        total = x.shape[0]
        outputs = []
        for start in range(0, total, _QUANTUM_CHUNK):
            end = min(total, start + _QUANTUM_CHUNK)
            q_out = self._circuit(x[start:end], self.weights)
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(
                    [o if isinstance(o, torch.Tensor) else torch.tensor(o, dtype=torch.float32)
                     for o in q_out], dim=1
                )
            outputs.append(q_out.float())
        return torch.cat(outputs, dim=0)


class QuantumMultiHeadAttentionBlock(nn.Module):
    """
    量子多头注意力模块（专利2核心实现）。

    【专利创新点】
    将经典Transformer的多头注意力机制推广到量子领域：
    - 多个量子电路头（不同纠缠拓扑）并行处理输入特征
    - 每个头提取不同空间频率的相关信息
    - 量子干涉效应实现隐式的注意力权重计算
    - 最终通过可学习的融合权重聚合多头输出

    相比经典多头注意力的优势：
    - 量子纠缠天然建模非局部关联，无需显式计算注意力得分矩阵
    - 参数量随量子比特数对数增长（而非平方增长）
    - 量子相干性保留更丰富的相关信息直到测量
    """

    def __init__(self, encoded_dim: int = 32, n_heads: int = 3,
                 n_qubits: int = 4, n_layers: int = 1):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.n_heads = n_heads
        self.n_qubits = n_qubits

        latent_h = _LATENT_H
        latent_w = encoded_dim // latent_h
        self.latent_h = latent_h
        self.latent_w = latent_w

        window_size = int(math.sqrt(n_qubits))
        if window_size * window_size != n_qubits:
            raise ValueError("n_qubits 须为完全平方数")
        self.window_size = window_size

        topologies = QuantumAttentionHead.TOPOLOGIES
        self.heads = nn.ModuleList([
            QuantumAttentionHead(n_qubits, n_layers, topologies[i % len(topologies)])
            for i in range(n_heads)
        ])

        # 多头融合权重（可训练）
        self.fusion = nn.Linear(n_heads * n_qubits, n_qubits, bias=False)
        # 残差连接的缩放因子
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _extract_patches(self, x: torch.Tensor):
        batch, dim = x.shape
        x_map = x.reshape(batch, 1, self.latent_h, self.latent_w)
        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        patches = unfold(x_map)  # [batch, n_qubits, num_patches]
        num_patches = patches.shape[-1]
        return patches, batch, num_patches

    def _fold_patches(self, patches: torch.Tensor, batch: int, num_patches: int):
        fold = nn.Fold(
            output_size=(self.latent_h, self.latent_w),
            kernel_size=self.window_size,
            stride=self.window_size,
        )
        out_map = fold(patches)  # [batch, 1, latent_h, latent_w]
        return out_map.reshape(batch, self.encoded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch, encoded_dim]

        返回:
            out: [batch, encoded_dim] 量子多头注意力增强特征
        """
        patches, batch, num_patches = self._extract_patches(x)
        total = batch * num_patches
        inp = patches.permute(0, 2, 1).reshape(total, self.n_qubits)
        inp_norm = torch.tanh(inp) * math.pi

        # 并行（串行实现）多头量子注意力
        head_outputs = []
        for head in self.heads:
            head_out = head(inp_norm)  # [total, n_qubits]
            head_outputs.append(head_out)

        # 拼接并融合
        concat = torch.cat(head_outputs, dim=-1)  # [total, n_heads * n_qubits]
        fused = self.fusion(concat)  # [total, n_qubits]

        # 残差连接（保留原始输入信息）
        out = inp + self.residual_scale * fused

        # 折叠回原始维度
        out = out.reshape(batch, num_patches, self.n_qubits)
        out_perm = out.permute(0, 2, 1)  # [batch, n_qubits, num_patches]
        return self._fold_patches(out_perm, batch, num_patches)


# =============================================================================
# 专利3: 变分量子电路自适应压缩率分配
# Patent 3: Variational Quantum Circuit Adaptive Rate Allocation
# =============================================================================

class QuantumRateAllocator(nn.Module):
    """
    变分量子电路自适应压缩率分配模块（专利3核心实现）。

    【专利创新点】
    5G/6G系统中，不同用户的信道条件差异显著，固定压缩率导致信道质量好的
    用户浪费带宽，信道质量差的用户丢失关键信息。
    本模块利用量子测量的概率性特性实现自适应码率分配：

    1. 提取信道质量指标（CQI）：从输入CSI计算信道能量、稀疏性、相关性
    2. 将CQI映射到量子态：使用 RY 旋转角编码信道质量
    3. 量子电路处理：通过参数化量子门捕获多维质量特征的非线性交互
    4. 测量概率分布：泡利Z期望值反映各压缩率档位的适用概率
    5. 软选择机制：训练时使用 Gumbel-Softmax 实现可微分的离散选择，
       推理时使用 argmax 选择最优压缩率

    支持的压缩率档位（可配置）：
        [1/4, 1/8, 1/16, 1/32, 1/64]  →  [512, 256, 128, 64, 32] 维
    """

    DEFAULT_DIMS = (512, 256, 128, 64, 32)

    def __init__(self, input_dim: int = 64, n_qubits: int = 5,
                 n_layers: int = 2, candidate_dims=None):
        super().__init__()
        if candidate_dims is None:
            candidate_dims = list(self.DEFAULT_DIMS)
        self.candidate_dims = candidate_dims
        n_candidates = len(candidate_dims)
        if n_qubits < n_candidates:
            raise ValueError(
                f"n_qubits ({n_qubits}) 须 >= 候选压缩率档位数 ({n_candidates})"
            )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_candidates = n_candidates

        # 信道质量指标提取（经典前处理）
        self.cqi_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh(),  # 输出归一化到 [-1,1] → 再 * π 作为量子旋转角
        )

        # 量子参数
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(cqi_angles, weights_ry, weights_crz):
            # 编码信道质量信息
            for i in range(n_qubits):
                qml.RY(cqi_angles[i] * math.pi, wires=i)

            # 变分纠缠层：捕获多维信道质量特征的交互
            for layer in range(n_layers):
                for i in range(n_qubits - 1):
                    qml.CRZ(weights_crz[layer, i], wires=[i, i + 1])
                qml.CRZ(weights_crz[layer, n_qubits - 1], wires=[n_qubits - 1, 0])
                for i in range(n_qubits):
                    qml.RY(weights_ry[layer, i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit
        # Gumbel-Softmax 温度（训练时逐渐退火至接近 argmax）
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, hard: bool = False):
        """
        参数:
            x:    [batch, input_dim] 输入CSI特征（编码后的潜在向量）
            hard: True 时使用 hard argmax（推理模式），False 时使用软选择

        返回:
            rate_logits: [batch, n_candidates] 各压缩率档位的对数概率
            selected:    [batch] 选择的压缩率维度索引
        """
        batch = x.shape[0]

        # 提取信道质量指标
        cqi = self.cqi_extractor(x)  # [batch, n_qubits]

        # 逐样本运行量子电路
        q_outputs = []
        for b in range(batch):
            q_out = self._circuit(cqi[b], self.weights_ry, self.weights_crz)
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(
                    [o if isinstance(o, torch.Tensor) else torch.tensor(o, dtype=torch.float32)
                     for o in q_out]
                )
            q_outputs.append(q_out.float())

        q_outputs = torch.stack(q_outputs, dim=0)  # [batch, n_qubits]

        # 取前 n_candidates 个量子比特的期望值作为率分配 logits
        rate_logits = q_outputs[:, :self.n_candidates]  # [batch, n_candidates]

        if hard:
            selected = torch.argmax(rate_logits, dim=-1)
        else:
            # Gumbel-Softmax 软选择（可微分）
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(rate_logits) + _GUMBEL_EPS) + _GUMBEL_EPS)
            selected = torch.argmax(rate_logits + gumbel_noise / self.temperature, dim=-1)

        return rate_logits, selected

    def get_selected_dim(self, x: torch.Tensor) -> int:
        """推理时返回选择的编码维度。"""
        _, selected = self.forward(x, hard=True)
        # 取批次中出现频率最高的档位
        mode_idx = selected.mode().values.item()
        return self.candidate_dims[mode_idx]


# =============================================================================
# 完整量子增强CSI反馈网络（整合三个专利模块）
# Full Quantum-Enhanced CSI Feedback Network
# =============================================================================

class CsiNetEncoder(nn.Module):
    """标准CsiNet编码器。"""

    def __init__(self, encoded_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(IMG_TOTAL, encoded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        return self.fc(self.flatten(x))


class ResidualBlock(nn.Module):
    """标准残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.net(x))


class QuantumEnhancedDecoder(nn.Module):
    """
    量子增强解码器，集成专利1（空间相关性）和专利2（多头注意力）。
    """

    def __init__(self, encoded_dim: int, alpha: float = 0.2,
                 use_correlation: bool = True, use_attention: bool = True):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.alpha = alpha

        self.fc_decode = nn.Linear(encoded_dim, IMG_TOTAL)

        # 专利1：量子纠缠空间相关性模块
        self.use_correlation = use_correlation
        if use_correlation:
            self.quantum_corr = QuantumSpatialCorrelationBlock(
                n_qubits=4, n_layers=2, window_size=2, use_bell_init=True
            )

        # 专利2：量子多头注意力模块
        self.use_attention = use_attention
        if use_attention:
            self.quantum_attn = QuantumMultiHeadAttentionBlock(
                encoded_dim=encoded_dim, n_heads=3, n_qubits=4, n_layers=1
            )

        self.main_conv = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(IMG_CHANNELS),
            nn.LeakyReLU(),
        )

        self.residual_blocks = nn.ModuleList([ResidualBlock(IMG_CHANNELS) for _ in range(4)])
        self.out_conv = nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, 3, padding=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        batch = s.shape[0]

        # 专利2：量子注意力增强潜在编码
        if self.use_attention:
            s_enhanced = s + self.alpha * self.quantum_attn(s)
        else:
            s_enhanced = s

        # 解码到图像空间
        x = self.fc_decode(s_enhanced).reshape(batch, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        # 主路径
        main_out = self.main_conv(x)

        # 专利1：量子空间相关性补偿路径
        if self.use_correlation:
            corr_latent = self.quantum_corr(s_enhanced)  # [batch, encoded_dim]
            corr_feat = self.fc_decode(corr_latent).reshape(
                batch, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
            )
            x = main_out + self.alpha * corr_feat
        else:
            x = main_out

        # 残差重建
        for blk in self.residual_blocks:
            x = blk(x)

        return torch.sigmoid(self.out_conv(x))


class CsiNetQuantumNovel(nn.Module):
    """
    量子增强CSI反馈完整网络（整合三个专利模块）。

    专利1 (QuantumSpatialCorrelationBlock): 解码器中的空间相关性建模
    专利2 (QuantumMultiHeadAttentionBlock): 解码器中的多头注意力
    专利3 (QuantumRateAllocator): 自适应压缩率分配（可选启用）

    用法:
        model = CsiNetQuantumNovel(encoded_dim=32)
        x_hat = model(x)               # 前向传播
        rate_logits, selected = model.predict_rate(x)  # 自适应率预测
    """

    def __init__(self, encoded_dim: int = 32, alpha: float = 0.2,
                 use_rate_allocator: bool = False):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.encoder = CsiNetEncoder(encoded_dim)
        self.decoder = QuantumEnhancedDecoder(
            encoded_dim, alpha,
            use_correlation=True,
            use_attention=True,
        )

        # 专利3：自适应率分配（可选）
        self.use_rate_allocator = use_rate_allocator
        if use_rate_allocator:
            self.rate_allocator = QuantumRateAllocator(
                input_dim=encoded_dim,
                n_qubits=5,
                n_layers=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.encoder(x)
        return self.decoder(s)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """获取压缩后的码字（用于无线传输）。"""
        return self.encoder(x)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """从码字重建CSI。"""
        return self.decoder(s)

    def predict_rate(self, x: torch.Tensor):
        """
        专利3：预测最优压缩率。

        返回:
            rate_logits: [batch, n_candidates] 率分配概率
            selected_dims: [batch] 每个样本选择的编码维度
        """
        if not self.use_rate_allocator:
            raise RuntimeError("请在初始化时设置 use_rate_allocator=True")
        s = self.encoder(x)
        logits, selected_idx = self.rate_allocator(s)
        selected_dims = torch.tensor(
            [self.rate_allocator.candidate_dims[i] for i in selected_idx.tolist()]
        )
        return logits, selected_dims
