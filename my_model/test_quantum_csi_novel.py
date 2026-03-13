"""
测试量子增强CSI反馈模块（quantum_csi_novel.py）的单元测试。
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 测试专利1：量子纠缠空间相关性建模
# ---------------------------------------------------------------------------

class TestQuantumSpatialCorrelationBlock:
    def _make_block(self, encoded_dim=32):
        from my_model.quantum_csi_novel import QuantumSpatialCorrelationBlock
        return QuantumSpatialCorrelationBlock(
            n_qubits=4, n_layers=1, window_size=2, use_bell_init=True
        )

    def test_output_shape(self):
        block = self._make_block()
        x = torch.randn(2, 32)
        out = block(x)
        assert out.shape == (2, 32), f"期望 (2,32)，实际 {out.shape}"

    def test_output_is_float32(self):
        block = self._make_block()
        x = torch.randn(2, 32)
        out = block(x)
        assert out.dtype == torch.float32

    def test_gradients_flow(self):
        block = self._make_block()
        x = torch.randn(2, 32)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"参数 {name} 没有梯度"

    def test_bell_phases_are_trainable(self):
        block = self._make_block()
        assert block.bell_phases.requires_grad

    def test_invalid_input_dim_raises(self):
        from my_model.quantum_csi_novel import QuantumSpatialCorrelationBlock
        block = QuantumSpatialCorrelationBlock(n_qubits=4, n_layers=1, window_size=2)
        with pytest.raises(ValueError):
            block(torch.randn(2, 33))  # 33 不能被 4 整除

    def test_wrong_n_qubits_raises(self):
        from my_model.quantum_csi_novel import QuantumSpatialCorrelationBlock
        with pytest.raises(ValueError):
            QuantumSpatialCorrelationBlock(n_qubits=5, n_layers=1, window_size=2)

    def test_without_bell_init(self):
        from my_model.quantum_csi_novel import QuantumSpatialCorrelationBlock
        block = QuantumSpatialCorrelationBlock(
            n_qubits=4, n_layers=1, window_size=2, use_bell_init=False
        )
        x = torch.randn(2, 32)
        out = block(x)
        assert out.shape == (2, 32)


# ---------------------------------------------------------------------------
# 测试专利2：量子多头注意力
# ---------------------------------------------------------------------------

class TestQuantumAttentionHead:
    def test_output_shape(self):
        from my_model.quantum_csi_novel import QuantumAttentionHead
        batch_size = 3
        n_qubits = 4
        for topology in ("linear", "star", "full"):
            head = QuantumAttentionHead(n_qubits=n_qubits, n_layers=1, topology=topology)
            x = torch.randn(batch_size, n_qubits) * math.pi
            out = head(x)
            assert out.shape == (batch_size, n_qubits), \
                f"[{topology}] 期望 ({batch_size},{n_qubits})，实际 {out.shape}"

    def test_invalid_topology_raises(self):
        from my_model.quantum_csi_novel import QuantumAttentionHead
        with pytest.raises(ValueError):
            QuantumAttentionHead(n_qubits=4, n_layers=1, topology="invalid")

    def test_gradients(self):
        from my_model.quantum_csi_novel import QuantumAttentionHead
        head = QuantumAttentionHead(n_qubits=4, n_layers=1, topology="linear")
        x = torch.randn(2, 4) * math.pi
        out = head(x)
        out.sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"{name} 无梯度"


class TestQuantumMultiHeadAttentionBlock:
    def test_output_shape_dim32(self):
        from my_model.quantum_csi_novel import QuantumMultiHeadAttentionBlock
        block = QuantumMultiHeadAttentionBlock(encoded_dim=32, n_heads=3, n_qubits=4)
        x = torch.randn(2, 32)
        out = block(x)
        assert out.shape == (2, 32)

    def test_output_shape_dim64(self):
        from my_model.quantum_csi_novel import QuantumMultiHeadAttentionBlock
        block = QuantumMultiHeadAttentionBlock(encoded_dim=64, n_heads=2, n_qubits=4)
        x = torch.randn(2, 64)
        out = block(x)
        assert out.shape == (2, 64)

    def test_residual_scale_trainable(self):
        from my_model.quantum_csi_novel import QuantumMultiHeadAttentionBlock
        block = QuantumMultiHeadAttentionBlock(encoded_dim=32, n_heads=2, n_qubits=4)
        assert block.residual_scale.requires_grad

    def test_gradients(self):
        from my_model.quantum_csi_novel import QuantumMultiHeadAttentionBlock
        block = QuantumMultiHeadAttentionBlock(encoded_dim=32, n_heads=2, n_qubits=4)
        x = torch.randn(2, 32)
        out = block(x)
        out.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} 无梯度"

    def test_non_square_n_qubits_raises(self):
        from my_model.quantum_csi_novel import QuantumMultiHeadAttentionBlock
        with pytest.raises(ValueError):
            QuantumMultiHeadAttentionBlock(encoded_dim=32, n_heads=2, n_qubits=3)


# ---------------------------------------------------------------------------
# 测试专利3：量子自适应压缩率分配
# ---------------------------------------------------------------------------

class TestQuantumRateAllocator:
    def test_output_shapes(self):
        from my_model.quantum_csi_novel import QuantumRateAllocator
        alloc = QuantumRateAllocator(input_dim=32, n_qubits=5, n_layers=1,
                                     candidate_dims=[512, 128, 64, 32, 16])
        x = torch.randn(2, 32)
        logits, selected = alloc(x, hard=False)
        assert logits.shape == (2, 5)
        assert selected.shape == (2,)

    def test_hard_selection(self):
        from my_model.quantum_csi_novel import QuantumRateAllocator
        alloc = QuantumRateAllocator(input_dim=32, n_qubits=5, n_layers=1,
                                     candidate_dims=[512, 128, 64, 32, 16])
        x = torch.randn(2, 32)
        _, selected = alloc(x, hard=True)
        for idx in selected.tolist():
            assert 0 <= idx < 5

    def test_get_selected_dim(self):
        from my_model.quantum_csi_novel import QuantumRateAllocator
        dims = [512, 128, 64, 32, 16]
        alloc = QuantumRateAllocator(input_dim=32, n_qubits=5, n_layers=1,
                                     candidate_dims=dims)
        x = torch.randn(4, 32)
        chosen = alloc.get_selected_dim(x)
        assert chosen in dims

    def test_too_few_qubits_raises(self):
        from my_model.quantum_csi_novel import QuantumRateAllocator
        with pytest.raises(ValueError):
            QuantumRateAllocator(input_dim=32, n_qubits=3, n_layers=1,
                                 candidate_dims=[512, 128, 64, 32, 16])

    def test_gradients(self):
        from my_model.quantum_csi_novel import QuantumRateAllocator
        alloc = QuantumRateAllocator(input_dim=32, n_qubits=5, n_layers=1,
                                     candidate_dims=[512, 128, 64, 32, 16])
        x = torch.randn(2, 32)
        logits, _ = alloc(x, hard=False)
        logits.sum().backward()
        for name, param in alloc.named_parameters():
            assert param.grad is not None, f"{name} 无梯度"


# ---------------------------------------------------------------------------
# 测试完整网络：CsiNetQuantumNovel
# ---------------------------------------------------------------------------

class TestCsiNetQuantumNovel:
    def test_forward_shape(self):
        from my_model.quantum_csi_novel import CsiNetQuantumNovel
        model = CsiNetQuantumNovel(encoded_dim=32, alpha=0.1)
        x = torch.randn(2, 2, 32, 32)
        out = model(x)
        assert out.shape == (2, 2, 32, 32), f"期望 (2,2,32,32)，实际 {out.shape}"

    def test_output_in_0_1(self):
        """输出层使用 sigmoid，应在 [0,1] 范围内。"""
        from my_model.quantum_csi_novel import CsiNetQuantumNovel
        model = CsiNetQuantumNovel(encoded_dim=32, alpha=0.1)
        x = torch.randn(2, 2, 32, 32)
        out = model(x)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_encode_decode(self):
        from my_model.quantum_csi_novel import CsiNetQuantumNovel
        model = CsiNetQuantumNovel(encoded_dim=32, alpha=0.1)
        x = torch.randn(2, 2, 32, 32)
        s = model.encode(x)
        assert s.shape == (2, 32)
        x_hat = model.decode(s)
        assert x_hat.shape == (2, 2, 32, 32)

    def test_rate_allocator_disabled_raises(self):
        from my_model.quantum_csi_novel import CsiNetQuantumNovel
        model = CsiNetQuantumNovel(encoded_dim=32, use_rate_allocator=False)
        x = torch.randn(2, 2, 32, 32)
        with pytest.raises(RuntimeError):
            model.predict_rate(x)

    def test_rate_allocator_enabled(self):
        from my_model.quantum_csi_novel import CsiNetQuantumNovel
        model = CsiNetQuantumNovel(encoded_dim=32, use_rate_allocator=True)
        x = torch.randn(2, 2, 32, 32)
        logits, dims = model.predict_rate(x)
        assert logits.shape[0] == 2
        assert dims.shape == (2,)

    def test_end_to_end_gradient(self):
        """验证损失可以反向传播到编码器参数。"""
        from my_model.quantum_csi_novel import CsiNetQuantumNovel
        model = CsiNetQuantumNovel(encoded_dim=32, alpha=0.1)
        x = torch.randn(2, 2, 32, 32)
        x_hat = model(x)
        loss = nn.MSELoss()(x_hat, x)
        loss.backward()
        enc_weight = model.encoder.fc.weight
        assert enc_weight.grad is not None, "编码器 fc 权重无梯度"
