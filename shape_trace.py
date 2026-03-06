#!/usr/bin/env python
"""
精确追踪量子补偿块中的形状变化
"""
import torch
import torch.nn as nn

print("=" * 70)
print("追踪量子补偿块的数据形状")
print("=" * 70)

# 测试参数
batch, ch, h, w = 2, 2, 8, 8
window_size = 2
n_qubits = 4

print(f"\n输入配置:")
print(f"  batch={batch}, channels={ch}, height={h}, width={w}")
print(f"  window_size={window_size}, n_qubits={n_qubits}")

# 步骤 1: Unfold
print(f"\n步骤 1: Unfold")
x = torch.randn(batch, ch, h, w)
print(f"  输入 x: {x.shape}")

unfold = nn.Unfold(kernel_size=window_size, stride=window_size)
patches = unfold(x)
print(f"  unfold(x): {patches.shape}")
print(f"    解释: [batch={batch}, ch*window_size²={ch*window_size*window_size}, num_patches={patches.shape[-1]}]")

num_patches = patches.shape[-1]

# 步骤 2: Permute 和 Reshape
print(f"\n步骤 2: Permute + Reshape")
patches = patches.permute(0, 2, 1)
print(f"  permute(0, 2, 1): {patches.shape}")

patches = patches.reshape(batch, num_patches, ch, 4)
print(f"  reshape(batch={batch}, num_patches={num_patches}, ch={ch}, 4): {patches.shape}")

# 步骤 3: Flatten 用于量子电路
print(f"\n步骤 3: Flatten 用于量子电路")
total_samples = batch * num_patches * ch
all_inputs = patches.reshape(total_samples, 4)
print(f"  reshape(batch*num_patches*ch, 4): {all_inputs.shape}")
print(f"    = [{batch}*{num_patches}*{ch}, 4] = [{total_samples}, 4]")

# 步骤 4: 模拟量子电路(每个4值输入 → n_qubits输出)
print(f"\n步骤 4: 模拟量子电路")
print(f"  每个输入 [4] 通过量子电路得到 [n_qubits={n_qubits}] 输出")
all_outputs = torch.randn(total_samples, n_qubits)  # 模拟量子输出
print(f"  量子输出: {all_outputs.shape}")

# 步骤 5: Reshape 回原结构
print(f"\n步骤 5: Reshape 回结构")
all_outputs = all_outputs.reshape(batch, num_patches, ch, n_qubits)
print(f"  reshape(batch={batch}, num_patches={num_patches}, ch={ch}, n_qubits={n_qubits}): {all_outputs.shape}")

# 步骤 6: 为 Fold 重塑
print(f"\n步骤 6: 为 Fold 重塑")
all_outputs_permute = all_outputs.permute(0, 2, 3, 1)
print(f"  permute(0, 2, 3, 1): {all_outputs_permute.shape}")
print(f"    维度顺序: [batch, ch, n_qubits, num_patches]")

all_outputs_final = all_outputs_permute.reshape(batch, ch*n_qubits, num_patches)
print(f"  reshape(batch={batch}, ch*n_qubits={ch*n_qubits}, num_patches={num_patches}): {all_outputs_final.shape}")

# 步骤 7: Fold
print(f"\n步骤 7: Fold")
fold = nn.Fold(output_size=(h, w), kernel_size=window_size, stride=window_size)
output = fold(all_outputs_final)
print(f"  fold 输入: {all_outputs_final.shape}")
print(f"  fold 输出: {output.shape}")
print(f"  ✓ 最终输出通道数: {output.shape[1]}")

print(f"\n" + "=" * 70)
print("结论")
print("=" * 70)
print(f"输入通道数:  {ch}")
print(f"输出通道数:  {output.shape[1]} (= ch × n_qubits = {ch} × {n_qubits})")
print(f"\n⚠️  通道数从 {ch} 扩展到 {ch*n_qubits}!")
print(f"\n这意味着:")
print(f"  - 每个通道的 4 个像素被 4 量子比特电路处理")
print(f"  - 输出是 4 个测量值，被视为 {n_qubits} 个新通道")
print(f"  - 原始 {ch} 个通道 → {output.shape[1]} 个通道")
