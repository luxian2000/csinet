"""
验证 Unfold 分块计算过程
"""
import torch
import torch.nn as nn

# 配置参数
batch_size = 1
encoded_dim = 32
latent_h = 4
latent_w = encoded_dim // latent_h  # 8
window_size = 2

print("=" * 60)
print("Unfold 分块详细计算过程")
print("=" * 60)
print()

# 1. 输入重塑
x = torch.randn(batch_size, encoded_dim)
x_map = x.reshape(batch_size, 1, latent_h, latent_w)
print(f"1. 输入重塑:")
print(f"   原始形状：[{batch_size}, {encoded_dim}]")
print(f"   重塑后：{x_map.shape}")
print(f"   解释：[B, C=1, H={latent_h}, W={latent_w}]")
print()

# 2. Unfold 操作
unfold = nn.Unfold(kernel_size=window_size, stride=window_size)
patches = unfold(x_map)

print(f"2. Unfold 操作:")
print(f"   kernel_size: {window_size}×{window_size}")
print(f"   stride: {window_size}")
print(f"   输出形状：{patches.shape}")
print(f"   解释：[B, C*kernel*kernel={1*2*2}, num_patches]")
print()

# 3. 计算 patches 数量
num_patches = patches.shape[-1]
print(f"3. Patches 数量:")
print(f"   num_patches = {num_patches}")
print()

# 4. 理论计算验证
# 对于空间尺寸 H×W = 4×8，使用 2×2 窗口，步长 2
# Height 方向可以提取：(4-2)//2 + 1 = 2 个
# Width 方向可以提取：(8-2)//2 + 1 = 4 个
# 总计：2 × 4 = 8 个
expected_h = (latent_h - window_size) // window_size + 1
expected_w = (latent_w - window_size) // window_size + 1
expected_total = expected_h * expected_w

print(f"4. 理论验证:")
print(f"   Height 方向：({latent_h}-{window_size})//{window_size}+1 = {expected_h}")
print(f"   Width 方向：({latent_w}-{window_size})//{window_size}+1 = {expected_w}")
print(f"   总计：{expected_h} × {expected_w} = {expected_total} 个 patches")
print()

# 5. 可视化每个 patch 的空间位置
print(f"5. Patch 空间位置可视化:")
print(f"   输入特征图空间结构 (4 行×8 列):")
print(f"   " + "-" * 34)
for i in range(latent_h):
    row = "   |"
    for j in range(latent_w):
        # 判断这个位置属于哪个 patch
        patch_h = i // window_size
        patch_w = j // window_size
        patch_idx = patch_h * expected_w + patch_w
        row += f" P{patch_idx:2d} |"
    print(row)
print(f"   " + "-" * 34)
print()

print(f"6. 结论:")
print(f"   ✅ 从 [1×4×8] 的特征图中提取 **{num_patches}** 个 2×2 的 patches")
print(f"   ✅ 每个 patch 包含 {window_size*window_size} 个元素")
print(f"   ✅ 展平后形状：[{batch_size*num_patches}, {window_size*window_size}]")
print(f"   ✅ 这些 patches 被送入 4 量子比特量子电路处理")
print()
print("=" * 60)
