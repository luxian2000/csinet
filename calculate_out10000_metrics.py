"""
计算 out_10000 模型的 NMSE 和 Rho 指标
"""
import torch
import numpy as np
import scipy.io as sio
from pathlib import Path
import sys

# 导入模型定义
sys.path.insert(0, '/home/luxian/GitSpace/csinet/my_model')
from compensation_gpu import CsiNetQuantumCompensated, calculate_nmse_rho, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

# 配置
DATA_PATH = Path('/home/luxian/DataSpace/csinet/data')
MODEL_PATH = Path('/home/luxian/GitSpace/csinet/my_model/out_10000/best_model_quantum_gpu_outdoor_dim32_out_10000.pth')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("out_10000 模型 NMSE 和 Rho 计算")
print("=" * 60)

# 1. 检查频域数据文件
print("\n1. 检查频域数据文件...")
freq_file = DATA_PATH / 'DATA_HtestFout_all.mat'
if freq_file.exists():
    print(f"   ✓ 找到文件：{freq_file.name}")
    print(f"   文件大小：{freq_file.stat().st_size / (1024**3):.2f} GB")
else:
    print(f"   ✗ 未找到文件：{freq_file}")
    sys.exit(1)

# 2. 加载测试数据
print("\n2. 加载 outdoor 测试数据...")
test_file = DATA_PATH / 'DATA_Htestout.mat'
x_test = sio.loadmat(test_file)["HT"].astype(np.float32)
print(f"   ✓ HT shape: {x_test.shape}")

# 预处理
batch_size = x_test.shape[0]
x_test = x_test.reshape(batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
print(f"   ✓ 重塑后 shape: {x_test.shape}")

# 3. 加载频域数据
print("\n3. 加载频域数据...")
x_test_freq = sio.loadmat(freq_file)["HF_all"].astype(np.complex128)
x_test_freq = x_test_freq.reshape(-1, IMG_HEIGHT, 125)
print(f"   ✓ HF_all shape: {x_test_freq.shape}")

# 4. 加载模型
print("\n4. 加载模型...")
model = CsiNetQuantumCompensated(encoded_dim=32, alpha=0.25)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"   ✓ 模型加载成功：{MODEL_PATH.name}")
print(f"   设备：{DEVICE}")

# 5. 推理
print("\n5. 执行推理...")
x_test_tensor = torch.FloatTensor(x_test).to(DEVICE)
with torch.no_grad():
    x_hat = model(x_test_tensor).cpu().numpy()
print(f"   ✓ 预测完成，输出 shape: {x_hat.shape}")

# 6. 计算指标
print("\n6. 计算 NMSE 和 Rho...")
nmse, rho = calculate_nmse_rho(x_test, x_hat, x_test_freq)
print(f"   ✓ NMSE: {nmse:.2f} dB")
print(f"   ✓ Rho:  {rho:.4f}")

print("\n" + "=" * 60)
print("计算完成！")
print("=" * 60)
