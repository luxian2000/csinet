#!/usr/bin/env python
"""
快速测试 compensation_1_2.py 的核心模块
"""
import sys
sys.path.insert(0, '/Users/luxian/GitSpace/csinet/my_model')

import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset

# 导入关键模块
from compensation_1_2 import (
    QuantumCompensationBlock,
    CsiNetEncoder,
    QuantumCompensatedDecoder,
    CsiNetQuantumCompensated
)

# 配置
img_channels = 2
img_height = 32
img_width = 32
img_total = img_height * img_width * img_channels
batch_size = 4
encoded_dim = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print("=" * 60)
print("Step 1: Testing QuantumCompensationBlock")
print("=" * 60)

try:
    qc_block = QuantumCompensationBlock(n_qubits=4, n_layers=2, window_size=2)
    x = torch.randn(2, 2, 16, 16)
    y = qc_block(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Output dtype: {y.dtype}")
    assert y.shape == (2, 2, 16, 16), "Output shape mismatch!"
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 2: Testing CsiNetQuantumCompensated model")
print("=" * 60)

try:
    model = CsiNetQuantumCompensated(encoded_dim=encoded_dim, alpha=0.25)
    model = model.to(device)
    print(f"✓ Model created")
    
    x = torch.randn(batch_size, 2, 32, 32).to(device)
    y = model(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Output dtype: {y.dtype}")
    assert y.shape == x.shape, "Output shape should match input!"
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 3: Loading and preprocessing real data")
print("=" * 60)

try:
    data_path = '/Users/luxian/DataSpace/csinet/data'
    
    # Load test data
    mat = sio.loadmat(f'{data_path}/DATA_Htestin.mat')
    x_test = mat['HT'].astype(np.float32)[:64]  # Use first 64 samples
    
    batch_size_test = x_test.shape[0]
    x_test = x_test.reshape(batch_size_test, img_channels, img_height, img_width)
    x_test = torch.FloatTensor(x_test)
    
    print(f"✓ Loaded test data: {x_test.shape}")
    
    test_dataset = TensorDataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test inference
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            mse = ((output - data) ** 2).mean()
            print(f"  Batch {batch_idx+1}: Input {data.shape} -> Output {output.shape}, MSE={mse.item():.6f}")
            
            if batch_idx >= 2:
                break
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
