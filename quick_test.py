#!/usr/bin/env python
"""
简化版本：只用部分数据进行快速测试
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset

# 配置
img_channels = 2
img_height = 32
img_width = 32
img_total = img_height * img_width * img_channels
batch_size = 8
encoded_dim = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 量子补偿模块 ====================
class QuantumCompensationBlock(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, window_size=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def quantum_circuit(inputs, weights_crz, weights_ry):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[i], wires=i)
            
            for layer in range(n_layers):
                for i in range(n_qubits - 1):
                    qml.CRZ(weights_crz[layer, i], wires=[i, i+1])
                qml.CRZ(weights_crz[layer, n_qubits-1], wires=[n_qubits-1, 0])
                for i in range(n_qubits):
                    qml.RY(weights_ry[layer, i], wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = quantum_circuit

    def forward(self, x):
        batch, ch, h, w = x.shape
        
        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        patches = unfold(x)
        num_patches = patches.shape[-1]
        
        patches = patches.permute(0, 2, 1).reshape(batch, num_patches, ch, 4)
        
        total_samples = batch * num_patches * ch
        all_inputs = patches.reshape(total_samples, 4)
        all_inputs = torch.tanh(all_inputs) * np.pi
        
        all_outputs = []
        for i in range(total_samples):
            input_data = all_inputs[i]
            q_out = self.circuit(input_data, self.weights_crz, self.weights_ry)
            all_outputs.append(torch.stack(q_out))
        
        all_outputs = torch.stack(all_outputs, dim=0)
        all_outputs = all_outputs.reshape(batch, num_patches, ch, self.n_qubits)
        all_outputs = all_outputs.permute(0, 2, 3, 1).reshape(batch, ch*self.n_qubits, num_patches)
        
        fold = nn.Fold(output_size=(h, w), kernel_size=self.window_size, stride=self.window_size)
        output = fold(all_outputs)
        
        # 转换为 float32 以匹配 Conv2d 权重类型
        output = output.float()
        
        return output


# ==================== 编码器 ====================
class CsiNetEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.conv1 = nn.Conv2d(img_channels, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.lr1 = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(img_total, encoded_dim)
        
    def forward(self, x):
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc_encode(x)
        return x


# ==================== 解码器 ====================
class QuantumCompensatedDecoder(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.encoded_dim = encoded_dim
        self.img_total = img_total
        
        self.fc_decode = nn.Linear(encoded_dim, img_total)
        self.downsample = nn.AvgPool2d(2)
        self.quantum_comp = QuantumCompensationBlock(n_qubits=4, n_layers=2, window_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.main_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.main_bn = nn.BatchNorm2d(img_channels)
        
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(img_channels) for _ in range(2)
        ])
        
        self.output_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, s):
        batch_size = s.shape[0]
        
        x = self.fc_decode(s)
        x = x.reshape(batch_size, img_channels, img_height, img_width)
        
        main_out = self.main_conv(x)
        main_out = self.main_bn(main_out)
        main_out = F.leaky_relu(main_out)
        
        comp = self.downsample(x)
        comp = self.quantum_comp(comp)
        # 确保类型一致（量子输出是 float64，需要转为 float32）
        comp = comp.float()
        comp = self.upsample(comp)
        
        fused = (1 - self.alpha) * main_out + self.alpha * comp
        
        residual = fused
        for block in self.residual_blocks:
            out = block(residual)
            residual = residual + out
            residual = F.leaky_relu(residual)
        
        out = self.output_conv(residual)
        out = self.sigmoid(out)
        
        return out


# ==================== 完整模型 ====================
class CsiNetQuantumCompensated(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.encoder = CsiNetEncoder(encoded_dim)
        self.decoder = QuantumCompensatedDecoder(encoded_dim, alpha)
        
    def forward(self, x):
        s = self.encoder(x)
        x_hat = self.decoder(s)
        return x_hat


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    data_path = '/Users/luxian/DataSpace/csinet/data'
    
    # 加载测试数据（仅用小部分）
    mat = sio.loadmat(f'{data_path}/DATA_Htestin.mat')
    x_test = mat['HT'].astype(np.float32)[:100]  # 仅用前100个
    
    batch_size_test = x_test.shape[0]
    x_test = x_test.reshape(batch_size_test, img_channels, img_height, img_width)
    
    x_test = torch.FloatTensor(x_test)
    test_dataset = TensorDataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Loaded test data: {x_test.shape}")
    
    # 创建模型
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    
    model = CsiNetQuantumCompensated(encoded_dim=encoded_dim, alpha=0.25)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试
    print("\n" + "=" * 60)
    print("Running forward pass...")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)
            print(f"Batch {batch_idx+1}: Input shape {data.shape}")
            
            output = model(data)
            print(f"  Output shape: {output.shape}")
            
            # Compute MSE
            loss = ((output - data) ** 2).mean()
            print(f"  MSE Loss: {loss.item():.6f}")
            
            if batch_idx >= 2:  # Only test first 3 batches
                break
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
