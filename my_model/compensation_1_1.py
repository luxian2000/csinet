import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset
import math
import time
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 配置参数 ====================
envir = 'indoor'  # 'indoor' or 'outdoor'
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

# 压缩率设置
compression_rates = {
    1/4: 512,    # 512/2048 = 1/4
    1/16: 128,   # 128/2048 = 1/16
    1/32: 64,    # 64/2048 = 1/32
    1/64: 32     # 32/2048 = 1/64
}
encoded_dim = 32 

# 训练参数
initial_lr = 5e-3
batch_size = 4  # Reduced for testing
epochs = 1  # Just test one epoch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 量子补偿模块 ====================
class QuantumCompensationBlock(nn.Module):
    """
    使用4量子比特，2×2窗口滑动处理
    """
    def __init__(self, n_qubits=4, n_layers=2, window_size=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        
        # 定义量子设备
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # 可训练参数：每层的参数
        # weights_crz: [n_layers, n_qubits]  CRZ门的旋转角度
        # weights_ry: [n_layers, n_qubits]   RY门的旋转角度
        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        # 定义量子电路
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def quantum_circuit(inputs, weights_crz, weights_ry):
            """
            inputs: vector of length 4 (values for 2×2 window)
            weights_crz: CRZ gate parameters
            weights_ry: RY gate parameters
            """
            # Ensure inputs is 1D
            if inputs.dim() == 2:
                inputs = inputs.squeeze(0)
            
            # Embedding layer: H gate + RY(input data)
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[i], wires=i)
            
            # Entanglement layer: multiple CRZ + RY
            for layer in range(n_layers):
                # CRZ entanglement (ring connection)
                for i in range(n_qubits - 1):
                    qml.CRZ(weights_crz[layer, i], wires=[i, i+1])
                # Last one connects to the first
                qml.CRZ(weights_crz[layer, n_qubits-1], wires=[n_qubits-1, 0])
                
                # RY phase adjustment
                for i in range(n_qubits):
                    qml.RY(weights_ry[layer, i], wires=i)
            
            # Measurement layer: measure PauliZ expectation value of each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = quantum_circuit
        
    def forward(self, x):
        """
        x: Input feature map [batch, channels, height, width]
        Assumes input is already downsampled feature map
        """
        batch, ch, h, w = x.shape
        
        # Use unfold to extract 2×2 windows
        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        patches = unfold(x)  # [batch, ch*4, num_patches]
        num_patches = patches.shape[-1]
        
        # Reshape to [batch * num_patches, ch, 4] for processing
        patches = patches.permute(0, 2, 1).reshape(batch * num_patches, ch, 4)
        
        # Process all patches through quantum circuit
        all_outputs = []
        
        for idx in range(patches.shape[0]):
            patch_idx = patches[idx]  # [ch, 4]
            
            # Process each channel
            channel_outputs = []
            for c in range(ch):
                input_data = torch.tanh(patch_idx[c]) * np.pi  # [4]
                
                # Apply quantum circuit
                q_out_list = self.circuit(input_data, self.weights_crz, self.weights_ry)
                q_out = torch.stack(q_out_list)  # [n_qubits]
                channel_outputs.append(q_out)
            
            # Stack channels: [ch, n_qubits]
            patch_output = torch.stack(channel_outputs, dim=0)
            all_outputs.append(patch_output)
        
        # Stack all patch outputs: [batch*num_patches, ch, n_qubits]
        compensated = torch.stack(all_outputs, dim=0)
        
        # Reshape back to [batch, num_patches, ch, n_qubits]
        compensated = compensated.reshape(batch, num_patches, ch, self.n_qubits)
        
        # Permute and reshape for fold: [batch, ch*n_qubits, num_patches]
        compensated = compensated.permute(0, 2, 3, 1).reshape(batch, ch*self.n_qubits, num_patches)
        
        # Recombine to feature map using fold
        fold = nn.Fold(output_size=(h, w), kernel_size=self.window_size, stride=self.window_size)
        output = fold(compensated)
        
        # Ensure output is float type
        output = output.float()
        
        return output


# ==================== CsiNet编码器 ====================
class CsiNetEncoder(nn.Module):
    """
    论文1中的CsiNet编码器部分
    将32×32×2的信道矩阵压缩为低维码字
    """
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoded_dim = encoded_dim
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(img_channels, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.lr1 = nn.LeakyReLU()
        
        # 压缩到码字
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(img_total, encoded_dim)
        
    def forward(self, x):
        """
        x: [batch, 2, 32, 32] 信道矩阵（实部+虚部）
        return: [batch, encoded_dim] 压缩后的码字
        """
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc_encode(x)
        return x


# ==================== 量子补偿解码器 ====================
class QuantumCompensatedDecoder(nn.Module):
    """
    量子补偿增强的解码器
    两条路径：
    1. 主路径：直接上采样（从码字生成特征图）
    2. 补偿路径：量子补偿模块处理后上采样
    然后加权融合，再经过残差网络
    """
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.alpha = alpha  # 量子补偿因子
        self.encoded_dim = encoded_dim
        self.img_total = img_total
        
        # 从码字到初始特征图（功能上的上采样）
        self.fc_decode = nn.Linear(encoded_dim, img_total)
        
        # 补偿路径：下采样 + 量子补偿 + 上采样
        self.downsample = nn.AvgPool2d(2)  # 32×32 → 16×16
        self.quantum_comp = QuantumCompensationBlock(
            n_qubits=4, n_layers=2, window_size=2
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.comp_adjust = nn.Conv2d(img_channels*4, img_channels, kernel_size=1)  # 量子补偿输出4通道/像素
        
        # 主路径：简单的特征提取
        self.main_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.main_bn = nn.BatchNorm2d(img_channels)
        
        # 融合后的残差网络（论文2中的10层残差网络简化版）
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(img_channels) for _ in range(5)  # 5个残差块，共10层卷积
        ])
        
        # 输出层
        self.output_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_residual_block(self, channels):
        """创建一个残差块（2层卷积）"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, s):
        """
        s: [batch, encoded_dim] compressed codeword
        return: [batch, 2, 32, 32] reconstructed channel matrix
        """
        batch_size = s.shape[0]
        
        # Generate initial feature map from codeword (functional upsampling)
        x = self.fc_decode(s)  # [batch, 2048]
        x = x.reshape(batch_size, img_channels, img_height, img_width)  # [batch, 2, 32, 32]
        
        # ===== Main Path =====
        main_out = self.main_conv(x)
        main_out = self.main_bn(main_out)
        main_out = F.leaky_relu(main_out)
        
        # ===== Compensation Path =====
        # Downsample to 16×16
        comp = self.downsample(x)  # [batch, 2, 16, 16]
        
        # Apply quantum compensation
        comp = self.quantum_comp(comp)  # [batch, 8, 16, 16] (2 channels × 4 qubits)
        
        # Adjust channel number from 8 to 2
        comp = self.comp_adjust(comp)  # [batch, 2, 16, 16]
        
        # Upsample back to 32×32
        comp = self.upsample(comp)  # [batch, 2, 32, 32]
        
        # ===== 融合 =====
        fused = (1 - self.alpha) * main_out + self.alpha * comp
        
        # ===== 残差网络 =====
        residual = fused
        for block in self.residual_blocks:
            out = block(residual)
            residual = residual + out  # 残差连接
            residual = F.leaky_relu(residual)
        
        # 输出层
        out = self.output_conv(residual)
        out = self.sigmoid(out)
        
        return out


# ==================== 完整的CsiNet量子补偿模型 ====================
class CsiNetQuantumCompensated(nn.Module):
    """
    完整的CsiNet + 量子补偿模型
    使用论文1的编码器，论文2的量子补偿解码器
    """
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.encoder = CsiNetEncoder(encoded_dim)
        self.decoder = QuantumCompensatedDecoder(encoded_dim, alpha)
        
    def forward(self, x):
        """
        x: [batch, 2, 32, 32] 输入信道矩阵
        return: [batch, 2, 32, 32] 重建的信道矩阵
        """
        s = self.encoder(x)
        x_hat = self.decoder(s)
        return x_hat
    
    def get_codeword(self, x):
        """获取压缩后的码字（用于传输）"""
        return self.encoder(x)


# ==================== 数据加载 ====================
def load_data(envir='indoor', data_path='/Users/luxian/DataSpace/csinet/data'):
    """
    加载论文 1 中的数据集
    """
    print(f"Loading {envir} data...")
    
    if envir == 'indoor':
        # 加载训练数据
        mat = sio.loadmat(f'{data_path}/DATA_Htrainin.mat')
        x_train = mat['HT'].astype(np.float32)
        
        # 加载验证数据
        mat = sio.loadmat(f'{data_path}/DATA_Hvalin.mat')
        x_val = mat['HT'].astype(np.float32)
        
        # 加载测试数据
        mat = sio.loadmat(f'{data_path}/DATA_Htestin.mat')
        x_test = mat['HT'].astype(np.float32)
        
        # 加载原始空频域数据（用于评估）
        mat = sio.loadmat(f'{data_path}/DATA_HtestFin_all.mat')
        X_test_freq = mat['HF_all'].astype(np.complex128)
        
    else:  # outdoor
        mat = sio.loadmat(f'{data_path}/DATA_Htrainout.mat')
        x_train = mat['HT'].astype(np.float32)
        
        mat = sio.loadmat(f'{data_path}/DATA_Hvalout.mat')
        x_val = mat['HT'].astype(np.float32)
        
        mat = sio.loadmat(f'{data_path}/DATA_Htestout.mat')
        x_test = mat['HT'].astype(np.float32)
        
        mat = sio.loadmat(f'{data_path}/DATA_HtestFout_all.mat')
        X_test_freq = mat['HF_all'].astype(np.complex128)
    
    # 数据预处理：调整为 [batch, channels, height, width] 格式
    def preprocess_data(data):
        # 原始数据形状: [batch, 2048] 其中 2048 = 32*32*2
        # 需要重塑为 [batch, 2, 32, 32]
        batch_size = data.shape[0]
        data = data.reshape(batch_size, img_channels, img_height, img_width)
        return data
    
    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)
    
    print(f"Train data shape: {x_train.shape}")
    print(f"Val data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test freq data shape: {X_test_freq.shape}")
    
    return x_train, x_val, x_test, X_test_freq


# ==================== 评估函数 ====================
def calculate_nmse_rho(x_test, x_hat, X_test_freq, img_height=32, img_width=32):
    """
    计算NMSE和余弦相似度（与论文1一致）
    """
    batch_size = x_test.shape[0]
    
    # 将重建结果转换回复数形式
    x_test_real = x_test[:, 0, :, :].reshape(batch_size, -1)
    x_test_imag = x_test[:, 1, :, :].reshape(batch_size, -1)
    x_test_C = (x_test_real - 0.5) + 1j * (x_test_imag - 0.5)
    
    x_hat_real = x_hat[:, 0, :, :].reshape(batch_size, -1)
    x_hat_imag = x_hat[:, 1, :, :].reshape(batch_size, -1)
    x_hat_C = (x_hat_real - 0.5) + 1j * (x_hat_imag - 0.5)
    
    # 重建空频域数据
    x_hat_F = x_hat_C.reshape(batch_size, img_height, img_width)
    
    # 补零并进行FFT
    X_hat = np.fft.fft(
        np.concatenate(
            (x_hat_F, np.zeros((batch_size, img_height, 257 - img_width))), 
            axis=2
        ), 
        axis=2
    )
    X_hat = X_hat[:, :, 0:125]
    
    # 计算余弦相似度
    n1 = np.sqrt(np.sum(np.conj(X_test_freq) * X_test_freq, axis=1))
    n2 = np.sqrt(np.sum(np.conj(X_hat) * X_hat, axis=1))
    aa = np.abs(np.sum(np.conj(X_test_freq) * X_hat, axis=1))
    rho = np.mean(aa / (n1 * n2 + 1e-10), axis=1)
    
    # 计算NMSE
    power = np.sum(np.abs(x_test_C)**2, axis=1)
    mse = np.sum(np.abs(x_test_C - x_hat_C)**2, axis=1)
    nmse = 10 * np.log10(np.mean(mse / (power + 1e-10)))
    
    return nmse, np.mean(rho)


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=100, lr=5e-3, device=torch.device('cpu')):
    """
    训练模型
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses


# ==================== 主程序 ====================
def main():
    # Load data
    x_train, x_val, x_test, X_test_freq = load_data(envir, data_path='/Users/luxian/DataSpace/csinet/data')
    
    # 转换为PyTorch张量
    x_train = torch.FloatTensor(x_train)
    x_val = torch.FloatTensor(x_val)
    x_test = torch.FloatTensor(x_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)
    test_dataset = TensorDataset(x_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    alpha = 0.25  # 量子补偿因子
    model = CsiNetQuantumCompensated(encoded_dim=encoded_dim, alpha=alpha)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 训练模型
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=epochs, lr=initial_lr, device=device
    )
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # 测试
    print("\nTesting...")
    tStart = time.time()
    
    all_outputs = []
    with torch.no_grad():
        for data in test_loader:
            data = data[0].to(device)
            output = model(data)
            all_outputs.append(output.cpu().numpy())
    
    x_hat = np.concatenate(all_outputs, axis=0)
    tEnd = time.time()
    
    print(f"Inference time per sample: {(tEnd - tStart)/x_test.shape[0]:.6f} sec")
    
    # 计算NMSE和余弦相似度
    x_test_np = x_test.numpy()
    nmse, rho = calculate_nmse_rho(x_test_np, x_hat, X_test_freq)
    
    print(f"\nResults for {envir} environment:")
    print(f"Compression ratio: {encoded_dim/img_total:.4f} (dim={encoded_dim})")
    print(f"Quantum compensation factor alpha: {alpha}")
    print(f"NMSE: {nmse:.2f} dB")
    print(f"Cosine similarity: {rho:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), f'csinet_quantum_{envir}_dim{encoded_dim}.pth')
    print("Model saved.")
    
    # 保存损失曲线
    np.savetxt(f'train_loss_{envir}_dim{encoded_dim}.csv', train_losses, delimiter=',')
    np.savetxt(f'val_loss_{envir}_dim{encoded_dim}.csv', val_losses, delimiter=',')


# ==================== 不同压缩率的对比实验 ====================
def run_comparison_experiments():
    """
    运行不同压缩率下的对比实验
    """
    results = {}
    
    for cr_name, dim in compression_rates.items():
        print(f"\n{'='*50}")
        print(f"Testing compression rate: {cr_name:.4f} (dim={dim})")
        print('='*50)
        
        # 更新全局变量
        global encoded_dim
        encoded_dim = dim
        
        # 重新运行主程序
        main()
        
        # 这里可以收集结果进行对比
        # 由于main()会打印结果，我们可以在实际运行时收集
    
    return results


if __name__ == "__main__":
    # 运行单个实验
    main()
    
    # 如果要运行所有压缩率的对比实验，取消下面的注释
    # run_comparison_experiments()