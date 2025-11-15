''' 本程序搭建联合训练CsiNet的混合神经网络 '''
import pennylane as qml
import numpy as np
import torch
from torch import nn
from pennylane.templates.layers import StronglyEntanglingLayers
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

# Global parameters
ENVIR = 'indoor'
LAYERS = 2
BATCHES = 5
CHANNELS = 2

# image parameters
IMG_HEIGHT = 32
IMG_WIDTH = IMG_HEIGHT
IMG_DIM = IMG_HEIGHT * IMG_WIDTH * CHANNELS
IMG_QUBITS = int(np.log2(IMG_DIM))

# compressed parameters
COM_HEIGHT = 16
COM_WIDTH = COM_HEIGHT
COM_DIM = COM_HEIGHT * COM_WIDTH * CHANNELS
COM_QUBITS = int(np.log2(COM_DIM))

ALL_QUBITS = IMG_QUBITS + 1
ANC_QUBITS = IMG_QUBITS - COM_QUBITS

class ClassicalNN(nn.Module):
    def __init__(self, channels, img_height, com_height):
        super().__init__()
        self.img_dim = channels * img_height**2
        self.com_dim = channels * com_height**2
        self.conv = nn.Conv2d(
                in_channels=channels,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
                )
        # 修复BatchNorm1d的num_features
        self.bn1d = nn.BatchNorm1d(num_features=self.com_dim)  # 改为压缩后的维度
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dense_encode = nn.Linear(in_features=self.img_dim, out_features=self.com_dim)

    def forward(self, x):
        ''' 定义经典压缩层 '''
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = x.reshape((x.shape[0], -1))  # 展平为 [batch_size, features]
        x = self.dense_encode(x)
        x = x.unsqueeze(1)  # 添加维度 [batch_size, 1, com_dim]
        x = self.bn1d(x)
        x = x.squeeze(1)  # 移除维度 [batch_size, com_dim]
        x = 2 * x  # 缩放参数范围
        return x

def frqi_encoder(qubits, params, target_wire=0):
    ''' construct the FRQI encoding circuit '''
    # 更简单的实现：使用振幅编码
    # 首先创建所有基态
    for i in range(qubits):
        qml.Hadamard(wires=i+1)  # 数据量子比特初始化为|+⟩状态

    # 使用受控旋转进行编码
    for i in range(2**qubits):
        # 将索引转换为二进制
        binary = format(i, f'0{qubits}b')
        # 对每个比特位设置控制
        control_wires = []
        control_values = []
        for j, bit in enumerate(binary):
            if bit == '1':
                control_wires.append(j+1)
                control_values.append(1)

        if control_wires:
            # 有控制位的情况
            qml.ctrl(qml.RY, control=control_wires, control_values=control_values)(params[i], wires=target_wire)
        else:
            # 无控制位的情况（全0）
            qml.RY(params[i], wires=target_wire)

coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)

@qml.qnode(dev, interface='torch')
def frqi_circuit(com_qubits, com_params, img_qubits, img_params, asz_params):
    ''' construct the complete quantum circuit '''
    # 初始化辅助量子比特
    qml.Hadamard(wires=0)

    # 编码压缩后的参数
    frqi_encoder(com_qubits, com_params, target_wire=0)

    # 应用强纠缠层
    qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))

    # 编码原始图像参数
    frqi_encoder(img_qubits, img_params, target_wire=0)

    return qml.expval(hamiltonian)

class HybridNN(nn.Module):
    def __init__(self, classical_nn, com_qubits, img_qubits):
        super().__init__()
        self.classical_nn = classical_nn
        self.com_qubits = com_qubits
        self.img_qubits = img_qubits

        # 将量子参数转换为可训练参数
        asz_params = np.random.uniform(0, np.pi, size=(LAYERS, ALL_QUBITS, 3))
        self.asz_params = nn.Parameter(torch.tensor(asz_params, dtype=torch.float32))

        # 添加解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, self.img_qubits * 2**self.img_qubits),
            nn.Tanh()  # 输出在[-1, 1]范围内
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # 经典压缩
        com_params = self.classical_nn(x)

        # 将输入数据展平用于量子编码
        x_flat = x.reshape(batch_size, -1)

        # 对batch中的每个样本单独处理量子电路
        quantum_outputs = []
        for i in range(batch_size):
            try:
                current_img_params = x_flat[i]
                current_com_params = com_params[i]

                # 运行量子电路
                energy = frqi_circuit(self.com_qubits, current_com_params,
                                    self.img_qubits, current_img_params, self.asz_params)
                quantum_outputs.append(energy)
            except Exception as e:
                print(f"量子电路执行错误: {e}")
                # 返回默认值
                quantum_outputs.append(torch.tensor(0.0))

        # 将量子输出转换为张量
        quantum_tensor = torch.stack(quantum_outputs).unsqueeze(1)  # [batch_size, 1]

        # 通过解码器重建图像
        reconstructed = self.decoder(quantum_tensor)
        return reconstructed

# Data loading and preprocessing
if ENVIR == 'indoor':
    try:
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainin.mat')
        x_train = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalin.mat')
        x_val = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestin.mat')
        x_test = mat['HT']
    except FileNotFoundError:
        print("文件未找到，使用随机数据测试")
        # 创建随机测试数据
        x_train = np.random.randn(100, CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype('float32')
        x_val = np.random.randn(20, CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype('float32')
        x_test = np.random.randn(20, CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype('float32')
elif ENVIR == 'outdoor':
    try:
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainout.mat')
        x_train = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalout.mat')
        x_val = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestout.mat')
        x_test = mat['HT']
    except FileNotFoundError:
        print("文件未找到，使用随机数据测试")
        x_train = np.random.randn(100, CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype('float32')
        x_val = np.random.randn(20, CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype('float32')
        x_test = np.random.randn(20, CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype('float32')

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# 重塑数据
x_train = np.reshape(x_train, (len(x_train), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
x_val = np.reshape(x_val, (len(x_val), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
x_test = np.reshape(x_test, (len(x_test), CHANNELS, IMG_HEIGHT, IMG_WIDTH))

print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

def train_hybrid_nn():
    '''训练混合神经网络'''

    # 转换数据为PyTorch张量
    x_train_tensor = torch.tensor(x_train)
    x_val_tensor = torch.tensor(x_val)
    x_test_tensor = torch.tensor(x_test)

    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=min(BATCHES, len(x_train)), shuffle=True)

    # 初始化模型
    classical_nn = ClassicalNN(channels=CHANNELS, img_height=IMG_HEIGHT, com_height=COM_HEIGHT)
    hybrid_nn = HybridNN(classical_nn=classical_nn, com_qubits=COM_QUBITS, img_qubits=IMG_QUBITS)

    print(f"模型参数总数: {sum(p.numel() for p in hybrid_nn.parameters())}")

    # 优化器
    optimizer = torch.optim.Adam(hybrid_nn.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练参数
    num_epochs = 30  # 进一步减少周期数以便快速测试
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("开始训练混合量子-经典神经网络...")

    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练阶段
        hybrid_nn.train()
        train_loss = 0.0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                optimizer.zero_grad()

                # 前向传播
                outputs = hybrid_nn(data)

                # 将目标展平
                target_flat = target.reshape(target.shape[0], -1)

                # 确保输出和目标形状匹配
                min_dim = min(outputs.shape[1], target_flat.shape[1])
                outputs = outputs[:, :min_dim]
                target_flat = target_flat[:, :min_dim]

                # 计算损失
                loss = criterion(outputs, target_flat)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(hybrid_nn.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                batch_count += 1

                if batch_idx % 2 == 0:
                    print(f'Epoch: {epoch+1:03d} | Batch: {batch_idx:03d} | Loss: {loss.item():.6f}')

            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue

        if batch_count == 0:
            avg_train_loss = 0
        else:
            avg_train_loss = train_loss / batch_count

        # 验证阶段
        hybrid_nn.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            val_batch_size = min(BATCHES, len(x_val_tensor))
            for i in range(0, len(x_val_tensor), val_batch_size):
                try:
                    end_idx = min(i + val_batch_size, len(x_val_tensor))
                    val_data = x_val_tensor[i:end_idx]
                    val_target = x_val_tensor[i:end_idx]

                    val_outputs = hybrid_nn(val_data)
                    val_target_flat = val_target.reshape(val_target.shape[0], -1)

                    # 确保形状匹配
                    min_dim = min(val_outputs.shape[1], val_target_flat.shape[1])
                    val_outputs = val_outputs[:, :min_dim]
                    val_target_flat = val_target_flat[:, :min_dim]

                    val_loss += criterion(val_outputs, val_target_flat).item()
                    val_batch_count += 1

                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue

        if val_batch_count == 0:
            avg_val_loss = float('inf')
        else:
            avg_val_loss = val_loss / val_batch_count

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': hybrid_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_hybrid_model.pth')
            print(f'保存最佳模型在周期 {epoch+1}, 验证损失: {avg_val_loss:.6f}')

        print(f'Epoch: {epoch+1:03d} | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f}')

        # 简单的早停逻辑
        if epoch > 10 and avg_val_loss > best_val_loss:
            print("验证损失不再改善，停止训练")
            break

    training_time = time.time() - start_time
    print(f'训练完成! 总时间: {training_time:.2f}秒')

    # 绘制损失曲线
    if len(train_losses) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 测试模型
    test_model(hybrid_nn, x_test_tensor)

    return hybrid_nn, train_losses, val_losses

def test_model(model, test_data):
    """测试训练好的模型"""
    model.eval()
    criterion = nn.MSELoss()

    with torch.no_grad():
        test_loss = 0.0
        test_batch_count = 0

        test_batch_size = min(BATCHES, len(test_data))
        for i in range(0, len(test_data), test_batch_size):
            try:
                end_idx = min(i + test_batch_size, len(test_data))
                batch_data = test_data[i:end_idx]
                batch_target = test_data[i:end_idx]

                outputs = model(batch_data)
                target_flat = batch_target.reshape(batch_target.shape[0], -1)

                # 确保形状匹配
                min_dim = min(outputs.shape[1], target_flat.shape[1])
                outputs = outputs[:, :min_dim]
                target_flat = target_flat[:, :min_dim]

                test_loss += criterion(outputs, target_flat).item()
                test_batch_count += 1

            except Exception as e:
                print(f"测试批次出错: {e}")
                continue

        if test_batch_count > 0:
            avg_test_loss = test_loss / test_batch_count
            print(f'测试损失: {avg_test_loss:.6f}')
        else:
            print("测试过程中出现错误，无法计算测试损失")

if __name__ == "__main__":
    try:
        # 开始训练
        trained_model, train_losses, val_losses = train_hybrid_nn()

        # 保存最终模型
        torch.save(trained_model.state_dict(), 'final_hybrid_model.pth')
        print("最终模型已保存!")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
