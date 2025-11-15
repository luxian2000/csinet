import pennylane as qml
import numpy as np
import torch
from torch import nn
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
    ''' 构造经典压缩神经网络 '''
    def __init__(self, channels, img_height, com_height):
        super().__init__()
        self.img_dim = channels * img_height**2
        self.com_dim = channels * com_height**2
        self.conv = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dense_encode = nn.Linear(in_features=self.img_dim, out_features=self.com_dim)
        self.bn1d = nn.BatchNorm1d(num_features=channels)

    def forward(self, x):
        ''' 定义经典压缩层 '''
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = x.reshape((x.shape[0], -1))  # 展平
        x = self.dense_encode(x)
        x = 2 * self.bn1d(x)  # 缩放参数范围到[-2, 2]
        return x


def frqi_encoder(qubits, params, target_wire=0):
    ''' 改进的FRQI编码电路 '''
    # 对数据量子比特应用Hadamard门创建叠加态
    for i in range(1, qubits + 1):
        qml.Hadamard(wires=i)
    # 使用受控旋转进行编码
    for index in range(min(2**qubits, len(params))):
        binary_str = bin(index)[2:].zfill(qubits)
        bits = [int(bit) for bit in binary_str]
        bits.reverse()
        qml.ctrl(qml.RY, control=range(1, qubits + 1), control_values=bits)(params[index], wires=target_wire)


coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)


@qml.qnode(dev, interface='torch')
def quantum_circuit(com_params, img_params, quantum_params):
    ''' 量子电路 - 使用参数移位规则计算梯度 '''
    # 编码压缩参数
    frqi_encoder(COM_QUBITS, com_params, target_wire=0)
    # 强纠缠层
    qml.StronglyEntanglingLayers(weights=quantum_params, wires=range(ALL_QUBITS))
    # 编码原始图像参数
    frqi_encoder(IMG_QUBITS, img_params, target_wire=0)
    return qml.expval(hamiltonian)


class HybridNN(nn.Module):
    def __init__(self, classical_nn, com_qubits, img_qubits):
        super().__init__()
        self.classical_nn = classical_nn
        self.com_qubits = com_qubits
        self.img_qubits = img_qubits
        self.all_qubits = img_qubits + 1
        # 量子参数 - 使用numpy数组存储，不注册为nn.Parameter
        self.quantum_params = np.random.uniform(0, np.pi, size=(LAYERS, self.all_qubits, 3))

    def forward(self, x):
        ''' 前向传播 '''
        # 经典神经网络前向传播
        com_params = self.classical_nn(x)  # [batch_size, com_dim]
        # 将输入数据展平
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # [batch_size, img_dim]
        # 对每个样本运行量子电路
        energies = []
        for i in range(batch_size):
            energy = quantum_circuit(
                com_params[i].detach().numpy(),  # 经典参数，不需要量子梯度
                x_flat[i].detach().numpy(),      # 输入数据，不需要量子梯度
                self.quantum_params              # 量子参数，需要参数移位规则
            )
            energies.append(energy)
        return torch.stack(energies)
    def get_quantum_params(self):
        """获取量子参数"""
        return self.quantum_params.copy()
    def set_quantum_params(self, new_params):
        """设置量子参数"""
        self.quantum_params = new_params.copy()


class QuantumGradientOptimizer:
    """使用参数移位规则优化量子参数的优化器"""
    def __init__(self, quantum_circuit, learning_rate=0.1):
        self.quantum_circuit = quantum_circuit
        self.lr = learning_rate
    def compute_gradient(self, com_params, img_params, quantum_params, shift=np.pi/2):
        """使用参数移位规则计算梯度"""
        gradient = np.zeros_like(quantum_params)
        # 对每个参数计算梯度
        for layer in range(quantum_params.shape[0]):
            for qubit in range(quantum_params.shape[1]):
                for param_idx in range(quantum_params.shape[2]):
                    # 参数移位规则：f(θ+π/2) - f(θ-π/2)
                    params_plus = quantum_params.copy()
                    params_plus[layer, qubit, param_idx] += shift
                    
                    params_minus = quantum_params.copy()
                    params_minus[layer, qubit, param_idx] -= shift
                    
                    # 计算两个点的期望值
                    f_plus = self.quantum_circuit(com_params, img_params, params_plus)
                    f_minus = self.quantum_circuit(com_params, img_params, params_minus)
                    
                    # 计算梯度
                    gradient[layer, qubit, param_idx] = (f_plus - f_minus) / 2
        
        return gradient
    
    def update_params(self, com_params, img_params, quantum_params):
        """更新量子参数"""
        gradient = self.compute_gradient(com_params, img_params, quantum_params)
        new_params = quantum_params - self.lr * gradient
        return new_params


def train_hybrid_nn():
    ''' 训练混合神经网络 '''
    # 初始化模型
    classical_nn = ClassicalNN(channels=CHANNELS, img_height=IMG_HEIGHT, com_height=COM_HEIGHT)
    hybrid_nn = HybridNN(classical_nn=classical_nn, com_qubits=COM_QUBITS, img_qubits=IMG_QUBITS)
    
    # 初始化优化器
    quantum_optimizer = QuantumGradientOptimizer(quantum_circuit, learning_rate=0.1)
    classical_optimizer = torch.optim.Adam(classical_nn.parameters(), lr=0.001)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 数据准备
    x_train_tensor = torch.tensor(x_train)
    x_val_tensor = torch.tensor(x_val)
    x_test_tensor = torch.tensor(x_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, torch.zeros(len(x_train_tensor)))
    train_loader = DataLoader(train_dataset, batch_size=BATCHES, shuffle=True)
    
    # 训练参数
    num_epochs = 50
    train_losses = []
    val_losses = []
    quantum_param_history = []
    
    print("开始训练混合量子-经典神经网络...")
    print("📊 训练策略:")
    print("  - 经典参数: 使用反向传播(BP)")
    print("  - 量子参数: 使用参数移位规则(Parameter-shift Rule)")
    print(f"训练样本数: {len(x_train)}")
    print(f"批次大小: {BATCHES}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        classical_nn.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 重置梯度
            classical_optimizer.zero_grad()
            
            # 前向传播 - 经典部分
            com_params = classical_nn(data)  # 经典压缩
            
            # 获取当前量子参数
            current_quantum_params = hybrid_nn.get_quantum_params()
            
            batch_size = data.shape[0]
            x_flat = data.reshape(batch_size, -1)
            
            # 对批次中的每个样本单独优化量子参数
            batch_quantum_gradients = []
            batch_energies = []
            
            for i in range(batch_size):
                # 使用参数移位规则更新量子参数
                new_quantum_params = quantum_optimizer.update_params(
                    com_params[i].detach().numpy(),
                    x_flat[i].detach().numpy(),
                    current_quantum_params
                )
                
                # 设置新的量子参数
                hybrid_nn.set_quantum_params(new_quantum_params)
                
                # 计算当前样本的能量（损失）
                energy = quantum_circuit(
                    com_params[i].detach().numpy(),
                    x_flat[i].detach().numpy(),
                    new_quantum_params
                )
                batch_energies.append(energy)
            
            # 计算经典部分的损失
            energies_tensor = torch.tensor(batch_energies, dtype=torch.float32)
            loss = criterion(energies_tensor, torch.zeros_like(energies_tensor))
            
            # 经典部分的反向传播
            loss.backward()
            classical_optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch+1:03d} | Batch {batch_idx:03d} | Loss: {loss.item():.6f}')
        
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        classical_nn.eval()
        with torch.no_grad():
            val_energies = []
            for i in range(len(x_val_tensor)):
                com_params_val = classical_nn(x_val_tensor[i:i+1])
                x_flat_val = x_val_tensor[i:i+1].reshape(1, -1)
                
                energy_val = quantum_circuit(
                    com_params_val[0].detach().numpy(),
                    x_flat_val[0].detach().numpy(),
                    hybrid_nn.get_quantum_params()
                )
                val_energies.append(energy_val)
            
            val_energies_tensor = torch.tensor(val_energies, dtype=torch.float32)
            val_loss = criterion(val_energies_tensor, torch.zeros_like(val_energies_tensor)).item()
            val_losses.append(val_loss)
        
        # 保存量子参数历史
        quantum_param_history.append(hybrid_nn.get_quantum_params().copy())
        
        print(f'Epoch {epoch+1:03d}/{num_epochs:03d} | '
              f'训练损失: {avg_train_loss:.6f} | '
              f'验证损失: {val_loss:.6f}')
        
        # 早停检查
        if epoch > 10 and val_loss > np.mean(val_losses[-5:]):
            print("验证损失上升，考虑早停...")
            if epoch > 20:
                break
    
    training_time = time.time() - start_time
    print(f'训练完成! 总时间: {training_time:.2f}秒')
    
    # 保存模型
    save_trained_model(classical_nn, hybrid_nn, quantum_param_history)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, quantum_param_history)
    
    # 测试模型
    test_model(classical_nn, hybrid_nn, x_test_tensor)
    
    return classical_nn, hybrid_nn, train_losses, val_losses


def save_trained_model(classical_nn, hybrid_nn, quantum_param_history):
    """保存训练好的模型"""
    # 保存经典神经网络
    torch.save(classical_nn.state_dict(), 'classical_nn.pth')
    
    # 保存量子参数
    np.save('quantum_params.npy', hybrid_nn.get_quantum_params())
    np.save('quantum_param_history.npy', np.array(quantum_param_history))
    
    print("模型已保存!")


def plot_training_curves(train_losses, val_losses, quantum_param_history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失', alpha=0.7)
    ax1.plot(val_losses, label='验证损失', alpha=0.7)
    ax1.set_xlabel('训练周期')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 量子参数变化
    quantum_params_flat = [params.flatten() for params in quantum_param_history]
    ax2.plot(quantum_params_flat, alpha=0.5)
    ax2.set_xlabel('训练周期')
    ax2.set_ylabel('量子参数值')
    ax2.set_title('量子参数变化')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_model(classical_nn, hybrid_nn, test_data):
    """测试模型"""
    classical_nn.eval()
    
    with torch.no_grad():
        test_energies = []
        for i in range(min(10, len(test_data))):  # 测试前10个样本
            com_params_test = classical_nn(test_data[i:i+1])
            x_flat_test = test_data[i:i+1].reshape(1, -1)
            
            energy_test = quantum_circuit(
                com_params_test[0].detach().numpy(),
                x_flat_test[0].detach().numpy(),
                hybrid_nn.get_quantum_params()
            )
            test_energies.append(energy_test)
        
        print(f"\n测试结果:")
        print(f"能量范围: [{min(test_energies):.3f}, {max(test_energies):.3f}]")
        print(f"能量均值: {np.mean(test_energies):.3f} ± {np.std(test_energies):.3f}")


if __name__ == "__main__":
    # 数据加载（保持原有代码）
    if ENVIR == 'indoor':
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainin.mat')
        x_train = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalin.mat')
        x_val = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestin.mat')
        x_test = mat['HT']
    else:
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainout.mat')
        x_train = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalout.mat')
        x_val = mat['HT']
        mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestout.mat')
        x_test = mat['HT']

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (len(x_train), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
    x_val = np.reshape(x_val, (len(x_val), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
    x_test = np.reshape(x_test, (len(x_test), CHANNELS, IMG_HEIGHT, IMG_WIDTH))

    print('数据加载完成，开始训练...')
    
    # 开始训练
    classical_nn_trained, hybrid_nn_trained, train_losses, val_losses = train_hybrid_nn()
