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
        x = self.bn1d(x) * 2
        return x


def frqi_encoder(qubits, params):
    ''' construct the FRQI encoding circuit '''
    for index in range(2**qubits):
        binary_str = bin(index)[2:].zfill(qubits)  # 补零确保长度一致
        bits = [int(bit) for bit in binary_str]
        bits.reverse()
        # 使用条件控制门
        control_wires = [i+1 for i, bit in enumerate(bits) if bit == 1]
        control_values = [1] * len(control_wires)
        if control_wires:
            qml.ctrl(qml.RY, control=control_wires, control_values=control_values)(params[index], wires=0)
        else:
            qml.RY(params[index], wires=0)


coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)


@qml.qnode(dev, interface='torch')
def frqi_circuit(com_qubits, com_params, img_qubits, img_params, asz_params):
    ''' construct the complete quantum circuit '''
    # 初始化辅助量子比特
    qml.Hadamard(wires=0)
    
    frqi_encoder(com_qubits, com_params)
    qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
    frqi_encoder(img_qubits, img_params)
    return qml.expval(hamiltonian)


class HybridNN(nn.Module):
    ''' 把上面定义的经典神经网络和量子神经网络组装成完整神经网络 '''
    def __init__(self, classical_nn, com_qubits, img_qubits):
        super().__init__()
        self.classical_nn = classical_nn
        self.com_qubits = com_qubits
        self.img_qubits = img_qubits
        self.all_qubits = img_qubits + 1
        
        # 将asz_params转换为可训练参数
        asz_params = np.random.uniform(0, np.pi, size=(LAYERS, self.all_qubits, 3))
        self.asz_params = nn.Parameter(torch.tensor(asz_params, dtype=torch.float32))

    def forward(self, x):
        # 经典神经网络处理
        com_params = self.classical_nn(x)
        
        # 将输入数据展平用于量子编码
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # 对batch中的每个样本单独处理
        energies = []
        for i in range(batch_size):
            current_com_params = com_params[i]
            current_img_params = x_flat[i]
            
            # 运行量子电路
            energy = frqi_circuit(
                self.com_qubits, 
                current_com_params, 
                self.img_qubits, 
                current_img_params, 
                self.asz_params
            )
            energies.append(energy)
        
        return torch.stack(energies)


# Data loading
if ENVIR == 'indoor':
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainin.mat')
    x_train = mat['HT']  # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalin.mat')
    x_val = mat['HT']  # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestin.mat')
    x_test = mat['HT']  # array

elif ENVIR == 'outdoor':
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainout.mat')
    x_train = mat['HT']  # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalout.mat')
    x_val = mat['HT']  # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestout.mat')
    x_test = mat['HT']  # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
print('x_train 的原始维度:', x_train.shape)

x_train = np.reshape(x_train, (len(x_train), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
x_val = np.reshape(x_val, (len(x_val), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
x_test = np.reshape(x_test, (len(x_test), CHANNELS, IMG_HEIGHT, IMG_WIDTH))
print('x_train 的塑形维度:', x_train.shape)


def train_hybrid_nn():
    ''' 训练混合神经网络 '''
    # 初始化模型
    classical_nn = ClassicalNN(channels=CHANNELS, img_height=IMG_HEIGHT, com_height=COM_HEIGHT)
    hybrid_nn = HybridNN(classical_nn=classical_nn, com_qubits=COM_QUBITS, img_qubits=IMG_QUBITS)

    # 检查可训练参数
    print("可训练参数:")
    for name, param in hybrid_nn.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    
    total_params = sum(p.numel() for p in hybrid_nn.parameters() if p.requires_grad)
    print(f"总可训练参数数: {total_params}")

    # 优化器 - 同时优化classical_nn和asz_params
    optimizer = torch.optim.Adam(hybrid_nn.parameters(), lr=0.001)
    
    # 损失函数 - 使用MSE损失，目标是使量子电路输出接近某个目标值
    criterion = nn.MSELoss()
    
    # 数据准备
    x_train_tensor = torch.tensor(x_train)
    x_val_tensor = torch.tensor(x_val)
    x_test_tensor = torch.tensor(x_test)
    
    # 创建目标值（这里假设我们希望量子电路输出接近0）
    # 您可以根据具体任务调整目标值
    target_value = 0.0
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, torch.full((len(x_train_tensor),), target_value))
    train_loader = DataLoader(train_dataset, batch_size=BATCHES, shuffle=True)
    
    # 训练参数
    num_epochs = 100
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("开始训练混合量子-经典神经网络...")
    print(f"训练样本数: {len(x_train)}")
    print(f"批次大小: {BATCHES}")
    print(f"目标值: {target_value}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        hybrid_nn.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = hybrid_nn(data)
            
            # 计算损失 - 使量子电路输出接近目标值
            loss = criterion(outputs, torch.full_like(outputs, target_value))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(hybrid_nn.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1:03d}/{num_epochs:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                      f'Loss: {loss.item():.6f}')
        
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        hybrid_nn.eval()
        with torch.no_grad():
            val_outputs = hybrid_nn(x_val_tensor)
            val_loss = criterion(val_outputs, torch.full_like(val_outputs, target_value)).item()
            val_losses.append(val_loss)
        
        # 学习率调度（可选）
        if epoch > 0 and epoch % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"学习率调整为: {optimizer.param_groups[0]['lr']}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': hybrid_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, 'best_hybrid_model.pth')
            print(f'✅ 保存最佳模型在周期 {epoch+1}, 验证损失: {val_loss:.6f}')
        
        print(f'Epoch {epoch+1:03d}/{num_epochs:03d} | '
              f'训练损失: {avg_train_loss:.6f} | '
              f'验证损失: {val_loss:.6f}')
        
        # 早停检查
        if epoch > 10 and val_loss > np.mean(val_losses[-5:]):
            print("⚠️  验证损失上升，考虑早停...")
            if epoch > 30:  # 至少训练30个周期
                break
    
    training_time = time.time() - start_time
    print(f'🎉 训练完成! 总时间: {training_time:.2f}秒')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', alpha=0.7)
    plt.plot(val_losses, label='验证损失', alpha=0.7)
    plt.xlabel('训练周期')
    plt.ylabel('损失值')
    plt.title('混合量子-经典神经网络训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 测试模型
    test_model(hybrid_nn, x_test_tensor, target_value)
    
    return hybrid_nn, train_losses, val_losses


def test_model(model, test_data, target_value):
    """测试训练好的模型"""
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        test_outputs = model(test_data)
        test_loss = criterion(test_outputs, torch.full_like(test_outputs, target_value)).item()
        
        print(f'\n📊 测试结果:')
        print(f'测试损失: {test_loss:.6f}')
        print(f'输出范围: [{test_outputs.min().item():.3f}, {test_outputs.max().item():.3f}]')
        print(f'输出均值: {test_outputs.mean().item():.3f} ± {test_outputs.std().item():.3f}')
        
        # 显示前几个样本的输出
        print(f'前5个样本输出: {test_outputs[:5].squeeze().tolist()}')


def load_and_evaluate(model_path, x_test):
    """加载保存的模型并进行评估"""
    classical_nn = ClassicalNN(channels=CHANNELS, img_height=IMG_HEIGHT, com_height=COM_HEIGHT)
    model = HybridNN(classical_nn=classical_nn, com_qubits=COM_QUBITS, img_qubits=IMG_QUBITS)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 加载模型完成，训练周期: {checkpoint['epoch'] + 1}")
    print(f"训练损失: {checkpoint['train_loss']:.6f}")
    print(f"验证损失: {checkpoint['val_loss']:.6f}")
    
    x_test_tensor = torch.tensor(x_test)
    test_model(model, x_test_tensor, target_value=0.0)
    
    return model


if __name__ == "__main__":
    # 开始训练
    print("🚀 启动混合量子-经典神经网络训练...")
    trained_model, train_losses, val_losses = train_hybrid_nn()
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), 'final_hybrid_model.pth')
    print("💾 最终模型已保存!")
    
    # 可选：加载并评估最佳模型
    try:
        print("\n🔍 评估最佳模型...")
        best_model = load_and_evaluate('best_hybrid_model.pth', x_test)
    except Exception as e:
        print(f"加载最佳模型时出错: {e}")
