import time
import pennylane as qml
import numpy as np
import torch
from torch import nn
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader

# Global parameters
ENVIR = 'indoor'
LAYERS = 4
BATCHES = 5
CHANNELS = 2

# image parameters
IMG_HEIGHT = 32
IMG_WIDTH = IMG_HEIGHT
IMG_DIM = IMG_HEIGHT * IMG_WIDTH * CHANNELS
IMG_QUBITS = int(np.log2(IMG_DIM))

# compressed parameters
COM_HEIGHT = 8
COM_WIDTH = COM_HEIGHT 
COM_DIM = COM_HEIGHT * COM_WIDTH * CHANNELS
COM_QUBITS = int(np.log2(COM_DIM))

ALL_QUBITS = IMG_QUBITS + 1
ANC_QUBITS = IMG_QUBITS - COM_QUBITS

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


class ClassicalNN(nn.Module):
    ''' 构造经典压缩神经网络 '''
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=CHANNELS, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dense_encode = nn.Linear(in_features=IMG_DIM, out_features=COM_DIM)
        self.bn1d = nn.BatchNorm1d(num_features=1)

    def forward(self, x):
        ''' 定义经典压缩层 '''
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = x.reshape((x.shape[0], -1))     # [batch, features]
        x = self.dense_encode(x)            # [batch, com_dim]
        x = x.unsqueeze(1)                 # [batch, 1, com_dim]
        x = self.bn1d(x)
        x = 2 * x                           # 注意在这里乘以2
        return x


def frqi_encoder(qubits, params, target=0):
    ''' construct the FRQI encoding circuit '''
    for index in range(2**qubits):
        binary_str = bin(index)[2:].zfill(qubits)
        bits = [int(bit) for bit in binary_str]
        bits.reverse()  # 原地逆序，高位在右，作用于序数大的比特位
        qml.ctrl(qml.RY, control=range(1, qubits + 1), control_values=bits)(params[index], wires=target)


coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)


@qml.qnode(dev, interface='torch')
def frqi_circuit(com_params, img_params, asz_params):
    ''' construct the complete quantum circuit '''
    for i in range(1, COM_QUBITS + 1):
        qml.Hadamard(wires=i)

    frqi_encoder(COM_QUBITS, com_params)
    qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
    frqi_encoder(IMG_QUBITS, img_params)

    for i in range(1, IMG_QUBITS + 1):
        qml.Hadamard(wires=i)

    return qml.expval(hamiltonian)


class HybridNN(nn.Module):
    ''' 把上面定义的经典神经网络和量子神经网络组装成完整神经网络 '''
    def __init__(self, classical_nn):
        super().__init__()
        self.classical_nn = classical_nn

        asz_params = np.random.uniform(0, np.pi, size=(LAYERS, ALL_QUBITS, 3))
        self.asz_params = nn.Parameter(torch.tensor(asz_params, dtype=torch.float32))

    def forward(self, x):
        ''' 在量子线路前，加上经典压缩网络 '''
        com_params = self.classical_nn(x)

        # 将输入数据展平用于量子编码
        x = (-1) * x
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # 对batck中的每个样本单独处理
        energies = []
        for i in range(batch_size):
            try:
                energy = frqi_circuit(com_params[i], x_flat[i], self.asz_params)
                energies.append(energy)
            except Exception as e:
                print(f"量子线路执行错误：{e}")
                energies.append(torch.tensor(0.0))

        # loss = frqi_circuit(com_params, x, self.asz_params)
        return torch.stack(energies)


def train_hybrid_nn():
    ''' 训练混合神经网络 '''
    # 初始化模型
    classical_nn = ClassicalNN()
    hybrid_nn = HybridNN(classical_nn=classical_nn)

    # 检查可训练参数
    print("可训练参数分析")
    total_params = 0
    for name, param in hybrid_nn.named_parameters():
        if param.requires_grad:
            print(f"   {name}: {param.shape} ({param.numel()} 个参数)")
            total_params += param.numel()
    print(f" 总可训练参数数量: {total_params}")

    # 优化器
    optimizer = torch.optim.Adam(hybrid_nn.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 损失函数 - 目标是使量子电路输出接近0（或其他目标值）
    criterion = nn.MSELoss()
    target_value = 0.0  # 可以根据任务调整

    # 数据准备
    x_train_tensor = torch.tensor(x_train)
    x_val_tensor = torch.tensor(x_val)
    x_test_tensor = torch.tensor(x_test)

    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, torch.zeros(len(x_train_tensor)))
    train_loader = DataLoader(train_dataset, batch_size=min(BATCHES, len(x_train)), shuffle=True)

    # 训练参数
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    quantum_param_norms = []  # 跟踪量子参数的变化

    print("开始训练混合量子-经典神经网络...")
    print(f" 训练样本数: {len(x_train)}")
    print(f" 验证样本数: {len(x_val)}")
    print(f" 批次大小: {BATCHES}")
    print(f" 目标值: {target_value}")
    print(f" 量子比特数: {ALL_QUBITS} (辅助: 1, 数据: {ALL_QUBITS-1})")

    start_time = time.time()

    for epoch in range(num_epochs):
        # === 训练阶段 ===
        hybrid_nn.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
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

                if batch_idx % 5 == 0:
                    print(f'📍 Epoch {epoch+1:03d} | Batch {batch_idx:03d} | Loss: {loss.item():.6f}')

            except Exception as e:
                print(f"❌ 训练批次 {batch_idx} 出错: {e}")
                continue


train_hybrid_nn()
