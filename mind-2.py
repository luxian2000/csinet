import numpy as np
import mindspore as ms
import scipy.io as sio

from mindspore import Tensor, ops
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, X, RX, RY, RZ
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import Hamiltonian
from mindquantum.algorithm.nisq import IQPEncoding, Ansatz10
from mindquantum.simulator import Simulator
from mindquantum.framework import MQLayer
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam, TrainOneStepCell
from mindspore.train import Accuracy, Model, LossMonitor
from mindspore.dataset import NumpySlicesDataset

# global params
envir = 'indoor'                                    # 'indoor' or 'outdoor'
n_layers = 2
n_batch = 5

# image params
img_channels = 2
img_height = 32
img_width = 32
img_dim = img_height * img_width * img_channels     # 一个信道矩阵的实数个数
img_qubits = int(np.log2(img_dim))                  # img_qubits+1 是用于编码一个原始矩阵的量子比特数量
total_qubits = img_qubits + 1

# network params
com_height = 16
com_width = 16                                      # 压缩后，矩阵的高和宽
com_dim = com_height * com_width * img_channels     # 一个压缩信道的实数个数
com_qubits = int(np.log2(com_dim))                  # com_qubits+1 是用于编码一个压缩后矩阵的量子比特数量
rec_qubits = img_qubits - com_qubits                # 用于恢复原始矩阵的增加的量子比特数量

# 以下 Compressor 是 CsiNet 中左侧压缩图片的神经网络：
# 下面的 Compressor 就是论文中的 Encoder
class Compressor(ms.nn.Cell):
    def __init__(self, img_channels, img_height, img_width):
        super(Compressor, self).__init__()
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.img_dim = img_height*img_width*img_channels

        self.com_dim = com_dim

        self.conv = ms.nn.Conv2d(
            in_channels = self.img_channels,
            out_channels = 2,
            kernel_size = 3,
            stride = 1,
            pad_mode = 'pad',
            padding = 1,
            has_bias = True
        )

        self.bn2 = ms.nn.BatchNorm2d(num_features=img_channels)
        self.bn1 = ms.nn.BatchNorm1d(num_features=1)
        self.leaky_relu = ms.nn.LeakyReLU(alpha = 0.2)
        self.dense_encode = ms.nn.Dense(in_channels=self.img_dim, out_channels=com_dim)

    def construct(self, x):
        # 第一个黄色箭头
        x = self.conv(x)
        #x = self.bn2(x)
        x = self.leaky_relu(x)

        # 第一个黑色箭头，Reshape: [N, 2, H, W] -> [N, 2*H*W] = [N, img_dim]
        x = x.reshape((x.shape[0], 1, self.img_dim))

        # 第一个红色箭头，Dense：[N, img_dim] -> [N, com_dim]
        encoded = self.dense_encode(x)
        encoded = self.bn1(encoded)
        return encoded


# Data loading
if envir == 'indoor':
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainin.mat')
    x_train = mat['HT'] # array
#    mat = sio.loadmat('../..DataSpace/csinet/data/DATA_Hvalin.mat')
#    x_val = mat['HT'] # array
#    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestin.mat')
#    x_test = mat['HT'] # array

elif envir == 'outdoor':
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainout.mat')
    x_train = mat['HT'] # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalout.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestout.mat')
    x_test = mat['HT'] # array

x_train = x_train.astype('float32')
#x_val = x_val.astype('float32')
#x_test = x_test.astype('float32')
print('x_train 的原始维度:', x_train.shape)

x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
#x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
print('x_train 的塑形维度:', x_train.shape)

compressor = Compressor(img_channels, img_height, img_width)
y_train = compressor(Tensor(x_train))                         # y 是通过无线信道传输到接收端的信号，由量子神经网络进行处理
print('x_train 压缩后的维度:', y_train.shape)
y_train = np.reshape(y_train, (len(x_train), img_channels, -1))
print('x_train 再塑形的维度:', y_train.shape)

# 把矩阵每个元素（复数）归一化：
y_train_normalized = np.zeros_like(y_train)
y_train_phase = np.zeros_like(y_train)
magnitude = [0]*len(y_train)
for i in range(y_train.shape[0]):
    real = y_train[i, 0, :]
    imag = y_train[i, 1, :]
    magnitude[i] = np.sqrt(real**2 + imag**2)
    y_train_normalized[i, 0, :] = real / magnitude[i]
    y_train_normalized[i, 1, :] = imag / magnitude[i]
    # 求出复数的相位：
    y_train_phase[i, :, :] = np.arcsin(y_train_normalized[i, :, :])
y_train_phase = np.reshape(y_train_phase, (len(y_train_phase), -1))
y_train_phase = 2*y_train_phase # 在这里乘以2，因为在RY门中不再乘以2
print('shape of y_train_phase', y_train_phase.shape)

# 构造量子线路
# 使用FRQI编码，下面构造编码量子线路：
img_parm = [f'img_{i}' for i in range(2**img_qubits)]       # 用于编码原始矩阵数据的量子线路的参数
com_parm = [f'com_{i}' for i in range(2**com_qubits)]       # 用于编码压缩矩阵数据的量子线路的参数
#com_parm = y_train_phase[0]
img_index = 2**(img_qubits-1)-1                             # 用于指示当前对哪个数进行编码
com_index = 2**com_qubits-1
print('com_qubits:', com_qubits)
print('length of com_parm:', len(com_parm))
print('com_index:', com_index)

encoder = Circuit()
checker = Circuit()

def com_encoder(index):
    global com_index
    global com_parm
    global com_qubits
    global encoder
    if index == 1:
        encoder += X.on(1)
        com_index = com_index ^ 1
        encoder += RY(com_parm[com_index]).on(0, range(1, com_qubits+1))
    else:
        com_encoder(index-1)
        encoder += X.on(index)
        com_index = com_index ^ (1 << (index-1))
        encoder += RY(com_parm[com_index]).on(0, range(1, com_qubits+1))
        com_encoder(index-1)

def img_checker(index):
    global img_index
    global img_parm
    global img_qubits
    global checker
    if index == 1:
        img_index = img_index ^ 1
        checker += RY(img_parm[img_index]).on(0, range(1, img_qubits+1))
        checker += X.on(1)
    else:
        img_checker(index-1)
        img_index = img_index ^ (1 << (index-1))
        checker += RY(img_parm[img_index]).on(0, range(1, img_qubits+1))
        checker += X.on(index)
        img_checker(index-1)

# 构造压缩恢复量子线路
encoder += UN(H, range(1, com_qubits+1))
encoder += RY(com_parm[com_index]).on(0, range(1, com_qubits+1))
com_encoder(com_qubits)
encoder = encoder.no_grad()
encoder.as_encoder()

# 构造对比量子线路
img_checker(img_qubits)
checker += RY(img_parm[img_index]).on(0, range(1, img_qubits+1))
checker += UN(H, range(1, img_qubits+1))
checker = checker.no_grad()
checker.as_encoder()

ansatz = Ansatz10(total_qubits, n_layers).circuit
ansatz.as_ansatz()

circuit = encoder.as_encoder() + ansatz.as_ansatz() + checker.as_encoder()
circuit.summary()

hams = ' '.join([f'Z{i}' for i in range(total_qubits)])
hams = Hamiltonian(QubitOperator(hams, -1))
print(hams)


# 搭建量子神经网络

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)
sim = Simulator('mqvector', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams, circuit)
QuantumNet = MQLayer(grad_ops)
print(QuantumNet)

loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)

model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})

train_loader = NumpySlicesDataset({'features': x_train, 'labels': y_train}, shuffle=False).batch(n_batch)
test_loader = NumpySlicesDataset({'features': x_test, 'labels': y_test}).batch(n_batch)

class StepAcc(ms.Callback):                                                      # 定义一个关于每一步准确率的回调函数
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []
    def on_train_step_end(self, run_context):
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])

monitor = LossMonitor(16)                                                       # 监控训练中的损失，每16步打印一次损失值
acc = StepAcc(model, test_loader)                                               # 使用建立的模型和测试样本计算预测的准确率
model.train(20, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)# 将上述建立好的模型训练20次
