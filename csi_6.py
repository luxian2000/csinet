import numpy as np
import mindspore as ms
import scipy.io as sio

from mindspore import Tensor, ops, nn
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
img_dim = img_height * img_width * img_channels     # 一个信道矩阵的实部和虚部的数量，以下把实部和虚部一起编码
img_qubits = int(np.log2(img_dim))                  # img_qubits+1 是FRQI方式用于编码一个原始矩阵的量子比特数量，也是整个量子线路的比特数量
total_qubits = img_qubits + 1

# network params
com_height = 16
com_width = 16                                      # 压缩后，矩阵的高和宽
com_dim = com_height * com_width * img_channels     # 一个压缩信道的实部和虚部数量
com_qubits = int(np.log2(com_dim))                  # com_qubits+1 是FRQI用于编码一个压缩后矩阵的量子比特数量
rec_qubits = img_qubits - com_qubits                # 用于恢复原始矩阵的增加的量子比特数量

######################################################### Data loading ###########################################################
if envir == 'indoor':
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainin.mat')
    x_train = mat['HT'] # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalin.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestin.mat')
    x_test = mat['HT'] # array

elif envir == 'outdoor':
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htrainout.mat')
    x_train = mat['HT'] # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Hvalout.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('../../DataSpace/csinet/data/DATA_Htestout.mat')
    x_test = mat['HT'] # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
print('x_train 的原始维度:', x_train.shape)

x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
print('x_train 的塑形维度:', x_train.shape)

# 创建数据加载器
def create_dataset(data, batch_size=32, shuffle=True):
    dataset = NumpySlicesDataset(data, shuffle=shuffle)
    return dataset.batch(batch_size)

train_loader = create_dataset(x_train, n_batch)
val_loader = create_dataset(x_val, n_batch, shuffle=False)
test_loader = create_dataset(x_test, n_batch, shuffle=False)
###################################################### End of Data loading #######################################################

# 以下 Compressor 是 CsiNet 中左侧压缩图片的神经网络, 就是论文中的 Encoder
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
        encoded = 2*encoded                     # 在这里乘以2，因为在RY门不再乘以2
        return encoded

#y_train = compressor(Tensor(x_train))                         # y 是通过无线信道传输到接收端的信号，由量子神经网络进行处理
#y_train = np.reshape(y_train, (len(x_train), img_channels, -1))

# 使用FRQI编码，下面构造编码量子线路：
#img_parm = [f'img_{i}' for i in range(2**img_qubits)]       # 用于编码原始矩阵数据的量子线路的参数
#com_parm = [f'com_{i}' for i in range(2**com_qubits)]       # 用于编码压缩矩阵数据的量子线路的参数

# 重写量子编码器函数，避免使用全局变量
def build_com_encoder(com_qubits):
    com_parm = [f'com_{i}' for i in range(2**com_qubits)]       # 用于编码压缩矩阵数据的量子线路的参数
    encoder = Circuit()
    encoder += UN(H, range(1, com_qubits+1))
    com_index = 2**com_qubits-1
    encoder += RY(com_parm[com_index]).on(0, range(1, com_qubits+1))
    
    # 递归编码函数
    def add_com_encoder(circ, index, current_index):
        if index == 1:
            circ += X.on(1)
            new_index = current_index ^ 1
            circ += RY(com_parm[new_index]).on(0, range(1, com_qubits+1))
            return new_index
        else:
            current_index = add_com_encoder(circ, index-1, current_index)
            circ += X.on(index)
            new_index = current_index ^ (1 << (index-1))
            circ += RY(com_parm[new_index]).on(0, range(1, com_qubits+1))
            current_index = add_com_encoder(circ, index-1, new_index)
            return current_index
    
    add_com_encoder(encoder, com_qubits, com_index)
    return encoder

def build_img_checker(img_qubits):
    img_parm = [f'img_{i}' for i in range(2**img_qubits)]       # 用于编码原始矩阵数据的量子线路的参数
    checker = Circuit()
    img_index = 2**(img_qubits-1)-1
    
    # 递归检查函数
    def add_img_checker(circ, index, current_index):
        if index == 1:
            new_index = current_index ^ 1
            circ += RY(img_parm[new_index]).on(0, range(1, img_qubits+1))
            circ += X.on(1)
            return new_index
        else:
            current_index = add_img_checker(circ, index-1, current_index)
            new_index = current_index ^ (1 << (index-1))
            circ += RY(img_parm[new_index]).on(0, range(1, img_qubits+1))
            circ += X.on(index)
            current_index = add_img_checker(circ, index-1, new_index)
            return current_index
    
    add_img_checker(checker, img_qubits, img_index)
    checker += RY(img_parm[img_index]).on(0, range(1, img_qubits+1))
    checker += UN(H, range(1, img_qubits+1))
    return checker

# 构造压缩恢复量子线路
encoder = build_com_encoder(com_qubits)
encoder = encoder.no_grad()
encoder.as_encoder()

# 构造对比量子线路
checker = build_img_checker(img_qubits)
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

# 使用混合量子经典网络
class HybridQuantumClassicNet(ms.nn.Cell):
    def __init__(self, img_channels, img_height, img_width, com_height, com_width):
        super(HybridQuantumClassicNet, self).__init__()
        self.compressor = Compressor(img_channels, img_height, img_width)

        self.com_index = com_height * com_width * img_channels - 1
        self.com_qubits = int(np.log2(self.com_index+1))
        self.com_parm = [f'com_{i}' for i in range(2**self.com_qubits)]

        self.encoder = Circuit()
        self.encoder += UN(H, range(1, self.com_qubits+1))
        com_encoder(self.com_qubits)
        self.encoder = self.encoder.no_grad()
        self.encoder.as_ansatz()

        self.ansatz = Ansatz10(total_qubits, n_layers).circuit.as_ansatz()
        
        self.img_index = img_height * img_width * img_channels
        self.img_qubits = int(np.log2(self.img_index))
        self.img_parm = [f'img_{i}' for i in range(2**img_qubits)]

        self.checker = Circuit()
        self.quantum_net = quantum_net 
        
    def com_encoder(self, index):
	    if index == 1:
            self.encoder
            self.encoder += X.on(1)
            self.com_index = self.com_index ^ 1
            self.encoder += RY(self.com_parm[self.com_index]).on(0, range(1, com_qubits+1))
	    else:
	        com_encoder(index-1)
	        self.encoder += X.on(index)
	        self.com_index = self.com_index ^ (1 << (index-1))
	        self.encoder += RY(self.com_parm[self.com_index]).on(0, range(1, com_qubits+1))
	        com_encoder(index-1)

	def img_checker(self, index):
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

    def construct(self, x):
        compressed_features = self.compressor(x)
        output = self.quantum_net(x, compressed_features)
        return output


# 创建自定义的量子处理层
class DualInputQuantumLayer(ms.nn.Cell):
    def __init__(self, circuit, hams):
        super(DualInputQuantumLayer, self).__init__()
        self.circuit = circuit
        self.compressor = compressor

    def construct(self, original_data, compressed_data):
        batch_size = original_data.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # 创建参数解析器，同时包含两种数据
            pr = {}
            
            # 添加压缩数据到encoder参数
            comp_flat = compressed_data[i].flatten()
            for j in range(min(len(comp_flat), 2**com_qubits)):
                pr[f'com_{j}'] = comp_flat[j]
            
            # 添加原始数据到checker参数  
            orig_flat = original_data[i].flatten()
            for j in range(min(len(orig_flat), 2**img_qubits)):
                pr[f'img_{j}'] = orig_flat[j]
            
            # 转换为numpy数组
            param_names = sorted(pr.keys())
            param_values = np.array([pr[name] for name in param_names], dtype=np.float32)
            
            output = self.grad_ops(param_values)
            outputs.append(output[0])
        
        return Tensor(outputs, dtype=ms.float32).reshape(batch_size, 1)

# 使用自定义层代替MQLayer
quantum_net = DualInputQuantumLayer(circuit, hams)





# 创建混合模型
hybrid_model = HybridQuantumClassicNet(compressor, quantum_net)

# 定义损失函数和优化器
loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opti = Adam(hybrid_model.trainable_params(), learning_rate=0.01)  # 降低学习率以提高稳定性

# 创建模型
model = Model(hybrid_model, loss, opti, metrics={'Acc': Accuracy()})

class StepAcc(ms.Callback):
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []
        
    def on_train_step_end(self, run_context):
        acc_value = self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc']
        self.acc.append(acc_value)
        print(f"Step accuracy: {acc_value}")

# 训练模型
monitor = LossMonitor(16)
acc_callback = StepAcc(model, val_loader)  # 使用验证集进行准确率评估

print("开始训练模型...")
model.train(20, train_loader, callbacks=[monitor, acc_callback], dataset_sink_mode=False)

# 测试模型
print("测试模型性能...")
test_result = model.eval(test_loader, dataset_sink_mode=False)
print(f"测试准确率: {test_result['Acc']}")
