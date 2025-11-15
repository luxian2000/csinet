import numpy as np                                        # 导入numpy库并简写为np
from sklearn import datasets                              # 导入datasets模块，用于加载鸢尾花的数据集
import matplotlib.pyplot as plt 

iris_dataset = datasets.load_iris()                       # 加载鸢尾花的数据集，并存在iris_dataset

print(iris_dataset.data.shape)                            # 打印iris_dataset的样本的数据维度
print(iris_dataset.feature_names)                         # 打印iris_dataset的样本的特征名称
print(iris_dataset.target_names)                          # 打印iris_dataset的样本包含的亚属名称
print(iris_dataset.target)                                # 打印iris_dataset的样本的标签的数组
print(iris_dataset.target.shape)                          # 打印iris_dataset的样本的标签的数据维度

X = iris_dataset.data[:100, :].astype(np.float32)         # 选取iris_dataset的data的前100个数据，将其数据类型转换为float32，并储存在X中
X_feature_names = iris_dataset.feature_names              # 将iris_dataset的特征名称储存在X_feature_names中
y = iris_dataset.target[:100].astype(int)                 # 选取iris_dataset的target的前100个数据，将其数据类型转换为int，并储存在y中
y_target_names = iris_dataset.target_names[:2]            # 选取iris_dataset的target_names的前2个数据，并储存在y_target_names中

print(X.shape)                                            # 打印样本的数据维度
print(X_feature_names)                                    # 打印样本的特征名称
print(y_target_names)                                     # 打印样本包含的亚属名称
print(y)                                                  # 打印样本的标签的数组
print(y.shape)                                            # 打印样本的标签的数据维度

alpha = X[:, :3] * X[:, 1:]           # 每一个样本中，利用相邻两个特征值计算出一个参数，即每一个样本会多出3个参数（因为有4个特征值），并储存在alpha中
X = np.append(X, alpha, axis=1)       # 在axis=1的维度上，将alpha的数据值添加到X的特征值中

print(X.shape)                        # 打印此时X的样本的数据维度

from sklearn.model_selection import train_test_split                                                   # 导入train_test_split函数，用于对数据集进行划分

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True) # 将数据集划分为训练集和测试集

print(X_train.shape)                                                                                   # 打印训练集中样本的数据类型
print(X_test.shape)                                                                                    # 打印测试集中样本的数据类型

# pylint: disable=W0104
from mindquantum.core.circuit import Circuit
from mindquantum.core.circuit import UN
from mindquantum.core.gates import H, X, RZ
from mindquantum.core.parameterresolver import PRGenerator

prg = PRGenerator('alpha')
encoder = Circuit()
encoder += UN(H, 4)                                  # H门作用在每1位量子比特
for i in range(4):                                   # i = 0, 1, 2, 3
    encoder += RZ(prg.new()).on(i)                 # RZ(alpha_i)门作用在第i位量子比特
for j in range(3):                                   # j = 0, 1, 2
    encoder += X.on(j+1, j)                          # X门作用在第j+1位量子比特，受第j位量子比特控制
    encoder += RZ(prg.new()).on(j+1)             # RZ(alpha_{j+4})门作用在第0位量子比特
    encoder += X.on(j+1, j)                          # X门作用在第j+1位量子比特，受第j位量子比特控制

encoder = encoder.no_grad()                          # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
encoder.summary()                                    # 总结Encoder
encoder.svg()

# pylint: disable=W0104
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz                                      # 导入HardwareEfficientAnsatz
from mindquantum.core.gates import RY                                                               # 导入量子门RY

ansatz = HardwareEfficientAnsatz(4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit     # 通过HardwareEfficientAnsatz搭建Ansatz
ansatz.summary()                                                                                    # 总结Ansatz
ansatz.svg()

# pylint: disable=W0104
circuit = encoder.as_encoder() + ansatz.as_ansatz()                  # 完整的量子线路由Encoder和Ansatz组成
circuit.summary()
circuit.svg()

from mindquantum.core.operators import QubitOperator           # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量

hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]   # 分别对第2位和第3位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
for h in hams:
    print(h)

# pylint: disable=W0104
import mindspore as ms                                                                         # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer                                                      # 导入MQLayer
from mindquantum.simulator import Simulator

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)                                                                                 # 设置生成随机数的种子
sim = Simulator('mqvector', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams, circuit, parallel_worker=5)
QuantumNet = MQLayer(grad_ops)                                                 # 搭建量子神经网络
QuantumNet

from mindspore.nn import SoftmaxCrossEntropyWithLogits                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam                                                  # 导入Adam模块用于定义优化参数
from mindspore.train import Accuracy, Model, LossMonitor                       # 导入Accuracy模块，用于评估预测准确率
import mindspore as ms
from mindspore.dataset import NumpySlicesDataset                               # 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集

loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')            # 通过SoftmaxCrossEntropyWithLogits定义损失函数，sparse=True表示指定标签使用稀疏格式，reduction='mean'表示损失函数的降维方法为求平均值
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)                  # 通过Adam优化器优化Ansatz中的参数，需要优化的是Quantumnet中可训练的参数，学习率设为0.1

model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})             # 建立模型：将MindSpore Quantum构建的量子机器学习层和MindSpore的算子组合，构成一张更大的机器学习网络

train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(5) # 通过NumpySlicesDataset创建训练样本的数据集，shuffle=False表示不打乱数据，batch(5)表示训练集每批次样本点有5个
test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test}).batch(5)                   # 通过NumpySlicesDataset创建测试样本的数据集，batch(5)表示测试集每批次样本点有5个


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



plt.plot(acc.acc)
plt.title('Statistics of accuracy', fontsize=20)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
