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

