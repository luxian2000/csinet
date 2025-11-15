import pennylane as qml
import numpy as np
import torch
import torch.nn as nn


dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)  
def circuit(x, diff_method="parameter-shift", interface="torch"):
    '''this is an example
    '''
    qml.RZ(x, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(x, wires=1)
    return qml.expval(qml.Z(1))


class Compressor(nn.Module):
    ''' 经典的压缩信道矩阵部分
    '''
    def __init__(self, img_channels, img_height, img_width, com_dim):
        super(Compressor, self).__init__()
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.img_dim = img_height * img_width * img_channels
        self.com_dim = com_dim

        self.conv = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,  # PyTorch中直接使用padding
            bias=True   # PyTorch中叫bias而不是has_bias
        )

        self.bn2 = nn.BatchNorm2d(num_features=img_channels)
        self.bn1 = nn.BatchNorm1d(num_features=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  # PyTorch中叫negative_slope而不是alpha
        self.dense_encode = nn.Linear(in_features=self.img_dim, out_features=com_dim)  # PyTorch中叫Linear而不是Dense

    def forward(self, x):  # PyTorch中使用forward而不是construct
        # 第一个黄色箭头
        x = self.conv(x)
        #x = self.bn2(x)
        x = self.leaky_relu(x)

        # 第一个黑色箭头，Reshape: [N, 2, H, W] -> [N, 1, img_dim]
        x = x.reshape((x.shape[0], 1, self.img_dim))

        # 第一个红色箭头，Linear：[N, 1, img_dim] -> [N, 1, com_dim]
        encoded = self.dense_encode(x)
        encoded = self.bn1(encoded)
        encoded = 2 * encoded  # 在这里乘以2，因为在RY门不再乘以2
        return encoded
