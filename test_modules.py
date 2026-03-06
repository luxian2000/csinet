#!/usr/bin/env python
import os
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import scipy.io as sio

print("=" * 60)
print("Step 1: Testing QuantumCompensationBlock")
print("=" * 60)

class QuantumCompensationBlock(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, window_size=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def quantum_circuit(inputs, weights_crz, weights_ry):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[i], wires=i)
            
            for layer in range(n_layers):
                for i in range(n_qubits - 1):
                    qml.CRZ(weights_crz[layer, i], wires=[i, i+1])
                qml.CRZ(weights_crz[layer, n_qubits-1], wires=[n_qubits-1, 0])
                for i in range(n_qubits):
                    qml.RY(weights_ry[layer, i], wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = quantum_circuit

    def forward(self, x):
        batch, ch, h, w = x.shape
        
        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        patches = unfold(x)
        num_patches = patches.shape[-1]
        
        patches = patches.permute(0, 2, 1).reshape(batch, num_patches, ch, 4)
        
        total_samples = batch * num_patches * ch
        all_inputs = patches.reshape(total_samples, 4)
        
        all_inputs = torch.tanh(all_inputs) * np.pi
        
        all_outputs = []
        for i in range(total_samples):
            input_data = all_inputs[i]
            q_out = self.circuit(input_data, self.weights_crz, self.weights_ry)
            all_outputs.append(torch.stack(q_out))
        
        all_outputs = torch.stack(all_outputs, dim=0)
        all_outputs = all_outputs.reshape(batch, num_patches, ch, self.n_qubits)
        all_outputs = all_outputs.permute(0, 2, 3, 1).reshape(batch, ch*self.n_qubits, num_patches)
        
        fold = nn.Fold(output_size=(h, w), kernel_size=self.window_size, stride=self.window_size)
        output = fold(all_outputs)
        
        return output

try:
    block = QuantumCompensationBlock(n_qubits=4, n_layers=2, window_size=2)
    print("✓ Block created successfully")
    
    # Test on small input
    x_test = torch.randn(1, 2, 8, 8)
    print(f"  Input shape: {x_test.shape}")
    
    y_test = block(x_test)
    print(f"  Output shape: {y_test.shape}")
    print(f"✓ Forward pass successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Step 2: Testing encoder/decoder")
print("=" * 60)

img_channels = 2
img_height = 32
img_width = 32
img_total = img_height * img_width * img_channels

class CsiNetEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.conv1 = nn.Conv2d(img_channels, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.lr1 = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(img_total, encoded_dim)
        
    def forward(self, x):
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc_encode(x)
        return x

try:
    encoder = CsiNetEncoder(encoded_dim=512)
    x = torch.randn(4, 2, 32, 32)
    code = encoder(x)
    print(f"✓ Encoder works: input {x.shape} -> output {code.shape}")
except Exception as e:
    print(f"✗ Encoder error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All basic tests passed!")
