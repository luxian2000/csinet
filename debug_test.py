#!/usr/bin/env python
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import sys

print("=" * 60)
print("Testing QuantumCompensationBlock initialization and forward")
print("=" * 60)

# Define the quantum block
class TestQuantumBlock(nn.Module):
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

print("✓ Class definition successful")

# Test instantiation
try:
    block = TestQuantumBlock(n_qubits=4, n_layers=2, window_size=2)
    print("✓ Block instantiation successful")
except Exception as e:
    print(f"✗ Instantiation error: {e}")
    sys.exit(1)

# Test forward pass on small data
try:
    batch_size = 2
    channels = 2
    height = 4
    width = 4
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"  Input shape: {x.shape}")
    
    output = block(x)
    print(f"  Output shape: {output.shape}")
    print("✓ Forward pass failed - need to check implementation")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Testing data loading")
print("=" * 60)

try:
    import scipy.io as sio
    data_path = '/Users/luxian/DataSpace/csinet/data'
    
    mat = sio.loadmat(f'{data_path}/DATA_Htrainin.mat')
    x_train = mat['HT'].astype(np.float32)
    print(f"✓ Data loaded: shape {x_train.shape}")
    
    # Test preprocessing
    batch_size = x_train.shape[0]
    img_height, img_width, img_channels = 32, 32, 2
    x_train = x_train.reshape(batch_size, img_channels, img_height, img_width)
    print(f"✓ Data reshaped: {x_train.shape}")
    
except Exception as e:
    print(f"✗ Data loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
