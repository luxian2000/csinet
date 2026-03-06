#!/usr/bin/env python
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import sys

print("Testing quantum circuit execution...")

n_qubits = 4
n_layers = 2

dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='parameter-shift')
def test_circuit(inputs, weights_crz, weights_ry):
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

weights_crz = torch.randn(n_layers, n_qubits) * 0.1
weights_ry = torch.randn(n_layers, n_qubits) * 0.1
input_data = torch.randn(4)

print("Running quantum circuit...")
q_out = test_circuit(input_data, weights_crz, weights_ry)
print(f"✓ Circuit output type: {type(q_out)}")
print(f"✓ Output: {q_out}")
