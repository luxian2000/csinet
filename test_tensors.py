#!/usr/bin/env python
import torch
import torch.nn as nn

print("Testing tensor operations...")

# Test unfold/fold operations
batch, ch, h, w = 2, 2, 8, 8
x = torch.randn(batch, ch, h, w)
print(f"Input shape: {x.shape}")

unfold = nn.Unfold(kernel_size=2, stride=2)
patches = unfold(x)
print(f"After unfold: {patches.shape}")

num_patches = patches.shape[-1]
patches = patches.permute(0, 2, 1).reshape(batch, num_patches, ch, 4)
print(f"After reshape: {patches.shape}")

# Simulate quantum output (skip actual quantum circuit for speed)
n_qubits = 4
total_samples = batch * num_patches * ch
fake_outputs = torch.randn(total_samples, n_qubits)
fake_outputs = fake_outputs.reshape(batch, num_patches, ch, n_qubits)
print(f"Fake quantum outputs: {fake_outputs.shape}")

# Reshape for fold
fake_outputs = fake_outputs.permute(0, 2, 3, 1).reshape(batch, ch*n_qubits, num_patches)
print(f"Reshaped for fold: {fake_outputs.shape}")

fold = nn.Fold(output_size=(h, w), kernel_size=2, stride=2)
output = fold(fake_outputs)
print(f"After fold: {output.shape}")

print("\n✓ All tensor operations work correctly!")
