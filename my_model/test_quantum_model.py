#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test for the quantum compensation module
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from compensation_1_1 import CsiNetQuantumCompensated

# Create a small test batch
batch_size = 4
img_channels = 2
img_height = 32
img_width = 32
encoded_dim = 32

# Create dummy input
x = torch.randn(batch_size, img_channels, img_height, img_width)

print(f"Input shape: {x.shape}")

# Create model
model = CsiNetQuantumCompensated(encoded_dim=encoded_dim, alpha=0.25)
print(f"Model created successfully")

# Forward pass
print("Starting forward pass...")
try:
    output = model(x)
    print(f"✓ Success! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
