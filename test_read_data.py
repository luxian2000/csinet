#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to read and display CSI data from MATLAB file
"""

import scipy.io as sio
import numpy as np

# File path
file_path = '/Users/luxian/DataSpace/csinet/data/DATA_Htestin.mat'

print("=" * 60)
print("Reading MATLAB data file...")
print("=" * 60)

# Load the .mat file
mat = sio.loadmat(file_path)

# Display all variables in the file
print("\nVariables in the file:")
for key in mat.keys():
    if not key.startswith('__'):
        print(f"  - {key}")

# Get the HT data (channel matrix)
x_test = mat['HT']

# Print data dimensions
print("\n" + "=" * 60)
print("Data Dimensions:")
print("=" * 60)
print(f"Shape of HT: {x_test.shape}")
print(f"Number of samples: {len(x_test)}")
print(f"Features per sample: {x_test.shape[1]}")
print(f"Data type: {x_test.dtype}")

# Print first few samples
print("\n" + "=" * 60)
print("First 5 Samples (showing first 10 features of each):")
print("=" * 60)

n_samples_to_show = 5
n_features_to_show = 10

for i in range(n_samples_to_show):
    print(f"\nSample {i+1}:")
    print(f"  Full shape: {x_test[i].shape}")
    print(f"  First {n_features_to_show} features: {x_test[i, :n_features_to_show]}")
    print(f"  Min value: {np.min(x_test[i]):.6f}")
    print(f"  Max value: {np.max(x_test[i]):.6f}")
    print(f"  Mean value: {np.mean(x_test[i]):.6f}")
    print(f"  Std deviation: {np.std(x_test[i]):.6f}")

# Additional statistics
print("\n" + "=" * 60)
print("Overall Statistics:")
print("=" * 60)
print(f"Global min: {np.min(x_test):.6f}")
print(f"Global max: {np.max(x_test):.6f}")
print(f"Global mean: {np.mean(x_test):.6f}")
print(f"Global std: {np.std(x_test):.6f}")

# Check for NaN or Inf values
print("\n" + "=" * 60)
print("Data Quality Check:")
print("=" * 60)
print(f"Contains NaN: {np.any(np.isnan(x_test))}")
print(f"Contains Inf: {np.any(np.isinf(x_test))}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
