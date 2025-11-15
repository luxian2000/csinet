# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:28:59 2024

@author: HP
"""

import h5py

def explore_h5_file(file_path, path='/'):
    with h5py.File(file_path, 'r') as f:
        def print_name(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        f[path].visititems(print_name)

# 指定你的 .h5 文件路径
file_path = 'saved_model/model_CsiNet_indoor_dim32.h5'

# 查看整个文件的内容
explore_h5_file(file_path)

# 如果你只想查看某个特定组的内容，可以指定路径
# explore_h5_file(file_path, path='/batch_normalization_1')