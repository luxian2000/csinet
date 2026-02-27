import tensorflow as tf
from keras.models import model_from_json
import numpy as np

def count_parameters(model):
    """计算模型参数数量"""
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for layer in model.layers:
        layer_params = layer.count_params()
        total_params += layer_params
        if layer.trainable:
            trainable_params += layer_params
        else:
            non_trainable_params += layer_params
            
    return total_params, trainable_params, non_trainable_params

# 加载CsiNet模型
def load_model(json_file):
    with open(json_file, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    return model

# 分析CsiNet模型 (以indoor_dim512为例)
print("=== CsiNet 模型参数分析 ===")
csinet_model = load_model('saved_model/model_CsiNet_indoor_dim512.json')
total, trainable, non_trainable = count_parameters(csinet_model)
print(f"总参数数量: {total:,}")
print(f"可训练参数数量: {trainable:,}")
print(f"不可训练参数数量: {non_trainable:,}")

print("\n=== CsiNet 各层参数详情 ===")
for i, layer in enumerate(csinet_model.layers):
    params = layer.count_params()
    if params > 0:
        print(f"层 {i}: {layer.name} ({layer.__class__.__name__}) - 参数数量: {params:,}")

# 分析CS-CsiNet模型 (以indoor_dim512为例)
print("\n=== CS-CsiNet 模型参数分析 ===")
cs_csinet_model = load_model('saved_model/model_CS-CsiNet_indoor_dim512.json')
total, trainable, non_trainable = count_parameters(cs_csinet_model)
print(f"总参数数量: {total:,}")
print(f"可训练参数数量: {trainable:,}")
print(f"不可训练参数数量: {non_trainable:,}")

print("\n=== CS-CsiNet 各层参数详情 ===")
for i, layer in enumerate(cs_csinet_model.layers):
    params = layer.count_params()
    if params > 0:
        print(f"层 {i}: {layer.name} ({layer.__class__.__name__}) - 参数数量: {params:,}")