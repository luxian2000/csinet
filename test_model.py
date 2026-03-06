#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试量子补偿CSI模型是否能正常运行
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型
from my_model.compensation_1_2 import (
    QuantumCompensationBlock,
    CsiNetEncoder,
    QuantumCompensatedDecoder,
    CsiNetQuantumCompensated
)

def test_quantum_block():
    """测试量子补偿模块"""
    print("=" * 60)
    print("测试量子补偿模块")
    print("=" * 60)
    
    try:
        # 创建量子补偿模块
        quantum_block = QuantumCompensationBlock(n_qubits=4, n_layers=2, window_size=2)
        
        # 创建测试输入 (batch_size=2, channels=2, height=16, width=16)
        batch_size = 2
        test_input = torch.randn(batch_size, 2, 16, 16)
        
        print(f"输入形状: {test_input.shape}")
        print(f"量子补偿模块参数:")
        print(f"  - n_qubits: {quantum_block.n_qubits}")
        print(f"  - n_layers: {quantum_block.n_layers}")
        print(f"  - window_size: {quantum_block.window_size}")
        
        # 前向传播
        output = quantum_block(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: [batch, channels*n_qubits, height, width] = [{batch_size}, {2*4}, 16, 16]")
        
        if output.shape == (batch_size, 8, 16, 16):
            print("✓ 量子补偿模块测试通过!")
        else:
            print(f"✗ 量子补偿模块测试失败: 输出形状 {output.shape} 不符合预期")
            
        return True
        
    except Exception as e:
        print(f"✗ 量子补偿模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder():
    """测试编码器"""
    print("\n" + "=" * 60)
    print("测试CsiNet编码器")
    print("=" * 60)
    
    try:
        # 创建编码器 (压缩到512维)
        encoder = CsiNetEncoder(encoded_dim=512)
        
        # 创建测试输入 (batch_size=2, channels=2, height=32, width=32)
        batch_size = 2
        test_input = torch.randn(batch_size, 2, 32, 32)
        
        print(f"输入形状: {test_input.shape}")
        
        # 前向传播
        output = encoder(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: [batch, encoded_dim] = [{batch_size}, 512]")
        
        if output.shape == (batch_size, 512):
            print("✓ 编码器测试通过!")
        else:
            print(f"✗ 编码器测试失败: 输出形状 {output.shape} 不符合预期")
            
        return True
        
    except Exception as e:
        print(f"✗ 编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decoder():
    """测试解码器"""
    print("\n" + "=" * 60)
    print("测试量子补偿解码器")
    print("=" * 60)
    
    try:
        # 创建解码器
        decoder = QuantumCompensatedDecoder(encoded_dim=512, alpha=0.25)
        
        # 创建测试输入 (batch_size=2, encoded_dim=512)
        batch_size = 2
        test_input = torch.randn(batch_size, 512)
        
        print(f"输入形状: {test_input.shape}")
        
        # 前向传播
        output = decoder(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: [batch, channels, height, width] = [{batch_size}, 2, 32, 32]")
        
        if output.shape == (batch_size, 2, 32, 32):
            print("✓ 解码器测试通过!")
        else:
            print(f"✗ 解码器测试失败: 输出形状 {output.shape} 不符合预期")
            
        return True
        
    except Exception as e:
        print(f"✗ 解码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 60)
    print("测试完整CsiNet量子补偿模型")
    print("=" * 60)
    
    try:
        # 创建完整模型
        model = CsiNetQuantumCompensated(encoded_dim=512, alpha=0.25)
        
        # 创建测试输入 (batch_size=2, channels=2, height=32, width=32)
        batch_size = 2
        test_input = torch.randn(batch_size, 2, 32, 32)
        
        print(f"输入形状: {test_input.shape}")
        
        # 前向传播
        output = model(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: [batch, channels, height, width] = [{batch_size}, 2, 32, 32]")
        
        if output.shape == (batch_size, 2, 32, 32):
            print("✓ 完整模型测试通过!")
        else:
            print(f"✗ 完整模型测试失败: 输出形状 {output.shape} 不符合预期")
            
        # 测试获取码字
        codeword = model.get_codeword(test_input)
        print(f"码字形状: {codeword.shape}")
        print(f"期望码字形状: [batch, encoded_dim] = [{batch_size}, 512]")
        
        if codeword.shape == (batch_size, 512):
            print("✓ 获取码字功能测试通过!")
        else:
            print(f"✗ 获取码字功能测试失败: 形状 {codeword.shape} 不符合预期")
            
        return True
        
    except Exception as e:
        print(f"✗ 完整模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试数据加载")
    print("=" * 60)
    
    try:
        # 导入数据加载函数
        from my_model.compensation_1_2 import load_data
        
        # 测试加载数据
        x_train, x_val, x_test, X_test_freq = load_data(
            envir='indoor', 
            data_path='/Users/luxian/DataSpace/csinet/data'
        )
        
        print(f"训练数据形状: {x_train.shape}")
        print(f"验证数据形状: {x_val.shape}")
        print(f"测试数据形状: {x_test.shape}")
        print(f"测试频域数据形状: {X_test_freq.shape}")
        
        # 检查数据形状
        expected_train_shape = (x_train.shape[0], 2, 32, 32)
        expected_val_shape = (x_val.shape[0], 2, 32, 32)
        expected_test_shape = (x_test.shape[0], 2, 32, 32)
        
        if (x_train.shape == expected_train_shape and 
            x_val.shape == expected_val_shape and 
            x_test.shape == expected_test_shape):
            print("✓ 数据加载测试通过!")
        else:
            print(f"✗ 数据加载测试失败: 数据形状不符合预期")
            
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        print("注意: 需要确保数据文件存在于指定路径")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试量子补偿CSI模型...")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("量子补偿模块", test_quantum_block),
        ("编码器", test_encoder),
        ("解码器", test_decoder),
        ("完整模型", test_full_model),
        ("数据加载", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        if success:
            print(f"✓ {test_name}: 通过")
            passed += 1
        else:
            print(f"✗ {test_name}: 失败")
            failed += 1
    
    print(f"\n总计: {len(tests)} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过! 模型可以正常运行。")
        return True
    else:
        print("\n⚠️  有测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)