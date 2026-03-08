#!/usr/bin/env python3
"""
测试量子线路是否在GPU上运行
"""

import torch
import pennylane as qml
import numpy as np
import time

print("=" * 60)
print("量子线路GPU运行测试")
print("=" * 60)

# 1. 检查CUDA和PennyLane
print("\n1. 系统检查:")
print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")

print(f"\nPennyLane版本: {qml.__version__}")

# 2. 检查可用设备
print("\n2. PennyLane可用设备:")
try:
    devices = qml.available_devices()
    print(f"  可用设备: {devices}")
    
    # 检查lightning.gpu
    if 'lightning.gpu' in devices:
        print("  ✓ lightning.gpu 可用")
    else:
        print("  ✗ lightning.gpu 不可用")
        
        # 检查是否安装了lightning.gpu
        try:
            import pennylane_lightning_gpu
            print("  ✓ pennylane-lightning-gpu 已安装")
        except ImportError:
            print("  ✗ pennylane-lightning-gpu 未安装")
            
except Exception as e:
    print(f"  检查设备失败: {e}")

# 3. 测试创建GPU设备
print("\n3. 测试创建GPU量子设备:")
try:
    # 尝试创建GPU设备
    dev_gpu = qml.device("lightning.gpu", wires=4)
    print(f"  ✓ 成功创建lightning.gpu设备")
    print(f"    设备信息: {dev_gpu}")
    
    # 测试简单量子电路
    @qml.qnode(dev_gpu)
    def simple_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    # 测试执行
    result = simple_circuit()
    print(f"  ✓ GPU量子电路执行成功: 结果 = {result:.4f}")
    
    # 测试批量执行
    print("\n4. 测试批量执行:")
    @qml.qnode(dev_gpu)
    def batch_circuit(x):
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    # 创建批量输入
    batch_input = torch.tensor([0.1, 0.2, 0.3, 0.4], device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  批量输入设备: {batch_input.device}")
    
    # 执行批量电路
    batch_results = []
    for x in batch_input:
        batch_results.append(batch_circuit(x.item()))
    
    print(f"  ✓ 批量执行成功: {batch_results}")
    
except Exception as e:
    print(f"  ✗ 创建/执行GPU设备失败: {e}")
    
    # 回退到CPU
    print("  ⚠ 尝试使用CPU设备...")
    try:
        dev_cpu = qml.device("default.qubit", wires=4)
        print(f"  ✓ 成功创建default.qubit设备")
        
        @qml.qnode(dev_cpu)
        def cpu_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        result = cpu_circuit()
        print(f"  ✓ CPU量子电路执行成功: 结果 = {result:.4f}")
    except Exception as e2:
        print(f"  ✗ 创建CPU设备也失败: {e2}")

# 5. 分析您的代码中的量子部分
print("\n5. 分析您的代码:")
print("  在 QuantumCompensationBlock 中:")
print("  - 尝试创建 lightning.gpu 设备")
print("  - 使用了 batch_obs=True 参数")
print("  - 但量子电路输入需要确保在GPU上")

# 6. 检查量子电路输入输出设备
print("\n6. 设备一致性检查:")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
    # 测试张量设备
    cpu_tensor = torch.randn(4)
    gpu_tensor = cpu_tensor.to(device)
    
    print(f"  CPU张量设备: {cpu_tensor.device}")
    print(f"  GPU张量设备: {gpu_tensor.device}")
    
    # 测试量子参数设备
    try:
        dev = qml.device("lightning.gpu", wires=4)
        
        @qml.qnode(dev, interface="torch")
        def test_circuit(params):
            for i in range(4):
                qml.RY(params[i], wires=i)
            return qml.expval(qml.PauliZ(0))
        
        # 测试不同设备的参数
        params_cpu = torch.randn(4, requires_grad=True)
        params_gpu = params_cpu.to(device)
        
        print(f"\n  测试量子电路参数设备:")
        print(f"   参数设备 (CPU): {params_cpu.device}")
        try:
            result_cpu = test_circuit(params_cpu)
            print(f"   使用CPU参数执行: {result_cpu:.4f}")
        except Exception as e:
            print(f"   使用CPU参数失败: {e}")
        
        print(f"   参数设备 (GPU): {params_gpu.device}")
        try:
            result_gpu = test_circuit(params_gpu)
            print(f"   使用GPU参数执行: {result_gpu:.4f}")
        except Exception as e:
            print(f"   使用GPU参数失败: {e}")
            
    except Exception as e:
        print(f"  测试失败: {e}")

# 7. 性能测试
print("\n7. 性能测试:")
try:
    # 创建GPU设备
    dev_gpu = qml.device("lightning.gpu", wires=8)
    
    @qml.qnode(dev_gpu)
    def perf_circuit(theta):
        for i in range(8):
            qml.Hadamard(wires=i)
            qml.RY(theta, wires=i)
        
        # 添加一些纠缠
        for i in range(7):
            qml.CNOT(wires=[i, i+1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(8)]
    
    # 预热
    for _ in range(5):
        _ = perf_circuit(0.5)
    
    # 基准测试
    import time
    times = []
    for _ in range(20):
        start = time.time()
        result = perf_circuit(0.5)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    print(f"  平均执行时间: {avg_time:.2f} ms")
    print(f"  每秒执行次数: {1/np.mean(times):.1f}")
    
except Exception as e:
    print(f"  性能测试失败: {e}")

# 8. 检查您的代码中的关键问题
print("\n8. 您的代码中的关键问题:")
print("  ✓ 尝试使用 lightning.gpu 设备")
print("  ✓ 设置了 batch_obs=True")
print("  ⚠ 注意: 量子电路输入需要与量子参数在同一设备上")
print("  ⚠ 注意: 检查是否安装了 pennylane-lightning[gpu]")
print("  ⚠ 注意: 确保CUDA驱动版本匹配")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

print("\n建议:")
print("1. 安装: pip install pennylane-lightning[gpu]")
print("2. 确保CUDA版本匹配")
print("3. 检查量子电路输入是否在GPU上")
print("4. 使用 torch.cuda.is_available() 验证CUDA状态")
print("5. 如果lightning.gpu失败，检查错误信息")
