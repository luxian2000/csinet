#!/usr/bin/env python3
"""
简单检查量子GPU支持
"""

import sys

# 首先检查是否安装了必要的包
try:
    import pennylane as qml
    print(f"✓ PennyLane 已安装: {qml.__version__}")
except ImportError:
    print("✗ PennyLane 未安装")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch 已安装: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA 不可用")
except ImportError:
    print("✗ PyTorch 未安装")

print("\n检查PennyLane GPU支持:")
try:
    # 检查可用设备
    devices = qml.available_devices()
    print(f"可用设备: {devices}")
    
    if 'lightning.gpu' in devices:
        print("✓ lightning.gpu 在可用设备列表中")
    else:
        print("✗ lightning.gpu 不在可用设备列表中")
        
        # 检查是否安装了lightning.gpu
        try:
            import pennylane_lightning_gpu
            print("✓ pennylane-lightning-gpu 已安装")
            
            # 但仍然不在可用设备列表中，可能是CUDA问题
            print("⚠ 已安装但不在设备列表中，可能是CUDA驱动问题")
        except ImportError:
            print("✗ pennylane-lightning-gpu 未安装")
            print("  请运行: pip install pennylane-lightning[gpu]")
            
except Exception as e:
    print(f"检查设备时出错: {e}")

print("\n尝试创建GPU设备:")
try:
    dev = qml.device("lightning.gpu", wires=4)
    print(f"✓ 成功创建GPU设备: {dev}")
    
    # 测试简单电路
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    result = circuit()
    print(f"✓ 量子电路执行成功: {result}")
    
    # 检查设备属性
    print(f"设备名称: {dev.name}")
    print(f"设备参数: {dev.__dict__}")
    
except Exception as e:
    print(f"✗ 创建/执行GPU设备失败: {e}")
    
    # 尝试CPU设备
    print("\n尝试CPU设备:")
    try:
        dev_cpu = qml.device("default.qubit", wires=4)
        print(f"✓ 成功创建CPU设备: {dev_cpu}")
        
        @qml.qnode(dev_cpu)
        def cpu_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        result = cpu_circuit()
        print(f"✓ CPU量子电路执行成功: {result}")
    except Exception as e2:
        print(f"✗ CPU设备也失败: {e2}")

print("\n" + "=" * 60)
print("总结:")
print("=" * 60)
print("\n根据您的代码 compensation_gpu.py:")
print("1. 第44-50行: 尝试创建 lightning.gpu 设备")
print("2. 如果失败会抛出 RuntimeError")
print("3. 这意味着如果代码能运行，lightning.gpu 应该可用")
print("\n但是，有几点需要注意:")
print("1. 量子电路输入需要与量子参数在同一设备上")
print("2. 您的代码中使用了 torch.tensor 创建输入，需要确保在GPU上")
print("3. 检查第120-140行的设备处理")
print("\n关键代码段:")
print("```python")
print("# 第120-125行")
print("original_device = x.device")
print("x_map = x.reshape(batch, 1, latent_h, latent_w)")
print("# 强制使用 GPU：把输入移动到与量子模块权重相同的设备（应为 CUDA）")
print("x_proc = x_map.to(self.weights_crz.device)")
print("```")
print("\n这表明您已经注意了设备一致性。")
