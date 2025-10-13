import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 数据加载
data_30 = np.load('CSI_channel_30km.npy')  # shape=(80000, 2560)

# 数据划分参数
TOTAL_SAMPLES = 80000
TRAIN_RATIO = 0.70    # 70% 训练
VAL_RATIO = 0.15      # 15% 验证  
TEST_RATIO = 0.15     # 15% 测试

# 计算各集合大小
train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_size = int(TOTAL_SAMPLES * VAL_RATIO)
test_size = TOTAL_SAMPLES - train_size - val_size

# 划分数据集
train_data = data_30[:train_size]
val_data = data_30[train_size:train_size + val_size]
test_data = data_30[train_size + val_size:]

print("数据划分结果:")
print(f"训练集: {len(train_data)} 个样本 ({TRAIN_RATIO*100:.1f}%)")
print(f"验证集: {len(val_data)} 个样本 ({VAL_RATIO*100:.1f}%)")
print(f"测试集: {len(test_data)} 个样本 ({TEST_RATIO*100:.1f}%)")

INPUT_DIM = 2560
OUTPUT_DIM = 256

N_LAYERS = 4
IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 8
ALL_QUBITS = IMG_QUBITS  # 12个量子比特

print(f"IMG_QUBITS: {IMG_QUBITS}, COM_QUBITS: {COM_QUBITS}, ALL_QUBITS: {ALL_QUBITS}")

# 创建保存参数的目录
os.makedirs('model_parameters', exist_ok=True)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def normlize(x):
    norm = torch.norm(x)
    if norm == 0:
        return x
    return x / norm

# 定义经典神经网络层
class ClassicalNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.bias = nn.Parameter(torch.randn(1, output_dim))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        output = torch.matmul(x, self.weight) + self.bias
        output = sigmoid(output)
        output = normlize(output[0])  # 确保输出是一维的
        return output

# 初始化经典神经网络
classical_nn = ClassicalNN(INPUT_DIM, OUTPUT_DIM)

def save_initial_parameters():
    """保存初始化的参数"""
    torch.save(classical_nn.state_dict(), 'model_parameters/initial_classical_nn.pt')
    print("Initial classical NN parameters saved!")

coe = [-1] * ALL_QUBITS
obs_list = [qml.PauliZ(0)] * ALL_QUBITS
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)

# 重新设计量子电路，避免计算图冲突
def create_quantum_circuit():
    """创建新的量子电路实例"""
    @qml.qnode(dev, interface='torch')
    def circuit(img_params, asz_params):
        ''' 单个样本的量子电路 '''
        # 压缩参数编码，已经归一化
        com_params = classical_nn(img_params)
        # 确保参数长度正确
        if len(com_params) < 2**COM_QUBITS:
            com_params_padded = torch.nn.functional.pad(com_params, (0, 2**COM_QUBITS - len(com_params)))
        else:
            com_params_padded = com_params[:2**COM_QUBITS]
        qml.AmplitudeEmbedding(com_params_padded, wires=range(COM_QUBITS), pad_with=0.0, normalize=True)
        # 强纠缠层
        qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
        # 图像参数编码
        img_params_norm = normlize(img_params)
        if len(img_params_norm) < 2**IMG_QUBITS:
            img_params_padded = torch.nn.functional.pad(img_params_norm, (0, 2**IMG_QUBITS - len(img_params_norm)))
        else:
            img_params_padded = img_params_norm[:2**IMG_QUBITS]
        qml.adjoint(qml.AmplitudeEmbedding(img_params_padded, wires=range(IMG_QUBITS), pad_with=0.0, normalize=True))

        return qml.expval(hamiltonian)

    return circuit

# 批量处理函数 - 完全重新设计
def process_batch_safe(img_batch, asz_params):
    ''' 安全的批量处理，每次创建新的计算图 '''
    batch_results = []

    for img_params in img_batch:
        # 确保输入是PyTorch张量
        if isinstance(img_params, np.ndarray):
            img_params = torch.from_numpy(img_params).float()

        # 为每个样本创建新的量子电路
        circuit = create_quantum_circuit()
        result = circuit(img_params, asz_params)

        # 确保结果是实数类型
        if isinstance(result, (complex, np.complex128)):
            result = torch.tensor(np.real(result), dtype=torch.float32)

        batch_results.append(result)

    return torch.stack(batch_results)

def validate_model(weights, val_samples=1000):
    """在验证集上评估模型"""
    try:
        # 使用部分验证集进行评估（避免计算时间过长）
        val_subset = val_data[:min(val_samples, len(val_data))]

        # 在验证时不需要梯度
        with torch.no_grad():
            results = process_batch_safe(val_subset, weights)
        return float(torch.mean(results))
    except Exception as e:
        print(f"Validation error: {e}")
        return float('inf')

# 重新设计训练函数
def train_batch_version():
    try:
        # 保存初始参数
        save_initial_parameters()

        # 使用训练集
        n_samples = 1000  # 先用1000个训练样本进行测试
        samples = train_data[:n_samples]

        shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=ALL_QUBITS)
        quantum_weights = torch.rand(shape, requires_grad=True)

        # 保存初始量子权重
        torch.save(quantum_weights.detach(), 'model_parameters/initial_quantum_weights.pt')
        print("Initial quantum weights saved!")

        # 优化器同时优化经典和量子参数
        all_parameters = list(classical_nn.parameters()) + [quantum_weights]
        opt = torch.optim.SGD(all_parameters, lr=0.01)

        n_epochs = 5  # 先用较少的epoch测试
        batch_size = 50

        # 记录训练历史
        training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'batch_losses': [],
            'weights_history': [],
            'data_split_info': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'actual_train_used': n_samples
            }
        }

        print("Starting training...")
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i:i+batch_size]

                def closure():
                    opt.zero_grad()

                    # 为每个batch创建全新的计算图
                    total_loss = 0.0
                    sample_count = 0

                    for img_params in batch:
                        # 确保输入是PyTorch张量
                        if isinstance(img_params, np.ndarray):
                            img_params = torch.from_numpy(img_params).float()
                        
                        # 为每个样本创建新的量子电路
                        circuit = create_quantum_circuit()
                        result = circuit(img_params, quantum_weights)

                        # 确保结果是实数类型
                        if isinstance(result, (complex, np.complex128)):
                            result = torch.tensor(np.real(result), dtype=torch.float32, requires_grad=True)

                        total_loss = total_loss + result
                        sample_count += 1

                    # 计算平均损失
                    if sample_count > 0:
                        avg_loss = total_loss / sample_count
                        avg_loss.backward()
                        return avg_loss
                    else:
                        return torch.tensor(0.0, requires_grad=True)

                # 记录训练前的权重
                pre_update_weights = quantum_weights.clone().detach()

                # 更新权重
                loss = opt.step(closure)
                current_loss = loss.item() if hasattr(loss, 'item') else float(loss)
                epoch_loss += current_loss
                batch_count += 1

                # 记录批次信息
                training_history['batch_losses'].append({
                    'epoch': epoch,
                    'batch': i // batch_size,
                    'loss': float(current_loss),
                    'pre_weights_norm': float(torch.norm(pre_update_weights)),
                    'post_weights_norm': float(torch.norm(quantum_weights.detach()))
                })

                if (i // batch_size) % 5 == 0:  # 每5个batch打印一次
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                # 计算验证损失
                val_loss = validate_model(quantum_weights, val_samples=500)
                training_history['epoch_losses'].append({
                    'epoch': epoch,
                    'avg_loss': float(avg_epoch_loss)
                })
                training_history['val_losses'].append({
                    'epoch': epoch,
                    'val_loss': float(val_loss)
                })
                # 保存每个epoch的权重
                epoch_weights = quantum_weights.clone().detach()
                training_history['weights_history'].append(epoch_weights.numpy())
                torch.save(epoch_weights, f'model_parameters/quantum_weights_epoch_{epoch}.pt')
                torch.save(classical_nn.state_dict(), f'model_parameters/classical_nn_epoch_{epoch}.pt')
                print(f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Loss = {val_loss:.6f}")
                print(f"Model weights for epoch {epoch} saved!")
                print("-" * 50)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds!")

        # 保存最终权重和训练历史
        torch.save(quantum_weights.detach(), 'model_parameters/final_quantum_weights.pt')
        torch.save(classical_nn.state_dict(), 'model_parameters/final_classical_nn.pt')
        torch.save(training_history, 'model_parameters/training_history.pt')
        print("Final model weights and training history saved!")

        return quantum_weights, training_history

    except Exception as e:
        print(f"Error in batch training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 测试训练好的模型
def test_trained_model(weights, test_samples=1000):
    """测试训练好的模型"""
    print("\nTesting trained model on test set...")
    try:
        # 使用部分测试集进行评估
        test_subset = test_data[:min(test_samples, len(test_data))]

        with torch.no_grad():
            results = process_batch_safe(test_subset, weights)

        print(f"Test results on {len(test_subset)} samples:")
        for i in range(min(5, len(results))):  # 只显示前5个结果
            print(f"  Sample {i}: {results[i].item():.6f}")
        if len(results) > 5:
            print(f"  ... (showing first 5 of {len(results)} results)")
        avg_result = torch.mean(results).item()
        std_result = torch.std(results).item()
        print(f"Average test result: {avg_result:.6f}")
        print(f"Standard deviation: {std_result:.6f}")
        print(f"Min: {torch.min(results).item():.6f}, Max: {torch.max(results).item():.6f}")
        return results
    except Exception as e:
        print(f"Error in testing: {e}")
        return None
    
# 主程序
if __name__ == "__main__":
    print("Starting quantum-classical hybrid model training with train/val/test split...")
    print("=" * 60)
    # 显示数据划分信息
    print(f"Data Split: {TRAIN_RATIO*100:.0f}% Train, {VAL_RATIO*100:.0f}% Validation, {TEST_RATIO*100:.0f}% Test")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print("=" * 60)

    # 开始训练
    final_weights, history = train_batch_version()

    if final_weights is not None:
        # 测试训练好的模型
        test_results = test_trained_model(final_weights)
        # 显示训练总结
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY:")
        print("=" * 60)
        print(f"Data split: {TRAIN_RATIO*100:.1f}% train, {VAL_RATIO*100:.1f}% val, {TEST_RATIO*100:.1f}% test")
        print(f"Training samples used: {history['data_split_info']['actual_train_used']}")
        print(f"Total training samples available: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print("Model parameters saved:")
        print(f"  - Classical NN parameters")
        print(f"  - Quantum weights shape: {final_weights.shape}")
        print(f"  - Number of epochs: {len(history['epoch_losses'])}")
        if len(history['epoch_losses']) > 0:
            print(f"  - Final train loss: {history['epoch_losses'][-1]['avg_loss']:.6f}")
        if len(history['val_losses']) > 0:
            print(f"  - Final validation loss: {history['val_losses'][-1]['val_loss']:.6f}")
        # 显示保存的文件
        print("\nSaved files in 'model_parameters' directory:")
        saved_files = os.listdir('model_parameters')
        for file in sorted(saved_files):
            print(f"  - {file}")
    else:
        print("Training failed!")