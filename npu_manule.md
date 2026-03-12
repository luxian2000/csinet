# ALPHAQUBIT 代码中使用 NPU 的方法与步骤

本文档基于当前仓库代码与脚本总结，目标是回答两件事：
1. 代码里是如何接入 NPU 的。
2. 实际运行时应该按什么步骤做。

适用范围：Ascend NPU + PyTorch (`torch_npu`)。

## 1. 核心设计模式

仓库里 NPU 接入基本遵循以下模式。

### 1.1 导入顺序与可用性检测

常见写法：先尝试 `import torch_npu`，再检查 `torch.npu.is_available()`。

```python
try:
    import torch_npu
    has_npu = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    has_npu = False
```

参考实现：
- `ai_models/train.py`
- `ai_models/model_mla.py`
- `ai_models/decode.py`
- `ai_models/fine_tune_npz.py`

### 1.2 设备选择策略

仓库中常见三种策略：
- `auto`：优先 NPU，再 CUDA，再 CPU。
- 显式 `--npu`：如果可用则强制走 NPU。
- 多进程多卡：每个子进程绑定一张物理卡。

### 1.3 多卡绑定策略

多 NPU 训练脚本普遍通过环境变量限制进程可见设备，再在进程内使用本地设备索引。

典型方式：
```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
torch.npu.set_device(0)  # 映射后的本地设备 0
```

参考：
- `run_multi_npu_training.py`
- `run_training_all.py`（会自动处理可见设备和进程环境）
- `run_finetune_multi_npu.py`

## 2. NPU 相关脚本分层

为避免入口混乱，建议按层理解。

### 2.1 主流程（推荐优先）

1. `run_training_all.py`
- 用于批量训练所有 `.npz` 数据集。
- `--npu` 时可自动发现设备并并行调度。

2. `run_finetune_all.py`
- 用于批量 fine-tune。
- 通过 `--npu` 传递到 `ai_models/fine_tune_npz.py`。

3. `run_multi_npu_training.py`
- 面向多 NPU、内存优化、任务队列式训练。
- 提供 `--npu-ids`、`--memory-efficient`、`--skip-existing`。

### 2.2 单模型/单数据训练入口

1. `ai_models/train.py`
- 配置驱动训练入口。
- 关键参数：`--config`、`--npu`、`--mla`。

2. `ai_models/model_mla.py`
- 支持 `--npu`、`--device_index`，适合被调度脚本调用。

3. `ai_models/fine_tune_npz.py`
- 单个 NPZ 的 fine-tune 入口。
- 关键参数：`--data`、`--pretrained`、`--npu`、`--mla`。

### 2.3 推理与评估

1. `ai_models/decode.py`
- 使用 `--device npu` 或默认 `auto` 选择 NPU。
- 输入模型与数据，输出指标和可选预测数组。

2. `test_finetuned_models.py`
- 支持 `--npu` 做批量推理评测。

### 2.4 远程运维辅助脚本

- `remote_scripts/quick_npu_test.sh`：快速环境体检。
- `remote_scripts/run_qnn_npu_test.sh`：远程完整测试流程。
- `remote_scripts/check_generated_data.sh`：检查生成数据完整性。

## 3. 环境准备步骤（实操）

### 步骤 1：激活 Python 环境

```bash
conda activate alphaqubit
```

### 步骤 2：验证 NPU 驱动与工具链

```bash
npu-smi info
```

如果服务器是 Ascend 环境，通常还需要：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 步骤 3：验证 Python 侧 NPU

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

若输出 `True` 且设备数大于 0，说明 Python 侧可用。

## 4. 训练步骤（推荐顺序）

## 4.1 批量预训练（推荐）

```bash
python run_training_all.py --npu --epochs 20 --batch-size 16
```

说明：
- 默认会在 `pretrain_data/`（必要时回退到 `simulated_data/`）找数据。
- 输出模型在 `ai_models/models/`。

可选：
```bash
python run_training_all.py --npu --data-root pretrain_data --max-samples 2048
```

## 4.2 批量微调（通用）

```bash
python run_finetune_all.py \
  --data-dir google_finetune_data/finetune \
  --pretrained alphaqubit_pauli_plus.pth \
  --npu \
  --epochs 30 \
  --batch-size 128 \
  --skip-existing
```

输出目录默认：`finetuned_models/`。

## 4.3 多 NPU 内存优化微调（大任务推荐）

```bash
python run_multi_npu_training.py \
  --data-dir google_finetune_data/finetune \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --memory-efficient \
  --skip-existing
```

`--memory-efficient` 会自动调整：
- 更小 batch
- 更高梯度累积
- 更低 dataloader worker
- 更积极的内存释放

## 5. 推理与评估步骤

### 5.1 单数据解码

```bash
python ai_models/decode.py \
  --model finetuned_models/finetuned_xxx.pth \
  --data output/xxx.npz \
  --device npu
```

可加：
- `--batch-size`
- `--output`
- `--predictions`
- `--mla`（模型结构匹配时）

### 5.2 批量测试 fine-tuned 模型

```bash
python test_finetuned_models.py --model-dir finetuned_models --npu
```

## 6. 代码中可复用的 NPU 编程要点

1. 设备兜底逻辑
- NPU 不可用时自动回退 CPU/CUDA，避免任务直接崩溃。

2. 多进程隔离
- 每个 NPU 一个独立进程，减少初始化冲突和内存污染。

3. 环境变量控制
- 常见变量：
  - `ASCEND_RT_VISIBLE_DEVICES`
  - `ASCEND_VISIBLE_DEVICES`
  - `PYTORCH_NPU_ALLOC_CONF`

4. 周期性清理缓存
- 训练循环中定期 `torch.npu.empty_cache()` + `gc.collect()`。

5. 混合精度
- 部分脚本用 `amp` 或 `GradScaler` 降低显存/NPU 内存压力。

## 7. 常见问题与处理

### 7.1 `torch_npu` 导入失败

现象：`ImportError: No module named torch_npu`

处理：
- 安装匹配版本的 `torch_npu`。
- 确认 CANN 与 PyTorch 版本兼容。

### 7.2 NPU 初始化失败/500001

处理建议：
1. 重新 `source` Ascend 环境。
2. 检查 `npu-smi info`。
3. 用 `system_npu_diagnosis.py` 收集诊断信息。

### 7.3 OOM（内存不足）

优先动作：
1. 使用 `--memory-efficient`。
2. 降低 `--batch-size`，提高 `--grad-accum`。
3. 减少并行卡数或并发任务数。

## 8. 推荐的最小可执行流程

```bash
# 1) 环境检测
python -c "import torch, torch_npu; print('NPU:', torch.npu.is_available(), 'count=', torch.npu.device_count())"

# 2) 预训练（并行调度）
python run_training_all.py --npu --epochs 2 --batch-size 8 --max-samples 1024

# 3) 微调（小规模冒烟）
python run_finetune_all.py --npu --epochs 1 --batch-size 32 --limit 2

# 4) 推理验证
python ai_models/decode.py --model finetuned_models/finetuned_xxx.pth --data output/xxx.npz --device npu
```

## 9. 入口选择建议

- 想要稳妥批处理：`run_training_all.py` + `run_finetune_all.py`
- 想要极限多卡与内存控制：`run_multi_npu_training.py`
- 想针对单个数据快速调试：`ai_models/fine_tune_npz.py --npu`
- 想单次推理验证：`ai_models/decode.py --device npu`

---

如果后续需要，我可以在此文档基础上再补一版“按你当前机器（本地 macOS + 远程 Ascend）分场景命令清单”，把本地与远程命令拆开。

## 10. 如何把数据加载到 NPU（代码模式说明）

下面列出仓库中常见且推荐的几种把数据移动到 NPU 的写法，并说明为什么这样做：

- 直接把模型和张量放到 `device`：

```python
# 选择设备（示例：'npu:0' 或由 auto 检测）
device = torch.device('npu:0')
model = model.to(device)

# 从 numpy 创建并移动到 NPU
X = torch.from_numpy(X_np.astype(np.float32)).to(device)
# 或者直接指定 device（少一次中间复制）
X = torch.tensor(X_np, dtype=torch.float32, device=device)
```

- 在训练/推理循环中，把 batch 张量显式 `.to(device)`：

```python
for xb, basis, mask, yb in dataloader:
    xb = xb.to(device, non_blocking=True)   # 非阻塞拷贝（若 DataLoader 支持）
    basis = basis.to(device)
    mask = mask.to(device)
    yb = yb.to(device)
    logits = model(xb, basis, mask)
```

- 对于基于 chunk 的流程（仓库中 `run_multi_npu_training.py` 的常见做法）：
  - 先把一个 chunk 从磁盘/NPZ 加载到主内存（numpy），再按 mini-batch 切片并把切片移动到 `device`：

```python
# load_chunk 返回 numpy->torch 的批量张量（在 CPU 上）
x_chunk, basis_chunk, labels_chunk = dataset.load_chunk(start, end)

# 在 mini-batch 内把切片迁移到 NPU
for batch_local in batches:
    x = x_chunk[batch_local].to(device)
    basis = basis_chunk[batch_local].to(device)
    labels = labels_chunk[batch_local].to(device)
    out = model(x, basis, final_mask)
```

说明要点：
- 优先把模型 `model.to(device)` 后再移动输入张量，避免 device mismatch。
- 使用 `non_blocking=True` 可在 DataLoader pin_memory 支持时加速拷贝，但 Ascend NPU 与 `pin_memory` 的行为可能与 CUDA 不同，请以目标环境测试为准。
- 推荐在每个 batch/chunk 处理后显式删除临时张量并 `gc.collect()`，并在需要时调用 `torch.npu.empty_cache()` 以降低 OOM 风险（仓库多个脚本已采用此策略）。

## 11. NPUk（单卡或第 k 张 NPU）上的数据如何运算（分片与计算流程）

仓库里对“多 NPU”场景的处理遵循两条主线：进程级隔离（每个进程绑定一张 NPU）或在单进程内手动分片。常见实现细节如下：

- 进程绑定设备（每个进程把可见设备映射为本地 `npu:0`）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=3   # 让当前进程只看到物理卡 3
python train_single_npu.py --npu
```

在进程中使用：

```python
torch.npu.set_device(0)  # 进程内把映射后的本地设备 0 作为计算设备
device = torch.device('npu:0')
```

- 单进程手动分片（仓库 `run_multi_npu_training.py` 范例）：
  - 把整个训练集按 chunk 划分，每个 NPU 负责若干 chunk；在每个 NPU 的工作进程内再把 chunk 按 mini-batch 切分并移动到本地 device。
  - 关键实现片段：先 `dataset.load_chunk()` 得到 chunk（CPU 内存），然后 `x_chunk[local_indices].to(device)` 把子批次移动到 NPU。见仓库实现示例：`run_multi_npu_training.py`。

- 计算与同步：
  - 正常前向/反向在 NPU 上进行（model, inputs 都在 `npu:0`）。
  - 若使用混合精度，按常规使用 `torch.cuda.amp` 的 API（仓库里用 `autocast()` + `GradScaler` 样式），需要确认 `torch_npu` 对应的 autocast/amp 支持；若不支持，使用纯 FP32 或库提供的 NPU AMP 方案。
  - 在步骤结束或需要保证结果可读时可调用 `torch.npu.synchronize()` 以等待设备完成操作。

- 梯度累积与参数更新：
  - 在 chunk 内按 `gradient_accumulation_steps` 累积梯度并在合适时机做 `optimizer.step()`。
  - 若每个 NPU 是独立进程，则各进程保存本地模型（或使用分布式通信同步参数，仓库中对 HCCL/分布式有注释但多处采用单卡/独立进程策略以规避 HCCL 问题）。

要点总结：
- NPUk 上的“运算”就是：把模型和该 NPU 的输入张量移到 `npu:0`（进程内部视角），执行前向/反向、累积梯度、更新参数并周期性清理内存。
- 多卡并行最稳妥的方式是「一个进程 + 一张卡」，通过环境变量（`ASCEND_RT_VISIBLE_DEVICES`）与 `torch.npu.set_device(0)` 完成绑定。

如果你希望，我可以把上述节扩展为针对你当前目标机器（例如远程 Ascend 服务器）的逐步命令清单，或把常用代码片段提取为可复用的 helper 函数并提交为补丁。