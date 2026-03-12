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