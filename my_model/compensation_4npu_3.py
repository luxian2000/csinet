import argparse
from contextlib import nullcontext
import json
import os
import time
from pathlib import Path
import warnings

import numpy as np
import scipy.io as sio
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    HAS_TORCH_NPU = False

# torch_npu on some releases emits this deprecation warning during backward.
warnings.filterwarnings(
    "ignore",
    message=r".*AutoNonVariableTypeMode is deprecated.*",
    category=UserWarning,
)


torch.manual_seed(42)
np.random.seed(42)

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2
IMG_TOTAL = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

COMPRESSION_RATES = {
    1 / 4: 512,
    1 / 16: 128,
    1 / 32: 64,
    1 / 64: 32,
}


class QuantumCompensationBlock(nn.Module):
    """
    可在 NPU 上端到端训练。
    """

    def __init__(self, n_qubits=16, n_layers=2, window_size=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        self.state_dim = 1 << n_qubits
        if n_qubits != window_size * window_size:
            raise ValueError("n_qubits must equal window_size*window_size for fold/unfold reconstruction.")

        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.input_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.backend = "torch_statevector_gates"

        # bit_table[q, basis] in {0.0, 1.0}: qubit q 在给定基态下的比特值。
        basis = torch.arange(self.state_dim, dtype=torch.long)
        qubits = torch.arange(n_qubits, dtype=torch.long).unsqueeze(1)
        bit_table = ((basis.unsqueeze(0) >> qubits) & 1).to(torch.float32)
        self.register_buffer("bit_table", bit_table, persistent=False)

    @staticmethod
    def _split_pairs(state, qubit, n_qubits):
        high = 1 << (n_qubits - qubit - 1)
        low = 1 << qubit
        return state.view(state.shape[0], high, 2, low), high, low

    def _apply_h(self, real, imag, qubit):
        real_view, _, _ = self._split_pairs(real, qubit, self.n_qubits)
        imag_view, _, _ = self._split_pairs(imag, qubit, self.n_qubits)

        a0r = real_view[:, :, 0, :]
        a1r = real_view[:, :, 1, :]
        a0i = imag_view[:, :, 0, :]
        a1i = imag_view[:, :, 1, :]

        inv_sqrt2 = 2.0 ** -0.5
        out0r = (a0r + a1r) * inv_sqrt2
        out1r = (a0r - a1r) * inv_sqrt2
        out0i = (a0i + a1i) * inv_sqrt2
        out1i = (a0i - a1i) * inv_sqrt2

        real_out = torch.stack([out0r, out1r], dim=2).reshape(real.shape)
        imag_out = torch.stack([out0i, out1i], dim=2).reshape(imag.shape)
        return real_out, imag_out

    def _apply_ry(self, real, imag, qubit, theta):
        real_view, _, _ = self._split_pairs(real, qubit, self.n_qubits)
        imag_view, _, _ = self._split_pairs(imag, qubit, self.n_qubits)

        a0r = real_view[:, :, 0, :]
        a1r = real_view[:, :, 1, :]
        a0i = imag_view[:, :, 0, :]
        a1i = imag_view[:, :, 1, :]

        if theta.dim() == 0:
            c = torch.cos(theta * 0.5).view(1, 1, 1)
            s = torch.sin(theta * 0.5).view(1, 1, 1)
        else:
            c = torch.cos(theta * 0.5).view(-1, 1, 1)
            s = torch.sin(theta * 0.5).view(-1, 1, 1)

        out0r = c * a0r - s * a1r
        out1r = s * a0r + c * a1r
        out0i = c * a0i - s * a1i
        out1i = s * a0i + c * a1i

        real_out = torch.stack([out0r, out1r], dim=2).reshape(real.shape)
        imag_out = torch.stack([out0i, out1i], dim=2).reshape(imag.shape)
        return real_out, imag_out

    def _apply_crz(self, real, imag, control, target, theta):
        bit_control = self.bit_table[control]
        bit_target = self.bit_table[target]

        mask10 = bit_control * (1.0 - bit_target)
        mask11 = bit_control * bit_target

        c = torch.cos(theta * 0.5)
        s = torch.sin(theta * 0.5)

        phase_real = 1.0 + (c - 1.0) * (mask10 + mask11)
        phase_imag = (-s) * mask10 + s * mask11

        new_real = real * phase_real.unsqueeze(0) - imag * phase_imag.unsqueeze(0)
        new_imag = real * phase_imag.unsqueeze(0) + imag * phase_real.unsqueeze(0)
        return new_real, new_imag

    def _z_expectation(self, real, imag):
        probs = real * real + imag * imag
        z_sign = 1.0 - 2.0 * self.bit_table
        return probs @ z_sign.transpose(0, 1)

    def _torch_quantum_forward(self, inputs):
        # 真实门级状态向量模拟：编码(H+RY) -> 变分层(CRZ ring + RY) -> 测量<Z>。
        batch = inputs.shape[0]
        device = inputs.device
        dtype = inputs.dtype

        basis0 = torch.zeros(batch, device=device, dtype=torch.long)
        real = F.one_hot(basis0, num_classes=self.state_dim).to(dtype=dtype)
        imag = torch.zeros_like(real)

        enc_angles = torch.tanh(inputs * self.input_scale) * np.pi
        for q in range(self.n_qubits):
            real, imag = self._apply_h(real, imag, q)
            real, imag = self._apply_ry(real, imag, q, enc_angles[:, q])

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                target = (q + 1) % self.n_qubits
                real, imag = self._apply_crz(real, imag, q, target, self.weights_crz[layer, q])
            for q in range(self.n_qubits):
                real, imag = self._apply_ry(real, imag, q, self.weights_ry[layer, q])

        z_exp = self._z_expectation(real, imag)
        return torch.clamp(z_exp, -1.0, 1.0)

    def forward(self, x):
        # x: [batch, encoded_dim]
        if x.dim() != 2:
            raise ValueError(f"QuantumCompensationBlock expects 2D latent input [B, D], got shape={tuple(x.shape)}")

        batch, dim = x.shape
        latent_h = 4
        if dim % latent_h != 0:
            raise ValueError(f"encoded dim must be divisible by {latent_h}, got {dim}")
        latent_w = dim // latent_h
        if latent_w % self.window_size != 0:
            raise ValueError(f"latent width {latent_w} must be divisible by window_size {self.window_size}")

        original_device = x.device

        x_map = x.reshape(batch, 1, latent_h, latent_w)
        # 将输入移动到量子补偿参数所在设备。
        x_proc = x_map.to(self.weights_crz.device)

        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        patches = unfold(x_proc)  # [batch, 16, num_patches]
        num_patches = patches.shape[-1]

        # 对 encoded_dim=32 的典型 latent shape=(4,8)，4x4 unfold 应得到 2 个 patches。
        if latent_h == 4 and latent_w == 8 and num_patches != 2:
            raise RuntimeError(f"Expected 2 patches for latent shape (4,8) with 4x4 unfold, got {num_patches}")

        total_samples = batch * num_patches
        all_inputs = patches.permute(0, 2, 1).reshape(total_samples, self.n_qubits)
        all_inputs = torch.tanh(all_inputs) * np.pi

        all_outputs = []
        # 按块处理以控制显存/内存峰值。
        chunk_size = 32
        for start in range(0, total_samples, chunk_size):
            end = min(total_samples, start + chunk_size)
            batch_inp = all_inputs[start:end].to(self.weights_crz.device)
            q_out = self._torch_quantum_forward(batch_inp)
            all_outputs.append(q_out)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_outputs = all_outputs.reshape(batch, num_patches, self.n_qubits).permute(0, 2, 1)

        fold = nn.Fold(output_size=(latent_h, latent_w), kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        output = fold(all_outputs).float()
        output = output.reshape(batch, dim)
        return output.to(original_device)


class CsiNetEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.lr1 = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(IMG_TOTAL, encoded_dim)

    def forward(self, x):
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc_encode(x)
        return x


class QuantumCompensatedDecoder(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        alpha = float(np.clip(alpha, 1e-4, 1 - 1e-4))
        self.alpha_logit = nn.Parameter(torch.tensor(np.log(alpha / (1 - alpha)), dtype=torch.float32))
        self.encoded_dim = encoded_dim
        self.latent_h = 4
        if encoded_dim % self.latent_h != 0:
            raise ValueError(f"encoded_dim must be divisible by {self.latent_h}, got {encoded_dim}")
        self.latent_w = encoded_dim // self.latent_h

        self.fc_decode = nn.Linear(encoded_dim, IMG_TOTAL)
        self.quantum_comp = QuantumCompensationBlock(n_qubits=16, n_layers=2, window_size=4)
        self.quantum_upsample = nn.Upsample(size=(IMG_HEIGHT, IMG_WIDTH), mode="bilinear", align_corners=False)
        self.quantum_proj = nn.Conv2d(1, IMG_CHANNELS, kernel_size=1)

        self.residual_blocks = nn.ModuleList([self._make_residual_block(IMG_CHANNELS) for _ in range(5)])

        self.output_conv = nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _make_residual_block(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, s):
        batch_size = s.shape[0]
        x = self.fc_decode(s)
        x = x.reshape(batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        q_latent = self.quantum_comp(s)
        q_latent_map = q_latent.reshape(batch_size, 1, self.latent_h, self.latent_w)
        q_up = self.quantum_upsample(q_latent_map)
        comp = self.quantum_proj(q_up)

        alpha = torch.sigmoid(self.alpha_logit)
        fused = (1 - alpha) * x + alpha * comp

        residual = fused
        for block in self.residual_blocks:
            residual = F.leaky_relu(residual + block(residual))

        out = self.sigmoid(self.output_conv(residual))
        return out


class CsiNetQuantumCompensated(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.encoder = CsiNetEncoder(encoded_dim)
        self.decoder = QuantumCompensatedDecoder(encoded_dim, alpha)

    def forward(self, x):
        s = self.encoder(x)
        x_hat = self.decoder(s)
        return x_hat


def load_data(data_path="/root/work/luxian/csinet/data"):
    # 所有数据统一使用 outdoor 数据集
    x_train = sio.loadmat(f"{data_path}/DATA_Htrainout.mat")["HT"].astype(np.float32)
    x_val = sio.loadmat(f"{data_path}/DATA_Hvalout.mat")["HT"].astype(np.float32)
    x_test = sio.loadmat(f"{data_path}/DATA_Htestout.mat")["HT"].astype(np.float32)
    x_test_freq = sio.loadmat(f"{data_path}/DATA_HtestFout_all.mat")["HF_all"].astype(np.complex128)

    def preprocess(data):
        bs = data.shape[0]
        return data.reshape(bs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)
    x_test_freq = x_test_freq.reshape(-1, IMG_HEIGHT, 125)

    return x_train, x_val, x_test, x_test_freq


def calculate_nmse_rho(x_test, x_hat, x_test_freq):
    batch_size = x_test.shape[0]

    x_test_real = x_test[:, 0, :, :].reshape(batch_size, -1)
    x_test_imag = x_test[:, 1, :, :].reshape(batch_size, -1)
    x_test_c = (x_test_real - 0.5) + 1j * (x_test_imag - 0.5)

    x_hat_real = x_hat[:, 0, :, :].reshape(batch_size, -1)
    x_hat_imag = x_hat[:, 1, :, :].reshape(batch_size, -1)
    x_hat_c = (x_hat_real - 0.5) + 1j * (x_hat_imag - 0.5)

    x_hat_f = x_hat_c.reshape(batch_size, IMG_HEIGHT, IMG_WIDTH)
    x_hat_full = np.fft.fft(
        np.concatenate((x_hat_f, np.zeros((batch_size, IMG_HEIGHT, 257 - IMG_WIDTH))), axis=2),
        axis=2,
    )[:, :, 0:125]

    n1 = np.sqrt(np.sum(np.abs(x_test_freq) ** 2, axis=(1, 2)))
    n2 = np.sqrt(np.sum(np.abs(x_hat_full) ** 2, axis=(1, 2)))
    aa = np.abs(np.sum(np.conj(x_test_freq) * x_hat_full, axis=(1, 2)))
    rho = aa / (n1 * n2 + 1e-10)

    power = np.sum(np.abs(x_test_c) ** 2, axis=1)
    mse = np.sum(np.abs(x_test_c - x_hat_c) ** 2, axis=1)
    nmse = 10 * np.log10(np.mean(mse / (power + 1e-10)))

    return float(nmse), float(np.mean(rho))


def _move_model_devices(model, device):
    # 将模型整体迁移到目标 device。
    model = model.to(device)
    model.decoder.quantum_comp = model.decoder.quantum_comp.to(device)
    return model


def _amp_context(device):
    # NPU-only runtime: keep full precision and avoid CUDA-specific AMP checks.
    return nullcontext()


def _inference_context():
    # Prefer the new user-facing API for inference-only workloads.
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


class _NoOpGradScaler:
    """Fallback scaler for environments without amp GradScaler support."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


def _build_grad_scaler(device):
    # NPU-only runtime: disable scaler to avoid CUDA/AMP dependency.
    return _NoOpGradScaler()


def _setup_distributed():
    # torchrun 会注入这些环境变量；单卡运行时默认回退到 world_size=1。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")
    return distributed, world_size, rank, local_rank


def _cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank):
    return int(rank) == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    x_test_np,
    x_test_freq,
    epochs,
    lr,
    device,
    best_model_path,
    latest_checkpoint_path,
    best_checkpoint_path,
    save_dir=None,
    resume_from="",
    distributed=False,
    rank=0,
    train_sampler=None,
):
    model = _move_model_devices(model, device)
    main_process = _is_main_process(rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    scaler = _build_grad_scaler(device)

    best_val_loss = float("inf")
    start_epoch = 0
    train_losses = []
    val_losses = []
    lr_history = []

    if resume_from and str(resume_from).strip():
        resume_path = Path(resume_from).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            _unwrap_model(model).load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "scaler_state" in checkpoint and hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(checkpoint["scaler_state"])

            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            train_losses = list(checkpoint.get("train_losses", []))
            val_losses = list(checkpoint.get("val_losses", []))
            lr_history = list(checkpoint.get("lr_history", []))
            print(
                f"Resumed full checkpoint from {resume_path} at epoch={start_epoch}, best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
        else:
            # 兼容旧格式：仅模型参数。
            _unwrap_model(model).load_state_dict(checkpoint)
            print(f"Loaded model weights only from {resume_path}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = 0.0
        num_train_batches = 0

        for (data,) in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with _amp_context(device):
                output = model(data)
                loss = criterion(output, data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            num_train_batches += 1

        if distributed:
            loss_stats = torch.tensor([train_loss, float(num_train_batches)], device=device, dtype=torch.float64)
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            total_loss = loss_stats[0].item()
            total_batches = max(1.0, loss_stats[1].item())
            avg_train_loss = total_loss / total_batches
        else:
            avg_train_loss = train_loss / max(1, len(train_loader))

        train_losses.append(avg_train_loss)

        avg_val_loss = float("nan")
        epoch_metrics = {}

        # 只在主进程进行验证与测试指标计算，避免每个 rank 重复评估。
        if main_process:
            model.eval()
            val_loss = 0.0
            with _inference_context():
                for (data,) in val_loader:
                    data = data.to(device, non_blocking=True)
                    with _amp_context(device):
                        output = model(data)
                        loss = criterion(output, data)
                    val_loss += loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))

            # Compute test metrics each epoch (NMSE & rho if frequency data available)
            test_outputs = []
            with _inference_context():
                for (data,) in test_loader:
                    data = data.to(device, non_blocking=True)
                    with _amp_context(device):
                        out = model(data)
                    test_outputs.append(out.float().cpu().numpy())
            x_hat = np.concatenate(test_outputs, axis=0)

            if x_test_freq is not None:
                try:
                    nmse, rho = calculate_nmse_rho(x_test_np, x_hat, x_test_freq)
                    epoch_metrics["nmse_db"] = nmse
                    epoch_metrics["rho"] = rho
                except Exception:
                    epoch_metrics["nmse_db"] = None
                    epoch_metrics["rho"] = None
            else:
                epoch_metrics["test_mse"] = float(np.mean((x_hat - x_test_np) ** 2))

        val_losses.append(avg_val_loss)

        # 每个 epoch 结束时都打印 NMSE 和 rho（无频域标签时显示 N/A）
        nmse_value = epoch_metrics.get("nmse_db", None) if main_process else None
        rho_value = epoch_metrics.get("rho", None) if main_process else None

        nmse_str = f"{nmse_value:.2f} dB" if isinstance(nmse_value, (int, float)) else "N/A"
        rho_str = f"{rho_value:.4f}" if isinstance(rho_value, (int, float)) else "N/A"

        # Log epoch summary to console and to snapshot file if provided
        current_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]

        metrics_str = f"NMSE: {nmse_str}, Rho: {rho_str}"
        if "test_mse" in epoch_metrics:
            metrics_str += f", Test MSE: {epoch_metrics['test_mse']:.6f}"

        summary_line = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f} "
            f"LR: {current_lrs} "
            f"{metrics_str}"
        )
        if main_process:
            print(summary_line, flush=True)
        if main_process and save_dir is not None:
            try:
                # 第一个 epoch 时覆盖写入，后续 epoch 追加
                write_mode = "w" if epoch == 0 else "a"
                with open(Path(save_dir) / "log_snapshot.txt", write_mode, encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    f.write(summary_line + "\n")
                    f.write("----\n")
            except Exception:
                pass
        scheduler.step()

        current_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
        lr_history.append(current_lrs)

        if main_process and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            unwrapped = _unwrap_model(model)
            torch.save(unwrapped.state_dict(), best_model_path)
            best_checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_state": unwrapped.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_history": lr_history,
            }
            torch.save(best_checkpoint, best_checkpoint_path)
            print(f"Saved best model (epoch {epoch+1}): {best_model_path}", flush=True)

        if main_process:
            unwrapped = _unwrap_model(model)
            latest_checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_state": unwrapped.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_history": lr_history,
            }
            torch.save(latest_checkpoint, latest_checkpoint_path)

    return train_losses, val_losses, lr_history


def make_sanity_loaders(batch_size=32, train_samples=64, val_samples=32, test_samples=32):
    x_train = torch.rand(train_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_val = torch.rand(val_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_test = torch.rand(test_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, val_loader, test_loader, x_test.numpy(), None


def run(args):
    # 强制使用 NPU，否则报错。
    if not HAS_TORCH_NPU:
        raise RuntimeError(
            "NPU is not available. Install torch_npu and ensure Ascend runtime is configured correctly."
        )

    distributed, world_size, rank, local_rank = _setup_distributed()
    main_process = _is_main_process(rank)

    # 默认不强制 world size，避免 python 直接启动时报错；可通过 --strict-world-size 开启严格校验。
    if world_size != args.expected_world_size:
        msg = (
            f"Expected WORLD_SIZE={args.expected_world_size}, but got WORLD_SIZE={world_size}. "
            f"Use torchrun --nproc_per_node={args.expected_world_size} for multi-NPU training."
        )
        if args.strict_world_size:
            raise RuntimeError(msg)
        if main_process:
            print(f"[WARN] {msg} Falling back to current WORLD_SIZE={world_size}.", flush=True)

    if hasattr(torch.npu, "set_device"):
        torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    print(f"Using device: {device}, rank={rank}, world_size={world_size}")

    model = CsiNetQuantumCompensated(encoded_dim=args.encoded_dim, alpha=args.alpha).to(device)
    print(f"Quantum backend: {model.decoder.quantum_comp.backend}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    if args.sanity:
        train_loader, val_loader_tmp, test_loader_tmp, x_test_np_tmp, x_test_freq_tmp = make_sanity_loaders(
            batch_size=args.batch_size,
            train_samples=args.sanity_train_samples,
            val_samples=args.sanity_val_samples,
            test_samples=args.sanity_test_samples,
        )
        # sanity 场景不走分布式切分，保持简单。
        train_sampler = None
        val_loader = val_loader_tmp if main_process else None
        test_loader = test_loader_tmp if main_process else None
        x_test_np = x_test_np_tmp if main_process else None
        x_test_freq = x_test_freq_tmp if main_process else None
    else:
        x_train, x_val, x_test, x_test_freq_full = load_data(args.data_path)
        # Optionally subset the real datasets to requested sizes
        if getattr(args, "train_samples", 0) and args.train_samples > 0:
            x_train = x_train[: args.train_samples]
        if getattr(args, "val_samples", 0) and args.val_samples > 0:
            x_val = x_val[: args.val_samples]
        if getattr(args, "test_samples", 0) and args.test_samples > 0:
            x_test = x_test[: args.test_samples]
            x_test_freq_full = x_test_freq_full[: args.test_samples]

        x_train_t = torch.FloatTensor(x_train)
        x_val_t = torch.FloatTensor(x_val)
        x_test_t = torch.FloatTensor(x_test)

        train_dataset = TensorDataset(x_train_t)
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=train_sampler,
                pin_memory=False,
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=False,
            )

        # 验证/测试只在主进程执行，避免重复评估开销。
        if main_process:
            val_loader = DataLoader(TensorDataset(x_val_t), batch_size=args.batch_size, shuffle=False, pin_memory=False)
            test_loader = DataLoader(TensorDataset(x_test_t), batch_size=args.batch_size, shuffle=False, pin_memory=False)
            x_test_np = x_test
            x_test_freq = x_test_freq_full
        else:
            val_loader = None
            test_loader = None
            x_test_np = None
            x_test_freq = None

    # Determine save directory: priority --outputdir, then deprecated --output-dir,
    # otherwise default out_100k_3.
    out_arg = getattr(args, "outputdir", "") or getattr(args, "output_dir", "")
    if out_arg and str(out_arg).strip():
        save_dir = Path(out_arg).expanduser().resolve()
    else:
        save_dir = Path(__file__).resolve().parent / "out_100k_3"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() if args.run_tag else ""
    if not run_tag:
        run_tag = time.strftime("%Y%m%d_%H%M%S")

    suffix = f"{args.envir}_dim{args.encoded_dim}_{run_tag}"
    best_model_path = save_dir / f"best_model_quantum_npu_{suffix}.pth"
    latest_checkpoint_path = save_dir / f"latest_checkpoint_quantum_npu_{suffix}.pth"
    best_checkpoint_path = save_dir / f"best_checkpoint_quantum_npu_{suffix}.pth"

    try:
        start = time.time()
        train_losses, val_losses, lr_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            x_test_np=x_test_np,
            x_test_freq=x_test_freq,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            best_model_path=best_model_path,
            latest_checkpoint_path=latest_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            save_dir=save_dir,
            resume_from=args.resume_from,
            distributed=distributed,
            rank=rank,
            train_sampler=train_sampler,
        )
        train_time = time.time() - start
        if main_process:
            print(f"Training time: {train_time:.2f}s")

        if distributed:
            dist.barrier()

        if main_process:
            model_to_eval = _unwrap_model(model)
            if best_model_path.exists():
                model_to_eval.load_state_dict(torch.load(best_model_path, map_location=device))
            else:
                print("Best model file not found after training; using latest in-memory model for evaluation.", flush=True)
            model_to_eval = _move_model_devices(model_to_eval, device)
            model_to_eval.eval()

            outputs = []
            infer_start = time.time()
            with _inference_context():
                for (data,) in test_loader:
                    data = data.to(device, non_blocking=True)
                    output = model_to_eval(data)
                    outputs.append(output.float().cpu().numpy())
            infer_end = time.time()

            x_hat = np.concatenate(outputs, axis=0)
            inference_time_per_sample = (infer_end - infer_start) / x_hat.shape[0]
            print(f"Inference time per sample: {inference_time_per_sample:.6f}s")

            metrics = {}
            if x_test_freq is not None:
                nmse, rho = calculate_nmse_rho(x_test_np, x_hat, x_test_freq)
                metrics["nmse_db"] = nmse
                metrics["cosine_similarity"] = rho
                print(f"NMSE: {nmse:.2f} dB")
                print(f"Cosine similarity: {rho:.4f}")
            else:
                sanity_mse = float(np.mean((x_hat - x_test_np) ** 2))
                metrics["sanity_mse"] = sanity_mse
                print(f"Sanity MSE: {sanity_mse:.6f}")

            final_model_path = save_dir / f"csinet_quantum_npu_{suffix}.pth"
            train_loss_path = save_dir / f"train_loss_quantum_npu_{suffix}.csv"
            val_loss_path = save_dir / f"val_loss_quantum_npu_{suffix}.csv"
            lr_path = save_dir / f"lr_history_quantum_npu_{suffix}.csv"

            torch.save(model_to_eval.state_dict(), final_model_path)
            np.savetxt(train_loss_path, train_losses, delimiter=",")
            np.savetxt(val_loss_path, val_losses, delimiter=",")
            np.savetxt(lr_path, np.array(lr_history), delimiter=",")

            summary = {
                "args": vars(args),
                "device": str(device),
                "rank": int(rank),
                "world_size": int(world_size),
                "quantum_backend": model_to_eval.decoder.quantum_comp.backend,
                "train_time_sec": float(train_time),
                "inference_time_per_sample_sec": float(inference_time_per_sample),
                "train_samples": int(len(train_loader.dataset)),
                "val_samples": int(len(val_loader.dataset)),
                "test_samples": int(len(test_loader.dataset)),
                "best_model_path": str(best_model_path),
                "latest_checkpoint_path": str(latest_checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "final_model_path": str(final_model_path),
                "train_loss_csv": str(train_loss_path),
                "val_loss_csv": str(val_loss_path),
                "lr_history_csv": str(lr_path),
                "metrics": metrics,
            }

            summary_path = save_dir / f"run_summary_quantum_npu_{suffix}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"Saved best model: {best_model_path}")
            print(f"Saved latest checkpoint: {latest_checkpoint_path}")
            print(f"Saved best checkpoint: {best_checkpoint_path}")
            print(f"Saved final model: {final_model_path}")
            print(f"Saved train loss: {train_loss_path}")
            print(f"Saved val loss: {val_loss_path}")
            print(f"Saved lr history: {lr_path}")
            print(f"Saved run summary: {summary_path}")
    finally:
        _cleanup_distributed(distributed)


def build_parser():
    parser = argparse.ArgumentParser(description="CsiNet quantum-classical hybrid (NPU-oriented)")
    parser.add_argument("--envir", type=str, default="outdoor", choices=["outdoor"], help="data environment (outdoor only)")
    parser.add_argument("--data-path", type=str, default="/root/work/luxian/csinet/data")
    parser.add_argument("--encoded-dim", type=int, default=32, choices=sorted(set(COMPRESSION_RATES.values())))
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--expected-world-size", type=int, default=4, help="expected number of NPU processes (default: 4)")
    parser.add_argument(
        "--strict-world-size",
        action="store_true",
        help="if set, mismatch between WORLD_SIZE and --expected-world-size will raise an error",
    )
    parser.add_argument("--output-dir", type=str, default="", help="(deprecated) output directory; prefer --outputdir")
    parser.add_argument("--outputdir", type=str, default="out_100k_3", help="output directory for saved artifacts (default: out_100k_3)")
    parser.add_argument("--resume-from", type=str, default="", help="path to a checkpoint (.pth) for resuming training")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--train-samples", type=int, default=100000, help="number of training samples to use (default: 100000; <=0 means all)")
    parser.add_argument("--val-samples", type=int, default=30000, help="number of validation samples to use (default: 30000; <=0 means all)")
    parser.add_argument("--test-samples", type=int, default=20000, help="number of test samples to use (default: 20000; <=0 means all)")

    # 快速验证入口，避免完整数据训练耗时。
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--sanity-train-samples", type=int, default=64)
    parser.add_argument("--sanity-val-samples", type=int, default=32)
    parser.add_argument("--sanity-test-samples", type=int, default=32)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
