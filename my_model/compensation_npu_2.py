import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    HAS_TORCH_NPU = False


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
    量子补偿模块（纯 Torch 近似实现）。

    说明：不依赖 pennylane，通过可微分的旋转/纠缠近似算子模拟量子块，
    可在 NPU 上端到端训练。
    """

    def __init__(self, n_qubits=16, n_layers=2, window_size=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        if n_qubits != window_size * window_size:
            raise ValueError("n_qubits must equal window_size*window_size for fold/unfold reconstruction.")

        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.input_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.mix_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.backend = "torch_npu_approx"

    def _torch_quantum_layer(self, state, layer_idx):
        # 环形相邻耦合，近似 CRZ 纠缠效应。
        neigh = torch.roll(state, shifts=-1, dims=1)
        crz = self.weights_crz[layer_idx].unsqueeze(0)
        ry = self.weights_ry[layer_idx].unsqueeze(0)

        phase_mix = torch.cos(crz) * state + torch.sin(crz) * neigh
        rotated = torch.sin(phase_mix + ry)

        mix = torch.sigmoid(self.mix_gate)
        return mix * rotated + (1.0 - mix) * phase_mix

    def _torch_quantum_forward(self, inputs):
        # H + RY 的近似编码，保持输出范围在 [-1, 1]。
        state = torch.tanh(inputs * self.input_scale)
        state = (torch.sin(state) + torch.cos(state)) * (2.0 ** -0.5)

        for layer in range(self.n_layers):
            state = self._torch_quantum_layer(state, layer)

        # 近似 <Z> 期望。
        return torch.tanh(state)

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
        chunk_size = 256
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


class _NoOpGradScaler:
    """Fallback scaler for environments without amp GradScaler support."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def _build_grad_scaler(device):
    # NPU-only runtime: disable scaler to avoid CUDA/AMP dependency.
    return _NoOpGradScaler()


def train_model(model, train_loader, val_loader, test_loader, x_test_np, x_test_freq, epochs, lr, device, best_model_path, save_dir=None):
    model = _move_model_devices(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    scaler = _build_grad_scaler(device)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    lr_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

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

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device, non_blocking=True)
                with _amp_context(device):
                    output = model(data)
                    loss = criterion(output, data)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        # Compute test metrics each epoch (NMSE & rho if frequency data available)
        test_outputs = []
        with torch.no_grad():
            for (data,) in test_loader:
                data = data.to(device, non_blocking=True)
                with _amp_context(device):
                    out = model(data)
                test_outputs.append(out.float().cpu().numpy())
        x_hat = np.concatenate(test_outputs, axis=0)

        epoch_metrics = {}
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

        # 每个 epoch 结束时都打印 NMSE 和 rho（无频域标签时显示 N/A）
        nmse_value = epoch_metrics.get("nmse_db", None)
        rho_value = epoch_metrics.get("rho", None)

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
        print(summary_line, flush=True)
        if save_dir is not None:
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (epoch {epoch+1}): {best_model_path}", flush=True)

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

    if hasattr(torch.npu, "set_device"):
        torch.npu.set_device(0)
    device = torch.device("npu:0")
    print(f"Using device: {device}")

    model = CsiNetQuantumCompensated(encoded_dim=args.encoded_dim, alpha=args.alpha)
    print(f"Quantum backend: {model.decoder.quantum_comp.backend}")

    if args.sanity:
        train_loader, val_loader, test_loader, x_test_np, x_test_freq = make_sanity_loaders(
            batch_size=args.batch_size,
            train_samples=args.sanity_train_samples,
            val_samples=args.sanity_val_samples,
            test_samples=args.sanity_test_samples,
        )
    else:
        x_train, x_val, x_test, x_test_freq = load_data(args.data_path)
        # Optionally subset the real datasets to requested sizes
        if getattr(args, "train_samples", 0) and args.train_samples > 0:
            x_train = x_train[: args.train_samples]
        if getattr(args, "val_samples", 0) and args.val_samples > 0:
            x_val = x_val[: args.val_samples]
        if getattr(args, "test_samples", 0) and args.test_samples > 0:
            x_test = x_test[: args.test_samples]

        x_train = torch.FloatTensor(x_train)
        x_val = torch.FloatTensor(x_val)
        x_test_tensor = torch.FloatTensor(x_test)

        train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False, pin_memory=False)
        test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=args.batch_size, shuffle=False, pin_memory=False)
        x_test_np = x_test

    # Determine save directory: priority --outputdir, then deprecated --output-dir,
    # otherwise default out_10k_2.
    out_arg = getattr(args, "outputdir", "") or getattr(args, "output_dir", "")
    if out_arg and str(out_arg).strip():
        save_dir = Path(out_arg).expanduser().resolve()
    else:
        save_dir = Path(__file__).resolve().parent / "out_10k_2"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() if args.run_tag else ""
    if not run_tag:
        run_tag = time.strftime("%Y%m%d_%H%M%S")

    suffix = f"{args.envir}_dim{args.encoded_dim}_{run_tag}"
    best_model_path = save_dir / f"best_model_quantum_npu_{suffix}.pth"

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
        save_dir=save_dir,
    )
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f}s")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = _move_model_devices(model, device)
    model.eval()

    outputs = []
    infer_start = time.time()
    with torch.no_grad():
        for (data,) in test_loader:
            data = data.to(device, non_blocking=True)
            output = model(data)
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

    torch.save(model.state_dict(), final_model_path)
    np.savetxt(train_loss_path, train_losses, delimiter=",")
    np.savetxt(val_loss_path, val_losses, delimiter=",")
    np.savetxt(lr_path, np.array(lr_history), delimiter=",")

    summary = {
        "args": vars(args),
        "device": str(device),
        "quantum_backend": model.decoder.quantum_comp.backend,
        "train_time_sec": float(train_time),
        "inference_time_per_sample_sec": float(inference_time_per_sample),
        "train_samples": int(len(train_loader.dataset)),
        "val_samples": int(len(val_loader.dataset)),
        "test_samples": int(len(test_loader.dataset)),
        "best_model_path": str(best_model_path),
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
    print(f"Saved final model: {final_model_path}")
    print(f"Saved train loss: {train_loss_path}")
    print(f"Saved val loss: {val_loss_path}")
    print(f"Saved lr history: {lr_path}")
    print(f"Saved run summary: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="CsiNet quantum-classical hybrid (NPU-oriented)")
    parser.add_argument("--envir", type=str, default="outdoor", choices=["outdoor"], help="data environment (outdoor only)")
    parser.add_argument("--data-path", type=str, default="/root/work/luxian/csinet/data")
    parser.add_argument("--encoded-dim", type=int, default=32, choices=sorted(set(COMPRESSION_RATES.values())))
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="", help="(deprecated) output directory; prefer --outputdir")
    parser.add_argument("--outputdir", type=str, default="", help="output directory for saved artifacts (default: my_model/out_10k_2)")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--train-samples", type=int, default=0, help="number of training samples to use (0=all)")
    parser.add_argument("--val-samples", type=int, default=0, help="number of validation samples to use (0=all)")
    parser.add_argument("--test-samples", type=int, default=0, help="number of test samples to use (0=all)")

    # 快速验证入口，避免完整数据训练耗时。
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--sanity-train-samples", type=int, default=64)
    parser.add_argument("--sanity-val-samples", type=int, default=32)
    parser.add_argument("--sanity-test-samples", type=int, default=32)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
