import argparse
import json
import time
from pathlib import Path

import numpy as np
import pennylane as qml
try:
    # optional GPU-accelerated Lightning device (if installed with CUDA support)
    from pennylane_lightning.lightning_gpu import LightningGPU
except Exception:
    LightningGPU = None
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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
    量子补偿模块（保持量子-经典混合结构）。

    说明：当量子后端不支持 CUDA 时，量子电路在 CPU 上执行，
    经典网络仍可在 GPU 上训练与推理。
    """

    def __init__(self, n_qubits=4, n_layers=2, window_size=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        if n_qubits != window_size * window_size:
            raise ValueError("n_qubits must equal window_size*window_size for fold/unfold reconstruction.")

        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        # 优先尝试 GPU 加速的 lightning（如果可用），失败则回退到 lightning.qubit / default.qubit
        self.backend = None
        self.dev = None
        if LightningGPU is not None:
            try:
                # 启用 batch_obs 以便 qnode 接受批量输入并由设备批量处理
                self.dev = LightningGPU(wires=n_qubits, batch_obs=True)
                self.backend = "lightning.gpu"
            except Exception:
                self.dev = None

        if self.dev is None:
            for backend in ("lightning.qubit", "default.qubit"):
                try:
                    self.dev = qml.device(backend, wires=n_qubits)
                    self.backend = backend
                    break
                except Exception:
                    continue

        if self.dev is None:
            raise RuntimeError("No available PennyLane simulator backend found.")

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def quantum_circuit(inputs, weights_crz, weights_ry):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[..., i], wires=i)

            for layer in range(n_layers):
                for i in range(n_qubits - 1):
                    qml.CRZ(weights_crz[layer, i], wires=[i, i + 1])
                qml.CRZ(weights_crz[layer, n_qubits - 1], wires=[n_qubits - 1, 0])

                for i in range(n_qubits):
                    qml.RY(weights_ry[layer, i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = quantum_circuit

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
        # 如果后端支持 GPU（例如 lightning.gpu），则在对应设备上执行量子电路；
        # 否则将数据移动到 CPU 进行仿真。
        use_gpu_quantum = (self.backend is not None) and ("gpu" in str(self.backend))
        x_proc = x_map if use_gpu_quantum else x_map.to("cpu")

        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        patches = unfold(x_proc)  # [batch, 4, num_patches]
        num_patches = patches.shape[-1]

        total_samples = batch * num_patches
        all_inputs = patches.permute(0, 2, 1).reshape(total_samples, self.n_qubits)
        all_inputs = torch.tanh(all_inputs) * np.pi

        all_outputs = []
        # 尝试批量调用 qnode。若设备/接口支持批量输入（例如 LightningGPU + batch_obs=True），
        # 可以一次性传入 shape=(B, n_qubits) 的张量并返回 shape=(B, n_qubits) 的结果。
        # 为了兼容不同后端，同时按块处理以控制内存。
        chunk_size = 256
        for start in range(0, total_samples, chunk_size):
            end = min(total_samples, start + chunk_size)
            batch_inp = all_inputs[start:end]
            if use_gpu_quantum and hasattr(self.weights_crz, 'device'):
                batch_inp = batch_inp.to(self.weights_crz.device)

            q_out = self.circuit(batch_inp, self.weights_crz, self.weights_ry)

            # q_out 可能是 list-of-tensors (每个元素长度=batch) 或者 tensor (batch, n_qubits)
            if isinstance(q_out, (list, tuple)):
                # 转成 tensor (batch, n_qubits)
                q_out = torch.stack([o if isinstance(o, torch.Tensor) else torch.tensor(o) for o in q_out], dim=1)
            else:
                # 若 q_out 是 tensor，确保维度为 (batch, n_qubits)
                if q_out.dim() == 1:
                    q_out = q_out.unsqueeze(1)
                elif q_out.dim() == 2 and q_out.shape[0] == self.n_qubits and q_out.shape[1] == (end - start):
                    q_out = q_out.transpose(0, 1)

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
        self.quantum_comp = QuantumCompensationBlock(n_qubits=4, n_layers=2, window_size=2)
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


def load_data(envir="indoor", data_path="/home/luxian/DataSpace/csinet/data"):
    if envir == "indoor":
        x_train = sio.loadmat(f"{data_path}/DATA_Htrainin.mat")["HT"].astype(np.float32)
        x_val = sio.loadmat(f"{data_path}/DATA_Hvalin.mat")["HT"].astype(np.float32)
        x_test = sio.loadmat(f"{data_path}/DATA_Htestin.mat")["HT"].astype(np.float32)
        x_test_freq = sio.loadmat(f"{data_path}/DATA_HtestFin_all.mat")["HF_all"].astype(np.complex128)
    else:
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
    # 将经典部分迁移到目标 device。若量子后端支持 GPU（例如 lightning.gpu），
    # 则把量子模块也迁移到相同 device；否则保持量子模块在 CPU。
    model = model.to(device)
    qc_backend = getattr(model.decoder.quantum_comp, "backend", None)
    if qc_backend and ("gpu" in str(qc_backend)):
        model.decoder.quantum_comp = model.decoder.quantum_comp.to(device)
    else:
        model.decoder.quantum_comp = model.decoder.quantum_comp.to("cpu")
    return model


def train_model(model, train_loader, val_loader, epochs, lr, device, best_model_path):
    model = _move_model_devices(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

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

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
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
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(data)
                    loss = criterion(output, data)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        scheduler.step()

        current_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
        lr_history.append(current_lrs)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (epoch {epoch+1}): {best_model_path}", flush=True)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f} "
            f"LR: {current_lrs}",
            flush=True,
        )

    return train_losses, val_losses, lr_history


def make_sanity_loaders(batch_size=32, train_samples=64, val_samples=32, test_samples=32):
    x_train = torch.rand(train_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_val = torch.rand(val_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_test = torch.rand(test_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader, x_test.numpy(), None


def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        x_train, x_val, x_test, x_test_freq = load_data(args.envir, args.data_path)
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

        train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=args.batch_size, shuffle=False, pin_memory=True)
        x_test_np = x_test

    # Determine save directory: priority --outputdir, then deprecated --output-dir, otherwise default saved_model
    out_arg = getattr(args, "outputdir", "") or getattr(args, "output_dir", "")
    if out_arg and str(out_arg).strip():
        save_dir = Path(out_arg).expanduser().resolve()
    else:
        save_dir = Path(__file__).resolve().parent / "saved_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() if args.run_tag else ""
    if not run_tag:
        run_tag = time.strftime("%Y%m%d_%H%M%S")

    suffix = f"{args.envir}_dim{args.encoded_dim}_{run_tag}"
    best_model_path = save_dir / f"best_model_quantum_gpu_{suffix}.pth"

    start = time.time()
    train_losses, val_losses, lr_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        best_model_path=best_model_path,
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
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

    final_model_path = save_dir / f"csinet_quantum_gpu_{suffix}.pth"
    train_loss_path = save_dir / f"train_loss_quantum_gpu_{suffix}.csv"
    val_loss_path = save_dir / f"val_loss_quantum_gpu_{suffix}.csv"
    lr_path = save_dir / f"lr_history_quantum_gpu_{suffix}.csv"

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

    summary_path = save_dir / f"run_summary_quantum_gpu_{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved best model: {best_model_path}")
    print(f"Saved final model: {final_model_path}")
    print(f"Saved train loss: {train_loss_path}")
    print(f"Saved val loss: {val_loss_path}")
    print(f"Saved lr history: {lr_path}")
    print(f"Saved run summary: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="CsiNet quantum-classical hybrid (GPU-oriented)")
    parser.add_argument("--envir", type=str, default="indoor", choices=["indoor", "outdoor"])
    parser.add_argument("--data-path", type=str, default="/home/luxian/DataSpace/csinet/data")
    parser.add_argument("--encoded-dim", type=int, default=32, choices=sorted(set(COMPRESSION_RATES.values())))
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="", help="(deprecated) output directory; prefer --outputdir")
    parser.add_argument("--outputdir", type=str, default="", help="output directory for saved artifacts (default: my_model/saved_model)")
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
