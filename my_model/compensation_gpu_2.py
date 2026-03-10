import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pennylane as qml
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

DEFAULT_DATA_PATH = "/home/luxian/DataSpace/csinet/data"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out_10k_2_gpu"


class QuantumCompensationBlock(nn.Module):
    def __init__(self, n_qubits=16, n_layers=2, window_size=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        if n_qubits != window_size * window_size:
            raise ValueError("n_qubits must equal window_size*window_size")

        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        try:
            self.dev = qml.device("lightning.gpu", wires=n_qubits, batch_obs=True)
            self.backend = "lightning.gpu"
        except Exception as exc:
            raise RuntimeError(
                "Failed to create lightning.gpu. Install pennylane-lightning[gpu] and ensure CUDA is available."
            ) from exc

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
        if x.dim() != 2:
            raise ValueError(f"Expected latent tensor [B, D], got {tuple(x.shape)}")

        batch, dim = x.shape
        latent_h = 4
        if dim % latent_h != 0:
            raise ValueError(f"encoded dim must be divisible by {latent_h}, got {dim}")
        latent_w = dim // latent_h
        if latent_w % self.window_size != 0:
            raise ValueError(f"latent width {latent_w} must be divisible by window_size {self.window_size}")

        original_device = x.device
        x_map = x.reshape(batch, 1, latent_h, latent_w)
        x_proc = x_map.to(self.weights_crz.device)

        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        patches = unfold(x_proc)
        num_patches = patches.shape[-1]

        total_samples = batch * num_patches
        all_inputs = patches.permute(0, 2, 1).reshape(total_samples, self.n_qubits)
        all_inputs = torch.tanh(all_inputs) * np.pi

        all_outputs = []
        chunk_size = 256
        for start in range(0, total_samples, chunk_size):
            end = min(total_samples, start + chunk_size)
            q_out = self.circuit(all_inputs[start:end].to(self.weights_crz.device), self.weights_crz, self.weights_ry)
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack([o if isinstance(o, torch.Tensor) else torch.tensor(o) for o in q_out], dim=1)
            all_outputs.append(q_out)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_outputs = all_outputs.reshape(batch, num_patches, self.n_qubits).permute(0, 2, 1)

        fold = nn.Fold(output_size=(latent_h, latent_w), kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        output = fold(all_outputs).float().reshape(batch, dim)
        return output.to(original_device)


class CsiNetEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.act = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(IMG_TOTAL, encoded_dim)

    def forward(self, x):
        return self.fc_encode(self.flatten(self.act(self.bn1(self.conv1(x)))))


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

        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, kernel_size=3, padding=1),
                nn.BatchNorm2d(IMG_CHANNELS),
                nn.LeakyReLU(),
                nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, kernel_size=3, padding=1),
                nn.BatchNorm2d(IMG_CHANNELS),
            )
            for _ in range(5)
        ])
        self.output_conv = nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s):
        batch = s.shape[0]
        x = self.fc_decode(s).reshape(batch, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        q_latent = self.quantum_comp(s).reshape(batch, 1, self.latent_h, self.latent_w)
        comp = self.quantum_proj(self.quantum_upsample(q_latent))

        alpha = torch.sigmoid(self.alpha_logit)
        residual = (1 - alpha) * x + alpha * comp

        for block in self.residual_blocks:
            residual = F.leaky_relu(residual + block(residual))

        return self.sigmoid(self.output_conv(residual))


class CsiNetQuantumCompensated(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.encoder = CsiNetEncoder(encoded_dim)
        self.decoder = QuantumCompensatedDecoder(encoded_dim, alpha)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data(envir="outdoor", data_path=DEFAULT_DATA_PATH):
    if envir == "indoor":
        train = sio.loadmat(f"{data_path}/DATA_Htrainin.mat")["HT"].astype(np.float32)
        val = sio.loadmat(f"{data_path}/DATA_Hvalin.mat")["HT"].astype(np.float32)
        test = sio.loadmat(f"{data_path}/DATA_Htestin.mat")["HT"].astype(np.float32)
        test_freq = sio.loadmat(f"{data_path}/DATA_HtestFin_all.mat")["HF_all"].astype(np.complex128)
    else:
        train = sio.loadmat(f"{data_path}/DATA_Htrainout.mat")["HT"].astype(np.float32)
        val = sio.loadmat(f"{data_path}/DATA_Hvalout.mat")["HT"].astype(np.float32)
        test = sio.loadmat(f"{data_path}/DATA_Htestout.mat")["HT"].astype(np.float32)
        test_freq = sio.loadmat(f"{data_path}/DATA_HtestFout_all.mat")["HF_all"].astype(np.complex128)

    def preprocess(data):
        bs = data.shape[0]
        return data.reshape(bs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    return preprocess(train), preprocess(val), preprocess(test), test_freq.reshape(-1, IMG_HEIGHT, 125)


def calculate_nmse_rho(x_test, x_hat, x_test_freq):
    bs = x_test.shape[0]

    x_test_real = x_test[:, 0, :, :].reshape(bs, -1)
    x_test_imag = x_test[:, 1, :, :].reshape(bs, -1)
    x_test_c = (x_test_real - 0.5) + 1j * (x_test_imag - 0.5)

    x_hat_real = x_hat[:, 0, :, :].reshape(bs, -1)
    x_hat_imag = x_hat[:, 1, :, :].reshape(bs, -1)
    x_hat_c = (x_hat_real - 0.5) + 1j * (x_hat_imag - 0.5)

    x_hat_f = x_hat_c.reshape(bs, IMG_HEIGHT, IMG_WIDTH)
    x_hat_full = np.fft.fft(
        np.concatenate((x_hat_f, np.zeros((bs, IMG_HEIGHT, 257 - IMG_WIDTH))), axis=2),
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


def _to_device(model, device):
    model = model.to(device)
    model.decoder.quantum_comp = model.decoder.quantum_comp.to(device)
    return model


def _parse_epoch_from_filename(path):
    match = re.search(r"epoch_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def _save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val_loss, train_losses, val_losses):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": float(best_val_loss),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def _load_resume(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_val_loss", float("inf"))), list(
            ckpt.get("train_losses", [])
        ), list(ckpt.get("val_losses", []))

    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        parsed_epoch = _parse_epoch_from_filename(path)
        start_epoch = parsed_epoch if parsed_epoch is not None else 0
        return start_epoch, float("inf"), [], []

    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _find_latest_resume_candidate(save_dir):
    patterns = [
        "checkpoint_last_*.pth",
        "checkpoint_epoch_*.pth",
        "model_epoch_*.pth",
        "best_model_quantum_gpu_*.pth",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(save_dir.glob(pat))
    if not candidates:
        return ""
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return str(latest)


def _make_loaders(x_train, x_val, x_test, batch_size):
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(x_train)), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(x_val)), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(x_test)), batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader


def _append_epoch_metrics(metrics_csv_path, row):
    file_exists = metrics_csv_path.exists()
    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "nmse_db", "rho", "checkpoint", "model"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_and_eval(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA GPU.")
    device = torch.device("cuda:0")

    save_dir = Path(args.outputdir).expanduser().resolve() if args.outputdir else DEFAULT_OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() if args.run_tag else time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.envir}_dim{args.encoded_dim}_{run_tag}"

    model = _to_device(CsiNetQuantumCompensated(args.encoded_dim, args.alpha), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_model_path = save_dir / f"best_model_quantum_gpu_{suffix}.pth"
    epoch_metrics_csv = save_dir / f"epoch_metrics_{suffix}.csv"
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_epoch = 0

    resume_path = ""
    if args.resume_from:
        resume_path = str(Path(args.resume_from).expanduser().resolve())
    elif args.resume_latest:
        resume_path = _find_latest_resume_candidate(save_dir)

    if resume_path:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        print(f"Resuming from: {resume_path}", flush=True)
        start_epoch, best_val_loss, train_losses, val_losses = _load_resume(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        model = _to_device(model, device)

    x_train, x_val, x_test, x_test_freq = load_data(args.envir, args.data_path)
    x_train = x_train[: args.train_samples]
    x_val = x_val[: args.val_samples]
    x_test = x_test[: args.test_samples]
    x_test_freq = x_test_freq[: args.test_samples]

    train_loader, val_loader, test_loader = _make_loaders(x_train, x_val, x_test, args.batch_size)

    print(f"Device: {device}")
    print(f"Quantum backend: {model.decoder.quantum_comp.backend}")
    print(f"Save directory: {save_dir}")
    print(f"Samples train/val/test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        for (data,) in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(data)
                loss = criterion(output, data)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = model(data)
                    loss = criterion(output, data)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        outputs = []
        with torch.no_grad():
            for (data,) in test_loader:
                data = data.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    out = model(data)
                outputs.append(out.float().cpu().numpy())
        x_hat = np.concatenate(outputs, axis=0)
        nmse_db, rho = calculate_nmse_rho(x_test, x_hat, x_test_freq)

        scheduler.step()

        epoch_model_path = save_dir / f"model_epoch_{epoch + 1:03d}_{suffix}.pth"
        checkpoint_epoch_path = save_dir / f"checkpoint_epoch_{epoch + 1:03d}_{suffix}.pth"
        checkpoint_last_path = save_dir / f"checkpoint_last_{suffix}.pth"

        torch.save(model.state_dict(), epoch_model_path)
        _save_checkpoint(
            checkpoint_epoch_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_loss,
            train_losses,
            val_losses,
        )
        _save_checkpoint(
            checkpoint_last_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_loss,
            train_losses,
            val_losses,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        _append_epoch_metrics(
            epoch_metrics_csv,
            {
                "epoch": epoch + 1,
                "train_loss": f"{avg_train_loss:.8f}",
                "val_loss": f"{avg_val_loss:.8f}",
                "nmse_db": f"{nmse_db:.6f}",
                "rho": f"{rho:.6f}",
                "checkpoint": str(checkpoint_epoch_path),
                "model": str(epoch_model_path),
            },
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train {avg_train_loss:.6f} Val {avg_val_loss:.6f} "
            f"NMSE {nmse_db:.2f} dB Rho {rho:.4f}",
            flush=True,
        )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model = _to_device(model, device)

    model.eval()
    infer_start = time.time()
    outputs = []
    with torch.no_grad():
        for (data,) in test_loader:
            data = data.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(data)
            outputs.append(output.float().cpu().numpy())
    infer_time = time.time() - infer_start
    x_hat = np.concatenate(outputs, axis=0)
    final_nmse, final_rho = calculate_nmse_rho(x_test, x_hat, x_test_freq)

    final_model_path = save_dir / f"final_model_quantum_gpu_{suffix}.pth"
    train_loss_path = save_dir / f"train_loss_{suffix}.csv"
    val_loss_path = save_dir / f"val_loss_{suffix}.csv"
    summary_path = save_dir / f"run_summary_{suffix}.json"

    torch.save(model.state_dict(), final_model_path)
    np.savetxt(train_loss_path, train_losses, delimiter=",")
    np.savetxt(val_loss_path, val_losses, delimiter=",")

    summary = {
        "args": vars(args),
        "device": str(device),
        "quantum_backend": model.decoder.quantum_comp.backend,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "epoch_metrics_csv": str(epoch_metrics_csv),
        "final_nmse_db": float(final_nmse),
        "final_rho": float(final_rho),
        "inference_time_per_sample_sec": float(infer_time / max(1, len(test_loader.dataset))),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Final NMSE: {final_nmse:.2f} dB")
    print(f"Final Rho: {final_rho:.4f}")
    print(f"Saved summary: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Refactored CsiNet quantum-compensated training")
    parser.add_argument("--envir", type=str, default="outdoor", choices=["indoor", "outdoor"])
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--encoded-dim", type=int, default=32, choices=[32, 64, 128, 512])
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--outputdir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="", help="Resume from a specific checkpoint/model path")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the newest checkpoint/model in outputdir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_and_eval(args)
