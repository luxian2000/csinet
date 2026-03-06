import argparse
import time
import json
from pathlib import Path
import numpy as np
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


class TorchCompensationBlock(nn.Module):
    """
    纯 PyTorch 的补偿模块。
    使用 2x2 patch + 小型 MLP 进行局部补偿，完全支持 CUDA。
    """

    def __init__(self, channels=2, hidden_dim=16, window_size=2):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.unfold = nn.Unfold(kernel_size=window_size, stride=window_size)
        self.fold = None

        self.patch_mlp = nn.Sequential(
            nn.Linear(window_size * window_size, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, window_size * window_size),
            nn.Tanh(),
        )

    def forward(self, x):
        batch, ch, h, w = x.shape
        if ch != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {ch}")

        patches = self.unfold(x)
        num_patches = patches.shape[-1]

        patches = patches.reshape(batch, ch, self.window_size * self.window_size, num_patches)
        patches = patches.permute(0, 1, 3, 2).contiguous()

        compensated = self.patch_mlp(patches)
        compensated = compensated.permute(0, 1, 3, 2).contiguous()
        compensated = compensated.reshape(batch, ch * self.window_size * self.window_size, num_patches)

        if self.fold is None or self.fold.output_size != (h, w):
            self.fold = nn.Fold(output_size=(h, w), kernel_size=self.window_size, stride=self.window_size)

        out = self.fold(compensated)
        return out


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


class TorchCompensatedDecoder(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.alpha = alpha

        self.fc_decode = nn.Linear(encoded_dim, IMG_TOTAL)

        self.downsample = nn.AvgPool2d(2)
        self.comp_block = TorchCompensationBlock(channels=IMG_CHANNELS, hidden_dim=16, window_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.main_conv = nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, kernel_size=3, padding=1)
        self.main_bn = nn.BatchNorm2d(IMG_CHANNELS)

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

        main_out = F.leaky_relu(self.main_bn(self.main_conv(x)))

        comp = self.downsample(x)
        comp = self.comp_block(comp)
        comp = self.upsample(comp)

        fused = (1 - self.alpha) * main_out + self.alpha * comp

        residual = fused
        for block in self.residual_blocks:
            residual = F.leaky_relu(residual + block(residual))

        out = self.sigmoid(self.output_conv(residual))
        return out


class CsiNetTorchCompensated(nn.Module):
    def __init__(self, encoded_dim, alpha=0.25):
        super().__init__()
        self.encoder = CsiNetEncoder(encoded_dim)
        self.decoder = TorchCompensatedDecoder(encoded_dim, alpha)

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

    def preprocess_data(data):
        batch_size = data.shape[0]
        return data.reshape(batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)
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

    return nmse, float(np.mean(rho))


def train_model(model, train_loader, val_loader, epochs, lr, device, best_model_path):
    model = model.to(device)
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

        # record learning rate(s) for this epoch
        current_lrs = [float(g.get('lr', 0.0)) for g in optimizer.param_groups]
        lr_history.append(current_lrs)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f}"
        )

    return train_losses, val_losses, lr_history


def make_sanity_loaders(batch_size=64, train_samples=1024, val_samples=256, test_samples=256):
    x_train = torch.rand(train_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_val = torch.rand(val_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_test = torch.rand(test_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader, x_test.numpy()


def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = CsiNetTorchCompensated(encoded_dim=args.encoded_dim, alpha=args.alpha)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.sanity:
        train_loader, val_loader, test_loader, x_test_np = make_sanity_loaders(
            batch_size=args.batch_size,
            train_samples=args.sanity_train_samples,
            val_samples=args.sanity_val_samples,
            test_samples=args.sanity_test_samples,
        )
        x_test_freq = None
    else:
        x_train, x_val, x_test, x_test_freq = load_data(args.envir, args.data_path)
        x_train = torch.FloatTensor(x_train)
        x_val = torch.FloatTensor(x_val)
        x_test_tensor = torch.FloatTensor(x_test)

        train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=args.batch_size, shuffle=False, pin_memory=True)
        x_test_np = x_test

    save_dir = Path(__file__).resolve().parent / "saved_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model_gpu.pth"

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
    model = model.to(device)
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
        print(f"NMSE: {nmse:.2f} dB")
        print(f"Cosine similarity: {rho:.4f}")
        metrics["nmse_db"] = float(nmse)
        metrics["cosine_similarity"] = float(rho)
    else:
        mse = float(np.mean((x_hat - x_test_np) ** 2))
        print(f"Sanity MSE: {mse:.6f}")
        metrics["sanity_mse"] = mse

    final_model_path = save_dir / f"csinet_torch_gpu_{args.envir}_dim{args.encoded_dim}.pth"
    torch.save(model.state_dict(), final_model_path)
    train_loss_path = save_dir / f"train_loss_gpu_{args.envir}_dim{args.encoded_dim}.csv"
    val_loss_path = save_dir / f"val_loss_gpu_{args.envir}_dim{args.encoded_dim}.csv"
    np.savetxt(train_loss_path, train_losses, delimiter=",")
    np.savetxt(val_loss_path, val_losses, delimiter=",")

    run_summary = {
        "args": vars(args),
        "device": str(device),
        "total_parameters": int(total_params),
        "train_time_sec": float(train_time),
        "inference_time_per_sample_sec": float(inference_time_per_sample),
        "train_samples": int(len(train_loader.dataset)),
        "val_samples": int(len(val_loader.dataset)),
        "test_samples": int(len(test_loader.dataset)),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "train_loss_csv": str(train_loss_path),
        "val_loss_csv": str(val_loss_path),
        "metrics": metrics,
        "lr_history": lr_history,
    }
    summary_path = save_dir / f"run_summary_gpu_{args.envir}_dim{args.encoded_dim}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # save lr history as CSV (epochs x param_groups)
    lr_path = save_dir / f"lr_history_gpu_{args.envir}_dim{args.encoded_dim}.csv"
    try:
        np.savetxt(lr_path, np.array(lr_history), delimiter=",")
    except Exception:
        # fallback: write JSON if numerical array conversion fails
        with open(save_dir / f"lr_history_gpu_{args.envir}_dim{args.encoded_dim}.json", "w", encoding="utf-8") as f:
            json.dump(lr_history, f, ensure_ascii=False, indent=2)

    print(f"Saved lr history: {lr_path}")

    print(f"Saved best model: {best_model_path}")
    print(f"Saved final model: {final_model_path}")
    print(f"Saved train loss: {train_loss_path}")
    print(f"Saved val loss: {val_loss_path}")
    print(f"Saved run summary: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="CsiNet Torch CUDA/GPU compensation model")
    parser.add_argument("--envir", type=str, default="indoor", choices=["indoor", "outdoor"])
    parser.add_argument("--data-path", type=str, default="/home/luxian/DataSpace/csinet/data")
    parser.add_argument("--encoded-dim", type=int, default=32, choices=sorted(set(COMPRESSION_RATES.values())))
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sanity", action="store_true", help="Use synthetic data for quick GPU verification")
    parser.add_argument("--sanity-train-samples", type=int, default=512)
    parser.add_argument("--sanity-val-samples", type=int, default=128)
    parser.add_argument("--sanity-test-samples", type=int, default=128)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
