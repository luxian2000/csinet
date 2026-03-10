import argparse
import importlib.util
from pathlib import Path

import torch


def load_module(path):
    spec = importlib.util.spec_from_file_location("comp", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, default="my_model/snapshots/checkpoint_snapshot_20260310_161949.pth")
    parser.add_argument("--encoded-dim", type=int, default=32)
    parser.add_argument("--envir", type=str, default="outdoor", choices=["indoor", "outdoor"])
    parser.add_argument("--data-path", type=str, default="/home/luxian/DataSpace/csinet/data")
    parser.add_argument("--max-samples", type=int, default=0, help="limit number of test samples (0=all)")
    args = parser.parse_args()

    mod = load_module(Path(__file__).resolve().parent / "compensation_gpu_2.py")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = mod.CsiNetQuantumCompensated(encoded_dim=args.encoded_dim)

    # load snapshot
    snap = Path(args.snapshot).expanduser().resolve()
    if not snap.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap}")

    ckpt = torch.load(str(snap), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # assume it's a state_dict
        state = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    model.load_state_dict(state)
    model = mod._move_model_devices(model, device)
    model.eval()

    # load data
    x_train, x_val, x_test, x_test_freq = mod.load_data(args.envir, args.data_path)

    import numpy as np

    x_test_np = x_test
    x_test_tensor = torch.FloatTensor(x_test)

    # run inference in batches (optionally limit samples)
    bs = 200
    total = x_test_tensor.shape[0]
    if args.max_samples and args.max_samples > 0:
        total = min(total, args.max_samples)

    outputs = []
    with torch.no_grad():
        for i in range(0, total, bs):
            batch = x_test_tensor[i : i + bs].to(device)
            out = model(batch)
            outputs.append(out.float().cpu().numpy())
    x_hat = np.concatenate(outputs, axis=0)
    x_test_np = x_test_np[: x_hat.shape[0]]

    nmse, rho = mod.calculate_nmse_rho(x_test_np, x_hat, x_test_freq)
    print(f"NMSE (dB): {nmse:.4f}")
    print(f"Rho: {rho:.6f}")


if __name__ == "__main__":
    main()
