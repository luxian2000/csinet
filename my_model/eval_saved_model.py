import sys
import os
from pathlib import Path
import torch
import numpy as np
from importlib import util

MODEL_PATH = Path("my_model/out_10k_2_gpu/best_model_quantum_gpu_outdoor_dim32_20260309_120205.pth")
SOURCE = Path("my_model/compensation_gpu_2.py")

if not MODEL_PATH.exists():
    print(f"Model file not found: {MODEL_PATH}")
    sys.exit(2)
if not SOURCE.exists():
    print(f"Source file not found: {SOURCE}")
    sys.exit(2)

spec = util.spec_from_file_location("compmod", str(SOURCE))
mod = util.module_from_spec(spec)
# Ensure working directory on repo root so relative data paths work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print("Failed to import model source:", e)
    raise

# Construct model
encoded_dim = 32
try:
    model = mod.CsiNetQuantumCompensated(encoded_dim=encoded_dim, alpha=0.25)
except Exception as e:
    print("Failed to instantiate model (maybe pennylane backend missing):", e)
    raise

# Load state_dict
try:
    sd = torch.load(str(MODEL_PATH), map_location="cpu")
    if isinstance(sd, dict):
        model.load_state_dict(sd)
    else:
        # might be full model object
        model = sd
    print("Loaded model successfully.")
except Exception as e:
    print("Failed to load state_dict:", e)
    raise

# Load test data (outdoor) using the module's default data path
try:
    x_train, x_val, x_test, x_test_freq = mod.load_data(envir="outdoor")
except Exception as e:
    print("Failed to load dataset:", e)
    raise

# Run inference on CPU in batches using first 200 real test samples
from torch.utils.data import DataLoader, TensorDataset
TEST_SAMPLE_COUNT = 1000
if x_test.shape[0] > TEST_SAMPLE_COUNT:
    x_test = x_test[:TEST_SAMPLE_COUNT]
    if x_test_freq is not None:
        x_test_freq = x_test_freq[:TEST_SAMPLE_COUNT]

x_test_tensor = torch.FloatTensor(x_test)
loader = DataLoader(TensorDataset(x_test_tensor), batch_size=64, shuffle=False)
model.eval()
outputs = []
with torch.no_grad():
    for (data,) in loader:
        out = model(data)
        outputs.append(out.cpu().numpy())

x_hat = np.concatenate(outputs, axis=0)
print("x_test shape:", x_test.shape)
print("x_hat shape:", x_hat.shape)

if x_test_freq is not None:
    nmse, rho = mod.calculate_nmse_rho(x_test, x_hat, x_test_freq)
    print(f"NMSE (dB): {nmse:.6f}")
    print(f"Rho: {rho:.6f}")
else:
    mse = float(np.mean((x_hat - x_test) ** 2))
    print("No frequency labels; MSE:", mse)
