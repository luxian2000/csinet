import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def _load_csv_series(path):
    if not path.exists():
        return np.array([])
    data = np.genfromtxt(path, delimiter=",")
    if data.size == 0:
        return np.array([])
    if np.isscalar(data):
        return np.array([float(data)])
    return np.array(data)


def _find_latest_summary(output_dir):
    candidates = sorted(output_dir.glob("run_summary_quantum_npu_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _parse_epoch_metrics_from_log(log_path):
    """Parse NMSE/Rho from log_snapshot lines if available."""
    nmse_list = []
    rho_list = []
    if not log_path.exists():
        return nmse_list, rho_list

    pattern_nmse = re.compile(r"NMSE:\s*([+-]?\d+(?:\.\d+)?)\s*dB")
    pattern_rho = re.compile(r"Rho:\s*([+-]?\d+(?:\.\d+)?)")

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m_nmse = pattern_nmse.search(line)
        m_rho = pattern_rho.search(line)
        if m_nmse:
            nmse_list.append(_safe_float(m_nmse.group(1)))
        if m_rho:
            rho_list.append(_safe_float(m_rho.group(1)))

    return nmse_list, rho_list


def _load_checkpoint_epoch(checkpoint_path):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    try:
        import torch  # local import to avoid hard dependency when only report is needed

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            return int(ckpt["epoch"])
    except Exception:
        return None

    return None


def _render_plots(train_losses, val_losses, lr_history, nmse_per_epoch, rho_per_epoch, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plots_enabled": False, "reason": f"matplotlib unavailable: {exc}"}

    paths = {}

    if train_losses.size > 0 or val_losses.size > 0:
        plt.figure(figsize=(8, 5))
        if train_losses.size > 0:
            plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train_loss")
        if val_losses.size > 0:
            plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training/Validation Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        loss_path = out_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()
        paths["loss_curve"] = str(loss_path)

    lr_array = np.array(lr_history)
    if lr_array.size > 0:
        plt.figure(figsize=(8, 5))
        if lr_array.ndim == 1:
            lr_array = lr_array.reshape(-1, 1)
        for idx in range(lr_array.shape[1]):
            plt.plot(np.arange(1, lr_array.shape[0] + 1), lr_array[:, idx], label=f"group_{idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate History")
        plt.grid(alpha=0.3)
        plt.legend()
        lr_path = out_dir / "lr_curve.png"
        plt.tight_layout()
        plt.savefig(lr_path, dpi=150)
        plt.close()
        paths["lr_curve"] = str(lr_path)

    nmse_array = np.array(nmse_per_epoch, dtype=float)
    if nmse_array.size > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(nmse_array) + 1), nmse_array, label="nmse_db", color="tab:red")
        plt.xlabel("Epoch")
        plt.ylabel("NMSE (dB)")
        plt.title("NMSE per Epoch")
        plt.grid(alpha=0.3)
        plt.legend()
        nmse_path = out_dir / "nmse_curve.png"
        plt.tight_layout()
        plt.savefig(nmse_path, dpi=150)
        plt.close()
        paths["nmse_curve"] = str(nmse_path)

    rho_array = np.array(rho_per_epoch, dtype=float)
    if rho_array.size > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(rho_array) + 1), rho_array, label="rho", color="tab:green")
        plt.xlabel("Epoch")
        plt.ylabel("Rho")
        plt.title("Rho per Epoch")
        plt.grid(alpha=0.3)
        plt.legend()
        rho_path = out_dir / "rho_curve.png"
        plt.tight_layout()
        plt.savefig(rho_path, dpi=150)
        plt.close()
        paths["rho_curve"] = str(rho_path)

    return {"plots_enabled": True, "paths": paths}


def analyze(summary_path, analysis_dir, render_plots=True):
    summary_path = Path(summary_path).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_loss_path = Path(summary.get("train_loss_csv", ""))
    val_loss_path = Path(summary.get("val_loss_csv", ""))
    lr_path = Path(summary.get("lr_history_csv", ""))
    latest_ckpt_path = Path(summary.get("latest_checkpoint_path", "")) if summary.get("latest_checkpoint_path") else None
    best_ckpt_path = Path(summary.get("best_checkpoint_path", "")) if summary.get("best_checkpoint_path") else None

    train_losses = _load_csv_series(train_loss_path)
    val_losses = _load_csv_series(val_loss_path)
    lr_history = _load_csv_series(lr_path)

    if train_losses.ndim > 1:
        train_losses = train_losses.reshape(-1)
    if val_losses.ndim > 1:
        val_losses = val_losses.reshape(-1)

    best_epoch = None
    best_val = None
    if val_losses.size > 0:
        idx = int(np.nanargmin(val_losses))
        best_epoch = idx + 1
        best_val = float(val_losses[idx])

    final_train = float(train_losses[-1]) if train_losses.size > 0 else np.nan
    final_val = float(val_losses[-1]) if val_losses.size > 0 else np.nan

    final_metrics = summary.get("metrics", {})
    nmse_db = final_metrics.get("nmse_db", None)
    rho = final_metrics.get("cosine_similarity", final_metrics.get("rho", None))

    log_snapshot = summary_path.parent / "log_snapshot.txt"
    nmse_per_epoch, rho_per_epoch = _parse_epoch_metrics_from_log(log_snapshot)

    latest_epoch_in_ckpt = _load_checkpoint_epoch(latest_ckpt_path) if latest_ckpt_path else None
    best_epoch_in_ckpt = _load_checkpoint_epoch(best_ckpt_path) if best_ckpt_path else None

    plot_result = {"plots_enabled": False}
    if render_plots:
        plot_result = _render_plots(
            train_losses=train_losses,
            val_losses=val_losses,
            lr_history=lr_history,
            nmse_per_epoch=nmse_per_epoch,
            rho_per_epoch=rho_per_epoch,
            out_dir=analysis_dir,
        )

    best_nmse_db = float(np.nanmin(np.array(nmse_per_epoch, dtype=float))) if len(nmse_per_epoch) > 0 else None
    best_rho = float(np.nanmax(np.array(rho_per_epoch, dtype=float))) if len(rho_per_epoch) > 0 else None

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_path": str(summary_path),
        "device": summary.get("device"),
        "quantum_backend": summary.get("quantum_backend"),
        "train_samples": summary.get("train_samples"),
        "val_samples": summary.get("val_samples"),
        "test_samples": summary.get("test_samples"),
        "train_time_sec": summary.get("train_time_sec"),
        "inference_time_per_sample_sec": summary.get("inference_time_per_sample_sec"),
        "epochs_recorded": int(max(len(train_losses), len(val_losses))),
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "best_val_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_nmse_db": nmse_db,
        "final_rho": rho,
        "best_nmse_db": best_nmse_db,
        "best_rho": best_rho,
        "nmse_per_epoch_count": len(nmse_per_epoch),
        "rho_per_epoch_count": len(rho_per_epoch),
        "latest_checkpoint_epoch": latest_epoch_in_ckpt,
        "best_checkpoint_epoch": best_epoch_in_ckpt,
        "artifacts": {
            "best_model_path": summary.get("best_model_path"),
            "latest_checkpoint_path": summary.get("latest_checkpoint_path"),
            "best_checkpoint_path": summary.get("best_checkpoint_path"),
            "final_model_path": summary.get("final_model_path"),
        },
        "plot_info": plot_result,
    }

    report_json = analysis_dir / "analysis_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_md = analysis_dir / "analysis_report.md"
    lines = [
        "# Training Analysis Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Summary file: {report['summary_path']}",
        f"- Device: {report['device']}",
        f"- Quantum backend: {report['quantum_backend']}",
        "",
        "## Dataset",
        f"- Train samples: {report['train_samples']}",
        f"- Val samples: {report['val_samples']}",
        f"- Test samples: {report['test_samples']}",
        "",
        "## Core Metrics",
        f"- Epochs recorded: {report['epochs_recorded']}",
        f"- Final train loss: {report['final_train_loss']}",
        f"- Final val loss: {report['final_val_loss']}",
        f"- Best val epoch: {report['best_val_epoch']}",
        f"- Best val loss: {report['best_val_loss']}",
        f"- Final NMSE (dB): {report['final_nmse_db']}",
        f"- Final Rho: {report['final_rho']}",
        f"- Best NMSE over epochs (dB): {report['best_nmse_db']}",
        f"- Best Rho over epochs: {report['best_rho']}",
        f"- NMSE points parsed from log: {report['nmse_per_epoch_count']}",
        f"- Rho points parsed from log: {report['rho_per_epoch_count']}",
        "",
        "## Timing",
        f"- Train time (sec): {report['train_time_sec']}",
        f"- Inference time per sample (sec): {report['inference_time_per_sample_sec']}",
        "",
        "## Checkpoint Status",
        f"- Latest checkpoint epoch: {report['latest_checkpoint_epoch']}",
        f"- Best checkpoint epoch: {report['best_checkpoint_epoch']}",
        "",
        "## Generated Files",
        f"- JSON report: {report_json}",
        f"- Loss curve: {plot_result.get('paths', {}).get('loss_curve', 'N/A')}",
        f"- LR curve: {plot_result.get('paths', {}).get('lr_curve', 'N/A')}",
        f"- NMSE curve: {plot_result.get('paths', {}).get('nmse_curve', 'N/A')}",
        f"- Rho curve: {plot_result.get('paths', {}).get('rho_curve', 'N/A')}",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return report_json, report_md, plot_result


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze CsiNet NPU training logs and artifacts")
    parser.add_argument("--outputdir", type=str, default="out_100k_3", help="training output directory")
    parser.add_argument("--summary", type=str, default="", help="explicit run_summary json path")
    parser.add_argument("--analysis-dir", type=str, default="", help="analysis output directory (default: <outputdir>/analysis_latest)")
    parser.add_argument("--no-plots", action="store_true", help="disable plot generation")
    return parser


def main():
    args = build_parser().parse_args()

    output_dir = Path(args.outputdir).expanduser().resolve()
    if args.summary and str(args.summary).strip():
        summary_path = Path(args.summary).expanduser().resolve()
    else:
        summary_path = _find_latest_summary(output_dir)
        if summary_path is None:
            raise FileNotFoundError(f"No run summary found under: {output_dir}")

    if args.analysis_dir and str(args.analysis_dir).strip():
        analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    else:
        analysis_dir = output_dir / "analysis_latest"

    report_json, report_md, plot_result = analyze(
        summary_path=summary_path,
        analysis_dir=analysis_dir,
        render_plots=not args.no_plots,
    )

    print(f"Summary analyzed: {summary_path}")
    print(f"Analysis JSON: {report_json}")
    print(f"Analysis Markdown: {report_md}")
    if plot_result.get("plots_enabled", False):
        print(f"Plot files: {plot_result.get('paths', {})}")
    else:
        print(f"Plot generation skipped/failed: {plot_result}")


if __name__ == "__main__":
    main()
