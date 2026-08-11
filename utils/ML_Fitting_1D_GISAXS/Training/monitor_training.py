#!/usr/bin/env python3
"""Report Slurm execution, saved progress, and convergence for one training run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


STEP_RE = re.compile(
    r"epoch\s+(\d+)\s+train step\s+(\d+)/(\d+)\s+global_step=(\d+)\s+loss=([^\s]+)"
)


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def slurm_status(job_id: str | None):
    if not job_id:
        return {"state": "UNKNOWN", "job_id": None}
    command = [
        "sacct", "-X", "-j", str(job_id), "-n", "-P",
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    main = next((row for row in rows if row[0] == str(job_id)), rows[0] if rows else None)
    if main is None:
        return {"state": "UNKNOWN", "job_id": str(job_id), "error": result.stderr.strip()}
    return {
        "job_id": str(job_id),
        "state": main[1].split()[0],
        "exit_code": main[2],
        "elapsed": main[3],
        "node": main[4],
    }


def latest_log_progress(log_path: Path | None):
    if log_path is None or not log_path.exists():
        return None
    last = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = STEP_RE.search(line)
            if match:
                last = {
                    "epoch": int(match.group(1)),
                    "step": int(match.group(2)),
                    "train_steps": int(match.group(3)),
                    "global_step": int(match.group(4)),
                    "loss": float(match.group(5)),
                }
    return last


def convergence(history):
    if not history:
        return {"state": "no_epoch_completed", "epochs": 0}
    val = np.asarray([float(row["val"]["total_loss"]) for row in history], dtype=np.float64)
    train = np.asarray([float(row["train"]["total_loss"]) for row in history], dtype=np.float64)
    best_idx = int(np.argmin(val))
    result = {
        "epochs": len(history),
        "latest_epoch": int(history[-1]["epoch"]),
        "latest_train_loss": float(train[-1]),
        "latest_val_loss": float(val[-1]),
        "best_val_loss": float(val[best_idx]),
        "best_val_epoch": int(history[best_idx]["epoch"]),
        "latest_val_K_accuracy": float(history[-1]["val"].get("component_count_accuracy", np.nan)),
        "latest_val_type_accuracy": float(history[-1]["val"].get("nonempty_type_accuracy", np.nan)),
    }
    if not np.all(np.isfinite(val)) or not np.all(np.isfinite(train)):
        result.update(state="diverged_nonfinite", overfit_warning=True)
        return result
    if len(history) < 3:
        result.update(state="warming_up", overfit_warning=False)
        return result
    window = min(5, len(history))
    x = np.arange(window, dtype=np.float64)
    val_slope = float(np.polyfit(x, val[-window:], 1)[0])
    train_slope = float(np.polyfit(x, train[-window:], 1)[0])
    scale = max(float(np.mean(np.abs(val[-window:]))), 1e-6)
    relative_val_slope = val_slope / scale
    if relative_val_slope < -0.005:
        state = "improving"
    elif relative_val_slope > 0.005:
        state = "degrading"
    else:
        state = "plateau_candidate"
    result.update(
        state=state,
        recent_window=window,
        relative_val_slope_per_epoch=relative_val_slope,
        overfit_warning=bool(train_slope < 0.0 and val_slope > 0.0),
        epochs_since_best=int(history[-1]["epoch"]) - int(history[best_idx]["epoch"]),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--log", help="Optional explicit Slurm stdout path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    log_path = Path(args.log) if args.log else None
    if log_path is None and args.job_id:
        matches = sorted(Path("Training/HCPoutput").glob(f"*-{args.job_id}.out"))
        log_path = matches[-1] if matches else None
    history = load_json(model_dir / "history.json", [])
    heartbeat = load_json(model_dir / "training_status.json", {})
    # Sort by the numeric training step (lexicographic order puts ckpt-500
    # after ckpt-1000).
    checkpoints = sorted(
        (model_dir / "checkpoints").glob("ckpt-*.index"),
        key=lambda path: int(path.stem.removeprefix("ckpt-")),
    )
    report = {
        "checked": datetime.now().isoformat(timespec="seconds"),
        "slurm": slurm_status(args.job_id),
        "model_dir": str(model_dir),
        "heartbeat": heartbeat,
        "latest_log_progress": latest_log_progress(log_path),
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "convergence": convergence(history),
        "log_path": str(log_path) if log_path else None,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"Slurm: {report['slurm']['state']} job={args.job_id} elapsed={report['slurm'].get('elapsed', '?')}")
    progress = report["latest_log_progress"] or heartbeat
    if progress:
        print(
            f"Progress: epoch={progress.get('epoch', '?')} step={progress.get('step', '?')}/"
            f"{progress.get('train_steps', '?')} global_step={progress.get('global_step', '?')}"
        )
    print(f"Checkpoint: {report['latest_checkpoint'] or 'none'}")
    conv = report["convergence"]
    print(f"Convergence: {conv['state']} epochs={conv['epochs']}")
    if conv.get("latest_val_loss") is not None:
        print(
            f"Loss: train={conv['latest_train_loss']:.6g} val={conv['latest_val_loss']:.6g} "
            f"best_val={conv['best_val_loss']:.6g}@epoch{conv['best_val_epoch']}"
        )
        print(
            f"Validation: K_acc={conv['latest_val_K_accuracy']:.4f} "
            f"type_acc={conv['latest_val_type_accuracy']:.4f} overfit_warning={conv.get('overfit_warning', False)}"
        )
    if report["log_path"]:
        print(f"Log: {report['log_path']}")


if __name__ == "__main__":
    main()
