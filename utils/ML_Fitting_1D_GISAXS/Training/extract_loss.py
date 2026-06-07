#!/usr/bin/env python3
"""Extract training loss curves from Slurm .out logs.

Parses lines like:
- epoch 1 train step 20/1407 loss=15.02061
- epoch 1: train_loss=1.93972 val_loss=-2.22697 val_type_acc=0.754 val_nonempty_acc=0.554
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

STEP_RE = re.compile(
    r"^epoch\s+(\d+)\s+train step\s+(\d+)/(\d+)\s+loss=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$",
    re.IGNORECASE,
)

EPOCH_RE = re.compile(
    r"^epoch\s+(\d+):\s+train_loss=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"val_loss=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s+val_type_acc=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?"
    r"(?:\s+val_nonempty_acc=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?",
    re.IGNORECASE,
)


def parse_log(log_path: Path):
    step_rows = []
    epoch_rows = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            m_step = STEP_RE.match(line)
            if m_step:
                epoch = int(m_step.group(1))
                step = int(m_step.group(2))
                total_steps = int(m_step.group(3))
                loss = float(m_step.group(4))
                global_step = (epoch - 1) * total_steps + step
                step_rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "total_steps": total_steps,
                        "global_step": global_step,
                        "loss": loss,
                    }
                )
                continue

            m_epoch = EPOCH_RE.match(line)
            if m_epoch:
                epoch_rows.append(
                    {
                        "epoch": int(m_epoch.group(1)),
                        "train_loss": float(m_epoch.group(2)),
                        "val_loss": float(m_epoch.group(3)),
                        "val_type_acc": float(m_epoch.group(4)) if m_epoch.group(4) is not None else None,
                        "val_nonempty_acc": float(m_epoch.group(5)) if m_epoch.group(5) is not None else None,
                    }
                )

    return step_rows, epoch_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(plot_path: Path, step_rows: list[dict], epoch_rows: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plot: matplotlib unavailable ({exc})")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)

    if step_rows:
        xs = [r["global_step"] for r in step_rows]
        ys = [r["loss"] for r in step_rows]
        axes[0].plot(xs, ys, linewidth=0.9)
        axes[0].set_title("Step Loss")
        axes[0].set_xlabel("Global Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No step loss parsed", ha="center", va="center")

    if epoch_rows:
        epochs = [r["epoch"] for r in epoch_rows]
        train_loss = [r["train_loss"] for r in epoch_rows]
        val_loss = [r["val_loss"] for r in epoch_rows]
        axes[1].plot(epochs, train_loss, marker="o", label="train_loss")
        axes[1].plot(epochs, val_loss, marker="o", label="val_loss")
        axes[1].set_title("Epoch Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No epoch summary parsed", ha="center", va="center")

    if epoch_rows:
        acc_rows = [
            r
            for r in epoch_rows
            if r["val_type_acc"] is not None and r["val_nonempty_acc"] is not None
        ]
        if acc_rows:
            acc_epochs = [r["epoch"] for r in acc_rows]
            val_type_acc = [r["val_type_acc"] for r in acc_rows]
            val_nonempty_acc = [r["val_nonempty_acc"] for r in acc_rows]
            axes[2].plot(acc_epochs, val_type_acc, marker="o", label="val_type_acc")
            axes[2].plot(acc_epochs, val_nonempty_acc, marker="o", label="val_nonempty_acc")
            axes[2].set_title("Validation Accuracy")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_ylim(0.0, 1.0)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No validation accuracy parsed", ha="center", va="center")
    else:
        axes[2].text(0.5, 0.5, "No epoch summary parsed", ha="center", va="center")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract loss changes from training .out logs")
    parser.add_argument("--log", required=True, help="Path to slurm .out log")
    parser.add_argument(
        "--out_prefix",
        default=None,
        help="Output prefix without extension (default: same folder + log stem)",
    )
    parser.add_argument("--no_plot", action="store_true", help="Do not create PNG plot")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        out_prefix = log_path.with_suffix("")

    step_rows, epoch_rows = parse_log(log_path)

    step_csv = out_prefix.parent / f"{out_prefix.name}_step_loss.csv"
    epoch_csv = out_prefix.parent / f"{out_prefix.name}_epoch_loss.csv"

    write_csv(step_csv, step_rows, ["epoch", "step", "total_steps", "global_step", "loss"])
    write_csv(
        epoch_csv,
        epoch_rows,
        ["epoch", "train_loss", "val_loss", "val_type_acc", "val_nonempty_acc"],
    )

    print(f"Parsed {len(step_rows)} step records -> {step_csv}")
    print(f"Parsed {len(epoch_rows)} epoch records -> {epoch_csv}")

    if not args.no_plot:
        plot_path = out_prefix.parent / f"{out_prefix.name}_loss.png"
        save_plot(plot_path, step_rows, epoch_rows)


if __name__ == "__main__":
    main()
