"""Training progress, history and plot persistence."""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def write_training_artifacts(model_dir: Path, history, step_history):
    write_json_atomic(model_dir / "history.json", history)
    write_json_atomic(model_dir / "step_history.json", step_history)
    write_step_history_csv(model_dir / "step_history.csv", step_history)
    plot_loss_curve(model_dir / "loss_curve.png", step_history, history)


def update_runtime_status(model_dir: Path, state: str, **fields):
    payload = {
        "state": state,
        "updated": datetime.now().isoformat(timespec="seconds"),
        **fields,
    }
    write_json_atomic(model_dir / "training_status.json", payload)


def write_step_history_csv(path: Path, step_history):
    fieldnames = [
        "global_step",
        "epoch",
        "step",
        "total_loss",
        "exist_loss",
        "type_loss",
        "param_loss",
        "weight_loss",
        "global_loss",
        "d_presence_loss",
        "spacing_loss",
        "reconstruction_loss",
        "count_loss",
        "component_count_accuracy",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_history:
            train = row.get("train", {})
            writer.writerow(
                {
                    "global_step": row.get("global_step"),
                    "epoch": row.get("epoch"),
                    "step": row.get("step"),
                    "total_loss": train.get("total_loss"),
                    "exist_loss": train.get("exist_loss"),
                    "type_loss": train.get("type_loss"),
                    "param_loss": train.get("param_loss"),
                    "weight_loss": train.get("weight_loss"),
                    "global_loss": train.get("global_loss"),
                    "d_presence_loss": train.get("d_presence_loss"),
                    "spacing_loss": train.get("spacing_loss"),
                    "reconstruction_loss": train.get("reconstruction_loss"),
                    "count_loss": train.get("count_loss"),
                    "component_count_accuracy": train.get("component_count_accuracy"),
                }
            )
    tmp.replace(path)


def load_json_list(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def count_samples(dataset_dir: Path, split: str) -> int:
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if "split_counts" in meta and split in meta["split_counts"]:
            return int(meta["split_counts"][split])

    total = 0
    for shard in sorted((dataset_dir / split).glob("*.npz")):
        with np.load(shard) as data:
            total += int(data["x"].shape[0])
    if total:
        return total
    for shard in sorted((dataset_dir / split).glob("*.tfrecord")):
        total += sum(1 for _ in tf.data.TFRecordDataset([str(shard)]))
    return total


def scalar_dict(metrics):
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def plot_loss_curve(path: Path, step_history, epoch_history):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    if step_history:
        x = [int(row["global_step"]) for row in step_history]
        y = [float(row["train"]["total_loss"]) for row in step_history]
        axes[0].plot(x, y, lw=0.9, label="train step total_loss")
        axes[0].set_xlabel("global step")
        axes[0].set_ylabel("loss")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No step history yet", ha="center", va="center")
    axes[0].grid(True, alpha=0.25)

    if epoch_history:
        epochs = [int(row["epoch"]) for row in epoch_history]
        train_loss = [float(row["train"]["total_loss"]) for row in epoch_history]
        val_loss = [float(row["val"]["total_loss"]) for row in epoch_history]
        axes[1].plot(epochs, train_loss, marker="o", label="train epoch total_loss")
        axes[1].plot(epochs, val_loss, marker="o", label="val epoch total_loss")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("loss")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No completed epoch yet", ha="center", va="center")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    tmp = path.with_suffix(path.suffix + ".tmp.png")
    fig.savefig(tmp, dpi=160)
    plt.close(fig)
    tmp.replace(path)


def write_json_atomic(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def mean_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(np.mean([float(m[k]) for m in metrics_list])) for k in keys}
