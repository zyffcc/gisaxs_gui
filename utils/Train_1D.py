"""Train the 1D GISAXS slot model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Training import data_loader
from Training.losses import LossWeights, compute_losses
from Training.model import build_model
from TrainSetBuild import schema


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS")
    p.add_argument("--model_dir", default="/data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_points", type=int, default=schema.MAX_POINTS)
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--reconstruction_loss_weight", type=float, default=0.0)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def mean_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(np.mean([float(m[k]) for m in metrics_list])) for k in keys}


def write_json_atomic(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_json_list(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def scalar_dict(metrics):
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def write_step_history_csv(path: Path, step_history):
    fieldnames = ["global_step", "epoch", "step", "total_loss", "exist_loss", "type_loss", "param_loss", "weight_loss", "global_loss"]
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
                }
            )
    tmp.replace(path)


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


def write_training_artifacts(model_dir: Path, history, step_history):
    write_json_atomic(model_dir / "history.json", history)
    write_json_atomic(model_dir / "step_history.json", step_history)
    write_step_history_csv(model_dir / "step_history.csv", step_history)
    plot_loss_curve(model_dir / "loss_curve.png", step_history, history)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    if args.max_points != schema.MAX_POINTS:
        raise ValueError(f"This first version expects max_points={schema.MAX_POINTS}; got {args.max_points}")
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if args.reconstruction_loss_weight != 0.0:
        print("Note: differentiable reconstruction loss is not used in this first version; physics verification runs in predict_topk.py.", flush=True)

    tf.keras.utils.set_random_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)
    (model_dir / "logs").mkdir(exist_ok=True)

    train_count = count_samples(dataset_dir, "train")
    val_count = count_samples(dataset_dir, "val")
    if args.quick_test:
        train_count = min(train_count, 64)
        val_count = min(val_count, 32)
        args.epochs = min(args.epochs, 2)
        args.save_interval = min(args.save_interval, 2)
    train_steps = max(1, math.ceil(train_count / args.batch_size))
    val_steps = max(1, math.ceil(val_count / args.batch_size))

    train_ds = data_loader.make_dataset(dataset_dir, "train", args.batch_size, shuffle=True, seed=args.seed, max_samples=train_count)
    val_ds = data_loader.make_dataset(dataset_dir, "val", args.batch_size, shuffle=False, seed=args.seed + 1, max_samples=val_count)

    logical_gpus = tf.config.list_logical_devices("GPU")
    strategy = None
    if args.multi_gpu:
        if len(logical_gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.", flush=True)
        else:
            print("--multi_gpu was passed, but fewer than two logical GPUs are visible; using single-device training.", flush=True)

    loss_weights = LossWeights()

    if strategy is not None:
        with strategy.scope():
            model = build_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    else:
        model = build_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    ckpt_epoch = tf.Variable(1, dtype=tf.int64, trainable=False)
    ckpt_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=ckpt_epoch, step=ckpt_step, global_step=global_step)
    manager = tf.train.CheckpointManager(ckpt, str(model_dir / "checkpoints"), max_to_keep=20)
    writer = tf.summary.create_file_writer(str(model_dir / "logs"))

    history = load_json_list(model_dir / "history.json")
    step_history = load_json_list(model_dir / "step_history.json")
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if int(ckpt_epoch.numpy()) == 1 and int(ckpt_step.numpy()) == 0 and int(global_step.numpy()) == 0:
            match = re.search(r"ckpt-(\d+)$", manager.latest_checkpoint)
            if match and not step_history:
                legacy_epoch = int(match.group(1))
                ckpt_epoch.assign(legacy_epoch + 1)
                global_step.assign(legacy_epoch * train_steps)
                print(
                    f"Interpreting legacy checkpoint {manager.latest_checkpoint} as completed epoch {legacy_epoch}.",
                    flush=True,
                )
        print(
            f"Restored checkpoint {manager.latest_checkpoint}: "
            f"epoch={int(ckpt_epoch.numpy())}, step={int(ckpt_step.numpy())}, global_step={int(global_step.numpy())}",
            flush=True,
        )
    elif step_history:
        global_step.assign(int(step_history[-1]["global_step"]))
        print(f"No checkpoint found, but loaded existing step history through global_step={int(global_step.numpy())}.", flush=True)

    def train_step_fn(inputs, labels):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            losses = compute_losses(labels, preds, loss_weights)
            loss = losses["total_loss"]
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    def val_step_fn(inputs, labels):
        preds = model(inputs, training=False)
        return compute_losses(labels, preds, loss_weights)

    if strategy is not None:
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds = strategy.experimental_distribute_dataset(val_ds)

        @tf.function
        def train_step(inputs, labels):
            per_replica = strategy.run(train_step_fn, args=(inputs, labels))
            return {k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None) for k, v in per_replica.items()}

        @tf.function
        def val_step(inputs, labels):
            per_replica = strategy.run(val_step_fn, args=(inputs, labels))
            return {k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None) for k, v in per_replica.items()}
    else:
        train_step = tf.function(train_step_fn)
        val_step = tf.function(val_step_fn)

    print(f"Training samples={train_count}, val samples={val_count}, steps={train_steps}/{val_steps}", flush=True)
    print(f"Intervals: log_interval={args.log_interval}, save_interval={args.save_interval}", flush=True)

    start_epoch = int(ckpt_epoch.numpy())
    resume_step = int(ckpt_step.numpy())
    if resume_step >= train_steps:
        start_epoch += 1
        resume_step = 0
        ckpt_epoch.assign(start_epoch)
        ckpt_step.assign(0)
    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch}; requested epochs={args.epochs}. Nothing to train.", flush=True)
        write_training_artifacts(model_dir, history, step_history)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = []
        for step, (inputs, labels) in enumerate(train_ds.take(train_steps), start=1):
            if epoch == start_epoch and step <= resume_step:
                continue
            m = train_step(inputs, labels)
            loss_value = float(m["total_loss"].numpy())
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Non-finite total_loss at epoch={epoch}, step={step}: {loss_value}")
            train_row = scalar_dict({k: v.numpy() for k, v in m.items()})
            train_metrics.append(train_row)
            global_step_value = int(global_step.assign_add(1).numpy())
            step_history.append(
                {
                    "global_step": global_step_value,
                    "epoch": int(epoch),
                    "step": int(step),
                    "train": train_row,
                }
            )
            with writer.as_default():
                for k, v in train_row.items():
                    tf.summary.scalar(f"train_step/{k}", v, step=global_step_value)
            if args.log_interval > 0 and (step % args.log_interval == 0 or step == train_steps):
                print(f"epoch {epoch} train step {step}/{train_steps} global_step={global_step_value} loss={loss_value:.5f}", flush=True)
            if args.save_interval > 0 and (global_step_value % args.save_interval == 0 or step == train_steps):
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(step)
                manager.save(checkpoint_number=global_step_value)
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                model.save(model_dir / "model.keras", overwrite=True)
                print(f"saved progress at epoch {epoch} step {step}/{train_steps} global_step={global_step_value}", flush=True)

        if not train_metrics:
            print(f"epoch {epoch}: no new training steps after resume skip; moving to validation.", flush=True)

        val_metrics = []
        for step, (inputs, labels) in enumerate(val_ds.take(val_steps), start=1):
            m = val_step(inputs, labels)
            loss_value = float(m["total_loss"].numpy())
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Non-finite validation total_loss at epoch={epoch}, step={step}: {loss_value}")
            val_metrics.append(scalar_dict({k: v.numpy() for k, v in m.items()}))

        tr = mean_metrics(train_metrics)
        va = mean_metrics(val_metrics)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={tr['total_loss']:.5f} val_loss={va['total_loss']:.5f} "
            f"val_type_acc={va['slot_type_accuracy']:.3f} val_nonempty_acc={va['nonempty_type_accuracy']:.3f}",
            flush=True,
        )
        with writer.as_default():
            for k, v in tr.items():
                tf.summary.scalar(f"train/{k}", v, step=epoch)
            for k, v in va.items():
                tf.summary.scalar(f"val/{k}", v, step=epoch)
        writer.flush()
        ckpt_epoch.assign(epoch + 1)
        ckpt_step.assign(0)
        manager.save(checkpoint_number=int(global_step.numpy()))
        write_training_artifacts(model_dir, history, step_history)
        model.save(model_dir / "model.keras", overwrite=True)
        print(f"saved epoch {epoch} checkpoint/model artifacts", flush=True)

    config = {
        "max_points": schema.MAX_POINTS,
        "max_slots": schema.MAX_SLOTS,
        "num_types": schema.NUM_TYPES,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
    }
    with (model_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    src_meta = dataset_dir / "metadata.json"
    if src_meta.exists():
        shutil.copy2(src_meta, model_dir / "dataset_metadata.json")

    model.save(model_dir / "model.keras", overwrite=True)
    model.save(model_dir / "saved_model")
    write_training_artifacts(model_dir, history, step_history)
    print(f"Training complete. Model written to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
