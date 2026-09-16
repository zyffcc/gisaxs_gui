"""Train the 1D GISAXS slot model."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Training import data_loader
from Training.losses import LossWeights, compute_losses
from Training.model import build_model, build_training_model
from Training.differentiable_physics import configure_dataset_physics
from TrainSetBuild import schema


STOP_REQUESTED = False


def request_graceful_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpoint will be saved after the current step.", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS")
    p.add_argument("--model_dir", default="/data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS")
    p.add_argument("--warmstart_model", default=None, help="Optional previous model.keras; matching legacy V7 layers are copied by name.")
    p.add_argument("--multisolution_dir", default=None, help="Optional aligned 16-mode training sidecar.")
    p.add_argument("--multisolution_fraction", type=float, default=0.50)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_points", type=int, default=schema.MAX_POINTS)
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--reconstruction_loss_weight", type=float, default=0.0)
    p.add_argument("--reconstruction_start_epoch", type=int, default=6)
    p.add_argument("--reconstruction_ramp_epochs", type=int, default=5)
    p.add_argument("--reconstruction_q_stride", type=int, default=16)
    p.add_argument("--reconstruction_samples_per_batch", type=int, default=2)
    p.add_argument("--type_loss_weight", type=float, default=1.0)
    p.add_argument("--set_count_loss_weight", type=float, default=0.5)
    p.add_argument("--quality_loss_weight", type=float, default=0.20)
    p.add_argument(
        "--physics_curriculum",
        action="store_true",
        help="Use the 60-epoch staged branch-conditioned multi-scale physics schedule.",
    )
    p.add_argument("--branch_condition_probability", type=float, default=0.50)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_skipped_nonfinite_batches", type=int, default=10)
    p.add_argument(
        "--num_hypotheses",
        type=int,
        default=16,
        help="Maximum over-complete proposal capacity; the activation gate returns a variable subset.",
    )
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


def update_runtime_status(model_dir: Path, state: str, **fields):
    payload = {
        "state": state,
        "updated": datetime.now().isoformat(timespec="seconds"),
        **fields,
    }
    write_json_atomic(model_dir / "training_status.json", payload)


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
    fieldnames = [
        "global_step", "epoch", "step", "total_loss", "set_coverage_loss", "dustbin_loss", "tier_quality_loss",
        "ground_truth_modes", "exist_loss", "type_loss",
        "param_loss", "weight_loss", "global_loss", "d_presence_loss", "spacing_loss", "reconstruction_loss", "physics_coverage_loss",
        "count_loss", "component_count_accuracy",
        "diversity_loss", "hypothesis_usage_balance_loss", "hypothesis_confidence_loss", "active_hypotheses", "hypothesis_usage_entropy",
        "winner_activation_loss", "activation_quality_loss", "candidate_complexity_loss",
        "expected_active_candidates", "pseudo_active_candidates",
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
                    "set_coverage_loss": train.get("set_coverage_loss"),
                    "dustbin_loss": train.get("dustbin_loss"),
                    "tier_quality_loss": train.get("tier_quality_loss"),
                    "ground_truth_modes": train.get("ground_truth_modes"),
                    "exist_loss": train.get("exist_loss"),
                    "type_loss": train.get("type_loss"),
                    "param_loss": train.get("param_loss"),
                    "weight_loss": train.get("weight_loss"),
                    "global_loss": train.get("global_loss"),
                    "d_presence_loss": train.get("d_presence_loss"),
                    "spacing_loss": train.get("spacing_loss"),
                    "reconstruction_loss": train.get("reconstruction_loss"),
                    "physics_coverage_loss": train.get("physics_coverage_loss"),
                    "count_loss": train.get("count_loss"),
                    "component_count_accuracy": train.get("component_count_accuracy"),
                    "diversity_loss": train.get("diversity_loss"),
                    "hypothesis_usage_balance_loss": train.get("hypothesis_usage_balance_loss"),
                    "hypothesis_confidence_loss": train.get("hypothesis_confidence_loss"),
                    "active_hypotheses": train.get("active_hypotheses"),
                    "hypothesis_usage_entropy": train.get("hypothesis_usage_entropy"),
                    "winner_activation_loss": train.get("winner_activation_loss"),
                    "activation_quality_loss": train.get("activation_quality_loss"),
                    "candidate_complexity_loss": train.get("candidate_complexity_loss"),
                    "expected_active_candidates": train.get("expected_active_candidates"),
                    "pseudo_active_candidates": train.get("pseudo_active_candidates"),
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
    signal.signal(signal.SIGTERM, request_graceful_stop)
    signal.signal(signal.SIGINT, request_graceful_stop)
    if args.max_points != schema.MAX_POINTS:
        raise ValueError(f"This first version expects max_points={schema.MAX_POINTS}; got {args.max_points}")
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if args.reconstruction_loss_weight < 0:
        raise ValueError("--reconstruction_loss_weight must be >= 0")
    if args.reconstruction_start_epoch < 1:
        raise ValueError("--reconstruction_start_epoch must be >= 1")
    if args.reconstruction_ramp_epochs < 1:
        raise ValueError("--reconstruction_ramp_epochs must be >= 1")
    if args.reconstruction_q_stride < 1:
        raise ValueError("--reconstruction_q_stride must be >= 1")
    if args.reconstruction_samples_per_batch < 1:
        raise ValueError("--reconstruction_samples_per_batch must be >= 1")
    if args.type_loss_weight < 0:
        raise ValueError("--type_loss_weight must be >= 0")
    if args.set_count_loss_weight < 0:
        raise ValueError("--set_count_loss_weight must be >= 0")
    if args.quality_loss_weight < 0:
        raise ValueError("--quality_loss_weight must be >= 0")
    if not 0.0 <= args.branch_condition_probability <= 1.0:
        raise ValueError("--branch_condition_probability must be between 0 and 1")
    if args.max_skipped_nonfinite_batches < 0:
        raise ValueError("--max_skipped_nonfinite_batches must be >= 0")
    if args.num_hypotheses < 2:
        raise ValueError("--num_hypotheses must be >= 2 for multi-solution training")
    if not 0.0 <= args.multisolution_fraction <= 1.0:
        raise ValueError("--multisolution_fraction must be between zero and one")

    tf.keras.utils.set_random_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    dataset_metadata = {}
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        dataset_metadata = json.loads(metadata_path.read_text())
    global_norm_version = dataset_metadata.get(
        "global_normalization_version", schema.V5_GLOBAL_NORM_VERSION
    )
    profile = configure_dataset_physics(dataset_metadata)
    print(f"Training physics dataset profile: {profile}", flush=True)
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
    train_steps = max(1, train_count // args.batch_size)
    val_steps = max(1, val_count // args.batch_size)

    train_ds = data_loader.make_dataset(
        dataset_dir,
        "train",
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        max_samples=train_count,
        drop_remainder=True,
        multisolution_dir=args.multisolution_dir,
        multisolution_fraction=args.multisolution_fraction if args.multisolution_dir else 0.0,
    )
    val_ds = data_loader.make_dataset(
        dataset_dir,
        "val",
        args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
        max_samples=val_count,
        drop_remainder=False,
        multisolution_dir=args.multisolution_dir,
        multisolution_fraction=0.0,
    )
    # Apply cardinality limits while these are still regular tf.data.Dataset
    # objects. DistributedDataset intentionally does not expose .take().
    train_ds = train_ds.take(train_steps)
    val_ds = val_ds.take(val_steps)

    logical_gpus = tf.config.list_logical_devices("GPU")
    strategy = None
    if args.multi_gpu:
        if len(logical_gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.", flush=True)
        else:
            raise RuntimeError(f"--multi_gpu requires at least two visible GPUs; found {len(logical_gpus)}")

    def apply_warmstart(target_model):
        if not args.warmstart_model:
            return 0
        source_path = Path(args.warmstart_model)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        legacy = None
        load_errors = []
        for include_resolution_head in (True, False):
            candidate = build_model(
                num_hypotheses=args.num_hypotheses,
                include_resolution_presence_head=include_resolution_head,
            )
            try:
                candidate.load_weights(source_path)
            except ValueError as exc:
                load_errors.append(exc)
                continue
            legacy = candidate
            break
        if legacy is None:
            raise ValueError(
                f"Warm-start artifact is incompatible with both supported model variants: {source_path}"
            ) from load_errors[-1]
        source_layers = {layer.name: layer for layer in legacy.layers}
        copied = 0
        for layer in target_model.layers:
            source = source_layers.get(layer.name)
            if source is None or not source.weights:
                continue
            source_weights, target_weights = source.get_weights(), layer.get_weights()
            if len(source_weights) == len(target_weights) and all(
                left.shape == right.shape for left, right in zip(source_weights, target_weights)
            ):
                layer.set_weights(source_weights); copied += 1
        del legacy
        print(f"Warm-started {copied} weighted layers from {source_path}", flush=True)
        return copied

    if strategy is not None:
        with strategy.scope():
            model, inference_model = build_training_model(
                num_hypotheses=args.num_hypotheses
            )
            warmstart_layer_count = apply_warmstart(model)
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
            optimizer.build(model.trainable_variables)
            reconstruction_weight_var = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="reconstruction_loss_weight")
            reconstruction_q_stride_var = tf.Variable(args.reconstruction_q_stride, dtype=tf.int32, trainable=False, name="reconstruction_q_stride")
            reconstruction_samples_var = tf.Variable(args.reconstruction_samples_per_batch, dtype=tf.int32, trainable=False, name="reconstruction_samples_per_batch")
            reconstruction_sampling_mode_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="reconstruction_sampling_mode")
            reconstruction_min_points_var = tf.Variable(64, dtype=tf.int32, trainable=False, name="reconstruction_min_points")
            reconstruction_max_points_var = tf.Variable(128, dtype=tf.int32, trainable=False, name="reconstruction_max_points")
            branch_condition_probability_var = tf.Variable(args.branch_condition_probability, dtype=tf.float32, trainable=False, name="branch_condition_probability")
    else:
        model, inference_model = build_training_model(
            num_hypotheses=args.num_hypotheses
        )
        warmstart_layer_count = apply_warmstart(model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
        optimizer.build(model.trainable_variables)
        reconstruction_weight_var = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="reconstruction_loss_weight")
        reconstruction_q_stride_var = tf.Variable(args.reconstruction_q_stride, dtype=tf.int32, trainable=False, name="reconstruction_q_stride")
        reconstruction_samples_var = tf.Variable(args.reconstruction_samples_per_batch, dtype=tf.int32, trainable=False, name="reconstruction_samples_per_batch")
        reconstruction_sampling_mode_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="reconstruction_sampling_mode")
        reconstruction_min_points_var = tf.Variable(64, dtype=tf.int32, trainable=False, name="reconstruction_min_points")
        reconstruction_max_points_var = tf.Variable(128, dtype=tf.int32, trainable=False, name="reconstruction_max_points")
        branch_condition_probability_var = tf.Variable(args.branch_condition_probability, dtype=tf.float32, trainable=False, name="branch_condition_probability")
    loss_weights = LossWeights(
        type=args.type_loss_weight,
        set_count=args.set_count_loss_weight,
        quality=args.quality_loss_weight,
        reconstruction=reconstruction_weight_var,
        reconstruction_q_stride=reconstruction_q_stride_var,
        reconstruction_samples_per_batch=reconstruction_samples_var,
        reconstruction_sampling_mode=reconstruction_sampling_mode_var,
        reconstruction_multiscale_min_points=reconstruction_min_points_var,
        reconstruction_multiscale_max_points=reconstruction_max_points_var,
    )

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

    update_runtime_status(
        model_dir,
        "initialized",
        epoch=int(ckpt_epoch.numpy()),
        step=int(ckpt_step.numpy()),
        global_step=int(global_step.numpy()),
        train_steps=train_steps,
        val_steps=val_steps,
        replicas=1 if strategy is None else strategy.num_replicas_in_sync,
    )

    def branch_condition_inputs(inputs, labels):
        """Mix free-mode examples with explicit ground-truth geometry branches."""
        if "solution_mask" in labels:
            # Full solution-set supervision contains mostly cross-branch
            # alternatives.  Some V5 records carry the anchor generation
            # branch as a hard input constraint, which would make those labels
            # impossible for the decoder to represent.  Neutralize only the
            # structural constraints; q/intensity features and the shared
            # global-parameter constraints remain sample-specific.
            unconstrained = dict(inputs)
            unconstrained["type_allowed"] = tf.ones_like(inputs["type_allowed"])
            unconstrained["force_exist"] = -tf.ones_like(inputs["force_exist"])
            unconstrained["param_low_norm"] = tf.zeros_like(inputs["param_low_norm"])
            unconstrained["param_high_norm"] = tf.ones_like(inputs["param_high_norm"])
            unconstrained["param_range_mask"] = tf.ones_like(inputs["param_range_mask"])
            unconstrained["d_allowed"] = tf.ones_like(inputs["d_allowed"])
            unconstrained["d_spacing_rule"] = tf.zeros_like(inputs["d_spacing_rule"])
            return unconstrained
        batch = tf.shape(labels["slot_type"])[0]
        choose = tf.random.uniform([batch], dtype=tf.float32) < branch_condition_probability_var
        conditioned = dict(inputs)
        fixed_types = tf.one_hot(
            tf.cast(labels["slot_type"], tf.int32), schema.NUM_TYPES, dtype=tf.float32
        )
        fixed_exist = tf.where(
            tf.cast(labels["slot_exist"], tf.float32) > 0.5,
            tf.ones_like(labels["slot_exist"], tf.float32),
            tf.zeros_like(labels["slot_exist"], tf.float32),
        )
        conditioned["type_allowed"] = tf.where(
            choose[:, tf.newaxis, tf.newaxis], fixed_types, tf.cast(inputs["type_allowed"], tf.float32)
        )
        conditioned["force_exist"] = tf.where(
            choose[:, tf.newaxis], fixed_exist, tf.cast(inputs["force_exist"], tf.float32)
        )
        return conditioned

    def train_step_fn(inputs, labels):
        with tf.GradientTape() as tape:
            preds = model(branch_condition_inputs(inputs, labels), training=True)
            losses = compute_losses(labels, preds, loss_weights)
            loss = losses["total_loss"]
            optimization_loss = loss if strategy is None else loss / float(strategy.num_replicas_in_sync)
        grads = tape.gradient(optimization_loss, model.trainable_variables)
        finite_grads = [tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None]
        local_finite = tf.reduce_all(
            tf.stack([tf.reduce_all(tf.math.is_finite(v)) for v in losses.values()] + finite_grads)
        )
        if strategy is not None:
            replica_context = tf.distribute.get_replica_context()
            # TensorFlow 2.15 only exposes SUM and MEAN for distributed
            # reductions.  Every replica must report a finite result.
            finite_replica_count = replica_context.all_reduce(
                tf.distribute.ReduceOp.SUM, tf.cast(local_finite, tf.int32)
            )
            globally_finite = finite_replica_count == strategy.num_replicas_in_sync
        else:
            globally_finite = local_finite

        # Keras' distributed optimizer performs a merge_call internally, so it
        # cannot live inside tf.cond while tracing strategy.run.  Keep one
        # unconditional optimizer path and feed it zero gradients whenever any
        # replica is non-finite.  The outer loop records/rejects such batches.
        safe_grads = [
            None if grad is None else tf.where(globally_finite, grad, tf.zeros_like(grad))
            for grad in grads
        ]
        optimizer.apply_gradients(zip(safe_grads, model.trainable_variables))
        update_applied = tf.cast(globally_finite, tf.float32)
        result = dict(losses)
        result["gradient_global_norm"] = tf.linalg.global_norm([g for g in grads if g is not None])
        result["update_applied"] = update_applied
        return result

    def val_step_fn(inputs, labels):
        preds = model(branch_condition_inputs(inputs, labels), training=False)
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
    print(
        f"Fixed train batches: drop_remainder=True, discarded_per_epoch={train_count - train_steps * args.batch_size}",
        flush=True,
    )
    print(
        f"Validation batches: drop_remainder=False, evaluated_up_to={min(val_count, val_steps * args.batch_size)}",
        flush=True,
    )

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

    nonfinite_events = load_json_list(model_dir / "nonfinite_batches.json")
    skipped_nonfinite_batches = len(nonfinite_events)

    def curriculum_state(epoch):
        if not args.physics_curriculum:
            if args.reconstruction_loss_weight > 0.0 and epoch >= args.reconstruction_start_epoch:
                ramp_step = epoch - args.reconstruction_start_epoch + 1
                weight = args.reconstruction_loss_weight * min(ramp_step / args.reconstruction_ramp_epochs, 1.0)
            else:
                weight = 0.0
            return {
                "phase": "legacy",
                "weight": weight,
                "q_stride": args.reconstruction_q_stride,
                "samples_per_replica": args.reconstruction_samples_per_batch,
                "sampling_mode": 0,
                "min_points": 64,
                "max_points": 128,
                "branch_probability": args.branch_condition_probability,
                "learning_rate": args.learning_rate,
            }
        if epoch <= 5:
            return {"phase": "coarse_supervised", "weight": 0.0, "q_stride": 16, "samples_per_replica": 1, "sampling_mode": 0, "min_points": 64, "max_points": 64, "branch_probability": 0.25, "learning_rate": args.learning_rate}
        if epoch <= 15:
            t = (epoch - 6) / 9.0
            return {"phase": "physics_ramp", "weight": 0.01 + 0.04 * t, "q_stride": 16, "samples_per_replica": 2, "sampling_mode": 0, "min_points": 64, "max_points": 64, "branch_probability": 0.50, "learning_rate": args.learning_rate}
        if epoch <= 40:
            t = (epoch - 16) / 24.0
            return {"phase": "physics_strong", "weight": 0.05 + 0.10 * t, "q_stride": 8, "samples_per_replica": 4, "sampling_mode": 0, "min_points": 96, "max_points": 128, "branch_probability": 0.75, "learning_rate": args.learning_rate * 0.5}
        t = min(max((epoch - 41) / 19.0, 0.0), 1.0)
        return {"phase": "multiscale_finetune", "weight": 0.15 + 0.15 * t, "q_stride": 1, "samples_per_replica": 4, "sampling_mode": 1, "min_points": 64, "max_points": 128, "branch_probability": 0.75, "learning_rate": args.learning_rate * 0.2}

    for epoch in range(start_epoch, args.epochs + 1):
        schedule = curriculum_state(epoch)
        physical_weight = float(schedule["weight"])
        reconstruction_weight_var.assign(physical_weight)
        reconstruction_q_stride_var.assign(int(schedule["q_stride"]))
        reconstruction_samples_var.assign(int(schedule["samples_per_replica"]))
        reconstruction_sampling_mode_var.assign(int(schedule["sampling_mode"]))
        reconstruction_min_points_var.assign(int(schedule["min_points"]))
        reconstruction_max_points_var.assign(int(schedule["max_points"]))
        branch_condition_probability_var.assign(float(schedule["branch_probability"]))
        optimizer.learning_rate.assign(float(schedule["learning_rate"]))
        print(
            f"epoch {epoch}: phase={schedule['phase']} reconstruction_loss_weight={physical_weight:.6g} "
            f"q_stride={schedule['q_stride']} samples_per_replica={schedule['samples_per_replica']} "
            f"sampling_mode={schedule['sampling_mode']} points={schedule['min_points']}-{schedule['max_points']} "
            f"branch_probability={schedule['branch_probability']:.3f} learning_rate={schedule['learning_rate']:.3g}",
            flush=True,
        )
        update_runtime_status(
            model_dir,
            "running",
            epoch=epoch,
            step=resume_step if epoch == start_epoch else 0,
            global_step=int(global_step.numpy()),
            train_steps=train_steps,
            reconstruction_loss_weight=float(physical_weight),
            curriculum=schedule,
        )
        train_metrics = []
        for step, (inputs, labels) in enumerate(train_ds, start=1):
            if epoch == start_epoch and step <= resume_step:
                continue
            m = train_step(inputs, labels)
            raw_metrics = {k: v.numpy() for k, v in m.items()}
            train_row = scalar_dict(raw_metrics)
            update_applied = train_row.pop("update_applied")
            loss_value = float(train_row["total_loss"])
            if update_applied < 0.5:
                skipped_nonfinite_batches += 1
                if strategy is None:
                    bad_slot_type = np.asarray(labels["slot_type"]).astype(int)
                    bad_slot_exist = np.asarray(labels["slot_exist"]).astype(float)
                else:
                    bad_slot_type = np.concatenate(
                        [np.asarray(v).astype(int) for v in strategy.experimental_local_results(labels["slot_type"])],
                        axis=0,
                    )
                    bad_slot_exist = np.concatenate(
                        [np.asarray(v).astype(float) for v in strategy.experimental_local_results(labels["slot_exist"])],
                        axis=0,
                    )
                event = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "global_step": int(global_step.numpy()),
                    "losses": train_row,
                    "slot_type": bad_slot_type.tolist(),
                    "slot_exist": bad_slot_exist.tolist(),
                }
                nonfinite_events.append(event)
                write_json_atomic(model_dir / "nonfinite_batches.json", nonfinite_events)
                update_runtime_status(
                    model_dir,
                    "skipped_nonfinite_batch",
                    epoch=epoch,
                    step=step,
                    global_step=int(global_step.numpy()),
                    train_steps=train_steps,
                    skipped_nonfinite_batches=skipped_nonfinite_batches,
                    diagnostic=event,
                )
                print(
                    f"WARNING: skipped non-finite batch epoch={epoch} step={step}; "
                    f"skipped_total={skipped_nonfinite_batches}",
                    flush=True,
                )
                if skipped_nonfinite_batches > args.max_skipped_nonfinite_batches:
                    raise RuntimeError(
                        f"Exceeded --max_skipped_nonfinite_batches={args.max_skipped_nonfinite_batches}; "
                        f"see {model_dir / 'nonfinite_batches.json'}"
                    )
                continue
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Non-finite total_loss at epoch={epoch}, step={step}: {loss_value}")
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
                update_runtime_status(
                    model_dir,
                    "running",
                    epoch=epoch,
                    step=step,
                    global_step=global_step_value,
                    train_steps=train_steps,
                    reconstruction_loss_weight=float(physical_weight),
                    curriculum=schedule,
                    latest_train=train_row,
                )
            if args.save_interval > 0 and (global_step_value % args.save_interval == 0 or step == train_steps):
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(step)
                manager.save(checkpoint_number=global_step_value)
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                inference_model.save(model_dir / "model.keras", overwrite=True)
                print(f"saved progress at epoch {epoch} step {step}/{train_steps} global_step={global_step_value}", flush=True)
            if STOP_REQUESTED:
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(step)
                manager.save(checkpoint_number=global_step_value)
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                inference_model.save(model_dir / "model.keras", overwrite=True)
                update_runtime_status(
                    model_dir,
                    "interrupted_checkpoint_saved",
                    epoch=epoch,
                    step=step,
                    global_step=global_step_value,
                    train_steps=train_steps,
                    latest_train=train_row,
                )
                print(f"Graceful stop checkpoint saved at epoch {epoch} step {step}.", flush=True)
                return

        if not train_metrics:
            print(f"epoch {epoch}: no new training steps after resume skip; moving to validation.", flush=True)

        val_metrics = []
        for step, (inputs, labels) in enumerate(val_ds, start=1):
            m = val_step(inputs, labels)
            loss_value = float(m["total_loss"].numpy())
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Non-finite validation total_loss at epoch={epoch}, step={step}: {loss_value}")
            val_metrics.append(scalar_dict({k: v.numpy() for k, v in m.items()}))
            if STOP_REQUESTED:
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(train_steps)
                manager.save(checkpoint_number=int(global_step.numpy()))
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                inference_model.save(model_dir / "model.keras", overwrite=True)
                update_runtime_status(
                    model_dir,
                    "interrupted_checkpoint_saved",
                    epoch=epoch,
                    step=train_steps,
                    global_step=int(global_step.numpy()),
                    train_steps=train_steps,
                )
                print(f"Graceful stop checkpoint saved during epoch {epoch} validation.", flush=True)
                return

        tr = mean_metrics(train_metrics)
        va = mean_metrics(val_metrics)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={tr['total_loss']:.5f} val_loss={va['total_loss']:.5f} "
            f"val_type_acc={va['slot_type_accuracy']:.3f} val_nonempty_acc={va['nonempty_type_accuracy']:.3f} "
            f"val_K_acc={va['component_count_accuracy']:.3f}",
            f"active_H={va.get('active_hypotheses', float('nan')):.2f} "
            f"gate_H={va.get('expected_active_candidates', float('nan')):.2f} "
            f"physics_H={va.get('pseudo_active_candidates', float('nan')):.2f} "
            f"diversity={va.get('diversity_loss', float('nan')):.4g}",
            flush=True,
        )
        best_row = min(history, key=lambda item: float(item["val"]["total_loss"]))
        update_runtime_status(
            model_dir,
            "epoch_complete",
            epoch=epoch,
            step=train_steps,
            global_step=int(global_step.numpy()),
            train_steps=train_steps,
            reconstruction_loss_weight=float(physical_weight),
            curriculum=schedule,
            latest_train=tr,
            latest_val=va,
            best_val_loss=float(best_row["val"]["total_loss"]),
            best_val_epoch=int(best_row["epoch"]),
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
        inference_model.save(model_dir / "model.keras", overwrite=True)
        print(f"saved epoch {epoch} checkpoint/model artifacts", flush=True)

    config = {
        "max_points": schema.MAX_POINTS,
        "max_slots": schema.MAX_SLOTS,
        "num_types": schema.NUM_TYPES,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_hypotheses": args.num_hypotheses,
        "multisolution_dir": args.multisolution_dir,
        "multisolution_fraction": args.multisolution_fraction,
        "warmstart_model": args.warmstart_model,
        "warmstart_layer_count": warmstart_layer_count,
        "architecture": {
            "name": "universal_v7_multisolution_set_decoder",
            "independent_mode_queries": True,
            "shared_curve_encoder": True,
            "explicit_branch_embedding": True,
            "shared_global_nuisance_head": True,
            "ordinal_tier_quality_head": [0.01, 0.03, 0.05],
            "set_to_set_sinkhorn_dustbin_loss": bool(args.multisolution_dir),
            "set_supervision_relaxes_anchor_structural_constraints": bool(args.multisolution_dir),
        },
        "variable_candidate_cardinality": {
            "independent_activation_head": True,
            "inactive_candidates_excluded_from_diversity": True,
            "physics_quality_pseudo_labels": True,
            "physical_verification_required_at_inference": True,
        },
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
        "dataset_profile": profile,
        "global_normalization_version": global_norm_version,
        "reconstruction_start_epoch": args.reconstruction_start_epoch,
        "reconstruction_ramp_epochs": args.reconstruction_ramp_epochs,
        "reconstruction_q_stride": args.reconstruction_q_stride,
        "reconstruction_samples_per_batch": args.reconstruction_samples_per_batch,
        "type_loss_weight": args.type_loss_weight,
        "set_count_loss_weight": args.set_count_loss_weight,
        "quality_loss_weight": args.quality_loss_weight,
        "physics_curriculum": bool(args.physics_curriculum),
        "physics_curriculum_schedule": {
            "epochs_1_5": {"weight": 0.0, "phase": "coarse_supervised"},
            "epochs_6_15": {"weight": [0.01, 0.05], "q_stride": 16, "samples_per_replica": 2},
            "epochs_16_40": {"weight": [0.05, 0.15], "q_stride": 8, "samples_per_replica": 4},
            "epochs_41_60": {"weight": [0.15, 0.30], "multiscale_points": [64, 128], "samples_per_replica": 4},
        },
        "branch_condition_probability": args.branch_condition_probability,
        "d_constraint_model": {
            "explicit_presence_head": True,
            "spacing_rules": schema.D_RULE_NAMES,
        },
    }
    with (model_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    src_meta = dataset_dir / "metadata.json"
    if src_meta.exists():
        shutil.copy2(src_meta, model_dir / "dataset_metadata.json")

    inference_model.save(model_dir / "model.keras", overwrite=True)
    inference_model.save(model_dir / "saved_model")
    write_training_artifacts(model_dir, history, step_history)
    update_runtime_status(
        model_dir,
        "complete",
        epoch=args.epochs,
        step=train_steps,
        global_step=int(global_step.numpy()),
        train_steps=train_steps,
        latest_train=history[-1]["train"] if history else None,
        latest_val=history[-1]["val"] if history else None,
    )
    print(f"Training complete. Model written to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
