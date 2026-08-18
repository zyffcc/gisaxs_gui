#!/usr/bin/env python3
"""Overfit debug on a tiny K=1 training subset.

Goal:
- Verify dataset -> label -> model optimization chain can overfit 256 samples.
- Do not change model architecture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Training.overfit_cli import parse_args
from Training.overfit_evaluation import (
    plot_curves,
)

from TrainSetBuild import schema
from TrainSetBuild.tfrecord_io import parse_example
from Training.losses import LossWeights, compute_losses
from Training.model import build_model
from TrainSetBuild.sampling import pad_2d, preprocess_curve


def read_k1_subset(train_dir: Path, num_samples: int, seed: int):
    shards = sorted(train_dir.glob("*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found in {train_dir}")

    ds = tf.data.TFRecordDataset([str(s) for s in shards])
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(50000, seed=seed, reshuffle_each_iteration=False)

    selected = []
    for sample in ds:
        sample_np = {k: v.numpy() for k, v in sample.items()}
        # Keep only strict K=1 examples.
        nonempty = int(np.sum(sample_np["slot_exist"] > 0.5))
        if nonempty != 1:
            continue
        selected.append(sample_np)
        if len(selected) >= num_samples:
            break

    if len(selected) < num_samples:
        raise RuntimeError(
            f"Only found {len(selected)} K=1 samples in {train_dir}, requested {num_samples}"
        )
    return selected


def build_low_noise_inputs(sample: dict, noise_frac: float, actual_max_points: int):
    point_mask = np.asarray(sample["point_mask"]).astype(bool)
    q = np.asarray(sample["q"], dtype=np.float32)[point_mask]
    i_clean = np.asarray(sample["I_clean"], dtype=np.float32)[point_mask]
    sigma = np.maximum(noise_frac * np.maximum(i_clean, 1e-30), 1e-8).astype(np.float32)

    if len(q) > actual_max_points:
        idx = np.linspace(0, len(q) - 1, actual_max_points).astype(int)
        q = q[idx]
        i_clean = i_clean[idx]
        sigma = sigma[idx]

    x, global_features = preprocess_curve(q, i_clean, sigma)
    n = min(len(q), actual_max_points)

    point_mask_pad = np.zeros(actual_max_points, dtype=bool)
    point_mask_pad[:n] = True

    inputs = {
        "x": pad_2d(x[:n], actual_max_points, 3).astype(np.float32),
        "point_mask": point_mask_pad,
        "global_features": global_features.astype(np.float32),
        "type_allowed": np.asarray(sample["type_allowed"], dtype=np.float32),
        "param_low_norm": np.asarray(sample["param_low_norm"], dtype=np.float32),
        "param_high_norm": np.asarray(sample["param_high_norm"], dtype=np.float32),
        "param_range_mask": np.asarray(sample["param_range_mask"], dtype=np.float32),
        "force_exist": np.asarray(sample["force_exist"], dtype=np.float32),
        "global_low_norm": np.asarray(sample["global_low_norm"], dtype=np.float32),
        "global_high_norm": np.asarray(sample["global_high_norm"], dtype=np.float32),
        "global_range_mask": np.asarray(sample["global_range_mask"], dtype=np.float32),
        "d_allowed": np.asarray(sample["d_allowed"], dtype=np.float32),
        "d_spacing_rule": np.asarray(sample["d_spacing_rule"], dtype=np.float32),
    }

    slot_type = np.asarray(sample["slot_type"], dtype=np.int32)
    slot_params_phys = np.asarray(sample["slot_params_phys"], dtype=np.float32)
    slot_param_mask = np.asarray(sample["slot_param_mask"], dtype=np.float32).copy()
    for slot in np.where(np.asarray(sample["slot_exist"], dtype=np.float32) > 0.5)[0]:
        slot_param_mask[slot] *= schema.effective_param_mask(
            int(slot_type[slot]), slot_params_phys[slot]
        )

    labels = {
        "slot_type": slot_type,
        "slot_exist": np.asarray(sample["slot_exist"], dtype=np.float32),
        "slot_params_norm": np.asarray(sample["slot_params_norm"], dtype=np.float32),
        "slot_param_mask": slot_param_mask,
        "slot_params_phys": slot_params_phys,
        "slot_weight": np.asarray(sample["slot_weight"], dtype=np.float32),
        "global_params_norm": np.asarray(sample["global_params_norm"], dtype=np.float32),
        "global_params_phys": np.asarray(sample["global_params_phys"], dtype=np.float32),
        "d_spacing_rule": np.asarray(sample["d_spacing_rule"], dtype=np.float32),
    }
    return inputs, labels, q[:n], i_clean[:n]


def stack_dicts(items: list[dict]):
    keys = items[0].keys()
    return {k: np.stack([d[k] for d in items], axis=0) for k in keys}


def simple_k1_loss(labels, preds):
    slot_exist = tf.cast(labels["slot_exist"], tf.float32)
    slot_type = tf.cast(labels["slot_type"], tf.int32)
    slot_params_norm = tf.cast(labels["slot_params_norm"], tf.float32)
    slot_param_mask = tf.cast(labels["slot_param_mask"], tf.float32)
    global_params_norm = tf.cast(labels["global_params_norm"], tf.float32)

    exist_logit = tf.cast(preds["exist_logit"], tf.float32)
    type_logits = tf.cast(preds["type_logits"], tf.float32)
    param_mu_norm = tf.cast(preds["param_mu_norm"], tf.float32)
    global_mu_norm = tf.cast(preds["global_mu_norm"], tf.float32)

    true_slot = tf.argmax(slot_exist, axis=1, output_type=tf.int32)
    batch_idx = tf.range(tf.shape(slot_exist)[0], dtype=tf.int32)
    slot_idx = tf.stack([batch_idx, true_slot], axis=1)

    exist_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=slot_exist, logits=exist_logit)
    exist_loss = tf.reduce_mean(exist_ce)

    type_logits_true = tf.gather_nd(type_logits, slot_idx)
    type_true = tf.gather_nd(slot_type, slot_idx)
    type_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            type_true, type_logits_true, from_logits=True
        )
    )

    param_idx = tf.stack([batch_idx, true_slot, type_true], axis=1)
    param_pred = tf.gather_nd(param_mu_norm, param_idx)
    param_true = tf.gather_nd(slot_params_norm, slot_idx)
    param_mask = tf.gather_nd(slot_param_mask, slot_idx)
    param_mse_per = tf.reduce_sum(
        param_mask * tf.square(param_pred - param_true), axis=1
    ) / tf.maximum(tf.reduce_sum(param_mask, axis=1), 1.0)
    param_mse = tf.reduce_mean(param_mse_per)

    global_mse = tf.reduce_mean(tf.square(global_mu_norm - global_params_norm))
    total_loss = exist_loss + type_loss + 20.0 * param_mse + 5.0 * global_mse

    return {
        "total_loss": total_loss,
        "exist_loss": exist_loss,
        "type_loss": type_loss,
        "param_mse": param_mse,
        "global_mse": global_mse,
    }


def mae_accumulators_from_batch(preds: dict, labels: dict):
    exist_logits = preds["exist_logit"].numpy()
    type_logits = preds["type_logits"].numpy()
    param_mu_norm = preds["param_mu_norm"].numpy()
    global_mu_norm = preds["global_mu_norm"].numpy()

    slot_type = labels["slot_type"].numpy()
    slot_exist = labels["slot_exist"].numpy()
    slot_params_norm = labels["slot_params_norm"].numpy()
    slot_param_mask = labels["slot_param_mask"].numpy()
    global_params_norm = labels["global_params_norm"].numpy()

    type_ok = 0
    param_mae_sum = 0.0
    param_mae_count = 0
    global_mae_sum = float(np.sum(np.abs(global_mu_norm - global_params_norm)))
    global_mae_count = int(np.prod(global_mu_norm.shape))

    for b in range(slot_type.shape[0]):
        true_slots = np.where(slot_exist[b] > 0.5)[0]
        if true_slots.size == 0:
            continue
        j_true = int(true_slots[0])
        tid_true = int(slot_type[b, j_true])

        j_pred = int(np.argmax(exist_logits[b]))
        type_scores = np.asarray(type_logits[b, j_pred], dtype=np.float64)
        type_scores[schema.TYPE_EMPTY] = -1e9
        tid_pred = int(np.argmax(type_scores))
        if tid_pred == tid_true:
            type_ok += 1

        pred_params = param_mu_norm[b, j_pred, tid_true]
        true_params = slot_params_norm[b, j_true]
        mask = slot_param_mask[b, j_true] > 0.5
        if np.any(mask):
            param_mae_sum += float(np.mean(np.abs(pred_params[mask] - true_params[mask])))
            param_mae_count += 1

    return {
        "type_ok": int(type_ok),
        "type_total": int(slot_type.shape[0]),
        "param_mae_sum": float(param_mae_sum),
        "param_mae_count": int(param_mae_count),
        "global_mae_sum": float(global_mae_sum),
        "global_mae_count": int(global_mae_count),
    }


def evaluate_metrics_batched(
    model,
    x_train: dict,
    y_train: dict,
    loss_weights: LossWeights,
    eval_batch_size: int,
    use_simple_k1_loss: bool = False,
):
    n = int(next(iter(x_train.values())).shape[0])
    loss_weighted_sum = 0.0
    type_ok = 0
    type_total = 0
    param_mae_sum = 0.0
    param_mae_count = 0
    global_mae_sum = 0.0
    global_mae_count = 0
    loss_parts = {
        "exist_loss": 0.0,
        "type_loss": 0.0,
        "param_mse": 0.0,
        "global_mse": 0.0,
    }

    current_eval_bs = max(1, int(eval_batch_size))
    start = 0
    while start < n:
        bsz = min(current_eval_bs, n - start)
        end = start + bsz
        xb = {k: tf.convert_to_tensor(v[start:end]) for k, v in x_train.items()}
        yb = {k: tf.convert_to_tensor(v[start:end]) for k, v in y_train.items()}
        try:
            preds = model(xb, training=False)
            losses = (
                simple_k1_loss(yb, preds)
                if use_simple_k1_loss
                else compute_losses(yb, preds, loss_weights)
            )
        except tf.errors.ResourceExhaustedError:
            if bsz == 1:
                raise
            current_eval_bs = max(1, bsz // 2)
            print(
                f"OOM during eval batch; reducing eval_batch_size to {current_eval_bs} and retrying.",
                flush=True,
            )
            continue

        loss_weighted_sum += float(losses["total_loss"].numpy()) * bsz
        for key in loss_parts:
            if key in losses:
                loss_parts[key] += float(losses[key].numpy()) * bsz
            elif key == "param_mse" and "param_loss" in losses:
                loss_parts[key] += float(losses["param_loss"].numpy()) * bsz
            elif key == "global_mse" and "global_loss" in losses:
                loss_parts[key] += float(losses["global_loss"].numpy()) * bsz

        acc = mae_accumulators_from_batch(preds, yb)
        type_ok += acc["type_ok"]
        type_total += acc["type_total"]
        param_mae_sum += acc["param_mae_sum"]
        param_mae_count += acc["param_mae_count"]
        global_mae_sum += acc["global_mae_sum"]
        global_mae_count += acc["global_mae_count"]
        start = end

    train_loss = loss_weighted_sum / max(n, 1)
    type_acc = type_ok / max(type_total, 1)
    param_mae = param_mae_sum / max(param_mae_count, 1)
    global_mae = global_mae_sum / max(global_mae_count, 1)
    out = {
        "train_loss": float(train_loss),
        "type_accuracy": float(type_acc),
        "param_mae": float(param_mae),
        "global_mae": float(global_mae),
    }
    for key, val in loss_parts.items():
        out[key] = float(val / max(n, 1))
    return out


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    actual_max_points = min(schema.MAX_POINTS, int(args.debug_max_points))
    if actual_max_points < 16:
        raise ValueError("--debug_max_points must be >= 16")

    train_dir = Path(args.train_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading K=1 subset...", flush=True)
    raw_samples = read_k1_subset(train_dir, args.num_samples, args.seed)

    inputs_list = []
    labels_list = []
    q_list = []
    i_list = []

    for sample in raw_samples:
        inputs, labels, q, i_clean = build_low_noise_inputs(
            sample, args.noise_frac, actual_max_points
        )
        inputs_list.append(inputs)
        labels_list.append(labels)
        q_list.append(q)
        i_list.append(i_clean)

    x_train = stack_dicts(inputs_list)
    y_train = stack_dicts(labels_list)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(args.num_samples, seed=args.seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size, drop_remainder=False)

    model = build_model(max_points=actual_max_points)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    loss_weights = LossWeights()

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            if args.simple_k1_loss:
                losses = simple_k1_loss(labels, preds)
            else:
                losses = compute_losses(labels, preds, loss_weights)
            loss = losses["total_loss"]
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    history = []
    for epoch in range(1, args.epochs + 1):
        batch_losses = []
        for inputs_b, labels_b in train_ds:
            m = train_step(inputs_b, labels_b)
            batch_losses.append(float(m["total_loss"].numpy()))

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate_metrics_batched(
                model,
                x_train,
                y_train,
                loss_weights,
                eval_batch_size=args.eval_batch_size,
                use_simple_k1_loss=args.simple_k1_loss,
            )
            row = {
                "epoch": epoch,
                **metrics,
                "batch_loss_mean": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            }
            history.append(row)
            print(
                f"epoch {epoch:03d} | "
                f"train_loss={metrics['train_loss']:.6f} | "
                f"type_accuracy={metrics['type_accuracy']:.4f} | "
                f"param_mae={metrics['param_mae']:.6f} | "
                f"global_mae={metrics['global_mae']:.6f} | "
                f"exist_loss={metrics['exist_loss']:.6f} | "
                f"type_loss={metrics['type_loss']:.6f} | "
                f"param_mse={metrics['param_mse']:.6f} | "
                f"global_mse={metrics['global_mse']:.6f}",
                flush=True,
            )

    with (out_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(
        f"Rendering saved/oracle/predicted curves for first {min(args.plot_n, len(q_list))} samples...",
        flush=True,
    )
    plot_curves(
        model,
        x_train,
        y_train,
        q_list,
        i_list,
        out_dir,
        eval_batch_size=args.eval_batch_size,
        plot_n=args.plot_n,
        use_true_global_for_plot=args.use_true_global_for_plot,
    )

    if not args.skip_model_save:
        model.save(out_dir / "overfit_debug_model.keras", overwrite=True)
    print(f"Done. Outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
