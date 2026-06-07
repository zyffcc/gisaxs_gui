#!/usr/bin/env python3
"""Inspect one TFRecord sample and compare stored curves to label forward model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import component_array_to_dict, evaluate_clean, global_array_to_dict
from TrainSetBuild.tfrecord_io import SAMPLE_SPECS, parse_example


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_dir",
        default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train",
        help="Training split directory containing TFRecord shards.",
    )
    p.add_argument("--sample_index", type=int, default=0, help="Index after optional shuffle/filtering.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed, matching overfit_debug.py by default.")
    p.add_argument("--shuffle_buffer", type=int, default=50000)
    p.add_argument("--any_sample", action="store_true", help="Do not filter to strict K=1 samples.")
    p.add_argument("--raw_order", action="store_true", help="Disable shuffle and read samples in shard order.")
    return p.parse_args()


def finite_summary(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    shape = arr.shape
    dtype = arr.dtype
    if arr.size == 0:
        return f"shape={shape} dtype={dtype} empty"
    if arr.dtype == np.bool_:
        vals = arr.astype(np.float64)
    elif np.issubdtype(arr.dtype, np.number):
        vals = arr.astype(np.float64)
    else:
        return f"shape={shape} dtype={dtype}"
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return f"shape={shape} dtype={dtype} all_nonfinite"
    return (
        f"shape={shape} dtype={dtype} "
        f"min={float(np.min(finite)):.9g} max={float(np.max(finite)):.9g} "
        f"mean={float(np.mean(finite)):.9g}"
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(a - b))))


def log_rmse(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-30
    return rmse(np.log(np.maximum(a, eps)), np.log(np.maximum(b, eps)))


def load_raw_feature_keys(shards: list[Path]) -> list[str]:
    for raw in tf.data.TFRecordDataset([str(s) for s in shards]).take(1):
        ex = tf.train.Example()
        ex.ParseFromString(bytes(raw.numpy()))
        return sorted(ex.features.feature.keys())
    return []


def select_sample(train_dir: Path, sample_index: int, seed: int, shuffle_buffer: int, strict_k1: bool, raw_order: bool):
    shards = sorted(train_dir.glob("*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found in {train_dir}")

    ds = tf.data.TFRecordDataset([str(s) for s in shards])
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if not raw_order:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)

    seen = 0
    for sample in ds:
        sample_np = {k: v.numpy() for k, v in sample.items()}
        nonempty = int(np.sum(sample_np["slot_exist"] > 0.5))
        if strict_k1 and nonempty != 1:
            continue
        if seen == sample_index:
            return sample_np, shards
        seen += 1
    mode = "strict K=1 " if strict_k1 else ""
    raise RuntimeError(f"Could not find {mode}sample_index={sample_index} in {train_dir}")


def valid_curve(sample: dict):
    mask = np.asarray(sample["point_mask"]).astype(bool)
    return (
        np.asarray(sample["q"], dtype=np.float64)[mask],
        np.asarray(sample["I_clean"], dtype=np.float64)[mask],
        np.asarray(sample["I_noisy"], dtype=np.float64)[mask],
        np.asarray(sample["sigma"], dtype=np.float64)[mask],
        np.asarray(sample["x"], dtype=np.float64)[mask],
        mask,
    )


def true_components_and_global(sample: dict):
    components_stored = []
    components_denorm_masked = []
    slot_exist = np.asarray(sample["slot_exist"])
    slot_type = np.asarray(sample["slot_type"])
    slot_params_phys = np.asarray(sample["slot_params_phys"])
    slot_params_norm = np.asarray(sample["slot_params_norm"])
    slot_weight = np.asarray(sample["slot_weight"])

    for slot in np.where(slot_exist > 0.5)[0]:
        tid = int(slot_type[slot])
        params_phys = np.asarray(slot_params_phys[slot], dtype=np.float64)
        param_mask = np.asarray(sample["slot_param_mask"][slot], dtype=np.float32)
        effective_mask = param_mask * schema.effective_param_mask(tid, params_phys)
        params_from_norm_raw = schema.denormalize_params(slot_params_norm[slot], tid)
        params_from_norm_masked = schema.denormalize_params_with_mask(slot_params_norm[slot], tid, effective_mask)
        if not np.allclose(params_phys, params_from_norm_raw, rtol=1e-5, atol=1e-6):
            print(f"WARNING slot {slot}: slot_params_phys differs from raw denormalized slot_params_norm")
            print(f"  stored_phys      = {params_phys}")
            print(f"  denorm_from_norm = {params_from_norm_raw}")
        if not np.allclose(params_phys, params_from_norm_masked, rtol=1e-5, atol=1e-6):
            print(f"WARNING slot {slot}: slot_params_phys differs from masked denormalized slot_params_norm")
            print(f"  stored_phys             = {params_phys}")
            print(f"  denorm_from_norm_masked = {params_from_norm_masked}")
            print(f"  effective_param_mask    = {effective_mask}")
        components_stored.append(component_array_to_dict(tid, params_phys, float(slot_weight[slot])))
        components_denorm_masked.append(component_array_to_dict(tid, params_from_norm_masked, float(slot_weight[slot])))

    global_phys = np.asarray(sample["global_params_phys"], dtype=np.float64)
    global_from_norm_raw = schema.denormalize_global(np.asarray(sample["global_params_norm"], dtype=np.float64))
    global_from_norm_masked = schema.denormalize_global_with_optional_zero(np.asarray(sample["global_params_norm"], dtype=np.float64))
    if not np.allclose(global_phys, global_from_norm_raw, rtol=1e-5, atol=1e-6):
        print("WARNING global_params_phys differs from raw denormalized global_params_norm")
        print(f"  stored_phys      = {global_phys}")
        print(f"  denorm_from_norm = {global_from_norm_raw}")
    if not np.allclose(global_phys, global_from_norm_masked, rtol=1e-5, atol=1e-6):
        print("WARNING global_params_phys differs from optional-zero denormalized global_params_norm")
        print(f"  stored_phys              = {global_phys}")
        print(f"  denorm_from_norm_masked  = {global_from_norm_masked}")
    return components_stored, components_denorm_masked, global_phys, global_from_norm_masked


def print_key_summaries(sample: dict, raw_feature_keys: list[str]):
    print("\n=== Raw TFRecord feature keys ===")
    for key in raw_feature_keys:
        known = "known" if key in SAMPLE_SPECS or key == "sampling_mode" else "unknown"
        print(f"{key}: {known}")

    print("\n=== Parsed sample keys / shape / min / max / mean ===")
    for key in sorted(sample.keys()):
        print(f"{key}: {finite_summary(sample[key])}")


def print_focus_fields(sample: dict, q, i_clean, i_noisy, sigma, x):
    global_features = np.asarray(sample["global_features"], dtype=np.float64)
    i_offset = float(global_features[3])
    i_scale = float(global_features[4])
    print("\n=== Focus fields ===")
    print(f"q: {finite_summary(q)}")
    print(f"I_clean: {finite_summary(i_clean)}")
    print(f"I_noisy: {finite_summary(i_noisy)}")
    print(f"sigma: {finite_summary(sigma)}")
    print(f"x(valid): {finite_summary(x)}")
    print(f"global_features: {global_features}")
    print(f"I_offset = global_features[3] = {i_offset:.12g}")
    print(f"I_scale  = global_features[4] = {i_scale:.12g}")
    print(f"slot_params_norm:\n{np.asarray(sample['slot_params_norm'])}")
    print(f"slot_weight: {np.asarray(sample['slot_weight'])}")
    print(f"global_params_norm: {np.asarray(sample['global_params_norm'])}")
    print(f"global_params_phys stored: {np.asarray(sample['global_params_phys'])}")
    print(f"raw denormalized global: {schema.denormalize_global(np.asarray(sample['global_params_norm']))}")
    print(f"optional-zero denormalized global: {schema.denormalize_global_with_optional_zero(np.asarray(sample['global_params_norm']))}")


def compare_curves(sample: dict, q, i_clean, i_noisy, x):
    components_stored, components_denorm_masked, global_phys, global_from_norm_masked = true_components_and_global(sample)
    stored_phys_oracle = evaluate_clean(q, components_stored, global_array_to_dict(global_phys))
    norm_denorm_masked_oracle = evaluate_clean(q, components_denorm_masked, global_array_to_dict(global_from_norm_masked))

    global_features = np.asarray(sample["global_features"], dtype=np.float64)
    i_offset = float(global_features[3])
    i_scale = float(global_features[4])
    x_log_i_inverse = np.exp(x[:, 1] * i_scale + i_offset)

    print("\n=== True label forward model ===")
    print(f"stored components = {[{'type': c['type_name'], 'weight': c['weight']} for c in components_stored]}")
    print(f"stored global = {global_array_to_dict(global_phys)}")
    print(f"masked denorm components = {[{'type': c['type_name'], 'weight': c['weight']} for c in components_denorm_masked]}")
    print(f"masked denorm global = {global_array_to_dict(global_from_norm_masked)}")
    print(f"stored_phys_oracle: {finite_summary(stored_phys_oracle)}")
    print(f"norm_denorm_masked_oracle: {finite_summary(norm_denorm_masked_oracle)}")

    print("\n=== Curve RMSE comparisons against stored_phys_oracle ===")
    comparisons = {
        "I_clean": i_clean,
        "I_noisy": i_noisy,
        "exp(x[:,1] * I_scale + I_offset)": x_log_i_inverse,
    }
    for key, curve in comparisons.items():
        print(f"{key}: RMSE={rmse(curve, stored_phys_oracle):.12g} logRMSE={log_rmse(curve, stored_phys_oracle):.12g}")

    print("\n=== Curve RMSE comparisons against norm_denorm_masked_oracle ===")
    for key, curve in comparisons.items():
        print(f"{key}: RMSE={rmse(curve, norm_denorm_masked_oracle):.12g} logRMSE={log_rmse(curve, norm_denorm_masked_oracle):.12g}")
    print(
        "stored_phys_oracle vs norm_denorm_masked_oracle: "
        f"RMSE={rmse(stored_phys_oracle, norm_denorm_masked_oracle):.12g} "
        f"logRMSE={log_rmse(stored_phys_oracle, norm_denorm_masked_oracle):.12g}"
    )

    print("\n=== Other possible normalized/intensity-like curve fields ===")
    found = False
    n = len(q)
    for key, value in sorted(sample.items()):
        arr = np.asarray(value)
        if key in comparisons or key in {"q", "I_clean", "I_noisy", "sigma", "x", "point_mask"}:
            continue
        key_lower = key.lower()
        is_curve_len = arr.shape == (schema.MAX_POINTS,)
        is_curve_matrix = arr.shape == (schema.MAX_POINTS, 3)
        looks_like_intensity = any(token in key_lower for token in ("i", "intensity", "norm", "curve", "log"))
        if not looks_like_intensity or not (is_curve_len or is_curve_matrix):
            continue
        found = True
        mask = np.asarray(sample["point_mask"]).astype(bool)
        if is_curve_len:
            curve = arr[mask].astype(np.float64)
            if np.all(np.isfinite(curve)) and np.all(curve > 0):
                print(f"{key}: RMSE={rmse(curve, stored_phys_oracle):.12g} logRMSE={log_rmse(curve, stored_phys_oracle):.12g}")
            else:
                print(f"{key}: not positive curve, {finite_summary(curve)}")
        else:
            for col in range(arr.shape[1]):
                curve = arr[mask, col].astype(np.float64)
                print(f"{key}[:,{col}]: raw normalized column, {finite_summary(curve)}")
                if n == len(curve) and np.all(np.isfinite(curve)):
                    inv = np.exp(curve * i_scale + i_offset)
                    print(
                        f"{key}[:,{col}] exp(col*I_scale+I_offset): "
                        f"RMSE={rmse(inv, stored_phys_oracle):.12g} logRMSE={log_rmse(inv, stored_phys_oracle):.12g}"
                    )
    if not found:
        print("No extra normalized/intensity-like curve fields found beyond x/I_clean/I_noisy/sigma.")


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    strict_k1 = not args.any_sample
    sample, shards = select_sample(
        train_dir=train_dir,
        sample_index=args.sample_index,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        strict_k1=strict_k1,
        raw_order=args.raw_order,
    )
    raw_feature_keys = load_raw_feature_keys(shards)
    q, i_clean, i_noisy, sigma, x, _ = valid_curve(sample)

    print(f"train_dir = {train_dir}")
    print(f"sample_index = {args.sample_index}")
    print(f"selection = {'raw shard order' if args.raw_order else f'shuffled seed={args.seed}'}")
    print(f"strict_k1 = {strict_k1}")
    print(f"valid_points = {len(q)}")

    print_key_summaries(sample, raw_feature_keys)
    print_focus_fields(sample, q, i_clean, i_noisy, sigma, x)
    compare_curves(sample, q, i_clean, i_noisy, x)


if __name__ == "__main__":
    main()
