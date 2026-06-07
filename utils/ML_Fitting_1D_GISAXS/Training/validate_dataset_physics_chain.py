#!/usr/bin/env python3
"""Validate dataset label -> physics forward consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import component_array_to_dict, evaluate_clean, global_array_to_dict
from TrainSetBuild.tfrecord_io import parse_example


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--logrmse_threshold", type=float, default=1e-5)
    return p.parse_args()


def log_rmse(a, b):
    eps = 1e-30
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((np.log(np.maximum(a, eps)) - np.log(np.maximum(b, eps))) ** 2)))


def components_from_stored(sample):
    components = []
    for slot in np.where(sample["slot_exist"] > 0.5)[0]:
        type_id = int(sample["slot_type"][slot])
        components.append(
            component_array_to_dict(
                type_id,
                np.asarray(sample["slot_params_phys"][slot], dtype=np.float64),
                float(sample["slot_weight"][slot]),
            )
        )
    return components, np.asarray(sample["global_params_phys"], dtype=np.float64)


def components_from_norm_masked(sample):
    components = []
    for slot in np.where(sample["slot_exist"] > 0.5)[0]:
        type_id = int(sample["slot_type"][slot])
        mask = np.asarray(sample["slot_param_mask"][slot], dtype=np.float32)
        if "slot_params_phys" in sample:
            mask = mask * schema.effective_param_mask(type_id, sample["slot_params_phys"][slot])
        params_phys = schema.denormalize_params_with_mask(sample["slot_params_norm"][slot], type_id, mask)
        components.append(component_array_to_dict(type_id, params_phys, float(sample["slot_weight"][slot])))
    return components, schema.denormalize_global_with_optional_zero(sample["global_params_norm"])


def main():
    args = parse_args()
    shards = sorted(Path(args.train_dir).glob("*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found in {args.train_dir}")

    failures = 0
    checked = 0
    ds = tf.data.TFRecordDataset([str(s) for s in shards]).map(parse_example)
    for tensor_sample in ds.take(args.max_samples):
        sample = {k: v.numpy() for k, v in tensor_sample.items()}
        mask = np.asarray(sample["point_mask"]).astype(bool)
        q = np.asarray(sample["q"], dtype=np.float64)[mask]
        i_clean = np.asarray(sample["I_clean"], dtype=np.float64)[mask]

        stored_components, stored_global = components_from_stored(sample)
        norm_components, norm_global = components_from_norm_masked(sample)
        stored_curve = evaluate_clean(q, stored_components, global_array_to_dict(stored_global))
        norm_curve = evaluate_clean(q, norm_components, global_array_to_dict(norm_global))

        stored_logrmse = log_rmse(i_clean, stored_curve)
        norm_logrmse = log_rmse(i_clean, norm_curve)
        ok = stored_logrmse < args.logrmse_threshold and norm_logrmse < args.logrmse_threshold
        status = "OK" if ok else "FAIL"
        print(
            f"{checked:05d} {status} "
            f"stored_phys_logRMSE={stored_logrmse:.6g} "
            f"norm_denorm_masked_logRMSE={norm_logrmse:.6g}",
            flush=True,
        )
        failures += int(not ok)
        checked += 1

    print(f"checked={checked} failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
