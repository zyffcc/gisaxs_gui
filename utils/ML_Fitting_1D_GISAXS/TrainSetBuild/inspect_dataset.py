"""Inspect generated TFRecord/NPZ shards and export example curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import sampling, schema
from TrainSetBuild.tfrecord_io import parse_example


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--num_plots", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def sample_ground_truth(sample, shard_path: Path, storage: str, sample_index: int) -> dict:
    components = []
    for slot in range(schema.MAX_SLOTS):
        type_id = int(sample["slot_type"][slot])
        if type_id == schema.TYPE_EMPTY or float(sample["slot_exist"][slot]) <= 0.5:
            continue
        mask = sample["slot_param_mask"][slot].astype(bool)
        params_phys = {
            name: float(sample["slot_params_phys"][slot, idx])
            for idx, name in enumerate(schema.PARAM_NAMES)
            if mask[idx]
        }
        params_norm = {
            name: float(sample["slot_params_norm"][slot, idx])
            for idx, name in enumerate(schema.PARAM_NAMES)
            if mask[idx]
        }
        components.append(
            {
                "slot": slot,
                "type_id": type_id,
                "type": schema.TYPE_NAMES[type_id],
                "weight": float(sample["slot_weight"][slot]),
                "params_phys": params_phys,
                "params_norm": params_norm,
            }
        )

    global_params_phys = {
        name: float(sample["global_params_phys"][idx])
        for idx, name in enumerate(schema.GLOBAL_PARAM_NAMES)
    }
    global_params_norm = {
        name: float(sample["global_params_norm"][idx])
        for idx, name in enumerate(schema.GLOBAL_PARAM_NAMES)
    }
    point_mask = sample["point_mask"].astype(bool)
    sampling_mode = int(np.asarray(sample.get("sampling_mode", -1)).item())
    return {
        "source": {
            "storage": storage,
            "shard": str(shard_path),
            "sample_index_in_loaded_records": int(sample_index),
        },
        "sampling_mode": sampling_mode,
        "sampling_mode_name": sampling.SAMPLING_MODE_NAMES.get(sampling_mode, "unknown"),
        "n_points": int(np.sum(point_mask)),
        "q_min": float(np.min(sample["q"][point_mask])),
        "q_max": float(np.max(sample["q"][point_mask])),
        "components": components,
        "global_params_phys": global_params_phys,
        "global_params_norm": global_params_norm,
        "slot_type": sample["slot_type"].astype(int).tolist(),
        "slot_exist": sample["slot_exist"].astype(float).tolist(),
        "slot_weight": sample["slot_weight"].astype(float).tolist(),
    }


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    shards = sorted((dataset_dir / args.split).glob("*.tfrecord"))
    storage = "tfrecord"
    if not shards:
        shards = sorted((dataset_dir / args.split).glob("*.npz"))
        storage = "npz"
    if not shards:
        raise FileNotFoundError(f"No shards found under {dataset_dir / args.split}")
    rng = np.random.default_rng(args.seed)
    shard_path = shards[int(rng.integers(0, len(shards)))]
    print(f"Loaded {shard_path}")

    if storage == "tfrecord":
        import tensorflow as tf

        records = []
        for sample in tf.data.TFRecordDataset([str(shard_path)]).map(parse_example).take(max(args.num_plots, 8)):
            records.append({k: v.numpy() for k, v in sample.items()})
        if not records:
            raise ValueError(f"No records found in {shard_path}")
        keys = list(records[0].keys())
        for key in keys:
            arr = np.stack([r[key] for r in records], axis=0)
            finite = np.isfinite(arr).all() if arr.dtype.kind in "fc" else True
            msg = f"{key:22s} sample_shape={arr.shape[1:]} batch_checked={arr.shape[0]} dtype={arr.dtype} finite={finite}"
            if arr.dtype.kind in "fci" and arr.size:
                msg += f" min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g}"
            print(msg)
        samples = records
    else:
        data = np.load(shard_path)
        samples = [{key: data[key][i] for key in data.files} for i in range(data["q"].shape[0])]
        for key in data.files:
            arr = data[key]
            finite = np.isfinite(arr).all() if arr.dtype.kind in "fc" else True
            msg = f"{key:22s} shape={arr.shape} dtype={arr.dtype} finite={finite}"
            if arr.dtype.kind in "fci" and arr.size:
                msg += f" min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g}"
            print(msg)

    out_dir = dataset_dir / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = rng.choice(len(samples), size=min(args.num_plots, len(samples)), replace=False)
    for rank, idx in enumerate(picks):
        sample = samples[int(idx)]
        mask = sample["point_mask"].astype(bool)
        q = sample["q"][mask]
        noisy = sample["I_noisy"][mask]
        clean = sample["I_clean"][mask]
        sigma = sample["sigma"][mask]
        plt.figure(figsize=(7, 5))
        plt.loglog(q, noisy, ".", ms=3, label="noisy")
        plt.loglog(q, clean, "-", lw=1.2, label="clean")
        plt.xlabel("q / nm^-1")
        plt.ylabel("I")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"curve_{rank:02d}.png", dpi=160)
        plt.close()
        if rank == 0:
            csv = np.column_stack([q, noisy, sigma])
            np.savetxt(out_dir / "example_curve.csv", csv, delimiter=",", header="q,I,sigma", comments="")
            csv_full = np.column_stack([q, noisy, sigma, clean])
            np.savetxt(out_dir / "example_curve_with_clean.csv", csv_full, delimiter=",", header="q,I,sigma,I_clean", comments="")
            truth = sample_ground_truth(sample, shard_path, storage, int(idx))
            with (out_dir / "example_ground_truth.json").open("w", encoding="utf-8") as f:
                json.dump(truth, f, indent=2, default=to_jsonable)
    print(f"Plots and example_curve.csv written to {out_dir}")


if __name__ == "__main__":
    main()
