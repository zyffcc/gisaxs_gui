"""Build synthetic 1D GISAXS training shards as TFRecord or NPZ."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import sampling, schema
from TrainSetBuild.tfrecord_io import serialize_sample


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one integer.")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one float.")
    return values


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_points", type=int, default=schema.MAX_POINTS)
    p.add_argument("--poisson_scale_min", type=float, default=200.0)
    p.add_argument("--poisson_scale_max", type=float, default=3000.0)
    p.add_argument("--rel_noise_min", type=float, default=0.001)
    p.add_argument("--rel_noise_max", type=float, default=0.02)
    p.add_argument("--drop_noisy_floor", type=float, default=1e-20, help="Drop noisy points at or below this intensity floor; 0 disables.")
    p.add_argument("--q_conditioned_sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--visible_fraction", type=float, default=0.70)
    p.add_argument("--edge_fraction", type=float, default=0.20)
    p.add_argument("--out_of_window_fraction", type=float, default=0.10)
    p.add_argument("--gap_drop_prob", type=float, default=1.0, help="Probability of applying short intensity-drop gap augmentation per curve.")
    p.add_argument("--gap_drop_max_fraction", type=float, default=0.05, help="Maximum fraction of points allowed to be intensity-dropped.")
    p.add_argument("--format", choices=["tfrecord", "npz"], default="tfrecord")
    p.add_argument("--k_values", type=parse_int_list, default=[1, 2, 3, 4], help="Allowed active component counts, e.g. 1 or 1,2 or 3,4.")
    p.add_argument("--k_probs", type=parse_float_list, default=None, help="Optional probabilities matching --k_values, e.g. 0.5,0.5.")
    p.add_argument("--task_id", type=int, default=None, help="Parallel shard task id, 0-based. Defaults to SLURM_ARRAY_TASK_ID if set.")
    p.add_argument("--num_tasks", type=int, default=1, help="Number of parallel shard tasks.")
    p.add_argument("--shard_log_interval", type=int, default=5, help="Print one progress line every N finished shards per task.")
    p.add_argument("--sample_log_interval", type=int, default=0, help="Optional within-shard sample progress interval; 0 disables it.")
    p.add_argument("--n_jobs", type=int, default=1, help="Reserved for future multiprocessing; current version is single-process.")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def split_counts(num_samples: int):
    n_train = int(round(num_samples * 0.90))
    n_val = int(round(num_samples * 0.05))
    n_test = num_samples - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test}


def write_metadata(output_dir: Path, args, counts):
    meta = schema.metadata_dict()
    meta.update(
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "num_samples": int(args.num_samples),
            "split_counts": counts,
            "samples_per_shard": int(args.samples_per_shard),
            "storage_format": args.format,
            "k_values": list(map(int, args.k_values)),
            "k_probs": None if args.k_probs is None else list(map(float, args.k_probs)),
            "noise": {
                "poisson_scale_min": float(args.poisson_scale_min),
                "poisson_scale_max": float(args.poisson_scale_max),
                "rel_noise_min": float(args.rel_noise_min),
                "rel_noise_max": float(args.rel_noise_max),
                "drop_noisy_floor": float(args.drop_noisy_floor),
            },
            "q_conditioned_sampling": {
                "enabled": bool(args.q_conditioned_sampling),
                "mode_ids": sampling.SAMPLING_MODE_NAMES,
                "visible_fraction": float(args.visible_fraction),
                "edge_fraction": float(args.edge_fraction),
                "out_of_window_fraction": float(args.out_of_window_fraction),
                "coefficients": {
                    "sphere_R": sampling.SPHERE_R_COEFF,
                    "vertical_cylinder_R": sampling.VERTICAL_CYLINDER_R_COEFF,
                    "cylinder_R": sampling.CYLINDER_R_COEFF,
                    "D": sampling.STRUCTURE_PERIOD_COEFF,
                    "h": sampling.STRUCTURE_PERIOD_COEFF,
                },
                "observable_margin": 2.0,
                "edge_margin": 5.0,
                "cylinder_R_margin_multiplier": sampling.CYLINDER_R_MARGIN_MULTIPLIER,
            },
            "gap_drop": {
                "probability": float(args.gap_drop_prob),
                "max_fraction": float(args.gap_drop_max_fraction),
                "regions_per_curve": [1, 3],
                "points_per_region": [1, 10],
                "intensity_factor_range": [0.01, 0.70],
            },
            "parallel_num_tasks": int(args.num_tasks),
            "quick_test": bool(args.quick_test),
            "normalization": {
                "input": "[logq_norm, logI_norm, logsigma_norm]",
                "global_features": ["q_min_norm", "q_max_norm", "N_points/max_points", "I_offset", "I_scale"],
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_split(split: str, count: int, output_dir: Path, args, seed_offset: int):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    n_shards = math.ceil(count / args.samples_per_shard)
    assigned_shards = [idx for idx in range(n_shards) if idx % args.num_tasks == args.task_id]
    made_samples = 0
    made_shards = 0
    print(
        f"[{split}] task {args.task_id}/{args.num_tasks}: "
        f"assigned {len(assigned_shards)} of {n_shards} shards",
        flush=True,
    )
    for local_idx, shard_idx in enumerate(assigned_shards, start=1):
        start = shard_idx * args.samples_per_shard
        shard_n = min(args.samples_per_shard, count - start)
        suffix = "tfrecord" if args.format == "tfrecord" else "npz"
        shard_path = split_dir / f"shard_{shard_idx:06d}.{suffix}"
        if shard_path.exists() and not args.overwrite:
            made_samples += shard_n
            made_shards += 1
            if made_shards % args.shard_log_interval == 0 or made_shards == len(assigned_shards):
                print(
                    f"[{split}] task {args.task_id}/{args.num_tasks}: "
                    f"{made_shards}/{len(assigned_shards)} assigned shards done/skipped",
                    flush=True,
                )
            continue
        rng = np.random.default_rng(args.seed + seed_offset + shard_idx)
        tmp_path = shard_path.with_name(f".{shard_path.name}.tmp-{os.getpid()}")
        if args.format == "tfrecord":
            with tf.io.TFRecordWriter(str(tmp_path)) as writer:
                for i in range(shard_n):
                    sample = sampling.generate_sample(
                        rng,
                        max_points=args.max_points,
                        poisson_scale_min=args.poisson_scale_min,
                        poisson_scale_max=args.poisson_scale_max,
                        rel_noise_min=args.rel_noise_min,
                        rel_noise_max=args.rel_noise_max,
                        drop_noisy_floor=args.drop_noisy_floor,
                        q_conditioned_sampling=args.q_conditioned_sampling,
                        visible_fraction=args.visible_fraction,
                        edge_fraction=args.edge_fraction,
                        out_of_window_fraction=args.out_of_window_fraction,
                        k_values=args.k_values,
                        k_probs=args.k_probs,
                        gap_drop_prob=args.gap_drop_prob,
                        gap_drop_max_fraction=args.gap_drop_max_fraction,
                    )
                    writer.write(serialize_sample(sample))
                    if args.sample_log_interval > 0 and ((i + 1) % args.sample_log_interval == 0 or (i + 1) == shard_n):
                        print(f"[{split}] shard {shard_idx:06d}: {i + 1}/{shard_n}", flush=True)
            os.replace(tmp_path, shard_path)
        else:
            samples = []
            for i in range(shard_n):
                samples.append(
                    sampling.generate_sample(
                        rng,
                        max_points=args.max_points,
                        poisson_scale_min=args.poisson_scale_min,
                        poisson_scale_max=args.poisson_scale_max,
                        rel_noise_min=args.rel_noise_min,
                        rel_noise_max=args.rel_noise_max,
                        drop_noisy_floor=args.drop_noisy_floor,
                        q_conditioned_sampling=args.q_conditioned_sampling,
                        visible_fraction=args.visible_fraction,
                        edge_fraction=args.edge_fraction,
                        out_of_window_fraction=args.out_of_window_fraction,
                        k_values=args.k_values,
                        k_probs=args.k_probs,
                        gap_drop_prob=args.gap_drop_prob,
                        gap_drop_max_fraction=args.gap_drop_max_fraction,
                    )
                )
                if args.sample_log_interval > 0 and ((i + 1) % args.sample_log_interval == 0 or (i + 1) == shard_n):
                    print(f"[{split}] shard {shard_idx:06d}: {i + 1}/{shard_n}", flush=True)
            arrays = sampling.stack_samples(samples)
            with tmp_path.open("wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp_path, shard_path)
        made_samples += shard_n
        made_shards += 1
        if made_shards % args.shard_log_interval == 0 or made_shards == len(assigned_shards):
            print(
                f"[{split}] task {args.task_id}/{args.num_tasks}: "
                f"{made_shards}/{len(assigned_shards)} assigned shards done, "
                f"latest={shard_path.name}, samples_this_task={made_samples}",
                flush=True,
            )


def main():
    args = parse_args()
    if args.task_id is None:
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        args.task_id = int(slurm_task_id) if slurm_task_id is not None else 0
    if args.num_tasks < 1:
        raise ValueError("--num_tasks must be >= 1")
    if any(k < 1 or k > schema.MAX_SLOTS for k in args.k_values):
        raise ValueError(f"--k_values must be between 1 and MAX_SLOTS={schema.MAX_SLOTS}; got {args.k_values}")
    if args.k_probs is not None and len(args.k_probs) != len(args.k_values):
        raise ValueError(f"--k_probs length must match --k_values length; got {args.k_probs} vs {args.k_values}")
    if args.k_probs is not None and any(p < 0 for p in args.k_probs):
        raise ValueError("--k_probs must be non-negative.")
    if args.k_probs is not None and sum(args.k_probs) <= 0:
        raise ValueError("--k_probs sum must be positive.")
    if args.poisson_scale_min <= 0 or args.poisson_scale_max <= 0:
        raise ValueError("--poisson_scale_min/max must be positive.")
    if args.poisson_scale_min > args.poisson_scale_max:
        raise ValueError("--poisson_scale_min must be <= --poisson_scale_max.")
    if args.rel_noise_min < 0 or args.rel_noise_max < 0:
        raise ValueError("--rel_noise_min/max must be non-negative.")
    if args.rel_noise_min > args.rel_noise_max:
        raise ValueError("--rel_noise_min must be <= --rel_noise_max.")
    if args.drop_noisy_floor < 0:
        raise ValueError("--drop_noisy_floor must be non-negative.")
    mode_fracs = [args.visible_fraction, args.edge_fraction, args.out_of_window_fraction]
    if any(v < 0 for v in mode_fracs):
        raise ValueError("--visible_fraction, --edge_fraction, and --out_of_window_fraction must be non-negative.")
    if args.q_conditioned_sampling and sum(mode_fracs) <= 0:
        raise ValueError("At least one q-conditioned sampling fraction must be positive.")
    if not 0.0 <= args.gap_drop_prob <= 1.0:
        raise ValueError("--gap_drop_prob must be between 0 and 1.")
    if not 0.0 <= args.gap_drop_max_fraction <= 1.0:
        raise ValueError("--gap_drop_max_fraction must be between 0 and 1.")
    if not 0 <= args.task_id < args.num_tasks:
        raise ValueError(f"--task_id must satisfy 0 <= task_id < num_tasks; got {args.task_id}/{args.num_tasks}")
    if args.shard_log_interval < 1:
        raise ValueError("--shard_log_interval must be >= 1")
    if args.quick_test:
        args.num_samples = min(args.num_samples, 256)
        args.samples_per_shard = min(args.samples_per_shard, 128)
    if args.max_points > schema.MAX_POINTS:
        raise ValueError(f"max_points={args.max_points} exceeds schema.MAX_POINTS={schema.MAX_POINTS}")

    output_dir = Path(args.output_dir)
    counts = split_counts(args.num_samples)
    write_metadata(output_dir, args, counts)
    for sub in ("train", "val", "test"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    build_split("train", counts["train"], output_dir, args, 0)
    build_split("val", counts["val"], output_dir, args, 1000000)
    build_split("test", counts["test"], output_dir, args, 2000000)
    print(f"Done task {args.task_id}/{args.num_tasks}. Dataset written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
