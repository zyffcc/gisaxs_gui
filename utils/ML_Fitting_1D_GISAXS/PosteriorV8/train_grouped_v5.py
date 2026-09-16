"""Train V5.2 from explicit checked train and tuning-validation artifact shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from typing import Sequence

import tensorflow as tf

from .grouped_training_v5 import (
    V5GroupedTrainingConfig,
    inspect_v5_grouped_training,
    train_v5_grouped_model,
)
from .training_objective_v5 import DEFAULT_LOCAL_COVERAGE_WEIGHT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Checked ZIP shards remain separate but their arrays are currently resident. "
            "The trainer fails closed above a declared 64-GiB input-array limit; a checked "
            "streaming/mmap reader is required before exceeding that paper-scale gate. "
            "Full-sidecar batches also fail closed above 8192 expanded train rows per "
            "replica, 12000 validation rows, or 4 million positive-negative pairs; "
            "yellow preflights require a Maxwell one-step peak-memory smoke receipt."
        ),
    )
    parser.add_argument(
        "--train-dataset",
        required=True,
        action="append",
        type=Path,
        help="Checked train-split artifact; repeat for multiple immutable shards.",
    )
    parser.add_argument(
        "--validation-dataset",
        required=True,
        action="append",
        type=Path,
        help="Checked tuning-validation artifact; repeat for multiple immutable shards.",
    )
    parser.add_argument(
        "--train-sidecar",
        action="append",
        type=Path,
        help=(
            "Checked train frozen-search sidecar; repeat for the labeled parent subset. "
            "Its embedded parent SHA binding selects exactly one --train-dataset."
        ),
    )
    parser.add_argument(
        "--validation-sidecar",
        action="append",
        type=Path,
        help=(
            "Checked tuning-validation frozen-search sidecar; repeat for the labeled "
            "parent subset. Its embedded parent SHA binding selects exactly one "
            "--validation-dataset."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--full-epochs", type=int, default=0)
    parser.add_argument("--recipes-per-replica", type=int, default=4)
    parser.add_argument("--validation-recipes-per-batch", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument(
        "--full-steps-per-epoch",
        type=int,
        help=(
            "Optional independent full-stage step cap. By default the complete labeled "
            "sidecar subset is used, regardless of --steps-per-epoch."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--encoder-blocks", type=int, default=6)
    parser.add_argument("--mixture-components", type=int, default=12)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--search-yield-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-ranking-weight", type=float, default=1.0)
    parser.add_argument("--local-mdn-weight", type=float, default=1.0)
    parser.add_argument(
        "--local-coverage-weight",
        type=float,
        default=DEFAULT_LOCAL_COVERAGE_WEIGHT,
    )
    parser.add_argument(
        "--operational-top-l-alignment-weight", type=float, default=1.0
    )
    parser.add_argument("--logistic-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--local-coverage-temperature", type=float, default=0.05)
    parser.add_argument("--operational-hit-rms-threshold", type=float, default=0.05)
    parser.add_argument(
        "--operational-duplicate-rms-threshold", type=float, default=0.02
    )
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--non-deterministic-ops",
        action="store_true",
        help="Allow nondeterministic TensorFlow kernels while retaining fixed seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify artifact, contracts, split evidence, and batch feasibility without writes.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small local-MDN epoch and one recipe batch; never enables full labels.",
    )
    parser.add_argument(
        "--allow-unsafe-sidecar-expansion-for-engineering",
        action="store_true",
        help=(
            "Explicitly bypass the full-sidecar expanded-batch hard gate. The audit "
            "marks the run engineering-only and ineligible as a paper training receipt."
        ),
    )
    return parser


def _config(args: argparse.Namespace) -> V5GroupedTrainingConfig:
    values = {
        "warmup_epochs": args.warmup_epochs,
        "full_epochs": args.full_epochs,
        "recipes_per_replica": args.recipes_per_replica,
        "validation_recipes_per_batch": args.validation_recipes_per_batch,
        "steps_per_epoch": args.steps_per_epoch,
        "full_steps_per_epoch": args.full_steps_per_epoch,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "width": args.width,
        "encoder_blocks": args.encoder_blocks,
        "mixture_components": args.mixture_components,
        "gradient_clip_norm": args.gradient_clip_norm,
        "mixed_precision": args.mixed_precision,
        "deterministic_ops": not args.non_deterministic_ops,
        "search_yield_weight": args.search_yield_weight,
        "pairwise_ranking_weight": args.pairwise_ranking_weight,
        "local_mdn_weight": args.local_mdn_weight,
        "local_coverage_weight": args.local_coverage_weight,
        "operational_top_l_alignment_weight": (
            args.operational_top_l_alignment_weight
        ),
        "logistic_epsilon": args.logistic_epsilon,
        "local_coverage_temperature": args.local_coverage_temperature,
        "operational_hit_rms_threshold": args.operational_hit_rms_threshold,
        "operational_duplicate_rms_threshold": (
            args.operational_duplicate_rms_threshold
        ),
        "allow_unsafe_sidecar_expansion_for_engineering": (
            args.allow_unsafe_sidecar_expansion_for_engineering
        ),
    }
    if args.smoke:
        values.update(
            {
                "warmup_epochs": 1,
                "full_epochs": 0,
                "recipes_per_replica": 1,
                "validation_recipes_per_batch": 1,
                "steps_per_epoch": 1,
                "width": min(args.width, 16),
                "encoder_blocks": 1,
                "mixed_precision": False,
            }
        )
    return V5GroupedTrainingConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if socket.gethostname().split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError(
            "grouped training CLI, including --dry-run, is forbidden on Maxwell login nodes "
            "because artifact inspection materializes resident arrays; use a Slurm "
            "worker or a future manifest-only preflight"
        )
    config = _config(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if args.dry_run:
        replicas = max(1, len(tf.config.list_logical_devices("GPU")))
        _, audit = inspect_v5_grouped_training(
            args.train_dataset,
            args.validation_dataset,
            config,
            train_sidecar_paths=args.train_sidecar,
            validation_sidecar_paths=args.validation_sidecar,
            replicas=replicas,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": config.audit_payload(),
                    "training_audit": audit.audit_payload(),
                },
                sort_keys=True,
            )
        )
        return 0
    result = train_v5_grouped_model(
        args.train_dataset,
        args.validation_dataset,
        args.output_dir,
        config,
        train_sidecar_paths=args.train_sidecar,
        validation_sidecar_paths=args.validation_sidecar,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "best_model": str(result.best_model_path),
                "last_model": str(result.last_model_path),
                "history": str(result.history_path),
                "manifest": str(result.manifest_path),
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "full_checkpoints": [
                    str(value) for value in result.full_checkpoint_paths
                ],
                "checkpoint_selection_status": result.checkpoint_selection_status,
                "paper_model_eligible": result.paper_model_eligible,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
