"""CLI for single-process, multi-GPU Posterior V8 proposal training.

Invoke as ``python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_proposal``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .proposal_training import ProposalTrainingConfig, train_proposal
from .training_objective import TrainingObjectiveConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--encoder-blocks", type=int, default=6)
    parser.add_argument("--mixture-components", type=int, default=12)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument("--branch-pattern-weight", type=float, default=1.0)
    parser.add_argument("--continuous-weight", type=float, default=1.0)
    parser.add_argument("--topology-recall-k", type=int, default=8)
    parser.add_argument("--logistic-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--non-deterministic-ops",
        action="store_true",
        help="Allow nondeterministic TensorFlow kernels; fixed data/initialization seeds remain.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    objective = TrainingObjectiveConfig(
        topology_weight=args.topology_weight,
        branch_pattern_weight=args.branch_pattern_weight,
        continuous_weight=args.continuous_weight,
        topology_recall_k=args.topology_recall_k,
        logistic_epsilon=args.logistic_epsilon,
    )
    config = ProposalTrainingConfig(
        epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_points=args.max_points,
        width=args.width,
        encoder_blocks=args.encoder_blocks,
        mixture_components=args.mixture_components,
        shuffle_buffer=args.shuffle_buffer,
        gradient_clip_norm=args.gradient_clip_norm,
        mixed_precision=args.mixed_precision,
        deterministic_ops=not args.non_deterministic_ops,
        checkpoint_keep=args.checkpoint_keep,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        objective=objective,
    )
    result = train_proposal(
        args.shards,
        args.output_dir,
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "model_path": str(result.model_path),
                "manifest_path": str(result.manifest_path),
                "history_path": str(result.history_path),
                "completed_epochs": result.completed_epochs,
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
