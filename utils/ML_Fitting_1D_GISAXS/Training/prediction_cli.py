"""Focused helpers for TOP-K prediction: prediction cli."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--constraints_json")
    p.add_argument(
        "--initial_candidates_json",
        action="append",
        default=[],
        help=(
            "Optional previous top-candidate JSON used as verified warm starts. "
            "May be supplied more than once; previous solutions are retained only if they still pass the current forward model."
        ),
    )
    p.add_argument("--q_min", type=float)
    p.add_argument("--q_max", type=float)
    p.add_argument("--drop_low_intensity_floor", action="store_true")
    p.add_argument("--low_intensity_floor_percentile", type=float, default=0.5)
    p.add_argument("--low_intensity_floor_factor", type=float, default=5.0)
    p.add_argument("--drop_outliers", action="store_true")
    p.add_argument("--outlier_sigma", type=float, default=6.0)
    p.add_argument(
        "--outlier_max_run",
        type=int,
        default=10,
        help="Maximum consecutive outlier points to treat as a local bad gap.",
    )
    p.add_argument("--chi2_threshold", type=float, default=2.0)
    p.add_argument("--log_rmse_threshold", type=float, default=0.25)
    p.add_argument(
        "--score_mode",
        choices=["unweighted_log", "weighted_log", "robust_log", "hybrid_log_relative"],
        default="unweighted_log",
    )
    p.add_argument(
        "--rank_mode",
        choices=["physics", "posterior"],
        default="physics",
        help=(
            "Final TOP-K ordering. 'physics' ranks by the verified/refined forward-model score; "
            "'posterior' preserves posterior-frequency and complexity-prior ordering."
        ),
    )
    p.add_argument(
        "--complexity_penalty",
        type=float,
        default=1.0,
        help="Log-prior penalty per component beyond K=1; larger values prefer simpler fits more strongly.",
    )
    p.add_argument(
        "--fit_equivalence_tolerance",
        type=float,
        default=0.01,
        help=(
            "Absolute logRMSE interval treated as physically equivalent during final physics ranking. "
            "Within one interval, fewer components rank first; 0 disables this parsimony tie-break."
        ),
    )
    p.add_argument(
        "--score_equivalence_tolerance",
        type=float,
        default=0.0,
        help=(
            "Equivalence interval in the selected --score_mode units for final physics ranking. "
            "When positive this takes precedence over --fit_equivalence_tolerance; fewer components rank first within a band."
        ),
    )
    p.add_argument(
        "--parameter_mode_radius",
        type=float,
        default=0.10,
        help="RMS radius in normalized parameter space used to split one type combination into distinct TOP-K modes.",
    )
    p.add_argument("--robust_loss", choices=["huber", "cauchy"], default="huber")
    p.add_argument("--robust_f_scale", type=float, default=0.3)
    p.add_argument(
        "--sampling_std",
        type=float,
        default=0.03,
        help="Default normalized posterior sampling std.",
    )
    p.add_argument(
        "--sampling_scales",
        default="1.0",
        help="Comma-separated multipliers cycled across posterior samples, e.g. 0.5,1,2,4.",
    )
    p.add_argument(
        "--use_predicted_logstd",
        action="store_true",
        help="Use model log-std heads instead of --sampling_std.",
    )
    p.add_argument(
        "--include_mean_candidate",
        action="store_true",
        help="Verify and rank the deterministic posterior mean candidate.",
    )
    p.add_argument(
        "--refine_top_n",
        type=int,
        default=0,
        help="Refine the N parameter modes with the smallest verified forward-model score.",
    )
    p.add_argument(
        "--refine_best_per_k",
        action="store_true",
        help="Also refine the best verified mode for every represented K, preventing one K from monopolizing refinement.",
    )
    p.add_argument(
        "--refine_q_stride",
        type=int,
        default=1,
        help="Use every Nth q point during optimization, then evaluate/rank the refined result on the full curve.",
    )
    p.add_argument(
        "--refine_max_nfev",
        type=int,
        default=80,
        help="Maximum scipy least_squares function evaluations per refined candidate.",
    )
    p.add_argument(
        "--refine_ftol", type=float, default=1e-8, help="least_squares function tolerance."
    )
    p.add_argument("--refine_xtol", type=float, default=1e-8, help="least_squares step tolerance.")
    p.add_argument(
        "--refine_gtol", type=float, default=1e-8, help="least_squares gradient tolerance."
    )
    p.add_argument(
        "--refine_progress_interval",
        type=int,
        default=10,
        help="Print refine progress every N residual evaluations; 0 disables per-candidate progress.",
    )
    p.add_argument(
        "--refine_target_logrmse",
        type=float,
        default=0.0,
        help="Stop refinement early when best logRMSE reaches this target; <=0 disables.",
    )
    p.add_argument(
        "--refine_stall_patience",
        type=int,
        default=80,
        help="Stop refinement after this many residual calls without clear logRMSE improvement; <=0 disables.",
    )
    p.add_argument(
        "--refine_stall_tol",
        type=float,
        default=1e-4,
        help="Minimum logRMSE improvement needed to reset refine stall patience.",
    )
    p.add_argument(
        "--exact_nonempty",
        type=int,
        default=None,
        help="Require exactly K non-empty components per sampled candidate (e.g., 1 for single-component fits).",
    )
    p.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="Print sampling progress every N posterior samples; 0 disables.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--allow_unsafe_lambda",
        action="store_true",
        help="Allow Keras Lambda layer deserialization (safe_mode=False) for trusted models.",
    )
    return p.parse_args()


def validate_prediction_args(args):
    """Validate numerical CLI options and return normalized sampling scales."""
    try:
        sampling_scales = tuple(
            float(item.strip()) for item in str(args.sampling_scales).split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "--sampling_scales must be a comma-separated list of positive numbers"
        ) from exc
    if not sampling_scales or any(
        not np.isfinite(scale) or scale <= 0 for scale in sampling_scales
    ):
        raise ValueError("--sampling_scales must contain at least one positive finite number")
    if args.exact_nonempty is not None and args.exact_nonempty < 1:
        raise ValueError("--exact_nonempty must be >= 1")
    if args.refine_top_n < 0:
        raise ValueError("--refine_top_n must be >= 0")
    if args.refine_max_nfev < 1:
        raise ValueError("--refine_max_nfev must be >= 1")
    if args.refine_ftol <= 0 or args.refine_xtol <= 0 or args.refine_gtol <= 0:
        raise ValueError("--refine_ftol, --refine_xtol, and --refine_gtol must be > 0")
    if args.refine_progress_interval < 0:
        raise ValueError("--refine_progress_interval must be >= 0")
    if args.refine_stall_patience < 0:
        raise ValueError("--refine_stall_patience must be >= 0")
    if args.refine_stall_tol < 0:
        raise ValueError("--refine_stall_tol must be >= 0")
    if args.refine_q_stride < 1:
        raise ValueError("--refine_q_stride must be >= 1")
    if args.fit_equivalence_tolerance < 0.0:
        raise ValueError("--fit_equivalence_tolerance must be >= 0")
    if args.score_equivalence_tolerance < 0.0:
        raise ValueError("--score_equivalence_tolerance must be >= 0")
    if args.complexity_penalty < 0:
        raise ValueError("--complexity_penalty must be >= 0")
    if args.parameter_mode_radius <= 0:
        raise ValueError("--parameter_mode_radius must be > 0")
    return sampling_scales
