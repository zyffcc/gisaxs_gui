"""Posterior sampling plus physics verification for TOP-K component candidates.

The CLI workflow remains here; reusable numerical, curve, candidate, scoring and
refinement operations live in focused sibling modules and are re-exported for
backward compatibility.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import constraints, schema
from TrainSetBuild.physics_adapter import evaluate_clean, global_array_to_dict
from TrainSetBuild.sampling import D_HARD_CORE_MARGIN

from Training.prediction_candidates import (
    cluster_parameter_modes,
    combination_key,
    mean_candidate,
    sample_candidate,
)
from Training.prediction_cli import parse_args, validate_prediction_args
from Training.prediction_curve_io import (
    load_curve,
    load_model,
    make_input,
)
from Training.prediction_preprocessing import (
    apply_q_range,
    downsample_curve,
    drop_log_outliers,
    sigmoid_stable,
    softmax,
)
from Training.prediction_outputs import write_prediction_outputs
from Training.prediction_refinement import (
    refine_candidate,
)
from Training.prediction_scoring import (
    candidate_from_json_row,
    components_json,
    fit_metrics,
    score_from_metrics,
    score_weight,
)


from Training.prediction_compatibility import PUBLIC_NAMES, resolve_legacy_symbol

__all__ = PUBLIC_NAMES


def __getattr__(name):
    return resolve_legacy_symbol(name, __name__)


def main():
    args = parse_args()
    sampling_scales = validate_prediction_args(args)
    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "dataset_metadata.json"
    try:
        model_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
    except Exception:
        model_metadata = {}
    cylinder_revision = (model_metadata.get("forward_model") or {}).get(
        "random_cylinder_radial_amplitude"
    )
    if cylinder_revision != "2*cylindrical_J1(x)/x":
        print(
            "WARNING: this checkpoint predates the exact random-cylinder J1 forward-model revision. "
            "Physics verification now uses the corrected equation, but retraining is recommended before "
            "interpreting random-cylinder proposal probabilities quantitatively.",
            flush=True,
        )

    print(f"Loading input curve: {args.input_csv}", flush=True)
    q, I, sigma_arr, input_debug = load_curve(
        Path(args.input_csv),
        drop_low_intensity_floor=args.drop_low_intensity_floor,
        low_intensity_floor_percentile=args.low_intensity_floor_percentile,
        low_intensity_floor_factor=args.low_intensity_floor_factor,
    )
    input_debug["after_load_n_points"] = int(len(q))
    before_q_range_n = int(len(q))
    q, I, sigma_arr = apply_q_range(q, I, sigma_arr, q_min=args.q_min, q_max=args.q_max)
    input_debug["q_min_arg"] = None if args.q_min is None else float(args.q_min)
    input_debug["q_max_arg"] = None if args.q_max is None else float(args.q_max)
    input_debug["q_range_removed_n_points"] = int(before_q_range_n - len(q))
    input_debug["after_q_range_n_points"] = int(len(q))
    q_outlier = np.array([], dtype=np.float64)
    I_outlier = np.array([], dtype=np.float64)
    before_outlier_n = int(len(q))
    if args.drop_outliers:
        q, I, sigma_arr, q_outlier, I_outlier = drop_log_outliers(
            q,
            I,
            sigma_arr,
            outlier_sigma=args.outlier_sigma,
            window_size=21,
            max_run=args.outlier_max_run,
        )
    input_debug["drop_outliers"] = bool(args.drop_outliers)
    input_debug["outlier_removed_n_points"] = int(before_outlier_n - len(q))
    input_debug["after_outlier_filter_n_points"] = int(len(q))
    q_eval, I_eval, sigma_eval = downsample_curve(q, I, sigma_arr)
    input_debug["after_downsample_n_points"] = int(len(q_eval))
    input_debug["final_q_min"] = float(q_eval.min())
    input_debug["final_q_max"] = float(q_eval.max())
    input_debug["final_I_min"] = float(I_eval.min())
    input_debug["final_I_max"] = float(I_eval.max())
    with (out_dir / "input_curve_debug.json").open("w", encoding="utf-8") as f:
        json.dump(input_debug, f, indent=2)
    np.savez_compressed(
        out_dir / "input_curve_used.npz", q_eval=q_eval, I_eval=I_eval, sigma_eval=sigma_eval
    )
    print(
        f"Curve ready: {len(q)} valid points, using {len(q_eval)} points; "
        f"q=[{q_eval.min():.4g}, {q_eval.max():.4g}], I=[{I_eval.min():.4g}, {I_eval.max():.4g}]",
        flush=True,
    )

    cons_config = None
    if args.constraints_json:
        with Path(args.constraints_json).open("r", encoding="utf-8") as f:
            cons_config = json.load(f)
        print(f"Loaded constraints: {args.constraints_json}", flush=True)
    cons = constraints.from_json_dict(cons_config)
    d_hard_core_margin = float(
        ((cons_config or {}).get("d_constraint") or {}).get("margin", D_HARD_CORE_MARGIN)
    )
    if d_hard_core_margin <= 1.0:
        raise ValueError("d_constraint.margin must be strictly greater than 1.0")

    print(f"Loading model from: {model_dir}", flush=True)
    model = load_model(model_dir, allow_unsafe_lambda=args.allow_unsafe_lambda)
    print("Running neural network proposal pass...", flush=True)
    proposal_input = make_input(q_eval, I_eval, sigma_eval, cons)
    model_input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    proposal_input = {
        key: value for key, value in proposal_input.items() if key in model_input_names
    }
    pred = model(proposal_input, training=False)
    pred = {k: v.numpy() for k, v in pred.items()}
    exist_prob = sigmoid_stable(pred["exist_logit"][0])
    type_prob = np.stack(
        [softmax(pred["type_logits"][0, j]) for j in range(schema.MAX_SLOTS)], axis=0
    )
    print(
        "Model proposal summary: "
        f"exist_prob={np.array2string(exist_prob, precision=3)}, "
        f"top_types={[schema.TYPE_NAMES[int(np.argmax(type_prob[j]))] for j in range(schema.MAX_SLOTS)]}",
        flush=True,
    )

    groups = defaultdict(list)
    curve_bank = []
    sigma_log = np.maximum(sigma_eval / np.maximum(I_eval, 1e-30), 1e-3)

    print(
        f"Sampling {args.num_samples} posterior candidates with normalized std scales "
        f"{sampling_scales} and verifying with physics forward model...",
        flush=True,
    )
    rejected_empty = 0
    rejected_exact_nonempty = 0
    rejected_forward = 0
    rejected_nonfinite = 0
    kept = 0
    mean_candidate_kept = False
    mean_candidate_item = None
    mean_candidate_curve = None

    def verify_candidate(candidate, source):
        nonlocal \
            kept, \
            rejected_forward, \
            rejected_nonfinite, \
            mean_candidate_kept, \
            mean_candidate_item, \
            mean_candidate_curve
        components, global_phys = candidate
        try:
            I_fit = evaluate_clean(q_eval, components, global_array_to_dict(global_phys))
        except Exception:
            rejected_forward += 1
            return False
        if not np.all(np.isfinite(I_fit)):
            rejected_nonfinite += 1
            return False
        metrics = fit_metrics(
            I_fit,
            I_eval,
            sigma_log,
            robust_loss=args.robust_loss,
            robust_f_scale=args.robust_f_scale,
        )
        score = score_from_metrics(metrics, args.score_mode)
        item = {
            "source": source,
            "score": score,
            "log_rmse": metrics["log_rmse"],
            "weighted_log_chi2": metrics["weighted_log_chi2"],
            "robust_log": metrics["robust_log"],
            "relative_rmse": metrics["relative_rmse"],
            "linear_rmse": metrics["linear_rmse"],
            "log_residual": metrics["log_residual"],
            "linear_residual": metrics["linear_residual"],
            "components": components,
            "global_phys": global_phys,
            "I_fit": I_fit.astype(np.float32),
        }
        groups[combination_key(components)].append(item)
        kept += 1
        if source == "mean":
            mean_candidate_kept = True
            mean_candidate_item = item
            mean_candidate_curve = I_fit.astype(np.float32)
        return True

    warm_start_loaded = 0
    warm_start_rejected = 0
    for initial_path_text in args.initial_candidates_json:
        initial_path = Path(initial_path_text)
        with initial_path.open("r", encoding="utf-8") as f:
            saved_rows = json.load(f)
        if not isinstance(saved_rows, list):
            raise ValueError(f"Initial candidates must be a JSON list: {initial_path}")
        for saved_rank, saved_row in enumerate(saved_rows, start=1):
            try:
                candidate = candidate_from_json_row(saved_row)
                accepted = verify_candidate(
                    candidate, source=f"warm_start:{initial_path.name}:rank{saved_rank}"
                )
            except Exception as exc:
                warm_start_rejected += 1
                print(
                    f"Warm-start rank {saved_rank} rejected from {initial_path}: {exc}", flush=True
                )
                continue
            if accepted:
                warm_start_loaded += 1
            else:
                warm_start_rejected += 1
    if args.initial_candidates_json:
        print(
            f"Warm starts verified: loaded={warm_start_loaded}, rejected={warm_start_rejected}, "
            f"files={len(args.initial_candidates_json)}",
            flush=True,
        )

    if args.include_mean_candidate:
        candidate, reason = mean_candidate(
            pred,
            cons,
            exact_nonempty=args.exact_nonempty,
            return_reason=True,
            d_hard_core_margin=d_hard_core_margin,
        )
        if candidate is None:
            if reason == "exact_nonempty":
                rejected_exact_nonempty += 1
            else:
                rejected_empty += 1
            print(
                f"Mean candidate skipped: reason={reason or 'unknown'}, "
                f"exact_nonempty={args.exact_nonempty}.",
                flush=True,
            )
        else:
            ok = verify_candidate(candidate, source="mean")
            status = "kept" if ok else "rejected"
            print(f"Mean candidate {status}; kept={kept}, groups={len(groups)}", flush=True)

    if mean_candidate_item is not None:
        mean_debug = {
            "source": "mean",
            "exact_nonempty": args.exact_nonempty,
            "components": components_json(mean_candidate_item["components"]),
            "global_params": {
                name: float(mean_candidate_item["global_phys"][i])
                for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
            },
            "log_rmse": float(mean_candidate_item["log_rmse"]),
            "weighted_log_chi2": float(mean_candidate_item["weighted_log_chi2"]),
            "weighted_chi2": float(mean_candidate_item["weighted_log_chi2"]),
            "robust_log": float(mean_candidate_item["robust_log"]),
            "relative_rmse": float(mean_candidate_item["relative_rmse"]),
            "linear_rmse": float(mean_candidate_item["linear_rmse"]),
            "q": q_eval.astype(float).tolist(),
            "I_exp": I_eval.astype(float).tolist(),
            "I_fit": mean_candidate_curve.astype(float).tolist(),
        }
        with (out_dir / "mean_candidate_debug.json").open("w", encoding="utf-8") as f:
            json.dump(mean_debug, f, indent=2)
        np.savez_compressed(
            out_dir / "mean_candidate_curve.npz",
            q=q_eval,
            I_exp=I_eval,
            sigma=sigma_eval,
            I_fit=mean_candidate_curve,
        )

    for sample_idx in range(1, args.num_samples + 1):
        effective_sampling_std = (
            args.sampling_std * sampling_scales[(sample_idx - 1) % len(sampling_scales)]
        )
        candidate, reason = sample_candidate(
            pred,
            cons,
            rng,
            sampling_std=effective_sampling_std,
            use_predicted_logstd=args.use_predicted_logstd,
            exact_nonempty=args.exact_nonempty,
            return_reason=True,
            d_hard_core_margin=d_hard_core_margin,
        )
        if candidate is None:
            if reason == "exact_nonempty":
                rejected_exact_nonempty += 1
            else:
                rejected_empty += 1
            if args.progress_interval > 0 and sample_idx % args.progress_interval == 0:
                print(
                    f"Progress {sample_idx}/{args.num_samples}: kept={kept}, "
                    f"groups={len(groups)}, empty={rejected_empty}, exact_nonempty={rejected_exact_nonempty}, "
                    f"forward_fail={rejected_forward}, nonfinite={rejected_nonfinite}",
                    flush=True,
                )
            continue
        if not verify_candidate(candidate, source="sample"):
            if args.progress_interval > 0 and sample_idx % args.progress_interval == 0:
                print(
                    f"Progress {sample_idx}/{args.num_samples}: kept={kept}, "
                    f"groups={len(groups)}, empty={rejected_empty}, exact_nonempty={rejected_exact_nonempty}, "
                    f"forward_fail={rejected_forward}, nonfinite={rejected_nonfinite}",
                    flush=True,
                )
            continue
        if args.progress_interval > 0 and (
            sample_idx % args.progress_interval == 0 or sample_idx == args.num_samples
        ):
            best_so_far = min(
                (it["score"] for items in groups.values() for it in items), default=float("nan")
            )
            print(
                f"Progress {sample_idx}/{args.num_samples}: kept={kept}, groups={len(groups)}, "
                f"best_{args.score_mode}={best_so_far:.4g}, empty={rejected_empty}, "
                f"exact_nonempty={rejected_exact_nonempty}, "
                f"forward_fail={rejected_forward}, nonfinite={rejected_nonfinite}",
                flush=True,
            )

    if not groups:
        print("WARNING: all posterior samples failed. No candidates written.")
        return

    print(
        "Rejection summary: "
        f"rejected_empty={rejected_empty}, "
        f"rejected_exact_nonempty={rejected_exact_nonempty}, "
        f"rejected_forward={rejected_forward}, "
        f"rejected_nonfinite={rejected_nonfinite}",
        flush=True,
    )
    print(
        f"Aggregating {kept} valid candidates into TOP {args.top_k} parameter modes "
        f"(mean_candidate_kept={mean_candidate_kept})...",
        flush=True,
    )
    total_kept = sum(len(v) for v in groups.values())
    ranked = []
    for key, items in groups.items():
        parameter_modes = cluster_parameter_modes(items, radius=args.parameter_mode_radius)
        for mode_index, mode_items in enumerate(parameter_modes, start=1):
            scores = np.array([it["score"] for it in mode_items], dtype=np.float64)
            log_rmses = np.array([it["log_rmse"] for it in mode_items], dtype=np.float64)
            weighted_chi2s = np.array(
                [it["weighted_log_chi2"] for it in mode_items], dtype=np.float64
            )
            n_components = len(mode_items[0]["components"])
            score_weights = np.array(
                [score_weight(s, n_components, args.complexity_penalty) for s in scores],
                dtype=np.float64,
            )
            complexity_prior = float(np.exp(-args.complexity_penalty * max(n_components - 1, 0)))
            chi2_weights = np.exp(-0.5 * np.clip(weighted_chi2s, 0, 1e6)) * complexity_prior
            # The representative passed to physical refinement must be the
            # curve with the smallest directly interpretable forward-model
            # logRMSE.  Robust/posterior scores still contribute probability,
            # but selecting the representative by a different metric can hide
            # a much better starting point inside the same parameter mode.
            best = mode_items[int(np.argmin(log_rmses))]
            ranked.append(
                {
                    "combination": key,
                    "parameter_mode": mode_index,
                    "n_components": n_components,
                    "complexity_log_prior": -args.complexity_penalty * max(n_components - 1, 0),
                    "posterior_frequency": len(mode_items) / total_kept,
                    "score_weighted_probability": float(score_weights.sum()),
                    "chi2_weighted_probability": float(chi2_weights.sum()),
                    "fit_percent_by_log_rmse": float(np.mean(log_rmses < args.log_rmse_threshold)),
                    "fit_percent_by_chi2": float(np.mean(weighted_chi2s < args.chi2_threshold)),
                    "fit_percent": float(np.mean(log_rmses < args.log_rmse_threshold)),
                    "best_score": float(best["score"]),
                    "best_log_rmse": float(best["log_rmse"]),
                    "best_chi2_weighted": float(best["weighted_log_chi2"]),
                    "best_robust_log": float(best["robust_log"]),
                    "best_relative_rmse": float(best["relative_rmse"]),
                    "best_linear_rmse": float(best["linear_rmse"]),
                    "best_source": str(best.get("source", "sample")),
                    "best": best,
                    "items": mode_items,
                    "count": len(mode_items),
                }
            )
    score_norm = sum(r["score_weighted_probability"] for r in ranked) or 1.0
    chi2_norm = sum(r["chi2_weighted_probability"] for r in ranked) or 1.0
    for r in ranked:
        r["score_weighted_probability"] /= score_norm
        r["chi2_weighted_probability"] /= chi2_norm
    posterior_order = sorted(
        ranked, key=lambda r: (-r["score_weighted_probability"], r["best_score"])
    )
    for posterior_rank, row in enumerate(posterior_order, start=1):
        row["posterior_rank"] = posterior_rank

    # Refinement selection is deliberately based on verified physics error,
    # not posterior frequency or the parsimony prior.  Those are useful model
    # preferences, but must not prevent a better-fitting basin from being
    # optimized.
    physics_order = sorted(
        ranked, key=lambda r: (r["best_score"], r["best_log_rmse"], r["n_components"])
    )
    refine_targets = list(physics_order[: min(int(args.refine_top_n), len(physics_order))])
    if args.refine_best_per_k:
        for k in sorted({int(row["n_components"]) for row in physics_order}):
            best_for_k = next(row for row in physics_order if int(row["n_components"]) == k)
            if all(best_for_k is not existing for existing in refine_targets):
                refine_targets.append(best_for_k)

    if refine_targets:
        n_refine = len(refine_targets)
        print(
            f"Refining {n_refine} physics-selected modes "
            f"(top_n={args.refine_top_n}, best_per_k={args.refine_best_per_k}, "
            f"q_stride={args.refine_q_stride})...",
            flush=True,
        )
        print(
            "Note: refine eval=... counts residual calls, not scipy nfev. "
            "With numerical Jacobian, residual_calls can be roughly nfev * (n_variables + 1).",
            flush=True,
        )
        for idx, r in enumerate(refine_targets, start=1):
            unrefined = r["best"]
            n_components = len(unrefined["components"])
            type_names = [c["type_name"] for c in unrefined["components"]]
            n_refine_vars = sum(
                int(np.sum(schema.type_param_mask(int(c["type_id"]))))
                for c in unrefined["components"]
            )
            n_refine_vars += schema.G_MAX
            if n_components > 1:
                n_refine_vars += n_components
            print(
                f"Refine #{idx}/{n_refine} start: combination={r['combination']}, "
                f"types={type_names}, components={n_components}, variables={n_refine_vars}, "
                f"initial_logRMSE={unrefined['log_rmse']:.5g}, initial_score={unrefined['score']:.5g}",
                flush=True,
            )
            refined, refine_info = refine_candidate(
                unrefined,
                q_eval,
                I_eval,
                sigma_log,
                score_mode=args.score_mode,
                robust_loss=args.robust_loss,
                robust_f_scale=args.robust_f_scale,
                max_nfev=args.refine_max_nfev,
                ftol=args.refine_ftol,
                xtol=args.refine_xtol,
                gtol=args.refine_gtol,
                progress_interval=args.refine_progress_interval,
                progress_label=f" #{idx}/{n_refine}",
                refine_target_logrmse=args.refine_target_logrmse,
                refine_stall_patience=args.refine_stall_patience,
                refine_stall_tol=args.refine_stall_tol,
                cons=cons,
                optimization_q_stride=args.refine_q_stride,
                d_hard_core_margin=d_hard_core_margin,
            )
            r["refine_info"] = refine_info
            r["unrefined_best_score"] = float(unrefined["score"])
            r["unrefined_best_log_rmse"] = float(unrefined["log_rmse"])
            r["unrefined_best_chi2_weighted"] = float(unrefined["weighted_log_chi2"])
            r["unrefined_best_source"] = str(unrefined.get("source", "sample"))
            refine_accepted = float(refined["score"]) <= float(unrefined["score"])
            refine_info["accepted_by_score"] = bool(refine_accepted)
            selected = refined if refine_accepted else unrefined
            r["best"] = selected
            r["best_score"] = float(selected["score"])
            r["best_log_rmse"] = float(selected["log_rmse"])
            r["best_chi2_weighted"] = float(selected["weighted_log_chi2"])
            r["best_robust_log"] = float(selected["robust_log"])
            r["best_relative_rmse"] = float(selected["relative_rmse"])
            r["best_linear_rmse"] = float(selected["linear_rmse"])
            r["best_source"] = str(selected.get("source", "sample"))
            print(
                f"Refine #{idx} {r['combination']}: "
                f"logRMSE {refine_info.get('initial_log_rmse', np.nan):.4g} -> "
                f"{refine_info.get('final_log_rmse', np.nan):.4g}, "
                f"success={refine_info.get('success', False)}, "
                f"accepted_by_score={refine_accepted}, "
                f"nfev={refine_info.get('nfev', 0)}, "
                f"residual_calls={refine_info.get('residual_calls', 0)}, "
                f"early_stop={refine_info.get('early_stop_reason')}",
                flush=True,
            )

    # Re-rank only after every selected local optimization has been evaluated
    # on the full q grid.  Keep posterior/prior rank as a separate diagnostic.
    use_selected_score = (
        args.score_mode != "unweighted_log" or float(args.score_equivalence_tolerance) > 0.0
    )
    if use_selected_score:
        best_physics_value = min(float(r["best_score"]) for r in ranked)
        equivalence_tolerance = float(args.score_equivalence_tolerance)
    else:
        best_physics_value = min(float(r["best_log_rmse"]) for r in ranked)
        equivalence_tolerance = float(args.fit_equivalence_tolerance)

    def final_physics_key(row):
        physics_value = float(row["best_score"] if use_selected_score else row["best_log_rmse"])
        if equivalence_tolerance > 0.0:
            fit_band = int(
                np.floor(
                    max(physics_value - best_physics_value, 0.0) / equivalence_tolerance + 1e-12
                )
            )
            return fit_band, int(row["n_components"]), physics_value, float(row["best_log_rmse"])
        return physics_value, int(row["n_components"]), float(row["best_log_rmse"])

    physics_order = sorted(ranked, key=final_physics_key)
    for physics_rank, row in enumerate(physics_order, start=1):
        row["physics_rank"] = physics_rank
    if args.rank_mode == "physics":
        ranked = physics_order[: args.top_k]
    else:
        ranked = posterior_order[: args.top_k]

    write_prediction_outputs(
        args=args,
        ranked=ranked,
        curve_bank=curve_bank,
        q_eval=q_eval,
        I_eval=I_eval,
        sigma_eval=sigma_eval,
        q_outlier=q_outlier,
        I_outlier=I_outlier,
        sampling_scales=sampling_scales,
        warm_start_loaded=warm_start_loaded,
        warm_start_rejected=warm_start_rejected,
        rejected_empty=rejected_empty,
        rejected_exact_nonempty=rejected_exact_nonempty,
        rejected_forward=rejected_forward,
        rejected_nonfinite=rejected_nonfinite,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
