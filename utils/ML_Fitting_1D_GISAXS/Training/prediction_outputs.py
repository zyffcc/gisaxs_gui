"""Serialization and diagnostic plots for TOP-K prediction results."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from TrainSetBuild import schema
from Training.prediction_candidates import component_params_json, posterior_stats


def write_prediction_outputs(
    args,
    ranked,
    curve_bank,
    q_eval,
    I_eval,
    sigma_eval,
    q_outlier,
    I_outlier,
    sampling_scales,
    warm_start_loaded,
    warm_start_rejected,
    rejected_empty,
    rejected_exact_nonempty,
    rejected_forward,
    rejected_nonfinite,
    out_dir,
) -> None:
    """Write the stable JSON, CSV, arrays and plots produced by the CLI."""
    json_rows = []
    residual_bank = []
    linear_residual_bank = []
    for rank, r in enumerate(ranked, start=1):
        best = r["best"]
        refine_info = r.get("refine_info")
        curve_bank.append(best["I_fit"])
        residual_bank.append(best["log_residual"])
        linear_residual_bank.append(best["linear_residual"])
        json_rows.append(
            {
                "rank": rank,
                "rank_mode": args.rank_mode,
                "physics_rank": int(r["physics_rank"]),
                "posterior_rank": int(r["posterior_rank"]),
                "combination": r["combination"],
                "parameter_mode": int(r["parameter_mode"]),
                "parameter_mode_radius": float(args.parameter_mode_radius),
                "sampling_std": float(args.sampling_std),
                "sampling_scales": list(sampling_scales),
                "n_components": int(r["n_components"]),
                "complexity_penalty": float(args.complexity_penalty),
                "fit_equivalence_tolerance": float(args.fit_equivalence_tolerance),
                "score_equivalence_tolerance": float(args.score_equivalence_tolerance),
                "complexity_log_prior": float(r["complexity_log_prior"]),
                "score_mode": args.score_mode,
                "robust_loss": args.robust_loss,
                "robust_f_scale": args.robust_f_scale,
                "q_min": args.q_min,
                "q_max": args.q_max,
                "drop_outliers": bool(args.drop_outliers),
                "n_removed_outliers": int(len(q_outlier)),
                "include_mean_candidate": bool(args.include_mean_candidate),
                "initial_candidates_json": [str(p) for p in args.initial_candidates_json],
                "warm_start_loaded": int(warm_start_loaded),
                "warm_start_rejected": int(warm_start_rejected),
                "exact_nonempty": args.exact_nonempty,
                "refine_top_n": int(args.refine_top_n),
                "refine_best_per_k": bool(args.refine_best_per_k),
                "refine_q_stride": int(args.refine_q_stride),
                "refine_max_nfev": int(args.refine_max_nfev),
                "refine_target_logrmse": float(args.refine_target_logrmse),
                "refine_stall_patience": int(args.refine_stall_patience),
                "refine_stall_tol": float(args.refine_stall_tol),
                "refine_attempted": refine_info is not None,
                "refine_success": bool(refine_info.get("success", False)) if refine_info else False,
                "refine_accepted": bool(refine_info.get("accepted_by_score", False))
                if refine_info
                else False,
                "refine_nfev": int(refine_info.get("nfev", 0)) if refine_info else 0,
                "refine_residual_calls": int(refine_info.get("residual_calls", 0))
                if refine_info
                else 0,
                "refine_early_stop_reason": refine_info.get("early_stop_reason")
                if refine_info
                else None,
                "refine_best_log_rmse_seen": float(refine_info.get("best_log_rmse_seen", np.nan))
                if refine_info
                else None,
                "refine_message": str(refine_info.get("message", "")) if refine_info else "",
                "rejected_empty": int(rejected_empty),
                "rejected_exact_nonempty": int(rejected_exact_nonempty),
                "rejected_forward": int(rejected_forward),
                "rejected_nonfinite": int(rejected_nonfinite),
                "best_source": r["best_source"],
                "unrefined_best_source": r.get("unrefined_best_source", r["best_source"]),
                "score_weighted_probability": r["score_weighted_probability"],
                "posterior_frequency": r["posterior_frequency"],
                "chi2_weighted_probability": r["chi2_weighted_probability"],
                "fit_percent": r["fit_percent"],
                "fit_percent_by_log_rmse": r["fit_percent_by_log_rmse"],
                "fit_percent_by_chi2": r["fit_percent_by_chi2"],
                "best_score": r["best_score"],
                "best_log_rmse": r["best_log_rmse"],
                "best_chi2_weighted": r["best_chi2_weighted"],
                "best_robust_log": r["best_robust_log"],
                "best_relative_rmse": r["best_relative_rmse"],
                "best_linear_rmse": r["best_linear_rmse"],
                "unrefined_best_score": r.get("unrefined_best_score", r["best_score"]),
                "unrefined_best_log_rmse": r.get("unrefined_best_log_rmse", r["best_log_rmse"]),
                "unrefined_best_chi2_weighted": r.get(
                    "unrefined_best_chi2_weighted", r["best_chi2_weighted"]
                ),
                "components": [
                    {
                        "type": c["type_name"],
                        "weight": float(c["weight"]),
                        "params": component_params_json(c),
                    }
                    for c in best["components"]
                ],
                "global_params": {
                    name: float(best["global_phys"][i])
                    for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)
                },
                "posterior_parameter_stats": posterior_stats(r["items"]),
            }
        )

    with (out_dir / "top20_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)
    with (out_dir / "top20_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "rank_mode",
                "physics_rank",
                "posterior_rank",
                "combination",
                "parameter_mode",
                "parameter_mode_radius",
                "n_components",
                "complexity_penalty",
                "fit_equivalence_tolerance",
                "score_equivalence_tolerance",
                "complexity_log_prior",
                "score_mode",
                "robust_loss",
                "robust_f_scale",
                "q_min",
                "q_max",
                "drop_outliers",
                "n_removed_outliers",
                "include_mean_candidate",
                "initial_candidates_json",
                "warm_start_loaded",
                "warm_start_rejected",
                "exact_nonempty",
                "refine_top_n",
                "refine_best_per_k",
                "refine_q_stride",
                "refine_max_nfev",
                "refine_target_logrmse",
                "refine_stall_patience",
                "refine_stall_tol",
                "refine_attempted",
                "refine_success",
                "refine_accepted",
                "refine_nfev",
                "refine_residual_calls",
                "refine_early_stop_reason",
                "refine_best_log_rmse_seen",
                "refine_message",
                "rejected_empty",
                "rejected_exact_nonempty",
                "rejected_forward",
                "rejected_nonfinite",
                "best_source",
                "unrefined_best_source",
                "score_weighted_probability",
                "posterior_frequency",
                "chi2_weighted_probability",
                "fit_percent_by_log_rmse",
                "fit_percent_by_chi2",
                "best_score",
                "best_log_rmse",
                "best_chi2_weighted",
                "best_robust_log",
                "best_relative_rmse",
                "best_linear_rmse",
                "unrefined_best_score",
                "unrefined_best_log_rmse",
                "unrefined_best_chi2_weighted",
            ],
        )
        writer.writeheader()
        for row in json_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    np.savez_compressed(
        out_dir / "best_fit_curves.npz",
        q=q_eval,
        I_exp=I_eval,
        sigma=sigma_eval,
        I_fit=np.stack(curve_bank),
        q_removed_outliers=q_outlier,
        I_removed_outliers=I_outlier,
    )
    np.savez_compressed(
        out_dir / "residuals_top5.npz",
        q=q_eval,
        log_residual=np.stack(residual_bank[:5]),
        linear_residual=np.stack(linear_residual_bank[:5]),
    )

    plt.figure(figsize=(8, 6))
    plt.loglog(q_eval, I_eval, "k.", ms=3, label="input")
    for i, row in enumerate(json_rows[:5]):
        plt.loglog(
            q_eval,
            curve_bank[i],
            lw=1.0,
            label=f"#{row['rank']} {row['combination']} logRMSE={row['best_log_rmse']:.3g} chi2={row['best_chi2_weighted']:.3g}",
        )
    plt.xlabel("q / nm^-1")
    plt.ylabel("I")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_top5.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.loglog(q_eval, I_eval, "k.", ms=3, label="valid input")
    if len(q_outlier) > 0:
        plt.loglog(q_outlier, I_outlier, "rx", ms=5, label="removed outliers")
    for i, row in enumerate(json_rows[:5]):
        plt.loglog(
            q_eval,
            curve_bank[i],
            lw=1.0,
            label=f"#{row['rank']} {row['combination']} score={row['best_score']:.3g}",
        )
    plt.xlabel("q / nm^-1")
    plt.ylabel("I")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_top5_with_mask.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, row in enumerate(json_rows[:5]):
        plt.semilogx(q_eval, residual_bank[i], lw=1.0, label=f"#{row['rank']} {row['combination']}")
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel("q / nm^-1")
    plt.ylabel("log(I_fit) - log(I_exp)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_residuals_top5.png", dpi=180)
    plt.close()
    print(f"Wrote {len(json_rows)} candidates to {out_dir}")
