"""Render the actual GUI release result beside the frozen numerical reference."""

from pathlib import Path
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/stable_blue_20260922"


def main():
    evidence = json.loads(
        (ROOT / "validation/stable_release_20260922/ui/VERIFIED.json").read_text()
    )
    # Resolve the recorded run by basename, so a moved whole project can replay it.
    run_name = evidence["fit_output"].replace("\\", "/").rsplit("/", 1)[-1]
    result = ROOT / "AI_Fitting_Output" / run_name / "top20_candidates.json"
    if not result.exists():
        result = OUT / "gui_candidates.json"
    rows = json.loads(result.read_text())
    previous = np.load(ROOT / "validation/cause_audit_20260921/comparison_native.npz")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = []
    for col, side in enumerate(("positive", "negative")):
        row = next(r for r in rows if r["side"] == side)
        q = np.abs(np.asarray(row["native_q"]))
        y, fitted = np.asarray(row["observed"]), np.asarray(row["native_fit"])
        assert np.allclose(q, np.abs(previous[f"{side}_q"]), atol=1e-12, rtol=0)
        assert np.array_equal(y, previous[f"{side}_observed"])
        reference = previous[f"{side}_converged_integral"]
        old = previous[f"{side}_nn_rank1"]
        peak = (q >= 1) & (q <= 2)
        metric = dict(
            side=side,
            source=row["best_source"],
            observed_logrmse=row["best_log_rmse"],
            reference_logrmse=float(np.sqrt(np.mean(np.log(reference / y) ** 2))),
            distance_to_reference=float(np.sqrt(np.mean(np.log(fitted / reference) ** 2))),
            peak_height_bias=float(fitted[peak].max() / reference[peak].max() - 1),
            peak_position_difference=float(q[peak][fitted[peak].argmax()] - q[peak][reference[peak].argmax()]),
        )
        metrics.append(metric)
        for r in range(2):
            ax = axes[r, col]
            use = np.ones(len(q), dtype=bool) if r == 0 else (q >= 0.7) & (q <= 2.4)
            ax.scatter(q[use], y[use], s=7, color="#858b93", alpha=0.5, label="Measured")
            ax.plot(q[use], old[use], color="#d79537", lw=1.4, label="Previous general NN")
            ax.plot(q[use], reference[use], color="#2468b5", lw=2.1, label="Numerical reference")
            ax.plot(q[use], fitted[use], color="#18835a", lw=1.9, ls="--", label="Stable: NN + amplitude")
            ax.set_xlabel(r"$|q|$ (nm$^{-1}$)")
            ax.set_ylabel("Intensity (counts / pixel)")
            ax.grid(alpha=0.15)
            if r == 0:
                ax.set_yscale("log")
                ax.set_title(f"{side.capitalize()} | observed lnRMSE {row['best_log_rmse']:.4f}")
                ax.legend(fontsize=8)
            else:
                ax.set_title(f"Peak zoom | height vs reference: {metric['peak_height_bias']:+.1%}")
        assert row["nonlinear_refinement"] is False
    fig.suptitle(
        "CBF 00033: actual Stable auto GUI result, no nonlinear refinement\n"
        "Local single-RC branch; blue reference is a numerical fit, not ground truth",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "stable_gui_comparison.png", dpi=160)
    plt.close(fig)
    (OUT / "gui_candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "gui_comparison.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
