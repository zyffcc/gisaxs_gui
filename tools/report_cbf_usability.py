"""Rebuild the comparison report from preserved GUI job outputs."""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/usability_20260921"


def main():
    records = {}
    for key, path in (
        ("model", OUT / "gui_native_model/VERIFIED.json"),
        ("physical", OUT / "gui_quick/single/VERIFIED.json"),
    ):
        verified = json.loads(path.read_text(encoding="utf-8"))
        folder = Path(verified["fit_output"])
        records[key] = json.loads((folder / "top20_candidates.json").read_text(encoding="utf-8"))
    fig = Figure(figsize=(12, 4.5), tight_layout=True)
    metrics = []
    for i, side in enumerate(("positive", "negative"), 1):
        rows = {
            key: next(r for r in bank if r["side"] == side and r["rank"] == 1)
            for key, bank in records.items()
        }
        a, b = rows.values()
        for field in ("native_q", "observed", "sigma"):
            np.testing.assert_allclose(a[field], b[field], rtol=0, atol=1e-12)
        q, y = np.abs(a["native_q"]), np.array(a["observed"])
        ax = fig.add_subplot(1, 2, i)
        ax.plot(q, y, ".", color="#64748b", ms=3, alpha=0.6, label="Native CBF observations")
        for key, color, title in (
            ("model", "#e58a2d", "V5 + correction"),
            ("physical", "#2563eb", "Quick physical fit"),
        ):
            row = rows[key]
            prediction = np.array(row["native_fit"])
            # This explicitly named diagnostic window is not the whole-curve score.
            peak = (q > 0.15) & (q < 2) & (y > 20)
            peak_error = float(np.sqrt(np.mean(np.log(prediction[peak] / y[peak]) ** 2)))
            metrics.append(
                dict(
                    method=key,
                    side=side,
                    measured_points=len(q),
                    logrmse=row["best_log_rmse"],
                    peak_window_logrmse=peak_error,
                    peak_window="0.15 < |q| < 2 nm^-1 and I > 20 (diagnostic only)",
                    rms_sigma=row["signed_weighted_rms"],
                    components=row["combination"],
                    globals=row["global_params"],
                    warnings=row.get("warnings", []),
                )
            )
            ax.plot(
                np.abs(row["display_q"]),
                row["display_fit"],
                color=color,
                lw=1.7,
                label=f"{title} (lnRMSE {row['best_log_rmse']:.3f})",
            )
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(y) * 2)
        ax.set_xlabel("|q| (nm⁻¹)")
        ax.set_ylabel("Intensity (input units)")
        ax.set_title(f"{side.capitalize()} q side · {len(q)} measured columns")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.savefig(OUT / "workflow_comparison.png", dpi=160)
    (OUT / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
