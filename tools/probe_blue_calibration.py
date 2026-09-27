"""Predeclared one-pass amplitude calibration, with no shape optimization.

Selection uses fresh synthetic validation only. Experimental reference curves
are read after selecting one global variance tolerance. They are not truth.
"""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import nnls
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.blue_curve_distillation import AUDIT, OUT, build_model, decode, features, forward

DEST = ROOT / "validation/stable_blue_20260922"
SEED = 2026092201
TOLERANCES = (0.0, 0.03, 0.10)


def log_error(pred, target):
    valid = np.asarray(target) > 0
    return float(np.sqrt(np.mean(np.log(pred[valid] / np.asarray(target)[valid]) ** 2)))


def basis_from_prediction(q, u):
    p = decode(u)
    curve = forward(q, u)
    resolution = 1 / (1 + (q / p["sigma_res"]) ** p["nu_res"])
    particle = (curve - p["B"] - p["C"] * resolution) / p["A"]
    return curve, np.column_stack([particle, np.ones_like(q), resolution]), p


def calibrate(matrix, initial, y, count, tolerance):
    """One NNLS solve; variance is fixed from the original NN prediction."""
    tick = time.perf_counter()
    use = (count > 0) & np.isfinite(y)
    variance = initial[use] / count[use] + (tolerance * initial[use]) ** 2
    inv_sigma = 1 / np.sqrt(np.maximum(variance, 1e-12))
    weighted = matrix[use] * inv_sigma[:, None]
    norm = np.linalg.norm(weighted, axis=0)
    scaled = weighted / norm
    amplitudes, _ = nnls(scaled, y[use] * inv_sigma, maxiter=100)
    amplitudes = amplitudes / norm
    calibrated = matrix @ amplitudes
    if not np.isfinite(calibrated).all() or np.any(calibrated <= 0):
        raise ValueError("Nonfinite/nonpositive calibrated curve")
    return calibrated, dict(
        amplitudes=dict(zip(("A", "B", "C"), amplitudes.tolist())),
        seconds=time.perf_counter() - tick,
        weighted_design_condition=float(np.linalg.cond(scaled)),
        amplitudes_outside_training_domain=bool(
            amplitudes[0] < 500
            or amplitudes[0] > 7000
            or amplitudes[1] < 0.15
            or amplitudes[1] > 4
            or amplitudes[2] < 50000
            or amplitudes[2] > 900000
        ),
    )


def summary(records, name):
    error = np.array([r[name] for r in records])
    return dict(
        mean=float(error.mean()),
        median=float(np.median(error)),
        p90=float(np.quantile(error, 0.9)),
        max=float(error.max()),
    )


def run():
    DEST.mkdir(exist_ok=True, parents=True)
    protocol = dict(
        seed=SEED,
        synthetic_validation_rows=64,
        tolerances=list(TOLERANCES),
        selection="minimum mean clean lnRMSE on 64 NEW synthetic validation samples; ties prefer lower tolerance",
        variance="initial_NN_mean / count + (relative_tolerance * initial_NN_mean)^2",
        solve="One nonnegative linear least squares solve of A*particle + B + C*resolution; all NN shapes fixed",
        mask="Use only finite measured samples with positive exposure; no imputation in solve",
        evaluation="Synthetic clean error uses ALL original q, including randomly missing columns",
        synthetic_note="New validation, not independent confirmation of the chosen tolerance; no model training or old TEST reuse",
        calibration="Amplitudes may leave training bounds; explicitly reported; not nonlinear refinement or pure NN",
    )
    (DEST / "calibration_probe.protocol.json").write_text(json.dumps(protocol, indent=2))
    sc = np.load(OUT / "feature_scaling.npz")
    net = build_model()
    net.load_weights(str(OUT / "curve_best.weights.h5"))
    templates = json.loads((AUDIT / "inputs.json").read_text())
    units = qmc.LatinHypercube(11, seed=SEED).random(64)
    rng = np.random.default_rng(SEED + 1)
    synthetic = []
    for i, u in enumerate(units):
        d = templates[("positive", "negative")[i % 2]]
        q = np.array(d["q"])
        exposure = np.array(d["count"]) * np.exp(rng.uniform(np.log(0.5), np.log(2)))
        clean = forward(q, u)
        observed = rng.poisson(clean * exposure) / exposure
        exposure *= rng.random(len(q)) > 0.025
        feat = (features(q, observed, exposure) - sc["mean"]) / sc["scale"]
        predicted_u = net(feat[None], training=False).numpy()[0]
        initial, matrix, _ = basis_from_prediction(q, predicted_u)
        rec = dict(index=i, clean_raw=log_error(initial, clean), options={})
        for tolerance in TOLERANCES:
            pred, details = calibrate(matrix, initial, observed, exposure, tolerance)
            rec["options"][str(tolerance)] = dict(
                clean=log_error(pred, clean),
                **details,
                shape_parameters_unchanged=True,
            )
        synthetic.append(rec)
        if (i + 1) % 16 == 0:
            print(json.dumps(dict(synthetic_done=i + 1)), flush=True)
    synthetic_summary = dict(raw=summary(synthetic, "clean_raw"), options={})
    for tolerance in TOLERANCES:
        key = str(tolerance)
        details = [r["options"][key] for r in synthetic]
        synthetic_summary["options"][key] = dict(
            **summary(details, "clean"),
            median_seconds=float(np.median([r["seconds"] for r in details])),
            fraction_improved=float(
                np.mean([r["options"][key]["clean"] < r["clean_raw"] for r in synthetic])
            ),
            amplitudes_outside_training_domain=sum(
                r["amplitudes_outside_training_domain"] for r in details
            ),
        )
    chosen = min(TOLERANCES, key=lambda t: synthetic_summary["options"][str(t)]["mean"])
    selection = dict(chosen_tolerance=chosen, synthetic_summary=synthetic_summary)
    # Persist the actual decision BEFORE experimental comparisons are loaded.
    (DEST / "calibration_probe.selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    references = json.loads((OUT / "real_references.json").read_text())
    previous = json.loads((OUT / "evaluation.json").read_text())
    real = []
    for d in references:
        q, y, count, reference = [np.array(d[k]) for k in ("q", "y", "count", "reference")]
        feat = (features(q, y, count) - sc["mean"]) / sc["scale"]
        uu = net(feat[None], training=False).numpy()[0]
        initial, matrix, p = basis_from_prediction(q, uu)
        prior = next(
            r
            for r in previous["real"]
            if r["frame"] == d["frame"]
            and r["side"] == d["side"]
            and r["model"] == "curve_supervised"
        )
        np.testing.assert_allclose(initial, prior["forward_curve"], rtol=1e-5)
        peak = (q >= 1) & (q <= 2)

        def score(pred):
            return dict(
                observed=log_error(pred, y),
                reference=log_error(pred, reference),
                peak_height_relative_error=float(pred[peak].max() / reference[peak].max() - 1),
                peak_q_error=float(
                    abs(q[peak][np.argmax(pred[peak])] - q[peak][np.argmax(reference[peak])])
                ),
                curve=pred.tolist(),
            )

        rec = dict(
            frame=d["frame"],
            side=d["side"],
            q=q.tolist(),
            observed=y.tolist(),
            reference=reference.tolist(),
            neural_parameters=p,
            raw=score(initial),
            options={},
        )
        for tolerance in TOLERANCES:
            pred, details = calibrate(matrix, initial, y, count, tolerance)
            rec["options"][str(tolerance)] = dict(**score(pred), **details)
        real.append(rec)
        print(
            json.dumps(
                dict(
                    frame=rec["frame"],
                    side=rec["side"],
                    raw=rec["raw"]["reference"],
                    selected=rec["options"][str(chosen)]["reference"],
                )
            ),
            flush=True,
        )
    payload = dict(protocol=protocol, **selection, failures=0, synthetic=synthetic, real=real)
    (DEST / "calibration_probe.json").write_text(json.dumps(payload, indent=2))
    plot(real, chosen)
    report(payload)


def plot(real, chosen):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, r in zip(axes.flat, real):
        q = np.array(r["q"])
        ax.scatter(q, r["observed"], s=5, alpha=0.4, color="gray", label="Observed")
        ax.plot(q, r["reference"], color="#277DA1", lw=2, label="Numerical reference")
        ax.plot(q, r["raw"]["curve"], color="#AA3377", lw=1.3, label="Neural only")
        ax.plot(
            q,
            r["options"][str(chosen)]["curve"],
            color="#228833",
            lw=1.6,
            label="Neural + one amplitude solve",
        )
        ax.set_yscale("log")
        ax.set_xlim(0, 4.3)
        ax.set_title(f"{r['frame']} {r['side']}; fixed tolerance {chosen:.0%}")
        ax.set_xlabel("|q| (nm$^{-1}$)")
        ax.set_ylabel("Mean counts / pixel")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DEST / "calibration_probe.png", dpi=160)
    plt.close(fig)


def report(payload):
    chosen = payload["chosen_tolerance"]
    ss = payload["synthetic_summary"]
    lines = [
        "# One-pass amplitude calibration probe",
        "",
        "No training, gradients or nonlinear refinement. All 8 neural shape coordinates stay fixed; only A/B/C change.",
        "",
        f"64 new local-domain synthetic validation curves, Latin hypercube seed {SEED}; independent Poisson/missingness augmentation. Selected tolerance = {chosen:.0%} by mean clean lnRMSE before loading experimental comparisons.",
        "",
        "These 64 samples select the tolerance; results are validation, not independent confirmation. Experimental blue references are fitted estimates, not known truth. Amplitude coefficients are nonnegative but are not clipped to the training label range.",
        "",
        "| Method | Clean mean | Median | P90 | Improved fraction | Solve ms |",
        "|---|---:|---:|---:|---:|---:|",
        f"| NN only | {ss['raw']['mean']:.5f} | {ss['raw']['median']:.5f} | {ss['raw']['p90']:.5f} | — | — |",
    ]
    for key, d in ss["options"].items():
        lines.append(
            f"| tolerance {float(key):.0%} | {d['mean']:.5f} | {d['median']:.5f} | {d['p90']:.5f} | {d['fraction_improved']:.1%} | {1000 * d['median_seconds']:.3f} |"
        )
    lines += [
        "",
        "| Frame / side | Method | Observed lnRMSE | Reference lnRMSE | Peak height error | Peak q error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in payload["real"]:
        for label, d in [("NN only", r["raw"])] + [
            (f"tol {float(k):.0%}", v) for k, v in r["options"].items()
        ]:
            lines.append(
                f"| {r['frame']} {r['side']} | {label} | {d['observed']:.5f} | {d['reference']:.5f} | {d['peak_height_relative_error']:.1%} | {d['peak_q_error']:.5f} |"
            )
    lines += [
        "",
        "![All six sides](calibration_probe.png)",
        "",
        "Failures: 0. The numerical linear solve uses fixed weights from the initial prediction, never inverse observed noisy intensity. Input masking is retained; no observations are interpolated or selectively removed.",
    ]
    (DEST / "calibration_probe.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
