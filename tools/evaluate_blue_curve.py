"""Evaluate frozen local pilot on synthetic TEST and three CBF frames."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.blue_curve_distillation import OUT, AUDIT, features, forward, decode, build_model
from tools.audit_cbf_counting import band
from tools.audit_cbf_causes import Model


def frame_inputs():
    original = json.loads((AUDIT / "inputs.json").read_text())
    meta = json.loads((AUDIT / "provenance.json").read_text())
    root = ROOT / "TestSAXSdata"
    prefix = "jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_"
    base, _ = band(root / f"{prefix}00033.cbf", meta["mask"]["pixel_region"])
    valid = np.isfinite(base).any(axis=0)
    signed_q = np.r_[-np.array(original["negative"]["q"])[::-1], original["positive"]["q"]]
    result = []
    for frame in ("00033", "00005", "00045"):
        values, _ = band(root / f"{prefix}{frame}.cbf", meta["mask"]["pixel_region"])
        values = values[:, valid]
        count = np.isfinite(values).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = np.nanmean(values, axis=0)
        for side, sign in [("positive", 1), ("negative", -1)]:
            idx = np.flatnonzero((signed_q * sign > 0) & np.isfinite(y) & (count > 0))
            idx = idx[np.argsort(abs(signed_q[idx]))]
            d = dict(
                frame=frame,
                side=side,
                q=abs(signed_q[idx]).tolist(),
                y=y[idx].tolist(),
                count=count[idx].tolist(),
            )
            if frame == "00033":
                np.testing.assert_allclose(d["y"], original[side]["y"])
            result.append(d)
    return result


def reference_one(d):
    records = json.loads((AUDIT / "gauss48.json").read_text())
    teacher = next(
        r
        for r in records
        if r["task"]["side"] == d["side"]
        and r["task"]["arm"] == "resolution_wide"
        and "split" not in r["task"]
    )
    m = Model(d, "resolution_wide", [2], [True], quadrature="gauss48")
    tick = time.perf_counter()
    if d["frame"] == "00033":
        z = np.array(teacher["z"])
        nfev = 0
        success = True
    else:
        z0 = np.array(teacher["z"])
        fit = least_squares(
            lambda z: np.log(m.predict(z) / m.y),
            z0,
            bounds=(m.lo, m.hi),
            max_nfev=350,
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-8,
        )
        z = fit.x
        nfev = fit.nfev
        success = bool(fit.success)
    pred = m.predict(z)
    return dict(
        **d,
        reference=pred.tolist(),
        reference_z=z.tolist(),
        reference_names=m.names,
        reference_parameters=m.decode(z)[1],
        reference_logrmse=float(np.sqrt(np.mean(np.log(pred / m.y) ** 2))),
        reference_seconds=time.perf_counter() - tick,
        reference_nfev=nfev,
        reference_converged=success,
        note="Numerical reference, not known clean experimental truth. 00033 is a development frame; 00005/00045 were not used for training or checkpoint selection.",
    )


def references():
    OUT.mkdir(exist_ok=True, parents=True)
    jobs = frame_inputs()
    records = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(reference_one, jobs):
            records.append(r)
            (OUT / "real_references.json").write_text(json.dumps(records, indent=2))
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "frame",
                            "side",
                            "reference_logrmse",
                            "reference_seconds",
                            "reference_converged",
                        )
                    }
                ),
                flush=True,
            )


def evaluate():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(OUT / "dataset.npz")
    sc = np.load(OUT / "feature_scaling.npz")
    n = int(data["ntrain"]) + int(data["nval"])
    x = (data["x"][n:] - sc["mean"]) / sc["scale"]
    real = json.loads((OUT / "real_references.json").read_text())
    rx = np.array([features(r["q"], r["y"], r["count"]) for r in real])
    rx = (rx - sc["mean"]) / sc["scale"]
    result = dict(
        synthetic_test_rows=len(x),
        models={},
        real=[],
        scope="Single RC local branch. No refinement during network prediction. Real references are fitted curves, not ground truth.",
    )
    for label, weights in [
        ("parameters_only", "parameter_only.weights.h5"),
        ("curve_supervised", "curve_best.weights.h5"),
    ]:
        net = build_model()
        net.load_weights(str(OUT / weights))
        predicted = net(x, training=False).numpy()
        errors = []
        parameter_errors = []
        for j, uu in enumerate(predicted):
            i = n + j
            use = data["mask"][i] > 0
            curve = forward(data["q"][i, use], uu)
            errors.append(float(np.sqrt(np.mean(np.log(curve / data["clean"][i, use]) ** 2))))
            parameter_errors.append(float(np.sqrt(np.mean((uu - data["u"][i]) ** 2))))
        result["models"][label] = dict(
            clean_curve_median=float(np.median(errors)),
            clean_curve_p90=float(np.quantile(errors, 0.9)),
            clean_curve_mean=float(np.mean(errors)),
            fraction_clean_below_005=float(np.mean(np.array(errors) < 0.05)),
            parameter_normalized_rmse_median=float(np.median(parameter_errors)),
            errors=errors,
        )
        # Warm latency: feature extraction + one NN call + physical forward, CPU.
        net(rx[:1], training=False)
        for j, r in enumerate(real):
            q = np.array(r["q"])
            y = np.array(r["y"])
            reference = np.array(r["reference"])
            times = []
            for _ in range(5):
                tick = time.perf_counter()
                feat = (features(q, y, r["count"]) - sc["mean"]) / sc["scale"]
                uu = net(feat[None], training=False).numpy()[0]
                curve = forward(q, uu)
                times.append(time.perf_counter() - tick)
            rec = dict(
                frame=r["frame"],
                side=r["side"],
                model=label,
                parameters=decode(uu),
                normalized_parameters=uu.tolist(),
                observed_logrmse=float(np.sqrt(np.mean(np.log(curve / y) ** 2))),
                reference_logrmse=r["reference_logrmse"],
                vs_reference_logrmse=float(np.sqrt(np.mean(np.log(curve / reference) ** 2))),
                warm_median_seconds=float(np.median(times)),
                forward_curve=curve.tolist(),
                q=q.tolist(),
                refinement=False,
            )
            peak_region = (q >= 1.0) & (q <= 2.0)
            pq = q[peak_region]
            reference_peak = int(np.argmax(reference[peak_region]))
            predicted_peak = int(np.argmax(curve[peak_region]))
            rec["sidepeak_q_error_nm_inverse"] = float(abs(pq[predicted_peak] - pq[reference_peak]))
            rec["sidepeak_height_relative_error"] = float(
                curve[peak_region][predicted_peak] / reference[peak_region][reference_peak] - 1
            )
            result["real"].append(rec)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in rec.items()
                        if k not in ("q", "forward_curve", "parameters", "normalized_parameters")
                    }
                ),
                flush=True,
            )
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, r in zip(axes.flat, real):
        q = np.array(r["q"])
        ax.scatter(q, r["y"], s=4, c="#888", alpha=0.5, label="Measured")
        ax.plot(
            q,
            r["reference"],
            c="#2563b6",
            lw=2,
            label=f"Numerical reference: {r['reference_logrmse']:.3f}",
        )
        for label, color in [("parameters_only", "#dc9540"), ("curve_supervised", "#b7406b")]:
            rec = next(
                a
                for a in result["real"]
                if a["model"] == label and a["frame"] == r["frame"] and a["side"] == r["side"]
            )
            ax.plot(
                q,
                rec["forward_curve"],
                c=color,
                lw=1.5,
                label=f"{label}: {rec['observed_logrmse']:.3f}",
            )
        ax.set_yscale("log")
        ax.set_title(f"CBF {r['frame']} / {r['side']}")
        ax.set_xlabel("q (nm^-1)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Direct neural parameters + physical forward; no numerical refinement\n00033 = development; 00005 / 00045 = same-series held-out frames (not independent experiments)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "direct_comparison.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["references", "evaluate"])
    a = p.parse_args()
    references() if a.phase == "references" else evaluate()
