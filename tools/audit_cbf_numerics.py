"""Check quadrature and observed-noise averaging without changing fit inputs."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import j1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_cbf_causes import Model, OUT


def fine_component(q, p, factor):
    def nodes(mu, width, n, nsig):
        xx = np.linspace(max(mu - nsig * width, 0), mu + nsig * width, n)
        w = np.exp(-0.5 * ((xx - mu) / width) ** 2)
        return np.maximum(xx, 1e-8), w / w.sum()

    def sphere(x):
        safe = np.where(abs(x) < 0.1, 1, x)
        f = 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3
        return np.where(abs(x) < 0.1, 1 - x * x / 10 + x**4 / 280 - x**6 / 15120, f)

    def radial(x):
        safe = np.where(abs(x) < 1e-4, 1, x)
        return np.where(abs(x) < 1e-4, 1 - x * x / 8 + x**4 / 192, 2 * j1(safe) / safe)

    r, sr, h, sh, d, sd = (p[k] for k in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_D"))
    if p["type"] == 1:
        rr, w = nodes(r, r * sr, 25 * factor, 4)
        form = w @ sphere(rr[:, None] * q) ** 2
    elif p["type"] == 3:
        rr, w = nodes(r, r * sr, 26 * factor, 3)
        form = w @ radial(rr[:, None] * q) ** 2
    else:
        rr, wr = nodes(r, r * sr, 13 * factor, 4)
        hh, wh = nodes(h, h * sh, 13 * factor, 4)
        alpha = np.linspace(0, np.pi / 2, 24 * factor)
        wa = np.sin(alpha)
        wa /= wa.sum()
        fm = np.sum(
            wr[:, None, None]
            * radial(rr[:, None, None] * np.sin(alpha)[None, :, None] * q[None, None, :]) ** 2,
            axis=0,
        )
        hm = np.sum(
            wh[:, None, None]
            * np.sinc(
                hh[:, None, None] * np.cos(alpha)[None, :, None] * q[None, None, :] / (2 * np.pi)
            )
            ** 2,
            axis=0,
        )
        form = np.sum(wa[:, None] * fm * hm, axis=0)
    if p["structure"]:
        lp = -np.pi * q * q * (d * sd) ** 2
        phi = np.exp(lp)
        form *= (-np.expm1(2 * lp)) / np.maximum(
            (-np.expm1(lp)) ** 2 + 4 * phi * np.sin(0.5 * q * d) ** 2, 1e-15
        )
    return form


def main():
    inputs = json.loads((OUT / "inputs.json").read_text())
    records = json.loads((OUT / "range.json").read_text())
    result = []
    for side in ("positive", "negative"):
        best = min(
            (
                r
                for r in records
                if r["task"]["side"] == side and r["task"]["arm"] == "resolution_wide"
            ),
            key=lambda r: r["logrmse"],
        )
        t = best["task"]
        m = Model(inputs[side], t["arm"], t["types"], t["gates"])
        z = best["z"]
        co, parts = m.decode(z)
        pred = m.predict(z)
        checks = []
        for factor in (1, 2, 4, 8):
            fine = np.full_like(m.q, np.exp(co["bg"]))
            for j, p in enumerate(parts):
                fine += np.exp(co[f"a{j}"]) * fine_component(m.q, p, factor)
            fine += np.exp(co["rc"]) / (1 + (m.q / np.exp(co["rs"])) ** co["nu"])
            checks.append(
                dict(
                    quadrature_multiplier=factor,
                    vs_original_forward_logrmse=float(np.sqrt(np.mean(np.log(fine / pred) ** 2))),
                    vs_observed_logrmse=float(np.sqrt(np.mean(np.log(fine / m.y) ** 2))),
                )
            )
        bins = []
        contiguous = np.split(
            np.arange(len(m.q)), np.flatnonzero(np.diff(m.q) > 1.5 * np.median(np.diff(m.q))) + 1
        )
        for size in (1, 2, 4, 8, 16):
            groups = [
                chunk[i : i + size] for chunk in contiguous for i in range(0, len(chunk), size)
            ]
            y = np.array([np.average(m.y[g], weights=m.count[g]) for g in groups])
            f = np.array([np.average(pred[g], weights=m.count[g]) for g in groups])
            bins.append(
                dict(
                    max_columns_per_bin=size,
                    points=len(groups),
                    logrmse=float(np.sqrt(np.mean(np.log(f / y) ** 2))),
                    note="Diagnostic count-weighted binning of BOTH data and forward; changes q resolution. Not the original full-curve score.",
                )
            )
        result.append(dict(side=side, quadrature=checks, binning=bins))
    (OUT / "numerics_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
