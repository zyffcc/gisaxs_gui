"""Estimate a noise budget without changing observations or fitting scores.

Poisson counting and independent pixels are assumptions, not a measured noise
calibration. The fitted curve is used only as a plug-in simulation mean.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/preprocess_mask_20260921"


def main():
    verified = json.loads((OUT / "single/VERIFIED.json").read_text(encoding="utf-8"))
    job = Path(verified["fit_output"])
    rows = json.loads((job / "top20_candidates.json").read_text(encoding="utf-8"))
    request = json.loads((job / "request.json").read_text(encoding="utf-8"))
    options = request["options"]
    rng = np.random.default_rng(20260921)
    records = []
    for row in rows:
        if row["rank"] != 1:
            continue
        q, y, sigma, mu = (np.asarray(row[k], float) for k in ("native_q", "observed", "sigma", "native_fit"))
        q = abs(q)
        variance = np.maximum(sigma**2 - (options["relative_noise"] * abs(y))**2 - options["absolute_noise"]**2, 1e-30)
        count = np.where(y > 0, y / variance, 1 / np.sqrt(variance))
        count = np.rint(count).astype(int)
        assert np.all(count >= 1)
        positive = y > 0
        log_y = np.log(np.maximum(y, 1e-30))
        spacing = np.diff(q)
        consecutive = (spacing[:-1] < 1.5*np.median(spacing)) & (spacing[1:] < 1.5*np.median(spacing)) & positive[:-2] & positive[1:-1] & positive[2:]
        record = dict(
            side=row["side"], samples=len(y), full_observed_logrmse=row["best_log_rmse"],
            poisson_delta_method_log_rms=float(np.sqrt(np.mean(variance[positive] / y[positive]**2))),
            second_difference_log_rms=float(np.sqrt(np.mean(np.diff(log_y, n=2)[consecutive]**2) / 6)),
            selected_pixel_count_range=[int(count.min()), int(count.max())],
            simulations=[],
        )
        for factor in (1, 4, 9, 16):
            expected_counts = mu * count * factor
            draws = rng.poisson(expected_counts, size=(1000, len(mu))) / (count * factor)
            valid = draws > 0
            errors = np.where(valid, np.log(np.maximum(draws, 1e-30) / mu)**2, 0)
            scores = np.sqrt(errors.sum(axis=1) / valid.sum(axis=1))
            record["simulations"].append(dict(count_multiplier=factor, median_logrmse=float(np.median(scores)), q05=float(np.quantile(scores,.05)), q95=float(np.quantile(scores,.95)), fraction_below_005=float(np.mean(scores < .05))))
        records.append(record)
    result = dict(source=str(job), assumptions="Independent Poisson counts; fitted mean as a plug-in estimate; no calibrated noise floor or repeat-exposure validation. Zero counts omitted only from log metric, as in the existing score; retained in fitting.", records=records)
    (OUT / "noise_budget.json").write_text(json.dumps(result,indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
