"""Population coverage and whole-cut bootstrap intervals for the validation figure."""
from __future__ import annotations

import numpy as np


def coverage_by_cut(results, level: str, key: str, *, stage="fit", confidence=0.68,
                    resamples=10000, seed=0) -> dict:
    """Estimate coverage of the nominal 68% and 95% intervals for a parameter.

    Pass the results list from one recovery run. Predictions are paired to truth
    in radius order on successful cuts of the requested level and matching
    population count. Only populations with an estimate and both intervals enter.
    Each bootstrap sample draws as many whole cuts as contributed to the point
    estimate, with replacement. Divide its total number of hits by its total
    number of eligible populations. The bounds are central percentile intervals.
    """
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be between 0 and 1 and resamples must be positive")
    counts = {}
    for row in results:
        case = row["case"]
        if level != "all" and case["level"] != level:
            continue
        fit = row.get(stage) or {}
        truth = sorted(case["populations"], key=lambda p: p["R"])
        if not fit.get("ok") or len(fit["populations"]) != len(truth):
            continue
        eligible = hit68 = hit95 = 0
        for actual, estimate in zip(truth, fit["populations"]):
            value = actual.get(key)
            intervals = [estimate.get(key + suffix) for suffix in ("_68", "_95")]
            if (value is None or not np.isfinite(value) or estimate.get(key) is None
                    or not np.isfinite(estimate[key]) or any(bounds is None for bounds in intervals)):
                continue
            for bounds in intervals:
                if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                    raise ValueError(f"Invalid {key} interval on cut {case['index']}")
            eligible += 1
            hit68 += int(intervals[0][0] <= value <= intervals[0][1])
            hit95 += int(intervals[1][0] <= value <= intervals[1][1])
        if eligible:
            if case["index"] in counts:
                raise ValueError(f"Repeated cut {case['index']} in recovery run")
            counts[case["index"]] = (eligible, hit68, hit95)

    report = dict(cuts=len(counts), populations=0, hits68=0, hits95=0,
                  coverage68=None, coverage95=None, bootstrap68=None, bootstrap95=None,
                  confidence=confidence, resamples=resamples, seed=seed)
    if not counts:
        return report
    # Worker completion order can vary between runs.
    cuts = np.array([counts[i] for i in sorted(counts)])
    populations, hits68, hits95 = map(int, cuts.sum(axis=0))
    rng = np.random.default_rng(seed)
    shares = np.empty((resamples, 2))
    for start in range(0, resamples, 256):
        batch = shares[start:start + 256]
        selected = rng.integers(len(cuts), size=(len(batch), len(cuts)))
        totals = cuts[selected].sum(axis=1)
        batch[:] = totals[:, 1:] / totals[:, :1]
    tail = (1 - confidence) / 2
    limits68, limits95 = np.quantile(shares, [tail, 1 - tail], axis=0).T
    report.update(populations=populations, hits68=hits68, hits95=hits95,
                  coverage68=hits68 / populations, coverage95=hits95 / populations,
                  bootstrap68=limits68.tolist(), bootstrap95=limits95.tolist())
    return report
