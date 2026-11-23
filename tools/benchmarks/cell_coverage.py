"""Cell coverage from indexed centres and from unlabelled, unresolved simulated peak positions."""

from __future__ import annotations

from itertools import product
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.analyze.application.cell_assignments import fit_assignments
from src.gimap.features.analyze.application.indexing import PhaseSpec, assign_peaks
from src.gimap.features.analyze.domain.crystal.cell import UnitCell
from src.gimap.features.analyze.domain.crystal.refinement import CellPeak, refine_cell


CELLS = {
    "cubic": ((5., 5., 5.), (5.018, 5.018, 5.018)),
    "tetragonal": ((5., 5., 7.), (5.018, 5.018, 6.979)),
    "hexagonal": ((4., 4., 7., 90., 90., 120.), (4.012, 4.012, 6.979, 90., 90., 120.)),
    "trigonal": ((4., 4., 6., 90., 90., 120.), (4.012, 4.012, 5.982, 90., 90., 120.)),
    "trigonal_R": ((5., 5., 5., 78., 78., 78.), (5.01, 5.01, 5.01, 77.85, 77.85, 77.85)),
    "orthorhombic": ((5., 6., 7.), (5.018, 5.985, 7.014)),
    "monoclinic": ((5., 6., 7., 90., 106., 90.), (5.018, 5.985, 7.014, 90., 105.85, 90.)),
    "triclinic": ((5., 6., 7., 84., 97., 106.), (5.018, 5.985, 7.014, 84.1, 96.9, 106.15)),
}
SHIFT = 0.0015  # Å⁻¹


def _unlabelled(trials):
    """Only noisy centres, widths and errors enter indexing. Ground truth stays with the scorer."""
    rng = np.random.default_rng(70814)
    a, c = 6.018, 8.485
    spec = PhaseSpec("Tetragonal simulation", (6., 6., 8.40, 90., 90., 90.))
    # Bragg spacings generated independently of the indexing catalogue. Equivalents
    # share one radius. Instrumentally unresolved lines form one measured centroid.
    lines = sorted((2*np.pi*np.sqrt((h*h + k*k)/a**2 + l*l/c**2) + SHIFT, (h,k,l))
                   for h, k, l in product(range(5), repeat=3) if h >= k and h+k+l)
    width = .027
    groups = []
    for q, hkl in lines:
        if q > 3.4:
            break
        if groups and q - groups[-1][0][0] < .75*width:
            groups[-1].append((q, hkl))
        else:
            groups.append([(q, hkl)])
    estimates, errors, rows = [], [], []
    for draw in range(trials):
        measured = []
        for group in groups:
            weights = rng.uniform(.65, 1.35, len(group))
            sigma = rng.uniform(.00025, .0006)
            q = np.average([g[0] for g in group], weights=weights) + rng.normal(0., sigma)
            measured.append({"q": q, "q_err": sigma, "fwhm": width})
        assigned = assign_peaks(measured, [spec])
        (fit,), _, _ = fit_assignments([spec], assigned)
        used = [r["source_index"] for r in fit["residuals"]]
        rows.append({"draw": draw, "ok": fit["ok"], "reason": fit["reason"], "used": len(used),
                     "retained_overlaps": sum(len(groups[i]) > 1 for i in used),
                     "wrong_singletons": sum(len(groups[r["source_index"]]) == 1 and
                         tuple(r["hkl"]) != groups[r["source_index"]][0][1] for r in fit["residuals"]),
                     "reassignment_passes": len(fit["assignment_iterations"]) - 1,
                     "values": fit["values"], "errors": fit["errors"], "zero_shift": fit.get("zero_shift"),
                     "zero_shift_error": fit.get("zero_shift_error")})
        if fit["ok"] and fit["zero_shift_fitted"]:
            estimates.append([fit["values"]["a"], fit["values"]["c"], fit["zero_shift"]])
            errors.append([fit["errors"]["a"], fit["errors"]["c"], fit["zero_shift_error"]])
    parameters = {}
    if estimates:
        for j, (name, true) in enumerate((("a", a), ("c", c), ("zero_shift", SHIFT))):
            values, sigmas = np.array(estimates)[:, j], np.array(errors)[:, j]
            parameters[name] = {"truth": true, "mean": float(values.mean()), "mean_sigma": float(sigmas.mean()),
                                "coverage": float(np.mean(abs(values-true) <= sigmas))}
    return {"seed": 70814, "trials": trials, "successful_with_offset": len(estimates), "peaks": len(groups),
            "overlaps_per_draw": sum(len(group) > 1 for group in groups), "parameters": parameters, "draws": rows}


def _rings(cell):
    grid = np.array([h for h in product(range(5), range(-3, 4), range(-3, 4)) if any(h)])
    q = cell.q_of(grid)
    order = np.argsort(q)
    # Each measurement is a separate powder position, including in the cubic test.
    order = order[np.r_[True, np.diff(q[order]) > 1e-7]]
    return grid[order[np.linspace(0, len(order) - 1, min(30, len(order)), dtype=int)]]


def coverage(trials=100):
    """JSON-ready means and empirical one-standard-error coverage for each crystal system."""
    if trials < 1:
        raise ValueError("A positive number of noise draws is required.")
    rng = np.random.default_rng(90873)
    output = {}
    for case, (start, end) in CELLS.items():
        reference, target = UnitCell(*start), UnitCell(*end)
        hkl = _rings(reference)
        q = target.q_of(hkl) + SHIFT
        sigma = np.linspace(0.0002, 0.0008, len(q))
        measurements, uncertainties = [], []
        for _ in range(trials):
            noisy = rng.normal(q, sigma)
            peaks = [CellPeak(tuple(h), v, s) for h, v, s in zip(hkl, noisy, sigma)]
            fit = refine_cell(reference, case.split("_")[0], peaks)
            if not fit["ok"] or not fit["zero_shift_fitted"]:
                raise RuntimeError(f"{case}: {fit['reason']}")
            names = [*fit["values"], "zero_shift"]
            measurements.append([*fit["values"].values(), fit["zero_shift"]])
            uncertainties.append([*fit["errors"].values(), fit["zero_shift_error"]])
        truth = {**dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), target.as_tuple())),
                 "V": target.volume, "zero_shift": SHIFT}
        if case == "trigonal_R":
            cosine = math.cos(math.radians(target.alpha))
            truth.update(a=target.a * math.sqrt(2*(1-cosine)), c=target.a * math.sqrt(3*(1+2*cosine)), V=3*target.volume)
        values, errors = np.array(measurements), np.array(uncertainties)
        parameters = {}
        for j, name in enumerate(names):
            parameters[name] = {"truth": truth[name], "mean": float(values[:, j].mean()),
                                "coverage": float(np.mean(np.abs(values[:, j] - truth[name]) <= errors[:, j])),
                                "mean_sigma": float(errors[:, j].mean())}
        output[case] = {"trials": trials, "rings": len(q), "parameters": parameters}
    output["unlabelled_overlapping_positions"] = _unlabelled(trials)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = coverage(args.trials)
    for case, run in result.items():
        print(case, {k: v["coverage"] for k, v in run["parameters"].items()})
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
