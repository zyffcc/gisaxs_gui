"""Known cells, independent ring-centre errors and a fixed additive shift in q."""

from __future__ import annotations

from itertools import product
import math

import numpy as np

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
    return output
