"""Phase-locked fringe fit: H of a time group from the whole cut, without counting orders.

Each side's cut is fitted with ln I = c0 + c1 ln q_z + c2 ln^2 q_z + A cos(H Q), A < 0, Q inside the layer (minima at
H Q = 2 pi n), H on a grid (``H_GRID``): the cosine is projected out of the baseline and the H that lowers chi^2 most
is kept, its error from the curvature of chi^2 (scaled by the reduced chi^2 when above 1). The fit tells H only when
the range holds enough of the fringe: at least ``MIN_PERIODS`` periods, and the best H at least ``ALIAS_DCHI2`` better
(in chi^2) than the best one more than ``ALIAS_FRACTION`` away from it (an alias); with one or two periods in range it
aliases, and the minima tracking gives H instead.

A group's phase-locked H is accepted when both sides pass those tests and agree (within ``SIDE_REL_TOL`` or three
times their combined error), and when it lies within ``TREND_REL_TOL`` of a straight line through the accepted
groups (iterated; only the longest run of accepted groups with at most ``BLOCK_GAP_GROUPS`` missing ones is kept:
the time the fringes are seen).

alpha_c: with at least ``AC_MIN_GROUPS`` accepted groups holding ``AC_MIN_PERIODS`` periods or more, the alpha_c
(between ``AC_RANGE`` times the bulk value) at which the phase-locked fits of those groups have the lowest summed
chi^2 (``alpha_c_scan``): it sets how alpha_f maps onto Q, and the fringe is periodic in Q only with the right one.
"""

from __future__ import annotations

import math

import numpy as np

H_GRID = (1.5, 80.0, 0.05)
ALIAS_FRACTION = 0.25
ALIAS_DCHI2 = 10.0
MIN_PERIODS = 1.5
SIDE_REL_TOL = 0.03
TREND_REL_TOL = 0.10
BLOCK_GAP_GROUPS = 5
TREND_ROUNDS = 6
AC_MIN_GROUPS = 8
AC_MIN_PERIODS = 2.0
"""Only accepted groups with at least this many periods in range (both sides) enter the alpha_c scan."""
AC_RANGE = (0.6, 1.25)
"""The alpha_c scan of the phase-locked fit runs over these multiples of the bulk value."""
AC_STEP = 0.0025
AC_ERROR_FLOOR = 0.01


def heights_grid() -> np.ndarray:
    return np.arange(*H_GRID)


def fit(q_in, ln_qz, y, e, grid=None):
    """{H, error, gain, alias_dchi2, amplitude, periods} of one side's cut (cells with counts), or None."""
    ok = (y > 0) & (e > 0)
    if ok.sum() < 12:
        return None
    grid = heights_grid() if grid is None else grid
    x, L, w = q_in[ok], ln_qz[ok], y[ok].astype(float)
    lr = np.log(y[ok] / e[ok])
    sw = np.sqrt(w)
    X0 = np.stack([np.ones_like(L), L, L * L], axis=1)
    c0, *_ = np.linalg.lstsq(X0 * sw[:, None], lr * sw, rcond=None)
    r0 = (lr - X0 @ c0) * sw
    chi0 = float(r0 @ r0)
    basis, _ = np.linalg.qr(X0 * sw[:, None])
    C = np.cos(np.outer(grid, x)) * sw
    C -= (C @ basis) @ basis.T
    num = C @ r0
    den = np.einsum("ij,ij->i", C, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = num / den
        gain = np.where((amplitude < 0) & (den > 0), num * num / den, 0.0)
    j = int(np.argmax(gain))
    if gain[j] <= 0:
        return None
    reduced = max((chi0 - gain[j]) / max(len(x) - 4, 1), 1.0)
    near = slice(max(j - 3, 0), min(j + 4, len(grid)))
    curvature = np.polyfit(grid[near] - grid[j], -gain[near], 2)[0] if j - 3 >= 0 and j + 4 <= len(grid) else 0.0
    error = math.sqrt(reduced / curvature) if curvature > 0 else float("nan")
    far = np.abs(grid - grid[j]) > ALIAS_FRACTION * grid[j]
    alias = float(np.max(np.where(far, gain, -np.inf))) if far.any() else 0.0
    return {"H": float(grid[j]), "error": error, "gain": float(gain[j] / reduced),
            "alias_dchi2": float((gain[j] - max(alias, 0.0)) / reduced), "amplitude": float(-amplitude[j]),
            "periods": float(grid[j] * (x.max() - x.min()) / (2 * math.pi)), "chi2": float(chi0 - gain[j]),
            "cells": int(len(x))}


def side_check(a, b):
    """(H, error) of the group when both sides pass and agree, else (None, reason)."""
    if a is None or b is None:
        return None, "no fit on one side"
    for side in (a, b):
        if not side["periods"] >= MIN_PERIODS:
            return None, f"fewer than {MIN_PERIODS} periods in range"
        if not side["alias_dchi2"] >= ALIAS_DCHI2:
            return None, "an alias fits about as well"
        if not math.isfinite(side["error"]):
            return None, "no error"
    combined = math.hypot(a["error"], b["error"])
    h = (a["H"] + b["H"]) / 2
    if abs(a["H"] - b["H"]) > max(3 * combined, SIDE_REL_TOL * h):
        return None, "the sides disagree"
    return (h, combined / 2), ""


def trend_accept(times, heights, passed) -> np.ndarray:
    """Groups whose H lies within TREND_REL_TOL of a straight line through the passing groups (the longest run)."""
    times, heights = np.asarray(times, dtype=float), np.asarray(heights, dtype=float)
    passed = np.asarray(passed, bool)
    keep = passed.copy()
    for _ in range(TREND_ROUNDS):
        if keep.sum() < 4:
            return np.zeros_like(keep)
        line = np.polyval(np.polyfit(times[keep], heights[keep], 1), times)
        new = passed & (np.abs(heights - line) <= TREND_REL_TOL * np.abs(line))
        index = np.flatnonzero(new)
        if len(index) == 0:
            return new
        blocks = np.split(index, np.flatnonzero(np.diff(index) > BLOCK_GAP_GROUPS) + 1)
        largest = max(blocks, key=len)
        new = np.zeros_like(new)
        new[largest] = True
        if np.array_equal(new, keep):
            break
        keep = new
    return keep if keep.sum() >= 4 else np.zeros_like(keep)


def alpha_c_scan(cuts: list, alpha_i: float, k: float, bulk: float, upper: float, q_inside, grid=None) -> dict:
    """alpha_c that fits the accepted groups best: the summed chi^2 of the phase-locked fits (each searched within
    25% of its accepted H) over alpha_c. ``cuts``: [(H, af, ln q_z, counts, exposure)] of the accepted groups' sides.
    Its error: where the summed chi^2 rises by its reduced value (at least ``AC_ERROR_FLOOR``)."""
    record = {"fitted": False, "groups_sides": len(cuts)}
    if len(cuts) < 2 * AC_MIN_GROUPS:
        record["reason"] = (f"the phase-locked fit is accepted in fewer than {AC_MIN_GROUPS} groups with "
                            f"{AC_MIN_PERIODS:g} periods or more in range")
        return record
    grid = heights_grid() if grid is None else grid
    low, high = AC_RANGE[0] * bulk, min(AC_RANGE[1] * bulk, upper)
    values = np.arange(low, high + 1e-9, AC_STEP)
    totals, cells = [], sum(c[3].size for c in cuts)
    for ac in values:
        total = 0.0
        for h, af, ln_qz, y, e in cuts:
            near = grid[(grid > 0.75 * h) & (grid < 1.25 * h)]
            result = fit(q_inside(af, alpha_i, ac, k), ln_qz, y, e, near) if len(near) else None
            total += result["chi2"] if result else np.inf
        totals.append(total)
    totals = np.array(totals)
    best = int(np.argmin(totals))
    if not np.isfinite(totals[best]) or best in (0, len(values) - 1):
        record["reason"] = "the best alpha_c of the phase-locked fit lies at the edge of the scan"
        return record
    reduced = max(totals[best] / max(cells - 4 * len(cuts), 1), 1.0)
    inside = values[totals <= totals[best] + reduced]
    record.update(fitted=True, alpha_c_deg=float(values[best]),
                  error_deg=max(float(inside.max() - inside.min()) / 2, AC_ERROR_FLOOR),
                  scan_deg=[float(low), float(high), AC_STEP])
    return record


def settings() -> dict:
    return {"H_grid_nm": list(H_GRID), "min_periods": MIN_PERIODS, "alias_fraction": ALIAS_FRACTION,
            "alias_dchi2": ALIAS_DCHI2, "side_rel_tol": SIDE_REL_TOL, "trend_rel_tol": TREND_REL_TOL,
            "block_gap_groups": BLOCK_GAP_GROUPS, "alpha_c_min_groups": AC_MIN_GROUPS,
            "alpha_c_min_periods": AC_MIN_PERIODS,
            "alpha_c_range_of_bulk": list(AC_RANGE), "alpha_c_error_floor_deg": AC_ERROR_FLOOR}


__all__ = ["alpha_c_scan", "fit", "heights_grid", "settings", "side_check", "trend_accept"]
