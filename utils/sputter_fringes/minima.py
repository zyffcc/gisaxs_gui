"""Clean alpha_f ranges, the smooth baseline, static detector rows and the minima of a vertical cut.

Clean ranges (from the data and the geometry):

- A cell whose exposure (summed over the scan and the band's q_y bins) is below ``GAP_DENSITY`` times the median is
  not used; a run of such cells at least ``GAP_MIN_DEG`` long is a gap (a detector module gap, the detector's edge)
  and is widened by one cell on each side (module edges); a single such cell is only left out, it does not split
  the range.
- Below max(alpha_i + ``SPECULAR_MARGIN_DEG``, alpha_c + ``YONEDA_MARGIN_DEG``) nothing is used (the specular
  reflection, the Yoneda region of the metal and the steep kz near alpha_c).
- Runs shorter than ``MIN_RANGE_DEG`` are not used.

Baseline: ln I = a + b ln q_z (a power law of the vacuum q_z), fitted to the whole clean range of the cut (weighted
by the counts) and subtracted; a local baseline would absorb the fringes.

Static rows: a cell whose narrow residual (the residual minus its running median over 7 cells) has the same value
over the whole scan (the median over the time groups), larger than ``STATIC_MIN`` and ``STATIC_SIGMAS`` robust
standard deviations of all cells, is a detector feature and is left out of every cut.

Minimum: a local minimum of the residual's running mean over ``SMOOTH_CELLS`` cells (the lowest within 3 cells);
for each scale w of ``SCALES_DEG`` that fits inside the range, a weighted parabola within min(w, 0.05) deg places
it (the vertex within 0.02 deg of the local minimum) and its depth is the mean of the two shoulders at +-w minus the
centre (each the summed counts over the summed exposure of the cells within max(0.01, w/4) deg). It counts when its
depth exceeds ``DEPTH_SIGMAS`` times its Poisson error and ``DEPTH_MIN`` (the detector rows' systematic floor) at one
scale at least (the most significant is kept). Its position error comes from a Poisson bootstrap of the counts
(``BOOT_DRAWS`` draws, the parabola at the same scale around the draw's own lowest point); it is kept when at least
``BOOT_NEED`` draws agree within ``AGREE_DEG``, and the standard deviation of those (at least ``SD_FLOOR_DEG``) is its
error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SMOOTH_CELLS = 5
LOCAL_CELLS = 3
SCALES_DEG = (0.03, 0.06, 0.12)
FIT_HALF_MAX_DEG = 0.05
VERTEX_SHIFT_DEG = 0.02
DEPTH_SIGMAS = 3.0
DEPTH_MIN = 0.006
AGREE_DEG = 0.05
BOOT_DRAWS = 30
BOOT_NEED = 10
SD_FLOOR_DEG = 0.002
GAP_DENSITY = 0.5
GAP_MIN_DEG = 0.02
MIN_RANGE_DEG = 0.12
SPECULAR_MARGIN_DEG = 0.12
YONEDA_MARGIN_DEG = 0.20
STATIC_MIN = 0.006
STATIC_SIGMAS = 4.0
STATIC_WINDOW = 7


@dataclass(frozen=True)
class Minimum:
    af: float
    sd: float
    agree: int
    depth: float
    significance: float
    scale: float


def lower_bound(alpha_i_deg: float, alpha_c_deg: float) -> tuple:
    """(alpha_f below which nothing is used, which exclusion sets it)."""
    return max((alpha_i_deg + SPECULAR_MARGIN_DEG, "specular reflection (alpha_i + 0.12 deg)"),
               (alpha_c_deg + YONEDA_MARGIN_DEG, "Yoneda region of the metal (alpha_c + 0.2 deg)"))


def cell_quality(exposure_density) -> tuple:
    """(usable cells, gap cells) of one side from its exposure per cell: gaps are widened by one cell."""
    density = np.asarray(exposure_density, dtype=float)
    low = density < GAP_DENSITY * np.median(density)
    gap = np.zeros_like(low)
    i, n = 0, len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        if j - i + 1 >= round(GAP_MIN_DEG / 0.01):
            gap[max(i - 1, 0):min(j + 2, n)] = True
        i = j + 1
    return ~low & ~gap, gap


def gap_ranges(gap, cell_edges) -> list:
    out, i, n = [], 0, len(gap)
    while i < n:
        if not gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and gap[j + 1]:
            j += 1
        out.append((float(cell_edges[i]), float(cell_edges[j + 1])))
        i = j + 1
    return out


def clean_ranges(usable, gap, cell_edges, lower: float, upper: float = np.inf) -> list:
    """Clean alpha_f ranges [(lo, hi)] (cell edges) between the gaps, above ``lower``."""
    cell_edges = np.asarray(cell_edges, dtype=float)
    inside = (cell_edges[:-1] >= lower - 1e-9) & (cell_edges[1:] <= upper + 1e-9) & ~np.asarray(gap)
    usable = np.asarray(usable)
    out, i, n = [], 0, len(inside)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        cells = np.flatnonzero(usable[i:j + 1]) + i
        if len(cells) and cell_edges[cells[-1] + 1] - cell_edges[cells[0]] >= MIN_RANGE_DEG - 1e-9:
            out.append((float(cell_edges[cells[0]]), float(cell_edges[cells[-1] + 1])))
        i = j + 1
    return out


def baseline(af, y, e, qz) -> np.ndarray:
    """ln I of the power law a + b ln q_z fitted to the cells given (weighted by their counts)."""
    ok = (y > 0) & (e > 0)
    if ok.sum() < 3:
        return np.zeros_like(af)
    L = np.log(qz)
    coefficients = np.polyfit(L[ok], np.log(y[ok] / e[ok]), 1, w=np.sqrt(y[ok]))
    return np.polyval(coefficients, L)


def _running_median(values: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def static_cells(residuals: np.ndarray) -> np.ndarray:
    """Cells with a narrow residual that stays over the scan (``residuals``: groups x cells, NaN where unused)."""
    narrow = np.full(residuals.shape, np.nan)
    for i, row in enumerate(residuals):
        ok = np.isfinite(row)
        if ok.sum() >= STATIC_WINDOW:
            narrow[i, ok] = row[ok] - _running_median(row[ok], STATIC_WINDOW)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            template = np.nanmedian(narrow, axis=0)
    finite = np.isfinite(template)
    if finite.sum() < 10:
        return np.zeros(residuals.shape[1], bool)
    spread = 1.4826 * np.median(np.abs(template[finite] - np.median(template[finite])))
    threshold = max(STATIC_MIN, STATIC_SIGMAS * spread)
    return finite & (np.abs(np.nan_to_num(template)) > threshold)


def _smooth(r: np.ndarray) -> np.ndarray:
    """Running mean over SMOOTH_CELLS cells (+inf where the window is not complete)."""
    half = SMOOTH_CELLS // 2
    out = np.full(len(r), np.inf)
    if len(r) >= SMOOTH_CELLS:
        out[half:len(r) - half] = np.convolve(r, np.ones(SMOOTH_CELLS) / SMOOTH_CELLS, mode="valid")
    return out


def _parabola(a, r, w, a0):
    x = a - a0
    X = np.stack([np.ones_like(x), x, x * x], axis=1)
    A = X.T @ (X * w[:, None])
    try:
        c = np.linalg.solve(A, X.T @ (w * r))
    except np.linalg.LinAlgError:
        return None
    return (a0 - c[1] / (2 * c[2])) if c[2] > 0 else None


def _depth(a, y, eb, x0, scale):
    """(depth, its Poisson error) of the dip at ``x0`` against its shoulders at +-scale (``eb``: exposure times the
    baseline, so that ln(y / eb) is the residual)."""
    band = max(0.01, scale / 4) + 1e-9
    parts = []
    for centre in (x0 - scale, x0 + scale, x0):
        cells = np.abs(a - centre) < band
        if not cells.any() or y[cells].sum() <= 0:
            return None
        parts.append((np.log(y[cells].sum() / eb[cells].sum()), 1.0 / y[cells].sum()))
    (left, vl), (right, vr), (centre, vc) = parts
    return 0.5 * (left + right) - centre, float(np.sqrt(vc + 0.25 * (vl + vr)))


def _locate(a, y, eb, j_hint=None):
    """[(vertex, depth, significance, scale)] of the dips of one range."""
    r = np.log((y + 0.5) / eb)
    smooth = _smooth(r)
    w = y + 0.5
    found = []
    candidates = range(2, len(a) - 2) if j_hint is None else [j_hint]
    for j in candidates:
        if smooth[j] > smooth[max(j - LOCAL_CELLS, 0):j + LOCAL_CELLS + 1].min():
            continue
        best = None
        for scale in SCALES_DEG:
            if a[j] - a[0] < scale - 1e-9 or a[-1] - a[j] < scale - 1e-9:
                continue
            cells = np.abs(a - a[j]) <= min(scale, FIT_HALF_MAX_DEG) + 1e-9
            if cells.sum() < 5:
                continue
            x0 = _parabola(a[cells], r[cells], w[cells], a[j])
            if x0 is None or abs(x0 - a[j]) > VERTEX_SHIFT_DEG:
                continue
            dip = _depth(a, y, eb, x0, scale)
            if dip is None:
                continue
            depth, error = dip
            if depth > DEPTH_SIGMAS * error and depth > DEPTH_MIN and (best is None or depth / error > best[2]):
                best = (float(x0), float(depth), float(depth / error), scale)
        if best is not None:
            found.append(best)
    merged = []
    for item in sorted(found):
        if merged and abs(item[0] - merged[-1][0]) < 0.01:
            if item[2] > merged[-1][2]:
                merged[-1] = item
        else:
            merged.append(item)
    return merged


def _bootstrap(a, y, eb, vertex, scale, rng) -> tuple:
    found = []
    half = min(scale, FIT_HALF_MAX_DEG) + 1e-9
    near = np.abs(a - vertex) <= AGREE_DEG
    for draw in rng.poisson(y, size=(BOOT_DRAWS, len(y))).astype(float):
        r = np.log((draw + 0.5) / eb)
        smooth = _smooth(r)
        if not np.isfinite(smooth[near]).any():
            continue
        j = int(np.flatnonzero(near)[np.argmin(smooth[near])])
        cells = np.abs(a - a[j]) <= half
        if cells.sum() < 5:
            continue
        x0 = _parabola(a[cells], r[cells], draw[cells] + 0.5, a[j])
        if x0 is not None and abs(x0 - vertex) <= AGREE_DEG and a[cells].min() < x0 < a[cells].max():
            found.append(x0)
    sd = float(np.std(found)) if len(found) > 1 else 0.0
    return len(found), max(sd, SD_FLOOR_DEG)


def find_minima(af, counts, exposure, base, ranges: list, usable, rng) -> list:
    """The minima of one side's cut in its clean ``ranges`` (``base``: the baseline's ln I per cell)."""
    af, counts, exposure = (np.asarray(v, dtype=float) for v in (af, counts, exposure))
    out = []
    for lo, hi in ranges:
        cells = (af > lo) & (af < hi) & usable & (exposure > 0)
        if cells.sum() < 2 * SMOOTH_CELLS:
            continue
        a, y = af[cells], counts[cells]
        eb = exposure[cells] * np.exp(base[cells])
        for vertex, depth, significance, scale in _locate(a, y, eb):
            agree, sd = _bootstrap(a, y, eb, vertex, scale, rng)
            if agree >= BOOT_NEED:
                out.append(Minimum(vertex, sd, agree, depth, significance, scale))
    return out


def settings() -> dict:
    """The locator's constants (recorded in the summary)."""
    return {"smooth_cells": SMOOTH_CELLS, "scales_deg": list(SCALES_DEG), "depth_sigmas": DEPTH_SIGMAS,
            "depth_min_ln": DEPTH_MIN, "agree_deg": AGREE_DEG, "bootstrap_draws": BOOT_DRAWS,
            "bootstrap_need": BOOT_NEED, "sd_floor_deg": SD_FLOOR_DEG, "gap_density": GAP_DENSITY,
            "gap_min_deg": GAP_MIN_DEG, "min_range_deg": MIN_RANGE_DEG, "specular_margin_deg": SPECULAR_MARGIN_DEG,
            "yoneda_margin_deg": YONEDA_MARGIN_DEG, "static_min_ln": STATIC_MIN, "static_sigmas": STATIC_SIGMAS}


__all__ = ["Minimum", "baseline", "cell_quality", "clean_ranges", "find_minima", "gap_ranges", "lower_bound",
           "settings", "static_cells"]
