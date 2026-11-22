"""Absolute sample roll from calibrated detector rays and measured intensities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .detector_geometry import DetectorGeometry
from .roll_symmetry import fibre_symmetry, rod_symmetry


@dataclass(frozen=True)
class RollEstimate:
    deg: float
    error_deg: float


@dataclass(frozen=True)
class RollMeasurement:
    horizon: RollEstimate | None
    symmetry: RollEstimate | None
    deg: float | None
    error_deg: float | None
    method: str
    symmetry_feature: str = "fibre rings"

    @property
    def significant(self) -> bool:
        return self.deg is not None and abs(self.deg) >= 3 * self.error_deg

    def as_dict(self) -> dict:
        return {**asdict(self), "significant": self.significant}


def _horizon(data, valid, g):
    if min(data.shape) < 32:
        return None
    # Restrict the search to the vicinity of the horizon, allowing displacement of the
    # shadow or Yoneda edge relative to the geometric horizon. Each strip gets one vote.
    angles = np.linspace(-8., 8., 161)
    a, b, c = np.asarray([g.with_sample_roll(angle).horizon_plane() for angle in angles]).T
    zero = len(angles)//2
    cx, cy = g.beam_center_x_px, g.beam_center_y_px
    px, py = g.pixel_size_x_m, g.pixel_size_y_m
    if abs(a[zero]) > abs(b[zero]):  # steep detector trace: take row strips
        a, b = b, a
        cx, cy, px, py = cy, cx, py, px
        data, valid = data.T, valid.T
    if np.any(b == 0) or np.max(np.abs(a/b)) > 2:
        return None
    slopes, crossing = -a/b, cy + c/b
    rows, cols = data.shape
    widths = max(10, int(round(rows * .09)))
    step = max(1, rows // 800)
    offsets = np.arange(-widths, widths + 1, step)
    # Profiles follow the strongest downward intensity change on either side of the beam.
    up_to_down = np.sign(b[zero])  # intensity rises toward the illuminated side
    strip_x, scores = [], []
    blur = max(1.3, rows / 200)
    nstrip = min(40, max(12, cols // 12))
    for columns in np.array_split(np.arange(cols), nstrip):
        x = columns.mean() + .5 - cx
        pos = crossing + slopes * x
        lo = max(0, int(pos.min() - widths - 4*blur))
        hi = min(rows, int(pos.max() + widths + 4*blur + 1))
        if hi - lo < 12:
            continue
        block = data[lo:hi, columns[0]:columns[-1]+1]
        mask = valid[lo:hi, columns[0]:columns[-1]+1]
        count = mask.sum(axis=1)
        profile = np.sum(np.where(mask, block, 0.), axis=1, dtype=float) / count.clip(min=1)
        good = count >= max(2, columns.size // 2)
        if good.sum() < 12:
            continue
        # Winsorising rows prevents a single hot row/pixel carrying an entire strip.
        limits = np.percentile(profile[good], [1, 97])
        profile = np.interp(np.arange(len(good)), np.flatnonzero(good), profile[good].clip(*limits))
        grad = up_to_down * gaussian_filter1d(profile, blur, order=1)
        interior = good & (gaussian_filter1d(good.astype(float), blur) > .98)
        interior[:math.ceil(2*blur)] = interior[-math.ceil(2*blur):] = False
        if interior.sum() < 6 or np.max(grad[interior]) <= 0:
            continue
        noise = 1.4826 * np.median(np.abs(grad[interior] - np.median(grad[interior])))
        scale = max(3*noise, np.percentile(grad[interior], 98), 1e-12)
        score = np.where(interior, (grad / scale).clip(-1, 2), np.nan)
        score = np.interp((pos[:, None] + offsets - lo - .5).ravel(), np.arange(len(good)),
                          score, left=np.nan, right=np.nan).reshape(len(angles), -1)
        strip_x.append(x)
        scores.append(score)
    if len(scores) < 10:
        return None
    stack = np.asarray(scores)

    def best(subset):
        selected = stack[subset]
        available = np.isfinite(selected).sum(axis=0)
        score = np.nansum(selected, axis=0) / np.maximum(available, 1)
        score[available < max(8, .70*len(subset))] = -np.inf
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if score[i, j] < .45 or i in (0, len(angles)-1) or j in (0, len(offsets)-1):
            return None
        return float(angles[i])

    indices = np.arange(len(stack))
    result = best(indices)
    if result is None:
        return None
    blocks = np.array_split(indices, 5)
    estimates = [best(np.setdiff1d(indices, block)) for block in blocks]
    if any(item is None for item in estimates):
        return None
    # Deleting contiguous strips accounts for coherent curvature / displaced local edges.
    error = math.sqrt(4 * np.var(estimates))
    span = np.ptp(strip_x) * px / py
    return RollEstimate(result, max(.10, error, math.degrees(math.atan2(1., span))))


def measure_sample_roll(data, geometry: DetectorGeometry, *, valid=None, technique="giwaxs") -> RollMeasurement:
    """Estimate absolute ψ, independent of a previously applied sample roll.

    Positive ψ puts the surface normal toward lab right. Input pixels are the measured,
    unfilled frame. True values of `valid` identify usable pixels.
    """
    frame = np.asarray(data)
    if frame.ndim != 2:
        raise ValueError("Sample roll requires a two-dimensional frame.")
    keep = np.isfinite(frame)
    if valid is None and frame.dtype.kind in "iu":
        keep &= frame >= 0
    if valid is not None:
        if np.shape(valid) != frame.shape:
            raise ValueError("The validity mask must match the frame shape.")
        keep &= valid
    horizon = _horizon(frame, keep, geometry)
    gisaxs = str(technique).lower() == "gisaxs"
    feature = "specular rod" if gisaxs else "fibre rings"
    pair = (rod_symmetry if gisaxs else fibre_symmetry)(frame, keep, geometry)
    symmetry = RollEstimate(*pair) if pair is not None else None
    if horizon is None and symmetry is None:
        return RollMeasurement(None, None, None, None, "unavailable", feature)
    if horizon is None or symmetry is None:
        chosen = horizon or symmetry
        return RollMeasurement(horizon, symmetry, chosen.deg, chosen.error_deg, "horizon" if horizon else "symmetry",
                               feature)
    if abs(horizon.deg - symmetry.deg) > 2 * math.hypot(horizon.error_deg, symmetry.error_deg):
        return RollMeasurement(horizon, symmetry, None, None, "disagreement", feature)
    weights = np.array([horizon.error_deg, symmetry.error_deg]) ** -2
    value = float(np.average([horizon.deg, symmetry.deg], weights=weights))
    # Both estimators see the same frame, and sometimes the same shadow edge on rings.
    # Retain the best individual error and allow for the observed method disagreement.
    error = max(min(horizon.error_deg, symmetry.error_deg), abs(horizon.deg - symmetry.deg)/2)
    return RollMeasurement(horizon, symmetry, value, error, "combined", feature)
