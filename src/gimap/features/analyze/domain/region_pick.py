"""Put a cut region where the person clicked, snapped to the peak there.

A click is a point (q, χ). ``pick_region`` turns it into one of three region shapes:

* ``RING`` — the q window of the peak at the click, every χ: its I(χ) is the azimuthal profile
  of that ring (orientation), its I(q) the peak itself.
* ``SECTOR`` — every q, the χ window at the click: its I(q) is the profile along that direction.
* ``SPOT`` — both windows: a Bragg spot of an oriented film.

Snapping (``peak_near``): the profile around the click is binned at the detector's own step (q)
or 1° (χ); from the click, climb the 3-bin running mean to the nearest maximum. The background is
the straight line through the lowest point on either side within the search range, so a sloping
background or a neighbouring peak does not shift the result. The centre is the centroid of the
part above half maximum, the width the full width at half maximum. The window is centre ± FWHM
(the whole peak) and never extends past those lowest points into a neighbouring peak. A maximum
less than ``SIGNIFICANCE`` times the point-to-point noise above the background, with fewer than
``MIN_PEAK_POINTS`` bins above half height, not falling to half its height within the data on both sides, or centred
farther from the click than ``Q_REACH_FRACTION`` / ``CHI_REACH_DEG``, is not the peak clicked:
the window is then a fixed width around the click (``Pick.q_peak`` / ``chi_peak`` are ``None``).

Pixels are used directly (not the cake grid), so the result is as precise as the data.
Regions are folded onto |χ| (``both_sides``): GIWAXS patterns are symmetric in ±q∥.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .binning import BinnedMean
from .regions import CutRegion

RING, SECTOR, SPOT = "ring", "sector", "spot"
PICK_KINDS = (RING, SECTOR, SPOT)
SIGNIFICANCE = 4.0
"""A peak must stand this many times the point-to-point noise above its background."""
MIN_PEAK_POINTS = 2
"""…and have at least this many (unsmoothed) bins above half its height: one bright bin is edge or hot pixels."""
PROFILE_CHI_DEG = 10.0
"""Half width of the χ band around a click whose I(q) locates the q peak."""
SECTOR_HALF_DEG = 5.0
"""Half width of a sector (and of a spot) in χ when no azimuthal peak stands out at the click."""
MIN_CHI_HALF_DEG, MAX_CHI_HALF_DEG = 1.0, 30.0
CHI_SEARCH_DEG = 30.0
CHI_REACH_DEG = 6.0
"""A χ peak whose centre is farther than this from the click is not the one clicked."""
Q_REACH_FRACTION = 0.02
"""Likewise for q, as a fraction of q (at least 6 detector q steps)."""
FALLBACK_Q_FRACTION = 0.015
"""Half width of the q window, as a fraction of q, when no peak stands out (at least 2 q steps)."""


@dataclass(frozen=True)
class Peak:
    center: float
    fwhm: float
    low: float
    """The window: centre ± FWHM, cut at the lowest points on either side."""
    high: float
    height: float
    """Above the background, in the unit of the intensity."""
    significance: float
    """``height`` in units of the point-to-point noise."""


@dataclass(frozen=True)
class Pick:
    region: CutRegion
    kind: str
    q_peak: Optional[Peak]
    chi_peak: Optional[Peak]
    q_clicked: float
    chi_clicked: float


def _running_mean(y: np.ndarray, width: int = 3) -> np.ndarray:
    if y.size < width:
        return y.copy()
    padded = np.pad(y, width // 2, mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _noise(y: np.ndarray) -> float:
    """Point-to-point scatter: a point against the mean of its two neighbours (robust)."""
    if y.size < 3:
        return float("inf")
    residual = y[1:-1] - 0.5 * (y[:-2] + y[2:])
    return float(np.median(np.abs(residual)) * 1.4826 / math.sqrt(1.5))


def _crossing(x: np.ndarray, net: np.ndarray, start: int, stop: int, half: float) -> Optional[float]:
    """Where ``net`` falls below ``half`` walking from ``start`` towards ``stop`` (interpolated), or ``None``."""
    step = 1 if stop > start else -1
    for index in range(start, stop, step):
        after = index + step
        if net[after] < half:
            span = net[index] - net[after]
            fraction = (net[index] - half) / span if span > 0 else 0.5
            return float(x[index] + fraction * (x[after] - x[index]))
    return None


def peak_near(x, y, x0: float, *, search: float, reach: Optional[float] = None) -> Optional[Peak]:
    """The peak of ``y(x)`` nearest ``x0`` within ``x0 ± search``, centred within ``reach`` of it."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x - x0) <= search)
    if keep.sum() < 7:
        return None
    order = np.argsort(x[keep], kind="stable")
    x, y = x[keep][order], y[keep][order]
    top = int(np.argmin(np.abs(x - x0)))
    if abs(x[top] - x0) > 2.0 * float(np.median(np.diff(x))):
        return None  # no data where the click is (a gap, or beyond the detector in this band)
    smooth = _running_mean(y)
    while True:  # climb to the nearest maximum
        left = smooth[top - 1] if top > 0 else -np.inf
        right = smooth[top + 1] if top < x.size - 1 else -np.inf
        if left > smooth[top] and left >= right:
            top -= 1
        elif right > smooth[top]:
            top += 1
        else:
            break
    if top in (0, x.size - 1):
        return None  # the data end here: a maximum at the edge of the coverage is not a peak
    left_low = int(np.argmin(smooth[: top + 1]))
    right_low = top + int(np.argmin(smooth[top:]))
    if right_low > left_low:
        slope = (smooth[right_low] - smooth[left_low]) / (x[right_low] - x[left_low])
    else:
        slope = 0.0
    background = smooth[left_low] + slope * (x - x[left_low])
    net = smooth - background
    height = float(net[top])
    noise = _noise(y)
    if height <= 0 or not np.isfinite(noise) or height < SIGNIFICANCE * max(noise, 1e-12):
        return None
    half = 0.5 * height
    low_cross = _crossing(x, net, top, left_low, half)
    high_cross = _crossing(x, net, top, right_low, half)
    if low_cross is None or high_cross is None:
        return None  # it does not fall to half its height on both sides: cut by the edge of the data, not a peak
    raw_above = (y - background)[left_low : right_low + 1] >= half
    if int(raw_above.sum()) < MIN_PEAK_POINTS:
        return None
    inside = (x >= low_cross) & (x <= high_cross) & (net > 0)
    centre = float((x[inside] * net[inside]).sum() / net[inside].sum()) if inside.any() else float(x[top])
    if reach is not None and abs(centre - x0) > reach:
        return None
    fwhm = max(high_cross - low_cross, float(np.median(np.diff(x))))
    low = max(centre - fwhm, float(x[left_low]))
    high = min(centre + fwhm, float(x[right_low]))
    return Peak(centre, fwhm, low, high, height, height / max(noise, 1e-12))


def _mirrored_chi(chi: np.ndarray, values: np.ndarray, both_sides: bool) -> tuple[np.ndarray, np.ndarray]:
    """On |χ|, a peak at 0° or 90° continues across the edge: mirror the profile there."""
    if not both_sides or chi.size == 0:
        return chi, values
    return (
        np.concatenate([-chi[::-1], chi, 180.0 - chi[::-1]]),
        np.concatenate([values[::-1], values, values[::-1]]),
    )


def _chi_of(maps, both_sides: bool) -> np.ndarray:
    return np.abs(maps.chi_deg) if both_sides else maps.chi_deg


def q_profile(maps, values, usable, q0: float, chi_range, both_sides: bool, search: float):
    """I(q) near ``q0`` of the pixels in ``chi_range``, one bin per q step of the detector."""
    step = float(getattr(maps, "q_step", 0.0)) or max(1e-4, 0.002 * q0)
    low, high = max(0.0, q0 - search), q0 + search
    chi = _chi_of(maps, both_sides)
    inside = usable & (maps.q >= low) & (maps.q <= high) & (chi >= chi_range[0]) & (chi <= chi_range[1])
    accumulator = BinnedMean.linear(low, high, max(8, int(round((high - low) / step))))
    accumulator.add(maps.q[inside], values[inside])
    x, mean, _sigma, _pixels = accumulator.result()
    return x, mean


def chi_profile(maps, values, usable, q_range, both_sides: bool):
    """I(χ) (on |χ| when folded) of the pixels in ``q_range``, bins no narrower than 1° or a pixel."""
    chi = _chi_of(maps, both_sides)
    low_limit = 0.0 if both_sides else -90.0
    inside = usable & (maps.q >= q_range[0]) & (maps.q <= q_range[1])
    q_mid = max(0.5 * (q_range[0] + q_range[1]), 1e-9)
    step = max(1.0, math.degrees(2.0 * float(getattr(maps, "q_step", 0.0)) / q_mid))
    accumulator = BinnedMean.linear(low_limit, 90.0, max(10, int(round((90.0 - low_limit) / step))))
    accumulator.add(chi[inside], values[inside])
    x, mean, _sigma, _pixels = accumulator.result()
    return x, mean


def snap_q(
    maps, values, usable, q0: float, chi_range, both_sides: bool, *, reach: float = 0.0,
) -> tuple[tuple[float, float], Optional[Peak]]:
    """The q window of the peak at ``q0`` (or a fixed width around ``q0`` when none stands out).

    ``reach``: how far from ``q0`` the peak may be (at least the default for a click).
    """
    step = float(getattr(maps, "q_step", 0.0)) or max(1e-4, 0.002 * q0)
    reach = max(float(reach), 6.0 * step, Q_REACH_FRACTION * q0)
    search = max(12.0 * step, 0.05 * q0, 2.0 * reach)
    x, y = q_profile(maps, values, usable, q0, chi_range, both_sides, search)
    peak = peak_near(x, y, q0, search=search, reach=reach)
    if peak is None:
        half = max(2.0 * step, FALLBACK_Q_FRACTION * q0)
        return (max(0.0, q0 - half), q0 + half), None
    low, high = min(peak.low, peak.center - step), max(peak.high, peak.center + step)
    return (max(0.0, low), high), peak


def snap_chi(
    maps, values, usable, chi0: float, q_range, both_sides: bool, *, reach: float = 0.0,
) -> tuple[tuple[float, float], Optional[Peak]]:
    """The χ window of the azimuthal peak at ``chi0`` in ``q_range`` (or ± ``SECTOR_HALF_DEG``)."""
    chi0 = abs(chi0) if both_sides else chi0
    limit = (0.0, 90.0) if both_sides else (-90.0, 90.0)
    x, y = chi_profile(maps, values, usable, q_range, both_sides)
    x, y = _mirrored_chi(x, y, both_sides)
    peak = peak_near(x, y, chi0, search=CHI_SEARCH_DEG, reach=max(CHI_REACH_DEG, float(reach)))
    if peak is None:
        half = SECTOR_HALF_DEG
        centre = chi0
    else:
        half = min(MAX_CHI_HALF_DEG, max(MIN_CHI_HALF_DEG, 0.5 * (peak.high - peak.low), 0.5 * peak.fwhm))
        centre = peak.center
    # At 0° or 90° the window is cut at the edge: on |χ| that already covers both sides of the peak.
    low, high = max(limit[0], centre - half), min(limit[1], centre + half)
    if high - low < 2.0 * MIN_CHI_HALF_DEG:
        low, high = max(limit[0], high - 2.0 * MIN_CHI_HALF_DEG), min(limit[1], low + 2.0 * MIN_CHI_HALF_DEG)
    return (low, high), peak


def pick_region(
    kind: str, q: float, chi: float, maps, values, usable, *, name: str, both_sides: bool = True, chi_band=None,
) -> Pick:
    """A region of ``kind`` at the clicked ``(q, χ)`` (see the module docstring).

    ``chi_band``: the χ range whose I(q) locates the q peak; by default ±``PROFILE_CHI_DEG`` around
    the click, then the whole ring when no peak stands out there (a click on a 1-D I(q) plot has no
    χ and passes the whole range).
    """
    if kind not in PICK_KINDS:
        raise ValueError(f"Unknown region shape {kind!r}.")
    q = float(q)
    chi = float(chi)
    chi_here = abs(chi) if both_sides else chi
    full_chi = (0.0, 90.0) if both_sides else (-90.0, 90.0)
    band = chi_band or (max(full_chi[0], chi_here - PROFILE_CHI_DEG), min(90.0, chi_here + PROFILE_CHI_DEG))
    q_range, q_peak = snap_q(maps, values, usable, q, band, both_sides)
    if q_peak is None and band != full_chi:  # nothing (or no data) around the click: the whole ring
        q_range, q_peak = snap_q(maps, values, usable, q, full_chi, both_sides)
    chi_range, chi_peak = full_chi, None
    if kind in (SECTOR, SPOT):
        chi_range, chi_peak = snap_chi(maps, values, usable, chi_here, q_range, both_sides)
    region = CutRegion(name, None if kind == SECTOR else q_range, chi_range, both_sides)
    return Pick(region, kind, q_peak if kind != SECTOR else None, chi_peak, q, chi)


def region_kind(region: CutRegion) -> str:
    """``ring`` (every χ), ``sector`` (every q) or ``spot`` (both limited)."""
    full = (0.0, 90.0) if region.both_sides else (-90.0, 90.0)
    every_chi = region.chi_range[0] <= full[0] + 1e-6 and region.chi_range[1] >= full[1] - 1e-6
    if region.q_range is None:
        return SECTOR
    return RING if every_chi else SPOT


def snap_region(region: CutRegion, maps, values, usable) -> Pick:
    """``region`` moved onto the peak nearest its centre, keeping its shape and name.

    The peak may lie up to 1.5 region widths from the centre (asked for, not a click)."""
    kind = region_kind(region)
    chi_mid = 0.5 * (region.chi_range[0] + region.chi_range[1])
    q_range, q_peak = region.q_range, None
    if region.q_range is not None:
        q_mid = 0.5 * (region.q_range[0] + region.q_range[1])
        reach = 1.5 * (region.q_range[1] - region.q_range[0])
        q_range, q_peak = snap_q(maps, values, usable, q_mid, region.chi_range, region.both_sides, reach=reach)
    chi_range, chi_peak = region.chi_range, None
    if kind in (SECTOR, SPOT):
        window = q_range if q_range is not None else (float(np.nanmin(maps.q)), float(np.nanmax(maps.q)))
        reach = 1.5 * (region.chi_range[1] - region.chi_range[0])
        chi_range, chi_peak = snap_chi(maps, values, usable, chi_mid, window, region.both_sides, reach=reach)
    q_mid = 0.5 * (q_range[0] + q_range[1]) if q_range is not None else float("nan")
    snapped = CutRegion(region.name, q_range, chi_range, region.both_sides)
    return Pick(snapped, kind, q_peak, chi_peak, q_mid, chi_mid)


__all__ = [
    "PICK_KINDS", "Peak", "Pick", "RING", "SECTOR", "SIGNIFICANCE", "SPOT", "chi_profile", "peak_near",
    "pick_region", "q_profile", "region_kind", "snap_chi", "snap_q", "snap_region",
]
