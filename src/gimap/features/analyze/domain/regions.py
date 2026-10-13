"""Cut regions of a GIWAXS pattern: a q range × a χ range, each cut along q and along χ.

A region is what a person draws on the unwrapped (cake) view: q from ``q_range`` (the whole
measured range when ``None``), χ from ``chi_range`` in the reduction convention (0° along the
surface normal, ±90° in the sample plane). With ``both_sides`` the χ range is taken on |χ|, so
the region covers both halves of the pattern (the usual case: GIWAXS is symmetric in ±q∥) and
its I(χ) is folded onto |χ|.

Each region gives two curves: ``<key>`` = I(q) averaged over the region's χ, limited to its q
range, and ``<key>_chi`` = I(χ) averaged over its q range, limited to its χ range. Pixels below
the sample horizon never count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .binning import BinnedMean, counts_and_scale, merge_sparse_bins
from .models import Curve

CHI_STEP_DEG = 1.0
MIN_CHI_BINS = 10
REGION_KEY = "region"
"""Curve keys of user regions: ``region1``, ``region1_chi``, ``region2`` …"""


@dataclass(frozen=True)
class CutRegion:
    name: str
    q_range: Optional[tuple[float, float]] = None
    chi_range: tuple[float, float] = (0.0, 90.0)
    both_sides: bool = True

    def __post_init__(self) -> None:
        low, high = sorted(float(value) for value in self.chi_range)
        limit = (0.0, 90.0) if self.both_sides else (-90.0, 90.0)
        low, high = max(limit[0], low), min(limit[1], high)
        if high <= low:
            raise ValueError(f"Region {self.name!r}: the χ range is empty.")
        object.__setattr__(self, "chi_range", (low, high))
        if self.q_range is not None:
            q_low, q_high = sorted(float(value) for value in self.q_range)
            if q_high <= q_low or q_high <= 0:
                raise ValueError(f"Region {self.name!r}: the q range is empty.")
            object.__setattr__(self, "q_range", (max(0.0, q_low), q_high))


def region_key(index: int) -> str:
    return f"{REGION_KEY}{index + 1}"


def _chi(maps, both_sides: bool) -> np.ndarray:
    return np.abs(maps.chi_deg) if both_sides else maps.chi_deg


def region_mask(q_range, chi_range, both_sides: bool, maps, usable: np.ndarray) -> np.ndarray:
    """Pixels of a region (``q_range`` ``None``: every q)."""
    chi = _chi(maps, both_sides)
    inside = usable & (chi >= chi_range[0]) & (chi <= chi_range[1])
    if q_range is not None:
        inside &= (maps.q >= q_range[0]) & (maps.q <= q_range[1])
    return inside


def _profile(
    key, title, x_label, x_values, values, mask, *, low, high, bins, region, merge: bool = True, counts: bool = True,
) -> Curve:
    accumulator = BinnedMean.linear(low, high, bins)
    poisson, scale = counts_and_scale(counts, mask)
    accumulator.add(x_values[mask], values[mask], scale)
    result = accumulator.result(counts=poisson)
    x, mean, sigma, pixels = merge_sparse_bins(*result) if merge else result
    return Curve(key, title, x, mean, sigma, pixels, x_label, region=region)


def region_curves(
    region: CutRegion, key: str, maps, values: np.ndarray, usable: np.ndarray, *,
    q_low: float, q_high: float, bins: int, counts: bool = True,
) -> tuple[Curve, Curve]:
    """I(q) and I(χ) of one region (see the module docstring)."""
    low, high = region.q_range or (q_low, q_high)
    low, high = max(low, q_low), min(high, q_high)
    if high <= low:
        high = low + 1e-9
    chi_low, chi_high = region.chi_range
    inside = region_mask((low, high), region.chi_range, region.both_sides, maps, usable)
    record = {
        "cut_region": region.name, "q": (low, high), "chi_deg": (chi_low, chi_high), "both_sides": region.both_sides,
    }
    fraction = (high - low) / max(q_high - q_low, 1e-12)
    q_bins = max(20, int(round(bins * fraction)))
    chi_label = "|χ| (°)" if region.both_sides else "χ (°)"
    chi_text = f"|χ| {chi_low:g}–{chi_high:g}°" if region.both_sides else f"χ {chi_low:g}–{chi_high:g}°"
    along_q = _profile(
        key, f"{region.name}: I(q), {chi_text}", "q (Å⁻¹)", maps.q, values, inside,
        low=low, high=high, bins=q_bins, region=record, counts=counts,
    )
    # One χ bin no narrower than a pixel at this q (a ring near the beam spans few pixels per degree).
    pixel_deg = np.degrees(2.0 * getattr(maps, "q_step", 0.0) / max(0.5 * (low + high), 1e-9))
    chi_step = max(CHI_STEP_DEG, float(pixel_deg))
    chi_bins = max(MIN_CHI_BINS, int(round((chi_high - chi_low) / chi_step)))
    along_chi = _profile(
        f"{key}_chi", f"{region.name}: I(χ), q {low:.3f}–{high:.3f} Å⁻¹", chi_label, _chi(maps, region.both_sides),
        values, inside, low=chi_low, high=chi_high, bins=chi_bins, region=record, merge=False, counts=counts,
    )
    return along_q, along_chi


def window_profile(maps, values, usable, q_range, chi_range, both_sides: bool, *, counts: bool = True):
    """``(q, I, σ)`` of a region's χ range over ``q_range`` at the detector's own q step (to fit a peak:
    the window can be wider than the region, so the background on both sides of the peak is in it)."""
    low, high = float(q_range[0]), float(q_range[1])
    step = float(getattr(maps, "q_step", 0.0)) or (high - low) / 100.0
    inside = region_mask((low, high), chi_range, both_sides, maps, usable)
    accumulator = BinnedMean.linear(low, high, max(10, int(round((high - low) / step))))
    poisson, scale = counts_and_scale(counts, inside)
    accumulator.add(maps.q[inside], values[inside], scale)
    x, mean, sigma, _pixels = accumulator.result(counts=poisson)
    return x, mean, sigma


def region_outline(q_range: tuple[float, float], chi_range: tuple[float, float], *, points: int = 64):
    """``(q∥, qz)`` of a region's border in the q∥–qz plane (one side, χ as given).

    q∥ = q sin χ, qz = q cos χ: two arcs at the q limits joined by the two radial lines.
    """
    (q0, q1), (c0, c1) = q_range, chi_range
    chi = np.radians(np.linspace(c0, c1, points))
    outer, inner = q1 * np.sin(chi), q1 * np.cos(chi)
    back = chi[::-1]
    x = np.concatenate([outer, q0 * np.sin(back), outer[:1]])
    z = np.concatenate([inner, q0 * np.cos(back), inner[:1]])
    return x, z


def default_region_near(q_center: float, *, name: str, width: Optional[float] = None) -> CutRegion:
    """A full-χ ring region around ``q_center`` (±1.5 % of q, at least ±0.01 Å⁻¹)."""
    half = width if width is not None else max(0.01, 0.015 * float(q_center))
    return CutRegion(name, (max(0.0, q_center - half), q_center + half), (0.0, 90.0), True)


__all__ = [
    "CutRegion", "REGION_KEY", "default_region_near", "region_curves", "region_key",
    "region_mask", "region_outline", "window_profile",
]
