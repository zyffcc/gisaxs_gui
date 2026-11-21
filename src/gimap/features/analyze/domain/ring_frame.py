"""Does a frame show complete, sharp rings about the direct beam?

A powder calibrant measured in transmission (silver behenate, LaB6) gives sharp rings concentric with the beam that
run through every direction the detector covers, also below the beam. A grazing-incidence frame does not: the sample
shadows what lies below its horizon, and its features are rods, a Yoneda band and broad maxima. The automatic
analysis uses this to tell a calibrant frame from a GISAXS frame when no incidence angle is known; nothing is
reduced from it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SECTORS = 24
"""Azimuthal sectors (15° each) a ring is looked for in."""
MIN_RINGS = 3
"""Complete rings a frame must show to count as a ring frame."""
MIN_PRESENT = 0.8
"""Share of the sectors the detector covers at a ring's radius in which the ring must stand out."""
MIN_COVERED_DEG = 90.0
"""Azimuth the detector must cover at a ring's radius: a short arc is no ring."""
MAX_WIDTH_PX = 25.0
"""Widest ring (FWHM, pixels) that still counts as sharp."""
_BACKGROUND_WINDOW = 61
_MIN_PIXELS = 8


@dataclass(frozen=True)
class RingFrame:
    rings: tuple[dict, ...]
    """Every sharp maximum of the azimuthal mean: ``radius_px``, ``covered_deg``, ``present`` (share of the covered
    sectors that show it), ``complete``."""

    @property
    def complete_rings(self) -> int:
        return sum(1 for ring in self.rings if ring["complete"])

    @property
    def is_ring_frame(self) -> bool:
        return self.complete_rings >= MIN_RINGS

    def to_dict(self) -> dict:
        return {"complete_rings": self.complete_rings, "is_ring_frame": self.is_ring_frame,
                "rings": [dict(ring) for ring in self.rings]}


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    stacked = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.nanmedian(stacked, axis=1)


def rings_about_beam(image: np.ndarray, valid: np.ndarray, center_x: float, center_y: float) -> RingFrame:
    """The sharp rings concentric with ``(center_x, center_y)`` (canonical pixels) and how completely each runs
    through the directions the detector covers."""
    image, valid = np.asarray(image, dtype=np.float64), np.asarray(valid, dtype=bool)
    rows, columns = np.indices(image.shape, dtype=np.float32)
    dx, dy = columns + 0.5 - float(center_x), rows + 0.5 - float(center_y)
    radius = np.hypot(dx, dy)
    sector = np.floor((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * SECTORS).astype(np.int64) % SECTORS
    ring = radius.astype(np.int64)
    size = int(ring.max()) + 1
    flat = (sector * size + ring)[valid]
    counts = np.bincount(flat, minlength=SECTORS * size).reshape(SECTORS, size).astype(np.float64)
    sums = np.bincount(flat, weights=image[valid], minlength=SECTORS * size).reshape(SECTORS, size)
    total = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(total >= 3 * _MIN_PIXELS, sums.sum(axis=0) / total, np.nan)
        by_sector = np.where(counts >= _MIN_PIXELS, sums / counts, np.nan)
    if np.isfinite(profile).sum() < 3 * _BACKGROUND_WINDOW:
        return RingFrame(())
    background = _rolling_median(profile, _BACKGROUND_WINDOW)
    excess = profile - background
    noise = 1.4826 * np.nanmedian(np.abs(excess - np.nanmedian(excess)))
    found = []
    for index in range(5, size - 5):
        height = excess[index]
        window = excess[index - 3:index + 4]
        if not np.isfinite(height) or not np.isfinite(window).all() or height < np.nanmax(window):
            continue
        if height < max(6.0 * noise, 0.3 * abs(background[index])):
            continue
        above = excess >= 0.5 * height  # full width at half maximum, in pixels
        left = index
        while left > 0 and above[left - 1]:
            left -= 1
        right = index
        while right < size - 1 and above[right + 1]:
            right += 1
        if right - left + 1 > MAX_WIDTH_PX:
            continue
        found.append(_ring(index, float(height), by_sector, size))
    return RingFrame(tuple(found))


def _ring(index: int, height: float, by_sector: np.ndarray, size: int) -> dict:
    """One ring: in how many of the sectors that hold its radius and both sides of it does it stand out?"""
    covered = present = 0
    for values in by_sector:
        peak = values[max(0, index - 2):index + 3]
        sides = np.r_[values[max(0, index - 14):max(0, index - 7)], values[index + 8:min(size, index + 15)]]
        if np.isfinite(peak).sum() < 3 or np.isfinite(sides).sum() < 6:
            continue
        covered += 1
        present += int(np.nanmax(peak) - np.nanmedian(sides) > 0.3 * height)
    covered_deg = covered * 360.0 / SECTORS
    share = present / covered if covered else 0.0
    return {"radius_px": float(index) + 0.5, "covered_deg": covered_deg, "present": share,
            "complete": bool(covered_deg >= MIN_COVERED_DEG and share >= MIN_PRESENT)}


__all__ = ["MIN_COVERED_DEG", "MIN_PRESENT", "MIN_RINGS", "RingFrame", "rings_about_beam"]
