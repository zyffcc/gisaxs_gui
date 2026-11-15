"""Features of a grazing-incidence image that are not powder rings: the specular rod and the Yoneda band.

A calibrant measured at grazing incidence (AgBH on a wafer) shows, besides the arcs of its rings, the specular rod -
a vertical streak through the direct beam - and the Yoneda band - a horizontal band above the sample horizon. Both
are bright. The rod crosses every annulus, so the brightest pixels near a ring include it and pull a ring fit; the
band, around the direct beam, is a strong radial peak at its height: AgBH fits took it for the first order and gave
half the distance. The rod is found from the column medians and left out of ring finding and fitting
(``specular_rod_columns``, ``without_columns``); a radial peak whose maxima follow a horizontal line rather than its
circle is not a ring (``follows_horizontal_line``). Below the sample horizon the wafer shadows the rings: the rings
are looked for in the lit half of the image only (``LitHalf``), since the dark half holds an azimuthal percentile down
and loses a ring as soon as the centre is a pixel off. A transmission image has none of these and is not changed. A
broad band that the band test does not resolve is caught by the harmonic check of AgBH (``peak_matcher``).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

from .preprocessing import AnalysisImage

ROD_MIN_EXCESS = 0.3
"""Least excess of a column's median over its neighbours' (log signal after background) for a specular rod."""
ROD_MIN_ROWS = 0.25
"""Least share of usable pixels in a column for its median to count."""
ROD_GROW_PX = 3
BAND_ARC_DEG = (15.0, 45.0)
"""Where a ring and a horizontal line are compared: from 15 to 45 degrees off the vertical, on both sides."""
BAND_LINE_TOLERANCE = 0.05


def specular_rod_columns(analysis: AnalysisImage) -> Optional[tuple[int, int]]:
    """First and last column of the specular rod, or None (no rod: a transmission image)."""
    usable = analysis.valid.mean(axis=0) >= ROD_MIN_ROWS
    if not np.any(usable):
        return None
    signal = np.where(analysis.valid, analysis.signal, np.nan)
    with warnings.catch_warnings():                       # columns without usable pixels: their median is not used
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.where(usable, np.nan_to_num(np.nanmedian(signal, axis=0)), 0.0)
    excess = level - median_filter(level, size=41, mode="nearest")
    peak = int(np.argmax(excess))
    if excess[peak] < ROD_MIN_EXCESS:
        return None
    left = right = peak
    while left > 0 and excess[left - 1] > 0.3 * excess[peak] and peak - left < 40:
        left -= 1
    while right < excess.size - 1 and excess[right + 1] > 0.3 * excess[peak] and right - peak < 40:
        right += 1
    return max(0, left - ROD_GROW_PX), min(excess.size - 1, right + ROD_GROW_PX)


def without_columns(analysis: AnalysisImage, columns: Optional[tuple[int, int]]) -> AnalysisImage:
    """The image with these columns left out (no longer valid, no signal); the image itself when None."""
    if columns is None:
        return analysis
    valid = analysis.valid.copy()
    signal = analysis.signal.copy()
    valid[:, columns[0]:columns[1] + 1] = False
    signal[:, columns[0]:columns[1] + 1] = 0.0
    return replace(analysis, signal=signal, valid=valid, valid_fraction=float(np.mean(valid)))


class LitHalf:
    """The half of a grazing-incidence image, above or below the beam row, that the sample does not shadow: the one
    with the higher mean signal (row sums are taken once, a centre costs nothing)."""

    def __init__(self, analysis: AnalysisImage):
        valid = analysis.valid
        self._sums = np.concatenate([[0.0], np.cumsum(np.where(valid, analysis.signal, 0.0).sum(axis=1))])
        self._counts = np.concatenate([[0.0], np.cumsum(valid.sum(axis=1))])
        self._rows = np.arange(valid.shape[0])[:, None]
        self._valid = valid

    def _mean(self, start: int, stop: int) -> float:
        count = self._counts[stop] - self._counts[start]
        return float((self._sums[stop] - self._sums[start]) / count) if count > 0 else -np.inf

    def valid(self, center_y_px: float) -> np.ndarray:
        """The usable pixels of the lit half for a beam at this row."""
        row = int(np.clip(round(center_y_px), 0, self._valid.shape[0]))
        if self._mean(0, row) >= self._mean(row, self._valid.shape[0]):
            return self._valid & (self._rows < row)
        return self._valid & (self._rows >= row)


def follows_horizontal_line(analysis: AnalysisImage, center_x_px: float, center_y_px: float, radius_px: float) -> bool:
    """True when a radial peak above the beam is a horizontal band (the Yoneda band) rather than a ring: along rays
    15 to 45 degrees off the vertical, a ring keeps its maximum at its radius, a horizontal line at the peak's height
    moves it out to radius / cos(angle) (41 % farther at 45 degrees). Only a clear case counts: the line explains the
    maxima within 5 % and at most half as far off as the circle does."""
    psi = np.radians(np.r_[np.linspace(*BAND_ARC_DEG, 7), -np.linspace(*BAND_ARC_DEG, 7)])
    radii = np.linspace(0.8 * radius_px, 1.55 * radius_px, 200)
    image = np.where(analysis.valid, analysis.signal, np.nan)
    found = []
    for angle in psi:
        xs, ys = center_x_px + radii * np.sin(angle), center_y_px - radii * np.cos(angle)
        values = map_coordinates(image, [ys, xs], order=1, cval=np.nan)
        if np.isfinite(values).sum() < 50:
            continue
        values = gaussian_filter1d(np.nan_to_num(values), 2.0)
        found.append((abs(angle), radii[int(np.argmax(values))]))
    if len(found) < 6:
        return False
    angle, at = np.array(found).T
    circle = float(np.median(np.abs(at / radius_px - 1.0)))
    line = float(np.median(np.abs(at * np.cos(angle) / radius_px - 1.0)))
    return line < BAND_LINE_TOLERANCE and line < 0.5 * circle


__all__ = ["LitHalf", "follows_horizontal_line", "specular_rod_columns", "without_columns"]
