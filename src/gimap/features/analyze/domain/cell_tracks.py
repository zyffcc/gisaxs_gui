"""Follow the indexed rings of one radial I(q) through native (unresampled) frame profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crystal.refinement import CellPeak
from .peak_fit import GAUSSIAN, fit_peak
from .series_map import SeriesMap, track_peak


@dataclass(frozen=True)
class RingTrack:
    phase: str
    hkl: tuple[int, int, int]
    q: float
    fwhm: float
    low: float
    high: float


def ring_tracks(results) -> tuple[RingTrack, ...]:
    """One non-overlapping search interval per seed, bounded by neighbouring measured ring centres."""
    seeds = [(result["phase"], seed) for result in results for seed in result.get("seeds", ())]
    centres = sorted({seed["q"] for _, seed in seeds} | {q for result in results for q in result.get("unassigned_q", ())
                                                          if q is not None and np.isfinite(q)} | {
        item["q"] for result in results for item in result.get("excluded", ())
        if item.get("q") is not None and np.isfinite(item["q"])
    })
    tracks = []
    for phase, seed in seeds:
        q = seed["q"]
        index = np.searchsorted(centres, q)
        low = 0.5 * (centres[index - 1] + q) if index > 0 else 0.0
        high = 0.5 * (centres[index + 1] + q) if index + 1 < len(centres) else np.inf
        width = float(seed.get("fwhm") or 0)
        tracks.append(RingTrack(phase, tuple(seed["hkl"]), q, width if np.isfinite(width) else 0., low, high))
    return tuple(tracks)


def track_frame(x, intensity, sigma, tracks, previous, check_cancel):
    """The series-map centroid initializes a Gaussian + linear background fitted to each window.

    The search follows the last successful centre by at most 2.5 indexed FWHM (at least five
    sample spacings to either side). A fit whose half-maximum is cut by the window or whose
    centre lies in a gap wider than the fitted FWHM cannot supply a position.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(intensity, dtype=float)
    series = SeriesMap(x, y[None, :], ("",), "q (Å⁻¹)", "radial")
    steps = np.diff(x)
    spacing = np.median(steps[steps > 0]) if np.any(steps > 0) else 0.0
    peaks, fitted = {}, []
    for ring in tracks:
        check_cancel()
        centre = previous.get(ring, ring.q)
        half = max(2.5 * ring.fwhm, 5 * spacing)
        low, high = max(ring.low, centre - half), min(ring.high, centre + half)
        item = {"phase": ring.phase, "hkl": list(ring.hkl), "window": [float(low), float(high)]}
        fitted.append(item)
        centroid = track_peak(series, low, high).position[0]
        if not np.isfinite(centroid):
            item["reason"] = "No peak above the local background."
            continue
        fit = fit_peak(x, y, sigma, (low, high), profile=GAUSSIAN, start={"center": centroid})
        if not fit.ok:
            item["reason"] = fit.message
            continue
        measured = fit.x
        index = int(np.searchsorted(measured, fit.center))
        if (not np.isfinite(fit.center_err) or fit.center_err <= 0
                or not 0 < index < len(measured)
                or min(fit.center - measured[0], measured[-1] - fit.center) < 0.5 * fit.fwhm
                or measured[index] - measured[index - 1] > fit.fwhm):
            item["reason"] = "Peak centre or width is unresolved in the measured bins."
            continue
        previous[ring] = fit.center
        item.update(q=fit.center, sigma=fit.center_err, fwhm=fit.fwhm)
        peaks.setdefault(ring.phase, []).append(CellPeak(ring.hkl, fit.center, fit.center_err))
    return peaks, fitted
