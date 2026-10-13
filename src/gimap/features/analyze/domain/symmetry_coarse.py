"""Locate a horizontal mirror axis without trusting the saved beam centre.

The coarse scan compares measured column pairs over the available detector
width.  Requiring both coverage and substantial profile contrast prevents a
flat tail from looking more symmetric than the scattering itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.signal import find_peaks

MIN_PAIRED_SAMPLES = 40
MIN_PAIR_DISTANCE_PX = 1.0
"""Left/right pairs use distinct columns, including measured near-centre structure."""
MIN_COVERAGE = 0.60
MIN_CONTRAST_SUPPORT = 0.50
MAX_NORMALIZED_LOSS = 0.25


def coarse_center(profile: np.ndarray, cancelled: Callable[[], bool] | None = None) -> float:
    """Return a coarse symmetry axis in column-index coordinates.

    ``profile`` contains raw band intensities, with NaN at unmeasured columns.
    Candidate axes span the detector rather than a neighbourhood of the
    initial geometry.  Integer and half-integer candidates pair stored column
    values directly, so masked gaps are never bridged by interpolation.

    The score is mean Huber loss of the mirrored asinh profile, divided by
    paired-profile variance.  At least half of the prominent measured
    contrast must participate: matching featureless tails is insufficient.
    Similar, separately supported minima are ambiguous and raise
    ``ValueError`` rather than selecting a repeated scattering lobe.
    """
    values = np.asarray(profile, dtype=np.float64)
    if cancelled is not None and cancelled():
        raise ValueError("Centre search cancelled.")
    if values.ndim != 1:
        raise ValueError("The horizontal cut profile must be one-dimensional.")
    measured = values[np.isfinite(values)]
    if measured.size < 2 * MIN_PAIRED_SAMPLES:
        raise ValueError("Too few measured pixels pair up across the beam centre (detector gaps).")
    positive = measured[measured > 0]
    scale = max(float(np.percentile(positive, 25)), 1e-12) if positive.size else 1.0
    transformed = np.arcsinh(values / scale)
    transformed[~np.isfinite(values)] = np.nan
    measured = transformed[np.isfinite(transformed)]
    contrast = float(np.var(measured))
    if contrast <= 1e-16:
        raise ValueError("The horizontal cut has too little structure to find a symmetry axis.")
    baseline = float(np.median(measured))
    energies = []
    for polarity in (1.0, -1.0):
        deviation = np.maximum(polarity * (transformed - baseline), 0.0)
        nonzero = deviation[np.isfinite(deviation) & (deviation > 0)]
        if not nonzero.size:
            continue
        # Compare the measured neighbours in order for extrema proposals.
        # Filling gaps with an artificial minimum would create false peaks
        # at their edges; no intensity is synthesized inside a gap.
        finite_columns = np.flatnonzero(np.isfinite(deviation))
        peaks, _ = find_peaks(deviation[finite_columns])
        peaks = finite_columns[peaks]
        peaks = peaks[(peaks >= MIN_PAIRED_SAMPLES + 3) & (peaks <= values.size - MIN_PAIRED_SAMPLES - 4)]
        if not peaks.size:
            continue
        maximum = float(np.max(deviation[peaks]))
        prominence = float(np.percentile(nonzero, 75))
        energy = np.square(np.maximum(deviation - prominence, 0.0))
        if float(np.nansum(energy)) <= 1e-16:
            energy = np.square(deviation)
        energies.append((maximum, energy))
    # Seed from the strongest interior peak or dip.  A monotonic tail whose
    # extreme is at the frame boundary cannot establish a mirror axis.
    # Polarity is relative to the median, independent of stored value sign.
    if not energies:
        raise ValueError("Too few structured measured pixels pair up to locate a symmetry axis.")
    strongest = max(energies, key=lambda item: item[0])
    energy = np.asarray([strongest[1]])
    total_energy = np.nansum(energy, axis=1)
    last = values.size - 1
    # Half-pixel axes at ordinary detector widths; at most about 4,000 global
    # candidates on larger frames.  The subsequent local search refines them.
    step = max(1, int(math.ceil(values.size / 2000)))
    candidates: list[tuple[float, int, float, float]] = []
    centers = range(2 * (MIN_PAIRED_SAMPLES + 3), 2 * (last - MIN_PAIRED_SAMPLES - 3) + 1, step)
    for position, doubled_center in enumerate(centers):
        if position % 32 == 0 and cancelled is not None and cancelled():
            raise ValueError("Centre search cancelled.")
        center = 0.5 * doubled_center
        first = max(0, doubled_center - last)
        stop = int(math.floor(center - MIN_PAIR_DISTANCE_PX)) + 1
        left_columns = np.arange(first, stop)
        right_columns = doubled_center - left_columns
        left = transformed[left_columns]
        right = transformed[right_columns]
        keep = np.isfinite(left) & np.isfinite(right)
        count = int(keep.sum())
        if count < max(MIN_PAIRED_SAMPLES, int(math.ceil(MIN_COVERAGE * left.size))):
            continue
        captured = float(np.max(np.sum(energy[:, left_columns[keep]] + energy[:, right_columns[keep]], axis=1)
                                / total_energy))
        if captured < MIN_CONTRAST_SUPPORT:
            continue
        left, right = left[keep], right[keep]
        paired = np.concatenate((left, right))
        paired_contrast = float(np.var(paired))
        if paired_contrast < 0.1 * contrast:
            continue
        residual = np.abs(left - right)
        loss = np.where(residual <= 1.0, 0.5 * residual**2, residual - 0.5)
        score = float(np.mean(loss)) / paired_contrast
        candidates.append((score, count, center, captured))
    if not candidates:
        raise ValueError("Too few structured measured pixels pair up to locate a symmetry axis.")
    candidates.sort()
    best_score = candidates[0][0]
    if best_score > MAX_NORMALIZED_LOSS:
        raise ValueError("The horizontal cut is not sufficiently mirror-symmetric to locate a centre.")
    # A subpixel axis does not fall exactly on the coarse grid.  Small score
    # differences therefore cannot establish that a repeated wave is the
    # physical axis; require prominent contrast and broad measured support.
    tolerance = max(0.005, 0.01 * best_score)
    contenders = [candidate for candidate in candidates if candidate[0] <= best_score + tolerance]
    # Repeated waves may have many near-perfect mirror axes.  Prefer broad
    # measured support, but require a unique supported neighbourhood.
    # A broad smooth peak has one wide minimum, not many ambiguous axes.
    # Compare separated neighbourhoods, keeping neighbouring trials together.
    groups: list[list[tuple[float, int, float, float]]] = []
    for candidate in sorted(contenders, key=lambda item: item[2]):
        if not groups or candidate[2] - groups[-1][-1][2] > max(3.0, 2.0 * step):
            groups.append([])
        groups[-1].append(candidate)
    representatives = [min(group) for group in groups]
    support = max(representatives, key=lambda candidate: (candidate[3], candidate[1]))
    supported = [candidate for candidate in representatives if candidate[3] >= 0.95 * support[3]
                 and candidate[1] >= 0.95 * support[1]]
    best = min(supported)
    if any(other is not best and other[1] >= 0.9 * best[1] and other[3] >= 0.8 * best[3]
           for other in representatives):
        raise ValueError("The horizontal cut has several similarly supported symmetry axes.")
    return float(best[2])


__all__ = ["coarse_center"]
