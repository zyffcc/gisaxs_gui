"""A beam-centre proposal polished to the pixel before rings are looked for.

Arc voting (``center_estimator``) places a centre in a histogram cell several pixels wide. With an offset centre, a
narrow ring spreads over several radii and can disappear from the radial profile. For both transmission and
grazing-incidence frames, a coarse-to-fine grid search around the leading arc proposals maximises the sharpness
of the bright pixels' radial profile. The engine tests the polished centres alongside the original proposals.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .center_estimator import CenterProposal
from .preprocessing import AnalysisImage

BRIGHT_SHARE = 0.25
"""Share of the usable pixels, the brightest, whose radial profile is made sharp."""
STEPS_PX = (4.0, 2.0, 1.0, 0.5)
"""Grid steps of the search, coarse to fine; each searches two steps on every side of the best so far."""
POLISHED_METHODS = ("concentric arc gradients",)
LIMIT = 4
MIN_RADIUS_PX = 40


def _sharpness(analysis: AnalysisImage):
    valid = analysis.valid
    if not np.any(valid):
        return None
    cut = np.quantile(analysis.signal[valid], 1.0 - BRIGHT_SHARE)
    rows, cols = np.nonzero(valid & (analysis.signal >= cut))
    if rows.size < 100:
        return None
    weights = analysis.signal[rows, cols].astype(np.float64)
    rows, cols = rows.astype(np.float64), cols.astype(np.float64)

    def score(x: float, y: float) -> float:
        bins = np.hypot(cols - x, rows - y).astype(np.int64)
        profile = np.bincount(bins, weights=weights)
        detail = profile - gaussian_filter1d(profile, 6.0)
        detail[:MIN_RADIUS_PX] = 0.0
        return float(np.sum(np.clip(detail, 0.0, None) ** 2))

    return score


def polished_centers(analysis: AnalysisImage, centers: list[CenterProposal]) -> list[CenterProposal]:
    """Polished copies of the leading arc proposals (none that lands on a proposal already there)."""
    score = _sharpness(analysis)
    if score is None:
        return []
    added: list[CenterProposal] = []
    for center in [item for item in centers if item.method in POLISHED_METHODS][:LIMIT]:
        best = (score(center.x_px, center.y_px), center.x_px, center.y_px)
        for step in STEPS_PX:
            x0, y0 = best[1], best[2]
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    if dx or dy:
                        value = score(x0 + dx * step, y0 + dy * step)
                        if value > best[0]:
                            best = (value, x0 + dx * step, y0 + dy * step)
        x, y = best[1], best[2]
        if all(np.hypot(x - item.x_px, y - item.y_px) >= 1.0 for item in centers + added):
            added.append(CenterProposal(x, y, center.score, center.method, center.support))
    return added


__all__ = ["polished_centers"]
