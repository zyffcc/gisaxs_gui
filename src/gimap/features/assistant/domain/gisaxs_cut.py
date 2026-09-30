"""The horizontal GISAXS cut I(qy): which halves to use, the curve to fit, the dominant spacing.

After the beam centre is on the symmetry axis the two halves (qy < 0, qy > 0) should agree.
They are averaged when they do; when one half is cut short by the detector edge, a gap or a
beam stop, the other one is used; when both are complete but disagree, both are kept (shown in
two colours) and the better-covered one is fitted — each choice with its reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MIN_COVERAGE = 0.6
"""A half must have points in this fraction of its own |qy| range (gaps, a beam-stop shadow)."""
MIN_REACH = 0.3
"""A half reaching less than this fraction of the other's largest |qy| is too short to use."""
MISMATCH_LIMIT = 0.15
"""Median |ln(I₋/I₊)| up to which the halves agree well enough to average (≈ 16 %)."""
SPECULAR_POINTS = 3
"""Points next to qy = 0 left out of comparisons (specular rod, beam stop)."""
MAX_FIT_POINTS = 1000


@dataclass(frozen=True)
class HalvesChoice:
    side: str
    """``mean``, ``negative``, ``positive`` or ``both_abs`` (both halves kept on |qy|)."""
    fit_side: str
    """The curve a fit uses: ``mean``, ``negative`` or ``positive``."""
    reason: str
    coverage: dict = field(default_factory=dict)
    """Share of the |qy| range with measured points, per half."""
    reach: dict = field(default_factory=dict)
    """Largest |qy| measured on each half (Å⁻¹)."""
    mismatch: Optional[float] = None


def _halves(q: np.ndarray, intensity: np.ndarray, pixels: Optional[np.ndarray]):
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    measured = np.isfinite(q) & np.isfinite(intensity) & (intensity > 0)
    if pixels is not None:
        measured &= np.asarray(pixels) > 0
    negative = measured & (q < 0)
    positive = measured & (q > 0)
    return q, intensity, negative, positive


def _coverage(values: np.ndarray, low: float, high: float, step: float) -> float:
    if high <= low or step <= 0:
        return 0.0
    edges = np.arange(low, high + step, step)
    counts, _ = np.histogram(values, bins=edges)
    return float((counts > 0).mean()) if counts.size else 0.0


def choose_halves(
    q: np.ndarray,
    intensity: np.ndarray,
    pixels: Optional[np.ndarray] = None,
    *,
    min_coverage: float = MIN_COVERAGE,
    mismatch_limit: float = MISMATCH_LIMIT,
) -> HalvesChoice:
    """Decide how to use the two halves of I(qy); ``q`` is signed qy (Å⁻¹)."""
    q, intensity, negative, positive = _halves(q, intensity, pixels)
    if not negative.any() and not positive.any():
        raise ValueError("The horizontal cut has no measured points.")
    if not negative.any() or not positive.any():
        only = "positive" if positive.any() else "negative"
        where = "qy > 0" if only == "positive" else "qy < 0"
        return HalvesChoice(only, only, f"Only the {where} half is on the detector.")
    abs_neg, abs_pos = np.sort(-q[negative]), np.sort(q[positive])
    step = float(np.median(np.diff(np.sort(np.abs(q[negative | positive]))))) or 1e-6
    step = max(step, 1e-9) * 1.5
    low = max(abs_neg[min(SPECULAR_POINTS, abs_neg.size - 1)], abs_pos[min(SPECULAR_POINTS, abs_pos.size - 1)])
    reach = {"negative": float(abs_neg.max()), "positive": float(abs_pos.max())}
    coverage = {
        "negative": _coverage(abs_neg, low, reach["negative"], step),
        "positive": _coverage(abs_pos, low, reach["positive"], step),
    }
    names = {"negative": "qy < 0", "positive": "qy > 0"}
    longer = max(reach, key=reach.get)
    better = max(coverage, key=lambda key: (coverage[key], reach[key]))
    for side in ("negative", "positive"):
        other = "negative" if side == "positive" else "positive"
        short = reach[side] < MIN_REACH * reach[other]
        gappy = coverage[side] < min_coverage and coverage[side] < coverage[other] - 0.1
        if short or gappy:
            why = (
                f"reaches only |qy| = {reach[side]:.3g} Å⁻¹ (the {names[other]} half {reach[other]:.3g} Å⁻¹)"
                if short else
                f"has points in only {coverage[side]:.0%} of its |qy| range (detector gaps or the beam-stop shadow; "
                f"the {names[other]} half: {coverage[other]:.0%})"
            )
            return HalvesChoice(other, other, f"The {names[side]} half {why}, so the {names[other]} half is used.", coverage, reach)
    common_high = min(reach.values())
    extend = (
        f"; beyond {common_high:.3g} Å⁻¹ the longer {names[longer]} half continues alone to {reach[longer]:.3g} Å⁻¹"
        if reach[longer] > common_high * 1.02 else ""
    )
    grid = abs_pos[(abs_pos >= low) & (abs_pos <= common_high)]
    order = np.argsort(-q[negative])
    mismatch = None
    if grid.size >= 5:
        left = np.interp(grid, -q[negative][order], intensity[negative][order])
        right = np.interp(grid, q[positive], intensity[positive])
        ratio = np.log(left / right)
        mismatch = float(np.median(np.abs(ratio[np.isfinite(ratio)]))) if np.isfinite(ratio).any() else None
    if mismatch is not None and mismatch > mismatch_limit:
        return HalvesChoice(
            "both_abs", better,
            f"The halves differ by {100 * (math.exp(mismatch) - 1):.0f} % (median) up to |qy| = {common_high:.3g} Å⁻¹ "
            "even after the symmetry correction (a shadow, an absorber or real in-plane anisotropy): both are kept "
            f"and the {names[better]} half, the better covered one, is fitted.",
            coverage, reach, mismatch,
        )
    agree = f"agree within {100 * (math.exp(mismatch) - 1):.0f} %" if mismatch is not None else "overlap too little to compare"
    return HalvesChoice(
        "mean", "mean",
        f"Both halves are usable (points in {coverage['negative']:.0%} and {coverage['positive']:.0%} of their "
        f"|qy| ranges) and {agree}: they are averaged where both exist, which halves the noise{extend}.",
        coverage, reach, mismatch,
    )


def fit_curve(
    q: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    pixels: Optional[np.ndarray],
    side: str,
    *,
    max_points: int = MAX_FIT_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(|qy|, I, σ, note) of the half (or the mean of both) a fit uses, at most ``max_points`` points.

    The mean interpolates the qy < 0 half onto the qy > 0 points they share (σ combined). More
    points than a fit accepts are merged in consecutive groups (weighted mean, σ combined).
    """
    q, intensity, negative, positive = _halves(q, intensity, pixels)
    sigma = np.asarray(sigma, dtype=float)
    notes = []
    if side == "mean":
        reach_neg, reach_pos = float(-q[negative].min()), float(q[positive].max())
        common = min(reach_neg, reach_pos)
        keep = positive & (q <= common)
        x = q[keep]
        order = np.argsort(-q[negative])
        left_q, left_i, left_s = -q[negative][order], intensity[negative][order], sigma[negative][order]
        y = 0.5 * (intensity[keep] + np.interp(x, left_q, left_i))
        s = 0.5 * np.hypot(sigma[keep], np.interp(x, left_q, left_s))
        notes.append(f"mean of both halves up to |qy| = {common:.3g} Å⁻¹")
        beyond = (negative & (-q > common)) if reach_neg > reach_pos else (positive & (q > common))
        if beyond.any():
            extra = np.argsort(np.abs(q[beyond]))
            x = np.concatenate([x, np.abs(q[beyond])[extra]])
            y = np.concatenate([y, intensity[beyond][extra]])
            s = np.concatenate([s, sigma[beyond][extra]])
            notes.append(f"then the {'qy < 0' if reach_neg > reach_pos else 'qy > 0'} half alone to {max(reach_neg, reach_pos):.3g} Å⁻¹")
    elif side in ("negative", "positive"):
        keep = negative if side == "negative" else positive
        x, y, s = np.abs(q[keep]), intensity[keep], sigma[keep]
        order = np.argsort(x)
        x, y, s = x[order], y[order], s[order]
        notes.append(f"the qy {'<' if side == 'negative' else '>'} 0 half")
    else:
        raise ValueError(f"Unknown side {side!r}.")
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    fallback = np.nanmedian(s / np.maximum(y, 1e-30)) if np.isfinite(s).any() else 0.1
    s = np.where(np.isfinite(s), s, np.abs(y) * (fallback if np.isfinite(fallback) else 0.1) + 1e-30)
    if x.size > max_points:
        group = int(math.ceil(x.size / max_points))
        count = x.size // group
        x = x[: count * group].reshape(count, group)
        y = y[: count * group].reshape(count, group)
        s = s[: count * group].reshape(count, group)
        weights = 1.0 / s**2
        y = (weights * y).sum(axis=1) / weights.sum(axis=1)
        s = 1.0 / np.sqrt(weights.sum(axis=1))
        x = x.mean(axis=1)
        notes.append(f"{group} neighbouring points merged ({count} points for the fit)")
    return x, y, s, "; ".join(notes)


@dataclass(frozen=True)
class Spacing:
    q: float
    """|qy| of the side maximum (Å⁻¹)."""
    distance_nm: float
    """2π / q: the dominant in-plane distance."""
    snr: float
    kind: str = "maximum"
    """``maximum`` (a resolved side peak) or ``shoulder`` (a change of slope only: a hint, not a measurement)."""


SHOULDER_MIN_SNR = 5.0


def spacing_from_peaks(peaks, *, q_min: float) -> Optional[Spacing]:
    """The strongest side maximum away from qy = 0 as a distance 2π/q; else the strongest shoulder."""
    side = [peak for peak in peaks if abs(peak.q) >= q_min and "spike" not in peak.flags]
    resolved = [peak for peak in side if "broad" not in peak.flags and "fit_failed" not in peak.flags]
    if resolved:
        best, kind = max(resolved, key=lambda peak: peak.snr), "maximum"
    else:
        shoulders = [peak for peak in side if peak.snr >= SHOULDER_MIN_SNR]
        if not shoulders:
            return None
        best, kind = max(shoulders, key=lambda peak: peak.snr), "shoulder"
    q = abs(float(best.q))
    return Spacing(q, 2.0 * math.pi / q / 10.0, float(best.snr), kind)


__all__ = ["HalvesChoice", "MIN_REACH", "MISMATCH_LIMIT", "MIN_COVERAGE", "Spacing", "choose_halves", "fit_curve", "spacing_from_peaks"]
