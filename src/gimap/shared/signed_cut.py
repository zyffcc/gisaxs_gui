"""A cut through the beam, I(q) with signed q (the horizontal GISAXS cut): its two halves and the one
curve a fit takes of them. Every fit prepares its cut here.

After the beam centre is on the symmetry axis the two halves (q < 0, q > 0) should agree. A model is a
function of |q| and a fit takes one curve: the mean of the halves, or one of them (``FIT_HALVES``). Both
halves laid over each other on |q| are for looking; ``fit_halves`` turns that choice into the mean.

* ``choose_halves`` — which halves to use, with the reason: averaged when they agree; when one is cut
  short by the detector edge, a gap or a beam stop, the other one; when both are complete but disagree,
  both are kept (shown in two colours) and the better-covered one is fitted.
* ``mean_of_halves`` — the mean, in whatever unit q has: a point is the mean of two measurements only
  where the other half has measured points next to it. In a hole of the other half, and where one half
  starts later or reaches further, the measured half stands alone with its own σ; nothing is drawn
  across a hole. It says which points are which.
* ``prepare_cut`` / ``fit_curve`` — the curve of a fit with q in Å⁻¹ (the automatic analysis, the quick
  physical fit): the mean or one half, merged to at most ``MAX_FIT_POINTS`` points, and a note of what
  was done. A chosen half without points is ``MissingHalf``: the other half never stands in for it.
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
"""A fit takes at most this many points of one curve: more are merged (``merge_points``)."""
FIT_HALVES = ("mean", "positive", "negative")
"""What a fit can take of a signed cut. ``both`` / ``both_abs`` (display) are not among them."""
MISSING_HALF = {"positive": "No points on the q > 0 half", "negative": "No points on the q < 0 half"}
"""The message of ``MissingHalf`` for each half — these two sentences everywhere."""
BAD_SIGMA = "Every supplied sigma must be finite and positive"
"""Why a curve whose every point is an observation is refused for a σ that cannot weight its point — the
sentence the fit itself has for it, so the reason is the same whether or not points had to be merged."""
HOLE_STEPS = 2.5
"""Neighbouring points of a half further apart than this many of its steps there have a hole between them
(a detector gap, a mask, the beam stop): one missing point is not a hole, two are."""
STEP_WINDOW = 9
"""Intervals over which the step of a half is taken (their median: a hole among them does not count)."""
EDGE = 0.01
"""Share of a step by which a point may lie outside the other half's range, or off one of its points, and
still count as measured there (the two halves of a centred cut differ in the last digits of |q|)."""


class MissingHalf(ValueError):
    """The chosen half of a cut has no usable points (``MISSING_HALF``)."""


def fit_halves(choice, default: str = "mean") -> str:
    """The halves a fit takes for a stored or foreign choice: ``both``, ``both_abs``, ``None`` and anything
    unknown become ``default``."""
    return choice if choice in FIT_HALVES else default


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


def _halves(q: np.ndarray, intensity: np.ndarray, pixels: Optional[np.ndarray], positive_only: bool = True):
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    measured = np.isfinite(q) & np.isfinite(intensity)
    if positive_only:
        measured &= intensity > 0
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


# -- the mean of the two halves -------------------------------------------------------------------

def local_steps(q: np.ndarray) -> np.ndarray:
    """The usual distance between neighbouring points at every interval of ascending ``q``: the median of
    the ``STEP_WINDOW`` intervals around it, so that a hole does not count as a step and a grid whose
    steps grow along the curve is judged by the steps where it is."""
    widths = np.diff(np.asarray(q, dtype=float))
    if widths.size < 3:
        return np.full(widths.shape, float(np.median(widths)) if widths.size else 0.0)
    half = min(STEP_WINDOW // 2, (widths.size - 1) // 2)
    padded = np.pad(widths, half, mode="reflect")
    steps = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    usual = widths[widths > 0]
    return np.where(steps > 0, steps, float(np.median(usual)) if usual.size else 0.0)  # repeated q: not a step of 0


def _measured_around(at: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Whether ``half`` (the ascending |q| of one half) has measured points on both sides of each ``at``
    (ascending: the points of the other half) with no hole between them, or a point at ``at`` itself.
    The finer sampling of the two halves says what a step is: a few scattered points of a half are not a
    curve to interpolate on. A half of one point has nothing to interpolate between: it is measured at
    the |q| of that point only."""
    if not half.size or not at.size:
        return np.zeros(at.shape, dtype=bool)
    own = np.zeros(at.shape)
    if at.size > 1:
        own = local_steps(at)
        own = np.minimum(np.append(own, own[-1]), np.insert(own, 0, own[0]))  # per point: the finer side
    if half.size < 2:
        return np.abs(at - half[0]) <= np.maximum(EDGE * own, 1e-9 * np.abs(at))
    steps = local_steps(half)
    right = np.clip(np.searchsorted(half, at, side="left"), 1, half.size - 1)
    step = steps[right - 1]
    step = np.where(own > 0, np.minimum(step, own), step)
    inside = (at >= half[0] - EDGE * steps[0]) & (at <= half[-1] + EDGE * steps[-1])
    on_a_point = np.minimum(np.abs(at - half[right - 1]), np.abs(half[right] - at)) <= EDGE * step
    return inside & ((half[right] - half[right - 1] <= HOLE_STEPS * step) | on_a_point)


@dataclass(frozen=True)
class HalvesMean:
    """The mean of the two halves of a cut on |q| (ascending)."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray]
    """½·√(σ₊² + σ₋²) of a mean, the half's own σ elsewhere; ``None`` when a half has no σ."""
    halves: np.ndarray
    """Per point: 2 — the mean of both halves; 1 — the q > 0 half alone; −1 — the q < 0 half alone."""

    @property
    def averaged(self) -> int:
        """Points that are the mean of two measurements."""
        return int(np.count_nonzero(self.halves == 2))

    @property
    def single(self) -> int:
        """Points that come from one half alone."""
        return int(self.halves.size) - self.averaged


def mean_of_halves(positive, negative, *, grid: Optional[str] = None) -> HalvesMean:
    """The mean of the halves; each is ``(|q| ascending, I, σ or None)``, q in any unit.

    One half gives the |q| of the result: ``grid`` (``positive`` / ``negative``), else the one with more
    points (the q > 0 half when they have as many) — and always that one when the half asked for has
    fewer than two points (nothing can be interpolated to one point). A point of it is averaged with the
    other half — interpolated between its two neighbours there, σ likewise — only where the other half
    has measured points on both sides without a hole between them (``HOLE_STEPS``), or a point at the same
    |q|. Elsewhere it stays as measured, and the points of the other half that lie in a hole of this
    one, before its first or beyond its last point are added as they are — never a point at a |q| this
    half has itself, so two halves on the same |q| (the beam on a pixel edge) give every |q| once.

    Points of the other half that are too far apart to interpolate between (a single point, a few
    scattered ones) go into the mean only where they have the |q| of a point of this half; between its
    points they are left out. A half without points leaves the other one as it is.
    """
    sizes = {"positive": len(positive[0]), "negative": len(negative[0])}
    if grid not in sizes or sizes[grid] < 2:
        grid = "positive" if sizes["positive"] >= sizes["negative"] else "negative"
    first, second, sign = (positive, negative, 1) if grid == "positive" else (negative, positive, -1)
    q, y, s = (None if part is None else np.asarray(part, dtype=float) for part in first)
    other_q, other_y, other_s = (None if part is None else np.asarray(part, dtype=float) for part in second)
    both = _measured_around(q, other_q)
    intensity = y.copy()
    sigma = None if s is None or (other_s is None and other_q.size) else s.copy()
    if both.any():
        intensity[both] = 0.5 * (y[both] + np.interp(q[both], other_q, other_y))
        if sigma is not None:
            sigma[both] = 0.5 * np.hypot(s[both], np.interp(q[both], other_q, other_s))
    alone = ~_measured_around(other_q, q)  # the other half where this one has no points
    halves = np.concatenate([np.where(both, 2, sign), np.full(int(alone.sum()), -sign)]).astype(np.int8)
    q = np.concatenate([q, other_q[alone]])
    order = np.argsort(q, kind="stable")
    intensity = np.concatenate([intensity, other_y[alone]])[order]
    if sigma is not None and alone.any():
        sigma = np.concatenate([sigma, other_s[alone]])
    return HalvesMean(q[order], intensity, None if sigma is None else sigma[order], halves[order])


# -- the curve of a fit ---------------------------------------------------------------------------

def merge_points(q, intensity, sigma, max_points: int = MAX_FIT_POINTS):
    """``(q, I, σ, group)`` with at most ``max_points`` points: consecutive points merged in groups of
    ``group`` (the mean weighted by 1/σ² with the σ of that mean; the plain mean without σ). ``group`` is 1
    when nothing had to be merged. The last points that do not fill a group are left out. Points with a
    σ that is not finite and positive cannot be weighted: ``ValueError`` (``BAD_SIGMA``) when they would
    have to be merged."""
    q, intensity = np.asarray(q, dtype=float), np.asarray(intensity, dtype=float)
    if q.size <= max_points:
        return q, intensity, sigma, 1
    group = int(math.ceil(q.size / max_points))
    count = q.size // group
    shape = (count, group)
    q, intensity = q[: count * group].reshape(shape), intensity[: count * group].reshape(shape)
    if sigma is None:
        return q.mean(axis=1), intensity.mean(axis=1), None, group
    sigma = np.asarray(sigma, dtype=float)
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError(BAD_SIGMA)
    weights = 1.0 / sigma[: count * group].reshape(shape) ** 2
    total = weights.sum(axis=1)
    return q.mean(axis=1), (weights * intensity).sum(axis=1) / total, 1.0 / np.sqrt(total), group


@dataclass(frozen=True)
class PreparedCut:
    """The curve a fit takes of a signed cut: |q| ascending, I, σ (``None`` when none was given)."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray]
    note: str
    """What the points are, in English, its parts joined by ``"; "`` (for the record and the report)."""
    halves_used: str
    """``mean``, ``positive`` or ``negative``: under ``mean`` a cut with one half is that half, and so is a
    cut whose other half is too sparse to be averaged with it or kept beside it."""
    averaged: int
    """Points that are the mean of two measurements, before merging."""
    single: int
    """Points from one half alone, before merging."""
    merged: int = 1
    """How many neighbouring points make one point of the fit (1: none were merged)."""

    @property
    def points(self) -> int:
        return int(self.q.size)


def prepare_cut(q, intensity, sigma, pixels, side: str, *, max_points: int = MAX_FIT_POINTS,
                every_point: bool = False) -> PreparedCut:
    """The half (or the mean of both) a fit uses of the cut I(qy), signed qy in Å⁻¹: measured points only
    (finite, I > 0, ``pixels`` > 0 when given), at most ``max_points`` of them.

    ``side``: ``mean`` (``mean_of_halves`` on the |qy| of the qy > 0 half; a cut with one half only, or
    with a half too sparse to give one mean: the other half), ``positive`` or ``negative`` (``MissingHalf``
    when it has no measured point). A σ that is missing for single points is replaced by the usual relative
    σ of the others; without any σ (``None``) the result has none.

    The note of a mean counts its points — how many are averaged, how many come from one half — when the
    ranges it gives do not say it: a hole in one half, a half that begins more than a point after the
    other, halves that share no |qy|. The numbers are always in ``averaged`` and ``single``.

    ``every_point``: the cut holds measured points only (the detector columns of Analyze), so a point with
    I ≤ 0 is an observation too — a zero count, a value under the subtracted background — and no σ is made
    up for a point: a σ of the points taken that is not finite and positive is ``ValueError``
    (``BAD_SIGMA``).
    """
    if side not in FIT_HALVES:
        raise ValueError(f"Unknown side {side!r}.")
    q, intensity, negative, positive = _halves(q, intensity, pixels, positive_only=not every_point)
    given = None if sigma is None else np.asarray(sigma, dtype=float)
    names = {"negative": "qy < 0", "positive": "qy > 0"}
    if every_point and given is not None:
        taken = given[{"mean": negative | positive, "positive": positive, "negative": negative}[side]]
        if not np.all(np.isfinite(taken) & (taken > 0)):
            raise ValueError(BAD_SIGMA)

    def half(keep):
        order = np.argsort(np.abs(q[keep]), kind="stable")
        return np.abs(q[keep])[order], intensity[keep][order], None if given is None else given[keep][order]

    notes, averaged = [], 0
    mean = None
    if side == "mean" and negative.any() and positive.any():
        mean = mean_of_halves(half(positive), half(negative), grid="positive")  # as this curve always was
        x, y, s, averaged = mean.q, mean.intensity, mean.sigma, mean.averaged
    if mean is not None and (averaged or np.unique(mean.halves).size > 1):
        used = "mean"
        reach = {"negative": float(-q[negative].min()), "positive": float(q[positive].max())}
        common = min(reach.values())
        longer = max(reach, key=reach.get)
        notes.append(f"mean of both halves up to |qy| = {common:.3g} Å⁻¹")
        if reach[longer] > common:
            notes.append(f"then the {names[longer]} half alone to {reach[longer]:.3g} Å⁻¹")
        alone = mean.halves != 2
        inside = int(np.count_nonzero(alone & (x <= common)))
        before = int(np.count_nonzero(alone & (x < max(float(-q[negative].max()), float(q[positive].min())))))
        # One point before both halves have begun is no hole: the |qy| of two halves lie between each other.
        if inside > min(before, 1) or not averaged:
            notes.append(f"{averaged} points averaged, {mean.single} from one half alone "
                         f"({inside} of them inside that range, where the other half has no points)")
    elif mean is not None:  # nothing of the other half could be averaged with this one, or kept beside it
        used = "positive" if mean.halves[0] > 0 else "negative"
        notes += [f"the {names[used]} half",
                  f"the {names['negative' if used == 'positive' else 'positive']} half is too sparse to average with"]
    else:
        used = side if side != "mean" else ("positive" if positive.any() else "negative")
        keep = positive if used == "positive" else negative
        if not keep.any():
            if side == "mean":
                raise ValueError("The horizontal cut has no measured points.")
            raise MissingHalf(MISSING_HALF[side])
        x, y, s = half(keep)
        notes.append(f"the {names[used]} half")
        if side == "mean":
            notes.append(f"the {names['negative' if used == 'positive' else 'positive']} half has no points")
    single = int(x.size) - averaged
    if s is not None and not every_point:
        s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
        fallback = np.nanmedian(s / np.maximum(y, 1e-30)) if np.isfinite(s).any() else 0.1
        s = np.where(np.isfinite(s), s, np.abs(y) * (fallback if np.isfinite(fallback) else 0.1) + 1e-30)
    x, y, s, group = merge_points(x, y, s, max_points)
    if group > 1:
        notes.append(f"{group} neighbouring points merged ({x.size} points for the fit)")
    return PreparedCut(x, y, s, "; ".join(notes), used, averaged, single, group)


def fit_curve(
    q: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    pixels: Optional[np.ndarray],
    side: str,
    *,
    max_points: int = MAX_FIT_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(|qy|, I, σ, note) of the half (or the mean of both) a fit uses, at most ``max_points`` points:
    ``prepare_cut`` with a σ for every point (10 % of I when the cut has none)."""
    cut = prepare_cut(q, intensity, sigma, pixels, side, max_points=max_points)
    s = cut.sigma if cut.sigma is not None else 0.1 * np.abs(cut.intensity) + 1e-30
    return cut.q, cut.intensity, s, cut.note


__all__ = [
    "BAD_SIGMA", "EDGE", "FIT_HALVES", "HOLE_STEPS", "MAX_FIT_POINTS", "MIN_COVERAGE", "MIN_REACH", "MISMATCH_LIMIT",
    "MISSING_HALF", "SPECULAR_POINTS", "STEP_WINDOW", "HalvesChoice", "HalvesMean", "MissingHalf", "PreparedCut", "choose_halves",
    "fit_curve", "fit_halves", "local_steps", "mean_of_halves", "merge_points", "prepare_cut",
]
