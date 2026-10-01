"""Curves of a series made comparable, the frames that do not belong, and the main directions of change.

* ``comparable``: log10 I on the bins measured in (nearly) every frame, optionally with each frame's
  mean level removed — an overall intensity change (beam, thickness, exposure) is then not a new
  structure; the level is kept separately.
* ``odd_frames``: a frame compared with the frames before it and with those after it, before any
  principal components (an odd frame would otherwise become a component of its own). It is odd when it
  matches neither side, by far more than the series moves from frame to frame there — a frame right
  after a sudden change matches the frames that follow, and a fast but steady change moves its
  neighbours too. A difference concentrated in a few bins is the detector (a flickering hot spot, a
  module edge) rather than the sample.
* ``principal``: principal components of the rows kept, every row projected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

COVERAGE = 0.95
"""A bin is compared when this share of the frames has data there."""
ODD_RATIO = 5.0
"""An odd frame differs from both sides this many times more than the series moves per frame there …"""
ODD_FLOOR = 5.0
"""… and this many times more than a typical step between frames (not just noise in a still series)."""
NEIGHBOURS = 3
"""Frames on each side a frame is compared with."""
NARROW_BINS = 0.01
"""Share of the bins that holds the difference of a detector artefact …"""
NARROW_SHARE = 0.8
"""… and how much of the difference it holds."""


@dataclass(frozen=True)
class Comparable:
    x: np.ndarray
    """``(rows, bins)``: log10 I, each row's mean removed when ``shape_only``."""
    q: np.ndarray
    """The bins compared."""
    level: np.ndarray
    """Each row's mean log10 I over those bins (its overall intensity)."""
    columns: np.ndarray
    """Indices of the compared bins in the original grid."""
    shape_only: bool


@dataclass(frozen=True)
class OddFrame:
    row: int
    z: float
    """How far it is from both sides, in units of the series' movement per frame there."""
    q: float
    """Where it differs most (the median q of its largest differences)."""
    narrow: bool
    """The difference sits in a few bins: probably the detector, not the sample."""

    @property
    def reason(self) -> str:
        if self.narrow:
            return f"differs only near q {self.q:.4g}: probably the detector, not the sample"
        return f"differs from the frames before and after it ({self.z:.0f}× their step), most near q {self.q:.4g}"


@dataclass(frozen=True)
class Components:
    scores: np.ndarray
    """``(rows, k)``: every row in the space of the components (odd rows projected too)."""
    loadings: np.ndarray
    centre: np.ndarray
    explained: np.ndarray
    """Share of the variance (of the rows kept) of every component, largest first."""


def comparable(q, image, *, q_range: Optional[Sequence[float]] = None, coverage: float = COVERAGE,
               shape_only: bool = True) -> Comparable:
    """``image``: ``(rows, len(q))`` intensities (NaN where a frame has no data)."""
    q = np.asarray(q, dtype=float)
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or image.shape[1] != q.size:
        raise ValueError("The curves must share one q grid: (frames, points).")
    with np.errstate(invalid="ignore", divide="ignore"):
        log = np.where(image > 0, np.log10(image), np.nan)
    inside = np.isfinite(q)
    if q_range is not None:
        low, high = sorted(float(value) for value in q_range)
        inside &= (q >= low) & (q <= high)
    keep = inside & (np.mean(np.isfinite(log), axis=0) >= coverage)
    if keep.sum() < 3:
        raise ValueError("Fewer than three q points have data in nearly every frame: widen the q range.")
    columns = np.flatnonzero(keep)
    log = log[:, columns]
    fill = np.nanmedian(log, axis=0)
    log = np.where(np.isfinite(log), log, fill)
    level = log.mean(axis=1)
    x = log - level[:, None] if shape_only else log
    return Comparable(x=x, q=q[columns], level=level, columns=columns, shape_only=shape_only)


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def odd_frames(data: Comparable, *, ratio: float = ODD_RATIO, floor: float = ODD_FLOOR,
               neighbours: int = NEIGHBOURS) -> tuple[OddFrame, ...]:
    """Frames that match neither the frames before them nor those after them."""
    x, rows = data.x, data.x.shape[0]
    if rows < 4:
        return ()
    steps = np.array([_rms(x[i + 1] - x[i]) for i in range(rows - 1)])
    typical = max(float(np.median(steps)), 1e-12)
    found = []
    top_count = max(3, int(round(NARROW_BINS * x.shape[1])))
    for row in range(rows):
        sides = []
        for side in (range(max(0, row - neighbours), row), range(row + 1, min(rows, row + neighbours + 1))):
            side = list(side)
            if side:
                difference = x[row] - np.median(x[side], axis=0)
                sides.append((_rms(difference), difference))
        distance, difference = min(sides, key=lambda item: item[0])
        around = [j for j in range(max(0, row - neighbours - 1), min(rows - 1, row + neighbours + 1)) if j not in (row - 1, row)]
        movement = max(float(np.median(steps[around])) if around else typical, typical)
        score = distance / movement
        if score <= ratio or distance <= floor * typical:
            continue
        energy = difference ** 2
        top = np.argsort(-energy)[:top_count]
        share = float(energy[top].sum() / max(energy.sum(), 1e-300))
        found.append(OddFrame(row=int(row), z=float(score), q=float(np.median(data.q[top])), narrow=share > NARROW_SHARE))
    return tuple(found)


def principal(x, keep: Optional[Sequence[int]] = None, *, variance: float = 0.95, most: int = 10) -> Components:
    """Components that explain ``variance`` of the rows kept (at most ``most``), every row projected."""
    x = np.asarray(x, dtype=float)
    rows = np.arange(x.shape[0]) if keep is None else np.asarray(keep, dtype=int)
    if rows.size < 2:
        raise ValueError("At least two frames are needed.")
    centre = x[rows].mean(axis=0)
    _u, singular, vt = np.linalg.svd(x[rows] - centre, full_matrices=False)
    power = singular ** 2
    explained = power / power.sum() if power.sum() > 0 else np.zeros_like(power)
    count = int(min(most, np.searchsorted(np.cumsum(explained), variance) + 1, vt.shape[0]))
    loadings = vt[:max(1, count)]
    return Components(scores=(x - centre) @ loadings.T, loadings=loadings, centre=centre, explained=explained)


__all__ = ["COVERAGE", "Comparable", "Components", "OddFrame", "comparable", "odd_frames", "principal"]
