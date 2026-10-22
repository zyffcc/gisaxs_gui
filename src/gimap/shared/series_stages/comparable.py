"""Curves of a series made comparable, the frames that do not belong, and the main directions of change.

* ``comparable``: log10 I on the bins measured in (nearly) every frame, optionally with each frame's
  mean level removed — an overall intensity change (beam, thickness, exposure) is then not a new
  structure; the level is kept separately.
* ``odd_frames``: a frame compared with the frames before it and with those after it (the median of
  ``NEIGHBOURS`` on each side), before any principal components (an odd frame would otherwise become a
  component of its own). It is odd when it matches neither side, by far more than the series moves from
  frame to frame there — a frame right after a sudden change matches the frames that follow, and a fast
  but steady change moves its neighbours too. Two measures, either one suffices:

  - over the whole curve (RMS): a frame that is different everywhere;
  - in the few bins (``NARROW_BINS``) where it differs from both sides *in the same direction*: a spike.
    Over the whole curve a spike is diluted — a hot column 25× too bright in 1 % of the bins is 1.4 in
    log10 there but only ~0.14 RMS, under 4× a typical step of a noisy series — so it needs its own
    measure. The same direction keeps real changes out: a steady change differs in opposite directions
    from the frames before and after, a step matches the frames on one side.

  A difference concentrated in a few bins is the detector (a flickering hot spot, a hot column, a module
  edge) rather than the sample. An odd frame does not vote for its neighbours: the search is repeated
  without it, so two or three glitched frames in a row, or two at the end of the series, are all found
  (before, only the middle or the outermost one); four or more alike are a state of their own (at the
  ends: from three). At an end only one side can be compared, so a change in the first or last one or
  two frames cannot be told from an artefact: it is reported as odd, with its reason, as before.
* ``principal``: principal components of the rows kept, every row projected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

COVERAGE = 0.95
"""A bin is compared when this share of the frames has data there."""
ODD_RATIO = 5.0
"""An odd frame differs from both sides this many times more than the series moves per frame there …"""
ODD_FLOOR = 5.0
"""… and this many times more than a typical step between frames (not just noise in a still series)."""
SPIKE_RATIO = 8.0
"""… or, in its few most different bins, from both sides in the same direction this many times more than
those bins move per frame there. Higher than ``ODD_RATIO`` because these bins are the largest of the frame:
in noise alone they reach ~3.5× (measured on Gaussian and few-count Poisson series of 5–3200 frames), where
the whole curve stays under 1×. A spike of 10× in 1 % of the bins moves the whole curve by about one step, so
one too small to be found here hardly bends the components."""
NEIGHBOURS = 3
"""Frames on each side a frame is compared with."""
NARROW_BINS = 0.01
"""Share of the bins that holds the difference of a detector artefact …"""
NARROW_SHARE = 0.8
"""… and how much of the difference it holds."""
PASSES = 3
"""The search for odd frames is repeated at most this often, each time without the odd frames found."""
ROBUST = 1.4826
"""Median of |x| → standard deviation for normally distributed x."""


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
    """How far it is from both sides, in units of the series' movement per frame there (over the whole curve, or
    for a spike in its few most different bins)."""
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


def _same_direction(sides: list) -> np.ndarray:
    """The part of a frame's difference that it has from both sides in the same direction (at an end of the
    series, from its one side): zero where it lies between the frames before and after, as in a steady change."""
    if len(sides) == 1:
        return sides[0]
    before, after = sides
    return np.where(before * after > 0, np.sign(before) * np.minimum(np.abs(before), np.abs(after)), 0.0)


def _where(difference: np.ndarray, q: np.ndarray, count: int) -> tuple[np.ndarray, float, bool]:
    """The ``count`` bins holding most of ``difference``, their median q, and whether they hold nearly all of it."""
    energy = difference ** 2
    top = np.argsort(-energy)[:count]
    share = float(energy[top].sum() / max(energy.sum(), 1e-300))
    return top, float(np.median(q[top])), share > NARROW_SHARE


def _odd_once(x: np.ndarray, q: np.ndarray, *, ratio: float, floor: float, neighbours: int,
              spike: float) -> list[OddFrame]:
    """One look at every row of ``x`` (rows: positions in ``x``)."""
    rows = x.shape[0]
    jumps = np.abs(np.diff(x, axis=0))  # (rows - 1, bins): every step at every bin
    steps = np.sqrt(np.mean(jumps ** 2, axis=1))
    typical = max(float(np.median(steps)), 1e-12)
    usual = ROBUST * np.median(jumps, axis=0)  # every bin's typical step over the series (robust to a few spikes)
    count = max(3, int(round(NARROW_BINS * x.shape[1])))
    found = []
    for row in range(rows):
        sides = [x[row] - np.median(x[list(side)], axis=0)
                 for side in (range(max(0, row - neighbours), row), range(row + 1, min(rows, row + neighbours + 1)))
                 if len(side)]
        difference = min(sides, key=_rms)
        distance = _rms(difference)
        # the steps near the frame, without the two that touch it
        around = [j for j in range(max(0, row - neighbours - 1), min(rows - 1, row + neighbours + 1)) if j not in (row - 1, row)]
        movement = max(float(np.median(steps[around])) if around else typical, typical)
        if distance / movement > ratio and distance > floor * typical:  # different over the whole curve
            _top, where, narrow = _where(difference, q, count)
            found.append(OddFrame(row=int(row), z=float(distance / movement), q=where, narrow=narrow))
            continue
        both = _same_direction(sides)
        top, where, narrow = _where(both, q, count)
        moves = _rms(usual[top])  # how far those bins move per frame: over the series, and near the frame
        if around:
            moves = max(moves, float(np.median(np.sqrt(np.mean(jumps[np.ix_(around, top)] ** 2, axis=1)))))
        score = _rms(both[top]) / max(moves, 1e-12)
        if score > spike:  # a spike: a few bins far off both sides, the same way
            found.append(OddFrame(row=int(row), z=float(score), q=where, narrow=narrow))
    return found


def odd_frames(data: Comparable, *, ratio: float = ODD_RATIO, floor: float = ODD_FLOOR,
               neighbours: int = NEIGHBOURS, spike: float = SPIKE_RATIO) -> tuple[OddFrame, ...]:
    """Frames that match neither the frames before them nor those after them. An odd frame does not vote: the
    frames that had one among their neighbours are looked at again without it (``PASSES``)."""
    x, rows = data.x, data.x.shape[0]
    if rows < 4:
        return ()
    found: dict[int, OddFrame] = {}
    kept = previous = np.arange(rows)
    recent = None
    for _ in range(PASSES):
        frames = [replace(frame, row=int(kept[frame.row]))
                  for frame in _odd_once(x[kept], data.q, ratio=ratio, floor=floor, neighbours=neighbours, spike=spike)]
        if recent is not None:  # only a frame whose neighbours were left out can have changed
            place = {int(row): index for index, row in enumerate(previous)}
            frames = [frame for frame in frames
                      if any(abs(place[frame.row] - place[row]) <= neighbours for row in recent)]
        if not frames:
            break
        found.update((frame.row, frame) for frame in frames)
        previous, recent = kept, [frame.row for frame in frames]
        kept = np.array([row for row in range(rows) if row not in found], dtype=int)
        if kept.size < 4:
            break
    return tuple(found[row] for row in sorted(found))


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


__all__ = ["COVERAGE", "SPIKE_RATIO", "Comparable", "Components", "OddFrame", "comparable", "odd_frames", "principal"]
