"""Stages of a series: where along the frames the curves change course.

The scores of the main components are cut into stages, each followed by a straight line in frame
order (optimal segmentation by dynamic programming): a steady drift stays one stage, a boundary is
where the direction or the pace of the change switches. A stage is added only while it explains at
least ``GAIN`` of the change one stage leaves unexplained (real curves are not exactly piecewise
straight, so a noise criterion alone would keep splitting) and more than noise would (a BIC-like
penalty with the noise from second differences, so a still series stays one stage).

A stage is a description, not a phase: a smooth change is also cut where its pace bends. What
changes between stages (which q grows or falls, relative to the rest of the curve) says whether a
boundary is structural.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from .comparable import Comparable, OddFrame, comparable, odd_frames, principal

MOST = 8
"""Most stages offered."""
SHORTEST = 5
"""Fewest frames in a stage."""
GAIN = 0.05
"""Share of the one-stage residual each further stage must explain."""
MAX_ROWS = 1500
"""Longer series are averaged in blocks of consecutive frames before they are cut."""
CHANGES_SHOWN = 3


@dataclass(frozen=True)
class StageChange:
    stage: int
    """The later stage (2 = from stage 1 to stage 2)."""
    rises: tuple
    """``(q, percent)`` where the curve grows most relative to the rest of it."""
    falls: tuple

    def text(self) -> str:
        grows = ", ".join(f"{q:.4g} (+{percent:.0f} %)" for q, percent in self.rises)
        drops = ", ".join(f"{q:.4g} ({percent:.0f} %)" for q, percent in self.falls)
        parts = [f"grows most at q {grows}" if grows else "", f"falls most at q {drops}" if drops else ""]
        return f"Stage {self.stage - 1} → {self.stage}: " + "; ".join(part for part in parts if part)


@dataclass(frozen=True)
class SeriesStages:
    q: np.ndarray
    """The q points compared."""
    rows: int
    odd: tuple
    """``OddFrame``s: left out of the components and the cut, still assigned to a stage."""
    explained: np.ndarray
    scores: np.ndarray
    """``(rows, k)``: every frame in the space of the main components."""
    level: np.ndarray
    """Every frame's mean log10 I over the compared q (its overall intensity)."""
    boundaries: dict
    """Number of stages → the rows where stages 2, 3, … begin."""
    unexplained: dict
    """Number of stages → the share of the one-stage residual left."""
    suggested: int
    count: int
    """The number shown: ``suggested`` unless chosen."""
    changes: dict
    """Number of stages → ``StageChange`` per boundary."""
    representatives: dict
    """Number of stages → per stage, the frame nearest to its mean curve."""
    half_row: Optional[int] = None
    """The first frame by which half of the change (main component) has happened."""
    ninety_row: Optional[int] = None

    @property
    def edges(self) -> tuple:
        return (0, *self.boundaries.get(self.count, ()), self.rows)

    @property
    def odd_rows(self) -> frozenset:
        return frozenset(frame.row for frame in self.odd)

    def stage_of(self, row: int) -> int:
        """1-based stage of a frame."""
        edges = self.edges
        return int(np.searchsorted(np.asarray(edges[1:-1]), int(row), side="right")) + 1

    def ranges(self) -> list[tuple[int, int]]:
        """``(first, last)`` row of every stage."""
        edges = self.edges
        return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]

    def stage_changes(self) -> tuple:
        return self.changes.get(self.count, ())

    def stage_representatives(self) -> tuple:
        return self.representatives.get(self.count, ())

    def with_count(self, count: int) -> "SeriesStages":
        if count not in self.boundaries:
            raise ValueError(f"{count} stages were not computed (1–{max(self.boundaries)}).")
        return replace(self, count=int(count))


# -- the cut --------------------------------------------------------------------------------------

def _segment_costs(scores: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``cost[i, j]``: squared residual of straight lines (in ``t``) through rows ``i … j-1``."""
    n = len(scores)

    def cumulative(values):
        return np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)])

    s1, st, stt = cumulative(np.ones(n)), cumulative(t), cumulative(t * t)
    sx, stx, sxx = cumulative(scores), cumulative(t[:, None] * scores), cumulative(np.sum(scores ** 2, axis=1))
    cost = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        j = np.arange(i + 1, n + 1)
        m = s1[j] - s1[i]
        spread_t = (stt[j] - stt[i]) - (st[j] - st[i]) ** 2 / m
        total_x = sx[j] - sx[i]
        cross = (stx[j] - stx[i]) - (st[j] - st[i])[:, None] * total_x / m[:, None]
        residual = (sxx[j] - sxx[i]) - np.sum(total_x ** 2, axis=1) / m
        safe = np.where(spread_t > 1e-12, spread_t, 1.0)
        slope_part = np.where(spread_t > 1e-12, np.sum(cross ** 2, axis=1) / safe, 0.0)
        cost[i, j] = np.maximum(residual - slope_part, 0.0)
    return cost


def segment(scores, t=None, *, most: int = MOST, shortest: int = SHORTEST) -> dict:
    """Number of stages → ``(cost, boundaries)`` (boundaries as positions in ``scores``)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    t = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
    cost = _segment_costs(scores, t)
    for i in range(n + 1):
        cost[i, i + 1:min(n + 1, i + shortest)] = np.inf
    best = np.full((most + 1, n + 1), np.inf)
    back = np.zeros((most + 1, n + 1), dtype=int)
    best[1] = cost[0]
    for k in range(2, most + 1):
        for j in range(1, n + 1):
            total = best[k - 1, :j] + cost[:j, j]
            i = int(np.argmin(total))
            best[k, j], back[k, j] = total[i], i
    found = {}
    for k in range(1, most + 1):
        if not math.isfinite(best[k, n]):
            break
        bounds, j = [], n
        for level in range(k, 1, -1):
            j = int(back[level, j])
            bounds.append(j)
        found[k] = (float(best[k, n]), tuple(sorted(bounds)))
    return found


def noise_penalty(scores) -> float:
    """What one more stage would explain in a series of pure noise (BIC-like): (2k + 1) log n times the
    noise variance per component, the noise taken from second differences (a straight drift is not noise)."""
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    if n < 4:
        return 0.0
    second = scores[2:] - 2.0 * scores[1:-1] + scores[:-2]
    per_component = float(np.median(np.sum(second ** 2, axis=1))) / 6.0 / k
    return per_component * (2 * k + 1) * math.log(n)


def choose(found: dict, gain: float = GAIN, penalty: float = 0.0) -> int:
    """The number of stages: add one while it explains at least ``gain`` of the one-stage residual and
    more than ``penalty`` (what noise would explain)."""
    total = found[1][0]
    if total <= 0:
        return 1
    chosen = 1
    for k in sorted(found)[1:]:
        if (found[k - 1][0] - found[k][0]) < max(gain * total, penalty):
            break
        chosen = k
    return chosen


# -- what a stage is ------------------------------------------------------------------------------

def _changes(log: np.ndarray, q: np.ndarray, edges: Sequence[int], kept: np.ndarray) -> tuple:
    means = []
    for first, end in zip(edges[:-1], edges[1:]):
        rows = kept[(kept >= first) & (kept < end)]
        means.append(np.nanmean(log[rows if rows.size else np.arange(first, end)], axis=0))
    spacing = 0.02 * float(np.nanmax(q) - np.nanmin(q))
    changes = []
    for stage in range(1, len(means)):
        difference = means[stage] - means[stage - 1]
        difference = difference - np.nanmean(difference)
        picked = {}
        for sign in (1, -1):
            values = np.where(np.isfinite(difference), sign * difference, -np.inf)
            chosen = []
            for index in np.argsort(-values):
                if len(chosen) == CHANGES_SHOWN or values[index] <= 0:
                    break
                if all(abs(q[index] - q[other]) > spacing for other in chosen):
                    chosen.append(index)
            picked[sign] = tuple((float(q[i]), float(100.0 * (10 ** difference[i] - 1.0))) for i in chosen)
        changes.append(StageChange(stage=stage + 1, rises=picked[1], falls=picked[-1]))
    return tuple(changes)


def _representatives(x: np.ndarray, edges: Sequence[int], kept: np.ndarray) -> tuple:
    rows = []
    for first, end in zip(edges[:-1], edges[1:]):
        members = kept[(kept >= first) & (kept < end)]
        if not members.size:
            members = np.arange(first, end)
        mean = x[members].mean(axis=0)
        rows.append(int(members[np.argmin(np.sum((x[members] - mean) ** 2, axis=1))]))
    return tuple(rows)


def oriented(parts, rows):
    """Components turned so that each rises from the first frames to the last (their sign is arbitrary)."""
    rows = np.asarray(rows, dtype=int)
    if rows.size < 2:
        return parts
    edge = max(1, min(10, rows.size // 4))
    change = parts.scores[rows[-edge:]].mean(axis=0) - parts.scores[rows[:edge]].mean(axis=0)
    sign = np.where(change < 0, -1.0, 1.0)
    return replace(parts, scores=parts.scores * sign, loadings=parts.loadings * sign[:, None])


def progress(values, rows) -> tuple[Optional[int], Optional[int]]:
    """The rows by which half and 90 % of the change of ``values`` (in row order) have happened."""
    values, rows = np.asarray(values, dtype=float), np.asarray(rows)
    if values.size < 6:
        return None, None
    smooth = np.array([np.median(values[max(0, i - 2):i + 3]) for i in range(values.size)])
    start, end = np.median(smooth[:min(5, smooth.size)]), np.median(smooth[-min(10, smooth.size):])
    if abs(end - start) < 1e-12:
        return None, None
    share = (smooth - start) / (end - start)
    found = []
    for limit in (0.5, 0.9):
        reached = np.flatnonzero(share >= limit)
        found.append(int(rows[reached[0]]) if reached.size else None)
    return found[0], found[1]


# -- all of it ------------------------------------------------------------------------------------

def find_stages(q, image, *, q_range: Optional[Sequence[float]] = None, shape_only: bool = True,
                most: int = MOST, shortest: int = SHORTEST, gain: float = GAIN,
                data: Optional[Comparable] = None) -> SeriesStages:
    """Odd frames, main components, stages (1 … ``most``) and what changes, for a series of curves."""
    image = np.asarray(image, dtype=float)
    if image.shape[0] < 3:
        raise ValueError("At least three frames are needed to find stages.")
    data = data or comparable(q, image, q_range=q_range, shape_only=shape_only)
    odd: tuple[OddFrame, ...] = odd_frames(data)
    odd_rows = {frame.row for frame in odd}
    kept = np.array([row for row in range(image.shape[0]) if row not in odd_rows], dtype=int)
    if kept.size < 3:
        kept, odd = np.arange(image.shape[0]), ()
    parts = principal(data.x, kept)
    parts = oriented(parts, kept)
    block = max(1, math.ceil(kept.size / MAX_ROWS))
    starts = kept[::block]
    scores = np.array([parts.scores[kept[i:i + block]].mean(axis=0) for i in range(0, kept.size, block)])
    found = segment(scores, starts.astype(float), most=most, shortest=max(1, math.ceil(shortest / block)))
    suggested = choose(found, gain, noise_penalty(scores))
    with np.errstate(invalid="ignore", divide="ignore"):
        log = np.where(image > 0, np.log10(image), np.nan)[:, data.columns]
    boundaries, changes, representatives = {}, {}, {}
    total = found[1][0]
    for count, (_cost, bounds) in found.items():
        rows = tuple(int(starts[b]) for b in bounds)
        boundaries[count] = rows
        edges = (0, *rows, image.shape[0])
        changes[count] = _changes(log, data.q, edges, kept)
        representatives[count] = _representatives(data.x, edges, kept)
    half, ninety = progress(parts.scores[kept, 0], kept) if suggested > 1 else (None, None)
    return SeriesStages(
        q=data.q, rows=int(image.shape[0]), odd=odd, explained=parts.explained, scores=parts.scores, level=data.level,
        boundaries=boundaries, unexplained={k: (c / total if total > 0 else 0.0) for k, (c, _b) in found.items()},
        suggested=suggested, count=suggested, changes=changes, representatives=representatives,
        half_row=half, ninety_row=ninety,
    )


__all__ = [
    "GAIN", "MOST", "SHORTEST", "SeriesStages", "StageChange", "choose", "find_stages", "noise_penalty", "oriented", "progress",
    "segment",
]
