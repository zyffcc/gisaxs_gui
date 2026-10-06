"""Several series side by side: one common q grid, one space of the main changes, and per series its
odd frames, its own stages, how fast it changed and where it ended.

* The curves are compared in shape (log10 I, each frame's mean level removed) on the q range every
  series covers; the grid is the first series' points there, the others interpolated (never extrapolated).
* Odd frames are found per series (``shared/series_stages``) and left out of the components.
* End state: the mean shape of the last frames, start state of the first ones; two series differ by the
  RMS of the difference of their end (or start) states, as a percentage of intensity. Three or more series are grouped (Ward) where the merge
  distances jump, so “alike” and “different” are read from the data, not from a fixed threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.series_stages import (
    Comparable,
    OddFrame,
    SeriesStages,
    comparable,
    find_stages,
    odd_frames,
    principal,
    progress,
)

END_FRAMES = 10
"""Frames averaged for the end state of a series."""
SPLIT_RATIO = 2.0
"""Groups are split where a merge distance is at least this many times the one before …"""
SPLIT_FLOOR = 2.0
"""… and the series it separates differ by more than this percentage."""

# Why a comparison cannot be made: the English of each error (``str(exc)``), from these templates so that the
# page can say it in the interface language (exact keys, or a template whose values it fills in again).
NO_SERIES = "Add a series first."
NO_DATA = "{name} has no data."
NO_COMMON_RANGE = "The series share no common q range."
DIFFERENT_AXES = "The series have different x axes: {axes}"
COMPARE_ERRORS = (NO_SERIES, NO_DATA, NO_COMMON_RANGE, DIFFERENT_AXES)


@dataclass(frozen=True)
class SeriesData:
    name: str
    x: np.ndarray
    image: np.ndarray
    """``(frames, len(x))`` intensities, NaN where a frame has no data."""
    labels: tuple
    x_label: str = "q (Å⁻¹)"
    source: str = ""
    """Where it came from: “Analyze” or the folder of its curve files."""
    paths: tuple = ()
    """The curve files (to read them again for a project)."""

    @property
    def rows(self) -> int:
        return int(self.image.shape[0])

    def renamed(self, name: str) -> "SeriesData":
        return SeriesData(name, self.x, self.image, self.labels, self.x_label, self.source, self.paths)


@dataclass(frozen=True)
class SeriesResult:
    name: str
    rows: int
    odd: tuple
    """``OddFrame``s (rows of this series)."""
    scores: np.ndarray
    """``(rows, k)``: the frames in the common space of the main changes."""
    stages: Optional[SeriesStages]
    """The series' own stages (on the common grid); ``None`` when it is too short to cut (fewer than
    ``shared/series_stages`` ``SHORTEST`` frames kept)."""
    half_row: Optional[int]
    ninety_row: Optional[int]
    end: np.ndarray
    """Mean shape (compared log I) of the last frames."""
    start: np.ndarray
    """Mean shape of the first frames."""
    end_curve: np.ndarray
    """Mean intensity of the last frames on the common grid."""


@dataclass(frozen=True)
class Comparison:
    q: np.ndarray
    """The common grid (every point; ``compared`` says which were compared)."""
    compared: np.ndarray
    x_label: str
    explained: np.ndarray
    results: tuple
    distance: np.ndarray
    """``(n, n)``: how far the end states are apart, in percent of intensity (RMS, shape)."""
    start_distance: np.ndarray
    """``(n, n)``: the same for the start states."""
    groups: tuple
    """1-based group of every series."""
    end_frames: int
    shape_only: bool

    @property
    def names(self) -> list[str]:
        return [result.name for result in self.results]

    def group_text(self) -> str:
        """“116, 117, 123 alike; 120 different”: the label of the groups in the table and the record
        (English; the page writes its own words from ``groups``). Empty below three series."""
        if len(self.results) < 3:
            return ""
        members: dict[int, list[str]] = {}
        for name, group in zip(self.names, self.groups):
            members.setdefault(group, []).append(name)
        if len(members) == 1:
            return "All alike at the end (no clear split)."
        return "; ".join(", ".join(names) + (" alike" if len(names) > 1 else " different")
                         for names in members.values())


def common_grid(series: Sequence[SeriesData]) -> np.ndarray:
    """The first series' x where every series has data."""
    lows, highs = [], []
    for item in series:
        finite = np.isfinite(item.x) & np.isfinite(item.image).any(axis=0)
        if not finite.any():
            raise ValueError(NO_DATA.format(name=item.name))
        lows.append(float(np.min(item.x[finite])))
        highs.append(float(np.max(item.x[finite])))
    low, high = max(lows), min(highs)
    grid = series[0].x[(series[0].x >= low) & (series[0].x <= high)]
    if grid.size < 3:
        raise ValueError(NO_COMMON_RANGE)
    return np.sort(grid)


def on_grid(item: SeriesData, grid: np.ndarray) -> np.ndarray:
    """Every frame of ``item`` interpolated onto ``grid`` (NaN outside its data)."""
    order = np.argsort(item.x)
    x = item.x[order]
    out = np.full((item.rows, grid.size), np.nan)
    for row, values in enumerate(item.image[:, order]):
        good = np.isfinite(x) & np.isfinite(values)
        if good.sum() < 2:
            continue
        inside = (grid >= x[good][0]) & (grid <= x[good][-1])
        out[row, inside] = np.interp(grid[inside], x[good], values[good])
    return out


def _block(data: Comparable, rows: slice) -> Comparable:
    return Comparable(x=data.x[rows], q=data.q, level=data.level[rows], columns=data.columns, shape_only=data.shape_only)


def _groups(end: np.ndarray, distance: np.ndarray) -> tuple:
    n = len(end)
    if n < 3:
        return (1,) * n
    from scipy.cluster.hierarchy import fcluster, linkage

    tree = linkage(end, method="ward")
    heights = np.sort(tree[:, 2])
    ratios = heights[1:] / np.maximum(heights[:-1], 1e-12)
    jump = int(np.argmax(ratios))
    if ratios[jump] < SPLIT_RATIO or float(np.max(distance)) < SPLIT_FLOOR:
        return (1,) * n
    threshold = 0.5 * (heights[jump] + heights[jump + 1])
    labels = fcluster(tree, t=threshold, criterion="distance")
    order: dict[int, int] = {}
    return tuple(order.setdefault(int(label), len(order) + 1) for label in labels)  # groups in series order


def compare(series: Sequence[SeriesData], *, q_range: Optional[Sequence[float]] = None, shape_only: bool = True,
            end_frames: int = END_FRAMES) -> Comparison:
    if not series:
        raise ValueError(NO_SERIES)
    labels = {item.x_label for item in series}
    if len(labels) > 1:
        raise ValueError(DIFFERENT_AXES.format(axes=", ".join(sorted(labels))))
    grid = common_grid(series)
    images = [on_grid(item, grid) for item in series]
    stacked = np.vstack(images)
    data = comparable(grid, stacked, q_range=q_range, shape_only=shape_only)
    cuts = np.cumsum([0] + [item.rows for item in series])
    odd_by_series, kept = [], []
    for index in range(len(series)):
        rows = slice(int(cuts[index]), int(cuts[index + 1]))
        found: tuple[OddFrame, ...] = odd_frames(_block(data, rows))
        odd_by_series.append(found)
        bad = {frame.row for frame in found}
        kept += [int(cuts[index]) + row for row in range(series[index].rows) if row not in bad]
    parts = principal(data.x, kept)
    results, ends, starts = [], [], []
    for index, item in enumerate(series):
        rows = slice(int(cuts[index]), int(cuts[index + 1]))
        bad = {frame.row for frame in odd_by_series[index]}
        own = np.array([row for row in range(item.rows) if row not in bad] or list(range(item.rows)))
        scores = parts.scores[rows]
        half, ninety = progress(scores[own, 0], own)
        last = own[-min(int(end_frames), own.size):]
        block = _block(data, rows)
        end, start = block.x[last].mean(axis=0), block.x[own[:min(int(end_frames), own.size)]].mean(axis=0)
        ends.append(end)
        starts.append(start)
        try:
            stages = find_stages(grid, images[index], data=block) if item.rows >= 3 else None
        except (ValueError, KeyError):  # too short to cut: ValueError below SHORTEST kept frames (KeyError: a net)
            stages = None  # this series without stages; the others are still compared
        with np.errstate(all="ignore"):
            end_curve = np.nanmean(images[index][last], axis=0)
        results.append(SeriesResult(name=item.name, rows=item.rows, odd=odd_by_series[index], scores=scores, stages=stages,
                                    half_row=half, ninety_row=ninety, end=end, start=start, end_curve=end_curve))
    results = _oriented(results)
    ends, starts = np.array(ends), np.array(starts)
    distance, start_distance = _apart(ends), _apart(starts)
    return Comparison(q=grid, compared=data.q, x_label=series[0].x_label, explained=parts.explained,
                      results=tuple(results), distance=distance, start_distance=start_distance,
                      groups=_groups(ends, distance), end_frames=int(end_frames), shape_only=shape_only)


def _apart(states: np.ndarray) -> np.ndarray:
    """Percent of intensity between every two states (RMS of their log10 difference)."""
    rms = np.sqrt(np.mean((states[:, None, :] - states[None, :, :]) ** 2, axis=2))
    return 100.0 * (10 ** rms - 1.0)


def _oriented(results: list) -> list:
    """Every component turned so that, over all series, it rises from the first frames to the last."""
    from dataclasses import replace

    change = np.zeros(results[0].scores.shape[1])
    for result in results:
        odd = {frame.row for frame in result.odd}
        rows = [row for row in range(result.rows) if row not in odd] or list(range(result.rows))
        edge = max(1, min(10, len(rows) // 4))
        change += result.scores[rows[-edge:]].mean(axis=0) - result.scores[rows[:edge]].mean(axis=0)
    sign = np.where(change < 0, -1.0, 1.0)
    if np.all(sign > 0):
        return results
    turned = []
    for result in results:
        rows = [row for row in range(result.rows) if row not in {frame.row for frame in result.odd}] or list(range(result.rows))
        half, ninety = progress(result.scores[rows, 0] * sign[0], np.asarray(rows))
        turned.append(replace(result, scores=result.scores * sign, half_row=half, ninety_row=ninety))
    return turned


__all__ = ["COMPARE_ERRORS", "DIFFERENT_AXES", "END_FRAMES", "NO_COMMON_RANGE", "NO_DATA", "NO_SERIES", "Comparison",
           "SeriesData", "SeriesResult", "common_grid", "compare", "on_grid"]
