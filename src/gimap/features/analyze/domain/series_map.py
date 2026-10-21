"""Intensity against frame and x: the curves of a series stacked into one image.

Rows are frames (or groups of summed frames) in list order, columns a common x grid. Nothing is
invented between the points (``stack_curves``): frames of one geometry and cut share their grid, or
differ only in the detector columns left out, and keep every value as measured (both halves of a cut
on |x| keep both points where their |x| coincide); curves on other grids are put on one even grid with
their own step, never beyond a curve's ends and never across a hole of it (a detector gap), where the
cells stay NaN. A trace in a hole or beyond the ends has no value either (``SeriesMap.trace``).

The x of a curve is not evenly spaced (one point per valid detector column, merged bins at low q, holes
at gaps), an image is: ``display_grid`` gives the picture of a map on an even grid, for drawing only.
A signed cut (qy on both sides of the beam) can be stacked as one half, both, or their mean
(``fold_signed``, ``VIEWS``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.signed_cut import HOLE_STEPS, local_steps, mean_of_halves

VIEW_BOTH, VIEW_POSITIVE, VIEW_NEGATIVE, VIEW_FOLDED, VIEW_MEAN = "both", "positive", "negative", "folded", "mean"
VIEWS = (VIEW_BOTH, VIEW_POSITIVE, VIEW_NEGATIVE, VIEW_FOLDED, VIEW_MEAN)
"""How a signed cut is stacked: as measured (both signs), the x > 0 half, the x < 0 half on |x|, every point of
both halves on |x|, or the mean of the halves (``shared/signed_cut.mean_of_halves``: never across a hole)."""
DISPLAY_COLUMNS = 8192
"""The most columns the picture of a map gets (``display_grid``)."""
UNION_GROWTH = 1.5
"""Curves share one grid when all their x together are at most this many times the longest curve's."""
MOMENT_WIDTH = 2.3548
"""Standard deviation → the FWHM a Gaussian of that deviation has."""


@dataclass(frozen=True)
class RowFacts:
    """What one row of a map is made of (the frame axis, the exports and the marks of a short sum use it)."""

    number: int
    """The frame as the frame axis counts it, from 1: the frame in its file for one multi-frame source, its
    place among all the listed frames otherwise."""
    source: str = ""
    """The file of the frame (the first module file of a set, the master of a chunked series)."""
    frame_index: int = 0
    frame_count: int = 1
    summed: int = 1
    """Frames summed into the row (1: a single frame)."""
    group: int = 1
    """Frames a row sums when its group is complete (the Sum N in force)."""
    timestamp: str = ""
    exposure_s: Optional[float] = None
    frame_time_s: Optional[float] = None

    @property
    def incomplete(self) -> bool:
        """Fewer frames than the other rows sum: the last group of a series whose length is no multiple of N."""
        return self.summed < self.group


@dataclass(frozen=True)
class SeriesMap:
    x: np.ndarray
    """Common x grid of every row (the unit is in ``x_label``)."""
    image: np.ndarray
    """``(rows, len(x))`` intensities; NaN where a frame has no data."""
    labels: tuple[str, ...]
    """What each row is (file, frame number, summed frames)."""
    x_label: str
    curve: str
    """Key of the curve stacked (``radial``, ``horizontal``, ...)."""
    refs: tuple = field(default=())
    """What opens each row again (``(path, frame_index)``), in row order."""
    title: str = ""
    """The English title of the curve stacked (“I(χ), q = 0.981–1.011 Å⁻¹”); empty: only the key is known."""
    view: str = VIEW_BOTH
    """The halves of a signed cut the rows hold (``VIEWS``); ``both``: the x as measured."""
    facts: tuple = field(default=())
    """``RowFacts`` of every row, in row order (empty: nothing is known but the labels)."""
    times: Optional[np.ndarray] = None
    """Seconds since the first row, or ``None`` when the frames carry no time."""
    time_from: str = ""
    """Where ``times`` come from: ``timestamps`` (of the frames) or ``exposure`` (frame × exposure time)."""
    built_with: dict = field(default_factory=dict, compare=False)
    """What the map was built with, for its exports: the list, the cut of the series, the settings record."""

    @property
    def rows(self) -> int:
        return int(self.image.shape[0])

    @property
    def numbers(self) -> tuple[int, ...]:
        """The frame number of every row (``RowFacts.number``; the row counted from 1 when nothing is known)."""
        if len(self.facts) == self.rows:
            return tuple(int(fact.number) for fact in self.facts)
        return tuple(range(1, self.rows + 1))

    def number(self, row: int) -> int:
        return self.numbers[int(np.clip(row, 0, self.rows - 1))]

    def display(self) -> tuple[np.ndarray, float, float]:
        """``(picture, left, width)`` of the map on an even x grid (``display_grid``)."""
        return display_grid(self.x, self.image)

    def profile(self, row: int) -> np.ndarray:
        """The curve of one row on the common grid."""
        return self.image[int(np.clip(row, 0, self.rows - 1))]

    def most_changing_x(self) -> float:
        """The x where the last quarter of the rows differs most from the first, in units of its noise.

        The noise of a column is taken from its second differences along the frames (a steady rise or
        fall cancels in them), so it can be told from a trend also in a series of a few frames, where the
        scatter inside a quarter of one or two rows says nothing; below four rows: the square root of the
        level. No column counts as quieter than the usual one, so a bright but steady background does
        not win over a weak line that grows or fades, and a column that happens to scatter little does
        not win over a real change.
        """
        image = self.image.astype(float)
        quarter = max(1, self.rows // 4)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # columns without data: NaN, left out below
            change = np.abs(np.nanmean(image[-quarter:], axis=0) - np.nanmean(image[:quarter], axis=0))
            if self.rows >= 4:
                noise = 1.4826 * np.nanmedian(np.abs(np.diff(image, n=2, axis=0)), axis=0) / math.sqrt(6.0)
            else:
                noise = np.sqrt(np.abs(np.nanmean(image, axis=0)))
            usual = noise[np.isfinite(noise) & (noise > 0)]
            score = change / np.maximum(noise, float(np.median(usual)) if usual.size else 1.0)
        score = np.where(np.isfinite(score), score, -np.inf)
        return float(self.x[int(np.argmax(score))]) if np.isfinite(score).any() else float(self.x[self.x.size // 2])

    def trace_columns(self, x_value: float, half_width: Optional[float] = None) -> tuple[np.ndarray, str]:
        """``(columns, how)``: the columns a trace over ``x_value ± half_width`` (default: one grid step) averages.
        ``how`` is ``window`` (the points inside it), ``nearest`` (a window narrower than the points around it,
        inside the data and not in a hole of x — merged bins at low q: the nearest point) or ``none`` (the window
        lies in a hole of x — a detector gap — or beyond the ends of the map: nothing was measured there)."""
        x = np.asarray(self.x, dtype=float)
        step = float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else 0.0
        half = max(float(half_width) if half_width is not None else step, 0.5 * step)
        window = np.abs(x - float(x_value)) <= half
        if window.any():
            return window, "window"
        after = int(np.searchsorted(x, float(x_value)))
        if after == 0 or after >= x.size or _holes(x)[after - 1]:
            return window, "none"
        window[after - 1 if float(x_value) - x[after - 1] <= x[after] - float(x_value) else after] = True
        return window, "nearest"

    def trace(self, x_value: float, half_width: Optional[float] = None) -> np.ndarray:
        """Mean intensity per row over ``x_value ± half_width`` (default: one grid step); NaN in every row when
        nothing was measured there (``trace_columns``: a detector gap, beyond the ends) — no value is made up."""
        window, how = self.trace_columns(x_value, half_width)
        if how == "none":
            return np.full(self.rows, np.nan)
        with np.errstate(all="ignore"):
            cells = self.image[:, window]
            counts = np.isfinite(cells).sum(axis=1)
            total = np.nansum(cells, axis=1)
            return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


@dataclass(frozen=True)
class PeakTrack:
    """A peak followed through the rows of a series map inside one x window (NaN: no peak there)."""

    window: tuple[float, float]
    position: np.ndarray
    """Centroid of the intensity above the local background (x unit of the map)."""
    fwhm: np.ndarray
    """The moment width: 2.355 × the standard deviation of that intensity inside the window — the FWHM of
    a Gaussian peak that lies well inside it. It grows with the window (tails, a sloping wing), so it is
    not the FWHM of a fitted profile (Batch Export ▸ peak fit gives that one, with its error)."""
    area: np.ndarray
    """Intensity above the background integrated over x (trapezoids on the x the row has; intensity × x unit)."""
    height: np.ndarray
    """Largest value above the background."""

    def table(self) -> list[tuple[str, np.ndarray]]:
        return [("position", self.position), ("fwhm", self.fwhm), ("area", self.area), ("height", self.height)]


def track_peak(series: SeriesMap, low: float, high: float) -> PeakTrack:
    """Centroid, width, area and height of the peak in ``[low, high]`` for every row.

    The background is the straight line through the mean of the first and of the last three
    points of the window, so a sloping background does not move the centroid. Rows with less
    than five points in the window, or nothing above the background, give NaN.
    """
    low, high = sorted((float(low), float(high)))
    window = (series.x >= low) & (series.x <= high)
    x = series.x[window]
    rows = series.rows
    position, fwhm, area, height = (np.full(rows, np.nan) for _ in range(4))
    if x.size < 5:
        return PeakTrack((low, high), position, fwhm, area, height)
    edge = max(1, min(3, x.size // 5))
    for row in range(rows):
        y = series.image[row, window].astype(float)
        finite = np.isfinite(y)
        if finite.sum() < 5:
            continue
        xs, ys = x[finite], y[finite]
        left_x, left_y = float(np.mean(xs[:edge])), float(np.mean(ys[:edge]))
        right_x, right_y = float(np.mean(xs[-edge:])), float(np.mean(ys[-edge:]))
        slope = (right_y - left_y) / (right_x - left_x) if right_x > left_x else 0.0
        net = ys - (left_y + slope * (xs - left_x))
        positive = np.clip(net, 0.0, None)
        total = float(positive.sum())
        if total <= 0:
            continue
        centre = float((xs * positive).sum() / total)
        spread = float(np.sqrt(max(((xs - centre) ** 2 * positive).sum() / total, 0.0)))
        position[row], fwhm[row] = centre, MOMENT_WIDTH * spread
        area[row] = float(np.sum(0.5 * (net[1:] + net[:-1]) * np.diff(xs)))  # the x of a map is not evenly spaced
        height[row] = float(net.max())
    return PeakTrack((low, high), position, fwhm, area, height)


def _sorted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _usual_step(x: np.ndarray) -> float:
    widths = np.diff(x)
    widths = widths[widths > 0]
    return float(np.median(widths)) if widths.size else 0.0


def _holes(x: np.ndarray) -> np.ndarray:
    """Per interval of ascending ``x``: wider than ``HOLE_STEPS`` of the steps usual around it — nothing was
    measured in it (a detector gap, the beam stop). A grid whose steps grow along the curve has none."""
    widths = np.diff(x)
    return widths > HOLE_STEPS * local_steps(x) if widths.size else np.zeros(0, dtype=bool)


def _one_grid(curves: Sequence[tuple[np.ndarray, np.ndarray]]) -> Optional[tuple[np.ndarray, list[np.ndarray]]]:
    """``(grid, columns)``: the x of every curve as one grid and, per curve, the column of each of its points —
    when the curves were measured on the same points and differ only in the points they have (frames of one
    geometry and cut whose valid detector columns differ); ``None`` when their points lie between each other's
    (another cut row, another geometry). An x that one curve has more than once (both halves of a signed cut
    on |x|) gets as many columns as the most any curve has there, filled in order: no point overwrites another."""
    steps = [step for step in (_usual_step(x) for x, _y in curves) if step > 0]
    longest = max(x.size for x, _y in curves)
    if not steps or longest < 2:
        return None
    every = np.sort(np.concatenate([x for x, _y in curves]))
    same = 1e-6 * min(steps)
    grid = every[np.concatenate([[True], np.diff(every) > same])]
    if grid.size > 1 and float(np.diff(grid).min()) < 0.5 * min(steps):
        return None
    places = [np.searchsorted(grid, x, side="right") - 1 for x, _y in curves]  # (ascending: the x are sorted)
    repeats = np.ones(grid.size, dtype=int)
    for place in places:
        if place.size:
            repeats = np.maximum(repeats, np.bincount(place, minlength=grid.size))
    if repeats.sum() > UNION_GROWTH * longest:
        return None
    first = np.concatenate([[0], np.cumsum(repeats)[:-1]])
    # The n-th point of a curve at one x goes to the n-th column of that x.
    columns = [first[place] + np.arange(place.size) - np.searchsorted(place, place, side="left") for place in places]
    return np.repeat(grid, repeats), columns


def stack_curves(curves: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """``(x, image)``: the curves on one grid, one row each (see the module docstring)."""
    if not curves:
        raise ValueError("No curves to stack.")
    first_x = np.asarray(curves[0][0], dtype=float)
    if all(np.asarray(x).shape == first_x.shape and np.allclose(np.asarray(x, dtype=float), first_x, equal_nan=True)
           for x, _y in curves):
        image = np.vstack([np.asarray(y, dtype=np.float32) for _x, y in curves])
        order = np.argsort(first_x, kind="stable")
        return first_x[order], image[:, order]
    sorted_curves = [_sorted(x, y) for x, y in curves]
    if not any(x.size for x, _y in sorted_curves):
        raise ValueError("The curves have no points.")
    found = _one_grid(sorted_curves)
    if found is not None:  # the same points, some left out: every value where it was measured
        grid, columns = found
        image = np.full((len(curves), grid.size), np.nan, dtype=np.float32)
        for row, ((_x, y), column) in enumerate(zip(sorted_curves, columns)):
            image[row, column] = y
        return grid, image
    low = min(float(x[0]) for x, _y in sorted_curves if x.size)
    high = max(float(x[-1]) for x, _y in sorted_curves if x.size)
    steps = [step for step in (_usual_step(x) for x, _y in sorted_curves) if step > 0]
    step = float(np.median(steps)) if steps else 0.0
    count = int((high - low) / step + 1e-9) + 1 if step > 0 else 1
    if 2 <= count <= 8 * DISPLAY_COLUMNS:
        grid = low + step * np.arange(count)  # the curves' own step exactly
    else:
        grid = np.linspace(low, high, max(2, min(count, 8 * DISPLAY_COLUMNS)))
    image = np.full((len(curves), grid.size), np.nan, dtype=np.float32)
    for row, (x, y) in enumerate(sorted_curves):
        if x.size < 2:
            continue
        values = np.interp(grid, x, y, left=np.nan, right=np.nan)
        after = np.clip(np.searchsorted(x, grid), 1, x.size - 1)  # the interval of x each grid point lies in
        values[_holes(x)[after - 1] & (grid > x[after - 1]) & (grid < x[after])] = np.nan
        image[row] = values
    return grid, image


def display_grid(x: np.ndarray, image: np.ndarray, *, max_columns: int = DISPLAY_COLUMNS) -> tuple[np.ndarray, float, float]:
    """``(picture, left, width)``: the map on an even x grid, for drawing only (values and x stay as they are).

    An image has columns of one width, the x of a map has not: drawn column by column, a line sits at the
    wrong x (by 0.06 Å⁻¹ for an I(q) with merged bins at low q). The picture has the usual step of ``x``;
    each of its columns shows the nearest point of ``x``, and the columns inside a hole of ``x`` — further
    from a point than that point stands for — stay NaN, so a detector gap is as wide as it is.
    ``left`` is the x of the picture's left edge and ``width`` the x it spans."""
    x = np.asarray(x, dtype=float)
    image = np.asarray(image)
    step = _usual_step(x)
    if x.size < 2 or step <= 0:
        return image, (float(x[0]) if x.size else 0.0) - 0.5, 1.0
    span = float(x[-1] - x[0])
    columns = int(round(span / step)) + 1
    if columns > max_columns:
        columns, step = int(max_columns), span / (int(max_columns) - 1)
    centres = x[0] + step * np.arange(columns)
    after = np.clip(np.searchsorted(x, centres), 1, x.size - 1)
    before = after - 1
    nearest = np.where(centres - x[before] <= x[after] - centres, before, after)
    picture = np.array(image[:, nearest], dtype=np.float32)
    reach = 0.5 * np.maximum(step, local_steps(x)[before])  # how far a point stands for the x around it
    picture[:, _holes(x)[before] & (np.abs(centres - x[nearest]) > reach)] = np.nan
    return picture, float(x[0] - 0.5 * step), float(columns * step)


def has_both_signs(x: np.ndarray) -> bool:
    """A signed cut: measured points on both sides of x = 0 (the two halves of I(qy))."""
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    return finite.size > 1 and float(finite.min()) < 0.0 < float(finite.max())


def fold_signed(x: np.ndarray, y: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    """One curve of a signed cut as the halves chosen (``VIEWS``): ``(x, y)`` ascending in x.

    ``both``: as measured. ``positive`` / ``negative``: that half on |x|. ``folded``: every measured point
    of both halves on |x| (two points where both halves have one). ``mean``: the mean of the halves on the
    |x| of the x > 0 half, a point averaged only where the other half has measured points on both sides —
    never across a hole of it, where the one half is taken as measured (``mean_of_halves``). The point at
    x = 0 belongs to no half."""
    if view not in VIEWS:
        raise ValueError(f"Unknown halves {view!r}; expected one of {VIEWS}.")
    x, y = _sorted(x, y)
    if view == VIEW_BOTH:
        return x, y
    measured = np.isfinite(y)
    plus, minus = (x > 0) & measured, (x < 0) & measured
    positive = (x[plus], y[plus])
    negative = (-x[minus][::-1], y[minus][::-1])
    if view == VIEW_POSITIVE:
        return positive
    if view == VIEW_NEGATIVE:
        return negative
    if view == VIEW_FOLDED:
        return _sorted(np.concatenate([positive[0], negative[0]]), np.concatenate([positive[1], negative[1]]))
    mean = mean_of_halves((*positive, None), (*negative, None), grid=VIEW_POSITIVE)
    return np.asarray(mean.q, dtype=float), np.asarray(mean.intensity, dtype=float)


def folded_label(x_label: str) -> str:
    """The axis of a half drawn on |x|: “qy (Å⁻¹)” → “|qy| (Å⁻¹)”."""
    text = str(x_label or "")
    unit = text.rfind(" (")
    name = (text[:unit] if unit >= 0 else text).strip().strip("|") or "x"
    return f"|{name}|" + (text[unit:] if unit >= 0 else "")


def series_map(
    curves: Sequence[tuple[np.ndarray, np.ndarray]], labels: Sequence[str], *, x_label: str, curve: str, refs=(),
    view: str = VIEW_BOTH, **more,
) -> SeriesMap:
    """The curves stacked. A signed cut (``has_both_signs`` in any curve) is folded to ``view`` first, and
    its axis is then |x|; curves of one sign have no halves, their ``view`` stays ``both``. ``more``: the
    further fields of ``SeriesMap`` (``title``, ``facts``, ``times``, ``time_from``, ``built_with``)."""
    if view != VIEW_BOTH and any(has_both_signs(x) for x, _y in curves):
        curves = [fold_signed(x, y, view) for x, y in curves]
        x_label = folded_label(x_label)
    else:
        view = VIEW_BOTH
    x, image = stack_curves(curves)
    return SeriesMap(x, image, tuple(labels), x_label, curve, tuple(refs), view=view, **more)


__all__ = [
    "DISPLAY_COLUMNS", "MOMENT_WIDTH", "PeakTrack", "RowFacts", "SeriesMap", "VIEWS", "VIEW_BOTH", "VIEW_FOLDED",
    "VIEW_MEAN", "VIEW_NEGATIVE", "VIEW_POSITIVE", "display_grid", "fold_signed", "folded_label", "has_both_signs",
    "series_map", "stack_curves", "track_peak",
]
