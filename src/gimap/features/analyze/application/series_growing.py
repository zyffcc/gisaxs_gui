"""The Series map of a live series, made again only for the rows that arrived (``GrowingSeriesMap``).

A watched folder or a running batch draws its map again every second or so while rows arrive. Made from all its
rows each time (``build_series_map``), a map of 20 000 rows took 1.5 s of a busy computer — the facts of every row,
every row's x compared with the first, the rows stacked and the picture made again — and longer with every thousand
rows more; in the folded halves (``fold_signed``), a mean of the halves for every row, 30 s. Kept from the map
before, only the rows that arrived are made: their facts, their rows of the image and of the picture.

The map is the map ``build_series_map`` makes of the same rows, value for value (``tests/test_series_growing.py``).
It grows so only while the rows before are the same rows (the same objects, in the same places) with the frame
numbers they had (looked up again row by row unless the numbers say they number every frame as before:
``live_rows.numbers_agree``, 46 ms at 50 000 rows), the curve, halves, title and Sum N are the same, and every row has the points of the first row
(frames of one geometry). Otherwise it is made whole, as before, and a map of rows with other points (merged onto
one grid: ``stack_curves``) always. The times of the rows grow with them (``series_times.RowTimes``: the time
stamps of a row are read once); whether they are times of the frames depends on all of them.

The image and the picture grow in place (``shared.growing_rows``): the map shown before is a view of their first
rows and stays as it was. One map is made at a time (the caller sees to it: ``series_later.LiveStack``).
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import is_
from typing import Mapping, NamedTuple, Optional, Sequence

import numpy as np

from src.gimap.shared.growing_rows import GrowingRows

from ..domain import SeriesMap
from ..domain.series_map import (
    VIEW_BOTH, DisplayColumns, display_columns, display_rows, fold_signed, halves_of, has_both_signs, same_points,
    stack_growing, stack_rows,
)
from .live_rows import kept_numbers, numbers_agree
from .series_map import frame_key, row_facts
from .series_times import RowTimes


_NO_NUMBER = object()


class GrownMap(NamedTuple):
    series: SeriesMap
    picture: np.ndarray
    """The map on an even x grid (``SeriesMap.display``), with ``left`` and ``width``."""
    left: float
    width: float
    same: int
    """Rows at the start that are the rows of the map made before, unchanged in the picture (0: all made anew)."""


@dataclass
class _Kept:
    options: tuple
    """(curve, halves asked for, title, Sum N)."""
    rows: list
    keys: list
    """``frame_key`` of every row: its frame number is looked up again with the numbers of the next map."""
    facts: list
    labels: list
    refs: list
    folded: bool
    x_label: str
    first_x: np.ndarray
    order: Optional[np.ndarray]
    x: np.ndarray
    image: GrowingRows
    columns: DisplayColumns
    picture: Optional[GrowingRows]
    """``None``: the picture is the image (``display_rows``)."""
    times: RowTimes
    numbers: Optional[Mapping] = None
    """The frame numbers of the map (a dictionary as it was: copied)."""
    unnumbered: int = 0
    """Rows whose frame ``numbers`` has not (numbered by their row: a number given later may be another)."""


class GrowingSeriesMap:
    """``build_series_map`` for a series whose rows arrive one after another (see the module docstring)."""

    def __init__(self) -> None:
        self._kept: Optional[_Kept] = None

    def forget(self) -> None:
        """The next map is made whole (and nothing of the one before is held)."""
        self._kept = None

    def build(
        self, rows: Sequence[tuple], curve: str, *, view: str = VIEW_BOTH, title: str = "",
        numbers: Optional[Mapping] = None, group: int = 1, built_with: Optional[Mapping] = None,
    ) -> GrownMap:
        """The map of ``rows`` (as ``build_series_map``), with its picture."""
        if not rows:
            raise ValueError("No frame gave the curve.")
        options = (curve, view, title, max(1, int(group)))
        kept, self._kept = self._kept, None  # (kept again once made: a failure leaves nothing half grown)
        same, unnumbered = self._same(rows, options, numbers, kept)
        if same:
            kept.unnumbered = unnumbered
        if not same or not self._grow(kept, rows[same:], numbers):
            kept, same = self._whole(rows, options, numbers), 0
        kept.numbers = kept_numbers(numbers)
        times, time_from = kept.times.times()
        series = SeriesMap(
            kept.x, kept.image.view(), tuple(kept.labels), kept.x_label, curve, tuple(kept.refs),
            view=view if kept.folded else VIEW_BOTH, title=title, facts=tuple(kept.facts), times=times,
            time_from=time_from, built_with=dict(built_with or {}),
        )
        picture = series.image if kept.picture is None else kept.picture.view()
        if kept.first_x is not None:  # (rows of other points: made whole every time)
            self._kept = kept
        return GrownMap(series, picture, kept.columns.left, kept.columns.width, same)

    def same_rows(self, rows: Sequence[tuple], options: tuple, numbers: Optional[Mapping],
                  kept: Optional[_Kept] = None) -> int:
        """How many rows at the start of ``rows`` are the rows of the map kept, as they were (0: none can be kept)."""
        return self._same(rows, options, numbers, self._kept if kept is None else kept)[0]

    @staticmethod
    def _same(rows: Sequence[tuple], options: tuple, numbers: Optional[Mapping],
              kept: Optional[_Kept]) -> tuple[int, int]:
        """``(same_rows, rows of them without a frame number in numbers)``."""
        if kept is None or kept.options != options or len(rows) < len(kept.rows):
            return 0, 0
        if not all(map(is_, rows, kept.rows)):
            return 0, 0
        numbers = numbers or None
        if (numbers is None and kept.numbers is None) or (
                kept.numbers is not None and not kept.unnumbered and numbers_agree(kept.numbers, numbers)):
            return len(kept.rows), kept.unnumbered  # (every row numbered as before: none to look up)
        lookup, unnumbered = numbers or {}, 0
        for row, (key, fact) in enumerate(zip(kept.keys, kept.facts)):
            number = lookup.get(key, _NO_NUMBER)
            if number is _NO_NUMBER:
                number, unnumbered = row + 1, unnumbered + 1
            if int(number) != fact.number:
                return 0, 0  # (a frame listed before them moves the numbers of the rows after)
        return len(kept.rows), unnumbered

    def _grow(self, kept: _Kept, rows: Sequence[tuple], numbers: Optional[Mapping]) -> bool:
        """Add ``rows`` to the map kept; ``False`` (nothing changed): the map is to be made whole."""
        if not rows:
            return True
        curves = [(x, y) for x, y, *_rest in rows]
        if kept.options[1] != VIEW_BOTH:
            if kept.folded:
                curves = [fold_signed(x, y, kept.options[1]) for x, y in curves]
            elif any(has_both_signs(x) for x, _y in curves):
                return False  # (the first signed curve: every row is folded now)
        if not all(same_points(x, kept.first_x) for x, _y in curves):
            return False
        added = stack_rows(curves, kept.order)
        missing: list = []
        facts = row_facts(rows, numbers=numbers, group=kept.options[3], start=len(kept.rows), unnumbered=missing)
        kept.unnumbered += len(missing)
        kept.image.extend(added)
        if kept.picture is not None:
            kept.picture.extend(display_rows(added, kept.columns))
        self._note(kept, rows, facts)
        return True

    def _whole(self, rows: Sequence[tuple], options: tuple, numbers: Optional[Mapping]) -> _Kept:
        curve, view, _title, group = options
        missing: list = []
        facts = row_facts(rows, numbers=numbers, group=group, unnumbered=missing)
        curves, x_label, used = halves_of([(x, y) for x, y, *_rest in rows], view, rows[0][2])
        x, image, grid = stack_growing(curves)
        first_x, order = grid if grid is not None else (None, None)
        columns = display_columns(x)
        picture = display_rows(image, columns)
        kept = _Kept(options, [], [], [], [], [], used != VIEW_BOTH, x_label, first_x, order, x, GrowingRows(image),
                     columns, None if picture is image else GrowingRows(picture), RowTimes(), unnumbered=len(missing))
        self._note(kept, rows, facts)
        return kept

    @staticmethod
    def _note(kept: _Kept, rows: Sequence[tuple], facts: Sequence) -> None:
        kept.rows.extend(rows)
        kept.facts.extend(facts)
        kept.times.extend(facts)
        kept.keys.extend(frame_key(fact.source, fact.frame_index) for fact in facts)
        kept.labels.extend(name for *_rest, name, _ref in rows)
        kept.refs.extend(ref for *_rest, ref in rows)


__all__ = ["GrownMap", "GrowingSeriesMap"]
