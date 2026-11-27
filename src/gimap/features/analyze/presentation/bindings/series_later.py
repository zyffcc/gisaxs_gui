"""A live Series map of many rows, stacked on a thread of its own: the window only shows it.

A running batch or a watched folder draws its map again as rows arrive (``batch_live.py``, ``watch_series.py``).
Stacking a map of many rows takes long — 20 000 rows of 1 000 points: about a second — and on the window's thread
the window froze for as long at every redraw. A live map of more than ``LATER_ROWS`` rows, or one whose last
stacking took longer than ``LATER_BUILD_S`` (a busy computer: 700 rows took 0.4 s while every core was in use), is
therefore stacked on the map's own thread (``_series_build_later``), one map at a time, and shown when it is done;
a smaller one is stacked and shown at once, as before.

The rows are taken as they are when stacking begins — in order, one incidence angle (``_series_split_angles``).
Rows that arrive meanwhile wait in ``_series_rows`` for the next map: until then they are only appended or put in
the place of a row of the same frame, so the rows of the map shown are the first rows of ``_series_rows``.

A map stacked for rows that are no longer the map's is not shown (``_series_later_forget``): the map was cleared, a
new run began, it was drawn at once meanwhile (another curve, halves or angle; the last map of a batch), or the page
closed.

Every row that arrives for a live map shares its x with the rows before when it is the same (``_series_live_row``,
``application.series_map.share_x``): a map of 20 000 rows of 1 000 points then holds 160 MB less.

A live map grows from the map before (``LiveStack``: ``application.series_growing``, ``GrowingDisplay``): only the
rows that arrived are stacked, and only they are made into the picture drawn — the same map as one stacked whole.
At 20 000 rows a map took 1.0–1.5 s of a busy computer, now about 0.1 s; and a picture of the folded halves of a
cut, a mean of the halves for each row, 30 s. The last map of a run grows too; a map of a curve, halves or Sum N
chosen, and any map made at once in the window that is no live map, is made whole, as before (``_show_series_map``).
"""

from __future__ import annotations

import logging
import threading
import time
from operator import is_
from pathlib import Path

from src.gimap.app.presentation.components.detector_display import GrowingDisplay
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application.series_growing import GrowingSeriesMap
from ...application.series_map import share_x
from .series_stages import series_name

LATER_ROWS = 1000
"""A live map of more rows than this is stacked on the map's own thread (fewer: at once — a few hundredths of a
second, and a map that shows every frame the moment it arrives)."""
LATER_BUILD_S = 0.05
"""A live map whose last stacking took longer than this is stacked on the map's own thread, whatever its rows."""


def made_here(series) -> None:
    """The frame numbers and short sums of every row of ``series`` worked out now, on the thread that made it: the
    window asks for them at its first drawing (25 ms at 50 000 rows, every redraw of a live map)."""
    series.numbers, series.short_rows  # noqa: B018 - (cached properties)


class LiveStack:
    """The stacking of a live map, kept from one map to the next (see the module docstring): ``stack`` gives what
    ``_series_stacking`` gives. One map at a time — the map's own thread or the window — and one asked for while
    another is being made is made whole by the caller (``None``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._map, self._display = GrowingSeriesMap(), GrowingDisplay()
        self._forgotten = False
        self._reset()

    def _reset(self) -> None:
        self._map.forget()
        self._display.forget()
        self._rows, self._files, self._stems = [], {}, []

    def forget(self) -> None:
        """The next map is made whole, and nothing of the map before is held (at once, or once the map being made
        is done)."""
        if self._lock.acquire(blocking=False):
            try:
                self._reset()
            finally:
                self._lock.release()
        else:
            self._forgotten = True

    def stack(self, rows: list, key: str, options: dict, built_with: dict, log: bool):
        """``(map, name, (picture, left, width), (valid, prepared))`` of ``rows``; ``None``: another map is being made."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._forgotten:
                self._forgotten = False
                self._reset()
            try:
                name = self._name(rows)
                grown = self._map.build(rows, key, built_with={"name": name, **built_with}, **options)
                made_here(grown.series)
                valid, shown, sample = self._display.show(grown.picture, grown.same, log)
            except BaseException:
                self._reset()
                raise
            if self._forgotten:
                self._forgotten = False
                self._reset()
            return grown.series, name, (grown.picture, grown.left, grown.width), (valid, (log, shown, sample))
        finally:
            self._lock.release()

    def _name(self, rows: list) -> str:
        """``series_name(row_files(rows))``, the files of the rows before kept."""
        if len(rows) < len(self._rows) or not all(map(is_, rows, self._rows)):
            self._rows, self._files, self._stems = [], {}, []
        for row in rows[len(self._rows):]:
            source = row[4][0]
            text = source if isinstance(source, str) else str(source)
            if text.casefold() not in self._files:  # (as ``row_files``: each file once, in row order)
                path = source if isinstance(source, Path) else Path(text)
                self._files[text.casefold()] = path
                self._stems.append(path.stem)
        self._rows.extend(rows[len(self._rows):])
        return series_name(list(self._files.values()), self._stems)


class SeriesLaterMixin:
    """Needs ``_series_rows``, ``_series_split_angles``, ``_series_stacking``, ``_series_show_built`` and
    ``_series_say`` (``series_display.py``, ``series_mixed.py``)."""

    def _series_live_row(self, row: tuple) -> tuple:
        """A row that arrives for the live map, its x the x of a row before when equal (``share_x``)."""
        known = getattr(self, "_series_x_known", None)
        if known is None:
            known = self._series_x_known = []
        return share_x(row, known)

    def _series_later_forget(self) -> None:
        """A map being stacked now is not shown when done."""
        self._series_later = None

    def _series_live_stack(self) -> LiveStack:
        stacker = getattr(self, "_series_live", None)
        if stacker is None:
            stacker = self._series_live = LiveStack()
        return stacker

    def _series_live_forget(self) -> None:
        """The next live map is stacked whole (a new run, the map cleared, a map that is no live map shown), and no
        rows of the map before are held (``_series_exposures`` keeps them too)."""
        self._series_exposures_kept = None
        self._series_angles_kept = self._watch_rows_listed = None  # (``_series_split_angles``, the watch's order)
        stacker = getattr(self, "_series_live", None)
        if stacker is not None:
            stacker.forget()

    def _series_later_busy(self) -> bool:
        """A map is being stacked on the map's thread."""
        return getattr(self, "_series_later", None) is not None

    def _series_redraw_live(self, *, keep_view: bool, keep_q, before=None, then=None) -> bool:
        """Draw the live map again: ``before()``, the map, ``then()`` — at once for a map of up to ``LATER_ROWS``
        rows, else stacked on the map's thread and shown when done. ``False``: nothing was started, a map is still
        being stacked (its rows came first; the rows that arrived since are drawn at the next redraw)."""
        if self._series_later_busy():
            return False
        if len(self._series_rows) <= LATER_ROWS and getattr(self, "_series_build_s", 0.0) <= LATER_BUILD_S:
            if before is not None:
                before()
            self._show_series_map(keep_view=keep_view, keep_q=keep_q, live=True)
            if then is not None:
                then()
            return True
        self._series_split_angles()
        stack = self._series_stacking(live=True)
        token = self._series_later = object()
        runner = getattr(self, "_series_later_runner", None)
        if runner is None:  # its own thread: stacking a map is no work of the page's (``tasks.is_busy``)
            runner = self._series_later_runner = TaskRunner(self, max_threads=1)

        def timed():
            began = time.monotonic()
            built = stack()
            return built, time.monotonic() - began

        runner.submit(
            "series-map", timed,
            on_done=lambda result: self._series_later_done(token, result, keep_view, keep_q, before, then),
            on_error=lambda message, details: self._series_later_failed(token, message, details),
        )
        return True

    def _series_later_done(self, token, result, keep_view, keep_q, before, then) -> None:
        if token is not getattr(self, "_series_later", None):
            return  # (its rows are no longer the map's)
        self._series_later = None
        built, self._series_build_s = result
        if before is not None:
            before()
        self._series_show_built(built, keep_view=keep_view, keep_q=keep_q)
        if then is not None:
            then()

    def _series_later_failed(self, token, message: str, details: str) -> None:
        if token is not getattr(self, "_series_later", None):
            return
        self._series_later = None
        logging.getLogger(__name__).debug("The live Series map could not be stacked:\n%s", details)
        self._series_say(message)


__all__ = ["LATER_BUILD_S", "LATER_ROWS", "LiveStack", "SeriesLaterMixin", "made_here"]
