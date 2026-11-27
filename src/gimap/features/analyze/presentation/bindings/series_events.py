"""Events of the run on the Series map: a dashed line across the map above the first frame measured after each event
(a spray pulse, a shutter, a step of temperature), named by the event.

The events come from a CSV file (``application/series_events.py``) chosen in the options under the plots. A file that
grows while the run goes on (the log of a spray controller) is read again whenever the map is drawn and the file has
changed. An event needs the time of the frames: a map without a time per frame says so, and so does a map none of
whose frames lies near the events (another clock or time zone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr, trf

from ...application.series_events import event_rows, events_record, read_events
from ..texts import message_text

EVENT_MARKS = ("events", "Events of the run")
EVENT_COLOUR = "#ff4fd8"
"""Magenta: apart from the stage lines and the colour maps, in both themes."""
EVENTS_FILTER = "Events (*.csv *.txt)"
EVENTS_NONE = "No events file"
EVENTS_MARKED = "{n} of {total} events of {name} on the map"
EVENTS_WAITING = "{total} events of {name}: marked on the map once it is built"
EVENTS_OUTSIDE = ("None of the {total} events of {name} lies within the frames' times: check that the events and the "
                  "frames were stamped by the same clock and time zone.")
EVENTS_NO_TIME = "The frames carry no time: the events of {name} cannot be placed."
EVENTS_FAILED = "Could not read the events: {error}"


def _stamp(path: Path) -> Optional[tuple]:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


class SeriesEventsMixin:
    """Needs ``series_events_button``, ``series_events_clear_button``, ``series_events_label``, ``series_map_view``,
    ``_series_map``, ``_series_extent``, ``_last_folder`` and ``_status``."""

    def _connect_series_events(self) -> None:
        self._series_events_path: Optional[Path] = None
        self._series_events: tuple = ()
        self._series_events_stamp = None
        self._series_event_items: list = []
        self.series_map_view.marks.add_layer(*EVENT_MARKS)
        self.series_events_button.clicked.connect(self.choose_series_events)
        self.series_events_clear_button.clicked.connect(self.clear_series_events)
        self._series_say_events()

    def choose_series_events(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Events of the Run"), self._last_folder, tr(EVENTS_FILTER))
        if path:
            self.load_series_events(Path(path))

    def load_series_events(self, path: Path) -> bool:
        """Mark the events of ``path`` on the map (and on every map after it, until cleared)."""
        try:
            events = read_events(Path(path))
        except (OSError, ValueError) as exc:
            self._status(trf(EVENTS_FAILED, error=message_text(exc)), "error")
            return False
        self._series_events_path, self._series_events = Path(path), events
        self._series_events_stamp = _stamp(Path(path))
        self.series_events_clear_button.show()
        self._series_mark_events()
        return True

    def clear_series_events(self) -> None:
        self._series_events_path, self._series_events, self._series_events_stamp = None, (), None
        self.series_events_clear_button.hide()
        self._series_mark_events()

    def _series_reload_events(self) -> None:
        """The events file again when it changed since it was read (a log written while the run goes on)."""
        path = self._series_events_path
        if path is None:
            return
        stamp = _stamp(path)
        if stamp is not None and stamp != self._series_events_stamp:
            try:
                self._series_events = read_events(path)
            except (OSError, ValueError):
                return  # (caught in the middle of a write: the next drawing reads it)
            self._series_events_stamp = stamp

    def _series_mark_events(self) -> None:
        """Lines across the map at the events' rows, and what the line in the options says of them."""
        import pyqtgraph as pg

        view = self.series_map_view
        for item in self._series_event_items:
            view.marks.forget(EVENT_MARKS[0], item)
            try:
                view.plot.removeItem(item)
            except RuntimeError:  # the scene is gone
                pass
        self._series_event_items = []
        series = self._series_map
        if series is None or self._series_events_path is None or self._series_extent is None:
            self._series_say_events()
            return
        placed = event_rows(series, self._series_events)
        series.built_with["events"] = {"file": str(self._series_events_path), "events": events_record(placed)}
        low, _high = self._series_extent
        pen = pg.mkPen(EVENT_COLOUR, width=1.5, style=Qt.DashLine)
        for event, row, _after in placed:
            if row is None:
                continue
            line = pg.InfiniteLine(pos=float(row), angle=0, pen=pen, movable=False)
            label = pg.TextItem(event.label, color=EVENT_COLOUR, anchor=(0.0, 1.0))
            label.setPos(low, float(row))
            for item, depth in ((line, 12), (label, 13)):
                item.setZValue(depth)
                view.plot.addItem(item, ignoreBounds=True)
                view.marks.track(EVENT_MARKS[0], item, True)
                self._series_event_items.append(item)
        self._series_say_events(placed)

    def _series_say_events(self, placed=None) -> None:
        label = self.series_events_label
        path = getattr(self, "_series_events_path", None)
        if path is None:
            label.setText(tr(EVENTS_NONE))
            return
        values = dict(name=path.name, total=len(self._series_events))
        series = self._series_map
        if series is None:
            label.setText(trf(EVENTS_WAITING, **values))
        elif series.times is None:
            label.setText(trf(EVENTS_NO_TIME, **values))
        elif placed is not None and not any(row is not None for _event, row, _after in placed):
            label.setText(trf(EVENTS_OUTSIDE, **values))
        else:
            inside = sum(row is not None for _event, row, _after in (placed or ()))
            label.setText(trf(EVENTS_MARKED, n=inside, **values))


__all__ = ["SeriesEventsMixin"]
