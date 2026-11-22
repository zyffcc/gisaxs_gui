"""The structure at any time of an analysed scan (2026-11-22, user: the structure of a frame that is not one of the
analysed windows; interpolation, or a better way).

The package does both (``iampred.at_time``): the joint fit's path interpolated at that time, shown within seconds,
then refined to the frames of that time, which replaces it (about a minute or more). Each time comes into the result as
a window of its own, marked "on demand", among the windows in time order: the slider, the tiles, the images, the
evolution and the export treat it as any window. A scan analysed again drops them. Needs the widgets of
``views/sputter_main_view.py`` (``at_time_*``), ``service``, ``tasks`` and the result view of ``client_panel.py``.
"""

from __future__ import annotations

from .client_flow import Failure, _guarded
from .texts import message_text, text

AT_TIME_KEY = "sputter-at-time"
REFINE_STEPS = 8
"""Refinement evaluations per parameter of a time asked for (as the package's own per-window refinement)."""
INTERPOLATING = "{time} s: interpolating the joint fit …"
REFINING = "{time} s: interpolated; refining to the frames of this time …"
AT_TIME_DONE = "{time} s: refined to the frames of this time (fit deviation {deviation})."
AT_TIME_OUTSIDE = "Choose a time between the first and the last window ({first}–{last} s)."
AT_TIME_TEXTS = (INTERPOLATING, REFINING, AT_TIME_DONE, AT_TIME_OUTSIDE)


class AtTimePanelMixin:
    def _init_at_time(self) -> None:
        self._at_time_busy = False
        self._at_time_stop = False
        self._at_time_words = ("", {})
        self.at_time_button.clicked.connect(self.analyse_time)

    def _at_time_result_shown(self) -> None:
        """A result is shown: the row only for a scan whose package can analyse a time, its range the windows'."""
        result = self.client_result
        usable = result is not None and bool(result.windows) and self.service.client.can_analyse_time(result)
        self.at_time_row.setVisible(usable)
        if not usable:
            return
        times = [window.time_s for window in result.windows]
        self.at_time_value.blockSignals(True)
        self.at_time_value.setRange(min(times), max(times))
        if not self._at_time_busy:
            window = self.client_window()
            self.at_time_value.setValue(window.time_s if window is not None else min(times))
        self.at_time_value.blockSignals(False)
        self._at_time_sync()

    def _at_time_sync(self) -> None:
        busy = self._at_time_busy or getattr(self, "_client_busy", False)
        self.at_time_button.setEnabled(not busy)
        self.at_time_value.setEnabled(not self._at_time_busy)
        template, values = self._at_time_words
        self.at_time_note.set_full_text(text(template, **values) if template else "")

    def _at_time_say(self, template: str = "", **values) -> None:
        self._at_time_words = (template, values)
        self._at_time_sync()

    def analyse_time(self) -> None:
        """The time in the box: interpolated now, then refined (two runs of the package, one after the other)."""
        result = self.client_result
        if result is None or self._at_time_busy or getattr(self, "_client_busy", False):
            return
        times = [window.time_s for window in result.windows]
        time_s = round(float(self.at_time_value.value()), 2)
        if not times or not min(times) <= time_s <= max(times):
            self._at_time_say(AT_TIME_OUTSIDE, first=f"{min(times):g}", last=f"{max(times):g}")
            return
        self._at_time_busy, self._at_time_stop = True, False
        self._at_time_run(result, time_s, 0)

    def _at_time_run(self, result, time_s: float, refine: int) -> None:
        self._at_time_say(REFINING if refine else INTERPOLATING, time=f"{time_s:g}")
        client, stop = self.service.client, lambda: self._at_time_stop
        self.tasks.submit(AT_TIME_KEY,
                          lambda: _guarded(lambda: client.analyse_time(result, time_s, refine, lambda _event: None, stop)),
                          on_done=lambda outcome: self._at_time_done(outcome, time_s, refine),
                          on_error=lambda message, details: self._at_time_done(Failure(message, details), time_s,
                                                                               refine))

    def _at_time_done(self, outcome, time_s: float, refine: int) -> None:
        if isinstance(outcome, Failure):
            self._at_time_busy = False
            self._at_time_say()
            if outcome.kind != "cancelled":
                self._notify(message_text(outcome.message), error=True)
            return
        shown = self.client_result
        if shown is None or str(shown.folder) != str(outcome.folder):  # another result came up meanwhile
            self._at_time_busy = False
            self._at_time_say()
            return
        self.show_client_result(outcome)
        index = next((i for i, window in enumerate(outcome.windows) if window.on_demand and
                      abs(window.time_s - time_s) < 1e-6), None)
        if index is not None:
            self.select_window(index)
        if not refine:
            self._at_time_run(outcome, time_s, REFINE_STEPS)
            return
        self._at_time_busy = False
        window = outcome.windows[index] if index is not None else None
        entry = window.entry("refined") if window is not None else None
        deviation = f"{entry.fit_deviation:.3g}" if entry is not None and entry.fit_deviation is not None else "–"
        self._at_time_say(AT_TIME_DONE, time=f"{time_s:g}", deviation=deviation)

    def _dispose_at_time(self) -> None:
        self._at_time_stop = True
        self.tasks.cancel(AT_TIME_KEY)

__all__ = ["AT_TIME_KEY", "AT_TIME_TEXTS", "AtTimePanelMixin", "REFINE_STEPS"]
