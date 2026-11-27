"""Show the newest result: while a folder batch runs, the canvas follows its newest completed prediction.

Without it the canvas stays empty during a batch, and during a watched measurement (Watch for new frames) the
estimate the run is for appears only when a row is opened. Ticked (Batch results ▸ Show the newest result, the
default), each completed prediction becomes the one to show. The canvas draws it ``AFTER_MS`` after it is done when
it has not drawn one for ``SHOW_MS``, otherwise when that time is up: a run predicting ten frames a second is drawn
once a second. Drawing holds the window's thread for about a tenth of a second (a Matplotlib figure); the short
wait lets the next frame, already started by then, be handed its successor first (Keep up with the detector gives
the model its frames from the window's thread), so the model is not kept waiting by the drawing.

The panels are those of a chosen row, without the Preprocessed one (that would read the frame again); the panel in
view stays in view. The first result of a run brings the Prediction result tab forward, later ones do not (the
user may have turned to the input). While the page is not in view nothing is drawn: the newest is drawn when it is
back. Choosing a row during a run unticks the box, so the chosen row stays; ticking it again follows the run from
its next result. At the end of the run the newest not drawn yet is drawn at once.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import QTimer

from src.gimap.app.presentation.i18n import tr

from ..multifile_results import PredictStatus

SHOW_MS = 1000
AFTER_MS = 20

CHOSEN = "Showing the chosen row: tick Show the newest result to follow the run again."


class LiveCanvasMixin:
    """Needs the page binding: ``ui`` (``predictionBatchFollowNewestCheck``, ``predictionPlotPanel``),
    ``_multifile_results_widget``, ``_multifile_prediction_active``, ``_current_image``, ``_display_prediction``,
    ``_update_current_file_display``, ``_set_predict_main_tab`` and ``_append_status_message``."""

    def _setup_live_canvas(self) -> None:
        self._live_row: Optional[int] = None
        self._live_shown_at = float("-inf")
        self._live_first = True
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.timeout.connect(self._show_newest)
        check = getattr(self.ui, "predictionBatchFollowNewestCheck", None)
        if check is not None:
            check.toggled.connect(self._follow_newest_toggled)

    def _follow_newest_wanted(self) -> bool:
        check = getattr(self.ui, "predictionBatchFollowNewestCheck", None)
        return check is not None and check.isChecked()

    def _follow_newest_toggled(self, on: bool) -> None:
        if not on:
            self._live_row = None
            self._live_timer.stop()

    def _live_run_started(self) -> None:
        self._live_row = None
        self._live_first = True
        self._live_shown_at = float("-inf")
        self._live_timer.stop()

    def _live_result_completed(self, row: int) -> None:
        """A prediction of the run is done: the newest to show, now or when ``SHOW_MS`` has passed."""
        if not self._follow_newest_wanted():
            return
        self._live_row = row
        if not self._live_timer.isActive():
            wait_ms = max(0.0, SHOW_MS - (time.monotonic() - self._live_shown_at) * 1000.0)  # (none drawn yet: 0)
            self._live_timer.start(max(AFTER_MS, int(wait_ms) + 1))

    def _live_run_ended(self) -> None:
        """The run is over: its newest result, if not drawn yet, at once."""
        self._live_timer.stop()
        if self._live_row is not None:
            self._show_newest()

    def _live_result_chosen(self) -> None:
        """A row was chosen: during a run the canvas keeps it (the box is unticked, and says so)."""
        check = getattr(self.ui, "predictionBatchFollowNewestCheck", None)
        if check is not None and check.isChecked() and self._multifile_prediction_active:
            check.setChecked(False)
            self._append_status_message(tr(CHOSEN), level="INFO")

    def _show_newest(self) -> None:
        row = self._live_row
        if row is None or not self._follow_newest_wanted():
            return
        canvas = getattr(self.ui, "predictionPlotPanel", None)  # (its tabs are hidden until there is something)
        if canvas is not None and not canvas.isVisible():
            self._live_timer.start(SHOW_MS)  # the page is not in view: drawn when it is back
            return
        self._live_row = None
        widget = self._multifile_results_widget
        result = widget.table_model.getResult(row) if widget is not None else None
        if result is None or result.status != PredictStatus.COMPLETED or not result.prediction_data:
            return
        outputs = result.prediction_data.get("prediction_data") or {}
        if not outputs:
            return
        self._live_shown_at = time.monotonic()
        kept, self._current_image = self._current_image, None  # no Preprocessed panel: not this frame's
        try:
            self._display_prediction(outputs, live=True)
        finally:
            self._current_image = kept
        first = result.file_path.splitlines()[0] if result.file_path else result.file_name
        self._update_current_file_display(first, max(1, int(getattr(result, "stack_count", 1))))
        if self._live_first:
            self._live_first = False
            self._set_predict_main_tab("result")


__all__ = ["AFTER_MS", "CHOSEN", "LiveCanvasMixin", "SHOW_MS"]
