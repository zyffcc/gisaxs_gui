"""Finish exports in a worker: large fit tables, figures and records must not block Analyze."""

from __future__ import annotations

import time

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

from ...application import FOLDERS
from ..batch_texts import batch_end_text, end_toast_text


class BatchFinishMixin:
    def _finish_batch(self) -> None:
        self._batch_timer.stop()
        if self._batch_workers is not None:
            self._batch_workers.shutdown(wait=False)
            self._batch_workers = None
        run = self._batch_run
        run.seconds = time.monotonic() - self._batch_started
        if not run.not_started and not run.write_error:
            run.stopped = self._batch_stopping = False
        if run.frames and not self._batch_map_only:
            self._batch_finishing = True
            self.batch_panel.pause_button.setEnabled(False)
            self.batch_panel.stop_button.setEnabled(False)
            self.cancel_button.hide()
            self.progress_bar.setRange(0, 0)
            self._status(tr("Saving tables and batch record…"))
            self.tasks.submit(
                "batch-finish", lambda: self.view_model.finish_batch(run),
                on_done=lambda _written: self._batch_finish_ui(run),
                on_error=lambda message, _details: self._batch_finish_ui(run, message),
            )
        else:
            self._batch_finish_ui(run)

    def _batch_finish_ui(self, run, tables_error: str = "") -> None:
        self._batch_finishing = False
        destination, done, seconds = run.destination, len(run.frames), run.seconds
        self.batch_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(self.tasks.is_busy())
        self.cancel_button.hide()
        live = self._batch_live
        if self._batch_map_only:
            self._batch_live = False
            self._series_finished(list(run.failures))
            self._batch_panel_finish(done, seconds, live=False)
            self._batch_map_only = False
            if done and not run.failures and not self._batch_stopping:
                self.batch_panel.hide()
            return
        text, level = batch_end_text(run, seconds=seconds, tables_error=tables_error)
        if level == "ok":
            self.notify_written(text, destination)
        else:
            self._status(text, level)
            show_toast(self, end_toast_text(run, stopped=self._batch_stopping, tables_error=tables_error), level=level,
                       action=(tr("Open Folder"), self._batch_open_folder))
        self._batch_live_finish()
        self._series_fit_changed(finished=True)
        self._batch_panel_finish(done, seconds, live=live)
        then, self._batch_then = self._batch_then, None
        if then is not None and done:
            try:
                then(destination / FOLDERS["fit_input"] if run.choices.fit_input else destination)
            except (RuntimeError, ValueError, OSError) as exc:
                self._status(trf("Could not open the series in Fitting: {error}", error=exc), "error")


__all__ = ["BatchFinishMixin"]
