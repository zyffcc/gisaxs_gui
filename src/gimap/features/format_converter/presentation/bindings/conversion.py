"""Conversion behavior for Format Converter."""

from __future__ import annotations

import time


from PyQt5.QtCore import QThread


from ..display_formatting import _duration
from ..progress_dialog import ConversionProgressDialog
from ..workers import _ConversionWorker


class ConversionMixin:
    """Own conversion presentation behavior."""

    def _start_conversion(self, options) -> None:
        destination = self.view_model.normalize_path(options.destination)
        self._progress_dialog = ConversionProgressDialog(destination, self.parent())
        self._conversion_thread = QThread(self)
        self._conversion_worker = _ConversionWorker(options, self.view_model)
        self._conversion_worker.moveToThread(self._conversion_thread)
        self._conversion_thread.started.connect(self._conversion_worker.run)
        self._conversion_worker.progress.connect(self._conversion_progress)
        self._conversion_worker.finished.connect(self._conversion_finished)
        self._conversion_worker.failed.connect(self._conversion_failed)
        self._conversion_worker.finished.connect(self._conversion_thread.quit)
        self._conversion_worker.failed.connect(self._conversion_thread.quit)
        self._conversion_thread.finished.connect(self._conversion_cleanup)
        self._progress_dialog.pause_button.clicked.connect(self._toggle_pause)
        self._progress_dialog.cancel_button.clicked.connect(self._cancel_conversion)
        self._conversion_started_at = time.time()
        self._progress_dialog.show()
        self.hide()
        self._conversion_thread.start()

    def _conversion_progress(
        self, completed: int, total: int, source_name: str, frame_index: int
    ) -> None:
        if self._progress_dialog is None:
            return
        total = max(1, total)
        self._progress_dialog.bar.setRange(0, total)
        self._progress_dialog.bar.setValue(completed)
        self._progress_dialog.title.setText(f"Converting {source_name}")
        self._progress_dialog.detail.setText(f"Frame {frame_index + 1} · {completed} / {total}")
        self._progress_dialog.job_status.set_state(
            "running",
            f"{source_name} · frame {frame_index + 1} · {completed} / {total}",
            progress=completed / total,
        )
        elapsed = time.time() - self._conversion_started_at
        remaining = (elapsed / completed * (total - completed)) if completed else 0
        remaining_text = _duration(remaining) if completed else "calculating…"
        self._progress_dialog.time_label.setText(
            f"Elapsed: {_duration(elapsed)}    Remaining: approximately {remaining_text}"
        )

    def _toggle_pause(self) -> None:
        if self._conversion_worker is None or self._progress_dialog is None:
            return
        self._paused = not self._paused
        self._conversion_worker.set_paused(self._paused)
        self._progress_dialog.pause_button.setText("Resume" if self._paused else "Pause")
        if self._paused:
            self._progress_dialog.title.setText("Conversion paused")
            self._progress_dialog.job_status.set_state(
                "paused",
                "Conversion paused",
                progress=self._progress_dialog.bar.value()
                / max(1, self._progress_dialog.bar.maximum()),
            )
        else:
            self._progress_dialog.job_status.set_state(
                "running",
                "Conversion resumed",
                progress=self._progress_dialog.bar.value()
                / max(1, self._progress_dialog.bar.maximum()),
            )

    def _cancel_conversion(self) -> None:
        if self._conversion_worker is not None:
            self._conversion_worker.cancel()
        if self._progress_dialog is not None:
            self._progress_dialog.title.setText("Cancelling after the current image…")
            self._progress_dialog.job_status.set_state(
                "running",
                "Cancelling after the current image…",
                progress=self._progress_dialog.bar.value()
                / max(1, self._progress_dialog.bar.maximum()),
            )
            self._progress_dialog.cancel_button.setEnabled(False)

    def _conversion_finished(self, report) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.complete(report)

    def _conversion_failed(self, message: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.fail(message)

    def _conversion_cleanup(self) -> None:
        self._conversion_worker = None
        if self._conversion_thread is not None:
            self._conversion_thread.deleteLater()
        self._conversion_thread = None
        self.close()

    def closeEvent(self, event) -> None:
        if self._conversion_thread is not None and self._conversion_thread.isRunning():
            event.ignore()
            return
        # A native file read cannot be killed safely. Keep this dialog alive until it returns.
        if self._preview_thread is not None and self._preview_thread.isRunning():
            self.hide()
            self._preview_thread.finished.connect(self.close)
            event.ignore()
            return
        event.accept()
