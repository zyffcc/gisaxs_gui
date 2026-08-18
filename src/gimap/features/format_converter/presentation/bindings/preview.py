"""Preview behavior for Format Converter."""

from __future__ import annotations


from PyQt5.QtCore import QThread

from PyQt5.QtGui import QPixmap


from ..display_formatting import _array_pixmap
from ..workers import _PreviewWorker


class PreviewMixin:
    """Own preview presentation behavior."""

    def _start_preview(self, source) -> None:
        self._preview_request += 1
        request_id = self._preview_request
        self._pending_preview_source = source
        self.preview_stats.setText(f"Loading preview for {source.name}…")
        for label in self.preview_labels:
            label.setText("Loading…")
            label.setPixmap(QPixmap())
        # Do not terminate a loader in native HDF5/Fabio code. Its late result is ignored.
        if self._preview_thread is not None and self._preview_thread.isRunning():
            return
        self._pending_preview_source = None
        self._preview_thread = QThread(self)
        self._preview_worker = _PreviewWorker(request_id, source, self.view_model)
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._preview_ready)
        self._preview_worker.failed.connect(self._preview_failed)
        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.failed.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._preview_cleanup)
        self._preview_thread.start()

    def _preview_ready(self, request_id: int, payload: list[dict]) -> None:
        if request_id != self._preview_request:
            return
        statistics = []
        for index, item in enumerate(payload):
            self.preview_labels[index].setText("")
            self.preview_labels[index].setPixmap(_array_pixmap(item["data"]))
            self.preview_captions[index].setText(f"{item['label']} · frame {item['frame']}")
            if index == 0:
                minimum = "n/a" if item["minimum"] is None else f"{item['minimum']:.6g}"
                maximum = "n/a" if item["maximum"] is None else f"{item['maximum']:.6g}"
                statistics = [
                    f"Image size: {item['shape'][1]} × {item['shape'][0]}",
                    f"Data type: {item['dtype']}",
                    f"Min / max: {minimum} / {maximum}",
                    f"NaN/invalid: {item['nan_count']:,}",
                    f"Negative: {item['negative_count']:,}",
                    f"Pixels at maximum (possible saturation): {item['max_count']:,}",
                ]
        self.preview_stats.setText("\n".join(statistics))

    def _preview_failed(self, request_id: int, message: str) -> None:
        if request_id == self._preview_request:
            self.preview_stats.setText(f"Preview unavailable: {message}")

    def _preview_cleanup(self) -> None:
        self._preview_worker = None
        if self._preview_thread is not None:
            self._preview_thread.deleteLater()
        self._preview_thread = None
        pending = self._pending_preview_source
        self._pending_preview_source = None
        if pending is not None and self.isVisible():
            self._start_preview(pending)
