"""Preview behavior for Format Converter.

The statistics come first; below them one thumbnail per distinct frame (one for a single image,
first, middle and last for a series). Thumbnails are display only: log scale, viridis; the
statistics are the stored values.
"""

from __future__ import annotations


from PyQt5.QtCore import QThread

from PyQt5.QtGui import QPixmap

from src.gimap.app.presentation.i18n import tr, trf

from ..display_formatting import _array_pixmap
from ..workers import _PreviewWorker

THUMBNAIL_MIN_WIDTH = 200
THUMBNAIL_MAX_WIDTH = 520
CAPTIONS = {
    "First": "First · frame {frame}",
    "Middle": "Middle · frame {frame}",
    "Last": "Last · frame {frame}",
}
THUMBNAIL_TIP = (
    "Display only: log(1 + value) of the values ≥ 0, viridis colour map. "
    "The statistics above are the stored values."
)


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def preview_statistics(item: dict, *, several: bool) -> list[str]:
    lines = [trf("Statistics of frame {frame}:", frame=item["frame"])] if several else []
    lines += [
        trf("Image size: {width} × {height}", width=item["shape"][1], height=item["shape"][0]),
        trf("Data type: {dtype}", dtype=item["dtype"]),
        trf("Min / max: {minimum} / {maximum}", minimum=_number(item["minimum"]), maximum=_number(item["maximum"])),
        trf("NaN/invalid: {count}", count=f"{item['nan_count']:,}"),
        trf("Negative: {count}", count=f"{item['negative_count']:,}"),
        trf("Pixels at maximum (possible saturation): {count}", count=f"{item['max_count']:,}"),
    ]
    return lines


def preview_caption(item: dict) -> str:
    template = CAPTIONS.get(item.get("label") or "")
    return trf(template, frame=item["frame"]) if template else trf("Frame {frame}", frame=item["frame"])


class PreviewMixin:
    """Own preview presentation behavior."""

    def _start_preview(self, source) -> None:
        self._preview_request += 1
        request_id = self._preview_request
        self._pending_preview_source = source
        self.preview_stats.setText(trf("Loading preview for {name}…", name=source.name))
        for label in self.preview_labels:
            label.setText(tr("Loading…"))
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

    def _thumbnail_width(self) -> int:
        available = self.preview_scroll.viewport().width() - 56
        return max(THUMBNAIL_MIN_WIDTH, min(THUMBNAIL_MAX_WIDTH, available))

    def _preview_ready(self, request_id: int, payload: list[dict]) -> None:
        if request_id != self._preview_request:
            return
        width = self._thumbnail_width()
        for index, (caption, label) in enumerate(zip(self.preview_captions, self.preview_labels)):
            shown = index < len(payload)
            caption.setVisible(shown)
            label.setVisible(shown)
            if not shown:
                continue
            item = payload[index]
            label.setText("")
            label.setPixmap(_array_pixmap(item["data"], width, int(width * 0.85)))
            label.setToolTip(tr(THUMBNAIL_TIP))
            caption.setText(preview_caption(item))
        statistics = preview_statistics(payload[0], several=len(payload) > 1) if payload else []
        self.preview_stats.setText("\n".join(statistics))

    def _preview_failed(self, request_id: int, message: str) -> None:
        if request_id == self._preview_request:
            self.preview_stats.setText(trf("Preview unavailable: {message}", message=message))

    def _preview_cleanup(self) -> None:
        self._preview_worker = None
        if self._preview_thread is not None:
            self._preview_thread.deleteLater()
        self._preview_thread = None
        pending = self._pending_preview_source
        self._pending_preview_source = None
        if pending is not None and self.isVisible():
            self._start_preview(pending)
