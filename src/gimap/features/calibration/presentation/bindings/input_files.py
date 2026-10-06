"""Where the file dialogs of Geometry Calibration start, files dropped on it, and its toasts."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

# What the shared detector loader reads (the Open… filter), and an exported calibration.
IMAGE_SUFFIXES = (".nxs", ".cbf", ".tif", ".tiff", ".edf")
CALIBRATION_SUFFIX = ".json"


def dropped_calibration_file(mime) -> str:
    """The first dropped detector image or calibration JSON ('' when there is none)."""
    if mime is None or not mime.hasUrls():
        return ""
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.is_file() and path.suffix.casefold() in (*IMAGE_SUFFIXES, CALIBRATION_SUFFIX):
            return str(path)
    return ""


class CalibrationFilesMixin:
    """Own the start folder of the file dialogs, drag-and-drop and the window's toasts."""

    def _start_folder(self) -> str:
        """The folder of the image shown, else the folder the last image was read from."""
        current = self.path_edit.text().strip().strip('"')
        if current:
            folder = Path(current).parent
            if str(folder) not in ("", ".") and folder.is_dir():
                return str(folder)
        getter = getattr(self.view_model, "last_folder", None)
        return getter() if callable(getter) else ""

    def _remember_folder(self, path) -> None:
        remember = getattr(self.view_model, "remember_folder", None)
        if callable(remember) and path:
            remember(path)

    def _calibration_busy(self) -> bool:
        return any(
            thread is not None and thread.isRunning() for thread in (self._load_thread, self._cal_thread)
        )

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self._calibration_busy() and dropped_calibration_file(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        path = dropped_calibration_file(event.mimeData())
        if not path or self._calibration_busy():
            event.ignore()
            return
        event.acceptProposedAction()
        if path.casefold().endswith(CALIBRATION_SUFFIX):
            self.import_result_from(path)
        else:
            self.load_image(self.view_model.normalize_path(path))

    def _toast_host(self):
        """Toasts float over the preview and results, never over Apply and Close."""
        return getattr(self, "right", None) or self

    def _show_saved_toast(self, text: str, path) -> None:
        folder = Path(path).parent
        show_toast(
            self._toast_host(),
            text,
            level="ok",
            action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))),
        )


__all__ = ["CalibrationFilesMixin", "dropped_calibration_file"]
