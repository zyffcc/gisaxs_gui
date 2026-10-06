"""Files of the XRR window: choosing or dropping a series, exporting the curve with its record."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

SERIES_FILTER = "Detector data (*.nxs *.cbf);;NXS (*.nxs);;CBF (*.cbf)"
SERIES_SUFFIXES = (".nxs", ".cbf")
SOURCE_KIND_AUTO, SOURCE_KIND_NXS, SOURCE_KIND_CBF = 0, 1, 2


def dropped_series(mime) -> str:
    """The first dropped NXS or CBF file, or folder ('' when there is none)."""
    if mime is None or not mime.hasUrls():
        return ""
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.is_dir() or path.suffix.casefold() in SERIES_SUFFIXES:
            return str(path)
    return ""


class XrrFilesMixin:
    """Own the file dialogs, drops and the export of the XRR window."""

    def _start_folder(self) -> str:
        """The folder of the series shown, else the folder the window last read from."""
        current = self.source_picker.path()
        if current:
            location = Path(current)
            folder = location if location.is_dir() else location.parent
            if folder.is_dir():
                return str(folder)
        getter = getattr(self.view_model, "last_folder", None)
        return getter() if callable(getter) else ""

    def _remember_folder(self, path) -> None:
        remember = getattr(self.view_model, "remember_folder", None)
        if callable(remember):
            remember(path)

    def _browse_source(self) -> None:
        start = self._start_folder()
        if self.source_kind_combo.currentIndex() == SOURCE_KIND_CBF:
            path = QFileDialog.getExistingDirectory(self, tr("Select CBF series folder"), start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, tr("Select detector series"), start, SERIES_FILTER)
        if path:
            self.source_picker.set_path(path)
            self._remember_folder(path)

    def _busy(self) -> bool:
        return self._thread_running(self._inspect_thread) or self._thread_running(self._extract_thread)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self._busy() and dropped_series(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        path = dropped_series(event.mimeData())
        if not path or self._busy():
            event.ignore()
            return
        event.acceptProposedAction()
        self.use_series(path)

    def use_series(self, path: str) -> None:
        """A dropped series: as if chosen with Browse…, and its first frame is loaded at once."""
        location = Path(path)
        kind = self.source_kind_combo.currentIndex()
        if location.is_dir() and kind == SOURCE_KIND_NXS:
            self.source_kind_combo.setCurrentIndex(SOURCE_KIND_CBF)
        elif location.suffix.casefold() == ".nxs" and kind == SOURCE_KIND_CBF:
            self.source_kind_combo.setCurrentIndex(SOURCE_KIND_NXS)
        elif location.suffix.casefold() == ".cbf" and kind == SOURCE_KIND_NXS:
            self.source_kind_combo.setCurrentIndex(SOURCE_KIND_AUTO)
        self.source_picker.set_path(str(location))
        self._remember_folder(location)
        self._inspect_series()

    def _default_export_path(self) -> str:
        getter = getattr(self.view_model, "default_export_path", None)
        default = getter() if callable(getter) else None
        if default is None:
            folder = self._start_folder()
            default = Path(folder) / "xrr_curve.csv" if folder else Path("xrr_curve.csv")
        return str(default)

    def _export_curve(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Export XRR points"), self._default_export_path(), "CSV (*.csv)"
        )
        if not path:
            return
        name = Path(path).name
        try:
            exported = self.view_model.export(
                Path(path), geometry_sources=self._run_geometry_sources, calibration=self._calibration
            )
        except Exception as exc:
            self.job_status.set_state("failed", str(exc), progress=0.0)
            show_toast(
                self, trf("{name} could not be exported: {error}", name=name, error=exc), level="error"
            )
            return
        record = getattr(exported, "record_path", None)
        self.job_status.set_state("succeeded", trf("Exported {name}", name=name), progress=1.0)
        text = (
            trf("Exported {name} and its settings record {record}", name=name, record=Path(record).name)
            if record is not None
            else trf("Exported {name}", name=name)
        )
        folder = Path(path).parent
        show_toast(
            self,
            text,
            level="ok",
            action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))),
        )


__all__ = ["XrrFilesMixin", "dropped_series"]
