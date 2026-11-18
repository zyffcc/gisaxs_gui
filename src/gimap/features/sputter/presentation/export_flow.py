"""Export of the Analyze result shown, for a paper or a talk (a mixin of ``SputterPage``).

**Export…** (in the main screen's top bar beside *Open Result…*, and in Details beside *Open Report* / *Open
Folder*) asks for a folder and writes ``<scan>_export`` into it with the analysis package (``iampred.export``, in
the background): the evolution and the measured / model patterns as PDF, SVG and PNG at 300 dpi, the solution shown in
each window as a CSV with units in the column names — the user's own choice where they chose another solution —,
every solution of every window, a methods text with the numbers of the run, and a README. When it is done a toast
says where, with *Open Folder*; an older package that cannot export says so.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from .client_flow import Failure, _guarded
from .texts import message_text, text

EXPORT_KEY = "sputter_export"
EXPORT_BUTTON = "Export…"
EXPORT_TIP = ("Write the result shown for a paper or a talk: figures as PDF, SVG and PNG (300 dpi), the solution shown "
              "in each window as a CSV with units, every solution, and a methods text with this run's numbers.")
EXPORT_TITLE = "Choose the Folder for the Export"
EXPORTING = "Exporting {scan}…"
EXPORTED = "Exported {count} files to {folder}"
EXPORT_TEMPLATES = (EXPORT_BUTTON, EXPORT_TIP, EXPORT_TITLE, EXPORTING, EXPORTED)
"""The texts of the export that need Chinese."""
LAST_FOLDER = "export_folder"


class ExportMixin:
    """Needs ``client_result``, ``client_choices``, ``service``, ``tasks``, ``_stored`` / ``_store``, ``_notify``,
    ``export_button`` and ``details_export_button``."""

    def _init_export(self) -> None:
        self._exporting = False
        for button in self._export_buttons():
            button.setToolTip(tr(EXPORT_TIP))
            button.clicked.connect(self.export_result)
        self.sync_export_buttons()

    def _export_buttons(self) -> list:
        return [button for button in (getattr(self, "export_button", None), getattr(self, "details_export_button", None))
                if button is not None]

    def sync_export_buttons(self) -> None:
        """Export is there for an Analyze result with windows, and not while one is written."""
        result = getattr(self, "client_result", None)
        shown = result is not None and bool(result.windows) and not getattr(self, "_exporting", False)
        for button in self._export_buttons():
            button.setEnabled(shown)

    def export_result(self, folder=None) -> None:
        """Export the result shown into ``<folder>/<scan>_export`` (``folder`` None: asked)."""
        result, client = self.client_result, self.service.client
        if result is None or client is None or self._exporting:
            return
        if folder is None:
            start = str(self._stored().get(LAST_FOLDER) or Path(result.folder).parent)
            folder = QFileDialog.getExistingDirectory(self, tr(EXPORT_TITLE), start)
            if not folder:
                return
        stored = self._stored()
        stored[LAST_FOLDER] = str(folder)
        self._store(stored)
        choices = dict(getattr(self, "client_choices", {}) or {})
        self._exporting = True
        self.sync_export_buttons()
        show_toast(self, text(EXPORTING, scan=result.scan or Path(result.folder).name), level="info")
        self.tasks.submit(EXPORT_KEY, lambda: _guarded(lambda: client.export(result, Path(folder), choices)),
                          on_done=self._exported, on_error=lambda message, details: self._export_failed(message, details))

    def _exported(self, files) -> None:
        if isinstance(files, Failure):
            self._export_failed(files.message, files.details)
            return
        self._exporting = False
        self.sync_export_buttons()
        folder = Path(files[0]).parent if files else None
        if folder is None:
            return
        show_toast(self, text(EXPORTED, count=len(files), folder=str(folder)), level="success",
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))

    def _export_failed(self, message: str, details: str = "") -> None:
        self._exporting = False
        self.sync_export_buttons()
        if details and hasattr(self, "client_log"):
            self.client_log.appendPlainText(details)
        self._notify(message_text(message), error=True)


__all__ = ["EXPORT_KEY", "EXPORT_TEMPLATES", "ExportMixin"]
