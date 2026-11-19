"""Export… of samples measured after their deposition: the report of every series of the list (a mixin of
``SputterPage``).

While the main screen shows a series of static samples, **Export…** (beside Open Result…, and in Details) asks for a
folder and writes ``samples_report`` into it in the background: ``report.html`` — every series of the sample list (and
the series shown, if it is not in the list) with its table of values, the stage of each sample, and the package's two
figures of the series, in one file — and ``samples.csv`` with every sample of every series. The words are those of the
interface language. A toast says where, with Open Report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.shared.version import __version__

from ..application import read_series, report_html, series_of, write_csv
from .client_flow import Failure, _guarded
from .texts import text

STATIC_EXPORT_KEY = "sputter_static_export"
REPORT_FOLDER = "samples_report"
STATIC_EXPORT_TIP = ("Write the report of every series of the sample list: one HTML file with each series' values, the "
                     "stage of each sample and the figures of the fits, and a CSV with every sample.")
STATIC_EXPORT_TITLE = "Choose the Folder for the Report"
STATIC_EXPORTING = "Writing the report of {series} series…"
STATIC_EXPORTED = "Report of {series} series written to {folder}"
NOTHING_TO_REPORT = "No series of the list has analysed samples yet: analyse them first."
OPEN_REPORT = "Open Report"
STATIC_EXPORT_TEMPLATES = (STATIC_EXPORT_TIP, STATIC_EXPORT_TITLE, STATIC_EXPORTING, STATIC_EXPORTED,
                           NOTHING_TO_REPORT, OPEN_REPORT)
"""The texts of the report's export that need Chinese."""


def _write(series: list, folder: Path, facts: dict) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "report.html"
    page.write_text(report_html(series, lambda template, **values: text(template, **values), facts),
                    encoding="utf-8")
    return [page, write_csv(series, folder / "samples.csv")]


class StaticExportMixin:
    """Needs the sample list (``static_list.py``), ``client_result``, ``tasks`` and the export buttons."""

    def static_report_series(self) -> list:
        """(name, folder, bare substrate) of each series to report: those of the list, then the one shown."""
        found = []
        for name, samples in series_of(self.static_samples):
            bare = Path(samples[0].reference).name if samples and samples[0].reference else ""
            found.append((name, self.series_folder(name), bare))
        shown = self.client_result
        if shown is not None and shown.static and all(Path(folder) != Path(shown.folder) for _n, folder, _b in found):
            found.append((Path(shown.folder).name, Path(shown.folder), ""))
        return found

    def export_static_report(self, folder=None) -> None:
        """Write the report into ``<folder>/samples_report`` (``folder`` None: asked)."""
        if getattr(self, "_exporting", False):
            return
        wanted = [(name, path, bare) for name, path, bare in self.static_report_series()
                  if (Path(path) / "report.csv").is_file()]
        if not wanted:
            self._notify(tr(NOTHING_TO_REPORT), error=True)
            return
        if folder is None:
            start = str(self._stored().get("export_folder") or self.static_results_folder())
            folder = QFileDialog.getExistingDirectory(self, tr(STATIC_EXPORT_TITLE), start)
            if not folder:
                return
        stored = self._stored()
        stored["export_folder"] = str(folder)
        self._store(stored)
        client = self.service.client
        package = client.package().identity[:12] if client is not None else ""
        facts = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "version": __version__, "package": package or "?",
                 "geometry": self.geometry_summary.full_text() if hasattr(self.geometry_summary, "full_text")
                 else self.geometry_summary.text(), "mode": self.static_mode.currentText()}
        target = Path(folder) / REPORT_FOLDER
        self._exporting = True
        self.sync_export_buttons()
        show_toast(self, text(STATIC_EXPORTING, series=len(wanted)), level="info")

        def work():
            series = [one for one in (read_series(name, path, bare) for name, path, bare in wanted) if one is not None]
            return _write(series, target, facts)

        self.tasks.submit(STATIC_EXPORT_KEY, lambda: _guarded(work),
                          on_done=lambda files: self._static_exported(files, len(wanted)),
                          on_error=lambda message, details: self._export_failed(message, details))

    def _static_exported(self, files, count: int) -> None:
        if isinstance(files, Failure):
            self._export_failed(files.message, files.details)
            return
        self._exporting = False
        self.sync_export_buttons()
        page = Path(files[0])
        show_toast(self, text(STATIC_EXPORTED, series=count, folder=str(page.parent)), level="success",
                   action=(tr(OPEN_REPORT), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))))


__all__ = ["REPORT_FOLDER", "STATIC_EXPORT_KEY", "STATIC_EXPORT_TEMPLATES", "StaticExportMixin"]
