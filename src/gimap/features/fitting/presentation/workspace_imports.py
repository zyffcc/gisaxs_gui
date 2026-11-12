"""Import selected curve tables into the ordinary batch-fitting workspace."""

from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog

from src.gimap.app.presentation.i18n import tr

from .curve_import import CurveImportDialog
from .single.page import CURVE_FILTER


class CurveImportMixin:
    """Composition of the import preview and the two fitting pages."""

    def import_curves(self, paths=None) -> None:
        if self.series_page.running:
            self.series_page._status(tr("Stop the batch fit before importing curves."), "warning")
            return
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(
                self.context_container, tr("Import Curves"),
                str(self.fit_page._remembered("folder") or ""), CURVE_FILTER)
        if not paths:
            return
        dialog = CurveImportDialog(
            paths, self.view_model.inspect_curve_table, self.context_container,
            q_unit=self.fit_page.unit_setting())
        try:
            if dialog.exec_() != QDialog.Accepted or dialog.import_plan is None:
                return
            plan = dialog.import_plan
            unit = "nm" if plan.q_unit in {"nm", "nm^-1"} else "angstrom"
            if self.series_page.open_imported_sources(
                    plan.sources, independent=plan.independent, q_unit=unit):
                self.fit_page._remember(folder=str(Path(paths[0]).parent))
                self.show_context("insitu")
        finally:
            dialog.dispose()
            dialog.deleteLater()
