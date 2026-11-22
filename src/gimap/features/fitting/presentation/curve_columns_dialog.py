"""Choose Columns: a CSV whose header does not say which columns are q, I and σ (``COLUMNS_TO_CHOOSE``) is shown as
Import Curves shows a table — its header and first rows — and the person chooses q, I, σ (or none) and the unit of q.

A choice is checked as Import Curves checks its own (``curve_sources.validate_columns``: three different columns, a
named uncertainty column only as σ); the unit is the one the q column's name states, when it states one (the choice
is then fixed), else the one chosen here. The curve opens with these columns (a source token with a column
mapping): the record of a fit says which were read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTableWidgetItem

from src.gimap.app.presentation.i18n import apply_to, current_language, tr, trf

from ..application.curve_sources import ColumnMapping, CurveTablePreview, validate_columns
from ..application.models import COLUMNS_TO_CHOOSE, InSituSourceFrame
from .views.curve_columns_view import CurveColumnsView

NO_SIGMA = "none (no σ)"
SAME_COLUMN = "q, I and σ must be three different columns."
STATED = "Stated in the name of the q column."


class CurveColumnsDialog(QDialog, CurveColumnsView):
    """``stated_unit(mapping)``: the unit of q the file states for that choice of columns (``None``: none)."""

    def __init__(self, preview: CurveTablePreview, unit: str, stated_unit: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.preview, self._stated_unit, self._setting = preview, stated_unit, unit
        self.setup_curve_columns_view(self)
        apply_to(self, current_language())
        names = [name if preview.has_header else trf("Column {n}", n=index + 1)
                 for index, name in enumerate(preview.column_names)]
        self.columns_file_label.setText(str(preview.path))
        self.columns_note.setText(tr(COLUMNS_TO_CHOOSE))
        for index, name in enumerate(names):
            if index in preview.resolution_columns:
                continue
            for combo in (self.q_column_combo, self.intensity_column_combo, self.sigma_column_combo):
                combo.addItem(name, index)
        self.sigma_column_combo.insertItem(0, tr(NO_SIGMA), None)
        q_column = preview.q_columns[0] if preview.q_columns else 0
        self.q_column_combo.setCurrentIndex(max(0, self.q_column_combo.findData(q_column)))
        intensity = preview.intensity_columns[0] if preview.intensity_columns else min(1, len(names) - 1)
        self.intensity_column_combo.setCurrentIndex(max(0, self.intensity_column_combo.findData(intensity)))
        sigma = preview.uncertainty_columns[0] if preview.uncertainty_columns else None
        self.sigma_column_combo.setCurrentIndex(max(0, self.sigma_column_combo.findData(sigma)))
        self.columns_unit_combo.setCurrentIndex(max(0, self.columns_unit_combo.findData(unit)))
        self._fill_preview(names)
        for combo in (self.q_column_combo, self.intensity_column_combo, self.sigma_column_combo):
            combo.currentIndexChanged.connect(lambda _index: self._check())
        self.columns_buttons.accepted.connect(self.accept)
        self.columns_buttons.rejected.connect(self.reject)
        self._check()

    def _fill_preview(self, names) -> None:
        table = self.columns_preview
        table.setColumnCount(len(names))
        table.setHorizontalHeaderLabels(names)
        rows = self.preview.sample_rows[:10]
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values[:len(names)]):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    def mapping(self) -> Optional[ColumnMapping]:
        """The columns chosen, ``None`` when they are no curve (said under the choices)."""
        q, intensity = self.q_column_combo.currentData(), self.intensity_column_combo.currentData()
        sigma = self.sigma_column_combo.currentData()
        try:
            dq = self.preview.resolution_columns
            if len(dq) > 1:
                raise ValueError("Each curve must have one resolution column.")
            mapping = ColumnMapping(q, intensity, sigma, self.preview.column_names if self.preview.has_header else (),
                                    self.preview.column_count, dq[0] if dq else None)
            validate_columns(self.preview, mapping)
        except ValueError as exc:
            self.columns_problem.setText(tr(SAME_COLUMN) if len({q, intensity, sigma} - {None}) <
                                         2 + (sigma is not None) else tr(str(exc)))
            return None
        self.columns_problem.setText("")
        return mapping

    def _check(self) -> None:
        mapping = self.mapping()
        self.columns_buttons.button(QDialogButtonBox.Ok).setEnabled(mapping is not None)
        stated = None
        if mapping is not None and self._stated_unit is not None:
            try:
                stated = self._stated_unit(mapping)
            except (OSError, ValueError):
                stated = None
        if stated is not None:
            self.columns_unit_combo.setCurrentIndex(self.columns_unit_combo.findData(stated))
        self.columns_unit_combo.setEnabled(stated is None)
        self.columns_unit_combo.setToolTip(tr(STATED) if stated is not None else "")

    def chosen(self) -> Optional[tuple]:
        mapping = self.mapping()
        return None if mapping is None else (mapping, self.columns_unit_combo.currentData())


def choose_columns(parent, path, inspect: Callable, unit: str, stated_unit: Optional[Callable] = None) -> Optional[tuple]:
    """Ask for the columns of the table at ``path``: ``(ColumnMapping, unit of q)``, or ``None`` (cancelled, or the
    table cannot be read)."""
    try:
        preview = inspect(Path(str(path)))
    except (OSError, ValueError):
        return None
    if stated_unit is None and hasattr(parent, "view_model"):
        source = Path(str(path))
        stated_unit = lambda mapping: parent.view_model.stated_q_unit(  # noqa: E731
            InSituSourceFrame(source, columns=mapping).token)
    dialog = CurveColumnsDialog(preview, unit, stated_unit, parent)
    try:
        return dialog.chosen() if dialog.exec_() == QDialog.Accepted else None
    finally:
        dialog.deleteLater()


__all__ = ["NO_SIGMA", "SAME_COLUMN", "STATED", "CurveColumnsDialog", "choose_columns"]
