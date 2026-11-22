"""Behaviour of the Phases window: edit the phases, show their reflections, fit the pole figure of a ring.

The phases are the page's ``IndexingModel``; every change draws the overlay again (``page.indexing_changed``).
The pole figure is fitted on the task runner and drawn here: the folded (I(χ) − b) sin χ of the ring in
Analyze's I(χ) window (points, unmeasured bins left out) and the fitted distribution (line)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import QCheckBox, QDialog, QFileDialog, QTableWidgetItem, QWidget

from src.gimap.app.presentation.components.toast import show_toast
from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.theme import theme_color, theme_manager

from ..application.export_record import export_stem
from ..application.indexing import POWDER, PhaseSpec, table_rows
from ..application.pole_figure import pole_figure
from .crystal_orientation_panel import CrystalOrientationMixin
from .indexing_export import write_crystal_export, write_pole_export
from .indexing_texts import (
    MODEL_NAMES,
    orientation_text,
    phase_cell_text,
    phase_notes,
    pole_result_text,
    ring_text,
    symmetry_text,
)
from .views.indexing_view import IndexingView

POLE_POINTS, POLE_LEFT_OUT, POLE_MODEL = "#2563eb", "#9ca3af", "#f97316"


def _item(text: str, data=None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if data is not None:
        item.setData(Qt.UserRole, data)
    return item


def _number(value, digits: int) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


class IndexingPanel(QDialog, IndexingView, CrystalOrientationMixin):
    def __init__(self, page: QWidget):
        super().__init__(page)
        self.setWindowFlag(Qt.Tool, True)
        self._page = page
        self._filling = False
        self._rows: list[dict] = []
        self.setup_ui(self)
        self.buttons.rejected.connect(self.close)
        self.add_cif_button.clicked.connect(self._add_cif)
        self.add_cell_button.clicked.connect(self._add_cell)
        self.remove_button.clicked.connect(self._remove)
        self.phase_table.itemSelectionChanged.connect(self._fill_editor)
        self.name_edit.editingFinished.connect(self._edited)
        for spin in (*self.cell_spins, *self.angle_spins, *self.index_spins):
            spin.valueChanged.connect(lambda _value: self._edited())
        self.centring_combo.currentIndexChanged.connect(lambda _index: self._edited())
        self.orientation_combo.currentIndexChanged.connect(lambda _index: self._edited())
        for widget in (self.q_max_spin, self.critical_spin):
            widget.valueChanged.connect(lambda _value: self._settings_edited())
        for check in (self.refraction_check, self.reflected_check, self.labels_check):
            check.toggled.connect(lambda _on: self._settings_edited())
        self.use_reflection_button.clicked.connect(self._use_reflection)
        self.fit_button.clicked.connect(self.fit_pole)
        self.export_button.clicked.connect(self._export)
        self.background_check.toggled.connect(lambda on: self.background_spin.setEnabled(not on))
        self.direction_combo.currentIndexChanged.connect(self._direction_chosen)
        self._direction_chosen()
        self._displayed_pole = None
        self._setup_crystal_orientation()
        language_changed().connect(self._refresh_language)
        theme_manager().changed.connect(self._theme_changed)  # the warning colour of a phase
        self._fill_settings()

    def _refresh_language(self, *_args) -> None:
        self.refresh()
        if self._crystal_mode():
            self._show_crystal_ring()
        elif self._displayed_pole is not None:
            self._show_pole(self._displayed_pole)

    def _theme_changed(self, *_args) -> None:
        self.refresh()

    @property
    def model(self):
        return self._page.indexing

    # -- phases ---------------------------------------------------------------------------

    def _add_cif(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, tr("Add Phase from CIF"), "", tr("CIF files (*.cif);;All files (*)"))
        if not path:
            return
        try:
            index = self.model.add_cif(path)
        except (OSError, ValueError) as error:
            self.status_label.setText(trf("Could not read the CIF: {error}", error=str(error)))
            return
        self._changed(select=index)
        spec = self.model.phases[index]
        read = trf("Read {name} from {file}.", name=spec.name, file=Path(path).name)
        self.status_label.setText(read + (" ⚠ " + tr("See the Symmetry column.") if spec.warnings else ""))

    def _add_cell(self) -> None:
        index = self.model.add(PhaseSpec(trf("Phase {n}", n=len(self.model.phases) + 1),
                                         (5.0, 5.0, 5.0, 90.0, 90.0, 90.0), "P", orientation=POWDER))
        self._changed(select=index)

    def _remove(self) -> None:
        index = self._selected()
        if index is None:
            return
        self.model.remove(index)
        self._changed(select=min(index, len(self.model.phases) - 1))

    def _selected(self) -> Optional[int]:
        rows = self.phase_table.selectionModel().selectedRows() if self.phase_table.selectionModel() else []
        return rows[0].row() if rows and rows[0].row() < len(self.model.phases) else None

    def _fill_editor(self) -> None:
        index = self._selected()
        enabled = index is not None
        for widget in (self.name_edit, *self.cell_spins, *self.angle_spins, self.centring_combo,
                       self.orientation_combo, *self.index_spins, self.remove_button):
            widget.setEnabled(enabled)
        if index is None:
            self.symmetry_label.setText("")
            return
        spec = self.model.phases[index]
        self._filling = True
        try:
            self.name_edit.setText(spec.name)
            for spin, value in zip((*self.cell_spins, *self.angle_spins), spec.cell):
                spin.setValue(float(value))
            self.centring_combo.setCurrentText(spec.centring)
            self.centring_combo.setEnabled(not spec.operations and not spec.symmetry)  # a typed cell only
            self.orientation_combo.setCurrentIndex(max(0, self.orientation_combo.findData(spec.orientation)))
            for spin, value in zip(self.index_spins, spec.indices):
                spin.setValue(int(value))
                spin.setEnabled(spec.orientation != POWDER)
            notes = phase_notes(spec)
            self.symmetry_label.setText(symmetry_text(spec) + ("\n" + notes if notes else ""))
        finally:
            self._filling = False

    def _edited(self) -> None:
        index = self._selected()
        if self._filling or index is None:
            return
        indices = tuple(spin.value() for spin in self.index_spins)
        orientation = self.orientation_combo.currentData()
        if orientation != POWDER and not any(indices):
            self.status_label.setText(tr("The orientation needs indices other than 0 0 0."))
            return
        cell = tuple(spin.value() for spin in (*self.cell_spins, *self.angle_spins))
        try:
            PhaseSpec("check", cell).phase()
        except ValueError as error:
            self.status_label.setText(tr(str(error)))
            return
        self.model.replace(index, name=self.name_edit.text().strip() or self.model.phases[index].name, cell=cell,
                           centring=self.centring_combo.currentText(), orientation=orientation, indices=indices)
        for spin in self.index_spins:
            spin.setEnabled(orientation != POWDER)
        self._changed(select=index)

    def _visibility(self, index: int, visible: bool) -> None:
        self.model.replace(index, visible=bool(visible))
        self._page.indexing_changed()

    def _changed(self, select: Optional[int] = None) -> None:
        self._page.indexing_changed()
        if select is not None and 0 <= select < self.phase_table.rowCount():
            self.phase_table.selectRow(select)

    # -- settings -------------------------------------------------------------------------

    def _fill_settings(self) -> None:
        settings = self.model.settings
        self._filling = True
        try:
            self.q_max_spin.setValue(settings.q_max)
            self.critical_spin.setValue(settings.critical_angle_deg)
            self.refraction_check.setChecked(settings.refraction)
            self.reflected_check.setChecked(settings.reflected)
            self.labels_check.setChecked(settings.label_fraction > 0)
        finally:
            self._filling = False

    def _settings_edited(self) -> None:
        if self._filling:
            return
        self.reflected_check.setEnabled(self.refraction_check.isChecked())
        self.critical_spin.setEnabled(self.refraction_check.isChecked())
        self.model.set_settings(
            q_max=self.q_max_spin.value(), refraction=self.refraction_check.isChecked(),
            critical_angle_deg=self.critical_spin.value(), reflected=self.reflected_check.isChecked(),
            label_fraction=0.05 if self.labels_check.isChecked() else 0.0,
        )
        self._page.indexing_changed()

    # -- tables ---------------------------------------------------------------------------

    def refresh(self) -> None:
        """The phase list, the reflections and the ring of I(χ) as they are now."""
        selected = self._selected()
        with QSignalBlocker(self.phase_table):
            self.phase_table.setRowCount(len(self.model.phases))
            for row, spec in enumerate(self.model.phases):
                check = QCheckBox(self.phase_table)
                check.setChecked(spec.visible)
                check.toggled.connect(lambda on, index=row: self._visibility(index, on))
                self.phase_table.setCellWidget(row, 0, check)
                name = _item(spec.name)
                name.setForeground(_color(spec.color))
                if spec.formula:
                    name.setToolTip(trf("Formula: {formula}", formula=spec.formula))
                self.phase_table.setItem(row, 1, name)
                self.phase_table.setItem(row, 2, _item(phase_cell_text(spec)))
                symmetry = _item(symmetry_text(spec))
                notes = phase_notes(spec)
                symmetry.setToolTip(symmetry.text() + ("\n" + notes if notes else ""))
                if spec.warnings:  # kept while the phase is listed, not a status line the next action replaces
                    symmetry.setForeground(_color(theme_color("warning")))
                self.phase_table.setItem(row, 3, symmetry)
                self.phase_table.setItem(row, 4, _item(orientation_text(spec)))
        if selected is not None and selected < len(self.model.phases):
            self.phase_table.selectRow(selected)
        elif self.model.phases and selected is None:
            self.phase_table.selectRow(0)
        self._fill_editor()
        self._fill_reflections()
        self.ring_label.setText(ring_text(self._page.view_model.state.analysis))
        self._refresh_crystal_options()
        self.reflected_check.setEnabled(self.refraction_check.isChecked())
        self.critical_spin.setEnabled(self.refraction_check.isChecked())

    def _fill_reflections(self) -> None:
        overlays = self._page.indexing_overlays()
        rows = []
        for overlay in overlays:
            rows += table_rows(overlay)
        self._rows = rows
        table = self.reflection_table
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (row["phase"], row["hkl"], _number(row["d_A"], 4), _number(row["q"], 4),
                      _number(row["q_parallel"], 4), _number(row["qz"], 4), _number(row["chi_deg"], 1),
                      str(row["multiplicity"]), f"{row['f2']:.4g}", _number(row["intensity"], 1))
            for column, text in enumerate(values):
                table.setItem(index, column, _item(text, index if column == 0 else None))
        if not overlays and self.model.phases:
            self.status_label.setText(tr("The overlay needs a GIWAXS frame with a geometry."))

    # -- pole figure ----------------------------------------------------------------------

    def _direction_chosen(self, *_args) -> None:
        self.direction_spin.setEnabled(self.direction_combo.currentData() == "set")

    def _direction(self) -> Optional[float]:
        key = self.direction_combo.currentData()
        return {"normal": 0.0, "plane": 90.0, "fitted": None}.get(key, self.direction_spin.value())

    def _use_reflection(self) -> None:
        rows = self.reflection_table.selectionModel().selectedRows() if self.reflection_table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._rows):
            self.status_label.setText(tr("Select a reflection in the table first."))
            return
        row = self._rows[rows[0].row()]
        half = max(0.01, 0.015 * row["q"])
        if row.get("chi_deg") is not None:
            self.direction_combo.setCurrentIndex(self.direction_combo.findData("set"))
            self.direction_spin.setValue(float(row["chi_deg"]))
        self.tabs.setCurrentIndex(2)
        page = self._page
        page.view_model.set_chi_window(row["q"] - half, row["q"] + half)
        page._after_indexing_analysis = self.fit_pole
        page.run_analysis()

    def fit_pole(self) -> None:
        """Fit the pole figure of the ring in Analyze's I(χ) window (on the task runner)."""
        if self._crystal_mode():
            self._fit_crystal_orientation()
            return
        analysis = self._page.view_model.state.analysis
        background = None if self.background_check.isChecked() else self.background_spin.value()
        model, direction = self.model_combo.currentData(), self._direction()
        chi_max = self.chi_max_spin.value()
        self.result_label.setText(tr("Fitting the pole figure …"))
        self._page.tasks.submit(
            "pole_figure",
            lambda: pole_figure(analysis, model=model, chi0_deg=direction, background=background, chi_max=chi_max),
            on_done=self._pole_ready,
            on_error=lambda message, _details: self.result_label.setText(tr(str(message))),
        )

    def _pole_ready(self, result) -> None:
        self.model.last_pole = result
        self._displayed_pole = result
        if not self._crystal_mode():
            self._show_pole(result)

    def _show_pole(self, result) -> None:
        self.pole_plot.set_labels("χ (°)", tr("N(χ) ∝ crystallites per degree"))
        fit, profile = result.fit, result.fit.profile
        curves, colors, markers = [], [], []
        if profile is not None:
            shown = profile.measured & np.isfinite(profile.value)
            fitted = shown & (profile.chi <= result.chi_max)
            curves.append((tr("Measured N(χ)"), profile.chi[fitted], profile.value[fitted]))
            colors.append(POLE_POINTS)
            markers.append(True)
            if (shown & ~fitted).any():
                curves.append((tr("not fitted"), profile.chi[shown & ~fitted], profile.value[shown & ~fitted]))
                colors.append(POLE_LEFT_OUT)
                markers.append(True)
        if fit.ok and fit.curve_chi.size:
            curves.append((tr(MODEL_NAMES.get(fit.model, fit.model)), fit.curve_chi, fit.curve_value))
            colors.append(POLE_MODEL)
            markers.append(False)
        self.pole_plot.set_curves(curves, colors, markers=markers)
        low, high = result.window
        self.pole_plot.set_title(trf("Pole figure, q = {low}–{high} Å⁻¹", low=f"{low:.3f}", high=f"{high:.3f}"))
        self.result_label.setText(pole_result_text(result))
        if not result.fit.ok:
            return
        self.background_spin.setValue(float(result.background))

    def _export(self) -> None:
        analysis = self._page.view_model.state.analysis
        result = self.model.last_pole
        crystal = self._crystal_mode()
        ring = self._selected_crystal_ring() if crystal else None
        if analysis is None or (crystal and ring is None) or (not crystal and result is None):
            self.status_label.setText(tr("Fit a pole figure first."))
            return
        folder = self._page.view_model.default_export_dir() or Path.home()
        suffix = "_crystal_orientation" if crystal else "_pole_figure"
        suggested = str(folder / f"{export_stem(analysis)}{suffix}.csv")
        path, _filter = QFileDialog.getSaveFileName(self, tr("Export Pole Figure"), suggested, tr("CSV files (*.csv)"))
        if not path:
            return
        indexing = self.model.record(analysis.geometry)
        if crystal:
            phase = self._crystal_result[self.crystal_ring_combo.currentData()[0]]
            written = write_crystal_export(Path(path), analysis, indexing, self._crystal_result, phase, ring)
        else:
            written = write_pole_export(Path(path), analysis, result, indexing)
        show_toast(self._page, trf("Wrote {name} and its record", name=written[0].name), level="ok",
                   action=(tr("Open Folder"), lambda: _open_folder(written[0].parent)))
        self.status_label.setText(trf("Wrote {files}.", files=", ".join(str(item) for item in written)))


def _color(name: str):
    from PyQt5.QtGui import QColor

    return QColor(name)


def _open_folder(folder: Path) -> None:
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


__all__ = ["IndexingPanel", "write_pole_export"]
