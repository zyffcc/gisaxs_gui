"""Behaviour of the GISAXS cuts table in the Cuts step: the rows, the editor, adding, duplicating, removing, renaming,
showing and hiding, the unit, and the question when a frame of another set-up comes.

The rows are derived from the set-up after every analysis: the automatic horizontal cut (at the Yoneda band, the one
Fitting takes), the automatic vertical cut, then the person's cuts (``AnalyzeState.gisaxs.regions``, kept in q). A cut
has one colour everywhere: its row, its curve, its outline on the detector and on the q map, its pixels under Sources.
Typed values are converted to q at once (``application.gisaxs_cuts``: one exact rule per axis); the automatic cuts
stay bands of whole rows and columns, so typing them sets their rows and columns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

from ...application import GISAXS, gisaxs_cut_tools as tools
from ..texts import message_text
from .gisaxs_cut_display import AUTO_COLORS, GisaxsCutDisplayMixin

CUT_COLORS = ("#db2777", "#16a34a", "#7c3aed", "#2563eb", "#dc2626", "#65a30d", "#4f46e5", "#a21caf")
"""The person's cuts, in turn (none orange or cyan: those are the automatic bands')."""
AUTO_NAMES = {"horizontal": "Yoneda cut", "vertical": "Vertical cut"}
KIND_WORDS = {
    "auto_horizontal": "Auto I(qy)", "auto_vertical": "Auto I(qz)", "horizontal": "Band I(qy)", "vertical": "Band I(qz)",
    "box_qy": "Box I(qy)", "box_qz": "Box I(qz)", "sector": "Sector I(qyz)", "line": "Line I(s)",
}
STEPS = {tools.UNIT_Q: 0.0005, tools.UNIT_ANGLE: 0.01, tools.UNIT_PIXEL: 1.0}


@dataclass(frozen=True)
class CutRow:
    key: str
    """The curve key: ``horizontal``, ``vertical``, ``cut1`` …"""
    name: str
    kind: str
    color: str
    index: Optional[int] = None
    """The cut's place in ``gisaxs.regions`` (``None``: an automatic cut)."""
    cut: Optional[object] = None

    @property
    def auto(self) -> bool:
        return self.index is None

    @property
    def axis(self) -> str:
        """What its curve is against: ``qy``, ``qz``, ``sector`` or ``line``."""
        if self.auto:
            return "qy" if self.kind == "horizontal" else "qz"
        return self.cut.axis

    @property
    def band_axis(self) -> str:
        """Where a band lies: ``qz`` for a horizontal band, ``qy`` for a vertical one."""
        return "qz" if self.kind == "horizontal" else "qy"

    @property
    def kind_word(self) -> str:
        if self.auto:
            return KIND_WORDS[f"auto_{self.kind}"]
        if self.kind == tools.BOX:
            return KIND_WORDS[f"box_{self.cut.along}"]
        return KIND_WORDS[self.kind]


def cut_rows(settings) -> list[CutRow]:
    """The rows of the cuts table for these GISAXS settings."""
    rows = [CutRow(kind, AUTO_NAMES[kind], kind, AUTO_COLORS[kind]) for kind in ("horizontal", "vertical")]
    for index, cut in enumerate(settings.regions):
        rows.append(CutRow(tools.cut_key(index), cut.name, cut.kind, CUT_COLORS[index % len(CUT_COLORS)], index, cut))
    return rows


def swatch(color: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


class GisaxsCutsMixin(GisaxsCutDisplayMixin):
    """Needs the cuts widgets (``views/gisaxs_cuts_view.py``), ``view_model``, ``run_analysis``, ``_status``."""

    def _connect_gisaxs_cuts(self) -> None:
        self._hidden_cuts: set[str] = set()
        self._cut_rows: list[CutRow] = []
        self._cut_table_state = None
        self._cut_unit = tools.UNIT_Q
        self._pending_cut_key: Optional[str] = None
        self._cut_checked = None
        self._cut_pending = None
        self.cut_unit_combo.currentIndexChanged.connect(self._cut_unit_chosen)
        for kind, action in self.cut_add_actions.items():
            action.triggered.connect(lambda _checked=False, kind=kind: self.add_gisaxs_cut(kind))
        self.cut_table.itemChanged.connect(self._cut_item_changed)
        self.cut_table.itemSelectionChanged.connect(self._cut_selected)
        self.cut_duplicate_button.clicked.connect(self.duplicate_gisaxs_cut)
        self.cut_remove_button.clicked.connect(self.remove_gisaxs_cut)
        for spin in self.cut_spins:
            spin.valueChanged.connect(self._cut_edited)
        self.cut_along_combo.activated.connect(self._cut_edited)
        self.cut_all_q_check.toggled.connect(self._cut_edited)
        self.cut_name_edit.editingFinished.connect(self._cut_renamed_in_editor)
        self.cut_keep_button.clicked.connect(self._keep_cuts)
        self.cut_drop_button.clicked.connect(self._drop_cuts)
        self._connect_cut_display()

    # -- the table -----------------------------------------------------------------------

    def _cut_geometry(self):
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None or analysis.kind != GISAXS:
            return None
        return analysis.geometry

    def _visible_cut_rows(self) -> list[CutRow]:
        """The ticked cuts of the set-up as it is now (the image is drawn before the table is filled again)."""
        return [row for row in cut_rows(self.view_model.state.gisaxs) if row.key not in self._hidden_cuts]

    def _auto_range(self, row: CutRow) -> Optional[tuple[float, float]]:
        """The q range of an automatic band (its continuous rows or columns on the beam-centre column / horizon)."""
        analysis, geometry = self.view_model.state.analysis, self._cut_geometry()
        if geometry is None:
            return None
        band = analysis.reduction.markers.get("horizontal_band" if row.kind == "horizontal" else "vertical_band")
        if band is None:
            return None
        return tools.range_from_unit(geometry, tools.UNIT_PIXEL, row.band_axis, band)

    def _where(self, row: CutRow) -> str:
        geometry = self._cut_geometry()
        if not row.auto:
            return tools.describe_cut(row.cut, geometry, self._cut_unit)
        q_range = self._auto_range(row)
        return "" if q_range is None else tools.span_text(geometry, self._cut_unit, row.band_axis, q_range)

    def _refresh_cut_rows(self, analysis, *, force: bool = False) -> None:
        """The table again after an analysis (kept as it is, with the row being edited, when nothing changed)."""
        if analysis is None or analysis.reduction is None or analysis.kind != GISAXS:
            self._cut_rows = []
            return
        rows = cut_rows(self.view_model.state.gisaxs)
        wheres = [self._where(row) for row in rows]
        state = (rows, wheres)  # (a tick is the table's own: nothing to rebuild for it)
        selected = self._selected_cut()
        wanted = self._pending_cut_key or (selected.key if selected is not None else None)
        self._pending_cut_key = None
        self._cut_rows = rows
        if not force and state == self._cut_table_state and self.cut_table.rowCount() == len(rows) \
                and wanted in (None, *(row.key for row in rows)) and (selected is None or selected.key == wanted):
            return
        self._cut_table_state = state
        table = self.cut_table
        with QSignalBlocker(table):
            table.setRowCount(len(rows))
            for number, (row, where) in enumerate(zip(rows, wheres)):
                name = QTableWidgetItem(swatch(row.color), tr(row.name) if row.auto else row.name)
                flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                name.setFlags(flags if row.auto else flags | Qt.ItemIsEditable)
                name.setCheckState(Qt.Unchecked if row.key in self._hidden_cuts else Qt.Checked)
                name.setData(Qt.UserRole, row.key)
                short = where.replace(" · ", "\n").replace(" Å⁻¹", "").replace(" px", "")  # (the unit list names it)
                place = QTableWidgetItem(tr(row.kind_word) + "\n" + short)
                place.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                place.setToolTip(tr(row.kind_word) + ": " + where)  # (the kind translated, the numbers as they are)
                table.setItem(number, 0, name)
                table.setItem(number, 1, place)
            table.resizeRowsToContents()
            keys = [row.key for row in rows]
            table.selectRow(keys.index(wanted) if wanted in keys else 0)
            self._fit_cut_table()
        self._cut_selected()

    def _fit_cut_table(self) -> None:
        """As tall as its rows (at most six before it scrolls)."""
        table = self.cut_table
        rows = sum(table.rowHeight(row) for row in range(min(6, table.rowCount())))
        table.setFixedHeight(2 * table.frameWidth() + table.horizontalHeader().height() + rows + 2)

    def _cut_texts(self) -> None:
        """The rows and the editor in the interface language again (after a language switch)."""
        self._refresh_cut_rows(self.view_model.state.analysis, force=True)

    def _selected_cut(self) -> Optional[CutRow]:
        row = self.cut_table.currentRow()
        return self._cut_rows[row] if 0 <= row < len(self._cut_rows) else None

    def _cut_item_changed(self, item) -> None:
        number = item.row()
        if item.column() != 0 or not 0 <= number < len(self._cut_rows):
            return
        row = self._cut_rows[number]
        hidden = item.checkState() != Qt.Checked
        if hidden != (row.key in self._hidden_cuts):
            (self._hidden_cuts.add if hidden else self._hidden_cuts.discard)(row.key)
            self._cuts_shown_changed()
            return
        if not row.auto and item.text().strip() and item.text().strip() != row.name:
            self._rename_cut(row, item.text())

    def _cuts_shown_changed(self) -> None:
        """A tick changed: the curves, outlines and bands again, nothing reduced again."""
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None:
            return
        self._redraw_curves(analysis)
        if self.view_combo.currentIndex() == 0:
            self._show_overlays(analysis)
        self._refresh_shapes()
        self._refresh_overlay()

    def _cut_unit_chosen(self, _index: int) -> None:
        self._cut_unit = self.cut_unit_combo.currentData() or tools.UNIT_Q
        self._refresh_cut_rows(self.view_model.state.analysis, force=True)

    # -- the editor ----------------------------------------------------------------------

    def _editor_row(self, number: int, caption: str, separator: str, unit: str, values, *, single: bool = False,
                    decimals: int = 4, step: float = 0.001) -> None:
        shown = bool(caption)
        first, second = self.cut_spins[2 * number], self.cut_spins[2 * number + 1]
        for widget in (self.cut_row_captions[number], first, self.cut_row_units[number]):
            widget.setVisible(shown)
        self.cut_row_separators[number].setVisible(shown and not single)
        second.setVisible(shown and not single)
        if not shown:
            return
        self.cut_row_captions[number].setText(caption)
        self.cut_row_separators[number].setText(separator)
        self.cut_row_units[number].setText(unit)
        for spin, value in zip((first, second), values):
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(float(value))

    def _cut_selected(self) -> None:
        row, geometry = self._selected_cut(), self._cut_geometry()
        self.cut_editor.setEnabled(row is not None and geometry is not None)
        self.cut_duplicate_button.setEnabled(row is not None and geometry is not None)
        self.cut_remove_button.setEnabled(row is not None and not row.auto)
        if row is None or geometry is None:
            return
        widgets = (*self.cut_spins, self.cut_along_combo, self.cut_all_q_check, self.cut_name_edit)
        blockers = [QSignalBlocker(widget) for widget in widgets]
        unit = self._cut_unit
        decimals, step = tools.DECIMALS[unit] + 1, STEPS[unit]
        self.cut_name_edit.setText(tr(row.name) if row.auto else row.name)
        self.cut_name_edit.setEnabled(not row.auto)
        self.cut_along_combo.setVisible(row.kind == tools.BOX)
        self.cut_all_q_check.setVisible(row.kind == tools.SECTOR)
        rows = [("", "", "", ())] * 3
        if row.kind in (tools.HORIZONTAL, tools.VERTICAL):
            axis = row.band_axis
            q_range = self._auto_range(row) if row.auto else getattr(row.cut, axis)
            low, high = tools.range_to_unit(geometry, unit, axis, q_range)
            symbol, suffix = tools.AXIS_SYMBOLS[(axis, unit)]
            rows[0] = (symbol, "±", suffix, (0.5 * (low + high), 0.5 * (high - low)))
        elif row.kind == tools.BOX:
            for number, axis in enumerate(("qy", "qz")):
                symbol, suffix = tools.AXIS_SYMBOLS[(axis, unit)]
                rows[number] = (symbol, "–", suffix, tools.range_to_unit(geometry, unit, axis, getattr(row.cut, axis)))
            self.cut_along_combo.setCurrentIndex(max(0, self.cut_along_combo.findData(row.cut.along)))
        elif row.kind == tools.SECTOR:
            low, high = row.cut.chi_range
            rows[0] = ("χ", "±", "°", (0.5 * (low + high), 0.5 * (high - low)))
            radial = row.cut.q_range
            rows[1] = ("qyz", "–", "Å⁻¹", radial or self._sector_reach())
            self.cut_all_q_check.setChecked(radial is None)
        else:
            suffix = tools.WIDTH_SYMBOLS[unit]
            for number, (caption, point) in enumerate((("Start point", row.cut.start), ("End point", row.cut.end))):
                values = [tools.axis_to_unit(geometry, unit, axis, value)[0] for axis, value in zip(("qy", "qz"), point)]
                rows[number] = (tr(caption), ",", suffix, values)
            rows[2] = (tr("Half width"), "", suffix, (tools.width_to_unit(geometry, unit, row.cut.half_width),))
        for number, (caption, separator, suffix, values) in enumerate(rows):
            fixed = row.kind == tools.SECTOR
            self._editor_row(number, caption, separator, suffix, values, single=len(values) == 1,
                             decimals=(1 if number == 0 else 4) if fixed else decimals,
                             step=(1.0 if number == 0 else 0.001) if fixed else step)
        self.cut_spins[2].setEnabled(not (row.kind == tools.SECTOR and row.cut.q_range is None))
        self.cut_spins[3].setEnabled(not (row.kind == tools.SECTOR and row.cut.q_range is None))
        self.cut_range_label.setText(self._cut_equivalents(row))
        del blockers

    def _sector_reach(self) -> tuple[float, float]:
        extent = tools.q_extent(self.view_model.state.analysis) if self._cut_geometry() is not None else None
        if extent is None:
            return 0.0, 0.1
        corners = [abs(complex(y, z)) for y in extent[0] for z in extent[1]]
        return 0.0, max(corners)

    def _cut_equivalents(self, row: CutRow) -> str:
        """The band's or box's position in the two other units (``=`` …), or the frame cut of an automatic band."""
        geometry = self._cut_geometry()
        if geometry is None:
            return ""
        step = f"{tools.q_per_pixel(geometry):.2e}"
        if row.kind == tools.SECTOR:
            return tr("χ from the qz axis (the sign of qy), qyz = √(qy² + qz²) from the direct beam; bins one pixel wide "
                      "(k·p/D = {step} Å⁻¹).").format(step=step)
        if row.kind == tools.LINE:
            symbols = ", ".join(tools.AXIS_SYMBOLS[(axis, self._cut_unit)][0] for axis in ("qy", "qz"))
            return tr("Points as ({axes}), kept in q; I against the distance s from the start point, in bins one pixel "
                      "wide (k·p/D = {step} Å⁻¹).").format(axes=symbols, step=step)
        others = [unit for unit in tools.UNITS if unit != self._cut_unit]
        if row.auto:
            q_ranges = {row.band_axis: self._auto_range(row)}
        else:
            q_ranges = {axis: getattr(row.cut, axis) for axis in ("qy", "qz") if getattr(row.cut, axis) is not None}
        parts = [tools.span_text(geometry, unit, axis, q_range) for unit in others
                 for axis, q_range in q_ranges.items() if q_range is not None]
        return "= " + " · ".join(parts)

    def _cut_edited(self, *_args) -> None:
        row, geometry = self._selected_cut(), self._cut_geometry()
        if row is None or geometry is None:
            return
        values = [spin.value() for spin in self.cut_spins]
        unit = self._cut_unit
        try:
            if row.auto:
                self._auto_band_typed(row, values[0], values[1])
                return
            cut = self._typed_cut(row, values, geometry, unit)
        except ValueError as exc:
            self._status(message_text(exc), "warning")
            return
        self.view_model.update_gisaxs_cut(row.index, cut)
        self._pending_cut_key = row.key
        self.run_analysis()

    def _typed_cut(self, row: CutRow, values, geometry, unit):
        cut = row.cut
        if row.kind == tools.HORIZONTAL:
            return replace(cut, qz=tools.range_from_unit(geometry, unit, "qz", (values[0] - values[1], values[0] + values[1])))
        if row.kind == tools.VERTICAL:
            return replace(cut, qy=tools.range_from_unit(geometry, unit, "qy", (values[0] - values[1], values[0] + values[1])))
        if row.kind == tools.BOX:
            return replace(cut, qy=tools.range_from_unit(geometry, unit, "qy", values[0:2]),
                           qz=tools.range_from_unit(geometry, unit, "qz", values[2:4]),
                           along=self.cut_along_combo.currentData() or tools.ALONG_QY)
        if row.kind == tools.SECTOR:
            every = self.cut_all_q_check.isChecked()
            return replace(cut, chi_range=(values[0] - values[1], values[0] + values[1]),
                           q_range=None if every else (values[2], values[3]))
        start = tuple(float(tools.axis_from_unit(geometry, unit, axis, value)[0]) for axis, value in zip(("qy", "qz"), values[0:2]))
        end = tuple(float(tools.axis_from_unit(geometry, unit, axis, value)[0]) for axis, value in zip(("qy", "qz"), values[2:4]))
        return replace(cut, start=start, end=end, half_width=tools.width_from_unit(geometry, unit, values[4]))

    def _auto_band_typed(self, row: CutRow, centre: float, half: float) -> None:
        """An automatic band typed in any unit: its rows (columns) are set, as a drag of the band does."""
        geometry = self._cut_geometry()
        q_range = tools.range_from_unit(geometry, self._cut_unit, row.band_axis, (centre - half, centre + half))
        low, high = tools.range_to_unit(geometry, tools.UNIT_PIXEL, row.band_axis, q_range)
        if row.kind == "horizontal":
            self.view_model.set_horizontal_band(low, high)
        else:
            self.view_model.set_vertical_band(low, high)
        self._pending_cut_key = row.key
        self.run_analysis()

    def _cut_renamed_in_editor(self) -> None:
        row = self._selected_cut()
        if row is not None and not row.auto and self.cut_name_edit.text().strip() not in ("", row.name):
            self._rename_cut(row, self.cut_name_edit.text())

    def _rename_cut(self, row: CutRow, name: str) -> None:
        self.view_model.update_gisaxs_cut(row.index, tools.renamed(row.cut, name))
        self._pending_cut_key = row.key
        self.run_analysis()

    # -- adding, duplicating, removing ------------------------------------------------------------

    def _gisaxs_ready(self) -> bool:
        if self._cut_geometry() is None:
            self._status(tr("Open a GISAXS frame with a geometry first."), "warning")
            return False
        return True

    def _cut_name(self, kind: str) -> str:
        words = {tools.HORIZONTAL: "Horizontal", tools.VERTICAL: "Vertical", tools.BOX: "Box", tools.SECTOR: "Sector",
                 tools.LINE: "Line"}
        taken = {cut.name for cut in self.view_model.gisaxs_cuts}
        number = 1
        while f"{words[kind]} {number}" in taken:
            number += 1
        return f"{words[kind]} {number}"

    def add_gisaxs_cut(self, kind: str, cut=None) -> Optional[int]:
        """Add a cut of ``kind`` (from the Add menu: placed where it is easy to see) or ``cut`` itself; it is selected
        to be typed. Returns its index."""
        if not self._gisaxs_ready():
            return None
        analysis = self.view_model.state.analysis
        if cut is None:
            extent = tools.q_extent(analysis)
            if extent is None:
                self._status(tr("Open a GISAXS frame with a geometry first."), "warning")
                return None
            yoneda =self._auto_range(self._cut_rows[0]) if self._cut_rows else None
            try:
                cut = tools.new_cut(kind, analysis.geometry, extent, name=self._cut_name(kind),
                                    yoneda_qz=None if yoneda is None else 0.5 * sum(yoneda))
            except ValueError as exc:  # (a frame too small to place it: say why)
                self._status(message_text(exc), "warning")
                return None
        index = self.view_model.add_gisaxs_cut(cut)
        self.view_model.keep_cuts_for(analysis)  # made on this frame's set-up
        self._pending_cut_key = tools.cut_key(index)
        self.run_analysis()
        movable = cut.kind in (tools.HORIZONTAL, tools.VERTICAL, tools.BOX)
        show_toast(self.window(), trf(
            "{name} added: type its position below the table, or drag it on the q map." if movable
            else "{name} added: type its position below the table.", name=cut.name), level="ok")
        return index

    def duplicate_gisaxs_cut(self) -> Optional[int]:
        """A copy of the selected cut (of an automatic band: a band of the person's at the same q)."""
        row, geometry = self._selected_cut(), self._cut_geometry()
        if row is None or geometry is None:
            return None
        taken = [cut.name for cut in self.view_model.gisaxs_cuts]
        if row.auto:
            q_range = self._auto_range(row)
            name = tools.copy_name(tr(row.name), taken)
            cut = (tools.GisaxsCut(name, tools.HORIZONTAL, qz=q_range) if row.kind == "horizontal"
                   else tools.GisaxsCut(name, tools.VERTICAL, qy=q_range))
        else:
            cut = tools.renamed(row.cut, tools.copy_name(row.name, taken))
        return self.add_gisaxs_cut(cut.kind, cut)

    def remove_gisaxs_cut(self) -> None:
        row = self._selected_cut()
        if row is None or row.auto:
            return
        self.view_model.remove_gisaxs_cut(row.index)
        self._hidden_cuts = {key for key in self._hidden_cuts if key in ("horizontal", "vertical")}
        self._status(trf("Removed {name} (Ctrl+Z brings it back).", name=row.name), "ok")
        self.run_analysis()

    # -- cuts of another set-up ------------------------------------------------------------------

    def _check_cut_setup(self, analysis) -> None:
        """A frame of another detector or geometry than the cuts were made on: say so once, and let the person keep
        or remove the cuts (they are in q, so meanwhile they cut this frame at the same q)."""
        if analysis is None or analysis is self._cut_checked:
            return
        self._cut_checked = analysis
        changes = self.view_model.cut_setup_changes(analysis)
        if not changes:
            self._cut_pending = None
            self.cut_notice.hide()
            return
        asked_before = self.cut_notice.isVisible() or self._cut_pending is not None
        self._cut_pending = analysis
        words = ", ".join(tr(change) for change in changes)
        self.cut_notice_label.setText(trf(
            "{n} cut(s) were made on another set-up (other {changes}). They are kept in q and cut this frame at the "
            "same q until you choose.", n=self.view_model.person_cut_count(), changes=words))
        self.cut_notice.show()
        if not asked_before and self.isVisible():
            show_toast(self.window(), trf("This frame has another {changes} than the cuts were made on: keep or remove "
                                          "them in the Cuts step.", changes=words),
                       level="warning", action=(tr("Show Cuts"), lambda: self.show_step("cuts")), timeout_ms=0)

    def _keep_cuts(self) -> None:
        if self._cut_pending is not None:
            self.view_model.keep_cuts_for(self._cut_pending)
        self._cut_pending = None
        self.cut_notice.hide()
        self._status(tr("Cuts kept for this set-up."), "ok")

    def _drop_cuts(self) -> None:
        pending = self._cut_pending or self.view_model.state.analysis
        self.view_model.drop_person_cuts()
        if pending is not None:
            self.view_model.keep_cuts_for(pending)
        self._cut_pending = None
        self._hidden_cuts = {key for key in self._hidden_cuts if key in ("horizontal", "vertical")}
        self.cut_notice.hide()
        self._status(tr("Cuts removed: only the automatic cuts stay (Ctrl+Z brings them back)."), "ok")
        self.run_analysis()


__all__ = ["AUTO_NAMES", "CUT_COLORS", "CutRow", "GisaxsCutsMixin", "KIND_WORDS", "cut_rows"]
