"""Behaviour of the GIWAXS cut regions: the list, the editor, adding, and dragging on the cake.

The rows are derived from the settings after every analysis (the full ring, the in-plane and
out-of-plane bands, the ring of I(χ), a custom sector, the person's regions), so the list, the
curves, the outlines on the q map and the rectangles on the cake always agree. A region has one
colour everywhere: its curves, its outline, its pixels under Sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import QListWidgetItem

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ...application import GISAXS, CutRegion, default_region_near, region_key, region_outline

STANDARD_COLORS = {
    "full": "#2563eb", "in_plane": "#f97316", "out_of_plane": "#16a34a", "ring": "#9333ea", "sector": "#dc2626",
}
REGION_COLORS = ("#0891b2", "#db2777", "#ca8a04", "#0d9488", "#7c3aed", "#65a30d", "#ea580c", "#4f46e5")
FULL_CHI = (-90.0, 90.0)


@dataclass(frozen=True)
class RegionRow:
    key: str
    name: str
    q_range: Optional[tuple[float, float]]
    chi_range: tuple[float, float]
    both_sides: bool
    color: str
    q_curve: Optional[str]
    chi_curve: Optional[str]
    editor: str
    index: Optional[int] = None

    def q_text(self) -> str:
        return "all" if self.q_range is None else f"{self.q_range[0]:.3f}–{self.q_range[1]:.3f}"

    def chi_text(self) -> str:
        low, high = self.chi_range
        return f"{'|χ| ' if self.both_sides else ''}{low:g}–{high:g}"


def region_rows(settings, markers: dict) -> list[RegionRow]:
    """The rows of the region list for these GIWAXS settings (``markers``: of the last reduction)."""
    in_plane = 90.0 - float(settings.in_plane_half_width_deg)
    out = float(settings.out_of_plane_half_width_deg)
    window = settings.chi_q_window or (markers or {}).get("chi_q_window")
    rows = [
        RegionRow("full", "Full ring", None, FULL_CHI, False, STANDARD_COLORS["full"], "radial", None, "note"),
        RegionRow("in_plane", "In-plane", None, (in_plane, 90.0), True, STANDARD_COLORS["in_plane"], "in_plane", None, "width"),
        RegionRow("out_of_plane", "Out-of-plane", None, (0.0, out), True, STANDARD_COLORS["out_of_plane"], "out_of_plane", None, "width"),
        RegionRow(
            "ring", "Ring for I(χ)", None if window is None else tuple(sorted(window)), FULL_CHI, False,
            STANDARD_COLORS["ring"], None, "azimuthal", "ring",
        ),
    ]
    sector = settings.sector
    if sector is not None:
        q_range = None if sector.q_min is None and sector.q_max is None else (sector.q_min or 0.0, sector.q_max or 100.0)
        rows.append(RegionRow(
            "sector", "Custom sector", q_range, tuple(sorted((sector.chi_min_deg, sector.chi_max_deg))), False,
            STANDARD_COLORS["sector"], "sector", "sector_chi" if q_range is not None else None, "generic",
        ))
    for index, region in enumerate(settings.regions):
        key = region_key(index)
        rows.append(RegionRow(
            key, region.name, region.q_range, region.chi_range, region.both_sides,
            REGION_COLORS[index % len(REGION_COLORS)], key, f"{key}_chi", "generic", index,
        ))
    return rows


def _full_chi(both_sides: bool) -> tuple[float, float]:
    return (0.0, 90.0) if both_sides else (-90.0, 90.0)


def _range_text(q_range, chi_range, both_sides: bool) -> str:
    """The region's window in plain numbers, with the d-spacing of the q range (d = 2π/q)."""
    if q_range is None:
        q_text = tr("every q")
    else:
        low, high = q_range
        d_text = f" (d {2 * np.pi / high:.3f}–{2 * np.pi / low:.3f} Å)" if low > 0 else ""
        q_text = f"q {low:.4f}–{high:.4f} Å⁻¹{d_text}"
    chi_text = f"{'|χ|' if both_sides else 'χ'} {chi_range[0]:.1f}–{chi_range[1]:.1f}°"
    return f"{q_text} · {chi_text}"


def _swatch(color: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


class RegionsMixin:
    """Needs the regions widgets (``views/regions_view.py``), ``shape_layer``, ``view_model``, ``run_analysis``."""

    def _connect_regions(self) -> None:
        self._hidden_regions: set[str] = set()
        self._region_rows: list[RegionRow] = []
        self.region_list.itemChanged.connect(self._region_item_changed)
        self.region_list.itemSelectionChanged.connect(self._region_selected)
        self.region_add_ring_action.triggered.connect(lambda: self._add_region("ring"))
        self.region_add_in_plane_action.triggered.connect(lambda: self._add_region("in_plane"))
        self.region_add_out_of_plane_action.triggered.connect(lambda: self._add_region("out_of_plane"))
        self.region_add_whole_action.triggered.connect(lambda: self._add_region("whole"))
        for spin in (self.region_q_center, self.region_q_half, self.region_chi_center, self.region_chi_half):
            spin.valueChanged.connect(self._region_edited)
        for check in (self.region_all_q_check, self.region_all_chi_check, self.region_both_check):
            check.toggled.connect(self._region_edited)
        self.region_name_edit.editingFinished.connect(self._region_edited)
        self.region_remove_button.clicked.connect(self._remove_region)
        for spin in (self.ring_q_center, self.ring_q_half):
            spin.valueChanged.connect(self._ring_edited)
        self.ring_auto_button.clicked.connect(self._ring_auto)
        self.shape_layer.rectChanged.connect(self._region_rect_moved)

    # -- the list ----------------------------------------------------------------------

    def region_colors(self) -> dict[str, str]:
        """Curve key → colour of its region (for the plots and the Sources overlay)."""
        colors = {}
        for row in self._region_rows:
            for key in (row.q_curve, row.chi_curve):
                if key:
                    colors[key] = row.color
        return colors

    def visible_region(self, curve_key: str) -> bool:
        return not any(curve_key in (row.q_curve, row.chi_curve) for row in self._region_rows if row.key in self._hidden_regions)

    def _refresh_regions(self, analysis) -> None:
        reduction = analysis.reduction if analysis is not None else None
        if reduction is None or reduction.kind == GISAXS:
            self._region_rows = []
            self.shape_layer.clear()
            return
        selected = self._selected_row()
        wanted = getattr(self, "_pending_region_key", None) or (selected.key if selected is not None else None)
        self._pending_region_key = None
        rows = region_rows(self.view_model.state.giwaxs, reduction.markers)
        if rows == self._region_rows and self.region_list.count() == len(rows) and wanted in (None, *(row.key for row in rows)):
            self._refresh_shapes()  # nothing changed in the list (e.g. a tick): keep it, and the item being edited
            return
        self._region_rows = rows
        listing = self.region_list
        with QSignalBlocker(listing):
            listing.clear()
            for row in self._region_rows:
                item = QListWidgetItem(_swatch(row.color), f"{tr(row.name)}\nq {row.q_text()} Å⁻¹ · {row.chi_text()}°")
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked if row.key in self._hidden_regions else Qt.Checked)
                item.setData(Qt.UserRole, row.key)
                listing.addItem(item)
            listing.setFixedHeight(min(300, 8 + 40 * len(self._region_rows)))
            keys = [row.key for row in self._region_rows]
            listing.setCurrentRow(keys.index(wanted) if wanted in keys else 0)
        self._region_selected()
        self._refresh_shapes()

    def _selected_row(self) -> Optional[RegionRow]:
        row = self.region_list.currentRow()
        if not 0 <= row < len(self._region_rows):
            return None
        return self._region_rows[row]

    def _region_item_changed(self, item) -> None:
        key = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._hidden_regions.discard(key)
        else:
            self._hidden_regions.add(key)
        analysis = self.view_model.state.analysis
        if analysis is not None and analysis.reduction is not None:
            self._show_curves(analysis)
            self._refresh_overlay()
            self._refresh_shapes()

    # -- the editor ----------------------------------------------------------------------

    def _region_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.region_editor.setCurrentWidget(self.region_editor_pages[row.editor])
        blockers = [QSignalBlocker(widget) for widget in (
            self.region_q_center, self.region_q_half, self.region_chi_center, self.region_chi_half,
            self.region_all_q_check, self.region_all_chi_check, self.region_both_check, self.region_name_edit,
            self.ring_q_center, self.ring_q_half,
        )]
        if row.key == "full":
            self.region_note_label.setText(tr(
                "The whole pattern above the horizon, every χ. Its bins and the q or 2θ axis are under More cut settings."
            ))
        elif row.editor == "width":
            in_plane = row.key == "in_plane"
            self.in_plane_spin.setVisible(in_plane)
            self.out_of_plane_spin.setVisible(not in_plane)
            self.region_width_caption.setText(tr("Half width around χ = ±90°" if in_plane else "Half width around χ = 0°"))
        elif row.editor == "ring":
            low, high = row.q_range or (0.0, 0.0)
            self.ring_q_center.setValue(0.5 * (low + high))
            self.ring_q_half.setValue(max(0.5 * (high - low), self.ring_q_half.minimum()))
        else:
            self.region_name_edit.setText(row.name)
            self.region_name_edit.setEnabled(row.index is not None)
            self.region_all_q_check.setChecked(row.q_range is None)
            low, high = row.q_range or self._q_limits()
            self.region_q_center.setValue(0.5 * (low + high))
            self.region_q_half.setValue(max(0.5 * (high - low), self.region_q_half.minimum()))
            full = _full_chi(row.both_sides)
            every_chi = row.chi_range[0] <= full[0] + 1e-6 and row.chi_range[1] >= full[1] - 1e-6
            self.region_all_chi_check.setChecked(every_chi)
            self.region_chi_center.setMinimum(full[0])
            self.region_chi_center.setValue(0.5 * (row.chi_range[0] + row.chi_range[1]))
            self.region_chi_half.setValue(0.5 * (row.chi_range[1] - row.chi_range[0]))
            self.region_both_check.setChecked(row.both_sides)
            self.region_both_check.setEnabled(row.index is not None)
            self.region_snap_button.setEnabled(row.index is not None)
            self._sync_region_spins()
            self.region_range_label.setText(_range_text(row.q_range, row.chi_range, row.both_sides))
            self.region_remove_button.setText(tr("Remove Region" if row.index is not None else "Remove Sector"))
        del blockers

    def _sync_region_spins(self) -> None:
        for spin in (self.region_q_center, self.region_q_half):
            spin.setEnabled(not self.region_all_q_check.isChecked())
        for spin in (self.region_chi_center, self.region_chi_half):
            spin.setEnabled(not self.region_all_chi_check.isChecked())

    def _q_limits(self) -> tuple[float, float]:
        analysis = self.view_model.state.analysis
        radial = analysis.reduction.curve("radial") if analysis is not None and analysis.reduction is not None else None
        if radial is None or radial.is_empty:
            return 0.0, 1.0
        x = radial.x
        if radial.x_label.startswith("2θ") and analysis.geometry is not None:
            from ...application import q_from_two_theta

            x = q_from_two_theta(x, analysis.geometry.wavelength_angstrom)
        return float(np.nanmin(x)), float(np.nanmax(x))

    def _region_edited(self, *_args) -> None:
        row = self._selected_row()
        if row is None or row.editor != "generic":
            return
        self._sync_region_spins()
        if self.region_all_q_check.isChecked():
            q_range = None
        else:
            center, half = self.region_q_center.value(), self.region_q_half.value()
            q_range = (max(0.0, center - half), center + half)
        both = self.region_both_check.isChecked() and row.index is not None
        full = _full_chi(both)
        if self.region_all_chi_check.isChecked():
            chi_range = full
        else:
            center, half = self.region_chi_center.value(), self.region_chi_half.value()
            center = abs(center) if both else center
            chi_range = (max(full[0], center - half), min(full[1], center + half))
        self.region_range_label.setText(_range_text(q_range, chi_range, both))
        if row.index is None:  # the custom sector: its own widgets, so the automation sees the same values
            self._set_sector(chi_range, q_range)
            return
        try:
            region = CutRegion(self.region_name_edit.text().strip() or row.name, q_range, chi_range, both)
        except ValueError as exc:
            self._status(str(exc), "warning")
            return
        self.view_model.update_region(row.index, region)
        self._options_changed()

    def _set_sector(self, chi_range, q_range) -> None:
        values = (
            (self.sector_chi_min, chi_range[0]), (self.sector_chi_max, chi_range[1]),
            (self.sector_q_min, 0.0 if q_range is None else q_range[0]), (self.sector_q_max, 0.0 if q_range is None else q_range[1]),
        )
        for spin, value in values:
            with QSignalBlocker(spin):
                spin.setValue(value)
        self._sector_changed()

    def _ring_edited(self, *_args) -> None:
        center, half = self.ring_q_center.value(), self.ring_q_half.value()
        low, high = max(0.0, center - half), center + half
        if high > low:
            self.view_model.set_chi_window(low, high)
            self._options_changed()

    def _ring_auto(self) -> None:
        self.view_model.clear_chi_window()
        self.run_analysis()

    # -- adding and removing -------------------------------------------------------------------

    def _add_region(self, kind: str) -> None:
        count = len(self.view_model.state.giwaxs.regions) + 1
        name = f"Region {count}"
        if kind == "ring":
            analysis = self.view_model.state.analysis
            window = analysis.reduction.markers.get("chi_q_window") if analysis is not None and analysis.reduction else None
            if window is None:
                self._status("No ring stands out in I(q): set the q range of the new region.", "warning")
                region = CutRegion(name, None, (0.0, 90.0), True)
            else:
                region = default_region_near(0.5 * (window[0] + window[1]), name=name)
        elif kind == "in_plane":
            region = CutRegion(name, None, (70.0, 90.0), True)
        elif kind == "out_of_plane":
            region = CutRegion(name, None, (0.0, 20.0), True)
        else:
            region = CutRegion(name, None, (0.0, 90.0), True)
        self.view_model.add_region(region)
        self._pending_region_key = region_key(count - 1)
        self.run_analysis()
        show_toast(self.window(), f"{name} added: drag it on the Cake view, or change it in the list.", level="ok",
                   action=("Show Cake", lambda: self.set_view(2)))

    def _remove_region(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.index is None:
            self.sector_check.setChecked(False)  # removes the custom sector
            return
        self.view_model.remove_region(row.index)
        self._hidden_regions.discard(row.key)
        self.run_analysis()

    # -- on the image ----------------------------------------------------------------------

    def _visible_rows(self) -> list[RegionRow]:
        return [row for row in self._region_rows if row.key not in self._hidden_regions]

    def region_shapes(self, view: int) -> tuple[list, list]:
        """``(outlines, rects)`` of the visible regions for the q map (1) or the cake (2)."""
        outlines, rects = [], []
        q_limits = self._q_limits()
        for row in self._visible_rows():
            q_range = row.q_range or q_limits
            sides = [row.chi_range]
            if row.both_sides:
                sides.append((-row.chi_range[1], -row.chi_range[0]))
            if view == 1 and row.key != "full":
                for chi in sides:
                    x, z = region_outline(q_range, chi)
                    outlines.append((x, z, row.color))
            elif view == 2 and row.key != "full":
                editable = row.editor in ("ring", "generic")
                for number, chi in enumerate(sides):
                    rects.append((row.key + ("~" if number else ""), q_range[0], chi[0], q_range[1], chi[1], row.color, editable))
        return outlines, rects

    def _region_rect_moved(self, key: str, x0: float, y0: float, x1: float, y1: float) -> None:
        mirrored = key.endswith("~")
        key = key.rstrip("~")
        row = next((item for item in self._region_rows if item.key == key), None)
        if row is None:
            return
        q_low, q_high = sorted((max(0.0, x0), max(0.0, x1)))
        chi_low, chi_high = sorted((max(-90.0, min(90.0, y0)), max(-90.0, min(90.0, y1))))
        if row.key == "ring":
            self.view_model.set_chi_window(q_low, q_high)
            self.run_analysis()
            return
        q_range = None if row.q_range is None and abs(q_low - self._q_limits()[0]) < 1e-3 else (q_low, q_high)
        if row.index is None:
            self._set_sector((chi_low, chi_high), q_range)
            return
        if row.both_sides:
            chi_low, chi_high = sorted((abs(chi_low), abs(chi_high))) if not mirrored else sorted((-chi_high, -chi_low))
            chi_low, chi_high = max(0.0, chi_low), max(0.0, chi_high)
        try:
            region = CutRegion(row.name, q_range, (chi_low, chi_high), row.both_sides)
        except ValueError as exc:
            self._status(str(exc), "warning")
            self._refresh_shapes()
            return
        self.view_model.update_region(row.index, region)
        self.run_analysis()


__all__ = ["REGION_COLORS", "RegionRow", "RegionsMixin", "STANDARD_COLORS", "region_rows"]
