"""Placing cut regions by hand: click on the image, draw on the Cake view, snap, save and load.

* Ring / Sector / Spot arm one click on the image — detector, q map or cake — or, for a ring,
  on the upper plot (I(q)). The click becomes (q, χ) and a region snapped to the peak there
  (``PlaceRegions``, in the background); a toast says what was found — peak centre, d, FWHM —
  or that no peak stands out, with Undo.
* Draw on the Cake View: two corners of a rectangle in (q, χ); a rectangle on one side of χ = 0
  becomes a region on both sides (folded onto |χ|), one across χ = 0 keeps its sign.
* Snap to Peak moves the selected region onto the peak nearest its centre.
* Save Cuts… / Load Cuts… keep a set of regions as JSON (``cut_set``) for the next data set;
  loading replaces the regions the person added (the standard cuts stay).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import POINT, show_toast
from src.gimap.app.presentation.i18n import tr

from ...application import (
    GISAXS,
    RECTANGLE,
    RING,
    SECTOR,
    CutRegion,
    cut_set,
    q_from_two_theta,
    region_key,
    regions_from_cut_set,
)
from .display import VIEW_CAKE, VIEW_Q_MAP

PICK_PURPOSE = "region-pick"
DRAW_PURPOSE = "region-draw"
CUT_FILTER = "GIMaP cut regions (*.json)"
PICK_HINTS = {
    "ring": "Click a peak on the image or on the I(q) plot (Esc cancels).",
    "sector": "Click on the image in the direction of the cut (Esc cancels).",
    "spot": "Click a spot on the image (Esc cancels).",
}


def pick_name(pick) -> str:
    """A name that says where the region is: ``Ring q 1.001``, ``Sector |χ| 45°`` …"""
    region = pick.region
    q_mid = None if region.q_range is None else 0.5 * (region.q_range[0] + region.q_range[1])
    chi_mid = 0.5 * (region.chi_range[0] + region.chi_range[1])
    side = "|χ|" if region.both_sides else "χ"
    if pick.kind == RING:
        return f"Ring q {q_mid:.3f}"
    if pick.kind == SECTOR:
        return f"Sector {side} {chi_mid:.0f}°"
    return f"Spot q {q_mid:.3f}, {side} {chi_mid:.0f}°"


def pick_note(pick) -> str:
    """What the snapping found, for the toast."""
    parts = []
    if pick.q_peak is not None:
        peak = pick.q_peak
        parts.append(tr("Peak at q = {q} Å⁻¹ (d = {d} Å), FWHM {w} Å⁻¹").format(
            q=f"{peak.center:.4f}", d=f"{2 * math.pi / peak.center:.3f}", w=f"{peak.fwhm:.4f}",
        ))
    elif pick.region.q_range is not None:
        parts.append(tr("No peak stands out at q = {q} Å⁻¹: a window around the click").format(q=f"{pick.q_clicked:.3f}"))
    if pick.chi_peak is not None:
        parts.append(tr("azimuthal width {w}° at χ = {chi}°").format(
            w=f"{pick.chi_peak.fwhm:.1f}", chi=f"{pick.chi_peak.center:.1f}",
        ))
    return "; ".join(parts)


class RegionPickMixin:
    """Needs the region widgets, ``shape_layer``, ``top_plot``, ``view_model``, ``tasks`` and the regions mixin."""

    def _connect_region_pick(self) -> None:
        self._pick_kind = ""
        for key, button in self.region_pick_buttons.items():
            button.toggled.connect(lambda on, key=key: self._pick_toggled(key, on))
        self.shape_layer.shapePicked.connect(self._shape_picked)
        self.shape_layer.drawingChanged.connect(self._region_drawing_changed)
        self.top_plot.positionClicked.connect(self._plot_clicked)
        self.region_draw_action.triggered.connect(self.draw_region)
        self.region_snap_button.clicked.connect(self.snap_selected_region)
        self.region_save_action.triggered.connect(lambda: self.save_cuts())
        self.region_load_action.triggered.connect(lambda: self.load_cuts())

    # -- arming ------------------------------------------------------------------------

    def _giwaxs_ready(self) -> bool:
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None or analysis.reduction.kind == GISAXS:
            self._status(tr("Open a GIWAXS frame with a geometry first."), "warning")
            return False
        return True

    def _set_pick_buttons(self, kind: str) -> None:
        for key, button in self.region_pick_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(key == kind)

    def _pick_toggled(self, kind: str, on: bool) -> None:
        if not on:
            if self.shape_layer.purpose == PICK_PURPOSE and self._pick_kind == kind:
                self.shape_layer.cancel_drawing()
            return
        if not self._giwaxs_ready():
            self._set_pick_buttons("")
            return
        self.pick_region(kind)

    def pick_region(self, kind: str) -> None:
        """Arm one click for a region of ``kind`` (``ring``, ``sector``, ``spot``)."""
        self._pick_kind = kind
        self.shape_layer.start_drawing(POINT, purpose=PICK_PURPOSE)
        self._set_pick_buttons(kind)
        self._status(tr(PICK_HINTS[kind]))

    def _region_drawing_changed(self, kind: str) -> None:
        picking = bool(kind) and self.shape_layer.purpose == PICK_PURPOSE
        self._set_pick_buttons(self._pick_kind if picking else "")

    # -- the click -----------------------------------------------------------------------

    def _shape_picked(self, purpose: str, kind: str, points: list) -> None:
        if purpose == PICK_PURPOSE and points:
            self._picked_on_view(*points[0])
        elif purpose == DRAW_PURPOSE and len(points) == 2:
            self._region_drawn(points)

    def _picked_on_view(self, x: float, y: float) -> None:
        analysis = self.view_model.state.analysis
        if analysis is None:
            return
        view = self.view_combo.currentIndex()
        if view == VIEW_CAKE:
            q, chi = x, y
        elif view == VIEW_Q_MAP:
            q, chi = math.hypot(x, y), math.degrees(math.atan2(x, y))
        else:
            try:
                q, chi = self.view_model.region_at_pixel(analysis, x, y)
            except ValueError as exc:
                self._status(str(exc), "warning")
                return
        if not math.isfinite(q) or q <= 0:
            self._status(tr("No q at that point: click on the pattern."), "warning")
            return
        self._place_region(self._pick_kind, q, chi)

    def _plot_clicked(self, x: float, _y: float) -> None:
        if not self._pick_kind or self.shape_layer.purpose != PICK_PURPOSE:
            return
        if self._pick_kind != RING:
            self._status(tr("The I(q) plot has no χ: click on the image for a sector or a spot."), "warning")
            return
        analysis = self.view_model.state.analysis
        q = x
        if self._uses_two_theta() and analysis is not None and analysis.geometry is not None:
            q = float(q_from_two_theta(x, analysis.geometry.wavelength_angstrom))
        self.shape_layer.cancel_drawing()
        if q > 0:
            self._place_region(RING, q, 45.0, chi_band=(0.0, 90.0))

    def _place_region(self, kind: str, q: float, chi: float, *, chi_band=None) -> None:
        analysis = self.view_model.state.analysis
        if analysis is None or not kind:
            return
        self._status(tr("Finding the peak …"))
        self.tasks.submit(
            "region-pick",
            lambda: self.view_model.pick_region(analysis, kind, q, chi, name="", chi_band=chi_band),
            on_done=self._region_picked,
            on_error=lambda message, _details: self._status(tr("Could not place the region: {error}").format(error=message), "error"),
        )

    def _region_picked(self, pick) -> None:
        region = replace(pick.region, name=pick_name(pick))
        index = self.view_model.add_region(region)
        self._pending_region_key = region_key(index)
        self.run_analysis()
        note = pick_note(pick)
        show_toast(
            self.window(), f"{region.name} " + tr("added") + (f" — {note}" if note else ""), level="ok",
            action=(tr("Undo"), lambda region=region: self._undo_region(region)),
        )
        self._status(f"{region.name}: {note}" if note else region.name, "ok")

    def _undo_region(self, region: CutRegion) -> None:
        regions = list(self.view_model.state.giwaxs.regions)
        if region in regions:
            self.view_model.remove_region(regions.index(region))
            self.run_analysis()

    # -- drawing on the cake ---------------------------------------------------------------

    def draw_region(self) -> None:
        """Two corners on the Cake view (q across, χ up) make a region."""
        if not self._giwaxs_ready():
            return
        if self.view_combo.currentIndex() != VIEW_CAKE:
            self.set_view(VIEW_CAKE)
        self.shape_layer.start_drawing(RECTANGLE, purpose=DRAW_PURPOSE)
        self._status(tr("Click two opposite corners on the Cake view: q across, χ up (Esc cancels)."))

    def _region_drawn(self, points: list) -> None:
        cake = getattr(self, "_cake", None)
        if self.view_combo.currentIndex() != VIEW_CAKE or cake is None or cake[0] is not self.view_model.state.analysis:
            self._status(tr("Draw the region once the Cake view is shown."), "warning")
            return
        (q0, c0), (q1, c1) = points
        q_low, q_high = sorted((max(0.0, q0), max(0.0, q1)))
        c_low, c_high = sorted((max(-90.0, min(90.0, c0)), max(-90.0, min(90.0, c1))))
        both = c_low >= 0 or c_high <= 0
        chi_range = tuple(sorted((abs(c_low), abs(c_high)))) if both else (c_low, c_high)
        name = f"Region {len(self.view_model.state.giwaxs.regions) + 1}"
        try:
            region = CutRegion(name, (q_low, q_high), chi_range, both)
        except ValueError as exc:
            self._status(str(exc), "warning")
            return
        index = self.view_model.add_region(region)
        self._pending_region_key = region_key(index)
        self.run_analysis()
        self._status(tr("{name} added: drag it to adjust, or type its centre and width below.").format(name=name), "ok")

    # -- snapping ------------------------------------------------------------------------

    def snap_selected_region(self) -> None:
        row = self._selected_row()
        analysis = self.view_model.state.analysis
        if row is None or row.index is None or analysis is None:
            return
        region = self.view_model.state.giwaxs.regions[row.index]
        self._status(tr("Finding the peak …"))
        self.tasks.submit(
            "region-snap",
            lambda: self.view_model.snap_region(analysis, region),
            on_done=lambda pick, index=row.index, old=region: self._region_snapped(index, old, pick),
            on_error=lambda message, _details: self._status(tr("Could not snap the region: {error}").format(error=message), "error"),
        )

    def _region_snapped(self, index: int, old: CutRegion, pick) -> None:
        regions = self.view_model.state.giwaxs.regions
        if index >= len(regions) or regions[index] != old:
            return  # the region changed meanwhile
        self.view_model.update_region(index, pick.region)
        self._pending_region_key = region_key(index)
        self.run_analysis()
        note = pick_note(pick)
        self._status(note or tr("No peak stands out near the region's centre; it was left as it was."),
                     "ok" if pick.q_peak is not None or pick.chi_peak is not None else "warning")

    # -- cut sets ------------------------------------------------------------------------

    def save_cuts(self, path: str | Path | None = None):
        regions = self.view_model.state.giwaxs.regions
        if not regions:
            self._status(tr("Add a region first: the standard cuts are always there."), "warning")
            return None
        if path is None:
            folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
            path, _ = QFileDialog.getSaveFileName(self, tr("Save Cuts"), str(folder / "cuts.json"), CUT_FILTER)
            if not path:
                return None
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cut_set(regions), indent=2), encoding="utf-8")
        except OSError as exc:
            self._status(tr("Could not save the cuts: {error}").format(error=exc), "error")
            return None
        self.notify_written(tr("Saved {name}").format(name=path.name), path.parent)
        return path

    def load_cuts(self, path: str | Path | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("Load Cuts"), self._last_folder, CUT_FILTER)
            if not path:
                return False
        path = Path(path)
        try:
            regions = regions_from_cut_set(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            self._status(tr("Could not read the cuts: {error}").format(error=exc), "error")
            return False
        self.view_model.set_regions(regions)
        self.run_analysis()
        self._status(tr("{count} regions loaded from {name}").format(count=len(regions), name=path.name), "ok")
        return True


__all__ = ["DRAW_PURPOSE", "PICK_PURPOSE", "RegionPickMixin", "pick_name", "pick_note"]
