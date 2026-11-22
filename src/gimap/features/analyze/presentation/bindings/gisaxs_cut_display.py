"""What the GISAXS cuts draw: their curves in the two plots, their outlines on the detector and the q map, and cuts
drawn on either view.

* **Curves.** The upper plot holds every cut along q_y (the automatic horizontal cut and the person's horizontal bands
  and boxes along q_y), the lower plot every cut along q_z; sectors (I(q_yz)) and lines (I(s)) are in the lower plot's
  list when there are any. With only the automatic cuts the plots are as they always were (one curve each).
* **Outlines.** On the detector the person's cuts are drawn where they lie (q mapped back to pixels: a band of
  constant q_z curves a little); the automatic bands are the draggable bands they always were. On the q map every cut
  is drawn, the automatic bands as the region of whole pixels they average; bands and boxes of the person's can be
  dragged and resized there.
* **Drawing.** Horizontal / Vertical / Sector take one click, Box two corners, Line its two ends, on the detector or
  the q map (``shape_layer`` with its own purpose).
"""

from __future__ import annotations

from dataclasses import replace

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.components import POINT
from src.gimap.app.presentation.components.shape_layer import LINE, RECTANGLE
from src.gimap.app.presentation.i18n import tr

from ...application import GISAXS, gisaxs_cut_tools as tools
from ..texts import message_text

AUTO_COLORS = {"horizontal": "#f97316", "vertical": "#0891b2"}
"""The automatic cuts: the colours of their bands on the detector (orange; cyan, darker to read on white)."""
DRAW_PURPOSE = "gisaxs-cut"
DRAW_SHAPES = {"horizontal": POINT, "vertical": POINT, "sector": POINT, "box": RECTANGLE, "line": LINE}
DRAW_HINTS = {
    "horizontal": "Click the height of the band on the image or the q map (Esc cancels).",
    "vertical": "Click the position of the band on the image or the q map (Esc cancels).",
    "box": "Click two opposite corners of the box (Esc cancels).",
    "sector": "Click in the direction of the sector (Esc cancels).",
    "line": "Click the two ends of the line (Esc cancels).",
}
LOWER_GISAXS = (("g_qz", "Cuts along qz"), ("g_sector", "Sectors"), ("g_line", "Lines"))
"""What the lower plot of a GISAXS frame can show (its list, there when the frame has sectors or lines)."""
LOWER_TITLES = {"g_qz": "I(qz) of the cuts", "g_sector": "I(qyz) of the sectors", "g_line": "I(s) along the lines"}
LOWER_AXES = {"g_qz": "qz", "g_sector": "sector", "g_line": "line"}
LOWER_LABELS = {"g_qz": "qz (Å⁻¹)", "g_sector": "qyz (Å⁻¹)", "g_line": "s (Å⁻¹)"}
VIEW_DETECTOR, VIEW_Q_MAP = 0, 1


class GisaxsCutDisplayMixin:
    """Needs ``top_plot``, ``bottom_plot``, ``lower_choice``, ``shape_layer``, ``detector_view``, ``view_combo`` and the
    cuts table (``GisaxsCutsMixin``)."""

    def _connect_cut_display(self) -> None:
        self._cut_draw_kind = ""
        self._gisaxs_lower_choice = "g_qz"
        for key, button in self.cut_draw_buttons.items():
            button.toggled.connect(lambda on, key=key: self._cut_draw_toggled(key, on))
        self.shape_layer.shapePicked.connect(self._cut_shape_picked)
        self.shape_layer.drawingChanged.connect(self._cut_drawing_changed)
        self.shape_layer.rectChanged.connect(self._cut_rect_moved)

    # -- curves ----------------------------------------------------------------------------

    def _cut_curves(self, reduction, axis: str):
        """``(names, curves, colours, keys)`` of the visible cuts along ``axis`` that have data, in table order."""
        shown = [(row, reduction.curve(row.key)) for row in self._visible_cut_rows() if row.axis == axis]
        shown = [(row, curve) for row, curve in shown if curve is not None and not curve.is_empty]
        names = [tr(row.name) if row.auto else row.name for row, _curve in shown]
        return names, [curve for _row, curve in shown], [row.color for row, _curve in shown], [row.key for row, _ in shown]

    def _show_gisaxs_curves(self, analysis) -> None:
        from .display import _horizontal_where, cut_lines, pixel_range

        top = self.top_plot
        reduction = analysis.reduction
        self._check_cut_setup(analysis)
        self._refresh_cut_rows(analysis)
        lines = cut_lines(analysis)
        names, curves, colors, keys = self._cut_curves(reduction, "qy")
        horizontal = reduction.curve("horizontal")
        if "horizontal" in keys and "gisaxs_cut" in horizontal.region:
            top.set_title(lines[0])
            top.title_label.setToolTip(lines[0])
        elif "horizontal" in keys:  # short title (the header also holds the halves, Save and Log); the sentence in the tooltip
            top.set_title(tr("I(qy) · rows {rows} · {where}").format(
                rows=pixel_range(*horizontal.region["rows"]), where=_horizontal_where(reduction, short=True)))
            top.title_label.setToolTip(lines[0])
        else:
            top.set_title("I(qy) of the cuts")
        top.set_labels("qy (Å⁻¹)", horizontal.y_label)
        if keys == ["horizontal"]:
            names = ["I(qy)"]  # the one curve, as it always was
        top.set_curves([(name, curve.x, curve.intensity) for name, curve in zip(names, curves)], colors)
        self._plot_keys["top"] = keys
        self._fill_gisaxs_lower(reduction)
        self._show_gisaxs_lower(analysis)

    def _fill_gisaxs_lower(self, reduction) -> None:
        present = {row.axis for row in self._cut_rows if not row.auto}
        items = [(key, title) for key, title in LOWER_GISAXS if key == "g_qz" or LOWER_AXES[key] in present]
        with QSignalBlocker(self.lower_choice):
            self.lower_choice.clear()
            for key, title in items:
                self.lower_choice.addItem(tr(title), key)
            self.lower_choice.setCurrentIndex(max(0, self.lower_choice.findData(self._gisaxs_lower_choice)))
        self.lower_choice.setToolTip(tr("Which cuts the lower plot shows"))
        self.lower_choice.setVisible(len(items) > 1)

    def _gisaxs_lower(self) -> bool:
        """The lower plot of a GISAXS frame after a choice in its list (``True``: drawn here)."""
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None or analysis.kind != GISAXS:
            return False
        self._gisaxs_lower_choice = self.lower_choice.currentData() or "g_qz"
        self._show_gisaxs_lower(analysis)
        self._refresh_overlay()
        return True

    def _show_gisaxs_lower(self, analysis) -> None:
        from .display import cut_lines, pixel_range

        bottom = self.bottom_plot
        reduction = analysis.reduction
        choice = self.lower_choice.currentData() if not self.lower_choice.isHidden() else "g_qz"
        choice = choice if choice in LOWER_AXES else "g_qz"
        names, curves, colors, keys = self._cut_curves(reduction, LOWER_AXES[choice])
        vertical = reduction.curve("vertical")
        if choice == "g_qz" and "vertical" in keys and "gisaxs_cut" in vertical.region:
            line = cut_lines(analysis)[-1]
            bottom.set_title(line)
            bottom.title_label.setToolTip(line)
        elif choice == "g_qz" and "vertical" in keys:
            bottom.set_title(tr("I(qz) · columns {columns}").format(columns=pixel_range(*vertical.region["columns"])))
            bottom.title_label.setToolTip(cut_lines(analysis)[-1])
        else:
            bottom.set_title(LOWER_TITLES[choice])
        bottom.set_labels(LOWER_LABELS[choice], vertical.y_label)
        legend = [False] if keys == ["vertical"] else None  # one automatic curve: the title names it
        bottom.set_curves([(name, curve.x, curve.intensity) for name, curve in zip(names, curves)], colors, legend=legend)
        self._plot_keys["bottom"] = keys

    # -- outlines ---------------------------------------------------------------------------------

    def gisaxs_cut_shapes(self, view: int) -> tuple[list, list]:
        """``(outlines, rects)`` of the visible cuts for the detector (0) or the q map (1)."""
        analysis, geometry = self.view_model.state.analysis, self._cut_geometry()
        if geometry is None:
            return [], []
        extent = tools.q_extent(analysis)
        outlines, rects = [], []
        for row in self._visible_cut_rows():
            if row.auto:
                curve = analysis.reduction.curve(row.key)
                if curve is not None and "gisaxs_cut" in curve.region and extent is not None:
                    cut = tools.cut_from_record(curve.region)
                    if view == VIEW_DETECTOR:
                        outlines.extend((x, y, row.color) for x, y in tools.detector_outlines(cut, geometry, extent))
                    else:
                        rects.append((f"g:{row.key}", *tools.cut_rect(cut, extent), row.color, True))
                elif view == VIEW_Q_MAP and curve is not None and "rows" in curve.region:
                    qy, qz = tools.band_outline_q(geometry, curve.region["rows"], curve.region["columns"])
                    outlines.append((qy, qz, row.color))
                continue
            if extent is None:
                continue
            if view == VIEW_DETECTOR:
                outlines.extend((x, y, row.color) for x, y in tools.detector_outlines(row.cut, geometry, extent))
                continue
            rect = tools.cut_rect(row.cut, extent)
            if rect is not None:
                rects.append((f"g:{row.key}", *rect, row.color, True))
            else:
                outlines.extend((qy, qz, row.color) for qy, qz in tools.q_outlines(row.cut, extent))
        return outlines, rects

    def _hide_unticked_bands(self) -> None:
        """The automatic bands on the detector follow their ticks."""
        view = self.detector_view
        for key, band in (("horizontal", view.horizontal_band), ("vertical", view.vertical_band)):
            if key in self._hidden_cuts:
                view.marks.track("bands", band, False)

    def _cut_rect_moved(self, key: str, x0: float, y0: float, x1: float, y1: float) -> None:
        if not key.startswith("g:"):
            return
        row = next((item for item in self._cut_rows if item.key == key[2:]), None)
        if row is None:
            return
        if row.auto:
            geometry = self._cut_geometry()
            if row.kind == tools.HORIZONTAL:
                self.view_model.set_horizontal_band(*tools.range_to_unit(geometry, tools.UNIT_PIXEL, "qz", (y0, y1)))
            else:
                self.view_model.set_vertical_band(*tools.range_to_unit(geometry, tools.UNIT_PIXEL, "qy", (x0, x1)))
            self._pending_cut_key = row.key
            self.run_analysis()
            return
        cut = row.cut
        try:
            if row.kind == tools.HORIZONTAL:
                cut = replace(cut, qz=(y0, y1))
            elif row.kind == tools.VERTICAL:
                cut = replace(cut, qy=(x0, x1))
            else:
                cut = replace(cut, qy=(x0, x1), qz=(y0, y1))
        except ValueError as exc:
            self._status(message_text(exc), "warning")
            self._refresh_shapes()
            return
        self.view_model.update_gisaxs_cut(row.index, cut)
        self._pending_cut_key = row.key
        self.run_analysis()

    # -- drawing ------------------------------------------------------------------------------------

    def _set_cut_draw_buttons(self, kind: str) -> None:
        for key, button in self.cut_draw_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(key == kind)

    def _cut_draw_toggled(self, kind: str, on: bool) -> None:
        if not on:
            if self.shape_layer.purpose == DRAW_PURPOSE and self._cut_draw_kind == kind:
                self.shape_layer.cancel_drawing()
            return
        if not self._gisaxs_ready():
            self._set_cut_draw_buttons("")
            return
        self.draw_gisaxs_cut(kind)

    def draw_gisaxs_cut(self, kind: str) -> None:
        """Arm drawing a cut of ``kind`` on the detector or the q map."""
        self._cut_draw_kind = kind
        self.shape_layer.start_drawing(DRAW_SHAPES[kind], purpose=DRAW_PURPOSE)
        self._set_cut_draw_buttons(kind)
        self._status(tr(DRAW_HINTS[kind]))

    def _cut_drawing_changed(self, shape: str) -> None:
        drawing = bool(shape) and self.shape_layer.purpose == DRAW_PURPOSE
        self._set_cut_draw_buttons(self._cut_draw_kind if drawing else "")

    def _cut_shape_picked(self, purpose: str, _shape: str, points: list) -> None:
        if purpose != DRAW_PURPOSE or not points:
            return
        geometry = self._cut_geometry()
        if geometry is None:
            return
        view = self.view_combo.currentIndex()
        if view == VIEW_DETECTOR:
            points = [tools.q_at(geometry, x, y) for x, y in points]
        elif view != VIEW_Q_MAP:
            return
        kind, name = self._cut_draw_kind, self._cut_name(self._cut_draw_kind)
        try:
            if len(points) == 1:
                cut = tools.cut_at(kind, geometry, *points[0], name=name, extent=tools.q_extent(self.view_model.state.analysis))
            else:
                cut = tools.cut_between(kind, geometry, points[0], points[1], name=name)
        except ValueError as exc:
            self._status(message_text(exc), "warning")
            return
        self.add_gisaxs_cut(kind, cut)


__all__ = ["AUTO_COLORS", "DRAW_HINTS", "DRAW_PURPOSE", "GisaxsCutDisplayMixin", "LOWER_GISAXS"]
