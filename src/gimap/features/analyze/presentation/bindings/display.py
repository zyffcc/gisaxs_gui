"""What Analyze draws: the detector or q map with overlays, and the curve plots."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QSignalBlocker

from ...application import (
    GISAXS,
    X_AXIS_TWO_THETA,
    FrameAnalysis,
    q_at_position,
    q_from_two_theta,
    two_theta_deg,
)

LOWER_PROFILES = (
    ("chi", "I(χ) of the regions"),
    ("box_qz", "q box: I(qz)"),
    ("box_qpar", "q box: I(q∥)"),
)
"""GIWAXS profiles the lower plot can show, in menu order."""
CHI_PROFILES = ("azimuthal", "sector_chi")
"""Curves shown with ``chi`` (and every region's I(χ))."""
BOX_COLOR = "#facc15"
MASK_OUTLINE = "#f43f5e"
VIEW_DETECTOR, VIEW_Q_MAP, VIEW_CAKE = range(3)


class DisplayMixin:
    """Own the image view, its overlays and both curve plots."""

    def set_view(self, index: int) -> None:
        """Show the detector (0), the q map (1) or the cake (2), when the frame has it."""
        if index != VIEW_DETECTOR and not self.view_combo.isEnabled():
            return
        with QSignalBlocker(self.view_combo):
            self.view_combo.setCurrentIndex(int(index))
        self._view_chosen(int(index))

    def _show_image(self, *, keep_view: bool) -> None:
        analysis = self.view_model.state.analysis
        view = self.detector_view
        if analysis is None:
            view.clear()
            self.shape_layer.clear()
            return
        reduction = analysis.reduction
        rsm = reduction.reciprocal_space_map if reduction is not None else None
        index = self.view_combo.currentIndex()
        view.set_aspect_locked(index != VIEW_CAKE)
        if index == VIEW_CAKE and reduction is not None and reduction.kind != GISAXS:
            self._show_cake(analysis, keep_view=keep_view)
            return
        if index == VIEW_Q_MAP and rsm is not None:
            (q0, q1), (z0, z1) = rsm.q_parallel_range, rsm.qz_range
            view.set_image(
                np.flipud(rsm.image), rect=(q0, z0, q1 - q0, z1 - z0), y_down=False,
                title=analysis.path.name, x_label=rsm.x_label, y_label="qz (Å⁻¹)", keep_view=keep_view,
                context="qmap",
            )
            view.clear_overlays()
            self._show_q_map_marks(analysis, rsm, keep_view=keep_view)
            box = self.view_model.state.giwaxs.box
            if box is not None:
                view.show_box(box.q_parallel[0], box.qz[0], box.q_parallel[1], box.qz[1])
            self._refresh_shapes()
            return
        view.set_image(
            analysis.data, valid=analysis.valid, title=analysis.path.name,
            x_label="x (pixel)", y_label="y (pixel)", keep_view=keep_view, context="detector",
        )
        self._show_overlays(analysis)
        self._refresh_shapes()

    def _show_q_map_marks(self, analysis: FrameAnalysis, rsm, *, keep_view: bool) -> None:
        """The beam centre (the direct beam: q∥ = qz = 0) and the sample horizon (qz = k sin αi) on the q map."""
        geometry = analysis.geometry
        if geometry is None:
            return
        view = self.detector_view
        axis = "qy" if str(rsm.x_label).startswith("qy") else "q_parallel"
        beam = q_at_position(geometry, geometry.beam_center_x_px, geometry.beam_center_y_px)
        view.show_beam_center(beam[axis], beam["qz"], movable=False, keep_in_view=True)
        horizon = q_at_position(geometry, geometry.beam_center_x_px, geometry.horizon_row())
        view.show_horizon(horizon["qz"])
        if not keep_view:
            view.fit_view()

    # -- the cake (χ against q) -------------------------------------------------------------

    def _show_cake(self, analysis: FrameAnalysis, *, keep_view: bool) -> None:
        cached = getattr(self, "_cake", None)
        if cached is not None and cached[0] is analysis:
            cake = cached[1]
            (q0, q1), (c0, c1) = cake.q_range, cake.chi_range
            self.detector_view.set_image(
                cake.image, valid=np.isfinite(cake.image), rect=(q0, c0, q1 - q0, c1 - c0), y_down=False,
                title=f"{analysis.path.name} — unwrapped", x_label="q (Å⁻¹)", y_label="χ (°)", keep_view=keep_view,
                context="cake",
            )
            self.detector_view.clear_overlays()
            self._refresh_shapes()
            return
        self.detector_view.title_label.setText("Unwrapping onto χ–q …")
        self.tasks.submit(
            "cake",
            lambda: self.view_model.cake(analysis),
            on_done=lambda cake: self._cake_ready(analysis, cake, keep_view),
            on_error=lambda message, _details: self._status(f"Could not unwrap the frame: {message}", "error"),
        )

    def _cake_ready(self, analysis: FrameAnalysis, cake, keep_view: bool) -> None:
        self._cake = (analysis, cake)
        if analysis is self.view_model.state.analysis and self.view_combo.currentIndex() == VIEW_CAKE:
            self._show_cake(analysis, keep_view=keep_view)

    def _refresh_shapes(self) -> None:
        """Masks drawn on the detector; region outlines on the q map; region rectangles on the cake."""
        index = self.view_combo.currentIndex()
        if index == VIEW_DETECTOR:
            outlines = []
            for shape in self.view_model.state.corrections.mask_shapes:
                points = list(shape.points)
                if shape.kind == "rectangle":
                    (x0, y0), (x1, y1) = points
                    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                points.append(points[0])
                outlines.append(([x for x, _y in points], [y for _x, y in points], MASK_OUTLINE))
            self.shape_layer.set_outlines(outlines)
            self.shape_layer.set_rects([])
            return
        outlines, rects = self.region_shapes(index)
        self.shape_layer.set_outlines(outlines)
        self.shape_layer.set_rects(rects)

    def _show_overlays(self, analysis: FrameAnalysis) -> None:
        view = self.detector_view
        view.clear_overlays()
        reduction = analysis.reduction
        if reduction is None:
            return
        markers = reduction.markers
        view.show_beam_center(*markers["beam_center"])
        view.show_horizon(markers.get("horizon_row"))
        if reduction.kind == GISAXS:
            view.show_horizontal_band(*markers["horizontal_band"])
            view.show_vertical_band(*markers["vertical_band"])

    def _show_curves(self, analysis: FrameAnalysis) -> None:
        top, bottom = self.top_plot, self.bottom_plot
        top.hide_window()
        reduction = analysis.reduction
        self.lower_choice.setVisible(reduction is not None and reduction.kind != GISAXS)
        self._plot_keys = {"top": [], "bottom": []}
        if reduction is None:
            top.clear_curves()
            bottom.clear_curves()
            top.set_title("No curves yet: the frame needs a geometry (step 2)")
            bottom.set_title("")
            return
        if reduction.kind == GISAXS:
            self._plot_keys = {"top": ["horizontal"], "bottom": ["vertical"]}
            horizontal, vertical = reduction.curve("horizontal"), reduction.curve("vertical")
            yoneda = reduction.markers.get("yoneda")
            source = reduction.markers.get("horizontal_source")
            where = (
                f"Yoneda, αf = {yoneda.alpha_f_deg:.3f}°" if source == "yoneda" and yoneda else source
            )
            top.set_title(f"Horizontal cut I(qy) · rows {horizontal.region['rows']} ({where})")
            top.set_labels(horizontal.x_label, horizontal.y_label)
            top.set_curves([("I(qy)", horizontal.x, horizontal.intensity)])
            bottom.set_title(f"Vertical cut I(qz) · columns {vertical.region['columns']}")
            bottom.set_labels(vertical.x_label, vertical.y_label)
            bottom.set_curves([("I(qz)", vertical.x, vertical.intensity)])
            return
        self._show_giwaxs_curves(analysis)

    def _region_curves(self, reduction, keys) -> list:
        """``(name, curve, colour)`` of the visible regions' curves among ``keys``, in list order."""
        colors = self.region_colors()
        names = {
            key: row.name for row in self._region_rows for key in (row.q_curve, row.chi_curve) if key
        }
        shown = []
        for key in keys:
            curve = reduction.curve(key)
            if curve is not None and not curve.is_empty and self.visible_region(key):
                shown.append((names.get(key, curve.title), curve, colors.get(key)))
        return shown

    def _show_giwaxs_curves(self, analysis: FrameAnalysis) -> None:
        top = self.top_plot
        reduction = analysis.reduction
        self._refresh_regions(analysis)
        q_keys = [row.q_curve for row in self._region_rows if row.q_curve]
        if reduction.curve("box_q") is not None:
            q_keys.append("box_q")
        family = self._region_curves(reduction, q_keys)
        top.set_title("I(q) of the regions — drag the purple band to move the ring of I(χ)")
        x_label = family[0][1].x_label if family else "q (Å⁻¹)"
        top.set_labels(x_label, "I (counts/pixel)")
        top.set_curves(
            [("q box" if curve.key == "box_q" else name, curve.x, curve.intensity) for name, curve, _color in family],
            [color or BOX_COLOR for _name, _curve, color in family],
        )
        self._plot_keys["top"] = [curve.key for _name, curve, _color in family]
        window = reduction.markers.get("chi_q_window")
        if window is not None:
            top.show_window(*self._window_for_display(window, analysis))
        self._fill_lower_choice(reduction)
        self._show_lower_profile()

    def _fill_lower_choice(self, reduction) -> None:
        wanted = self.lower_choice.currentData() or self._lower_profile
        if wanted in CHI_PROFILES:
            wanted = "chi"
        with QSignalBlocker(self.lower_choice):
            self.lower_choice.clear()
            for key, title in LOWER_PROFILES:
                if key == "chi" or reduction.curve(key) is not None:
                    self.lower_choice.addItem(title, key)
            index = self.lower_choice.findData(wanted)
            self.lower_choice.setCurrentIndex(max(0, index))

    def _lower_profile_chosen(self, _index: int) -> None:
        self._lower_profile = self.lower_choice.currentData()
        self._show_lower_profile()

    def _show_lower_profile(self) -> None:
        bottom = self.bottom_plot
        analysis = self.view_model.state.analysis
        reduction = analysis.reduction if analysis is not None else None
        key = self.lower_choice.currentData()
        if reduction is not None and key == "chi":
            chi_keys = [row.chi_curve for row in self._region_rows if row.chi_curve]
            shown = self._region_curves(reduction, chi_keys)
            if not shown:
                bottom.clear_curves()
                bottom.set_title("I(χ): no ring or region selected")
                self._plot_keys["bottom"] = []
                return
            bottom.set_title("I(χ) of the regions")
            labels = {curve.x_label for _name, curve, _color in shown}
            bottom.set_labels(labels.pop() if len(labels) == 1 else "χ or |χ| (°)", "I (counts/pixel)")
            bottom.set_curves([(name, curve.x, curve.intensity) for name, curve, _color in shown],
                              [color for _name, _curve, color in shown])
            self._plot_keys["bottom"] = [curve.key for _name, curve, _color in shown]
            return
        curve = reduction.curve(key) if reduction is not None and key else None
        if curve is None:
            bottom.clear_curves()
            bottom.set_title("I(χ): no ring selected")
            self._plot_keys["bottom"] = []
            return
        bottom.set_title(curve.title)
        bottom.set_labels(curve.x_label, curve.y_label)
        bottom.set_curves([(curve.title.split(",")[0], curve.x, curve.intensity)], [BOX_COLOR])
        self._plot_keys["bottom"] = [curve.key]

    # -- q ↔ 2θ for the I(χ) window ------------------------------------------------------

    def _uses_two_theta(self) -> bool:
        return self.view_model.state.giwaxs.x_axis == X_AXIS_TWO_THETA

    def _window_for_display(self, window, analysis: FrameAnalysis):
        if not self._uses_two_theta() or analysis.geometry is None:
            return window
        return tuple(two_theta_deg(np.asarray(window), analysis.geometry.wavelength_angstrom))

    def _chi_window_moved(self, low: float, high: float) -> None:
        analysis = self.view_model.state.analysis
        if self._uses_two_theta() and analysis is not None and analysis.geometry is not None:
            low, high = q_from_two_theta(np.array([low, high]), analysis.geometry.wavelength_angstrom)
        self.view_model.set_chi_window(float(low), float(high))
        self.run_analysis()

    # -- cursor readout ------------------------------------------------------------------

    def _readout(self, x: float, y: float) -> str:
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.geometry is None:
            return ""
        if self.view_combo.currentIndex() == VIEW_Q_MAP:
            return f"q∥ = {x:.4f}, qz = {y:.4f}, q = {np.hypot(x, y):.4f} Å⁻¹"
        if self.view_combo.currentIndex() == VIEW_CAKE:
            return f"q = {x:.4f} Å⁻¹, χ = {y:.1f}°"
        q = q_at_position(analysis.geometry, x, y)
        return (
            f"qy = {q['qy']:.4f}, qz = {q['qz']:.4f}, q = {q['q']:.4f} Å⁻¹, "
            f"αf = {q['alpha_f_deg']:.3f}°"
        )


__all__ = ["DisplayMixin", "LOWER_PROFILES", "VIEW_CAKE", "VIEW_DETECTOR", "VIEW_Q_MAP"]
