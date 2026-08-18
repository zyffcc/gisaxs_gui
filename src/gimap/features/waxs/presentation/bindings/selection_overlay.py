"""Selection Overlay coordination for WAXS."""

from __future__ import annotations


import numpy as np


from PyQt5.QtWidgets import (
    QComboBox,
    QMessageBox,
)


from matplotlib.patches import Circle, Rectangle, Wedge

from matplotlib.widgets import RectangleSelector


class SelectionOverlayMixin:
    """Own selection overlay presentation behavior."""

    def _sync_selection_defaults_to_image(self) -> None:
        if self.current_image is None:
            return
        height, width = self.current_image.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0
        default_width = max(10.0, width * 0.25)
        default_height = max(4.0, height * 0.03)
        default_outer = max(10.0, min(width, height) * 0.25)
        default_inner = max(0.0, default_outer * 0.5)

        for spin, value in (
            (self.center_x_spin, center_x),
            (self.center_y_spin, center_y),
            (self.line_center_x_spin, center_x),
            (self.line_center_y_spin, center_y),
            (self.line_width_spin, default_width),
            (self.line_height_spin, default_height),
            (self.circle_center_x_spin, center_x),
            (self.circle_center_y_spin, center_y),
            (self.circle_inner_spin, default_inner),
            (self.circle_outer_spin, default_outer),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _set_values_without_refresh(self, updates: tuple[tuple[object, object], ...]) -> None:
        for widget, value in updates:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _set_combo_text_without_refresh(self, combo: QComboBox, text: str) -> None:
        combo.blockSignals(True)
        combo.setCurrentText(text)
        combo.blockSignals(False)

    def _on_cut_type_changed(self) -> None:
        self._update_cut_tool_visibility()
        self.refresh_view()

    def _update_cut_tool_visibility(self) -> None:
        if not hasattr(self, "_roi_layout"):
            return
        active = self.cut_type_combo.currentText() if hasattr(self, "cut_type_combo") else "Q Range"
        for controls, visible in (
            (getattr(self, "_q_range_controls", ()), active == "Q Range"),
            (getattr(self, "_line_cut_controls", ()), active == "Line Cut"),
            (getattr(self, "_circle_cut_controls", ()), active == "Circle Cut"),
        ):
            for widget in controls:
                widget.setVisible(visible)
                label = self._roi_layout.labelForField(widget)
                if label is not None:
                    label.setVisible(visible)

    def _draw_overlays(self) -> None:
        if self.current_image is None:
            return
        ax = self.viewer.ax
        if self.show_center_check.isChecked() and not self._current_view_is_cut:
            center_x = self.center_x_spin.value()
            center_y = self.center_y_spin.value()
            ax.plot(
                center_x, center_y, marker="+", color="#22d3ee", markersize=14, markeredgewidth=2.0
            )

        if self.show_cut_region_check.isChecked():
            cut_type = self.cut_type_combo.currentText()
            if cut_type == "Line Cut" and not self._current_view_is_cut:
                x0, y0, width, height = self._line_region()
                ax.add_patch(
                    Rectangle(
                        (x0, y0), width, height, fill=False, edgecolor="#f97316", linewidth=1.8
                    )
                )
                ax.plot(
                    self.line_center_x_spin.value(),
                    self.line_center_y_spin.value(),
                    marker="x",
                    color="#f97316",
                    markersize=9,
                )
            elif cut_type == "Circle Cut" and not self._current_view_is_cut:
                cx = self.circle_center_x_spin.value()
                cy = self.circle_center_y_spin.value()
                inner = self.circle_inner_spin.value()
                outer = self.circle_outer_spin.value()
                start = self.circle_start_spin.value()
                end = self.circle_end_spin.value()
                if end < start:
                    end += 360.0
                ax.add_patch(
                    Wedge(
                        (cx, cy),
                        outer,
                        start,
                        end,
                        width=max(outer - inner, 1e-6),
                        fill=False,
                        edgecolor="#a855f7",
                        linewidth=1.8,
                    )
                )
                ax.add_patch(Circle((cx, cy), 3, fill=True, color="#a855f7"))
            elif cut_type == "Q Range" and self._current_view_is_cut:
                x0 = None if self.qr_min_spin.value() == -121.0 else self.qr_min_spin.value()
                x1 = None if self.qr_max_spin.value() == -121.0 else self.qr_max_spin.value()
                y0 = None if self.qz_min_spin.value() == -121.0 else self.qz_min_spin.value()
                y1 = None if self.qz_max_spin.value() == -121.0 else self.qz_max_spin.value()
                if None not in (x0, x1, y0, y1):
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            fill=False,
                            edgecolor="#f97316",
                            linewidth=1.8,
                        )
                    )

        self.viewer.canvas.draw_idle()

    def reset_mask(self) -> None:
        self._set_values_without_refresh(
            (
                (self.mask_min_spin, -1e12),
                (self.mask_max_spin, 1e12),
            )
        )
        self.apply_mask_check.blockSignals(True)
        self.apply_mask_check.setChecked(True)
        self.apply_mask_check.blockSignals(False)
        self.refresh_view()

    def apply_cut(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Apply Cut", "No image loaded.")
            return
        self._current_view_is_cut = True
        self._show_2d_view()
        self.refresh_view()

    def clear_cut(self) -> None:
        self._current_view_is_cut = False
        self._set_values_without_refresh(
            (
                (self.qr_min_spin, -121.0),
                (self.qr_max_spin, -121.0),
                (self.qz_min_spin, -121.0),
                (self.qz_max_spin, -121.0),
            )
        )
        self._show_2d_view()
        self.refresh_view()

    def _select_roi_hint(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "ROI Selection", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._show_2d_view()
        self.refresh_view()
        if self._roi_selector is not None:
            self._roi_selector.set_active(False)
            self._roi_selector = None
        self._roi_selector = RectangleSelector(
            self.viewer.ax,
            self._on_roi_selected,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
        )
        self._set_status("Drag a rectangle on the detector image to select a Q-range ROI.")
        self.viewer.canvas.draw_idle()

    def start_line_cut_selection(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Line Cut", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._set_combo_text_without_refresh(self.cut_type_combo, "Line Cut")
        self._update_cut_tool_visibility()
        self._show_2d_view()
        self.refresh_view()
        self._roi_selector = RectangleSelector(
            self.viewer.ax,
            self._on_line_cut_selected,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
        )
        self._set_status("Drag any rectangle on the image to define the line cut region.")
        self.viewer.canvas.draw_idle()

    def start_circle_cut_selection(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Circle Cut", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._set_combo_text_without_refresh(self.cut_type_combo, "Circle Cut")
        self._update_cut_tool_visibility()
        self._show_2d_view()
        self.refresh_view()
        self._circle_pick_points = [
            (self.circle_center_x_spin.value(), self.circle_center_y_spin.value())
        ]
        self._circle_pick_cid = self.viewer.canvas.mpl_connect(
            "button_press_event", self._on_circle_pick
        )
        self._set_status("Circle Cut: click inner/start point, then outer/end point.")

    def start_center_pick(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Pick Center", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._show_2d_view()
        self.refresh_view()
        self._center_pick_cid = self.viewer.canvas.mpl_connect(
            "button_press_event", self._on_center_pick
        )
        self._set_status("Pick Center: click the detector image to set the center.")

    def _cancel_interactive_tools(self) -> None:
        if self._roi_selector is not None:
            self._roi_selector.set_active(False)
            self._roi_selector = None
        if self._circle_pick_cid is not None:
            self.viewer.canvas.mpl_disconnect(self._circle_pick_cid)
            self._circle_pick_cid = None
        if self._center_pick_cid is not None:
            self.viewer.canvas.mpl_disconnect(self._center_pick_cid)
            self._center_pick_cid = None

    def _on_line_cut_selected(self, press_event, release_event) -> None:
        if (
            press_event.xdata is None
            or press_event.ydata is None
            or release_event.xdata is None
            or release_event.ydata is None
        ):
            return
        x0, x1 = sorted([float(press_event.xdata), float(release_event.xdata)])
        y0, y1 = sorted([float(press_event.ydata), float(release_event.ydata)])
        self._set_values_without_refresh(
            (
                (self.line_center_x_spin, (x0 + x1) / 2.0),
                (self.line_center_y_spin, (y0 + y1) / 2.0),
                (self.line_width_spin, max(1.0, x1 - x0)),
                (self.line_height_spin, max(1.0, y1 - y0)),
            )
        )
        self._cancel_interactive_tools()
        self.refresh_view()
        self._set_status("Line cut region selected.")

    def _on_center_pick(self, event) -> None:
        if event.inaxes != self.viewer.ax or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        self._set_values_without_refresh(
            (
                (self.center_x_spin, x),
                (self.line_center_x_spin, x),
                (self.circle_center_x_spin, x),
                (self.center_y_spin, y),
                (self.line_center_y_spin, y),
                (self.circle_center_y_spin, y),
            )
        )
        self._cancel_interactive_tools()
        self.refresh_view()
        self._set_status(f"Center picked: X={x:.2f}, Y={y:.2f}")

    def _on_circle_pick(self, event) -> None:
        if event.inaxes != self.viewer.ax or event.xdata is None or event.ydata is None:
            return
        self._circle_pick_points.append((float(event.xdata), float(event.ydata)))
        if len(self._circle_pick_points) == 1:
            self._set_values_without_refresh(
                (
                    (self.circle_center_x_spin, self._circle_pick_points[0][0]),
                    (self.circle_center_y_spin, self._circle_pick_points[0][1]),
                )
            )
            self._set_status("Circle Cut: click inner/start point.")
            self.refresh_view()
            return
        if len(self._circle_pick_points) == 2:
            cx, cy = self._circle_pick_points[0]
            x, y = self._circle_pick_points[1]
            self._set_values_without_refresh(
                (
                    (self.circle_inner_spin, max(0.0, float(np.hypot(x - cx, y - cy)))),
                    (self.circle_start_spin, self._angle_from_center(cx, cy, x, y)),
                )
            )
            self._set_status("Circle Cut: click outer/end point.")
            self.refresh_view()
            return

        cx, cy = self._circle_pick_points[0]
        x, y = self._circle_pick_points[2]
        outer = max(self.circle_inner_spin.value() + 1.0, float(np.hypot(x - cx, y - cy)))
        self._set_values_without_refresh(
            (
                (self.circle_outer_spin, outer),
                (self.circle_end_spin, self._angle_from_center(cx, cy, x, y)),
            )
        )
        self._cancel_interactive_tools()
        self.refresh_view()
        self._set_status("Circle cut region selected.")

    @staticmethod
    def _angle_from_center(cx: float, cy: float, x: float, y: float) -> float:
        return float(np.degrees(np.arctan2(y - cy, x - cx)))

    def _line_region(self) -> tuple[float, float, float, float]:
        width = max(1.0, self.line_width_spin.value())
        height = max(1.0, self.line_height_spin.value())
        x0 = self.line_center_x_spin.value() - width / 2.0
        y0 = self.line_center_y_spin.value() - height / 2.0
        return x0, y0, width, height

    def _on_roi_selected(self, press_event, release_event) -> None:
        if self.current_image is None:
            return
        if (
            press_event.xdata is None
            or press_event.ydata is None
            or release_event.xdata is None
            or release_event.ydata is None
        ):
            return

        x0, x1 = sorted([float(press_event.xdata), float(release_event.xdata)])
        y0, y1 = sorted([float(press_event.ydata), float(release_event.ydata)])

        if self._current_view_is_cut:
            self.qr_min_spin.setValue(x0)
            self.qr_max_spin.setValue(x1)
            self.qz_min_spin.setValue(y0)
            self.qz_max_spin.setValue(y1)
        else:
            height, width = self.current_image.shape[:2]
            col0 = max(0, min(width - 1, int(np.floor(x0))))
            col1 = max(0, min(width - 1, int(np.ceil(x1))))
            row0 = max(0, min(height - 1, int(np.floor(y0))))
            row1 = max(0, min(height - 1, int(np.ceil(y1))))
            if row1 < row0:
                row0, row1 = row1, row0
            if col1 < col0:
                col0, col1 = col1, col0
            qr, qz = self.view_model.compute_q_maps(
                self.current_image.shape,
                self._geometry_settings(),
            )
            roi_qr = qr[row0 : row1 + 1, col0 : col1 + 1]
            roi_qz = qz[row0 : row1 + 1, col0 : col1 + 1]
            if np.isfinite(roi_qr).any() and np.isfinite(roi_qz).any():
                self.qr_min_spin.setValue(float(np.nanmin(roi_qr)))
                self.qr_max_spin.setValue(float(np.nanmax(roi_qr)))
                self.qz_min_spin.setValue(float(np.nanmin(roi_qz)))
                self.qz_max_spin.setValue(float(np.nanmax(roi_qz)))

        if self._roi_selector is not None:
            self._roi_selector.set_active(False)
            self._roi_selector = None
        self._current_view_is_cut = True
        self.refresh_view()
        self._set_status("ROI selected and Q-range cut applied.")
