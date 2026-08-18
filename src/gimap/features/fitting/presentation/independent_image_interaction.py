"""Independent Image Interaction behavior."""

from __future__ import annotations


import copy

import numpy as np

from PyQt5.QtCore import Qt


from src.gimap.app.presentation.responsive_layout import (
    apply_density_profile,
)


from src.gimap.app.presentation.responsive_layout import (
    apply_density_profile,
)


class IndependentImageInteractionMixin:
    """Own independent image interaction behavior."""

    def _apply_screen_profile(self, profile, screen):
        apply_density_profile(self, profile)

    def _on_xlim_changed(self, ax):
        """No description."""
        self.current_xlim = ax.get_xlim()

    def _on_ylim_changed(self, ax):
        """No description."""
        self.current_ylim = ax.get_ylim()

    def _on_mouse_press(self, event):
        """No description."""
        if event.button == 1 and self.pick_center_mode and event.inaxes == self.ax:
            self._emit_picked_center(event)
            return

        if event.button == 3:
            if not self.selection_mode:
                if self.pick_center_mode:
                    self._exit_pick_center_mode()
                self.selection_mode = True
                self.setWindowTitle(self.SELECTION_TITLE)
                self.canvas.setCursor(Qt.CrossCursor)
                self.status_updated.emit(
                    "Selection mode activated - Drag to select, Right-click again to exit"
                )
            else:
                self._exit_selection_mode()
            return

        if event.button == 1 and self.selection_mode and event.inaxes == self.ax:
            self.selection_start = (event.xdata, event.ydata)
            if self.selection_rect:
                self.selection_rect.remove()
                self.selection_rect = None
            self.status_updated.emit("Selection started - drag to define region")
            return

        if event.button == 1 and not self.selection_mode and event.inaxes == self.ax:
            hit = self._hit_test_overlay(event)
            if hit:
                self._dragging_overlay = hit
                self._drag_start = (float(event.xdata), float(event.ydata))
                self._drag_original_info = copy.deepcopy(self.parameter_selection_info) or {}
                self._drag_original_center = self._get_detector_center_display_coordinates()
                self._drag_current_center = None
                if self.canvas is not None:
                    self.canvas.setCursor(Qt.ClosedHandCursor)
                self.status_updated.emit(
                    "Drag Cut Region" if hit == "region" else "Drag Detector Beam Center"
                )

    def _emit_picked_center(self, event):
        try:
            if event.xdata is None or event.ydata is None:
                return
            x_value = float(event.xdata)
            y_value = float(event.ydata)
            show_q_axis = self._should_show_q_axis()
            payload = {
                "is_q_space": bool(show_q_axis),
                "x": x_value,
                "y": y_value,
            }
            if show_q_axis:
                pixel_coords = self._convert_q_to_pixel_coordinates(x_value, y_value, 1e-9, 1e-9)
                payload.update(
                    {
                        "center_qy": x_value,
                        "center_qz": y_value,
                        "beam_center_x": float(pixel_coords.get("center_x", 0.0)),
                        "beam_center_y": float(pixel_coords.get("center_y", 0.0)),
                    }
                )
                self.status_updated.emit(
                    f"Picked center at q=({x_value:.6g}, {y_value:.6g}) -> "
                    f"pixel=({payload['beam_center_x']:.2f}, {payload['beam_center_y']:.2f})"
                )
            else:
                payload.update(
                    {
                        "beam_center_x": x_value,
                        "beam_center_y": y_value,
                    }
                )
                self.status_updated.emit(
                    f"Picked detector center: X={x_value:.2f}, Y={y_value:.2f}"
                )
            try:
                if self.ax is not None:
                    self._drag_current_center = (
                        x_value,
                        y_value,
                        float(payload.get("beam_center_x", x_value)),
                        float(payload.get("beam_center_y", y_value)),
                    )
                    self._set_center_visible(True, emit=True)
            except Exception:
                pass
            self.center_picked.emit(payload)
            self._exit_pick_center_mode()
        except Exception as exc:
            self.status_updated.emit(f"Pick center failed: {exc}")
            self._exit_pick_center_mode()

    def _on_mouse_move(self, event):
        """No description."""
        if (
            self._dragging_overlay
            and self._drag_start
            and self._drag_original_info is not None
            and event.inaxes == self.ax
        ):
            if event.xdata is None or event.ydata is None:
                return
            dx = float(event.xdata) - self._drag_start[0]
            dy = float(event.ydata) - self._drag_start[1]
            if self._dragging_overlay == "center":
                self._move_detector_center_marker(dx, dy)
            else:
                self._move_parameter_selection(dx, dy)
            return

        if not self.selection_mode and event.inaxes == self.ax:
            hit = self._hit_test_overlay(event)
            if self.canvas is not None:
                self.canvas.setCursor(Qt.OpenHandCursor if hit else Qt.ArrowCursor)

        if (
            self.selection_mode
            and self.selection_start
            and event.inaxes == self.ax
            and event.xdata
            and event.ydata
        ):
            if self.selection_rect:
                self.selection_rect.remove()

            x0, y0 = self.selection_start
            x1, y1 = event.xdata, event.ydata

            width = abs(x1 - x0)
            height = abs(y1 - y0)
            x_min = min(x0, x1)
            y_min = min(y0, y1)

            from matplotlib.patches import Rectangle

            self.selection_rect = Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                alpha=0.7,
            )
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        """No description."""
        if self._dragging_overlay:
            moved_info = copy.deepcopy(self.parameter_selection_info)
            moved_center = getattr(self, "_drag_current_center", None)
            dragging = self._dragging_overlay
            self._dragging_overlay = None
            self._drag_start = None
            self._drag_original_info = None
            self._drag_original_center = None
            self._drag_current_center = None
            if self.canvas is not None:
                self.canvas.setCursor(
                    Qt.OpenHandCursor if self._hit_test_overlay(event) else Qt.ArrowCursor
                )
            if dragging == "center" and moved_center:
                payload = {
                    "is_q_space": bool(self._should_show_q_axis()),
                    "x": float(moved_center[0]),
                    "y": float(moved_center[1]),
                    "beam_center_x": float(moved_center[2]),
                    "beam_center_y": float(moved_center[3]),
                }
                self.center_picked.emit(payload)
            elif moved_info:
                self.region_selected.emit(moved_info)
            return

        if (
            self.selection_mode
            and self.selection_start
            and event.button == 1
            and event.inaxes == self.ax
            and event.xdata
            and event.ydata
        ):
            x0, y0 = self.selection_start
            x1, y1 = event.xdata, event.ydata

            show_q_axis = self._should_show_q_axis()

            min_size_threshold = 0.001 if show_q_axis else 5
            if abs(x1 - x0) > min_size_threshold and abs(y1 - y0) > min_size_threshold:
                width = abs(x1 - x0)
                height = abs(y1 - y0)
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2

                image_shape = getattr(self, "current_image_shape", (1, 1))
                img_height, img_width = image_shape

                if show_q_axis:
                    selection_info = {
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height,
                        "is_q_space": True,
                        "bounds": {
                            "x_min": min(x0, x1),
                            "x_max": max(x0, x1),
                            "y_min": min(y0, y1),
                            "y_max": max(y0, y1),
                        },
                    }

                    self.setWindowTitle(
                        f"GIMaP Image Viewer - Q selection: "
                        f"center=({center_x:.6f}, {center_y:.6f}) nm^-1, "
                        f"size=({width:.6f} x {height:.6f}) nm^-1"
                    )
                else:
                    original_center_y = center_y

                    selection_info = {
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height,
                        "pixel_center_x": int(center_x),
                        "pixel_center_y": int(original_center_y),
                        "pixel_width": int(width),
                        "pixel_height": int(height),
                        "is_q_space": False,
                        "bounds": {
                            "x_min": min(x0, x1),
                            "x_max": max(x0, x1),
                            "y_min": min(y0, y1),
                            "y_max": max(y0, y1),
                        },
                    }

                    self.setWindowTitle(
                        f"GIMaP Image Viewer - Pixel selection: "
                        f"center=({selection_info['pixel_center_x']}, {selection_info['pixel_center_y']}), "
                        f"size=({selection_info['pixel_width']} x {selection_info['pixel_height']}) px"
                    )

                self.current_selection = selection_info
                self._set_cut_region_visible(True, emit=True)

                self.region_selected.emit(selection_info)

            self.selection_start = None
            if self.selection_rect:
                try:
                    self.selection_rect.remove()
                except Exception:
                    pass
                self.selection_rect = None
                self.canvas.draw_idle()

    def _hit_test_overlay(self, event):
        try:
            if self.ax is None or event.inaxes != self.ax:
                return None
            if event.xdata is None or event.ydata is None:
                return None
            point_px = self.ax.transData.transform((float(event.xdata), float(event.ydata)))
            if self.show_center:
                center = self._get_detector_center_display_coordinates()
                if center is not None:
                    center_px = self.ax.transData.transform((float(center[0]), float(center[1])))
                    center_dist = float(
                        np.hypot(point_px[0] - center_px[0], point_px[1] - center_px[1])
                    )
                    if center_dist <= self._overlay_press_tolerance_px * 1.8:
                        return "center"

            if self.parameter_selection_info is None:
                return None
            bounds = self.parameter_selection_info.get("bounds", {})
            x_min = float(bounds.get("x_min", 0))
            x_max = float(bounds.get("x_max", 0))
            y_min = float(bounds.get("y_min", 0))
            y_max = float(bounds.get("y_max", 0))
            if x_min == x_max or y_min == y_max:
                return None

            corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            min_edge_dist = None
            for start, end in zip(corners, corners[1:] + corners[:1]):
                p0 = self.ax.transData.transform(start)
                p1 = self.ax.transData.transform(end)
                dist = self._point_to_segment_distance(point_px, p0, p1)
                min_edge_dist = dist if min_edge_dist is None else min(min_edge_dist, dist)
            if (
                self.show_cut_region
                and min_edge_dist is not None
                and min_edge_dist <= self._overlay_press_tolerance_px
            ):
                return "region"
        except Exception:
            return None
        return None

    @staticmethod
    def _point_to_segment_distance(point, start, end):
        try:
            point = np.asarray(point, dtype=float)
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            seg = end - start
            denom = float(np.dot(seg, seg))
            if denom <= 0:
                return float(np.linalg.norm(point - start))
            t = max(0.0, min(1.0, float(np.dot(point - start, seg) / denom)))
            projection = start + t * seg
            return float(np.linalg.norm(point - projection))
        except Exception:
            return float("inf")

    def _get_detector_center_display_coordinates(self):
        try:
            shape = getattr(self, "current_image_shape", None) or self.last_image_shape
            if shape is None:
                return None
            height, width = shape
            beam_x = float(
                self.fitting_view_model.get_setting(
                    "fitting", "detector.beam_center_x", width / 2.0
                )
            )
            beam_y = float(
                self.fitting_view_model.get_setting(
                    "fitting", "detector.beam_center_y", height / 2.0
                )
            )
            if self._should_show_q_axis():
                qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
                if qy_mesh is None or qz_mesh is None:
                    self._get_q_axis_extent(shape)
                    qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
                if qy_mesh is not None and qz_mesh is not None:
                    row = int(np.clip(round(beam_y), 0, qy_mesh.shape[0] - 1))
                    col = int(np.clip(round(beam_x), 0, qy_mesh.shape[1] - 1))
                    return float(qy_mesh[row, col]), float(qz_mesh[row, col]), beam_x, beam_y
            return beam_x, beam_y, beam_x, beam_y
        except Exception:
            return None

    def _move_detector_center_marker(self, dx, dy):
        try:
            original = getattr(self, "_drag_original_center", None)
            if original is None:
                return
            display_x = float(original[0]) + dx
            display_y = float(original[1]) + dy
            if self._should_show_q_axis():
                pixel_coords = self._convert_q_to_pixel_coordinates(
                    display_x, display_y, 1e-9, 1e-9
                )
                beam_x = float(pixel_coords.get("center_x", original[2]))
                beam_y = float(pixel_coords.get("center_y", original[3]))
            else:
                beam_x = display_x
                beam_y = display_y
            self._drag_current_center = (display_x, display_y, beam_x, beam_y)
            self._redraw_parameter_selection()
            if self.canvas is not None:
                self.canvas.draw_idle()
        except Exception as exc:
            self.status_updated.emit(f"Move Beam Center failed: {exc}")

    def _move_parameter_selection(self, dx, dy):
        try:
            info = copy.deepcopy(self._drag_original_info)
            if not info:
                return
            bounds = info.get("bounds", {})
            x_min = float(bounds.get("x_min", 0)) + dx
            x_max = float(bounds.get("x_max", 0)) + dx
            y_min = float(bounds.get("y_min", 0)) + dy
            y_max = float(bounds.get("y_max", 0)) + dy
            info["bounds"] = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }
            width = abs(x_max - x_min)
            height = abs(y_max - y_min)
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            info["center_x"] = center_x
            info["center_y"] = center_y
            info["width"] = width
            info["height"] = height
            if info.get("is_q_space", False):
                pixel_coords = self._convert_q_to_pixel_coordinates(
                    center_x, center_y, width, height
                )
                info.update(
                    {
                        "pixel_center_x": int(pixel_coords.get("center_x", 0)),
                        "pixel_center_y": int(pixel_coords.get("center_y", 0)),
                        "pixel_width": int(pixel_coords.get("width", 0)),
                        "pixel_height": int(pixel_coords.get("height", 0)),
                    }
                )
            else:
                info.update(
                    {
                        "pixel_center_x": int(round(center_x)),
                        "pixel_center_y": int(round(center_y)),
                        "pixel_width": int(round(width)),
                        "pixel_height": int(round(height)),
                    }
                )
            self.parameter_selection_info = info
            self._redraw_parameter_selection()
            if self.canvas is not None:
                self.canvas.draw_idle()
        except Exception as exc:
            self.status_updated.emit(f"Move Cut Region failed: {exc}")

    def _on_key_press(self, event):
        """No description."""
        if event.key == "escape":
            self._exit_pick_center_mode()
            self._exit_selection_mode()
        elif event.key == "delete" or event.key == "backspace":
            self._clear_selection()

    def keyPressEvent(self, event):
        """Forward Qt key events to the Matplotlib interaction handlers."""
        try:
            if event.key() == Qt.Key_Escape:
                self._exit_pick_center_mode()
                self._exit_selection_mode()
            elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                self._clear_selection()
            else:
                super().keyPressEvent(event)
        except Exception:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Qt"""
        self.canvas.setFocus()
        super().mousePressEvent(event)

    def _exit_selection_mode(self):
        """Exit selection mode."""
        self.selection_mode = False
        self.selection_start = None
        self.canvas.unsetCursor()
        self.setWindowTitle(self.DEFAULT_TITLE)
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()

    def _clear_selection(self):
        """No description."""
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()
        self.current_selection = None
        if self.selection_mode:
            self.setWindowTitle(self.SELECTION_TITLE)
        else:
            self.setWindowTitle(self.DEFAULT_TITLE)
