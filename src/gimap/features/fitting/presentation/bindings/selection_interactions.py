"""Selection Interactions for fitting presentation."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.app.presentation.responsive_layout import (
    move_window_to_cursor_screen,
)

from ..binding_primitives import (
    IndependentFitWindow,
    _qobject_is_alive,
    is_matplotlib_available,
)
from ..curve_plotting import plot_cut_data_with_log_handling


class SelectionInteractionsMixin:
    """Own selection interactions behavior."""

    def _on_region_selected(self, selection_info):
        """ut Line"""
        try:
            is_q_space = selection_info.get("is_q_space", False)

            updated_controls = []

            if is_q_space:
                center_qy = selection_info.get("center_x", 0)
                center_qz = selection_info.get("center_y", 0)
                width_q = selection_info.get("width", 0)
                height_q = selection_info.get("height", 0)

                self.status_updated.emit(
                    f"Q-space region selected: center=({center_qy:.6f}, {center_qz:.6f}) nm^-1, "
                    f"size=({width_q:.6f} x {height_q:.6f}) nm^-1"
                )

                try:
                    if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
                        self.ui.gisaxsInputCenterVerticalValue.setValue(center_qz)
                        updated_controls.append("gisaxsInputCenterVerticalValue")

                    if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
                        self.ui.gisaxsInputCenterParallelValue.setValue(center_qy)
                        updated_controls.append("gisaxsInputCenterParallelValue")

                    if hasattr(self.ui, "gisaxsInputCutLineVerticalValue"):
                        self.ui.gisaxsInputCutLineVerticalValue.setValue(height_q)
                        updated_controls.append("gisaxsInputCutLineVerticalValue")

                    if hasattr(self.ui, "gisaxsInputCutLineParallelValue"):
                        self.ui.gisaxsInputCutLineParallelValue.setValue(width_q)
                        updated_controls.append("gisaxsInputCutLineParallelValue")

                    pixel_coords = self._convert_q_to_pixel_coordinates(
                        center_qy, center_qz, width_q, height_q
                    )
                    center_x = pixel_coords["center_x"]
                    center_y = pixel_coords["center_y"]
                    width = pixel_coords["width"]
                    height = pixel_coords["height"]

                except Exception as e:
                    self.status_updated.emit(f"Q-space parameter update failure: {str(e)}")
                    return
            else:
                center_x = selection_info.get("pixel_center_x", 0)
                center_y = selection_info.get("pixel_center_y", 0)
                width = selection_info.get("pixel_width", 0)
                height = selection_info.get("pixel_height", 0)

                self.status_updated.emit(
                    f"Pixel region selected: Center({center_x}, {center_y}), Size({width} x {height})"
                )

                if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
                    self.ui.gisaxsInputCenterVerticalValue.setValue(center_y)
                    updated_controls.append("gisaxsInputCenterVerticalValue")

                if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
                    self.ui.gisaxsInputCenterParallelValue.setValue(center_x)
                    updated_controls.append("gisaxsInputCenterParallelValue")

                if hasattr(self.ui, "gisaxsInputCutLineVerticalValue"):
                    self.ui.gisaxsInputCutLineVerticalValue.setValue(height)
                    updated_controls.append("gisaxsInputCutLineVerticalValue")

                if hasattr(self.ui, "gisaxsInputCutLineParallelValue"):
                    self.ui.gisaxsInputCutLineParallelValue.setValue(width)
                    updated_controls.append("gisaxsInputCutLineParallelValue")

            if is_q_space:
                main_view_selection_info = {
                    "bounds": {
                        "x_min": center_qy - width_q / 2,
                        "x_max": center_qy + width_q / 2,
                        "y_min": center_qz - height_q / 2,
                        "y_max": center_qz + height_q / 2,
                    },
                    "center_x": center_qy,
                    "center_y": center_qz,
                    "width": width_q,
                    "height": height_q,
                    "is_q_space": True,
                }
                self._persist_cut_region_parameters(center_qy, center_qz, width_q, height_q)
            else:
                main_view_selection_info = {
                    "bounds": {
                        "x_min": center_x - width / 2,
                        "x_max": center_x + width / 2,
                        "y_min": center_y - height / 2,
                        "y_max": center_y + height / 2,
                    },
                    "pixel_center_x": center_x,
                    "pixel_center_y": center_y,
                    "pixel_width": width,
                    "pixel_height": height,
                    "is_q_space": False,
                }
                self._persist_cut_region_parameters(center_x, center_y, width, height)

            self._draw_selection_on_main_view(main_view_selection_info)

            if updated_controls:
                coord_mode = "Q-space" if is_q_space else "pixel"
                self.status_updated.emit(
                    f"Updated Cut Line parameters ({coord_mode}): {', '.join(updated_controls)}"
                )
                if self.independent_window and self.independent_window.isVisible():
                    if is_q_space:
                        self.independent_window.setWindowTitle(
                            f"GIMaP Image Viewer - Q selection "
                            f"center=({center_qy:.6f}, {center_qz:.6f}) nm^-1, "
                            f"size=({width_q:.6f} x {height_q:.6f}) nm^-1"
                        )
                    else:
                        self.independent_window.setWindowTitle(
                            f"GIMaP Image Viewer - pixel selection "
                            f"center=({center_x}, {center_y}), size=({width} x {height})"
                        )
            else:
                self.status_updated.emit("No matching Cut Line controls found for parameter update")

        except Exception as e:
            self.status_updated.emit(f"Error updating Cut Line parameters: {str(e)}")

    _plot_cut_data_with_log_handling = staticmethod(plot_cut_data_with_log_handling)

    def _on_fit_graphics_view_double_click(self, event):
        """No description."""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib",
                )
                return

            if self.q is None or self.I is None:
                QMessageBox.information(
                    self.main_window, "No Data", "No data available for display."
                )
                return
            try:
                q_snapshot = np.asarray(self.q, dtype=float).reshape(-1)
                i_snapshot = np.asarray(self.I, dtype=float).reshape(-1)
                n_snapshot = min(q_snapshot.size, i_snapshot.size)
                if n_snapshot <= 0 or not np.any(
                    np.isfinite(q_snapshot[:n_snapshot]) & np.isfinite(i_snapshot[:n_snapshot])
                ):
                    QMessageBox.information(
                        self.main_window, "No Data", "No finite fitting plot data available."
                    )
                    return
            except Exception:
                QMessageBox.information(
                    self.main_window, "No Data", "Fitting plot data is not ready yet."
                )
                return

            if not _qobject_is_alive(self.independent_fit_window):
                self.independent_fit_window = None

            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                self.independent_fit_window.setAttribute(Qt.WA_DeleteOnClose, True)
                self.independent_fit_window.destroyed.connect(
                    lambda _obj=None: setattr(self, "independent_fit_window", None)
                )
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
                self.independent_fit_window.show_positive_cb.toggled.connect(
                    self._on_positive_only_changed
                )
                if hasattr(self.independent_fit_window, "show_negative_cb"):
                    self.independent_fit_window.show_negative_cb.toggled.connect(
                        self._on_positive_only_changed
                    )
                if hasattr(self.independent_fit_window, "q_unit_combo"):
                    self.independent_fit_window.q_unit_combo.currentTextChanged.connect(
                        self._on_positive_only_changed
                    )
                if hasattr(self.independent_fit_window, "y_range_combo"):
                    self.independent_fit_window.y_range_combo.currentTextChanged.connect(
                        self._on_positive_only_changed
                    )
                if hasattr(self.independent_fit_window, "input_point_delete_requested"):
                    self.independent_fit_window.input_point_delete_requested.connect(
                        self._exclude_ai_input_point_from_plot
                    )
                try:
                    self._sync_axis_filter_controls()
                except Exception:
                    pass

                move_window_to_cursor_screen(self.independent_fit_window)
                self.independent_fit_window.show()
                self.independent_fit_window.raise_()
                self.independent_fit_window.activateWindow()

            mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            try:
                if (
                    hasattr(self, "_is_in_fitting_mode")
                    and callable(self._is_in_fitting_mode)
                    and self._is_in_fitting_mode()
                ):
                    mode = "fitting"
            except Exception:
                pass
            try:
                has_fit = bool(
                    getattr(self, "has_fitting_data", False)
                    and getattr(self, "I_fitting", None) is not None
                )
                if mode == "fitting" and not has_fit:
                    mode = "normal"
            except Exception:
                pass

            if mode == "fitting":
                try:
                    self._update_gui_fitting_display()
                except Exception:
                    pass
                self._update_outside_window("fitting")
            else:
                self._update_outside_window(mode)

            if hasattr(self.independent_fit_window, "canvas"):
                self.independent_fit_window.canvas.setFocus()
                self.independent_fit_window.canvas.draw_idle()

            self.status_updated.emit(f"{mode.capitalize()} mode independent window updated")

        except Exception as e:
            self.status_updated.emit(f"Fit double-click error: {str(e)}")

    def _on_cutline_parameters_changed(self):
        """No description."""
        try:
            if getattr(self, "_initializing", True):
                return

            if not hasattr(self, "_cutline_update_timer"):
                from PyQt5.QtCore import QTimer

                self._cutline_update_timer = QTimer()
                self._cutline_update_timer.setSingleShot(True)
                self._cutline_update_timer.timeout.connect(self._delayed_cutline_update)

            self._cutline_update_timer.stop()
            self._cutline_update_timer.start(150)

        except Exception as e:
            pass

    def _delayed_cutline_update(self):
        """No description."""
        try:
            center_x = 0
            center_y = 0
            width = 0
            height = 0

            if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()

            if hasattr(self.ui, "gisaxsInputCutLineParallelValue"):
                width = self.ui.gisaxsInputCutLineParallelValue.value()
            if hasattr(self.ui, "gisaxsInputCutLineVerticalValue"):
                height = self.ui.gisaxsInputCutLineVerticalValue.value()

            if center_x == 0 and center_y == 0 and width == 0 and height == 0:
                self._clear_parameter_selection()
                return

            if width <= 0 or height <= 0:
                self._clear_parameter_selection()
                return

            selection_info = self._create_selection_from_parameters(
                center_x, center_y, width, height
            )
            self._persist_cut_region_parameters(center_x, center_y, width, height)

            self._update_parameter_selection_display(selection_info)

            if (
                self.current_cut_data is not None
                and hasattr(self, "current_stack_data")
                and self.current_stack_data is not None
            ):
                self._perform_cut()
                self.status_updated.emit(
                    f"Auto-updated cut with new parameters: Center({center_x}, {center_y}), Size({width} x {height})"
                )
            else:
                self.status_updated.emit(
                    f"Parameter selection: Center({center_x}, {center_y}), Size({width} x {height}) - Perform Cut to see results"
                )

        except Exception as e:
            self.status_updated.emit(f"Error updating parameter selection: {str(e)}")

    def _create_selection_from_parameters(self, center_x, center_y, width, height):
        """No description."""
        is_q_space = self._should_show_q_axis()
        half_width = width / 2
        half_height = height / 2

        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height

        selection_info = {
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height,
            "pixel_center_x": int(center_x),
            "pixel_center_y": int(center_y),
            "pixel_width": int(width),
            "pixel_height": int(height),
            "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "is_q_space": is_q_space,
            "is_parameter_based": True,
        }

        return selection_info

    def _update_parameter_selection_display(self, selection_info):
        """No description."""
        try:
            self.current_parameter_selection = selection_info

            if self.current_stack_data is not None:
                self._update_graphics_view_with_selection(self.current_stack_data, selection_info)

            self._sync_independent_window_selection()

        except Exception as e:
            self.status_updated.emit(f"Error updating parameter selection display: {str(e)}")

    def _sync_independent_window_selection(self):
        """No description."""
        try:
            if self.independent_window is None:
                return
            selection_info = getattr(self, "current_parameter_selection", None)
            if selection_info:
                if hasattr(self.independent_window, "set_parameter_selection"):
                    self.independent_window.set_parameter_selection(selection_info)
            else:
                self.independent_window.clear_parameter_selection()
        except Exception as e:
            self.status_updated.emit(f"Error syncing independent window selection: {str(e)}")

    def _refresh_current_parameter_selection_from_ui(self):
        """No description."""
        try:
            if not all(
                hasattr(self.ui, name)
                for name in (
                    "gisaxsInputCenterParallelValue",
                    "gisaxsInputCenterVerticalValue",
                    "gisaxsInputCutLineParallelValue",
                    "gisaxsInputCutLineVerticalValue",
                )
            ):
                return

            center_x = self.ui.gisaxsInputCenterParallelValue.value()
            center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            width = self.ui.gisaxsInputCutLineParallelValue.value()
            height = self.ui.gisaxsInputCutLineVerticalValue.value()

            if getattr(self, "_show_cut_region", False):
                if width <= 0:
                    width = 10.0
                if height <= 0:
                    height = 10.0

            if width > 0 and height > 0:
                self.current_parameter_selection = self._create_selection_from_parameters(
                    center_x, center_y, width, height
                )
                self._persist_cut_region_parameters(center_x, center_y, width, height)
            else:
                self.current_parameter_selection = None
        except Exception:
            pass

    def _clear_parameter_selection(self):
        """No description."""
        try:
            self.current_parameter_selection = None

            if self.current_stack_data is not None:
                self._update_graphics_view_with_selection(self.current_stack_data, None)

            if self.independent_window is not None and self.independent_window.isVisible():
                self.independent_window.clear_parameter_selection()

            self.status_updated.emit("Parameter selection cleared")

        except Exception as e:
            self.status_updated.emit(f"Error clearing parameter selection: {str(e)}")

    def _draw_selection_on_main_view(self, selection_info):
        """raphicsView"""
        try:
            if not hasattr(self.ui, "gisaxsInputGraphicsView") or self.current_stack_data is None:
                return

            self.current_parameter_selection = selection_info
            self._update_graphics_view_with_selection(self.current_stack_data, selection_info)
            self._sync_independent_window_selection()

        except Exception as e:
            self.status_updated.emit(f"Error drawing selection on main view: {str(e)}")

    def _downsample_for_preview(self, data, max_pixels=700_000):
        try:
            h, w = data.shape
            pixels = max(1, h * w)
            step = max(1, int(np.ceil(np.sqrt(pixels / max_pixels))))
            return data[::step, ::step], step
        except Exception:
            return data, 1

    def _preview_extent(self, image_shape, show_q_axis):
        if show_q_axis:
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            if qy_mesh is not None and qz_mesh is not None:
                return [qy_mesh.min(), qy_mesh.max(), qz_mesh.min(), qz_mesh.max()], True
            return None, False
        height, width = image_shape
        return [-0.5, width - 0.5, -0.5, height - 0.5], False

    def _draw_preview_selection(self, ax, selection_info):
        try:
            for artist in getattr(self, "_preview_selection_artists", []):
                try:
                    artist.remove()
                except Exception:
                    pass
            self._preview_selection_artists = []
            artists = []
            if selection_info and self._show_cut_region:
                bounds = selection_info.get("bounds", {})
                x_min = bounds.get("x_min", 0)
                x_max = bounds.get("x_max", 0)
                y_min = bounds.get("y_min", 0)
                y_max = bounds.get("y_max", 0)
                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                    alpha=0.8,
                )
                ax.add_patch(rect)
                artists.append(rect)
            artists.extend(self._draw_detector_center_on_axis(ax))
            self._preview_selection_artists = artists
        except Exception:
            pass

    def _draw_detector_center_on_axis(self, ax):
        try:
            if not self._show_center:
                return []
            center = self._get_detector_center_for_controller_axis()
            if center is None:
                return []
            marker = ax.plot(
                float(center[0]),
                float(center[1]),
                marker="+",
                color="cyan",
                markersize=14,
                markeredgewidth=2.5,
            )[0]
            return [marker]
        except Exception:
            return []

    def _get_detector_center_for_controller_axis(self):
        try:
            if self.current_stack_data is not None:
                height, width = self.current_stack_data.shape
            else:
                height, width = (1, 1)
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
                    return None
                row = int(np.clip(round(beam_y), 0, qy_mesh.shape[0] - 1))
                col = int(np.clip(round(beam_x), 0, qy_mesh.shape[1] - 1))
                return float(qy_mesh[row, col]), float(qz_mesh[row, col])
            return beam_x, beam_y
        except Exception:
            return None
