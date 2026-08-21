"""Cut Extraction for fitting presentation."""

from __future__ import annotations

import numpy as np

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.fitting.application import (
    CutSelection,
)

from ..binding_primitives import (
    _scientific_commands,
)
from ..detector_data_access import analysis_image_for


class CutExtractionMixin:
    """Own cut extraction behavior."""

    def _resolve_cut_points(self, points_override: int = None) -> int:
        """Resolve the target cut point count from override, UI, cache, or settings."""
        if points_override is not None and not isinstance(points_override, bool):
            try:
                return max(10, int(points_override))
            except Exception:
                pass

        try:
            if hasattr(self.ui, "fitDataPointsNumValue"):
                widget = self.ui.fitDataPointsNumValue
                if hasattr(widget, "lineEdit") and widget.lineEdit() is not None:
                    text_value = widget.lineEdit().text().strip()
                    if text_value:
                        parsed = int(float(text_value))
                        if parsed >= 10:
                            return parsed
                if hasattr(widget, "interpretText"):
                    widget.interpretText()
                if hasattr(widget, "value"):
                    return max(10, int(widget.value()))
                if hasattr(widget, "text") and str(widget.text()).strip():
                    return max(10, int(float(str(widget.text()).strip())))
        except Exception:
            pass

        current = getattr(self, "_points_num_current", None)
        if isinstance(current, (int, float)):
            try:
                return max(10, int(current))
            except Exception:
                pass

        try:
            return max(10, int(self.preferences.get("fit.points_num", self._points_num_default)))
        except Exception:
            return max(10, int(getattr(self, "_points_num_default", 300)))

    def _perform_cut(
        self,
        points_override: int = None,
        *,
        reveal_result: bool = True,
    ):
        """Execute the current Cut operation using existing horizontal/vertical cut logic."""
        try:
            self._begin_fitting_step("cut", "Extracting cut")
            if analysis_image_for(self) is None:
                self._fail_fitting_step("cut", "Import an image first")
                QMessageBox.warning(self.main_window, "Warning", "Please import an image first.")
                return

            vertical_value = (
                self.ui.gisaxsInputCutLineVerticalValue.value()
                if hasattr(self.ui, "gisaxsInputCutLineVerticalValue")
                else 0.0
            )
            parallel_value = (
                self.ui.gisaxsInputCutLineParallelValue.value()
                if hasattr(self.ui, "gisaxsInputCutLineParallelValue")
                else 0.0
            )

            if vertical_value <= 0 or parallel_value <= 0:
                self._fail_fitting_step("cut", "Select a valid cut region")
                QMessageBox.warning(self.main_window, "Warning", "Please select a valid region.")
                return

            n_points_cut = self._resolve_cut_points(points_override)
            self._points_num_current = int(n_points_cut)

            if vertical_value <= parallel_value:
                self._perform_horizontal_cut(
                    vertical_value, parallel_value, points_override=n_points_cut
                )
                self.status_updated.emit(
                    f"Horizontal cut performed: Vertical={vertical_value:.2f}, "
                    f"Parallel={parallel_value:.2f}, Points={n_points_cut}"
                )
            else:
                self._perform_vertical_cut(
                    vertical_value, parallel_value, points_override=n_points_cut
                )
                self.status_updated.emit(
                    f"Vertical cut performed: Vertical={vertical_value:.2f}, "
                    f"Parallel={parallel_value:.2f}, Points={n_points_cut}"
                )

            self.data_source = "cut"
            try:
                self._switch_to_normal_display_mode()
            except Exception:
                self.display_mode = "normal"
                if hasattr(self, "_display_mode"):
                    self._display_mode = "normal"
                if hasattr(self, "_fitting_mode_active"):
                    self._fitting_mode_active = False

            if hasattr(self.ui, "fitCurrentDataCheckBox"):
                try:
                    self.ui.fitCurrentDataCheckBox.blockSignals(True)
                    self.ui.fitCurrentDataCheckBox.setChecked(True)
                finally:
                    self.ui.fitCurrentDataCheckBox.blockSignals(False)

            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass

            if hasattr(self, "_set_curve_view_mode"):
                self._set_curve_view_mode("data", refresh=False)
            self._apply_roi_to_data_and_refresh()
            if not getattr(self, "_suppress_workflow_plot_updates", False):
                self._update_GUI_image("normal")
                self._update_outside_window("normal")
                cut = getattr(self, "current_cut_data", None)
                if (
                    isinstance(cut, dict)
                    and np.asarray(cut.get("x_coords", [])).size
                    and np.asarray(
                        cut.get("y_intensity", cut.get("intensity", cut.get("I", [])))
                    ).size
                ):
                    tabs = getattr(self.ui, "fittingPreviewTabs", None)
                    if reveal_result and tabs is not None:
                        tabs.setCurrentIndex(1)
                    self._complete_fitting_step("cut", f"Cut ready · {n_points_cut} points")
                    self._set_fitting_inline_feedback("", "info")

        except Exception as e:
            self._fail_fitting_step("cut", str(e))
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Cut operation failed:\n{str(e)}")

    def _perform_cut_operation(
        self, vertical_value, parallel_value, cut_type: str, points_override: int = None
    ):
        """Execute a horizontal or vertical cut with the current center and geometry.

        Args:
            vertical_value: Cut region height in the active coordinate system.
            parallel_value: Cut region width in the active coordinate system.
            cut_type: Either ``horizontal`` or ``vertical``.
        """
        try:
            center_x, center_y = self._get_cut_center_coordinates()

            show_q_axis = self._should_show_q_axis()

            if cut_type == "horizontal":
                q_mode_method = self._extract_horizontal_cut_q_mode
                pixel_mode_method = self._extract_horizontal_cut_pixel_mode
                x_label = self._horizontal_q_label()
                title = "Horizontal Cut"
            elif cut_type == "vertical":
                q_mode_method = self._extract_vertical_cut_q_mode
                pixel_mode_method = self._extract_vertical_cut_pixel_mode
                x_label = r"$q_z$ (nm$^{-1}$)"
                title = "Vertical Cut"
                self._last_vertical_cut_pixel_rows = None
            else:
                raise Exception(f"Unknown cut type: {cut_type}")

            if show_q_axis:
                cut_data, q_coords = q_mode_method(
                    center_x,
                    center_y,
                    vertical_value,
                    parallel_value,
                    points_override=points_override,
                )
                x_coordinates = q_coords
            else:
                cut_data, q_coords = pixel_mode_method(
                    center_x,
                    center_y,
                    vertical_value,
                    parallel_value,
                    points_override=points_override,
                )
                x_coordinates = q_coords

            self._plot_cut_result(x_coordinates, cut_data, x_label, "Intensity (a.u.)", title)

        except Exception as e:
            raise Exception(f"{cut_type.capitalize()} cut failed: {str(e)}")

    def _perform_horizontal_cut(self, vertical_value, parallel_value, points_override: int = None):
        """No description."""
        self._perform_cut_operation(
            vertical_value, parallel_value, "horizontal", points_override=points_override
        )

    def _perform_vertical_cut(self, vertical_value, parallel_value, points_override: int = None):
        """No description."""
        self._perform_cut_operation(
            vertical_value, parallel_value, "vertical", points_override=points_override
        )

    def _extract_cut_q_mode(
        self, center_qy, center_qz, height_q, width_q, cut_type: str, points_override: int = None
    ):
        """Extract a horizontal or vertical cut from the selected Q-space region."""
        try:
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            if qy_mesh is None or qz_mesh is None:
                raise Exception("Q-space meshgrids not available")
            selection = CutSelection(
                center_x=float(center_qy),
                center_y=float(center_qz),
                height=float(height_q),
                width=float(width_q),
                orientation=cut_type,
            )
            intensity_native, q_native, pixel_indices = _scientific_commands(self).cut.extract_q(
                analysis_image_for(self),
                qy_mesh,
                qz_mesh,
                selection,
            )
            if cut_type == "vertical":
                self._log_cut_debug(
                    f"Vertical Q cut: ROI q range {self._horizontal_q_axis()}="
                    f"[{center_qy - width_q / 2:.8g}, "
                    f"{center_qy + width_q / 2:.8g}], qz=[{center_qz - height_q / 2:.8g}, "
                    f"{center_qz + height_q / 2:.8g}]"
                )
                self._log_cut_debug(
                    f"Vertical Q cut: first/last pixel row used = "
                    f"{int(pixel_indices[0])}, {int(pixel_indices[-1])}"
                )
            valid_q, valid_intensity, _ = self._sort_filter_cut_pairs(
                q_native,
                intensity_native,
                context=f"{cut_type.capitalize()} Q cut",
                pixel_rows=pixel_indices if cut_type == "vertical" else None,
                log_vertical=cut_type == "vertical",
            )
            valid_q, valid_intensity = self._filter_cut_pairs_for_active_axis(
                valid_q,
                valid_intensity,
                context=f"{cut_type.capitalize()} Q cut",
            )
            n_points = self._resolve_cut_points(points_override)
            q_interp = np.linspace(valid_q.min(), valid_q.max(), n_points)
            try:
                method = (
                    self.ui.fitInterpolationMethodValue.currentText()
                    if hasattr(self.ui, "fitInterpolationMethodValue")
                    else self._interp_method_default
                )
            except Exception:
                method = self._interp_method_default
            intensity_interp = _scientific_commands(self).cut.interpolate(
                valid_q, valid_intensity, q_interp, method
            )
            try:
                self.status_updated.emit(
                    f"Cut(Q) extracted points: {len(q_interp)} (method={method})"
                )
            except Exception:
                pass
            return intensity_interp, q_interp
        except Exception as exc:
            raise Exception(f"Q-mode {cut_type} cut extraction failed: {str(exc)}") from exc

    def _extract_horizontal_cut_q_mode(
        self, center_qy, center_qz, height_q, width_q, points_override: int = None
    ):
        """Q"""
        return self._extract_cut_q_mode(
            center_qy, center_qz, height_q, width_q, "horizontal", points_override=points_override
        )

    def _extract_vertical_cut_q_mode(
        self, center_qy, center_qz, height_q, width_q, points_override: int = None
    ):
        """Q"""
        return self._extract_cut_q_mode(
            center_qy, center_qz, height_q, width_q, "vertical", points_override=points_override
        )

    def _extract_cut_pixel_mode(
        self, center_x, center_y, height, width, cut_type: str, points_override: int = None
    ):
        """Extract a horizontal or vertical cut from the selected pixel-space region."""
        try:
            selection = CutSelection(
                center_x=float(center_x),
                center_y=float(center_y),
                height=float(height),
                width=float(width),
                orientation=cut_type,
            )
            intensity_native, pixel_coords = _scientific_commands(self).cut.extract_pixel(
                analysis_image_for(self),
                selection,
            )
            if cut_type == "horizontal":
                native_q = self._convert_pixel_to_qy(pixel_coords)
            else:
                native_q = self._convert_pixel_to_qz(pixel_coords)
                image_height, image_width = analysis_image_for(self).shape
                x_min = max(0, int(center_x - width / 2))
                x_max = min(image_width - 1, int(center_x + width / 2))
                self._log_cut_debug(
                    f"Vertical Pixel cut: ROI display pixels x=[{x_min}, {x_max}], "
                    f"y=[{int(center_y - height / 2)}, {int(center_y + height / 2)}], "
                    f"array rows=[{int(pixel_coords[0])}, {int(pixel_coords[-1])}], "
                    "image_origin=lower"
                )
                if pixel_coords.size:
                    self._log_cut_debug(
                        f"Vertical Pixel cut: first/last pixel row used = "
                        f"{int(pixel_coords[0])}, {int(pixel_coords[-1])}"
                    )
            valid_q, valid_intensity, _ = self._sort_filter_cut_pairs(
                native_q,
                intensity_native,
                context=f"{cut_type.capitalize()} Pixel cut native q",
                pixel_rows=pixel_coords if cut_type == "vertical" else None,
                log_vertical=cut_type == "vertical",
            )
            valid_q, valid_intensity = self._filter_cut_pairs_for_active_axis(
                valid_q,
                valid_intensity,
                context=f"{cut_type.capitalize()} Pixel cut",
            )
            if valid_q.size < 2:
                raise Exception("Not enough finite q/intensity points in the selected region")
            n_points = self._resolve_cut_points(points_override)
            q_interp = np.linspace(valid_q.min(), valid_q.max(), n_points)
            try:
                method = (
                    self.ui.fitInterpolationMethodValue.currentText()
                    if hasattr(self.ui, "fitInterpolationMethodValue")
                    else self._interp_method_default
                )
            except Exception:
                method = self._interp_method_default
            intensity_interp = _scientific_commands(self).cut.interpolate(
                valid_q, valid_intensity, q_interp, method
            )
            if cut_type == "vertical":
                self._last_vertical_cut_pixel_rows = None
            try:
                self.status_updated.emit(
                    f"Cut(Pixel) extracted {len(q_interp)} unique-q points from "
                    f"{len(valid_q)} native samples (method={method})"
                )
            except Exception:
                pass
            return intensity_interp, q_interp
        except Exception as exc:
            raise Exception(f"Pixel-mode {cut_type} cut extraction failed: {str(exc)}") from exc

    def _extract_horizontal_cut_pixel_mode(
        self, center_x, center_y, height, width, points_override: int = None
    ):
        """No description."""
        return self._extract_cut_pixel_mode(
            center_x, center_y, height, width, "horizontal", points_override=points_override
        )

    def _extract_vertical_cut_pixel_mode(
        self, center_x, center_y, height, width, points_override: int = None
    ):
        """No description."""
        return self._extract_cut_pixel_mode(
            center_x, center_y, height, width, "vertical", points_override=points_override
        )

    def _get_detector_for_pixel_conversion(self):
        """No description."""
        try:
            height, width = analysis_image_for(self).shape

            pixel_size_x = self.fitting_view_model.get_setting(
                "fitting", "detector.pixel_size_x", 172.0
            )
            pixel_size_y = self.fitting_view_model.get_setting(
                "fitting", "detector.pixel_size_y", 172.0
            )
            beam_center_x = self.fitting_view_model.get_setting(
                "fitting", "detector.beam_center_x", width / 2.0
            )
            beam_center_y = self.fitting_view_model.get_setting(
                "fitting", "detector.beam_center_y", height / 2.0
            )
            distance = self.fitting_view_model.get_setting("fitting", "detector.distance", 2565.0)
            theta_in_deg = self.fitting_view_model.get_setting("beam", "grazing_angle", 0.4)
            wavelength = self.fitting_view_model.get_setting("beam", "wavelength", 0.1045)

            return _scientific_commands(self).q_space.create_detector(
                image_shape=(height, width),
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                distance=distance,
                theta_in_deg=theta_in_deg,
                wavelength=wavelength,
                crop_params=None,
            )
        except Exception:
            return None

    def _convert_pixel_coords_to_q(self, pixel_coords, conversion_type: str):
        """Convert detector pixel coordinates to Q coordinates.

        Args:
            pixel_coords: Pixel coordinates to convert.
            conversion_type: Target coordinate type, either ``qy`` or ``qz``.
        """
        try:
            height, width = analysis_image_for(self).shape
            coords = np.asarray(pixel_coords, dtype=float)

            try:
                qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            except Exception:
                qy_mesh, qz_mesh = None, None

            if conversion_type == "qy":
                if qy_mesh is not None and getattr(qy_mesh, "shape", None) == (height, width):
                    return _scientific_commands(self).cut.sample_mesh_line(
                        qy_mesh,
                        coords,
                        orientation="horizontal",
                        image_shape=(height, width),
                    )

                detector = self._get_detector_for_pixel_conversion()
                if detector is None:
                    raise Exception("Failed to create detector")
                center_y = height / 2.0
                _, q_coords, _ = detector.pixel_to_q_space(coords, center_y)
            elif conversion_type == "qz":
                if qz_mesh is not None and getattr(qz_mesh, "shape", None) == (height, width):
                    q_coords = _scientific_commands(self).cut.sample_mesh_line(
                        qz_mesh,
                        coords,
                        orientation="vertical",
                        image_shape=(height, width),
                    )
                    if q_coords.size:
                        self._log_cut_debug(
                            f"Vertical Pixel cut: first/last q before sorting = {q_coords[0]:.8g}, {q_coords[-1]:.8g}"
                        )
                    return q_coords

                detector = self._get_detector_for_pixel_conversion()
                if detector is None:
                    raise Exception("Failed to create detector")
                center_x = width / 2.0
                # qz meshgrids are flipped vertically for origin='lower'; mirror rows for fallback consistency.
                mirrored_rows = (height - 1) - coords
                _, _, q_coords = detector.pixel_to_q_space(center_x, mirrored_rows)
            else:
                raise Exception(f"Unknown conversion type: {conversion_type}")

            return np.asarray(q_coords, dtype=float)

        except Exception as e:
            try:
                self._add_fitting_message(
                    f"Pixel to {conversion_type} conversion failed: {str(e)}", "ERROR"
                )
            except Exception:
                self.status_updated.emit(f"Pixel to {conversion_type} conversion failed: {str(e)}")
            coords = np.asarray(pixel_coords, dtype=float)
            std = coords.std()
            if not np.isfinite(std) or std == 0:
                return coords
            return (coords - coords.mean()) / std

    def _convert_pixel_to_qy(self, pixel_coords):
        """qy"""
        return self._convert_pixel_coords_to_q(pixel_coords, "qy")

    def _convert_pixel_to_qz(self, pixel_coords):
        """qz"""
        return self._convert_pixel_coords_to_q(pixel_coords, "qz")
