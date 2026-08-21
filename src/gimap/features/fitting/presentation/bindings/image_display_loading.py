"""Image Display Loading for fitting presentation."""

from __future__ import annotations

import os

import time

from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.fitting.application import (
    ScatteringFileData,
)

from src.gimap.app.presentation.responsive_layout import (
    move_window_to_cursor_screen,
)


from ..binding_primitives import (
    IndependentMatplotlibWindow,
    _scientific_commands,
    is_matplotlib_available,
)
from ..detector_data_access import analysis_image_for


class ImageDisplayLoadingMixin:
    """Own image display loading behavior."""

    def _on_image_loaded(self, image_data, file_path):
        """No description."""
        try:
            source_files = tuple(
                Path(path) for path in getattr(self.async_image_loader, "_last_source_files", ())
            ) or (Path(file_path),)
            self.fitting_view_model.accept_loaded_image(
                ScatteringFileData(
                    image=image_data,
                    source_path=Path(file_path),
                    source_files=source_files,
                    frame_index=int(getattr(self, "_nxs_frame_index", 0)),
                )
            )
            self._sync_fitting_workflow()
            self.status_updated.emit(f"Image loading complete: {os.path.basename(file_path)}")
            workflow_frame = getattr(self, "_insitu_workflow_busy", False) and file_path == getattr(
                self, "_insitu_workflow_processing_file", None
            )
            if workflow_frame and not self._should_refresh_insitu_views_for_current_file():
                self._ingest_workflow_image_without_preview(image_data)
            else:
                self._display_image(image_data)
            analysis_image = analysis_image_for(self)
            if analysis_image is not None:
                self._after_insitu_workflow_image_loaded(analysis_image, file_path)
        except Exception as e:
            self.fitting_view_model.fail_image_load(str(e))
            self._sync_fitting_workflow()
            self.status_updated.emit(f"Error while displaying image: {str(e)}")
            if getattr(self, "_insitu_workflow_processing_file", None) == file_path:
                self._finalize_insitu_workflow_file(
                    load_status="failed", error_message=str(e), failed=True
                )

    def _on_image_loading_progress(self, progress, status):
        """No description."""
        try:
            self.status_updated.emit(f"Image loading... {progress}% - {status}")
            self.progress_updated.emit(progress)
        except Exception as e:
            self.status_updated.emit(f"Progress update error: {str(e)}")

    def _on_image_loading_error(self, error_message):
        """No description."""
        self.fitting_view_model.fail_image_load(str(error_message))
        self._sync_fitting_workflow()
        if getattr(self, "_insitu_workflow_busy", False):
            self._finalize_insitu_workflow_file(
                load_status="failed", error_message=str(error_message), failed=True
            )
            return
        QMessageBox.critical(self.main_window, "Image loading error", error_message)

    def _ingest_workflow_image_without_preview(self, image_data):
        """Update data needed for cut/fitting without repainting heavy image views."""
        self.current_raw_image = np.asarray(image_data, dtype=np.float32)
        self._reapply_input_image_options(refresh=False)
        try:
            sc = int(self.current_parameters.get("stack_count", 1))
        except Exception:
            sc = 1
        analysis_image = analysis_image_for(self)
        self.summed_data = analysis_image if sc and sc > 1 else None
        try:
            self._compute_q_meshgrids_and_store()
        except Exception:
            pass
        if self._current_vmin is None or self._current_vmax is None:
            try:
                display_image = self._get_current_display_image()
                self._handle_color_scale(display_image if display_image is not None else image_data)
            except Exception:
                pass
        try:
            self._update_cutline_labels_units()
        except Exception:
            pass

    def _display_image(self, image_data):
        """No description."""
        try:
            self.current_raw_image = np.asarray(image_data, dtype=np.float32)
            self._reapply_input_image_options(refresh=False)
            try:
                sc = int(self.current_parameters.get("stack_count", 1))
            except Exception:
                sc = 1
            analysis_image = analysis_image_for(self)
            self.summed_data = analysis_image if sc and sc > 1 else None
            self._compute_q_meshgrids_and_store()

            display_image = self._get_current_display_image()
            if display_image is None:
                raise RuntimeError("Analysis image is not ready")

            self._handle_color_scale(display_image)

            self._update_cutline_labels_units()
            self._refresh_current_parameter_selection_from_ui()

            if hasattr(self.ui, "gisaxsInputGraphicsView"):
                self._update_graphics_view(display_image)

            if self.independent_window is not None and self.independent_window.isVisible():
                state = self._current_detector_display_state()
                self.independent_window.set_detector_display_state(state)
                self._seed_independent_q_cache()
                self.independent_window.update_image(
                    display_image, state.vmin, state.vmax, use_log=state.log_intensity
                )
                self._sync_independent_window_selection()

            window_status = (
                " (+ Independent window)"
                if (self.independent_window and self.independent_window.isVisible())
                else ""
            )
            vmin_vmax_info = (
                f" [Vmin: {self._current_vmin:.3f}, Vmax: {self._current_vmax:.3f}]"
                if self._current_vmin is not None and self._current_vmax is not None
                else ""
            )
            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            self.status_updated.emit(
                f"{mode_text} image displayed: {image_data.shape}{vmin_vmax_info}{window_status}"
            )

        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")

    def _compute_q_meshgrids_and_store(self):
        """No description."""
        try:
            analysis_image = analysis_image_for(self)
            if analysis_image is None:
                return
            height, width = analysis_image.shape
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
            cache_key = (
                height,
                width,
                float(pixel_size_x),
                float(pixel_size_y),
                float(beam_center_x),
                float(beam_center_y),
                float(distance),
                float(theta_in_deg),
                float(wavelength),
            )
            if (
                self._q_mesh_cache_key == cache_key
                and self.qy_matrix is not None
                and self.qz_matrix is not None
                and self.qr_matrix is not None
            ):
                return
            t0 = time.perf_counter()
            detector = _scientific_commands(self).q_space.create_detector(
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
            qy_mesh, qz_mesh, qr_mesh = detector.get_q_coordinate_meshgrids()
            self.qy_matrix = qy_mesh
            self.qz_matrix = qz_mesh
            self.qr_matrix = qr_mesh
            self._q_mesh_cache_key = cache_key
            print(f"[Timing] q-space mesh calculation: {(time.perf_counter() - t0) * 1000:.2f} ms")
        except Exception:
            self.qy_matrix = None
            self.qz_matrix = None
            self.qr_matrix = None
            self._q_mesh_cache_key = None

    def _update_graphics_view(self, image_data):
        """GraphicsView"""
        self._update_graphics_view_with_selection(
            image_data, getattr(self, "current_parameter_selection", None)
        )

    def _prepare_image_data_for_display(self, image_data):
        """No description."""
        try:
            is_log = self._is_log_mode_enabled()
            cache_key = (id(image_data), bool(is_log))
            cached = self._image_display_cache.get(cache_key)
            if cached is not None:
                self._image_display_cache.move_to_end(cache_key)
                if is_log:
                    print("[Timing] log transform: 0.00 ms (cache hit)")
                return cached, is_log

            if is_log:
                t0 = time.perf_counter()
                safe_data = np.where(
                    np.isfinite(image_data),
                    np.maximum(image_data, 0.001),
                    np.nan,
                )
                processed_data = np.log(safe_data, dtype=np.float32)
                print(f"[Timing] log transform: {(time.perf_counter() - t0) * 1000:.2f} ms")
            else:
                processed_data = image_data.astype(np.float32)
            self._image_display_cache[cache_key] = processed_data
            self._image_display_cache.move_to_end(cache_key)
            while len(self._image_display_cache) > self._image_display_cache_limit:
                self._image_display_cache.popitem(last=False)

            return processed_data, is_log

        except Exception:
            return image_data.astype(np.float32), True

    def _refresh_image_display(self):
        """No description."""
        try:
            if analysis_image_for(self) is not None:
                self._refresh_current_parameter_selection_from_ui()
                display_image = self._get_current_display_image()
                if display_image is None:
                    return
                if hasattr(self.ui, "gisaxsInputGraphicsView"):
                    self._update_graphics_view(display_image)

                if self.independent_window is not None and self.independent_window.isVisible():
                    state = self._current_detector_display_state()
                    self.independent_window.set_detector_display_state(state)
                    self._seed_independent_q_cache()
                    self.independent_window.update_image(
                        display_image, state.vmin, state.vmax, use_log=state.log_intensity
                    )
                    self._sync_independent_window_selection()
        except Exception as e:
            self.status_updated.emit(f"Refresh display error: {str(e)}")

    def _on_graphics_view_double_click(self, event):
        """GraphicsView"""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib",
                )
                return

            if analysis_image_for(self) is None:
                QMessageBox.information(
                    self.main_window, "No Image", "Please import and display an image first."
                )
                return

            self._show_independent_window()

        except Exception as e:
            self.status_updated.emit(f"Double-click error: {str(e)}")

    def _show_independent_window(self):
        """atplotlib"""
        try:
            if self.independent_window is None or not self.independent_window.isVisible():
                self.independent_window = IndependentMatplotlibWindow(
                    self.main_window,
                    fitting_view_model=self.fitting_view_model,
                )
                self.independent_window.region_selected.connect(self._on_region_selected)
                self.independent_window.center_picked.connect(self._on_detector_center_picked)
                self.independent_window.display_state_changed.connect(
                    self._on_independent_detector_display_state_changed
                )
                self.independent_window.status_updated.connect(self.status_updated.emit)

            analysis_image = analysis_image_for(self)
            if analysis_image is not None:
                self._refresh_current_parameter_selection_from_ui()
                display_image = self._get_current_display_image()
                if display_image is None:
                    return
                self.independent_window.current_image_shape = analysis_image.shape
                state = self._current_detector_display_state()
                self.independent_window.set_detector_display_state(state)
                self._seed_independent_q_cache()
                self.independent_window.update_image(
                    display_image, state.vmin, state.vmax, use_log=state.log_intensity
                )
                self._sync_independent_window_selection()

            if not self.independent_window.isVisible():
                move_window_to_cursor_screen(self.independent_window)
            self.independent_window.show()
            self.independent_window.raise_()
            self.independent_window.activateWindow()
            self._sync_independent_window_selection()

            self.independent_window.canvas.setFocus()

            self.status_updated.emit(
                "Independent window opened - Right-click to activate selection, ESC to exit selection mode"
            )

        except Exception as e:
            self.status_updated.emit(f"Independent window error: {str(e)}")

    def _seed_independent_q_cache(self) -> None:
        """Share immutable q grids instead of recalculating them in the viewer."""

        window = getattr(self, "independent_window", None)
        if window is None:
            return
        window.seed_q_grid_cache(
            self._q_mesh_cache_key,
            self.qy_matrix,
            self.qz_matrix,
            self.qr_matrix,
        )

    def _on_detector_center_picked(self, center_info: dict):
        try:
            beam_x = float(center_info.get("beam_center_x", 0.0))
            beam_y = float(center_info.get("beam_center_y", 0.0))

            try:
                self.fitting_view_model.set_setting("fitting", "detector.beam_center_x", beam_x)
                self.fitting_view_model.set_setting("fitting", "detector.beam_center_y", beam_y)
                self.fitting_view_model.save_settings()
            except Exception:
                self.fitting_view_model.set_setting("fitting", "detector.beam_center_x", beam_x)
                self.fitting_view_model.set_setting("fitting", "detector.beam_center_y", beam_y)
                try:
                    self.fitting_view_model.save_settings()
                except Exception:
                    pass

            dialog = getattr(self, "detector_params_dialog", None)
            if dialog is not None and dialog.isVisible():
                try:
                    dialog.beam_center_x_spinbox.blockSignals(True)
                    dialog.beam_center_y_spinbox.blockSignals(True)
                    dialog.beam_center_x_spinbox.setValue(beam_x)
                    dialog.beam_center_y_spinbox.setValue(beam_y)
                    dialog.beam_center_x_spinbox.blockSignals(False)
                    dialog.beam_center_y_spinbox.blockSignals(False)
                    if hasattr(dialog, "_save_parameters"):
                        dialog._save_parameters()
                    if hasattr(dialog, "_get_current_parameters"):
                        dialog.parameters_changed.emit(dialog._get_current_parameters())
                except Exception:
                    pass

            try:
                self._q_mesh_cache_key = None
                self._compute_q_meshgrids_and_store()
            except Exception:
                pass

            try:
                if self.independent_window is not None:
                    self._seed_independent_q_cache()
                    self.independent_window._drag_current_center = None
            except Exception:
                pass

            self._show_center = True
            self._apply_image_display_options(refresh=False)
            self._on_detector_parameters_changed(
                {
                    "beam_center_x": beam_x,
                    "beam_center_y": beam_y,
                }
            )
            self._refresh_image_display()
            self._complete_fitting_step("center", "Detector center selected")
            self.status_updated.emit(
                f"Detector beam center set from image: X={beam_x:.2f}, Y={beam_y:.2f}"
            )
        except Exception as exc:
            self.status_updated.emit(f"Failed to set detector center from image: {exc}")
