"""Yoneda loading convenience and explicit symmetry-based horizontal calibration."""

from pathlib import Path
from time import perf_counter

from ...application import CutSelection
from ..binding_primitives import _scientific_commands
from ..detector_data_access import analysis_image_for, analysis_revision_for


class CenterSymmetryMixin:
    def _auto_yoneda_after_load(self, file_path):
        toggle = getattr(self.ui, "gisaxsAutoYonedaOnLoadCheckBox", None)
        if (
            Path(file_path).suffix.lower() == ".cbf"
            and toggle is not None
            and toggle.isChecked()
            and not getattr(self, "_insitu_workflow_busy", False)
        ):
            self._auto_find_center()

    def _optimize_center_x(self):
        dialog = getattr(self, "_workflow_v5_dialog", None)
        if (
            getattr(self, "_insitu_workflow_busy", False)
            or getattr(self, "_insitu_workflow_state", "Idle")
            in ("Watching", "Processing", "Paused")
            or getattr(self, "_ai_job_thread", None)
            or (dialog is not None and dialog.job is not None)
        ):
            self._set_fitting_inline_feedback("Wait for the active analysis to finish.", "warning")
            return
        try:
            image = analysis_image_for(self)
            if image is None:
                raise ValueError("Load a detector image first")
            state = getattr(self, "current_detector_image", None)
            if state is not None and state.preprocessing.mirror_fill_gaps:
                raise ValueError(
                    "Turn off mirror gap fill before estimating symmetry from measured pixels"
                )
            region = self._current_selection_pixel_region(
                q_mode=self._should_show_q_axis(), horizontal_axis=self._horizontal_q_axis()
            )
            if region is None:
                raise ValueError("Find Yoneda or select a horizontal cut band first")
            if not self._should_show_q_axis():
                selection = CutSelection(
                    center_x=self.ui.gisaxsInputCenterParallelValue.value(),
                    center_y=self.ui.gisaxsInputCenterVerticalValue.value(),
                    height=self.ui.gisaxsInputCutLineVerticalValue.value(),
                    width=self.ui.gisaxsInputCutLineParallelValue.value(),
                    orientation="horizontal",
                )
                x0, x1, r0, r1 = _scientific_commands(self).cut.pixel_bounds(image.shape, selection)
                region = (r0, r1, x0, x1)
            initial = (region[2] + region[3]) / 2
            beam_x_before = self.fitting_view_model.get_setting(
                "fitting", "detector.beam_center_x", initial
            )
            tick = perf_counter()
            result = _scientific_commands(self).image.optimize_center_x(image, region, initial)
            # Beam X defines q=0. Keep beam Y and the selected Yoneda rows intact.
            self.fitting_view_model.set_setting(
                "fitting", "detector.beam_center_x", result.center_x
            )
            self.fitting_view_model.save_settings()
            self._q_mesh_cache_key = None
            self._compute_q_meshgrids_and_store()
            delta = result.center_x - initial
            shifted = (
                region[0],
                region[1],
                max(0, round(region[2] + delta)),
                min(image.shape[1] - 1, round(region[3] + delta)),
            )
            if self._should_show_q_axis():
                self._apply_pixel_region_to_active_coordinates(
                    shifted, q_mode=True, horizontal_axis=self._horizontal_q_axis()
                )
            else:
                self._set_numeric_control_silently(
                    "gisaxsInputCenterParallelValue", result.center_x
                )
                cy = self.ui.gisaxsInputCenterVerticalValue.value()
                width = self.ui.gisaxsInputCutLineParallelValue.value()
                height = self.ui.gisaxsInputCutLineVerticalValue.value()
                self._persist_cut_region_parameters(result.center_x, cy, width, height)
                self.current_parameter_selection = self._create_selection_from_parameters(
                    result.center_x, cy, width, height
                )
            panel = getattr(self.ui, "fittingDetectorSetupPanel", None)
            if panel is not None:
                # A programmatic calibration must not enqueue another geometry
                # commit (which snaps the cut and expands its row bounds).
                control = panel.beam_center_x_spinbox
                blocked = control.blockSignals(True)
                try:
                    control.setValue(result.center_x)
                finally:
                    control.blockSignals(blocked)
            self._seed_independent_q_cache()
            self._record_cut_geometry_draft(
                self.ui.gisaxsInputCenterParallelValue.value(),
                self.ui.gisaxsInputCenterVerticalValue.value(),
                self.ui.gisaxsInputCutLineParallelValue.value(),
                self.ui.gisaxsInputCutLineVerticalValue.value(),
            )
            self._refresh_image_display()
            self._perform_cut()
            self._last_center_symmetry = {
                **result.to_dict(),
                "seconds": perf_counter() - tick,
                "analysis_revision": analysis_revision_for(self),
                "pixel_region": list(region),
                "beam_x_before": beam_x_before,
            }
            if isinstance(getattr(self, "current_cut_data", None), dict):
                self.current_cut_data["center_symmetry"] = dict(self._last_center_symmetry)
            message = (
                f"Center X {initial:.2f} → {result.center_x:.2f} px · symmetry loss "
                f"{result.score_before:.4g} → {result.score_after:.4g}. Cut updated."
            )
            self.status_updated.emit(message)
            self._set_fitting_inline_feedback(message, "info")
        except (ValueError, TypeError) as exc:
            self._set_fitting_inline_feedback(str(exc), "warning")
            self.status_updated.emit(str(exc))
