"""Bridge the portable native-node workflow to single and in-situ fitting."""

import numpy as np
from pathlib import Path
from ...application import CutSelection
from ..binding_primitives import _scientific_commands

from ..workflow_v5_dialog import WorkflowV5Dialog
from ...application.workflow_v5 import bundled_workflow, default_options


class WorkflowV5BindingMixin:
    def _native_cbf_input(self, options):
        state = getattr(self, "current_detector_image", None)
        source = self.current_parameters.get("imported_gisaxs_file", "")
        if state is None or Path(source).suffix.lower() != ".cbf":
            return None
        workflow_state = self.fitting_view_model.state
        cut_revision = (getattr(self, "current_cut_data", None) or {}).get("analysis_revision", workflow_state.cut_result_analysis_revision)
        if cut_revision != state.revision:
            raise ValueError("Preprocessing changed. Click Extract / Update to rebuild the cut before fitting.")
        image = state.analysis_image
        grid = self._detector_q_grid()
        if grid is None:
            return None
        q_mesh = grid.horizontal(self._horizontal_q_axis())
        selected = None
        if self._should_show_q_axis():
            region = self._current_selection_pixel_region(
                q_mode=True, horizontal_axis=self._horizontal_q_axis()
            )
            if region is None:
                raise ValueError("Select a horizontal detector region first")
            info = self.current_parameter_selection
            cx, cy, width, height = (info[k] for k in ("center_x", "center_y", "width", "height"))
            selected = (abs(q_mesh - cx) <= width / 2) & (abs(grid.qz - cy) <= height / 2)
        else:
            insitu_geometry = self._insitu_cut_geometry() if getattr(self, "_insitu_workflow_ai_record", None) is not None else {}
            selection = CutSelection(
                center_x=insitu_geometry.get("center_parallel_px", self.ui.gisaxsInputCenterParallelValue.value()),
                center_y=insitu_geometry.get("center_vertical_px", self.ui.gisaxsInputCenterVerticalValue.value()),
                height=insitu_geometry.get("cut_vertical_px", self.ui.gisaxsInputCutLineVerticalValue.value()),
                width=insitu_geometry.get("cut_parallel_px", self.ui.gisaxsInputCutLineParallelValue.value()),
                orientation="horizontal",
            )
            if selection.height > selection.width:
                return None
            x0, x1, r0, r1 = _scientific_commands(self).cut.pixel_bounds(image.shape, selection)
            region = (r0, r1, x0, x1)
        q, y, counting, metadata = _scientific_commands(self).cut.cbf_observations(
            image,
            q_mesh,
            region,
            selection_mask=selected,
        )
        metadata.update(
            q_source="region_mean_native",
            analysis_revision=state.revision,
            mask_source="detector_preprocessing",
            gap_margin_px=state.preprocessing.invalid_margin_px,
            masked_pixels=state.masked_pixels,
            threshold_enabled=state.preprocessing.threshold_enabled,
            mirror_replaced_pixels=state.mirror_replaced_pixels,
            stack_count=max(1, int(self.current_parameters.get("stack_count", 1))),
        )
        sigma = np.sqrt(
            counting**2 + (options["relative_noise"] * abs(y)) ** 2 + options["absolute_noise"] ** 2
        )
        self._workflow_observation_metadata = metadata
        return dict(x_coords=q, y_intensity=y, err=sigma, q_source_unit="nm")

    def _workflow_options(self):
        return {**default_options(), **self._ai_fitting_settings().get("workflow_v5", {})}

    def _save_workflow_options(self, options):
        self._save_ai_fitting_settings(workflow_v5=dict(options))

    def open_ai_fitting_workspace(self):
        dialog = getattr(self, "_workflow_v5_dialog", None)
        if dialog is None:
            dialog = WorkflowV5Dialog(
                self._current_ai_curve_arrays,
                self._workflow_options(),
                self.main_window,
                runner=self.fitting_view_model.context.jobs,
            )
            dialog.settings_changed.connect(self._save_workflow_options)
            dialog.candidate_selected.connect(self._apply_workflow_candidate)
            dialog.can_start = lambda: (
                getattr(self, "_ai_job_thread", None) is None
                and not getattr(self, "_insitu_workflow_busy", False)
                and getattr(self, "_insitu_workflow_state", "Idle")
                not in ("Watching", "Processing", "Paused")
            )
            dialog.sigma_estimated = lambda: getattr(self, "_workflow_sigma_estimated", False)
            dialog.observation_metadata = lambda: getattr(
                self, "_workflow_observation_metadata", {}
            )
            self._workflow_v5_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _refresh_ai_fitting_models(self):
        self._ai_fitting_models = []
        method = self._workflow_options()["method"]
        label = {
            "stable": "Single RC specialist (experimental; known single RC required)",
            "model": "General V5 (experimental)",
            "experimental": "Numerical physical fitting",
        }[method]
        self._set_ai_workspace_status(f"{label} · ready for native-point fitting", 0)

    def _open_insitu_prediction_settings(self):
        recipe = self.fitting_view_model.insitu.recipe
        if recipe is None:
            self._add_fitting_error("Use current setup before editing in-situ prediction settings.")
            return
        dialog = WorkflowV5Dialog(
            options=dict(recipe.model.get("workflow_v5", self._workflow_options())),
            parent=self.main_window,
            settings_only=True,
        )

        def save(options):
            from ...application import ReviseInSituRecipeRequest

            current = self.fitting_view_model.insitu.recipe
            revision = self.fitting_view_model.insitu.revise_recipe(
                ReviseInSituRecipeRequest(
                    current=current,
                    model={**current.to_dict()["model"], "workflow_v5": options},
                    scope="future",
                )
            )
            self.ui.fittingInsituSeriesPage.render_recipe(revision.recipe)
            dialog.status.setText(f"Saved settings v{revision.recipe.version} for future frames.")

        dialog.settings_changed.connect(save)
        self._insitu_prediction_dialog = dialog
        dialog.show()

    def _selected_ai_model_path(self):
        return bundled_workflow()

    def _workflow_input_selection(self):
        roi = None
        if (
            getattr(self, "_roi_controls_enabled", True)
            and self._roi_min is not None
            and self._roi_max is not None
        ):
            roi = [float(self._roi_min), float(self._roi_max)]
        return dict(
            axis_filter=self._get_independent_axis_filter_mode(),
            roi=roi,
            excluded_q=sorted(getattr(self, "_ai_excluded_input_q", set()) or set()),
        )

    def _current_ai_curve_arrays(self, apply_exclusions=True):
        # Keep the measured sign and intensity. Never route V5 through the legacy
        # positive-I cleaner, which discards the most informative noisy samples.
        use_cut = bool(getattr(self, "_insitu_workflow_ai_record", None)) or bool(
            getattr(self.ui, "fitCurrentDataCheckBox", None)
            and self.ui.fitCurrentDataCheckBox.isChecked()
        )
        data = getattr(self, "current_cut_data" if use_cut else "current_1d_data", None)
        options = self._workflow_options()
        recipe = self.fitting_view_model.insitu.recipe
        if getattr(self, "_insitu_workflow_ai_record", None) is not None and recipe is not None:
            options = dict(recipe.model.get("workflow_v5", options))
        self._workflow_observation_metadata = {}
        if use_cut:
            data = self._native_cbf_input(options) or data
        if not isinstance(data, dict):
            return None
        q = np.asarray(data.get("x_coords" if use_cut else "q", []), float).reshape(-1)
        y = np.asarray(data.get("y_intensity" if use_cut else "I", []), float).reshape(-1)
        if len(q) != len(y) or not len(q):
            return None
        mask = np.isfinite(q) & np.isfinite(y) & (q != 0)
        selection = self._workflow_input_selection()
        options = self._workflow_options()
        recipe = self.fitting_view_model.insitu.recipe
        if getattr(self, "_insitu_workflow_ai_record", None) is not None and recipe is not None:
            selection = dict(recipe.model.get("workflow_input_selection", selection))
            options = dict(recipe.model.get("workflow_v5", options))
        mode = selection["axis_filter"]
        if mode == "positive":
            mask &= q > 0
        elif mode == "negative":
            mask &= q < 0
        if selection.get("roi") is not None:
            lo, hi = sorted(selection["roi"])
            mask &= (q >= lo) & (q <= hi)
        if apply_exclusions:
            excluded = set(selection.get("excluded_q", []))
            mask &= np.array(
                [
                    self._ai_q_key(v) not in excluded and self._ai_q_key(abs(v)) not in excluded
                    for v in q
                ]
            )
            if self._workflow_observation_metadata and excluded:
                # Deleted display samples may lie between native detector columns.
                spacing = np.median(np.diff(np.sort(q)))
                for value in excluded:
                    value = float(value)
                    for target in [value, -value] if value > 0 else [value]:
                        nearest = int(np.argmin(abs(q - target)))
                        if abs(q[nearest] - target) <= spacing:
                            mask[nearest] = False
        if not mask.any():
            return None
        if self._workflow_observation_metadata:
            counts = np.asarray(
                self._workflow_observation_metadata["valid_pixel_counts"], dtype=float
            )
            if counts.shape != q.shape:
                raise ValueError("CBF valid-pixel counts do not match the measured curve")
            self._workflow_observation_metadata.update(
                valid_pixel_counts=counts[mask].tolist(),
                selected_points=int(mask.sum()),
                selected_positive=int(np.count_nonzero(mask & (q > 0))),
                selected_negative=int(np.count_nonzero(mask & (q < 0))),
                input_selection=selection,
            )
        sigma = data.get("err")
        self._workflow_sigma_estimated = sigma is None or bool(self._workflow_observation_metadata)
        if sigma is None:
            floor = options["absolute_noise"] or max(float(np.max(abs(y[mask]))) * 0.001, 1e-12)
            sigma = np.hypot(options["relative_noise"] * abs(y), floor)
        sigma = np.asarray(sigma, float).reshape(-1)
        if sigma.shape != q.shape:
            raise ValueError("Input sigma length does not match q")
        return (
            self._convert_q_values_for_model(q[mask], source="cut" if use_cut else data),
            y[mask],
            sigma[mask],
        )

    def _show_workflow_results(self, rows, output_dir):
        self.open_ai_fitting_workspace()
        self._workflow_v5_dialog.set_results(rows, output_dir)

    def _apply_workflow_candidate(self, row, refresh_plot=True):
        if refresh_plot and (
            getattr(self, "_insitu_workflow_busy", False)
            or getattr(self, "_insitu_workflow_state", "Idle")
            in ("Watching", "Processing", "Paused")
        ):
            # Browsing old single-curve results must not replace an active frame.
            return False
        q, fitted = np.asarray(row["native_q"]), np.asarray(row["native_fit"])
        self.I_fitting = fitted
        self.has_fitting_data = True
        self._has_fitting_data = True
        params = {}
        for i, component in enumerate(row["components"], 1):
            for key, value in component["params"].items():
                if value is not None:
                    params[f"component_{i}_{key}"] = float(value)
            params[f"component_{i}_weight"] = component["weight"]
        params.update({k: float(v) for k, v in row["global_params"].items() if v is not None})
        self.fitting = dict(
            q=q.copy(),
            I=fitted.copy(),
            meta=dict(
                source="native_v5",
                params=params,
                q_source_unit="nm",
                q_model_unit="nm",
                candidate=row,
            ),
        )
        self.display_mode = "fitting"
        self._display_mode = "fitting"
        self._fitting_mode_active = True
        if refresh_plot and getattr(self, "_insitu_workflow_ai_record", None) is None:
            self._set_curve_view_mode("compare", refresh=False)
            self._render_workflow_plot()
        return True

    def _render_workflow_plot(self):
        projection = self._ensure_curve_canvas()
        if projection is None:
            return
        fig, canvas, ax, _proxy = projection
        ax.clear()
        row = self.fitting["meta"]["candidate"]
        mode = self._get_curve_view_mode()
        if mode in ("data", "compare"):
            ax.errorbar(
                row["native_q"],
                row["observed"],
                yerr=row["sigma"],
                fmt="o",
                ms=3,
                alpha=0.6,
                label="Measured ± σ",
            )
        if mode in ("compare", "model"):
            ax.plot(row["display_q"], row["display_fit"], color="#2563eb", label="V5 model forward")
        if self._is_fit_log_y_enabled():
            ax.set_yscale("symlog", linthresh=max(min(row["sigma"]), 1e-12))
        ax.set_xlabel("q (nm⁻¹)")
        ax.set_ylabel("Intensity (input units)")
        ax.set_title(f"{row['side']} · {row['combination']}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        canvas.draw_idle()
