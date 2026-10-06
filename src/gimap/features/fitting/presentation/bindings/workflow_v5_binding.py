"""Bridge the portable native-node workflow to single and in-situ fitting."""

import numpy as np
from PyQt5.QtCore import Qt

from src.gimap.app.presentation.i18n import tr, trf

from ..workflow_v5_dialog import WorkflowV5Dialog
from ...application.workflow_v5 import bundled_workflow, default_options

SAME_CURVE = 0.01
"""Fitting's model reproduces a loaded solution when it differs from it by at most this (relative)."""


class WorkflowV5BindingMixin:
    def _native_curve_input(self, data, options):
        """Native detector columns written by Analyze: the counting contract of the CBF path.

        Analyze averages each detector column of the horizontal cut band (mean
        counts per pixel, Poisson σ, pixel count), exactly the observation the
        detector path used to build here, so V5 gets the same metadata and the
        same working tolerance on σ.  Other curves return ``None``.
        """
        observation = data.get("observation") or {}
        pixels, counting = data.get("pixels"), data.get("err")
        if observation.get("source") != "native_detector_columns" or pixels is None or counting is None:
            return None
        q = np.asarray(data.get("q", []), float).reshape(-1)
        y = np.asarray(data.get("I", []), float).reshape(-1)
        pixels = np.asarray(pixels, float).reshape(-1)
        counting = np.asarray(counting, float).reshape(-1)
        if not (q.shape == y.shape == pixels.shape == counting.shape) or not q.size:
            return None
        cbf = observation.get("file_format") == "cbf"
        self._workflow_observation_metadata = dict(
            source="native_cbf_columns" if cbf else "native_detector_columns",
            q_source="analyze_native_columns",
            measured_columns=int(q.size),
            valid_pixel_counts=pixels.tolist(),
            intensity_unit=observation.get("intensity_unit", "counts_per_pixel"),
            gap_margin_px=int(observation.get("gap_guard_px", 0)),
            threshold_enabled=bool(observation.get("threshold_enabled", False)),
            counting_model_valid=bool(observation.get("counting_model_valid", True)),
            mirror_replaced_pixels=0,
            stack_count=max(1, int(observation.get("summed_frames", 1))),
            uncertainty="Poisson sum/count approximation; additional relative noise is a working tolerance",
            sampling="Native measured columns; no interpolated points across detector gaps",
        )
        sigma = np.sqrt(
            counting**2 + (options["relative_noise"] * abs(y)) ** 2 + options["absolute_noise"] ** 2
        )
        return {**data, "err": sigma}

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
            dialog.candidate_selected.connect(self.show_workflow_candidate)
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
        self._set_ai_workspace_status(f"{label} · ready to fit", 0)

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
            version = revision.recipe.version
            dialog.show_status(lambda: trf("Saved settings v{version} for future frames.", version=version))

        dialog.settings_changed.connect(save)
        dialog.setAttribute(Qt.WA_DeleteOnClose)  # one made per opening: gone when closed (no language hook left)
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
            # The folded views (|q| overlay, ±q average, −q as |q|) show the fitting
            # range in |q|, so it selects both signs of the native columns.
            roi_abs=self._get_q_combination_mode() != "separate",
            excluded_q=sorted(getattr(self, "_ai_excluded_input_q", set()) or set()),
        )

    def _current_ai_curve_arrays(self, apply_exclusions=True):
        # Keep the measured sign and intensity. Never route V5 through the legacy
        # positive-I cleaner, which discards the most informative noisy samples.
        data = getattr(self, "current_1d_data", None)
        options = self._workflow_options()
        recipe = self.fitting_view_model.insitu.recipe
        if getattr(self, "_insitu_workflow_ai_record", None) is not None and recipe is not None:
            options = dict(recipe.model.get("workflow_v5", options))
        self._workflow_observation_metadata = {}
        if isinstance(data, dict):
            data = self._native_curve_input(data, options) or data
        if not isinstance(data, dict):
            return None
        q = np.asarray(data.get("q", []), float).reshape(-1)
        y = np.asarray(data.get("I", []), float).reshape(-1)
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
            ranged = np.abs(q) if selection.get("roi_abs") else q
            mask &= (ranged >= lo) & (ranged <= hi)
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
            self._convert_q_values_for_model(q[mask], source=data),
            y[mask],
            sigma[mask],
        )

    def _show_workflow_results(self, rows, output_dir):
        self.open_ai_fitting_workspace()
        self._workflow_v5_dialog.set_results(rows, output_dir)

    def show_workflow_candidate(self, row) -> bool:
        """A ``native_v5`` solution (1D Predict, or Analyze ▸ Results ▸ Show in Fitting) put into Components and
        Global and drawn by Fitting's model, which then refines it; drawn as it is when it cannot be put there."""
        if self._insitu_frame_active():
            return False
        model = str(row.get("combination") or "").replace("_", " ")
        try:
            converted = self.fitting_view_model.map_native_solution(row)
            self._load_parameter_mapping(converted.mapping)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            shown = self._apply_workflow_candidate(row)
            self._set_fitting_inline_feedback(
                tr("{model} is drawn, but not put into Components: {reason}").format(model=model, reason=exc), "warning"
            )
            return shown
        self._perform_manual_fitting(reveal_result=True)  # the model on the data
        tabs = getattr(self.ui, "fittingModeTabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(0)  # Components: where the values went
        deviation = converted.max_deviation
        if not np.isfinite(deviation) or deviation <= SAME_CURVE:
            message, kind = tr("{model} is in Components and Global; Fitting's model draws the same curve{within}.").format(
                model=model, within="" if not np.isfinite(deviation) else f" (≤ {100 * deviation:.2g} %)"), "info"
        else:
            message, kind = tr(
                "{model} is in Components and Global as a start: Fitting's model of it differs from the solution by up "
                "to {percent} % (its Vertical Cylinder weights radii by R⁴). Refine it here."
            ).format(model=model, percent=f"{100 * deviation:.3g}"), "warning"
        self._set_fitting_inline_feedback(message, kind)
        return True

    def _insitu_frame_active(self) -> bool:
        return bool(getattr(self, "_insitu_workflow_busy", False)) or getattr(
            self, "_insitu_workflow_state", "Idle"
        ) in ("Watching", "Processing", "Paused")

    def _apply_workflow_candidate(self, row, refresh_plot=True):
        if refresh_plot and self._insitu_frame_active():
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
