"""Presentation binding for the feature-owned Fitting In-situ series page."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation import apply_design_system, install_safe_wheel_behavior

from ..application import (
    InSituFittingPolicy,
    InSituTrackingPolicy,
    ReviseInSituRecipeRequest,
)
from .views import InSituSeriesPageView


_TRACKING_FROM_TEXT = {
    "Fixed": "fixed",
    "Detect each frame": "detect_each_frame",
    "Previous success": "previous_success",
}
_TRACKING_TO_TEXT = {value: key for key, value in _TRACKING_FROM_TEXT.items()}
_INITIAL_FROM_TEXT = {
    "Previous success": "previous_success",
    "Recipe values": "recipe_values",
    "AI each frame": "ai_each_frame",
}
_INITIAL_TO_TEXT = {value: key for key, value in _INITIAL_FROM_TEXT.items()}
_REFINEMENT_FROM_TEXT = {
    "Plot only": "plot_only",
    "Every frame": "every_frame",
    "Every N frames": "every_n",
    "On quality drop": "quality_drop",
}
_REFINEMENT_TO_TEXT = {value: key for key, value in _REFINEMENT_FROM_TEXT.items()}
_FAILURE_FROM_TEXT = {
    "Continue": "continue",
    "Fallback to recipe": "fallback_recipe",
    "Stop": "stop",
}
_FAILURE_TO_TEXT = {value: key for key, value in _FAILURE_FROM_TEXT.items()}
_SCOPE_FROM_TEXT = {
    "Future frames": "future",
    "Selected + future": "selected_and_future",
    "All frames (reprocess)": "all",
}


class InSituSeriesPage(QWidget):
    """Render recipe and workflow state without owning scientific calculations."""

    return_to_single_requested = pyqtSignal()
    capture_recipe_requested = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, view_model, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view_model = view_model
        self.ui = InSituSeriesPageView()
        self.ui.setupUi(self)
        self._displayed_records: list[object] = []
        self._connect()
        self.render_recipe(self.view_model.recipe)
        self.render_workflow(self.view_model.state)
        install_safe_wheel_behavior(self)
        apply_design_system(self)

    def _connect(self) -> None:
        self.ui.backToSingleButton.clicked.connect(self.return_to_single_requested)
        self.ui.captureRecipeButton.clicked.connect(self.capture_recipe_requested)
        self.ui.workflowButtonGroup.idClicked.connect(self._show_workflow_step)
        self.ui.workflowControls.applyRecipeButton.clicked.connect(self._apply_recipe_edits)
        self.ui.workflowControls.refinementCombo.currentTextChanged.connect(
            lambda text: self.ui.workflowControls.refineEverySpinBox.setEnabled(
                text == "Every N frames"
            )
        )
        self.ui.resultsTable.itemSelectionChanged.connect(self._render_selected_record_status)
        self.ui.workflowControls.refineEverySpinBox.setEnabled(False)

    def _show_workflow_step(self, index: int) -> None:
        key = self.ui.STEP_DEFINITIONS[index][0]
        self.ui.workflowControls.show_step(key)

    def workflow_widgets(self) -> dict[str, object]:
        controls = self.ui.workflowControls
        return {
            "run_mode": controls.runModeCombo,
            "auto_show": controls.autoShowCheckBox,
            "auto_cut": controls.autoCutCheckBox,
            "auto_fit": controls.autoFitCheckBox,
            "use_previous": controls.usePreviousCheckBox,
            "full_auto_fit": controls.fullAutoFitCheckBox,
            "auto_refine": controls.autoRefineCheckBox,
            "profile": controls.profileCombo,
            "live_settings": controls.liveSettingsWidget,
            "sequence_settings": controls.sequenceSettingsWidget,
            "sequence_folder": controls.sequenceFolderEdit,
            "sequence_browse": controls.sequenceBrowseButton,
            "sequence_pattern": controls.sequencePatternEdit,
            "sequence_start": controls.sequenceStartSpinBox,
            "sequence_end": controls.sequenceEndSpinBox,
            "sequence_step": controls.sequenceStepSpinBox,
            "poll": controls.pollSpinBox,
            "fit_every": controls.fitEverySpinBox,
            "ui_every": controls.uiEverySpinBox,
            "stable": controls.stableCheckBox,
            "start": self.ui.startWatchButton,
            "process": self.ui.startProcessButton,
            "pause": self.ui.pauseButton,
            "stop": self.ui.stopButton,
            "trend": controls.trendButton,
            "heatmap": controls.heatmapButton,
            "export": controls.exportButton,
            "clear_cache": controls.clearCacheButton,
            "open_cache": controls.openCacheButton,
            "status_labels": self.ui.statusValueLabels,
            "log": self.ui.logBrowser,
            "image_label": self.ui.currentImageLabel,
        }

    def render_recipe(self, recipe) -> None:
        enabled = recipe is not None
        controls = self.ui.workflowControls
        controls.applyRecipeButton.setEnabled(enabled)
        for button in (self.ui.startWatchButton, self.ui.startProcessButton):
            button.setEnabled(enabled)
        if not enabled:
            self.ui.recipeStatusLabel.setText("No Recipe")
            self.ui.recipeStatusLabel.setProperty("statusKind", "warning")
            self.ui.recipeMetaLabel.setText(
                "Analyze one representative frame, then explicitly transfer its setup."
            )
            self._repolish(self.ui.recipeStatusLabel)
            return

        self.ui.recipeStatusLabel.setText(f"Recipe v{recipe.version} ready")
        self.ui.recipeStatusLabel.setProperty("statusKind", "complete")
        origin = "Single analysis" if recipe.source == "single_analysis" else "In-situ edit"
        self.ui.recipeMetaLabel.setText(
            f"Source: {origin} · Created: {recipe.created_at} · "
            f"Scope: {self.view_model.recipe_scope.replace('_', ' ')}"
        )
        self._render_recipe_values(recipe)
        self._repolish(self.ui.recipeStatusLabel)
        for key in ("preprocess", "geometry", "cut", "fit"):
            self.set_step_state(key, "configured")

    def _render_recipe_values(self, recipe) -> None:
        controls = self.ui.workflowControls
        setup = recipe.experiment_setup
        preprocess = recipe.preprocessing
        cut = recipe.cut
        assignments = (
            (controls.distanceSpinBox, setup.get("distance_mm", 2000.0)),
            (controls.grazingSpinBox, setup.get("grazing_angle_deg", 0.2)),
            (controls.wavelengthSpinBox, setup.get("wavelength_nm", 0.1)),
            (controls.centerXSpinBox, setup.get("beam_center_x_px", 0.0)),
            (controls.centerYSpinBox, setup.get("beam_center_y_px", 0.0)),
            (controls.pixelXSpinBox, setup.get("pixel_size_x_um", 172.0)),
            (controls.pixelYSpinBox, setup.get("pixel_size_y_um", 172.0)),
            (controls.thresholdMinSpinBox, preprocess.get("threshold_min", -1e12)),
            (controls.thresholdMaxSpinBox, preprocess.get("threshold_max", 1e12)),
            (controls.mirrorMarginSpinBox, preprocess.get("mirror_gap_margin_px", 0)),
            (controls.cutCenterVerticalSpinBox, cut.get("center_vertical_px", 0.0)),
            (controls.cutCenterParallelSpinBox, cut.get("center_parallel_px", 0.0)),
            (controls.cutVerticalSpinBox, cut.get("cut_vertical_px", 10.0)),
            (controls.cutParallelSpinBox, cut.get("cut_parallel_px", 10.0)),
            (controls.yonedaThicknessSpinBox, cut.get("auto_horizontal_thickness_px", 5)),
            (controls.refineEverySpinBox, recipe.fitting.refine_every_n),
        )
        for editor, value in assignments:
            editor.blockSignals(True)
            editor.setValue(float(value) if hasattr(editor, "decimals") else int(value))
            editor.blockSignals(False)
        controls.flipUdCheckBox.setChecked(bool(preprocess.get("flip_ud", False)))
        controls.thresholdCheckBox.setChecked(
            bool(preprocess.get("threshold_enabled", False))
        )
        controls.mirrorFillCheckBox.setChecked(
            bool(preprocess.get("mirror_fill_gaps", False))
        )
        self._set_combo(controls.centerTrackingCombo, _TRACKING_TO_TEXT[recipe.tracking.center])
        self._set_combo(controls.yonedaTrackingCombo, _TRACKING_TO_TEXT[recipe.tracking.yoneda])
        self._set_combo(
            controls.fitInitializationCombo,
            _INITIAL_TO_TEXT[recipe.fitting.initialization],
        )
        self._set_combo(controls.refinementCombo, _REFINEMENT_TO_TEXT[recipe.fitting.refinement])
        self._set_combo(controls.failurePolicyCombo, _FAILURE_TO_TEXT[recipe.fitting.failure])
        controls.refineEverySpinBox.setEnabled(recipe.fitting.refinement == "every_n")
        controls.usePreviousCheckBox.setChecked(
            recipe.fitting.initialization == "previous_success"
        )
        controls.fullAutoFitCheckBox.setChecked(
            recipe.fitting.initialization == "ai_each_frame"
        )
        controls.autoRefineCheckBox.setChecked(recipe.fitting.refinement != "plot_only")

    def render_workflow(self, workflow) -> None:
        total = workflow.processed_count + len(workflow.pending_paths)
        progress = None if workflow.status == "running" and total == 0 else (
            0.0 if total == 0 else workflow.processed_count / total
        )
        status_map = {
            "idle": "idle",
            "running": "running",
            "paused": "paused",
            "cancelled": "cancelled",
            "completed": "succeeded",
            "error": "failed",
        }
        self.ui.jobStatus.set_state(
            status_map.get(workflow.status, "idle"),
            f"{workflow.processed_count} processed · {workflow.failed_count} failed",
            progress=progress,
        )
        self.render_records(workflow.records)

    def render_records(self, records: Sequence[object]) -> None:
        self._displayed_records = list(records)
        rows = []
        recipe_version = self.view_model.recipe.version if self.view_model.recipe else "-"
        for record in records:
            values = self._record_values(record)
            status = self._record_attr(record, "status", values.get("status", "-"))
            paths = self._record_attr(record, "paths", ())
            file_name = ", ".join(paths) if paths else str(values.get("file_name", "-"))
            load = values.get("load_status", "ok" if status == "succeeded" else status)
            preprocess = values.get("preprocess_status", load)
            geometry = values.get("geometry_status", load)
            rows.append(
                (
                    self._record_attr(record, "index", values.get("file_index", "-")),
                    file_name,
                    load,
                    preprocess,
                    geometry,
                    values.get("cut_status", "-"),
                    values.get("fit_status", "-"),
                    values.get("recipe_version", recipe_version),
                    values.get("chi_square", "-"),
                )
            )
        self.ui.resultsTable.set_rows(rows)

    def set_step_state(self, key: str, state: str) -> None:
        button = self.ui.workflowButtons.get(key)
        if button is None:
            return
        button.setProperty("workflowState", state)
        self._repolish(button)

    def _render_selected_record_status(self) -> None:
        row = self.ui.resultsTable.currentRow()
        if row < 0 or row >= len(self._displayed_records):
            return
        record = self._displayed_records[row]
        values = self._record_values(record)
        status = str(self._record_attr(record, "status", values.get("status", "pending")))
        self.set_step_state("source", self._normalize_step_state(values.get("load_status", status)))
        self.set_step_state(
            "preprocess", self._normalize_step_state(values.get("preprocess_status", status))
        )
        self.set_step_state(
            "geometry", self._normalize_step_state(values.get("geometry_status", status))
        )
        self.set_step_state("cut", self._normalize_step_state(values.get("cut_status", "pending")))
        self.set_step_state("fit", self._normalize_step_state(values.get("fit_status", "pending")))
        self.set_step_state("results", self._normalize_step_state(status))

    def _apply_recipe_edits(self) -> None:
        recipe = self.view_model.recipe
        if recipe is None:
            self.error_occurred.emit("Capture a Single analysis Recipe first.")
            return
        controls = self.ui.workflowControls
        try:
            scope = _SCOPE_FROM_TEXT[controls.changeScopeCombo.currentText()]
            request = ReviseInSituRecipeRequest(
                current=recipe,
                scope=scope,
                selected_frame_ids=self._selected_frame_ids() if scope == "selected_and_future" else (),
                experiment_setup=self._experiment_setup_values(),
                preprocessing=self._preprocessing_values(),
                cut=self._cut_values(),
                tracking=InSituTrackingPolicy(
                    center=_TRACKING_FROM_TEXT[controls.centerTrackingCombo.currentText()],
                    yoneda=_TRACKING_FROM_TEXT[controls.yonedaTrackingCombo.currentText()],
                ),
                fitting=InSituFittingPolicy(
                    initialization=_INITIAL_FROM_TEXT[controls.fitInitializationCombo.currentText()],
                    refinement=_REFINEMENT_FROM_TEXT[controls.refinementCombo.currentText()],
                    refine_every_n=controls.refineEverySpinBox.value(),
                    failure=_FAILURE_FROM_TEXT[controls.failurePolicyCombo.currentText()],
                ),
            )
            revision = self.view_model.revise_recipe(request)
            self.render_recipe(revision.recipe)
        except (KeyError, TypeError, ValueError) as exc:
            self.error_occurred.emit(str(exc))

    def _experiment_setup_values(self) -> dict[str, float]:
        c = self.ui.workflowControls
        return {
            "distance_mm": c.distanceSpinBox.value(),
            "grazing_angle_deg": c.grazingSpinBox.value(),
            "wavelength_nm": c.wavelengthSpinBox.value(),
            "beam_center_x_px": c.centerXSpinBox.value(),
            "beam_center_y_px": c.centerYSpinBox.value(),
            "pixel_size_x_um": c.pixelXSpinBox.value(),
            "pixel_size_y_um": c.pixelYSpinBox.value(),
        }

    def _preprocessing_values(self) -> dict[str, object]:
        c = self.ui.workflowControls
        return {
            "flip_ud": c.flipUdCheckBox.isChecked(),
            "threshold_enabled": c.thresholdCheckBox.isChecked(),
            "threshold_min": c.thresholdMinSpinBox.value(),
            "threshold_max": c.thresholdMaxSpinBox.value(),
            "mirror_fill_gaps": c.mirrorFillCheckBox.isChecked(),
            "mirror_gap_margin_px": c.mirrorMarginSpinBox.value(),
        }

    def _cut_values(self) -> dict[str, object]:
        c = self.ui.workflowControls
        return {
            "center_vertical_px": c.cutCenterVerticalSpinBox.value(),
            "center_parallel_px": c.cutCenterParallelSpinBox.value(),
            "cut_vertical_px": c.cutVerticalSpinBox.value(),
            "cut_parallel_px": c.cutParallelSpinBox.value(),
            "auto_horizontal_thickness_px": c.yonedaThicknessSpinBox.value(),
        }

    def _selected_frame_ids(self) -> tuple[str, ...]:
        rows = sorted({item.row() for item in self.ui.resultsTable.selectedItems()})
        return tuple(
            item.text()
            for row in rows
            if (item := self.ui.resultsTable.item(row, 0)) is not None and item.text()
        )

    @staticmethod
    def _record_values(record: object) -> Mapping[str, object]:
        if isinstance(record, Mapping):
            return record
        values = getattr(record, "values", {})
        return values if isinstance(values, Mapping) else {}

    @staticmethod
    def _record_attr(record: object, name: str, default):
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    @staticmethod
    def _normalize_step_state(value: object) -> str:
        text = str(value).lower()
        if text in {"ok", "succeeded", "complete", "completed"}:
            return "complete"
        if text in {"running", "loading", "cutting", "fitting"}:
            return "running"
        if text.startswith("fail") or text == "error":
            return "error"
        if text == "skipped":
            return "skipped"
        return "pending"

    @staticmethod
    def _set_combo(combo, text: str) -> None:
        combo.blockSignals(True)
        combo.setCurrentText(text)
        combo.blockSignals(False)

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


__all__ = ["InSituSeriesPage"]
