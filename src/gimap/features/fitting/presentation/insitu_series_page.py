"""Presentation binding for the Fitting In-situ series page (a series of 1D curves)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation import install_safe_wheel_behavior

from ..application import ReviseInSituRecipeRequest
from .views import InSituSeriesPageView
from .views.insitu_workflow_controls import PLOT_ONLY


_FAILURE_FROM_TEXT = {"Continue": "continue", "Stop": "stop"}
_FAILURE_TO_TEXT = {"continue": "Continue", "stop": "Stop", "fallback_recipe": "Continue"}
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

    def _connect(self) -> None:
        self.ui.backToSingleButton.clicked.connect(self.return_to_single_requested)
        self.ui.captureRecipeButton.clicked.connect(self.capture_recipe_requested)
        self.ui.workflowButtonGroup.idClicked.connect(self._show_workflow_step)
        self.ui.workflowControls.applyRecipeButton.clicked.connect(self._apply_recipe_edits)
        self.ui.resultsTable.itemSelectionChanged.connect(self._render_selected_record_status)

    def _show_workflow_step(self, index: int) -> None:
        key = self.ui.STEP_DEFINITIONS[index][0]
        self.ui.workflowControls.show_step(key)

    def workflow_widgets(self) -> dict[str, object]:
        controls = self.ui.workflowControls
        return {
            "run_mode": controls.runModeCombo,
            "fit_mode": controls.workflowModeCombo,
            "live_settings": controls.liveSettingsWidget,
            "sequence_settings": controls.sequenceSettingsWidget,
            "sequence_folder": controls.sequenceFolderEdit,
            "sequence_browse": controls.sequenceBrowseButton,
            "sequence_pattern": controls.sequencePatternEdit,
            "recursive": controls.recursiveCheckBox,
            "sequence_start": controls.sequenceStartSpinBox,
            "sequence_end": controls.sequenceEndSpinBox,
            "sequence_step": controls.sequenceStepSpinBox,
            "poll": controls.pollSpinBox,
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
                "Fit one representative curve in Single analysis, then use its setup here."
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
        self.set_step_state("fit", "configured")

    def _render_recipe_values(self, recipe) -> None:
        controls = self.ui.workflowControls
        self._set_combo(controls.failurePolicyCombo, _FAILURE_TO_TEXT[recipe.fitting.failure])
        workflow = recipe.model.get("workflow_v5")
        if workflow:
            mode = PLOT_ONLY if recipe.model.get("extract_only") else (
                0 if workflow.get("numerical", True) else 1
            )
            controls.workflowModeCombo.blockSignals(True)
            controls.workflowModeCombo.setCurrentIndex(mode)
            controls.workflowModeCombo.blockSignals(False)

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
            rows.append(
                (
                    self._record_attr(record, "index", values.get("file_index", "-")),
                    file_name,
                    values.get("load_status", "ok" if status == "succeeded" else status),
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
            mode = controls.fit_mode()
            model = recipe.to_dict()["model"]
            request = ReviseInSituRecipeRequest(
                current=recipe,
                scope=scope,
                selected_frame_ids=self._selected_frame_ids() if scope == "selected_and_future" else (),
                model={
                    **model,
                    "workflow_v5": {**model.get("workflow_v5", {}), "numerical": mode == 0},
                    "extract_only": mode == PLOT_ONLY,
                },
                fitting=replace(
                    recipe.fitting,
                    failure=_FAILURE_FROM_TEXT[controls.failurePolicyCombo.currentText()],
                ),
            )
            revision = self.view_model.revise_recipe(request)
            self.render_recipe(revision.recipe)
        except (KeyError, TypeError, ValueError) as exc:
            self.error_occurred.emit(str(exc))

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
        if text in {"running", "loading", "fitting"}:
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
