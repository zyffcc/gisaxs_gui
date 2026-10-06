"""Public Api coordination for Trainset."""

from __future__ import annotations

import copy
from pathlib import Path

from typing import Any, Dict

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from .detector_design import NO_REFERENCE_THRESHOLD


class PublicApiMixin:
    """Own public api presentation behavior."""

    def _update_capabilities(self) -> None:
        available = self.simulation_port.is_available()
        self.page.texts.set(
            self.page.preview_capability,
            "BornAgain local simulation available"
            if available
            else "BornAgain not installed locally · reference preview only",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return self._collect_config()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        if not isinstance(parameters, dict):
            return
        self.config = self.trainset_view_model.merge_config_with_defaults(parameters)
        self._apply_config_to_page(self.config)
        reference = self.config.get("project", {}).get("reference_file")
        if reference and Path(reference).exists():
            self._load_reference(str(reference))
        self._update_capabilities()
        self._update_geometry_label()
        runtime = self.config.get("runtime", {})
        hpc = self.config.get("hpc", {})
        if runtime.get("last_job_id") and hpc.get("user") and hpc.get("remote_path"):
            self.monitor_timer.start()
        # Whatever loaded the parameters (Load project, File ▸ Load parameters, Undo): this design
        # has not been validated, previewed or checked against the model. Last, so it also
        # overrides the edits _load_reference made. Undo restores its own progress afterwards;
        # the Monitor step keeps showing a job that still runs.
        self.page.set_validation_state("Not validated", "pending")
        self._invalidate_design_checks(last_step=3)

    def validate_parameters(self):
        valid, errors, warnings = self.trainset_view_model.validate_config(
            self._collect_config(),
            simulation_available=self.simulation_port.is_available(),
        )
        return valid, "\n".join(errors or warnings)

    def _reset_clicked(self) -> None:
        """The page's Reset button: reset at once, and offer Undo (the global reset has none)."""
        previous = copy.deepcopy(self._collect_config())
        progress = self._page_progress()
        self.reset_to_defaults()

        def undo() -> None:
            self.set_parameters(previous)
            reference = previous.get("project", {}).get("reference_file")
            if not reference or self.reference_image is not None:
                self._restore_page_progress(progress)
            # else the reference could not be read again: keep the reloaded (not validated) state
            if self.page.auto_remember_check.isChecked():
                self._autosave_timer.start(100)  # remember the restored design again
            self.status_updated.emit("Trainset settings restored")

        show_toast(
            self.page,
            tr("Trainset reset to defaults"),
            action=(tr("Undo"), undo),
            timeout_ms=10000,
        )

    def _page_progress(self) -> Dict[str, Any]:
        """What the page says about progress: steps, readiness gates, design stages and badge.

        Undo restores it with the settings, so the badge never claims a validation or preview
        that the step list or the gate table no longer shows.
        """
        page = self.page
        table = page.preview_gate_table
        return {
            "steps": page.step_entries(),  # (English template, values): shown again in either language
            "gates": [
                table.item(row, 1).text() if table.item(row, 1) is not None else ""
                for row in range(table.rowCount())
            ],
            "stages": page.design_stages_ready(),
            "design_tab": page.design_tabs.currentIndex(),
            "badge": (page.validation_text(), page.validation_state()),  # English: shown translated
        }

    def _restore_page_progress(self, progress: Dict[str, Any]) -> None:
        page = self.page
        for index, (state, values) in enumerate(progress["steps"]):
            page.set_step_state(index, state, **values)
        for row, state in enumerate(progress["gates"]):
            item = page.preview_gate_table.item(row, 1)
            if item is not None:
                item.setText(state)
        for index, ready in enumerate(progress["stages"]):
            page.set_design_stage_ready(index, ready)
        page.design_tabs.setCurrentIndex(progress["design_tab"])
        page.set_validation_state(*progress["badge"])

    def reset_to_defaults(self) -> None:
        remember = self.page.auto_remember_check.isChecked()
        self.monitor_timer.stop()
        self.config = self.trainset_view_model.default_config()
        self.config.setdefault("runtime", {})["auto_remember"] = remember
        self.reference_image = None
        self._apply_config_to_page(self.config)
        for canvas in (
            self.page.full_detector_canvas,
            self.page.roi_design_canvas,
            self.page.masked_design_canvas,
            self.page.mask_only_canvas,
        ):
            canvas.set_draw_mode("")
            canvas.set_data(None)
        for index in range(4):
            self.page.set_design_stage_ready(index, False)
        for index in range(len(self.page.STEPS)):
            self.page.set_step_state(index, "Not started")
        # The checks of the old design no longer hold (row 3 follows the storage check box).
        self._invalidate_design_checks()
        self.page.design_tabs.setCurrentIndex(0)
        self.page.set_validation_state("Not validated", "pending")
        self.page.texts.set(self.page.design_info, "No reference loaded")  # not the file just dropped
        self.page.texts.set(self.page.design_info, "", setter="setToolTip")
        self.page.texts.set(self.page.threshold_summary, NO_REFERENCE_THRESHOLD)
        self._update_capabilities()
        self._update_geometry_label()
        if remember:
            self._autosave_timer.start(100)
        self.status_updated.emit("TrainSet settings reset to built-in defaults")
