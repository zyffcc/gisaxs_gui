"""Public Api coordination for Trainset."""

from __future__ import annotations


from pathlib import Path

from typing import Any, Dict


class PublicApiMixin:
    """Own public api presentation behavior."""

    def _update_capabilities(self) -> None:
        available = self.simulation_port.is_available()
        self.page.preview_capability.setText(
            "BornAgain local simulation available"
            if available
            else "BornAgain not installed locally · reference preview only"
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

    def validate_parameters(self):
        valid, errors, warnings = self.trainset_view_model.validate_config(
            self._collect_config(),
            simulation_available=self.simulation_port.is_available(),
        )
        return valid, "\n".join(errors or warnings)

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
        self.page.design_tabs.setCurrentIndex(0)
        self.page.validation_badge.setText("Not validated")
        self.page.threshold_summary.setText(
            "Load a reference to calculate detector-gap and hot-pixel locations."
        )
        self._update_capabilities()
        self._update_geometry_label()
        if remember:
            self._autosave_timer.start(100)
        self.status_updated.emit("TrainSet settings reset to built-in defaults")
