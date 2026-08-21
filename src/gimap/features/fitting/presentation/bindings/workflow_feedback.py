"""Synchronize verified fitting workflow state with presentation widgets."""

from __future__ import annotations

import numpy as np

from ..detector_data_access import analysis_image_for


class WorkflowFeedbackMixin:
    def _set_numeric_control_silently(self, name: str, value: float) -> bool:
        """Update a derived/programmatic value without emitting a user-edit command."""
        widget = getattr(self.ui, name, None)
        if widget is None:
            return False
        old_block = widget.blockSignals(True)
        try:
            widget.setValue(value)
        finally:
            widget.blockSignals(old_block)
        return True

    def _has_existing_cut_result(self) -> bool:
        cut = getattr(self, "current_cut_data", None)
        if not isinstance(cut, dict):
            return False
        x_values = cut.get("x_coords")
        y_values = cut.get("y_intensity", cut.get("intensity", cut.get("I")))
        if x_values is None or y_values is None:
            return False
        return bool(
            np.asarray(x_values).size
            and np.asarray(y_values).size
        )

    def _refresh_existing_cut_preserving_view(self) -> bool:
        """Recalculate an existing cut without taking ownership of navigation."""
        if not self._has_existing_cut_result():
            return False
        self._perform_cut(reveal_result=False)
        return True

    def _record_cut_geometry_draft(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> None:
        changed = self.fitting_view_model.update_cut_geometry(
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
        )
        if not changed:
            return
        self._sync_fitting_workflow()
        if self.fitting_view_model.state.cut_status == "stale":
            self._set_fitting_inline_feedback(
                "Center or cut geometry changed. The previous cut is stale; "
                "click Extract / Update Cut when ready.",
                "warning",
            )

    def _mark_cut_stale(self, message: str) -> None:
        self.fitting_view_model.invalidate_cut(message)
        self._sync_fitting_workflow()
        if self.fitting_view_model.state.cut_status == "stale":
            self._set_fitting_inline_feedback(message, "warning")

    def _sync_fitting_workflow(self) -> None:
        header = getattr(self.ui, "fittingWorkflowHeader", None)
        if header is not None:
            header.render(self.fitting_view_model.state.workflow)
        self._sync_fitting_result_status()
        self._sync_fitting_action_availability()

    def _begin_fitting_step(self, key: str, message: str = "") -> None:
        self.fitting_view_model.begin_workflow_step(key, message)
        self._sync_fitting_workflow()
        self._set_fitting_inline_feedback("", "info")

    def _complete_fitting_step(self, key: str, message: str = "") -> None:
        self.fitting_view_model.complete_workflow_step(key, message)
        self._sync_fitting_workflow()

    def _fail_fitting_step(self, key: str, message: str) -> None:
        self.fitting_view_model.fail_workflow_step(key, message)
        self._sync_fitting_workflow()
        self._set_fitting_inline_feedback(message, "error")

    def _sync_fitting_result_status(self) -> None:
        chip = getattr(self.ui, "fittingResultStatusChip", None)
        if chip is None:
            return
        workflow = self.fitting_view_model.state.workflow
        fit_status = workflow.step("fit").status
        cut_status = workflow.step("cut").status
        if fit_status == "running":
            text, kind = "Fitting…", "running"
        elif fit_status == "complete":
            text, kind = "Fit ready", "complete"
        elif fit_status == "error":
            text, kind = "Fit needs attention", "error"
        elif cut_status == "complete":
            text, kind = "Cut ready", "ready"
        elif self.fitting_view_model.state.cut_status == "stale":
            text, kind = "Cut needs update", "stale"
        elif self.fitting_view_model.state.curve_status == "ready":
            text, kind = "1D curve ready", "ready"
        elif self.fitting_view_model.state.image_status == "ready":
            text, kind = "Image ready", "ready"
        else:
            text, kind = "Waiting for cut data", "idle"
        chip.setText(text)
        chip.setProperty("statusKind", kind)
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _sync_fitting_action_availability(self) -> None:
        image_ready = analysis_image_for(self) is not None
        cut = getattr(self, "current_cut_data", None)
        cut_ready = (
            isinstance(cut, dict)
            and bool(cut.get("x_coords") is not None)
            and self.fitting_view_model.state.cut_status == "ready"
        )
        one_d_ready = getattr(self, "current_1d_data", None) is not None
        for name in ("gisaxsInputCenterAutoFindingButton", "gisaxsInputCutButton"):
            button = getattr(self.ui, name, None)
            if button is not None:
                button.setEnabled(image_ready)
        use_cut = bool(
            getattr(self.ui, "fitCurrentDataCheckBox", None)
            and self.ui.fitCurrentDataCheckBox.isChecked()
        )
        fitting_input_ready = cut_ready if use_cut else (one_d_ready or cut_ready)
        for name in (
            "FittingManualFittingButton",
            "FittingAutoRefineButton",
            "FittingAutoFittingButton",
        ):
            button = getattr(self.ui, name, None)
            if button is not None:
                button.setEnabled(fitting_input_ready)

    def _set_fitting_inline_feedback(self, message: str, kind: str = "info") -> None:
        banner = getattr(self.ui, "fittingInlineFeedback", None)
        if banner is None:
            return
        banner.setText(str(message))
        banner.setProperty("feedbackKind", kind)
        banner.setVisible(bool(str(message).strip()))
        banner.style().unpolish(banner)
        banner.style().polish(banner)


__all__ = ["WorkflowFeedbackMixin"]
