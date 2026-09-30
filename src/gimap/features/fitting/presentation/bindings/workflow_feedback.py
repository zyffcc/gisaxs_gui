"""Synchronize the fitting workflow state with the presentation widgets."""

from __future__ import annotations


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

    def _sync_fitting_workflow(self) -> None:
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
        fit_status = self.fitting_view_model.state.workflow.step("fit").status
        if fit_status == "running":
            text, kind = "Fitting…", "running"
        elif fit_status == "complete":
            text, kind = "Fit ready", "complete"
        elif fit_status == "error":
            text, kind = "Fit needs attention", "error"
        elif getattr(self, "current_1d_data", None) is not None:
            text, kind = "Curve ready", "ready"
        else:
            text, kind = "No curve", "idle"
        chip.setText(text)
        chip.setProperty("statusKind", kind)
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _sync_fitting_action_availability(self) -> None:
        curve_ready = getattr(self, "current_1d_data", None) is not None
        for name in (
            "FittingManualFittingButton",
            "FittingGlobalSearchButton",
            "FittingAutoRefineButton",
            "FittingAutoFittingButton",
        ):
            button = getattr(self.ui, name, None)
            if button is not None:
                button.setEnabled(curve_ready)

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
