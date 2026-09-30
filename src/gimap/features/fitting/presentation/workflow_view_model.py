"""Workflow-state commands composed into the fitting ViewModel."""

from __future__ import annotations

from dataclasses import replace

from .workflow_state import begin_workflow_step, complete_workflow_step, fail_workflow_step


class FittingWorkflowViewModelMixin:
    """Record the progress of the Fit step without owning use-case execution."""

    def begin_workflow_step(self, key: str, message: str = "") -> None:
        self.state = replace(
            self.state, workflow=begin_workflow_step(self.state.workflow, key, message)
        )

    def complete_workflow_step(self, key: str, message: str = "") -> None:
        self.state = replace(
            self.state, workflow=complete_workflow_step(self.state.workflow, key, message)
        )

    def fail_workflow_step(self, key: str, message: str) -> None:
        self.state = replace(
            self.state, workflow=fail_workflow_step(self.state.workflow, key, message)
        )


__all__ = ["FittingWorkflowViewModelMixin"]
