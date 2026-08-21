"""Workflow-state commands composed into the fitting ViewModel."""

from __future__ import annotations

from dataclasses import replace

from .state import CutGeometryDraft
from .workflow_state import (
    begin_workflow_step,
    complete_workflow_step,
    fail_workflow_step,
    invalidate_workflow_step,
)


class FittingWorkflowViewModelMixin:
    """Update verified workflow progress without owning use-case execution."""

    def begin_workflow_step(self, key: str, message: str = "") -> None:
        self.state = replace(
            self.state,
            workflow=begin_workflow_step(self.state.workflow, key, message),
        )

    def complete_workflow_step(self, key: str, message: str = "") -> None:
        updates = {
            "workflow": complete_workflow_step(self.state.workflow, key, message),
        }
        if key == "cut":
            updates.update(
                cut_status="ready",
                cut_result_analysis_revision=self.state.analysis_revision,
                cut_result_geometry_revision=self.state.cut_geometry.revision,
            )
        elif key in {"setup", "center"} and self.state.cut_status == "ready":
            updates["cut_status"] = "stale"
        self.state = replace(self.state, **updates)

    def fail_workflow_step(self, key: str, message: str) -> None:
        updates = {
            "workflow": fail_workflow_step(self.state.workflow, key, message),
        }
        if key == "cut":
            updates["cut_status"] = "error"
        self.state = replace(self.state, **updates)

    def update_cut_geometry(
        self,
        *,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> bool:
        """Commit a UI geometry draft without recalculating the cut."""
        current = self.state.cut_geometry
        values = tuple(map(float, (center_x, center_y, width, height)))
        previous = (current.center_x, current.center_y, current.width, current.height)
        if values == previous:
            return False
        draft = CutGeometryDraft(*values, revision=current.revision + 1)
        cut_status = "stale" if self.state.cut_status in {"ready", "stale"} else "idle"
        self.state = replace(
            self.state,
            cut_geometry=draft,
            cut_status=cut_status,
            workflow=complete_workflow_step(
                self.state.workflow,
                "center",
                "Center or cut geometry changed",
            ),
            status_message=(
                "Cut settings changed; update the cut to refresh results"
                if cut_status == "stale"
                else "Cut geometry ready"
            ),
        )
        return True

    def invalidate_cut(self, message: str) -> None:
        cut_status = "stale" if self.state.cut_status == "ready" else self.state.cut_status
        self.state = replace(
            self.state,
            cut_status=cut_status,
            workflow=invalidate_workflow_step(self.state.workflow, "cut", message),
            status_message=message,
        )

    def accept_analysis_revision(self, revision: int) -> None:
        """Record the canonical detector input and invalidate results from older data."""

        revision = int(revision)
        previous = self.state.analysis_revision
        if previous == revision:
            return
        cut_status = self.state.cut_status
        workflow = self.state.workflow
        message = "Image preprocessing changed; update the cut before fitting"
        if previous is not None:
            if cut_status == "ready":
                cut_status = "stale"
            workflow = invalidate_workflow_step(workflow, "center", message)
        self.state = replace(
            self.state,
            analysis_revision=revision,
            cut_status=cut_status,
            workflow=workflow,
            status_message=message if previous is not None else self.state.status_message,
        )


__all__ = ["FittingWorkflowViewModelMixin"]
