"""Framework-neutral state of the fitting run (the Fit step)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


WorkflowStatus = Literal["available", "running", "complete", "error"]


@dataclass(frozen=True)
class WorkflowStepState:
    key: str
    title: str
    status: WorkflowStatus
    message: str = ""


@dataclass(frozen=True)
class FittingWorkflowState:
    steps: tuple[WorkflowStepState, ...]

    def step(self, key: str) -> WorkflowStepState:
        return next(step for step in self.steps if step.key == key)


WORKFLOW_STEPS = (("fit", "Fit"),)


def initial_workflow_state() -> FittingWorkflowState:
    return FittingWorkflowState(
        tuple(WorkflowStepState(key, title, "available") for key, title in WORKFLOW_STEPS)
    )


def begin_workflow_step(
    workflow: FittingWorkflowState, key: str, message: str = ""
) -> FittingWorkflowState:
    return _replace_step(workflow, key, status="running", message=message)


def complete_workflow_step(
    workflow: FittingWorkflowState, key: str, message: str = ""
) -> FittingWorkflowState:
    return _replace_step(workflow, key, status="complete", message=message)


def fail_workflow_step(
    workflow: FittingWorkflowState, key: str, message: str
) -> FittingWorkflowState:
    return _replace_step(workflow, key, status="error", message=message)


def _replace_step(
    workflow: FittingWorkflowState,
    key: str,
    *,
    status: WorkflowStatus,
    message: str,
) -> FittingWorkflowState:
    steps = list(workflow.steps)
    for index, step in enumerate(steps):
        if step.key == key:
            steps[index] = replace(step, status=status, message=message)
            return FittingWorkflowState(tuple(steps))
    raise KeyError(key)


__all__ = [
    "FittingWorkflowState",
    "WorkflowStepState",
    "WorkflowStatus",
    "WORKFLOW_STEPS",
    "initial_workflow_state",
    "begin_workflow_step",
    "complete_workflow_step",
    "fail_workflow_step",
]
