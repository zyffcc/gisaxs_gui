"""Framework-neutral state transitions for the guided fitting workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


WorkflowStatus = Literal[
    "blocked", "available", "running", "complete", "error", "stale"
]


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


WORKFLOW_STEPS = (
    ("import", "Import data"),
    ("setup", "Experiment setup"),
    ("center", "Find Yoneda"),
    ("cut", "Define cut"),
    ("fit", "Fit"),
)


def initial_workflow_state() -> FittingWorkflowState:
    return FittingWorkflowState(
        tuple(
            WorkflowStepState(key, title, "available" if index == 0 else "blocked")
            for index, (key, title) in enumerate(WORKFLOW_STEPS)
        )
    )


def begin_workflow_step(
    workflow: FittingWorkflowState, key: str, message: str = ""
) -> FittingWorkflowState:
    return _replace_step(workflow, key, status="running", message=message)


def complete_workflow_step(
    workflow: FittingWorkflowState,
    key: str,
    message: str = "",
    *,
    preserve_completed: tuple[str, ...] = (),
) -> FittingWorkflowState:
    """Complete one verified step and invalidate only affected downstream work."""
    steps = list(workflow.steps)
    index = _step_index(steps, key)
    steps[index] = replace(steps[index], status="complete", message=message)
    for downstream_index in range(index + 1, len(steps)):
        old = steps[downstream_index]
        if old.key in preserve_completed and old.status == "complete":
            continue
        status: WorkflowStatus = (
            "stale" if old.status in {"complete", "stale"} else "blocked"
        )
        steps[downstream_index] = replace(old, status=status, message="")

    # The first unmet prerequisite is the only newly available step.  A stale
    # step is already actionable and retains that explicit warning state.
    for candidate_index in range(index + 1, len(steps)):
        candidate = steps[candidate_index]
        if candidate.status == "complete":
            continue
        if candidate.status == "blocked":
            steps[candidate_index] = replace(candidate, status="available")
        break
    return FittingWorkflowState(tuple(steps))


def fail_workflow_step(
    workflow: FittingWorkflowState, key: str, message: str
) -> FittingWorkflowState:
    return _replace_step(workflow, key, status="error", message=message)


def invalidate_workflow_step(
    workflow: FittingWorkflowState, key: str, message: str = ""
) -> FittingWorkflowState:
    """Mark a calculated step and its calculated dependants stale."""
    steps = list(workflow.steps)
    index = _step_index(steps, key)
    target = steps[index]
    target_status: WorkflowStatus = (
        "stale" if target.status in {"complete", "stale"} else "available"
    )
    steps[index] = replace(target, status=target_status, message=message)
    for downstream_index in range(index + 1, len(steps)):
        old = steps[downstream_index]
        if old.status in {"complete", "stale"}:
            steps[downstream_index] = replace(old, status="stale", message="")
    return FittingWorkflowState(tuple(steps))


def _replace_step(
    workflow: FittingWorkflowState,
    key: str,
    *,
    status: WorkflowStatus,
    message: str,
) -> FittingWorkflowState:
    steps = list(workflow.steps)
    index = _step_index(steps, key)
    steps[index] = replace(steps[index], status=status, message=message)
    return FittingWorkflowState(tuple(steps))


def _step_index(steps: list[WorkflowStepState], key: str) -> int:
    for index, step in enumerate(steps):
        if step.key == key:
            return index
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
    "invalidate_workflow_step",
]
