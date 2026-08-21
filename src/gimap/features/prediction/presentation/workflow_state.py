"""Typed presentation state for the three-step Prediction workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WorkflowStepState = Literal[
    "upcoming",
    "active",
    "complete",
    "running",
    "error",
]


@dataclass(frozen=True)
class PredictionWorkflowSnapshot:
    """Facts rendered by the workflow header; no Qt objects or click history."""

    input_ready: bool = False
    model_ready: bool = False
    framework_ready: bool = False
    prediction_running: bool = False
    prediction_succeeded: bool = False
    error_step: int | None = None

    @property
    def active_step(self) -> int:
        if not self.input_ready:
            return 1
        if not (self.model_ready and self.framework_ready):
            return 2
        return 3

    def step_states(self) -> tuple[WorkflowStepState, ...]:
        active = self.active_step
        states: list[WorkflowStepState] = []
        for number in range(1, 4):
            if self.error_step == number:
                state: WorkflowStepState = "error"
            elif number == 3 and active == 3 and self.prediction_running:
                state = "running"
            elif number == 3 and active == 3 and self.prediction_succeeded:
                state = "complete"
            elif number < active:
                state = "complete"
            elif number == active:
                state = "active"
            else:
                state = "upcoming"
            states.append(state)
        return tuple(states)


__all__ = ["PredictionWorkflowSnapshot", "WorkflowStepState"]
