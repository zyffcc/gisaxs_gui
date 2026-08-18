"""Trainset presentation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Status = Literal["idle", "running", "ready", "error"]


@dataclass(frozen=True)
class TrainsetState:
    preview_status: Status = "idle"
    what_if_status: Status = "idle"
    error_message: str | None = None
    status_message: str = "Ready"
