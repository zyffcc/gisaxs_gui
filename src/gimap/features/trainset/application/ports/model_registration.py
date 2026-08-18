"""Trained-model discovery and Prediction module registration port."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import RegisterTrainsetModelRequest, RegisteredTrainsetModel


class TrainsetModelRegistrationPort(Protocol):
    def find_model(self, roots: tuple[Path, ...]) -> Path | None: ...

    def register(self, request: RegisterTrainsetModelRequest) -> RegisteredTrainsetModel: ...
