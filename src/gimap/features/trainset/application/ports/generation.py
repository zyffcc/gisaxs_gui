"""Trainset generation and project storage ports。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..models import GeneratedTrainset, GenerateTrainsetRequest


class DatasetGenerationPort(Protocol):
    def generate(
        self,
        request: GenerateTrainsetRequest,
        *,
        on_progress=None,
        pause=None,
    ) -> GeneratedTrainset: ...


class TrainsetConfigRepository(Protocol):
    def load(self, path: Path) -> dict[str, Any]: ...

    def save(self, config: dict[str, Any], path: Path) -> Path: ...
