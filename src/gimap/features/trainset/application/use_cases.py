"""Trainset framework-neutral use cases。"""

from __future__ import annotations

from pathlib import Path

from .models import GenerateTrainsetRequest
from .ports import DatasetGenerationPort, TrainsetConfigRepository


class GenerateTrainset:
    def __init__(self, generator: DatasetGenerationPort):
        self._generator = generator

    def execute(self, request: GenerateTrainsetRequest, *, on_progress=None, pause=None):
        if request.sample_count <= 0:
            raise ValueError("Trainset sample_count must be positive")
        if request.mode not in {"preview", "demo", "dry", "full"}:
            raise ValueError(f"Unsupported trainset generation mode: {request.mode}")
        return self._generator.generate(
            request, on_progress=on_progress, pause=pause
        )


class LoadTrainsetProject:
    def __init__(self, repository: TrainsetConfigRepository):
        self._repository = repository

    def execute(self, path: Path):
        return self._repository.load(Path(path))


class SaveTrainsetProject:
    def __init__(self, repository: TrainsetConfigRepository):
        self._repository = repository

    def execute(self, config, path: Path):
        return self._repository.save(config, Path(path))
