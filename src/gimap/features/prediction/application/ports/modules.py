"""Prediction module storage port。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...domain import PredictionModule


class ModuleRepository(Protocol):
    def discover(self) -> tuple[PredictionModule, ...]: ...

    def load(self, yaml_path: Path) -> PredictionModule: ...

    def update_model_path(self, module: PredictionModule, model_path: Path) -> None: ...
