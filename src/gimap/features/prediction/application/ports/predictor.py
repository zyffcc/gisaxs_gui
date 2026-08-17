"""模型 runtime 的 application port。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...domain import ModelRuntimeInfo, PredictionRequest, PredictionResult


class Predictor(Protocol):
    def inspect(self, model_path: Path, allow_unsafe_lambda: bool = False) -> ModelRuntimeInfo: ...

    def predict(self, request: PredictionRequest) -> PredictionResult: ...
