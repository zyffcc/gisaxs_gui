"""Prediction feature 的模型无关数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRuntimeInfo:
    artifact_path: Path
    runtime_name: str
    runtime_version: str = ""
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    input_shape: tuple[Any, ...] | None = None
    output_shape: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class PredictionRequest:
    model_path: Path
    inputs: Any
    allow_unsafe_lambda: bool = False
    precision_policy: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class PredictionResult:
    outputs: Any
    runtime: ModelRuntimeInfo | None = None
