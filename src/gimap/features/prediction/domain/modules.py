"""Prediction module 的模型无关结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    format: str = ""
    path: str = ""


@dataclass(frozen=True)
class PreprocessSpec:
    entry: str = ""
    steps: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputSpec:
    type: str = ""
    names: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()
    target_min: tuple[float, ...] = ()
    target_max: tuple[float, ...] = ()


@dataclass(frozen=True)
class PredictionModule:
    id: str
    name: str
    framework: str = ""
    version: str = ""
    folder: Path | None = None
    yaml_path: Path | None = None
    model: ModelSpec = field(default_factory=ModelSpec)
    preprocess: PreprocessSpec = field(default_factory=PreprocessSpec)
    input_type: str = "cbf"
    stack_axis: int = 0
    input_shape: tuple[int, ...] | None = None
    outputs: OutputSpec = field(default_factory=OutputSpec)

    def __post_init__(self) -> None:
        if not self.id and not self.name:
            raise ValueError("Prediction module requires an id or name")
