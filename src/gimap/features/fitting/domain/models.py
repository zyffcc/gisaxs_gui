"""Fitting domain 的显式数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


AxisFilterMode = Literal["all", "positive", "negative"]
CutOrientation = Literal["horizontal", "vertical"]
QUnit = Literal["nm", "angstrom"]


def _one_dimensional(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    return array


@dataclass(frozen=True)
class CurveData:
    q: np.ndarray
    intensity: np.ndarray
    error: np.ndarray | None = None
    q_source_unit: QUnit = "angstrom"
    source_path: str | None = None

    def __post_init__(self) -> None:
        q = _one_dimensional(self.q, "q")
        intensity = _one_dimensional(self.intensity, "intensity")
        if q.size != intensity.size:
            raise ValueError("q and intensity must have the same length")
        error = None if self.error is None else _one_dimensional(self.error, "error")
        if error is not None and error.size != q.size:
            raise ValueError("error and q must have the same length")
        if self.q_source_unit not in ("nm", "angstrom"):
            raise ValueError(f"Unsupported q unit: {self.q_source_unit}")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "error", error)


@dataclass(frozen=True)
class CutSelection:
    center_x: float
    center_y: float
    height: float
    width: float
    orientation: CutOrientation

    def __post_init__(self) -> None:
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(f"Unsupported cut orientation: {self.orientation}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Cut height and width must be greater than zero")


@dataclass(frozen=True)
class CutResult:
    q: np.ndarray
    intensity: np.ndarray
    native_point_count: int
    orientation: CutOrientation

    def __post_init__(self) -> None:
        q = _one_dimensional(self.q, "q")
        intensity = _one_dimensional(self.intensity, "intensity")
        if q.size != intensity.size:
            raise ValueError("Cut q and intensity must have the same length")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True)
class ParameterValue:
    name: str
    value: float
    lower: float | None = None
    upper: float | None = None
    scope: Literal["particle", "global"] = "particle"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter name cannot be empty")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"Invalid bounds for {self.name}: {self.lower} > {self.upper}")


@dataclass(frozen=True)
class FittingParameterSet:
    values: tuple[ParameterValue, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.values)

    def as_array(self) -> np.ndarray:
        return np.asarray([item.value for item in self.values], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {item.name: float(item.value) for item in self.values}
