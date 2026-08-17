"""Manual fitting 的 typed input/output。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import QUnit


@dataclass(frozen=True)
class ManualFitRequest:
    q: np.ndarray
    q_source_unit: QUnit
    shapes: tuple[str, ...]
    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        if q.size == 0:
            raise ValueError("Manual fitting requires q values")
        if not self.shapes:
            raise ValueError("Manual fitting requires at least one active shape")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "parameters", tuple(float(value) for value in self.parameters))


@dataclass(frozen=True)
class ManualFitResult:
    q: np.ndarray
    q_model: np.ndarray
    intensity: np.ndarray
    shapes: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        q_model = np.asarray(self.q_model, dtype=float).reshape(-1)
        intensity = np.asarray(self.intensity, dtype=float).reshape(-1)
        if q.size != intensity.size or q_model.size != intensity.size:
            raise ValueError("Manual fitting q and intensity arrays must have the same length")
        if len(self.parameter_names) != len(self.parameters):
            raise ValueError("Manual fitting parameter names and values must have the same length")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "q_model", q_model)
        object.__setattr__(self, "intensity", intensity)
