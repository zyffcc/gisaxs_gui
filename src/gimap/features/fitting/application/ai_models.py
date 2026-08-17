"""AI fitting application requests/results。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateGenerationRequest:
    model_path: Path
    output_dir: Path
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    profile: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    exact_nonempty: int | None = None
    allow_unsafe_lambda: bool = True
    clear_output_dir: bool = False

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        intensity = np.asarray(self.intensity, dtype=float).reshape(-1)
        sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        if not (q.size == intensity.size == sigma.size) or q.size < 1:
            raise ValueError("AI fitting q, intensity and sigma must have the same non-zero length")
        json.dumps(self.profile, allow_nan=False)
        json.dumps(self.constraints, allow_nan=False)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True)
class CandidateGenerationResult:
    output_dir: Path
    profile_name: str
    runtime_seconds: float
    configured_candidates: int
    candidates: tuple[dict[str, Any], ...]
    best_log_rmse: float | None
    exit_code: int


class CandidateJobError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(message)
