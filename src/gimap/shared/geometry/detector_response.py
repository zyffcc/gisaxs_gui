"""Detector readout and uncertainty settings carried by an instrument profile."""

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class DetectorResponse:
    counts_per_adu: float = 1.0
    offset_adu: float = 0.0
    error_model: str = "auto"
    sigma_path: Optional[str] = None

    def __post_init__(self):
        if not isfinite(self.counts_per_adu) or self.counts_per_adu <= 0:
            raise ValueError("Counts per ADU must be positive and finite.")
        if not isfinite(self.offset_adu):
            raise ValueError("The detector offset must be finite.")
        if self.error_model not in ("auto", "poisson", "scatter", "sigma"):
            raise ValueError("Unknown detector error model.")
