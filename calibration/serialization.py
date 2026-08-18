"""Legacy import path for feature-owned calibration JSON serialization."""

from src.gimap.features.calibration.infrastructure.adapters.serialization import (
    load_calibration,
    save_calibration,
)

__all__ = ["load_calibration", "save_calibration"]
