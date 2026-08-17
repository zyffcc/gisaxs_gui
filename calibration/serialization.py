"""Calibration JSON 旧入口的兼容门面。"""

from __future__ import annotations

from pathlib import Path

from src.gimap.features.calibration.infrastructure.adapters import (
    JsonCalibrationStorageAdapter,
)

from .models import CalibrationResult


_storage = JsonCalibrationStorageAdapter()


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    _storage.save(result, path)


def load_calibration(path: str | Path) -> CalibrationResult:
    return _storage.load(path)
