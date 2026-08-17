"""Calibration application 的外部依赖接口。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ...domain import CalibrationRequest, CalibrationResult, DetectorImage


ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]


class CalibrationImagePort(Protocol):
    def load(
        self,
        path: str | Path,
        dataset_path: str | None = None,
    ) -> DetectorImage: ...

    def exists(self, path: str | Path) -> bool: ...


class CalibrationRunnerPort(Protocol):
    def calibrate(
        self,
        request: CalibrationRequest,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> CalibrationResult: ...


class CalibrationStoragePort(Protocol):
    def save(self, result: CalibrationResult, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> CalibrationResult: ...


class GeometryParametersPort(Protocol):
    def current_geometry(self, defaults: dict[str, float]) -> dict[str, float]: ...

    def apply(self, result: CalibrationResult) -> dict[str, float]: ...

    def save(self) -> None: ...


class DetectorCatalogPort(Protocol):
    def load(self) -> dict[str, dict[str, Any]]: ...
