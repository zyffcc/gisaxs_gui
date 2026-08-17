"""Geometry Calibration application use cases。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..domain import CalibrationRequest, CalibrationResult, DetectorImage
from .ports import (
    CalibrationImagePort,
    CalibrationRunnerPort,
    CalibrationStoragePort,
    CancellationCheck,
    DetectorCatalogPort,
    GeometryParametersPort,
    ProgressCallback,
)


@dataclass(frozen=True)
class ImportedCalibration:
    result: CalibrationResult
    image: DetectorImage | None


@dataclass(frozen=True)
class LoadCalibrationImage:
    images: CalibrationImagePort

    def __call__(
        self,
        path: str | Path,
        dataset_path: str | None = None,
    ) -> DetectorImage:
        return self.images.load(path, dataset_path)


@dataclass(frozen=True)
class RunCalibration:
    runner: CalibrationRunnerPort

    def __call__(
        self,
        request: CalibrationRequest,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> CalibrationResult:
        return self.runner.calibrate(request, progress, cancelled)


@dataclass(frozen=True)
class ExportCalibration:
    storage: CalibrationStoragePort

    def __call__(self, result: CalibrationResult, path: str | Path) -> None:
        self.storage.save(result, path)


@dataclass(frozen=True)
class ImportCalibration:
    storage: CalibrationStoragePort
    images: CalibrationImagePort

    def __call__(self, path: str | Path) -> ImportedCalibration:
        result = self.storage.load(path)
        image = self.images.load(result.source_image) if self.images.exists(result.source_image) else None
        return ImportedCalibration(result=result, image=image)


@dataclass(frozen=True)
class ApplyCalibration:
    parameters: GeometryParametersPort

    def current_geometry(self, defaults: dict[str, float]) -> dict[str, float]:
        return self.parameters.current_geometry(defaults)

    def __call__(self, result: CalibrationResult) -> dict[str, float]:
        geometry = self.parameters.apply(result)
        self.parameters.save()
        return geometry


@dataclass(frozen=True)
class LoadDetectorCatalog:
    catalog: DetectorCatalogPort

    def __call__(self) -> dict:
        return self.catalog.load()
