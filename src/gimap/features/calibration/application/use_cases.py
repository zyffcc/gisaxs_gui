"""Geometry Calibration application use cases。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.gimap.shared.geometry import InstrumentProfile

from ..domain import (
    CalibrationRequest,
    CalibrationResult,
    DetectorImage,
    profile_from_calibration,
    profile_name_for,
)
from .ports import (
    CalibrationImagePort,
    CalibrationPathPort,
    CalibrationRunnerPort,
    CalibrationStoragePort,
    CancellationCheck,
    DetectorCatalogPort,
    GeometryParametersPort,
    InstrumentProfilePort,
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
class NormalizeCalibrationPath:
    paths: CalibrationPathPort

    def __call__(self, path: str | Path) -> str:
        return self.paths.normalize(path)


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
        if image is not None and "image_shape" not in result.metadata:
            # Calibrations saved before the frame shape was recorded.
            result.metadata["image_shape"] = [int(value) for value in image.data.shape[:2]]
        return ImportedCalibration(result=result, image=image)


@dataclass(frozen=True)
class ApplyCalibration:
    parameters: GeometryParametersPort

    def current_geometry(self, defaults: dict[str, float]) -> dict[str, float]:
        """Distance and beam centre (numpy indices) of the last applied calibration."""
        return self.parameters.current_geometry(defaults)

    def __call__(self, result: CalibrationResult) -> dict[str, float]:
        geometry = self.parameters.apply(result)
        self.parameters.save()
        return geometry


@dataclass(frozen=True)
class RecordInstrumentProfile:
    """Store the applied result as the instrument profile of its detector.

    The profile named after the detector and frame size is created or updated,
    so loading the next frame from this detector finds the geometry again.
    """

    profiles: InstrumentProfilePort

    def existing(self, result: CalibrationResult) -> Optional[InstrumentProfile]:
        """The saved profile that recording ``result`` would update, if any."""
        return self.profiles.find(profile_name_for(result))

    def __call__(self, result: CalibrationResult) -> InstrumentProfile:
        profile = profile_from_calibration(result, self.profiles.find(profile_name_for(result)))
        self.profiles.save(profile)
        return profile


@dataclass(frozen=True)
class LoadDetectorCatalog:
    catalog: DetectorCatalogPort

    def __call__(self) -> dict:
        return self.catalog.load()
