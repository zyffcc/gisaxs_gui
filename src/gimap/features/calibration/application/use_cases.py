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
from .errors import AmbiguousImageDatasetError
from .ports import (
    CalibrationImagePort,
    CalibrationPathPort,
    CalibrationRunnerPort,
    CalibrationStoragePort,
    CancellationCheck,
    DetectorCatalogPort,
    FrameProgress,
    GeometryParametersPort,
    InstrumentProfilePort,
    ProgressCallback,
)


@dataclass(frozen=True)
class ImportedCalibration:
    result: CalibrationResult
    image: DetectorImage | None


def frames_made_on(result: CalibrationResult) -> tuple[int, int]:
    """``(first frame (0-based), frames summed)`` of the image ``result`` was made on, as its record
    says; ``(0, 1)`` for a record that does not say (one frame, or saved before frames were recorded)."""
    loader = result.metadata.get("loader") if isinstance(result.metadata, dict) else None
    loader = loader if isinstance(loader, dict) else {}

    def number(key: str, least: int) -> int:
        try:
            return max(least, int(loader.get(key) or least))
        except (TypeError, ValueError):
            return least

    return number("frame_index", 0), number("summed_frames", 1)


@dataclass(frozen=True)
class LoadCalibrationImage:
    images: CalibrationImagePort

    def __call__(
        self,
        path: str | Path,
        dataset_path: str | None = None,
        *,
        frame: int = 0,
        sum_frames: int = 1,
        progress: FrameProgress | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> DetectorImage:
        """The first frame of the file; ``frame`` (0-based) / ``sum_frames`` choose another one or a sum."""
        if not frame and sum_frames <= 1:
            return self.images.load(path, dataset_path)
        return self.images.load(
            path, dataset_path, frame=int(frame), sum_frames=int(sum_frames), progress=progress, cancelled=cancelled
        )


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

    def __call__(self, path: str | Path, *, summed: bool = False) -> ImportedCalibration:
        """The saved result and, when its file is still there, the image it was made on. For a calibration
        made on summed frames that is the first of them unless ``summed`` (see ``source_image``)."""
        result = self.storage.load(path)
        image = self.source_image(result, summed=summed) if self.images.exists(result.source_image) else None
        if image is not None and "image_shape" not in result.metadata:
            # Calibrations saved before the frame shape was recorded.
            result.metadata["image_shape"] = [int(value) for value in image.data.shape[:2]]
        return ImportedCalibration(result=result, image=image)

    def source_image(
        self,
        result: CalibrationResult,
        *,
        summed: bool = False,
        progress: FrameProgress | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> DetectorImage:
        """The image the calibration was made on: the frame its record names, and of a file with several
        image datasets the one that was read. Never a question, and a file that changed since shows its
        first frame.

        A calibration made on a sum of frames (``frames_made_on``): only the first of them is read (quick,
        for a window that must not wait) unless ``summed``; then all of them are read and added, which
        takes as long as reading them took (``progress`` and ``cancelled`` as in ``LoadCalibrationImage``).
        """
        loader = result.metadata.get("loader") if isinstance(result.metadata, dict) else None
        loader = loader if isinstance(loader, dict) else {}
        source = result.source_image
        first, count = frames_made_on(result)
        frames: dict = {"frame": first} if first else {}  # no frame named: the call ports always had
        if summed and count > 1:
            frames = {"frame": first, "sum_frames": count, "progress": progress, "cancelled": cancelled}
        try:
            try:
                return self.images.load(source, **frames)
            except AmbiguousImageDatasetError as asked:
                recorded = loader.get("dataset_path")
                dataset = recorded if recorded in asked.paths else asked.paths[0]
                return self.images.load(source, dataset, **frames)
        except IndexError:
            return self.images.load(source)


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
