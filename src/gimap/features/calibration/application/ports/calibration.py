"""Calibration application 的外部依赖接口。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Protocol

from src.gimap.shared.geometry import InstrumentProfile

from ...domain import CalibrationRequest, CalibrationResult, DetectorImage


ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]
FrameProgress = Callable[[int, int], None]
"""``(frames read, frames to read)`` while the frames of a sum are read."""


class CalibrationImagePort(Protocol):
    def load(
        self,
        path: str | Path,
        dataset_path: str | None = None,
        *,
        frame: int = 0,
        sum_frames: int = 1,
        progress: FrameProgress | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> DetectorImage:
        """Frame ``frame`` (0-based) of the file, or the sum of ``sum_frames`` frames from it on.

        ``metadata`` says what was read: ``frame_index``, ``summed_frames`` and, for a sum,
        ``summed_frame_indices``; ``frame_count`` is the number of frames the file holds.
        """
        ...

    def exists(self, path: str | Path) -> bool: ...


class CalibrationPathPort(Protocol):
    def normalize(self, path: str | Path) -> str: ...


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


class InstrumentProfilePort(Protocol):
    def find(self, name: str) -> Optional[InstrumentProfile]: ...

    def save(self, profile: InstrumentProfile) -> None: ...


class DetectorCatalogPort(Protocol):
    def load(self) -> dict[str, dict[str, Any]]: ...


class CalibrationInputFolderPort(Protocol):
    """The folder Geometry Calibration last read an image from (kept between sessions)."""

    def last_folder(self) -> str: ...

    def remember(self, path: str | Path) -> None: ...
