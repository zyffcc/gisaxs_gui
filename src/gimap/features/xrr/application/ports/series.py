"""Application-owned ports for XRR detector series."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..models import (
    XrrCalibrationGeometry,
    XrrDetectorFrame,
    XrrExtractionProgress,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrFrameRef,
    XrrSeriesLayout,
    XrrSeriesSpec,
)


class XrrSeriesRepository(Protocol):
    def discover(self, spec: XrrSeriesSpec) -> tuple[XrrFrameRef, ...]: ...

    def describe(self, spec: XrrSeriesSpec) -> XrrSeriesLayout | None:
        """What the frames of ``discover`` are stored in. Repositories made before this method lack it:
        it is called through ``getattr(repository, "describe", None)``."""
        ...

    def series(self, spec: XrrSeriesSpec) -> tuple[tuple[XrrFrameRef, ...], XrrSeriesLayout | None]:
        """``discover`` and ``describe`` from one look at the files. Optional like ``describe``."""
        ...

    def load_frame(self, frame: XrrFrameRef) -> XrrDetectorFrame: ...


class XrrExtractionRunnerPort(Protocol):
    def run(
        self,
        request: XrrExtractionRequest,
        *,
        on_progress: Callable[[XrrExtractionProgress], None] | None = None,
    ) -> XrrExtractionResult: ...

    def cancel(self) -> bool: ...


class XrrCurveExportPort(Protocol):
    def export(self, path: Path, result: XrrExtractionResult) -> None: ...


class XrrExportRecordPort(Protocol):
    """Writes the JSON record (settings, geometry, formulas) that goes next to an export."""

    def write(self, path: Path, record: dict[str, Any]) -> None: ...


class XrrGeometryDefaultsPort(Protocol):
    """The geometry of the last applied Geometry Calibration, ``None`` when none was applied."""

    def last_calibration(self) -> XrrCalibrationGeometry | None: ...


class XrrInputFolderPort(Protocol):
    """The folder the XRR window last read data from (kept between sessions)."""

    def last_folder(self) -> str: ...

    def remember(self, path: str | Path) -> None: ...


__all__ = [
    "XrrCurveExportPort",
    "XrrExportRecordPort",
    "XrrExtractionRunnerPort",
    "XrrGeometryDefaultsPort",
    "XrrInputFolderPort",
    "XrrSeriesRepository",
]
