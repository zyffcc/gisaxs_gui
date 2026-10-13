"""Framework-neutral presentation state and commands for the XRR tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..application import (
    ExportXrrCurveRequest,
    ExportedXrrCurve,
    XrrCalibrationGeometry,
    XrrExtractionProgress,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrSeriesInspection,
    XrrSeriesSpec,
    default_export_path,
)
from ..application.ports import XrrInputFolderPort


@dataclass
class XrrViewState:
    inspection: XrrSeriesInspection | None = None
    result: XrrExtractionResult | None = None
    request: XrrExtractionRequest | None = None
    """The extraction that made ``result``: its settings go into the export record."""
    error_message: str = ""
    running: bool = False


class XrrViewModel:
    def __init__(
        self,
        *,
        inspect_series,
        run_extraction,
        export_curve,
        last_calibration=None,
        input_folders: XrrInputFolderPort | None = None,
    ):
        self._inspect_series = inspect_series
        self._run_extraction = run_extraction
        self._export_curve = export_curve
        self._last_calibration = last_calibration
        self._input_folders = input_folders
        self.state = XrrViewState()
        self._progress_points = []

    def inspect(self, spec: XrrSeriesSpec) -> XrrSeriesInspection:
        self.state.error_message = ""
        try:
            inspection = self._inspect_series.execute(spec)
        except Exception as exc:
            self.state.error_message = str(exc)
            raise
        self.state.inspection = inspection
        return inspection

    def extract(self, request: XrrExtractionRequest, *, on_progress=None):
        self.state.error_message = ""
        self.state.running = True
        self.state.request = request
        self._progress_points = []

        def report(progress: XrrExtractionProgress) -> None:
            self._progress_points.append(progress.point)
            self.state.result = XrrExtractionResult(tuple(self._progress_points))
            if on_progress is not None:
                on_progress(progress)

        try:
            result = self._run_extraction.execute(request, on_progress=report)
            if result.points:
                self.state.result = result
            elif self._progress_points:
                result = XrrExtractionResult(tuple(self._progress_points))
                self.state.result = result
            return result
        except Exception as exc:
            self.state.error_message = str(exc)
            raise
        finally:
            self.state.running = False

    def cancel(self) -> bool:
        return self._run_extraction.cancel()

    def export(
        self,
        path,
        *,
        geometry_sources: dict[str, str] | None = None,
        calibration: XrrCalibrationGeometry | None = None,
    ) -> ExportedXrrCurve:
        """The CSV at ``path`` and its JSON record (settings of the extraction) next to it;
        ``calibration``: the applied calibration values marked ``calibration`` came from."""
        if self.state.result is None:
            raise ValueError("There are no extracted XRR points to export.")
        exported = self._export_curve.execute(
            ExportXrrCurveRequest(
                Path(path),
                self.state.result,
                settings=self.state.request,
                geometry_sources=dict(geometry_sources or {}),
                calibration=calibration,
            )
        )
        return exported if exported is not None else ExportedXrrCurve(Path(path))

    def default_export_path(self) -> Path | None:
        """``<source folder>/<source stem>_xrr.csv`` of the extracted series (``None`` before a run)."""
        request = self.state.request
        if request is None:
            return None
        return default_export_path(request.series.source_path)

    def last_calibration(self) -> XrrCalibrationGeometry | None:
        """The geometry of the last applied Geometry Calibration, if one was applied."""
        return self._last_calibration.execute() if self._last_calibration is not None else None

    def last_folder(self) -> str:
        return self._input_folders.last_folder() if self._input_folders is not None else ""

    def remember_folder(self, path) -> None:
        if self._input_folders is not None and path:
            self._input_folders.remember(path)


__all__ = ["XrrViewModel", "XrrViewState"]
