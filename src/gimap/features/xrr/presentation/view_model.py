"""Framework-neutral presentation state and commands for the XRR tool."""

from __future__ import annotations

from dataclasses import dataclass

from ..application import (
    ExportXrrCurveRequest,
    XrrExtractionProgress,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrSeriesInspection,
    XrrSeriesSpec,
)


@dataclass
class XrrViewState:
    inspection: XrrSeriesInspection | None = None
    result: XrrExtractionResult | None = None
    error_message: str = ""
    running: bool = False


class XrrViewModel:
    def __init__(self, *, inspect_series, run_extraction, export_curve):
        self._inspect_series = inspect_series
        self._run_extraction = run_extraction
        self._export_curve = export_curve
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

    def export(self, path) -> None:
        if self.state.result is None:
            raise ValueError("There are no extracted XRR points to export.")
        self._export_curve.execute(ExportXrrCurveRequest(path, self.state.result))


__all__ = ["XrrViewModel", "XrrViewState"]
