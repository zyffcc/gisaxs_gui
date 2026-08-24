"""XRR series inspection, extraction, and export use cases."""

from __future__ import annotations

from collections.abc import Callable
import math

from ..domain import extract_circular_roi, qz_from_theta, specular_pixel
from .models import (
    ExportXrrCurveRequest,
    XrrExtractionProgress,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrPoint,
    XrrSeriesInspection,
    XrrSeriesSpec,
)
from .ports import XrrCurveExportPort, XrrExtractionRunnerPort, XrrSeriesRepository


class InspectXrrSeries:
    def __init__(self, repository: XrrSeriesRepository):
        self._repository = repository

    def execute(self, spec: XrrSeriesSpec) -> XrrSeriesInspection:
        frames = self._repository.discover(spec)
        if not frames:
            raise ValueError("No detector frames were found for the selected XRR series.")
        return XrrSeriesInspection(
            frame_count=len(frames),
            first_ref=frames[0],
            first_frame=self._repository.load_frame(frames[0]),
        )


class ExtractXrrSeries:
    """Load, measure, and release one detector frame at a time."""

    def __init__(self, repository: XrrSeriesRepository):
        self._repository = repository

    def execute(
        self,
        request: XrrExtractionRequest,
        *,
        on_progress: Callable[[XrrExtractionProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> XrrExtractionResult:
        frames = self._repository.discover(request.series)
        if not frames:
            raise ValueError("No detector frames were found for the selected XRR series.")
        points: list[XrrPoint] = []
        total = len(frames)
        for completed, ref in enumerate(frames, start=1):
            if is_cancelled is not None and is_cancelled():
                break
            theta_deg = self._theta_for(request.series, ref)
            detector = self._repository.load_frame(ref)
            roi_x, roi_y = specular_pixel(theta_deg, request.geometry)
            measurement = extract_circular_roi(
                detector.data,
                detector.invalid_mask,
                roi_x,
                roi_y,
                request.extraction.radius_px,
                request.extraction.aggregation,
            )
            point = XrrPoint(
                sequence_index=ref.sequence_index,
                source_name=ref.path.name,
                frame_index=ref.frame_index,
                theta_deg=theta_deg,
                qz_inv_angstrom=qz_from_theta(theta_deg, request.geometry.energy_kev),
                intensity=(
                    measurement.intensity if math.isfinite(measurement.intensity) else None
                ),
                roi_center_x_px=roi_x,
                roi_center_y_px=roi_y,
                valid_pixels=measurement.valid_pixels,
            )
            points.append(point)
            if on_progress is not None:
                on_progress(
                    XrrExtractionProgress(
                        completed=completed,
                        total=total,
                        point=point,
                        preview=detector.data,
                        preview_shape=tuple(int(value) for value in detector.data.shape),
                    )
                )
        return XrrExtractionResult(tuple(points))

    @staticmethod
    def _theta_for(spec: XrrSeriesSpec, ref) -> float:
        if spec.angle_mode == "nxs_dataset":
            if ref.theta_deg is None:
                raise ValueError(
                    "The selected NXS angle dataset did not provide one angle per frame."
                )
            return float(ref.theta_deg)
        return float(spec.theta_start_deg + ref.sequence_index * spec.theta_step_deg)


class RunXrrExtraction:
    def __init__(self, runner: XrrExtractionRunnerPort):
        self._runner = runner

    def execute(self, request: XrrExtractionRequest, *, on_progress=None):
        return self._runner.run(request, on_progress=on_progress)

    def cancel(self) -> bool:
        return self._runner.cancel()


class ExportXrrCurve:
    def __init__(self, exporter: XrrCurveExportPort):
        self._exporter = exporter

    def execute(self, request: ExportXrrCurveRequest) -> None:
        if not request.result.points:
            raise ValueError("There are no extracted XRR points to export.")
        self._exporter.export(request.path, request.result)


__all__ = ["ExportXrrCurve", "ExtractXrrSeries", "InspectXrrSeries", "RunXrrExtraction"]
