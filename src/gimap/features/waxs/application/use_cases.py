"""WAXS application use cases。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import (
    IntegrateWaxsImageRequest,
    ExportWaxsCurveRequest,
    ExportWaxsImageRequest,
    LoadedWaxsImage,
    LoadWaxsImageRequest,
    WaxsCutImageRequest,
    WaxsCutImageResult,
    WaxsCurve,
    WaxsDisplayLimitsRequest,
    WaxsDisplayRequest,
    WaxsQMapRequest,
)
from .ports import WaxsExportPort, WaxsImageRepository, WaxsPathPort
from ..domain import (
    circle_cut_profile,
    compute_q_maps,
    cut_image_by_q_range,
    estimate_display_limits,
    integrate_image,
    line_cut_profile,
    prepare_display_array,
    smooth_curve,
)


@dataclass(frozen=True)
class NormalizeWaxsPath:
    paths: WaxsPathPort

    def __call__(self, path: str | Path) -> str:
        return self.paths.normalize(path)


@dataclass(frozen=True)
class GetWaxsWorkingDirectory:
    paths: WaxsPathPort

    def __call__(self) -> str:
        return self.paths.current_directory()


@dataclass(frozen=True)
class ValidateWaxsDirectory:
    paths: WaxsPathPort

    def __call__(self, path: str | Path) -> bool:
        return self.paths.is_directory(path)


class LoadWaxsImage:
    def __init__(self, repository: WaxsImageRepository):
        self._repository = repository

    def execute(self, request: LoadWaxsImageRequest) -> LoadedWaxsImage:
        path = Path(request.path)
        frame_count = max(1, int(self._repository.frame_count(path)))
        frame_index = max(0, min(int(request.frame_index), frame_count - 1))
        image = np.asarray(
            self._repository.load_frame(path, frame_index), dtype=np.float32
        )
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D WAXS image, got shape {image.shape}")
        return LoadedWaxsImage(path, frame_index, frame_count, image)


class IntegrateWaxsImage:
    def execute(self, request: IntegrateWaxsImageRequest) -> WaxsCurve:
        selection = request.selection or {}
        if request.cut_kind == "line":
            x, intensity = line_cut_profile(
                request.image,
                float(selection["center_x"]),
                float(selection["center_y"]),
                float(selection["width"]),
                float(selection["height"]),
                request.mask_min,
                request.mask_max,
            )
        elif request.cut_kind == "circle":
            x, intensity = circle_cut_profile(
                request.image,
                float(selection["center_x"]),
                float(selection["center_y"]),
                float(selection["inner_radius"]),
                float(selection["outer_radius"]),
                float(selection["start_angle"]),
                float(selection["end_angle"]),
                int(request.integration.get("bins", 500)),
                mode=str(request.integration.get("mode", "radial")),
                mask_min=request.mask_min,
                mask_max=request.mask_max,
            )
        else:
            x, intensity = integrate_image(
                request.image,
                request.geometry,
                request.integration,
                request.mask_min,
                request.mask_max,
            )
        if request.integration.get("smooth", False):
            intensity = smooth_curve(
                intensity, int(request.integration.get("smooth_window", 7))
            )
        return WaxsCurve(x, intensity)


class ExportWaxsCurve:
    def __init__(self, exporter: WaxsExportPort):
        self._exporter = exporter

    def execute(self, request: ExportWaxsCurveRequest):
        self._exporter.export_curve(
            Path(request.path), request.x, request.intensity
        )
        return Path(request.path)


class ExportWaxsImage:
    def __init__(self, exporter: WaxsExportPort):
        self._exporter = exporter

    def execute(self, request: ExportWaxsImageRequest):
        self._exporter.export_image(Path(request.path), request.image, request.display)
        return Path(request.path)


class ComputeWaxsQMaps:
    def execute(self, request: WaxsQMapRequest) -> tuple[np.ndarray, np.ndarray]:
        return compute_q_maps(request.shape, request.geometry)


class CutWaxsImage:
    def execute(self, request: WaxsCutImageRequest) -> WaxsCutImageResult:
        image, extent = cut_image_by_q_range(request.image, request.geometry)
        return WaxsCutImageResult(image, extent)


class PrepareWaxsDisplay:
    def execute(self, request: WaxsDisplayRequest) -> np.ndarray:
        return prepare_display_array(
            request.image,
            log_scale=request.log_scale,
            mask_min=request.mask_min,
            mask_max=request.mask_max,
            flip_vertical=request.flip_vertical,
        )


class EstimateWaxsDisplayLimits:
    def execute(
        self, request: WaxsDisplayLimitsRequest
    ) -> tuple[float, float] | None:
        return estimate_display_limits(
            request.image,
            log_scale=request.log_scale,
            mask_min=request.mask_min,
            mask_max=request.mask_max,
            max_samples=request.max_samples,
            stride_hint=request.stride_hint,
        )
