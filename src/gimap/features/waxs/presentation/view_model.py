"""WAXS ViewModel；不依赖 Qt、filesystem 或 process implementation。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..application import (
    ExportWaxsCurveRequest,
    ExportWaxsImageRequest,
    IntegrateWaxsImageRequest,
    LoadWaxsImageRequest,
    WaxsCutImageRequest,
    WaxsDisplayLimitsRequest,
    WaxsDisplayRequest,
    WaxsQMapRequest,
)
from .state import WaxsState


class WaxsViewModel:
    def __init__(
        self,
        *,
        load_image,
        integrate_image,
        run_batch,
        export_curve,
        export_image,
        compute_q_maps,
        cut_image,
        prepare_display,
        estimate_display_limits,
    ):
        self._load_image = load_image
        self._integrate_image = integrate_image
        self._run_batch = run_batch
        self._export_curve = export_curve
        self._export_image = export_image
        self._compute_q_maps = compute_q_maps
        self._cut_image = cut_image
        self._prepare_display = prepare_display
        self._estimate_display_limits = estimate_display_limits
        self.state = WaxsState()

    def load_image(self, path: Path, frame_index: int = 0):
        self.state = replace(self.state, image_status="loading", error_message=None)
        try:
            loaded = self._load_image.execute(
                LoadWaxsImageRequest(Path(path), frame_index)
            )
        except Exception as exc:
            self.state = replace(
                self.state,
                image_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            image_status="ready",
            current_image=loaded,
            error_message=None,
            status_message=f"Loaded {loaded.path.name}",
        )
        return loaded

    def integrate(self, request: IntegrateWaxsImageRequest):
        self.state = replace(
            self.state, integration_status="running", error_message=None
        )
        try:
            curve = self._integrate_image.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state, integration_status="error", error_message=str(exc)
            )
            return None
        self.state = replace(
            self.state,
            integration_status="ready",
            current_curve=curve,
            error_message=None,
        )
        return curve

    def run_batch(self, request, *, on_progress=None):
        self.state = replace(
            self.state, batch_status="running", progress=0.0, error_message=None
        )

        def progress(value):
            fraction = value.completed / value.total if value.total else 0.0
            self.state = replace(
                self.state,
                progress=fraction,
                status_message=f"Processed {value.name}",
            )
            if on_progress:
                on_progress(value)

        try:
            result = self._run_batch.execute(request, on_progress=progress)
        except Exception as exc:
            self.state = replace(
                self.state, batch_status="error", error_message=str(exc)
            )
            return None
        self.state = replace(
            self.state,
            batch_status="cancelled" if result.cancelled else "ready",
            batch_result=result,
            progress=1.0 if not result.cancelled else self.state.progress,
            error_message=None,
        )
        return result

    def cancel_batch(self) -> bool:
        return self._run_batch.cancel()

    def set_batch_paused(self, paused: bool) -> bool:
        return self._run_batch.set_paused(paused)

    def export_curve(self, path: Path, curve=None):
        selected = curve or self.state.current_curve
        if selected is None:
            return None
        try:
            return self._export_curve.execute(
                ExportWaxsCurveRequest(Path(path), selected.x, selected.intensity)
            )
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc))
            return None

    def export_image(self, path: Path, image, display):
        try:
            return self._export_image.execute(
                ExportWaxsImageRequest(Path(path), image, dict(display))
            )
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc))
            return None

    def compute_q_maps(self, shape, geometry):
        return self._compute_q_maps.execute(
            WaxsQMapRequest(tuple(shape[:2]), dict(geometry))
        )

    def cut_image(self, image, geometry):
        return self._cut_image.execute(
            WaxsCutImageRequest(image, dict(geometry))
        )

    def prepare_display(
        self,
        image,
        *,
        log_scale,
        mask_min,
        mask_max,
        flip_vertical=False,
    ):
        return self._prepare_display.execute(
            WaxsDisplayRequest(
                image,
                bool(log_scale),
                float(mask_min),
                float(mask_max),
                bool(flip_vertical),
            )
        )

    def estimate_display_limits(
        self,
        image,
        *,
        log_scale,
        mask_min,
        mask_max,
    ):
        return self._estimate_display_limits.execute(
            WaxsDisplayLimitsRequest(
                image,
                bool(log_scale),
                float(mask_min),
                float(mask_max),
            )
        )
