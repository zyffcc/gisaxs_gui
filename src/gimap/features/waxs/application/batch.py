"""WAXS batch application workflows。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import (
    IntegrateWaxsImageRequest,
    LoadWaxsImageRequest,
    WaxsBatchItem,
    WaxsBatchProgress,
    WaxsBatchRequest,
    WaxsBatchResult,
    WaxsCurve,
    WaxsPreprocessFrameRequest,
)
from .ports import (
    WaxsBatchRunnerPort,
    WaxsExportPort,
    WaxsFileCatalog,
    WaxsImageRepository,
)
from .use_cases import IntegrateWaxsImage, LoadWaxsImage, PreprocessWaxsFrame


class ProcessWaxsBatch:
    """文件/frame 展开、积分、首曲线背景与命名规则。"""

    def __init__(
        self,
        images: WaxsImageRepository,
        catalog: WaxsFileCatalog,
        exporter: WaxsExportPort,
        *,
        preprocess_frame=None,
    ):
        self._images = images
        self._catalog = catalog
        self._exporter = exporter
        self._load_image = LoadWaxsImage(images)
        self._integrate = IntegrateWaxsImage()
        self._preprocess = preprocess_frame or PreprocessWaxsFrame(self._integrate)

    def execute(
        self,
        request: WaxsBatchRequest,
        *,
        on_progress=None,
        is_cancelled=None,
        wait_if_paused=None,
    ) -> WaxsBatchResult:
        work_items = []
        results = []
        source_states = {}
        for source in request.batch_sources:
            files = self._catalog.discover(source.folder, source.pattern)
            if not files:
                results.append(
                    WaxsBatchItem(
                        source.folder,
                        0,
                        source.folder.name,
                        "failed",
                        "No matching .nxs, .tif, or .tiff files found.",
                    )
                )
                if not request.continue_on_error:
                    return WaxsBatchResult(tuple(results))
                continue
            source_output = request.output_folder
            if request.sources:
                output_name = source.resolved_output_subfolder
                if Path(output_name).name != output_name or output_name in {".", ".."}:
                    raise ValueError(
                        "Each batch output subfolder must be a simple folder name."
                    )
                source_output = source_output / output_name
            source_key = str(source_output).casefold()
            if source_key in source_states:
                raise ValueError(
                    f"Batch output subfolder is duplicated: {source_output.name}"
                )
            source_states[source_key] = {
                "output": source_output,
                "curve_columns": [],
                "curve_names": [],
                "background": None,
                "background_columns": [],
                "x_axis": None,
                "normalization_factor": None,
            }
            for path in files:
                try:
                    count = max(1, int(self._images.frame_count(path)))
                except Exception as exc:
                    results.append(
                        WaxsBatchItem(path, 0, path.stem, "failed", str(exc))
                    )
                    if not request.continue_on_error:
                        return WaxsBatchResult(tuple(results))
                    continue
                work_items.extend(
                    (source_key, path, index, count) for index in range(count)
                )

        total = len(work_items) + len(results)
        completed = len(results)
        if on_progress:
            for initial_completed, item in enumerate(results, start=1):
                on_progress(
                    WaxsBatchProgress(
                        initial_completed,
                        total,
                        item.name,
                        item.status,
                    )
                )
        for source_key, path, frame_index, frame_count in work_items:
            if wait_if_paused:
                wait_if_paused()
            if is_cancelled and is_cancelled():
                return WaxsBatchResult(tuple(results), cancelled=True)
            suffix = f"_f{frame_index + 1:04d}" if frame_count > 1 else ""
            name = f"{path.stem}{suffix}"
            source_state = source_states[source_key]
            output_folder = source_state["output"]
            try:
                loaded = self._load_image.execute(
                    LoadWaxsImageRequest(path, frame_index)
                )
                image = loaded.image
                geometry = dict(request.geometry)
                curve = None
                needs_curve = bool(
                    request.export_curves
                    or request.export_curve_images
                    or request.export_background_subtracted
                    or request.calibration_enabled
                    or request.normalization_enabled
                )
                if needs_curve and not (
                    request.calibration_enabled or request.normalization_enabled
                ):
                    curve = self._integrate_curve(image, geometry, request)
                if request.calibration_enabled or request.normalization_enabled:
                    if request.normalization_mode not in {"source_first", "per_frame"}:
                        raise ValueError(
                            "Normalization mode must be 'source_first' or 'per_frame'."
                        )
                    reused_factor = (
                        source_state["normalization_factor"]
                        if request.normalization_mode == "source_first"
                        else None
                    )
                    processed = self._preprocess.execute(
                        self._preprocess_request(
                            image, geometry, request, reused_factor
                        )
                    )
                    image = processed.image
                    geometry = processed.geometry
                    curve = processed.curve
                    if (
                        request.normalization_enabled
                        and request.normalization_mode == "source_first"
                        and source_state["normalization_factor"] is None
                    ):
                        source_state["normalization_factor"] = (
                            processed.normalization_factor
                        )

                if request.export_images:
                    self._exporter.export_image(
                        output_folder / "2D_pixel" / f"{name}.png",
                        image,
                        {**request.display, "coordinate_mode": "pixel"},
                    )
                if request.export_q_images:
                    self._exporter.export_image(
                        output_folder / "2D_q" / f"{name}.png",
                        image,
                        {
                            **request.display,
                            "coordinate_mode": "q",
                            "geometry": geometry,
                            "q_range": request.q_range,
                        },
                    )
                if needs_curve:
                    if curve is None:
                        curve = self._integrate_curve(image, geometry, request)
                    if request.export_curves or request.export_background_subtracted:
                        if source_state["x_axis"] is None:
                            source_state["x_axis"] = curve.x
                            source_state["curve_columns"].append(curve.x)
                        source_state["curve_columns"].append(curve.intensity)
                        source_state["curve_names"].append(name)
                    curve_path = output_folder / "1D"
                    if request.export_curves:
                        self._exporter.export_curve(
                            curve_path / f"{name}.csv", curve.x, curve.intensity
                        )
                    if request.export_curve_images:
                        self._exporter.export_curve_image(
                            curve_path / f"{name}.png",
                            curve.x,
                            curve.intensity,
                            {
                                **request.display,
                                "x_label": self._curve_x_label(request.integration),
                                "curve_log_scale": bool(
                                    request.display.get("log_scale", False)
                                ),
                            },
                        )
                    if request.export_background_subtracted:
                        if source_state["background"] is None:
                            source_state["background"] = curve.intensity
                        corrected = curve.intensity - source_state["background"]
                        source_state["background_columns"].append(corrected)
                        self._exporter.export_curve(
                            curve_path / f"{name}_subbg.csv", curve.x, corrected
                        )
                item = WaxsBatchItem(path, frame_index, name, "succeeded")
            except Exception as exc:
                item = WaxsBatchItem(path, frame_index, name, "failed", str(exc))
            results.append(item)
            completed += 1
            if on_progress:
                on_progress(
                    WaxsBatchProgress(
                        completed, total, name, item.status
                    )
                )
            if item.status == "failed" and not request.continue_on_error:
                break

        for source_state in source_states.values():
            curve_path = source_state["output"] / "1D"
            if source_state["curve_columns"]:
                self._exporter.export_matrix(
                    curve_path / "output.csv",
                    tuple(source_state["curve_columns"]),
                    tuple(["x"] + source_state["curve_names"]),
                )
            if (
                source_state["background_columns"]
                and source_state["x_axis"] is not None
            ):
                self._exporter.export_matrix(
                    curve_path / "output_subbg.csv",
                    tuple(
                        [source_state["x_axis"]]
                        + source_state["background_columns"]
                    ),
                    tuple(
                        ["x"]
                        + source_state["curve_names"][
                            : len(source_state["background_columns"])
                        ]
                    ),
                )
        return WaxsBatchResult(tuple(results))

    def _integrate_curve(
        self,
        image: np.ndarray,
        geometry: dict,
        request: WaxsBatchRequest,
    ) -> WaxsCurve:
        return self._integrate.execute(
            IntegrateWaxsImageRequest(
                image,
                geometry,
                request.integration,
                request.mask_min,
                request.mask_max,
            )
        )

    @staticmethod
    def _preprocess_request(
        image: np.ndarray,
        geometry: dict,
        request: WaxsBatchRequest,
        normalization_factor: float | None,
    ) -> WaxsPreprocessFrameRequest:
        return WaxsPreprocessFrameRequest(
            image=image,
            geometry=geometry,
            integration=request.integration,
            mask_min=request.mask_min,
            mask_max=request.mask_max,
            calibration_enabled=request.calibration_enabled,
            calibration_target_q=request.calibration_target_q,
            calibration_half_width=request.calibration_half_width,
            normalization_enabled=request.normalization_enabled,
            normalization_target_q=request.normalization_target_q,
            normalization_half_width=request.normalization_half_width,
            normalization_target_intensity=request.normalization_target_intensity,
            normalization_factor=normalization_factor,
        )

    @staticmethod
    def _curve_x_label(integration: dict) -> str:
        mode = str(integration.get("x_axis", "q")).lower()
        if mode == "pixel":
            return "Radius (pixel)"
        if mode == "2theta":
            return "2θ (°)"
        return "q (Å⁻¹)"


class RunWaxsBatch:
    def __init__(self, runner: WaxsBatchRunnerPort):
        self._runner = runner

    def execute(self, request: WaxsBatchRequest, *, on_progress=None):
        return self._runner.run(request, on_progress=on_progress)

    def cancel(self) -> bool:
        return self._runner.cancel()

    def set_paused(self, paused: bool) -> bool:
        return self._runner.set_paused(paused)
