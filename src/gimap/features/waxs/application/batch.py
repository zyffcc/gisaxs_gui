"""WAXS batch application workflows。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import (
    LoadWaxsImageRequest,
    WaxsBatchItem,
    WaxsBatchProgress,
    WaxsBatchRequest,
    WaxsBatchResult,
)
from .ports import (
    WaxsBatchRunnerPort,
    WaxsExportPort,
    WaxsFileCatalog,
    WaxsImageRepository,
)
from .use_cases import IntegrateWaxsImage, LoadWaxsImage
from .models import IntegrateWaxsImageRequest


class ProcessWaxsBatch:
    """文件/frame 展开、积分、首曲线背景与命名规则。"""

    def __init__(
        self,
        images: WaxsImageRepository,
        catalog: WaxsFileCatalog,
        exporter: WaxsExportPort,
    ):
        self._images = images
        self._catalog = catalog
        self._exporter = exporter
        self._load_image = LoadWaxsImage(images)
        self._integrate = IntegrateWaxsImage()

    def execute(
        self,
        request: WaxsBatchRequest,
        *,
        on_progress=None,
        is_cancelled=None,
        wait_if_paused=None,
    ) -> WaxsBatchResult:
        files = self._catalog.discover(request.folder, request.pattern)
        if not files:
            raise RuntimeError("No matching .nxs, .tif, or .tiff files found.")
        work_items = []
        results = []
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
            work_items.extend((path, index, count) for index in range(count))

        curve_columns = []
        curve_names = []
        background = None
        background_columns = []
        x_axis = None
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
        for path, frame_index, frame_count in work_items:
            if wait_if_paused:
                wait_if_paused()
            if is_cancelled and is_cancelled():
                return WaxsBatchResult(tuple(results), cancelled=True)
            suffix = f"_f{frame_index + 1:04d}" if frame_count > 1 else ""
            name = f"{path.stem}{suffix}"
            try:
                loaded = self._load_image.execute(
                    LoadWaxsImageRequest(path, frame_index)
                )
                if request.export_images:
                    self._exporter.export_image(
                        request.output_folder / "images" / f"{name}.png",
                        loaded.image,
                        request.display,
                    )
                if request.export_curves or request.export_background_subtracted:
                    curve = self._integrate.execute(
                        IntegrateWaxsImageRequest(
                            loaded.image,
                            request.geometry,
                            request.integration,
                            request.mask_min,
                            request.mask_max,
                        )
                    )
                    if x_axis is None:
                        x_axis = curve.x
                        curve_columns.append(curve.x)
                    curve_columns.append(curve.intensity)
                    curve_names.append(name)
                    curve_path = request.output_folder / "1D"
                    if request.export_curves:
                        self._exporter.export_curve(
                            curve_path / f"{name}.csv", curve.x, curve.intensity
                        )
                    if request.export_background_subtracted:
                        if background is None:
                            background = curve.intensity
                        corrected = curve.intensity - background
                        background_columns.append(corrected)
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

        curve_path = request.output_folder / "1D"
        if curve_columns:
            self._exporter.export_matrix(
                curve_path / "output.csv",
                tuple(curve_columns),
                tuple(["x"] + curve_names),
            )
        if background_columns and x_axis is not None:
            self._exporter.export_matrix(
                curve_path / "output_subbg.csv",
                tuple([x_axis] + background_columns),
                tuple(["x"] + curve_names[: len(background_columns)]),
            )
        return WaxsBatchResult(tuple(results))


class RunWaxsBatch:
    def __init__(self, runner: WaxsBatchRunnerPort):
        self._runner = runner

    def execute(self, request: WaxsBatchRequest, *, on_progress=None):
        return self._runner.run(request, on_progress=on_progress)

    def cancel(self) -> bool:
        return self._runner.cancel()

    def set_paused(self, paused: bool) -> bool:
        return self._runner.set_paused(paused)
