"""Preview one WAXS batch item through the production preprocessing path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .batch import ProcessWaxsBatch
from .models import (
    IntegrateWaxsImageRequest,
    LoadWaxsImageRequest,
    WaxsBatchPreviewRequest,
    WaxsBatchPreviewResult,
    WaxsBatchRequest,
    WaxsPreprocessedFrame,
)
from .ports import WaxsFileCatalog, WaxsImageRepository
from .use_cases import IntegrateWaxsImage, LoadWaxsImage, PreprocessWaxsFrame


class PreviewWaxsBatchFrame:
    """Preview an expanded file/frame with the same preprocessing used by batch."""

    def __init__(
        self,
        images: WaxsImageRepository,
        catalog: WaxsFileCatalog,
        *,
        preprocess_frame=None,
    ):
        self._images = images
        self._catalog = catalog
        self._load = LoadWaxsImage(images)
        self._integrate = IntegrateWaxsImage()
        self._preprocess = preprocess_frame or PreprocessWaxsFrame(self._integrate)

    def execute(self, request: WaxsBatchPreviewRequest) -> WaxsBatchPreviewResult:
        items = self._expanded_items(request.source)
        if not items:
            raise RuntimeError("No matching WAXS frames found in the selected group.")
        index = max(0, min(int(request.item_index), len(items) - 1))
        factor = None
        if (
            request.batch.normalization_enabled
            and request.batch.normalization_mode == "source_first"
            and index > 0
        ):
            first_path, first_frame = items[0]
            first = self._load.execute(LoadWaxsImageRequest(first_path, first_frame))
            factor = self._process(first.image, request.batch, None).normalization_factor
        path, frame_index = items[index]
        loaded = self._load.execute(LoadWaxsImageRequest(path, frame_index))
        frame = self._process(loaded.image, request.batch, factor)
        return WaxsBatchPreviewResult(
            path=path,
            frame_index=frame_index,
            item_index=index,
            item_count=len(items),
            frame=frame,
        )

    def _process(
        self,
        image: np.ndarray,
        batch: WaxsBatchRequest,
        normalization_factor: float | None,
    ) -> WaxsPreprocessedFrame:
        if batch.calibration_enabled or batch.normalization_enabled:
            return self._preprocess.execute(
                ProcessWaxsBatch._preprocess_request(
                    image,
                    dict(batch.geometry),
                    batch,
                    normalization_factor,
                )
            )
        curve = self._integrate.execute(
            IntegrateWaxsImageRequest(
                image,
                batch.geometry,
                batch.integration,
                batch.mask_min,
                batch.mask_max,
            )
        )
        return WaxsPreprocessedFrame(image, curve, dict(batch.geometry), None)

    def _expanded_items(self, source) -> list[tuple[Path, int]]:
        items: list[tuple[Path, int]] = []
        for path in self._catalog.discover(source.folder, source.pattern):
            count = max(1, int(self._images.frame_count(path)))
            items.extend((path, frame_index) for frame_index in range(count))
        return items
