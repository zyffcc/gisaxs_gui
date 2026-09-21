"""WAXS application request/result models。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadWaxsImageRequest:
    path: Path
    frame_index: int = 0
    background_path: Path | None = None
    background_coefficient: float = 1.0
    background_frame_index: int = 0


@dataclass(frozen=True)
class LoadedWaxsImage:
    path: Path
    frame_index: int
    frame_count: int
    image: np.ndarray


@dataclass(frozen=True)
class IntegrateWaxsImageRequest:
    image: np.ndarray
    geometry: dict
    integration: dict
    mask_min: float
    mask_max: float
    cut_kind: str = "full"
    selection: dict | None = None


@dataclass(frozen=True)
class WaxsCurve:
    x: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class WaxsBatchSource:
    folder: Path
    pattern: str = "*.tif"
    output_subfolder: str | None = None

    @property
    def resolved_output_subfolder(self) -> str:
        value = (self.output_subfolder or "").strip()
        return value or self.folder.name


@dataclass(frozen=True)
class WaxsBatchRequest:
    folder: Path
    pattern: str
    output_folder: Path
    export_images: bool
    export_curves: bool
    export_background_subtracted: bool
    display: dict[str, Any]
    geometry: dict[str, Any]
    integration: dict[str, Any]
    mask_min: float
    mask_max: float
    timeout_seconds: float | None = None
    continue_on_error: bool = True
    sources: tuple[WaxsBatchSource, ...] = ()
    export_q_images: bool = False
    export_curve_images: bool = False
    q_range: dict[str, float | None] | None = None
    calibration_enabled: bool = False
    calibration_target_q: float = 2.132
    calibration_half_width: float = 0.035
    normalization_enabled: bool = False
    normalization_target_q: float = 2.132
    normalization_half_width: float = 0.035
    normalization_target_intensity: float = 1.0
    normalization_mode: str = "source_first"

    @property
    def batch_sources(self) -> tuple[WaxsBatchSource, ...]:
        if self.sources:
            return self.sources
        return (WaxsBatchSource(self.folder, self.pattern),)


@dataclass(frozen=True)
class WaxsBatchItem:
    path: Path
    frame_index: int
    name: str
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class WaxsBatchProgress:
    completed: int
    total: int
    name: str
    status: str


@dataclass(frozen=True)
class WaxsBatchResult:
    items: tuple[WaxsBatchItem, ...]
    cancelled: bool = False

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.items)


@dataclass(frozen=True)
class WaxsPreprocessFrameRequest:
    image: np.ndarray
    geometry: dict[str, Any]
    integration: dict[str, Any]
    mask_min: float
    mask_max: float
    calibration_enabled: bool = False
    calibration_target_q: float = 2.132
    calibration_half_width: float = 0.035
    normalization_enabled: bool = False
    normalization_target_q: float = 2.132
    normalization_half_width: float = 0.035
    normalization_target_intensity: float = 1.0
    normalization_factor: float | None = None


@dataclass(frozen=True)
class WaxsPreprocessedFrame:
    image: np.ndarray
    curve: WaxsCurve
    geometry: dict[str, Any]
    normalization_factor: float | None = None


@dataclass(frozen=True)
class WaxsBatchPreviewRequest:
    source: WaxsBatchSource
    item_index: int
    batch: WaxsBatchRequest


@dataclass(frozen=True)
class WaxsBatchPreviewResult:
    path: Path
    frame_index: int
    item_index: int
    item_count: int
    frame: WaxsPreprocessedFrame


@dataclass(frozen=True)
class ExportWaxsCurveRequest:
    path: Path
    x: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class ExportWaxsImageRequest:
    path: Path
    image: np.ndarray
    display: dict[str, Any]


@dataclass(frozen=True)
class WaxsQMapRequest:
    shape: tuple[int, int]
    geometry: dict[str, Any]


@dataclass(frozen=True)
class WaxsCutImageRequest:
    image: np.ndarray
    geometry: dict[str, Any]


@dataclass(frozen=True)
class WaxsCutImageResult:
    image: np.ndarray
    extent: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class WaxsDisplayRequest:
    image: np.ndarray
    log_scale: bool
    mask_min: float
    mask_max: float
    flip_vertical: bool = False


@dataclass(frozen=True)
class WaxsDisplayLimitsRequest:
    image: np.ndarray
    log_scale: bool
    mask_min: float
    mask_max: float
    max_samples: int = 200_000
    stride_hint: int = 20
