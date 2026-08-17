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
