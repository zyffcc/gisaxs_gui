"""WAXS typed presentation state。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..application import LoadedWaxsImage, WaxsBatchResult, WaxsCurve


Status = Literal["idle", "loading", "running", "ready", "cancelled", "error"]


@dataclass(frozen=True)
class WaxsState:
    image_status: Status = "idle"
    current_image: LoadedWaxsImage | None = None
    integration_status: Status = "idle"
    current_curve: WaxsCurve | None = None
    batch_status: Status = "idle"
    batch_result: WaxsBatchResult | None = None
    progress: float = 0.0
    status_message: str = "Ready"
    error_message: str | None = None
