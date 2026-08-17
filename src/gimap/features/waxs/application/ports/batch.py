"""WAXS batch catalog/export/runner ports。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ..models import WaxsBatchRequest, WaxsBatchResult


class WaxsFileCatalog(Protocol):
    def discover(self, folder: Path, pattern: str) -> tuple[Path, ...]: ...


class WaxsExportPort(Protocol):
    def export_image(self, path: Path, image: np.ndarray, display: dict) -> None: ...

    def export_curve(self, path: Path, x: np.ndarray, y: np.ndarray) -> None: ...

    def export_matrix(
        self, path: Path, columns: tuple[np.ndarray, ...], headers: tuple[str, ...]
    ) -> None: ...


class WaxsBatchRunnerPort(Protocol):
    def run(self, request: WaxsBatchRequest, *, on_progress=None) -> WaxsBatchResult: ...

    def cancel(self) -> bool: ...

    def set_paused(self, paused: bool) -> bool: ...
