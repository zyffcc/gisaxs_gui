"""Format Converter 需要的文件读取与转换执行 ports。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ...domain.models import ConversionRequest, ConversionResult, InputSource


ProgressCallback = Callable[[int, int, str, int], None]


class SourceRepositoryPort(Protocol):
    def normalize_path(self, path: str | Path) -> str: ...

    def inspect_source(self, path: str | Path) -> InputSource: ...

    def select_dataset(self, source: InputSource, dataset_path: str) -> None: ...

    def scan_folder(
        self,
        folder: str | Path,
        *,
        include_cbf: bool = True,
        include_tiff: bool = True,
        include_nxs: bool = True,
        recursive: bool = False,
    ) -> list[str]: ...

    def estimate_output(self, sources: list[InputSource], request: ConversionRequest) -> tuple[int, int]: ...

    def load_frame(self, source: InputSource, frame_index: int) -> tuple[np.ndarray, dict[str, Any]]: ...


class ConversionExecutorPort(Protocol):
    def execute(
        self,
        request: ConversionRequest,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult: ...

    def cancel(self) -> None: ...

    def set_paused(self, paused: bool) -> None: ...
