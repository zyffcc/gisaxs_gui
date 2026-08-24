"""Application-owned ports for XRR detector series."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..models import (
    XrrDetectorFrame,
    XrrExtractionProgress,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrFrameRef,
    XrrSeriesSpec,
)


class XrrSeriesRepository(Protocol):
    def discover(self, spec: XrrSeriesSpec) -> tuple[XrrFrameRef, ...]: ...

    def load_frame(self, frame: XrrFrameRef) -> XrrDetectorFrame: ...


class XrrExtractionRunnerPort(Protocol):
    def run(
        self,
        request: XrrExtractionRequest,
        *,
        on_progress: Callable[[XrrExtractionProgress], None] | None = None,
    ) -> XrrExtractionResult: ...

    def cancel(self) -> bool: ...


class XrrCurveExportPort(Protocol):
    def export(self, path: Path, result: XrrExtractionResult) -> None: ...


__all__ = ["XrrCurveExportPort", "XrrExtractionRunnerPort", "XrrSeriesRepository"]
