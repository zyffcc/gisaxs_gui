"""Fitting 文件能力的 application ports。"""

from __future__ import annotations

from typing import Protocol

from ..models import (
    ExportFitResultRequest,
    ExportedFitResult,
    LoadCurveRequest,
    LoadScatteringFileRequest,
    DiscoverInSituFramesRequest,
    InSituSourceFrame,
    ScatteringFileData,
    ScatteringSequenceInfo,
)
from ...domain import CurveData


class ScatteringFileRepository(Protocol):
    def load(self, request: LoadScatteringFileRequest) -> ScatteringFileData: ...

    def inspect_sequence(self, path) -> ScatteringSequenceInfo: ...

    def discover_insitu_frames(
        self, request: DiscoverInSituFramesRequest
    ) -> tuple[InSituSourceFrame, ...]: ...


class CurveRepository(Protocol):
    def load(self, request: LoadCurveRequest) -> CurveData: ...


class FitResultRepository(Protocol):
    def export(self, request: ExportFitResultRequest) -> ExportedFitResult: ...
