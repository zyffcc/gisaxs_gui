"""Application ports of the Fitting file capabilities."""

from __future__ import annotations

from typing import Protocol

from pathlib import Path

from ..models import (
    ExportCurveFigureRequest,
    ExportFitResultRequest,
    ExportedFitResult,
    LoadCurveRequest,
    DiscoverInSituFramesRequest,
    InSituSourceFrame,
)
from ...domain import CurveData


class InSituFrameRepository(Protocol):
    def discover_insitu_frames(
        self, request: DiscoverInSituFramesRequest
    ) -> tuple[InSituSourceFrame, ...]: ...


class CurveRepository(Protocol):
    def load(self, request: LoadCurveRequest) -> CurveData: ...


class FitResultRepository(Protocol):
    def export(self, request: ExportFitResultRequest) -> ExportedFitResult: ...


class CurveFigureWriter(Protocol):
    def write(self, request: ExportCurveFigureRequest) -> Path: ...
