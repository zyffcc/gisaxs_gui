"""Fitting file use cases: curves, fit results and in-situ curve series."""

from __future__ import annotations

from pathlib import Path

from .errors import FileOperationError
from .models import (
    ExportCurveFigureRequest,
    ExportFitResultRequest,
    DiscoverInSituFramesRequest,
    ExportOperationResult,
    FigureOperationResult,
    LoadCurveRequest,
    CurveOperationResult,
    InSituSourceFrame,
)
from .ports import CurveFigureWriter, CurveRepository, FitResultRepository, InSituFrameRepository
from .ports import FittingModelPort
from ..domain import ManualFitRequest, ManualFitResult, q_values_for_model


def _structured_file_error(path: Path, operation: str, exc: Exception) -> FileOperationError:
    if isinstance(exc, FileNotFoundError):
        code = "not_found"
    elif isinstance(exc, PermissionError):
        code = "permission_denied"
    elif isinstance(exc, (ValueError, TypeError)):
        text = str(exc).lower()
        code = "unsupported_format" if "unsupported" in text else "invalid_data"
    else:
        code = "write_failed" if operation == "write" else "read_failed"
    return FileOperationError(
        code=code,
        message=str(exc) or type(exc).__name__,
        path=str(path),
        details={"exception_type": type(exc).__name__, "operation": operation},
    )


class DiscoverInSituFrames:
    """List the curve files of an in-situ series, in natural order."""

    def __init__(self, repository: InSituFrameRepository):
        self._repository = repository

    def execute(
        self, request: DiscoverInSituFramesRequest
    ) -> tuple[InSituSourceFrame, ...]:
        return self._repository.discover_insitu_frames(request)


class LoadCurve:
    def __init__(self, repository: CurveRepository):
        self._repository = repository

    def execute(self, request: LoadCurveRequest) -> CurveOperationResult:
        try:
            return CurveOperationResult(value=self._repository.load(request))
        except Exception as exc:
            return CurveOperationResult(error=_structured_file_error(request.path, "read", exc))


class ExportFitResult:
    def __init__(self, repository: FitResultRepository):
        self._repository = repository

    def execute(self, request: ExportFitResultRequest) -> ExportOperationResult:
        try:
            return ExportOperationResult(value=self._repository.export(request))
        except Exception as exc:
            return ExportOperationResult(error=_structured_file_error(request.path, "write", exc))


class ExportCurveFigure:
    """Write the plotted curve layers as a publication figure."""

    def __init__(self, writer: CurveFigureWriter):
        self._writer = writer

    def execute(self, request: ExportCurveFigureRequest) -> FigureOperationResult:
        try:
            if not request.series:
                raise ValueError("Nothing is plotted yet: load a curve first")
            return FigureOperationResult(value=self._writer.write(request))
        except Exception as exc:
            return FigureOperationResult(error=_structured_file_error(request.path, "write", exc))


class RunManualFit:
    def __init__(self, model: FittingModelPort):
        self._model = model

    def execute(self, request: ManualFitRequest) -> ManualFitResult:
        q_model = q_values_for_model(request.q, request.q_source_unit)
        parameter_names = self._model.parameter_names(request.shapes)
        if len(parameter_names) != len(request.parameters):
            raise ValueError(
                "Manual fitting parameter count does not match the selected model"
            )
        intensity = self._model.evaluate(request.shapes, q_model, request.parameters)
        return ManualFitResult(
            q=request.q,
            q_model=q_model,
            intensity=intensity,
            shapes=request.shapes,
            parameter_names=parameter_names,
            parameters=request.parameters,
        )
