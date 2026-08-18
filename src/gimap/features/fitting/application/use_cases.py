"""Fitting 的文件加载和导出 use cases。"""

from __future__ import annotations

from pathlib import Path

from .errors import FileOperationError
from .models import (
    ExportFitResultRequest,
    ExportOperationResult,
    LoadCurveRequest,
    LoadScatteringFileRequest,
    CurveOperationResult,
    ScatteringOperationResult,
    ScatteringSequenceInfo,
)
from .ports import CurveRepository, FitResultRepository, ScatteringFileRepository
from .ports import FittingModelPort, RemoteFileCachePort
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


class LoadScatteringFile:
    def __init__(self, repository: ScatteringFileRepository):
        self._repository = repository

    def execute(self, request: LoadScatteringFileRequest) -> ScatteringOperationResult:
        try:
            return ScatteringOperationResult(value=self._repository.load(request))
        except Exception as exc:
            return ScatteringOperationResult(
                error=_structured_file_error(request.path, "read", exc)
            )


class InspectScatteringSequence:
    """Read detector navigation metadata through the file repository port."""

    def __init__(self, repository: ScatteringFileRepository):
        self._repository = repository

    def execute(self, path: Path) -> ScatteringSequenceInfo:
        return self._repository.inspect_sequence(Path(path))


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


class ManageRemoteFileCache:
    def __init__(self, cache: RemoteFileCachePort):
        self._cache = cache

    def default_directory(self) -> str:
        return self._cache.default_directory()

    def display_directory(self, cache_dir: str) -> str:
        return self._cache.display_directory(cache_dir)

    def resolve_directory(self, cache_dir: str) -> Path:
        return self._cache.resolve_directory(cache_dir)

    def is_remote(self, path: str) -> bool:
        return self._cache.is_remote(path)

    def target_path(self, source_path: str, cache_dir: str) -> Path:
        return self._cache.target_path(source_path, cache_dir)

    def prepare(self, source_path: str, cache_dir: str, max_gb: float, **callbacks):
        return self._cache.prepare(
            source_path,
            cache_dir,
            max_gb,
            **callbacks,
        )

    def clear(self, cache_dir: str) -> int:
        return self._cache.clear(cache_dir)
