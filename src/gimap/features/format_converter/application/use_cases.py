"""Format Converter application use cases。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..domain.models import ConversionRequest, ConversionResult, InputSource
from ..domain.rules import validate_options
from .ports import ConversionExecutorPort, ProgressCallback, SourceRepositoryPort


@dataclass(frozen=True)
class InspectSource:
    repository: SourceRepositoryPort

    def __call__(self, path: str | Path) -> InputSource:
        return self.repository.inspect_source(path)


@dataclass(frozen=True)
class NormalizePath:
    repository: SourceRepositoryPort

    def __call__(self, path: str | Path) -> str:
        return self.repository.normalize_path(path)


@dataclass(frozen=True)
class SelectDataset:
    repository: SourceRepositoryPort

    def __call__(self, source: InputSource, dataset_path: str) -> None:
        self.repository.select_dataset(source, dataset_path)


@dataclass(frozen=True)
class ScanFolder:
    repository: SourceRepositoryPort

    def __call__(
        self,
        folder: str | Path,
        *,
        include_cbf: bool = True,
        include_tiff: bool = True,
        include_nxs: bool = True,
        recursive: bool = False,
    ) -> list[str]:
        return self.repository.scan_folder(
            folder,
            include_cbf=include_cbf,
            include_tiff=include_tiff,
            include_nxs=include_nxs,
            recursive=recursive,
        )


@dataclass(frozen=True)
class EstimateOutput:
    repository: SourceRepositoryPort

    def __call__(self, request: ConversionRequest) -> tuple[int, int]:
        return self.repository.estimate_output(list(request.sources), request)


@dataclass(frozen=True)
class LoadPreview:
    repository: SourceRepositoryPort

    def __call__(self, source: InputSource, frame_index: int):
        return self.repository.load_frame(source, frame_index)


@dataclass(frozen=True)
class ConvertFile:
    executor: ConversionExecutorPort

    def __call__(
        self,
        request: ConversionRequest,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        validate_options(request.options)
        selected = [
            source
            for source in request.sources
            if source.included and source.status != "Failed" and source.selected_frames
        ]
        if not selected:
            raise ValueError("Select at least one image or frame to convert.")
        return self.executor.execute(request, progress)

    def cancel(self) -> None:
        self.executor.cancel()

    def set_paused(self, paused: bool) -> None:
        self.executor.set_paused(paused)


def convert_file(
    request: ConversionRequest,
    executor: ConversionExecutorPort,
    progress: ProgressCallback | None = None,
) -> ConversionResult:
    """函数式入口，便于脚本和测试在无 QApplication 环境中调用。"""
    return ConvertFile(executor)(request, progress)
