"""不依赖 PyQt widget 的 Format Converter ViewModel。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.gimap.app import AppContext

from ..application import (
    ConvertFile,
    EstimateOutput,
    InspectSource,
    LoadPreview,
    ScanFolder,
    SelectDataset,
)
from ..application.ports import ConversionExecutorPort, ProgressCallback, SourceRepositoryPort
from ..domain.models import ConversionOptions, ConversionRequest, ConversionResult, InputSource
from .state import FormatConverterState


@dataclass(frozen=True)
class AddPathsResult:
    added: int
    errors: tuple[str, ...]


class FormatConverterViewModel:
    """持有 dialog state，并将 commands 委托给 application use cases。"""

    def __init__(
        self,
        *,
        app_context: AppContext,
        repository: SourceRepositoryPort,
        executor: ConversionExecutorPort,
    ):
        self.app_context = app_context
        self.state = app_context.project_state.feature_state(
            "format_converter",
            FormatConverterState,
        )
        self.sources: list[InputSource] = []
        self._inspect_source = InspectSource(repository)
        self._select_dataset = SelectDataset(repository)
        self._scan_folder = ScanFolder(repository)
        self._estimate_output = EstimateOutput(repository)
        self._load_preview = LoadPreview(repository)
        self._convert_file = ConvertFile(executor)

    def add_paths(self, paths: list[str]) -> AddPathsResult:
        existing = {os.path.normcase(source.path) for source in self.sources}
        added = 0
        errors: list[str] = []
        for raw_path in paths:
            if not raw_path:
                continue
            try:
                source = self._inspect_source(raw_path)
            except Exception as exc:
                errors.append(f"{Path(raw_path).name}: {exc}")
                continue
            key = os.path.normcase(source.path)
            if key in existing:
                continue
            self.sources.append(source)
            existing.add(key)
            added += 1
        return AddPathsResult(added=added, errors=tuple(errors))

    def remove_indices(self, indices: list[int]) -> None:
        selected = set(indices)
        self.sources[:] = [
            source for index, source in enumerate(self.sources) if index not in selected
        ]

    def sort_sources(self) -> None:
        self.sources.sort(key=lambda source: source.name.lower())

    def select_dataset(self, source: InputSource, dataset_path: str) -> None:
        self._select_dataset(source, dataset_path)

    def scan_folder(self, folder: str, **options) -> list[str]:
        return self._scan_folder(folder, **options)

    def estimate_output(self, options: ConversionOptions) -> tuple[int, int]:
        request = ConversionRequest(sources=tuple(self.sources), options=options)
        return self._estimate_output(request)

    def load_preview(self, source: InputSource) -> list[dict]:
        frames = source.selected_frames or [0]
        picks = [frames[0], frames[len(frames) // 2], frames[-1]]
        payload = []
        for label, frame in zip(("First", "Middle", "Last"), picks):
            data, _metadata = self._load_preview(source, frame)
            array = np.asarray(data)
            finite = array[np.isfinite(array)]
            minimum = float(np.min(finite)) if finite.size else None
            maximum = float(np.max(finite)) if finite.size else None
            max_count = int(np.count_nonzero(array == maximum)) if maximum is not None else 0
            payload.append(
                {
                    "label": label,
                    "frame": frame + 1,
                    "data": array,
                    "shape": tuple(array.shape),
                    "dtype": str(array.dtype),
                    "minimum": minimum,
                    "maximum": maximum,
                    "nan_count": int(np.count_nonzero(~np.isfinite(array))),
                    "negative_count": int(np.count_nonzero(np.isfinite(array) & (array < 0))),
                    "max_count": max_count,
                }
            )
        return payload

    def convert(
        self,
        options: ConversionOptions,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        self.state.destination = options.destination
        self.state.output_format = options.output_format
        request = ConversionRequest(sources=tuple(self.sources), options=options)
        return self._convert_file(request, progress)

    def cancel(self) -> None:
        self._convert_file.cancel()

    def set_paused(self, paused: bool) -> None:
        self._convert_file.set_paused(paused)
