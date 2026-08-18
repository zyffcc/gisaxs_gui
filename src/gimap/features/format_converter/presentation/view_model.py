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
    NormalizePath,
    ScanFolder,
    SelectDataset,
)
from ..application.ports import ConversionExecutorPort, ProgressCallback, SourceRepositoryPort
from ..domain.models import ConversionOptions, ConversionRequest, ConversionResult, InputSource
from ..domain.rules import (
    is_supported_input_path,
    output_may_lose_float_values,
    output_naming_summary,
    render_output_example,
    select_source_frame_indices,
    validate_options,
    visible_output_formats,
)
from .state import ConversionReviewState, FormatConverterState, OutputPreviewState


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
        self._normalize_path = NormalizePath(repository)
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
                normalized_path = self._normalize_path(raw_path)
                source = self._inspect_source(normalized_path)
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

    def normalize_path(self, path: str) -> str:
        return self._normalize_path(path)

    @staticmethod
    def supports_input_path(path: str) -> bool:
        return is_supported_input_path(path)

    def remove_indices(self, indices: list[int]) -> None:
        selected = set(indices)
        self.sources[:] = [
            source for index, source in enumerate(self.sources) if index not in selected
        ]

    def sort_sources(self) -> None:
        self.sources.sort(key=lambda source: source.name.lower())

    def set_all_included(self, included: bool) -> None:
        for source in self.sources:
            source.included = included

    def set_source_included(self, index: int, included: bool) -> None:
        if 0 <= index < len(self.sources):
            self.sources[index].included = included

    def apply_frame_selection(
        self,
        indices: list[int],
        mode: str,
        *,
        current_file: str = "",
        current_frame: int = 1,
        range_start: int = 1,
        range_end: int | None = None,
        custom_frames: str = "",
        nth_frame: int = 1,
    ) -> None:
        for index in indices:
            source = self.sources[index]
            source_current_frame = 1
            if os.path.normcase(source.path) == os.path.normcase(current_file or ""):
                source_current_frame = current_frame
            source.selected_frames = select_source_frame_indices(
                source.file_type,
                source.frame_count,
                mode,
                current_frame=source_current_frame,
                range_start=range_start,
                range_end=range_end,
                custom_frames=custom_frames,
                nth_frame=nth_frame,
            )

    def select_dataset(self, source: InputSource, dataset_path: str) -> None:
        self._select_dataset(source, dataset_path)

    def scan_folder(self, folder: str, **options) -> list[str]:
        return self._scan_folder(folder, **options)

    def estimate_output(self, options: ConversionOptions) -> tuple[int, int]:
        request = ConversionRequest(sources=tuple(self.sources), options=options)
        return self._estimate_output(request)

    def output_format_visibility(self, *, container: bool = False) -> dict[str, bool]:
        input_types = (source.file_type for source in self.sources if source.included)
        return visible_output_formats(input_types, container=container)

    @staticmethod
    def make_options(**values) -> ConversionOptions:
        return ConversionOptions(**values)

    def output_preview(self, options: ConversionOptions) -> OutputPreviewState:
        try:
            example = render_output_example(options)
        except Exception:
            example = "Invalid naming template"
        count, estimated_bytes = self.estimate_output(options)
        output_files = 1 if options.container and count else count
        warning = ""
        if output_may_lose_float_values(options.output_format, options.data_mode):
            warning = (
                "CBF encoders may not preserve NaN values or every floating-point "
                "representation. A metadata sidecar is recommended."
            )
        return OutputPreviewState(
            example=example,
            image_count=count,
            file_count=output_files,
            estimated_bytes=estimated_bytes,
            dtype_warning=warning,
        )

    def conversion_review(self, options: ConversionOptions) -> ConversionReviewState:
        validate_options(options)
        selected_sources = [source for source in self.sources if source.included]
        image_count, estimated_bytes = self.estimate_output(options)
        type_counts: dict[str, int] = {}
        for source in selected_sources:
            type_counts[source.file_type] = type_counts.get(source.file_type, 0) + 1
        input_summary = ", ".join(
            f"{count} {kind} file(s)"
            for kind, count in sorted(type_counts.items())
        )
        naming = output_naming_summary(options)
        output_files = 1 if options.container and image_count else image_count
        return ConversionReviewState(
            input_summary=input_summary,
            image_count=image_count,
            output_files=output_files,
            estimated_bytes=estimated_bytes,
            destination=self._normalize_path(options.destination),
            naming=naming,
            is_large_output=image_count > 10_000 or estimated_bytes > 20 * 1024**3,
        )

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
