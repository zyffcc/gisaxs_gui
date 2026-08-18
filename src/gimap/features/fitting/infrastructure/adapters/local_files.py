"""Fitting 当前本地文件格式的 adapters。"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.gimap.shared.detector_io import (
    detect_nxs_frame_count,
    load_detector_image,
    nxs_series_paths,
)
from utils.load_SAXS_data import load_xy_any

from ...application.models import (
    ExportFitResultRequest,
    ExportedFitResult,
    LoadCurveRequest,
    LoadScatteringFileRequest,
    ScatteringFileData,
    ScatteringSequenceInfo,
)
from ...domain import CurveData


PathPreparer = Callable[[str], str]
ProgressCallback = Callable[[int, str], None]


def _natural_sort_key(path: Path):
    return [int(value) if value.isdigit() else value.lower() for value in re.split(r"(\d+)", path.name)]


class LocalScatteringFileRepository:
    def __init__(
        self,
        *,
        prepare_path: PathPreparer | None = None,
        progress: ProgressCallback | None = None,
    ):
        self._prepare_path = prepare_path or (lambda value: value)
        self._progress = progress or (lambda _percent, _message: None)

    def load(self, request: LoadScatteringFileRequest) -> ScatteringFileData:
        source = Path(request.path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Scattering file was not found: {source}")
        suffix = source.suffix.lower()
        if suffix not in {".cbf", ".nxs", ".tif", ".tiff"}:
            raise ValueError(f"Unsupported scattering image format: {suffix or '<none>'}")

        stack_count = max(1, int(request.stack_count))
        frame_index = max(0, int(request.frame_index))
        if suffix == ".nxs":
            images, source_files, effective_files = self._load_nxs(
                source,
                frame_index,
                stack_count,
            )
        else:
            extensions = {suffix} if suffix in {".cbf"} else {".tif", ".tiff"}
            selected = self._ordinary_stack(source, stack_count, extensions)
            images, source_files, effective_files = self._load_ordinary(selected)
        if not images:
            raise ValueError(f"No readable detector image was found from {source}")

        summed = np.asarray(images[0], dtype=np.float32).copy()
        for image in images[1:]:
            next_image = np.asarray(image, dtype=np.float32)
            if next_image.shape != summed.shape:
                raise ValueError("Detector images in a stack must have the same shape")
            summed += next_image
        return ScatteringFileData(
            image=summed,
            source_path=source,
            source_files=tuple(source_files),
            frame_index=frame_index,
            metadata={
                "format": suffix.lstrip("."),
                "stack_count": len(images),
                "effective_files": tuple(str(item) for item in effective_files),
            },
        )

    def inspect_sequence(self, path) -> ScatteringSequenceInfo:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".nxs":
            return ScatteringSequenceInfo(source, source, (source,), 1)
        series = tuple(nxs_series_paths(source))
        logical_path = series[0] if series else source
        frame_count = max(1, int(detect_nxs_frame_count(source)))
        return ScatteringSequenceInfo(source, logical_path, series or (source,), frame_count)

    def _load_nxs(self, source: Path, frame_index: int, stack_count: int):
        frame_count = max(1, int(detect_nxs_frame_count(source)))
        start = max(0, min(frame_index, frame_count - 1))
        actual_count = min(stack_count, frame_count - start)
        images = []
        for offset in range(actual_count):
            frame = start + offset
            self._progress(
                40 + int((offset / actual_count) * 40),
                f"Processing NXS frame {frame + 1}/{frame_count}",
            )
            images.append(np.asarray(load_detector_image(source, frame_idx=frame).data))
        source_files = [source] * actual_count
        return images, source_files, source_files

    def _ordinary_stack(self, source: Path, stack_count: int, extensions: set[str]):
        candidates = sorted(
            (path for path in source.parent.iterdir() if path.suffix.lower() in extensions),
            key=_natural_sort_key,
        )
        try:
            start = candidates.index(source)
        except ValueError as exc:
            raise FileNotFoundError(f"Scattering file is not in its parent directory: {source}") from exc
        actual_count = min(stack_count, len(candidates) - start)
        return candidates[start : start + actual_count]

    def _load_ordinary(self, selected: list[Path]):
        images = []
        effective_files: list[Path] = []
        source_files: list[Path] = []
        for index, source in enumerate(selected):
            self._progress(
                40 + int((index / max(1, len(selected))) * 40),
                f"Processing file {index + 1}/{len(selected)}: {source.name}",
            )
            effective = Path(self._prepare_path(str(source))).expanduser().resolve()
            try:
                image = load_detector_image(effective).data
            except Exception:
                if len(selected) == 1:
                    raise
                continue
            source_files.append(source)
            effective_files.append(effective)
            images.append(np.asarray(image, dtype=np.float32))
        return images, source_files, effective_files


class LocalCurveRepository:
    def load(self, request: LoadCurveRequest) -> CurveData:
        source = Path(request.path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Curve file was not found: {source}")
        if source.suffix.lower() not in {".dat", ".txt"}:
            raise ValueError(f"Unsupported curve format: {source.suffix or '<none>'}")
        loaded = load_xy_any(str(source))
        return CurveData(
            q=loaded.q,
            intensity=loaded.I,
            error=getattr(loaded, "err", None),
            q_source_unit=request.q_source_unit,
            source_path=str(source),
        )


class LocalFitResultRepository:
    def export(self, request: ExportFitResultRequest) -> ExportedFitResult:
        target = Path(request.path).expanduser()
        q = np.asarray(request.q, dtype=float).reshape(-1)
        intensity = np.asarray(request.intensity, dtype=float).reshape(-1)
        count = min(q.size, intensity.size)
        if count == 0:
            raise ValueError("Fit result contains no rows")
        combined = np.column_stack([q[:count], intensity[:count]])
        delimiter = "," if target.suffix.lower() == ".csv" else "\t"
        column_header = f"{request.x_column_name}{delimiter}{request.y_column_name}"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            if request.header_lines:
                handle.write("\n".join(request.header_lines) + "\n")
            handle.write(column_header + "\n")
            np.savetxt(handle, combined, delimiter=delimiter, fmt="%.6e")
        return ExportedFitResult(path=target.resolve(), row_count=count, delimiter=delimiter)
