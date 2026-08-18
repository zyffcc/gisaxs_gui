"""不依赖 GUI 和文件系统的格式转换规则。"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import ConversionJob, ConversionOptions, InputSource


def parse_custom_frames(text: str, frame_count: int) -> list[int]:
    """解析从 1 开始的逗号/范围语法，并返回从 0 开始的有序帧索引。"""
    if frame_count < 1:
        return []
    result: set[int] = set()
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start > end:
                raise ValueError(f"Frame range starts after it ends: {part}")
            if start < 1 or end > frame_count:
                raise ValueError(f"Frame range {part} is outside 1–{frame_count}.")
            result.update(range(start - 1, end))
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid frame selection: {part}")
        value = int(part)
        if value < 1 or value > frame_count:
            raise ValueError(f"Frame {value} is outside 1–{frame_count}.")
        result.add(value - 1)
    if not result:
        raise ValueError("Select at least one frame.")
    return sorted(result)


def is_supported_input_path(path: str | Path) -> bool:
    from .models import SUPPORTED_SUFFIXES

    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def select_frame_indices(
    frame_count: int,
    mode: str,
    *,
    current_frame: int = 1,
    range_start: int = 1,
    range_end: int | None = None,
    custom_frames: str = "",
    nth_frame: int = 1,
) -> list[int]:
    """Apply the dialog's one-based frame-selection contract."""
    if frame_count <= 1:
        return [0]
    if mode == "All":
        return list(range(frame_count))
    if mode == "Current frame":
        return [max(0, min(frame_count - 1, int(current_frame) - 1))]
    if mode == "Frame range":
        end = frame_count if range_end is None else int(range_end)
        frames = list(range(int(range_start) - 1, end))
        if not frames:
            raise ValueError("The frame range is empty.")
        return frames
    if mode == "Custom":
        return parse_custom_frames(custom_frames, frame_count)
    return list(range(0, frame_count, max(1, int(nth_frame))))


def select_source_frame_indices(
    file_type: str,
    frame_count: int,
    mode: str,
    **selection,
) -> list[int]:
    if file_type != "NXS":
        return [0]
    return select_frame_indices(frame_count, mode, **selection)


def visible_output_formats(
    input_types: Iterable[str],
    *,
    container: bool = False,
) -> dict[str, bool]:
    """Return the legacy format-button visibility policy as data."""
    source_types = set(input_types)
    same_format = {"TIFF": "TIFF", "CBF": "CBF", "NXS": "HDF5"}
    visibility: dict[str, bool] = {}
    for output_format in ("TIFF", "CBF", "HDF5", "NumPy"):
        hidden = len(source_types) == 1 and any(
            same_format.get(source_type) == output_format
            for source_type in source_types
        )
        if output_format == "HDF5" and (len(source_types) > 1 or container):
            hidden = False
        visibility[output_format] = not hidden
    return visibility


def render_output_example(options: ConversionOptions) -> str:
    if options.container:
        return "converted_images.h5"
    rendered = options.naming_template.format(
        source="scan_001",
        frame=123,
        img="img",
    )
    return rendered + options.suffix


def output_naming_summary(options: ConversionOptions) -> str:
    return (
        "converted_images.h5"
        if options.container
        else options.naming_template + options.suffix
    )


def output_may_lose_float_values(output_format: str, data_mode: str) -> bool:
    return output_format == "CBF" and data_mode in ("original", "float32")


def compact_frame_summary(frames: Iterable[int], frame_count: int) -> str:
    values = sorted(set(int(value) for value in frames))
    if not values:
        return "None"
    if values == list(range(max(1, int(frame_count)))):
        return f"1–{frame_count}" if frame_count > 1 else "1"
    if len(values) == 1:
        return str(values[0] + 1)
    consecutive = all(right == left + 1 for left, right in zip(values, values[1:]))
    if consecutive:
        return f"{values[0] + 1}–{values[-1] + 1}"
    if len(values) <= 6:
        return ", ".join(str(value + 1) for value in values)
    return f"{values[0] + 1}, …, {values[-1] + 1} ({len(values)} frames)"


def safe_filename(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r'[<>:"|?*]', "_", value)
    return value or "converted"


def render_output_stem(source: InputSource, frame_index: int, template: str) -> str:
    try:
        rendered = template.format(source=source.stem, frame=frame_index + 1, img="img")
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid naming template: {exc}") from exc
    return safe_filename(rendered)


def validate_options(options: ConversionOptions) -> None:
    _ = options.suffix
    if not options.destination.strip():
        raise ValueError("Choose an output destination.")
    try:
        options.naming_template.format(source="scan_001", frame=1, img="img")
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid naming template: {exc}") from exc


def build_jobs(
    sources: Iterable[InputSource],
    options: ConversionOptions,
    output_exists: Callable[[str], bool] | None = None,
) -> list[ConversionJob]:
    from .models import ConversionJob

    validate_options(options)
    destination = Path(options.destination)
    exists = output_exists or (lambda _path: False)
    planned: set[str] = set()
    jobs: list[ConversionJob] = []
    container_path: Path | None = None
    if options.container:
        container_path = destination / f"converted_images{options.suffix}"
        counter = 1
        while exists(str(container_path)):
            container_path = destination / f"converted_images_{counter:03d}{options.suffix}"
            counter += 1
    for source in sources:
        if not source.included or source.status == "Failed":
            continue
        for frame_index in source.selected_frames or []:
            stem = (
                safe_filename(source.stem)
                if source.frame_count == 1
                else render_output_stem(source, frame_index, options.naming_template)
            )
            candidate = container_path or destination / f"{stem}{options.suffix}"
            key = os.path.normcase(str(candidate))
            if not options.container and (key in planned or exists(str(candidate))) and options.add_suffix:
                counter = 1
                while True:
                    suffixed = destination / f"{stem}_{counter:03d}{options.suffix}"
                    candidate_key = os.path.normcase(str(suffixed))
                    if candidate_key not in planned and not exists(str(suffixed)):
                        candidate, key = suffixed, candidate_key
                        break
                    counter += 1
            elif not options.container and (key in planned or exists(str(candidate))):
                raise FileExistsError(
                    "Output already exists or is duplicated: "
                    f"{candidate}. Enable automatic suffixes or change the naming template."
                )
            planned.add(key)
            jobs.append(
                ConversionJob(
                    source=source,
                    frame_index=int(frame_index),
                    output_path=str(candidate),
                )
            )
    return jobs


def convert_dtype(data: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(data)
    if mode == "float32":
        return array.astype(np.float32, copy=False)
    if mode == "scale_uint16":
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return np.zeros(array.shape, dtype=np.uint16)
        low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            return np.zeros(array.shape, dtype=np.uint16)
        scaled = (np.nan_to_num(array, nan=low, posinf=high, neginf=low) - low) / (high - low)
        return np.rint(np.clip(scaled, 0, 1) * 65535).astype(np.uint16)
    if mode == "clip_uint16":
        clean = np.nan_to_num(array, nan=0.0, posinf=65535.0, neginf=0.0)
        return np.clip(clean, 0, 65535).astype(np.uint16)
    return array
