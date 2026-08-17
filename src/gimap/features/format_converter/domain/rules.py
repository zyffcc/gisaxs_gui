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
