"""Task model and conversion engine for the GUI format converter.

Image decoding intentionally goes through :mod:`calibration.image_loader`, the
same loader used by the embedded WAXS page and geometry tools.  This keeps NXS
module stitching, frame orientation, CBF metadata extraction, and TIFF handling
consistent across the application.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import h5py
import numpy as np

from calibration.image_loader import (
    _dataset_candidates,
    load_detector_image,
    select_nxs_dataset,
)


SUPPORTED_SUFFIXES = {".nxs": "NXS", ".cbf": "CBF", ".tif": "TIFF", ".tiff": "TIFF"}
OUTPUT_SUFFIXES = {"TIFF": ".tif", "CBF": ".cbf", "HDF5": ".h5", "NumPy": ".npy"}


def parse_custom_frames(text: str, frame_count: int) -> list[int]:
    """Parse one-based comma/range syntax and return sorted zero-based frames."""
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


@dataclass
class InputSource:
    path: str
    file_type: str
    frame_count: int = 1
    dataset_path: Optional[str] = None
    dataset_paths: list[str] = field(default_factory=list)
    dataset_shape: tuple[int, ...] = field(default_factory=tuple)
    selected_frames: list[int] = field(default_factory=lambda: [0])
    included: bool = True
    status: str = "Ready"
    error: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    @property
    def selection_summary(self) -> str:
        return compact_frame_summary(self.selected_frames, self.frame_count)


def inspect_source(path: str | Path) -> InputSource:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported input format: {suffix or source.name}")
    if not source.is_file():
        raise FileNotFoundError(str(source))
    item = InputSource(path=str(source), file_type=SUPPORTED_SUFFIXES[suffix])
    if suffix != ".nxs":
        return item

    with h5py.File(str(source), "r") as handle:
        ranked_candidates = _dataset_candidates(handle)
        candidates = [dataset_path for _score, dataset_path in ranked_candidates]
        try:
            recommended = select_nxs_dataset(handle)
        except ValueError:
            # The shared reader deliberately reports equally plausible datasets
            # as ambiguous.  The converter can still present that choice, so use
            # its highest-ranked candidate as the non-destructive default.
            if not candidates:
                raise
            recommended = candidates[0]
        item.dataset_paths = [recommended] + [path for path in candidates if path != recommended]
        item.dataset_path = recommended
        dataset = handle[recommended]
        item.dataset_shape = tuple(int(value) for value in dataset.shape)
        item.frame_count = int(dataset.shape[0]) if dataset.ndim == 3 else 1
        item.selected_frames = list(range(item.frame_count))
    return item


def select_dataset(item: InputSource, dataset_path: str) -> None:
    """Update an inspected NXS source after an advanced dataset selection."""
    if item.file_type != "NXS":
        return
    with h5py.File(item.path, "r") as handle:
        if dataset_path not in handle:
            raise ValueError(f"Dataset does not exist: {dataset_path}")
        dataset = handle[dataset_path]
        if dataset.ndim not in (2, 3) or not np.issubdtype(dataset.dtype, np.number):
            raise ValueError("The selected dataset is not a readable 2D/3D numeric image dataset.")
        item.dataset_path = dataset_path
        item.dataset_shape = tuple(int(value) for value in dataset.shape)
        item.frame_count = int(dataset.shape[0]) if dataset.ndim == 3 else 1
        item.selected_frames = list(range(item.frame_count))


def scan_folder(
    folder: str | Path,
    *,
    include_cbf: bool = True,
    include_tiff: bool = True,
    include_nxs: bool = True,
    recursive: bool = False,
) -> list[str]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    suffixes = set()
    if include_cbf:
        suffixes.add(".cbf")
    if include_tiff:
        suffixes.update((".tif", ".tiff"))
    if include_nxs:
        suffixes.add(".nxs")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(str(path) for path in iterator if path.is_file() and path.suffix.lower() in suffixes)


@dataclass
class ConversionOptions:
    output_format: str
    destination: str
    naming_template: str = "{source}_{frame:06d}"
    add_suffix: bool = True
    preserve_values: bool = True
    data_mode: str = "original"  # original, float32, scale_uint16, clip_uint16
    preserve_metadata: bool = True
    write_sidecar: bool = True
    single_metadata_file: bool = True
    container: bool = False

    @property
    def suffix(self) -> str:
        return OUTPUT_SUFFIXES[self.output_format]


@dataclass
class ConversionJob:
    source: InputSource
    frame_index: int
    output_path: str


@dataclass
class ConversionReport:
    started_at: float
    finished_at: float = 0.0
    succeeded: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False
    report_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": max(0.0, self.finished_at - self.started_at),
            "cancelled": self.cancelled,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }


def _safe_filename(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r'[<>:"|?*]', "_", value)
    return value or "converted"


def render_output_stem(source: InputSource, frame_index: int, template: str) -> str:
    try:
        rendered = template.format(source=source.stem, frame=frame_index + 1, img="img")
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid naming template: {exc}") from exc
    return _safe_filename(rendered)


def build_jobs(sources: Iterable[InputSource], options: ConversionOptions) -> list[ConversionJob]:
    destination = Path(options.destination).expanduser().resolve()
    planned: set[str] = set()
    jobs: list[ConversionJob] = []
    container_path: Optional[Path] = None
    if options.container:
        container_path = destination / f"converted_images{options.suffix}"
        counter = 1
        while container_path.exists():
            container_path = destination / f"converted_images_{counter:03d}{options.suffix}"
            counter += 1
    for source in sources:
        if not source.included or source.status == "Failed":
            continue
        frames = source.selected_frames or []
        for frame_index in frames:
            if source.frame_count == 1:
                stem = _safe_filename(source.stem)
            else:
                stem = render_output_stem(source, frame_index, options.naming_template)
            candidate = destination / f"{stem}{options.suffix}"
            if container_path is not None:
                candidate = container_path
            key = os.path.normcase(str(candidate))
            if not options.container and (key in planned or candidate.exists()) and options.add_suffix:
                counter = 1
                while True:
                    suffixed = destination / f"{stem}_{counter:03d}{options.suffix}"
                    candidate_key = os.path.normcase(str(suffixed))
                    if candidate_key not in planned and not suffixed.exists():
                        candidate, key = suffixed, candidate_key
                        break
                    counter += 1
            elif not options.container and (key in planned or candidate.exists()):
                raise FileExistsError(
                    f"Output already exists or is duplicated: {candidate}. Enable automatic suffixes or change the naming template."
                )
            planned.add(key)
            jobs.append(ConversionJob(source=source, frame_index=int(frame_index), output_path=str(candidate)))
    return jobs


def estimate_output(sources: Iterable[InputSource], options: ConversionOptions) -> tuple[int, int]:
    """Return selected image count and a conservative byte estimate."""
    selected = [source for source in sources if source.included and source.status != "Failed"]
    count = sum(len(source.selected_frames) for source in selected)
    total = 0
    for source in selected:
        if source.dataset_shape:
            pixels = math.prod(source.dataset_shape[-2:])
            bytes_per_pixel = 4
            try:
                with h5py.File(source.path, "r") as handle:
                    bytes_per_pixel = int(handle[source.dataset_path].dtype.itemsize)
            except Exception:
                pass
            total += pixels * bytes_per_pixel * len(source.selected_frames)
        else:
            try:
                total += max(Path(source.path).stat().st_size, 1) * len(source.selected_frames)
            except OSError:
                pass
    # CBF/HDF5 compression varies; this is deliberately a planning estimate.
    return count, int(total)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _convert_dtype(data: np.ndarray, mode: str) -> np.ndarray:
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
        return np.clip(np.nan_to_num(array, nan=0.0, posinf=65535.0, neginf=0.0), 0, 65535).astype(np.uint16)
    return array


class ConversionEngine:
    """Synchronous conversion loop intended to run inside a ``QThread``."""

    def __init__(self, options: ConversionOptions):
        self.options = options
        self._cancel = threading.Event()
        self._resume = threading.Event()
        self._resume.set()

    def cancel(self) -> None:
        self._cancel.set()
        self._resume.set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._resume.clear()
        else:
            self._resume.set()

    def run(
        self,
        sources: Iterable[InputSource],
        progress: Optional[Callable[[int, int, str, int], None]] = None,
    ) -> ConversionReport:
        destination = Path(self.options.destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        source_list = list(sources)
        jobs = build_jobs(source_list, self.options)
        report = ConversionReport(started_at=time.time())
        metadata_records: list[dict[str, Any]] = []
        container: Optional[h5py.File] = None
        try:
            if self.options.container and jobs:
                container = h5py.File(jobs[0].output_path, "w")
                container.attrs["NX_class"] = "NXroot"
                container.create_group("entry").attrs["NX_class"] = "NXentry"
            for position, job in enumerate(jobs, start=1):
                self._resume.wait()
                if self._cancel.is_set():
                    report.cancelled = True
                    break
                if progress:
                    progress(position - 1, len(jobs), job.source.name, job.frame_index)
                try:
                    if job.source.status != "Failed":
                        job.source.status = "Converting"
                    detector_image = load_detector_image(
                        job.source.path,
                        frame_idx=job.frame_index,
                        dataset_path=job.source.dataset_path,
                    )
                    data = _convert_dtype(detector_image.data, self.options.data_mode)
                    metadata = _json_safe(dict(detector_image.metadata or {}))
                    metadata.update({
                        "source": job.source.path,
                        "frame_number": job.frame_index + 1,
                        "output": job.output_path,
                        "shape": list(data.shape),
                        "dtype": str(data.dtype),
                    })
                    if container is not None:
                        self._write_container_frame(container, data, metadata, position)
                    else:
                        self._write_frame(Path(job.output_path), data, metadata)
                    metadata_records.append(metadata)
                    report.succeeded.append({"source": job.source.path, "frame": job.frame_index + 1, "output": job.output_path})
                    if self.options.write_sidecar and not self.options.single_metadata_file and container is None:
                        Path(job.output_path).with_suffix(".json").write_text(
                            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                except Exception as exc:
                    job.source.status = "Failed"
                    job.source.error = str(exc)
                    report.failed.append({"source": job.source.path, "frame": job.frame_index + 1, "error": str(exc)})
                if progress:
                    progress(position, len(jobs), job.source.name, job.frame_index)
        finally:
            if container is not None:
                container.close()
        if self.options.write_sidecar and self.options.single_metadata_file and metadata_records:
            metadata_path = destination / "conversion_metadata.json"
            metadata_path.write_text(json.dumps(metadata_records, indent=2, ensure_ascii=False), encoding="utf-8")
        failed_sources = {item["source"] for item in report.failed}
        for source in source_list:
            if source.included and source.path not in failed_sources:
                source.status = "Cancelled" if report.cancelled else "Completed"
        report.finished_at = time.time()
        report_path = destination / "conversion_report.json"
        report.report_path = str(report_path)
        report_path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _write_frame(self, path: Path, data: np.ndarray, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.options.output_format == "NumPy":
            np.save(str(path), data, allow_pickle=False)
            return
        if self.options.output_format == "HDF5":
            with h5py.File(str(path), "w") as handle:
                entry = handle.create_group("entry")
                entry.attrs["NX_class"] = "NXentry"
                dataset = entry.create_dataset("data/data", data=data, compression="gzip", shuffle=True)
                if self.options.preserve_metadata:
                    dataset.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            return
        if self.options.output_format == "TIFF":
            from fabio.tifimage import TifImage

            header = {"GIMaP_metadata": json.dumps(metadata, ensure_ascii=False)} if self.options.preserve_metadata else {}
            TifImage(data=data, header=header).write(str(path))
            return
        if self.options.output_format == "CBF":
            from fabio.cbfimage import CbfImage

            cbf_data = data
            if not np.issubdtype(cbf_data.dtype, np.integer):
                cbf_data = np.nan_to_num(cbf_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            header = {"GIMaP_metadata": json.dumps(metadata, ensure_ascii=False)} if self.options.preserve_metadata else {}
            CbfImage(data=cbf_data, header=header).write(str(path))
            return
        raise ValueError(f"Unsupported output format: {self.options.output_format}")

    def _write_container_frame(self, handle: h5py.File, data: np.ndarray, metadata: dict[str, Any], index: int) -> None:
        group = handle["entry"].require_group("data")
        group.attrs["NX_class"] = "NXdata"
        dataset = group.create_dataset(f"image_{index:06d}", data=data, compression="gzip", shuffle=True)
        if self.options.preserve_metadata:
            dataset.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
