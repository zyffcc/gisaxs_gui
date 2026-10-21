"""基于本地文件系统和现有 detector loader 的 Format Converter adapters。"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.gimap.shared.detector_io import (
    HDF5_SUFFIXES,
    NexusLayout,
    cached_layout,
    group_sources,
    load_detector_image,
    scan_stem,
    write_frame,
)
from src.gimap.shared.detector_io.nexus_layout import DATASET_NOT_FOUND
from src.gimap.shared.file_paths import normalize_path

from ...application.ports import ProgressCallback
from ...domain.models import (
    SUPPORTED_SUFFIXES,
    ConversionOptions,
    ConversionRequest,
    ConversionResult,
    InputSource,
)
from ...domain.rules import build_jobs, convert_dtype

NO_IMAGE = "{name} holds no detector image."


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


def discard_failed_output(path: Path, was_there: bool) -> None:
    """A writer that fails after opening its file leaves an empty one: it looks like a frame and cannot be
    read. Remove what the failed write created; a file that was there before goes only if it is now empty."""
    try:
        if path.is_file() and (not was_there or path.stat().st_size == 0):
            path.unlink()
    except OSError:
        pass


class LocalSourceRepository:
    """读取并检查本地 detector image sources。"""

    def normalize_path(self, path: str | Path) -> str:
        return str(Path(normalize_path(path)).expanduser().resolve())

    def inspect_source(self, path: str | Path) -> InputSource:
        source_path = Path(self.normalize_path(path)).expanduser().resolve()
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input format: {suffix or source_path.name}")
        if not source_path.is_file():
            raise FileNotFoundError(str(source_path))
        source = InputSource(path=str(source_path), file_type=SUPPORTED_SUFFIXES[suffix])
        if not source.is_hdf5:
            return source
        # One inspection for every reader: what the source consists of, its image dataset and its frames.
        layout = cached_layout(source_path)
        if layout.kind == "no_image":
            raise ValueError(NO_IMAGE.format(name=source_path.name))
        source.dataset_paths = [layout.dataset, *layout.alternatives]
        self._describe(source, layout)
        return source

    @staticmethod
    def _describe(source: InputSource, layout: NexusLayout) -> None:
        """Fill ``source`` from the layout: any module file stands for its set (``path`` is the first one)."""
        if layout.kind == "module_set":
            source.path = str(layout.path)
        source.scan_name = scan_stem(layout)  # a set of module files: its scan, without the module number
        source.dataset_path = layout.dataset
        lead = tuple(layout.lead_shape) if len(layout.lead_shape) > 1 else (
            (int(layout.frames),) if layout.lead_shape or layout.frames != 1 else ()
        )
        source.dataset_shape = tuple(int(value) for value in (*lead, *layout.frame_shape))
        source.frame_count = int(layout.frames)
        source.selected_frames = list(range(source.frame_count))
        source.layout = layout.kind
        source.members = [str(member) for member in layout.members]
        source.summary = layout.summary()
        source.warnings = list(layout.warnings)
        source.stored_dtype = layout.dtype

    def select_dataset(self, source: InputSource, dataset_path: str) -> None:
        if not source.is_hdf5:
            return
        layout = cached_layout(source.path, dataset_path)  # raises for a dataset that holds no frames
        if layout.kind == "no_image" or any(template == DATASET_NOT_FOUND for template, _values in layout.warnings):
            raise ValueError(f"Dataset does not exist: {dataset_path}")
        self._describe(source, layout)

    def group_paths(self, paths: list[str | Path]) -> list[str]:
        """One path per source of the files a person chose: the module files of a set become its first
        file, a master and its data files the master. A file without a detector image stays (adding it
        says why it cannot be an input)."""
        return [str(path) for path in group_sources(list(paths), keep_without_image=True)]

    def scan_folder(
        self,
        folder: str | Path,
        *,
        include_cbf: bool = True,
        include_tiff: bool = True,
        include_nxs: bool = True,
        recursive: bool = False,
    ) -> list[str]:
        root = Path(self.normalize_path(folder)).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        suffixes = set()
        if include_cbf:
            suffixes.add(".cbf")
        if include_tiff:
            suffixes.update((".tif", ".tiff"))
        if include_nxs:
            suffixes.update(HDF5_SUFFIXES)
        iterator = root.rglob("*") if recursive else root.glob("*")
        found = sorted(
            str(candidate)
            for candidate in iterator
            if candidate.is_file() and candidate.suffix.lower() in suffixes
        )
        # A set of module files is one input, the data files of a master are not inputs, and an HDF5 file
        # without a detector image (results, model weights) is not listed.
        return [str(path) for path in group_sources(found)]

    def estimate_output(
        self,
        sources: list[InputSource],
        request: ConversionRequest,
    ) -> tuple[int, int]:
        del request
        selected = [source for source in sources if source.included and source.status != "Failed"]
        count = sum(len(source.selected_frames) for source in selected)
        total = 0
        for source in selected:
            if source.dataset_shape:
                # The stored size of one frame; a stitched frame holds every module of the set.
                pixels = math.prod(source.dataset_shape[-2:])
                if source.layout == "module_set":
                    pixels *= max(1, len(source.members))
                bytes_per_pixel = 4
                try:
                    bytes_per_pixel = int(np.dtype(source.stored_dtype).itemsize) if source.stored_dtype else 4
                except TypeError:
                    pass
                total += pixels * bytes_per_pixel * len(source.selected_frames)
            else:
                try:
                    total += max(Path(source.path).stat().st_size, 1) * len(
                        source.selected_frames
                    )
                except OSError:
                    pass
        return count, int(total)

    def load_frame(
        self,
        source: InputSource,
        frame_index: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        detector_image = load_detector_image(
            source.path,
            frame_idx=frame_index,
            dataset_path=source.dataset_path,
        )
        return np.asarray(detector_image.data), dict(detector_image.metadata or {})


class LocalConversionExecutor:
    """同步执行转换；调用方可将其放入 worker thread。"""

    def __init__(self, source_repository: LocalSourceRepository | None = None):
        self.source_repository = source_repository or LocalSourceRepository()
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

    def execute(
        self,
        request: ConversionRequest,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        options = request.options
        destination = Path(options.destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        sources = list(request.sources)
        planning_options = replace(options, destination=str(destination))
        jobs = build_jobs(
            sources,
            planning_options,
            output_exists=lambda candidate: Path(candidate).exists(),
        )
        report = ConversionResult(started_at=time.time())
        metadata_records: list[dict[str, Any]] = []
        container: h5py.File | None = None
        try:
            if options.container and jobs:
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
                    data, raw_metadata = self.source_repository.load_frame(
                        job.source,
                        job.frame_index,
                    )
                    data = convert_dtype(data, options.data_mode)
                    metadata = _json_safe(raw_metadata)
                    metadata.update(
                        {
                            "source": job.source.path,
                            "frame_number": job.frame_index + 1,
                            "output": job.output_path,
                            "shape": list(data.shape),
                            "dtype": str(data.dtype),
                        }
                    )
                    if container is not None:
                        self._write_container_frame(container, data, metadata, position, options)
                    else:
                        self._write_frame(Path(job.output_path), data, metadata, options)
                    metadata_records.append(metadata)
                    report.succeeded.append(
                        {
                            "source": job.source.path,
                            "frame": job.frame_index + 1,
                            "output": job.output_path,
                        }
                    )
                    if options.write_sidecar and not options.single_metadata_file and container is None:
                        Path(job.output_path).with_suffix(".json").write_text(
                            json.dumps(metadata, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                except Exception as exc:
                    job.source.status = "Failed"
                    job.source.error = str(exc)
                    report.failed.append(
                        {
                            "source": job.source.path,
                            "frame": job.frame_index + 1,
                            "error": str(exc),
                        }
                    )
                if progress:
                    progress(position, len(jobs), job.source.name, job.frame_index)
        finally:
            if container is not None:
                container.close()

        if options.write_sidecar and options.single_metadata_file and metadata_records:
            metadata_path = destination / "conversion_metadata.json"
            metadata_path.write_text(
                json.dumps(metadata_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        failed_sources = {item["source"] for item in report.failed}
        for source in sources:
            if source.included and source.path not in failed_sources:
                source.status = "Cancelled" if report.cancelled else "Completed"
        report.finished_at = time.time()
        report_path = destination / "conversion_report.json"
        report.report_path = str(report_path)
        report_path.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _write_frame(
        path: Path,
        data: np.ndarray,
        metadata: dict[str, Any],
        options: ConversionOptions,
    ) -> None:
        formats = {"NumPy": "npy", "HDF5": "hdf5", "TIFF": "tiff", "CBF": "cbf", "EDF": "edf"}
        if options.output_format not in formats:
            raise ValueError(f"Unsupported output format: {options.output_format}")
        fmt = formats[options.output_format]
        kept = metadata if options.preserve_metadata else None
        was_there = path.exists()
        try:
            # (The shared writer keeps the record of a CBF or EDF frame as ASCII JSON: it reads back exactly.)
            write_frame(path, data, fmt, kept)
        except Exception:
            discard_failed_output(path, was_there)
            raise

    @staticmethod
    def _write_container_frame(
        handle: h5py.File,
        data: np.ndarray,
        metadata: dict[str, Any],
        index: int,
        options: ConversionOptions,
    ) -> None:
        group = handle["entry"].require_group("data")
        group.attrs["NX_class"] = "NXdata"
        dataset = group.create_dataset(
            f"image_{index:06d}",
            data=data,
            compression="gzip",
            shuffle=True,
        )
        if options.preserve_metadata:
            dataset.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)


class ConversionEngine:
    """保留旧构造方式的 compatibility wrapper。"""

    def __init__(self, options: ConversionOptions):
        self.options = options
        self.executor = LocalConversionExecutor()

    def cancel(self) -> None:
        self.executor.cancel()

    def set_paused(self, paused: bool) -> None:
        self.executor.set_paused(paused)

    def run(
        self,
        sources,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        request = ConversionRequest(sources=tuple(sources), options=self.options)
        return self.executor.execute(request, progress)
