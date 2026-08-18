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
    _dataset_candidates,
    load_detector_image,
    select_nxs_dataset,
)
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
        if suffix != ".nxs":
            return source

        with h5py.File(str(source_path), "r") as handle:
            ranked_candidates = _dataset_candidates(handle)
            candidates = [dataset_path for _score, dataset_path in ranked_candidates]
            try:
                recommended = select_nxs_dataset(handle)
            except ValueError:
                if not candidates:
                    raise
                recommended = candidates[0]
            source.dataset_paths = [recommended] + [
                candidate for candidate in candidates if candidate != recommended
            ]
            source.dataset_path = recommended
            dataset = handle[recommended]
            source.dataset_shape = tuple(int(value) for value in dataset.shape)
            source.frame_count = int(dataset.shape[0]) if dataset.ndim == 3 else 1
            source.selected_frames = list(range(source.frame_count))
        return source

    def select_dataset(self, source: InputSource, dataset_path: str) -> None:
        if source.file_type != "NXS":
            return
        with h5py.File(source.path, "r") as handle:
            if dataset_path not in handle:
                raise ValueError(f"Dataset does not exist: {dataset_path}")
            dataset = handle[dataset_path]
            if dataset.ndim not in (2, 3) or not np.issubdtype(dataset.dtype, np.number):
                raise ValueError("The selected dataset is not a readable 2D/3D numeric image dataset.")
            source.dataset_path = dataset_path
            source.dataset_shape = tuple(int(value) for value in dataset.shape)
            source.frame_count = int(dataset.shape[0]) if dataset.ndim == 3 else 1
            source.selected_frames = list(range(source.frame_count))

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
            suffixes.add(".nxs")
        iterator = root.rglob("*") if recursive else root.glob("*")
        return sorted(
            str(candidate)
            for candidate in iterator
            if candidate.is_file() and candidate.suffix.lower() in suffixes
        )

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
        path.parent.mkdir(parents=True, exist_ok=True)
        if options.output_format == "NumPy":
            np.save(str(path), data, allow_pickle=False)
            return
        if options.output_format == "HDF5":
            with h5py.File(str(path), "w") as handle:
                entry = handle.create_group("entry")
                entry.attrs["NX_class"] = "NXentry"
                dataset = entry.create_dataset("data/data", data=data, compression="gzip", shuffle=True)
                if options.preserve_metadata:
                    dataset.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            return
        if options.output_format == "TIFF":
            from fabio.tifimage import TifImage

            header = (
                {"GIMaP_metadata": json.dumps(metadata, ensure_ascii=False)}
                if options.preserve_metadata
                else {}
            )
            TifImage(data=data, header=header).write(str(path))
            return
        if options.output_format == "CBF":
            from fabio.cbfimage import CbfImage

            cbf_data = data
            if not np.issubdtype(cbf_data.dtype, np.integer):
                cbf_data = np.nan_to_num(
                    cbf_data,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
            header = (
                {"GIMaP_metadata": json.dumps(metadata, ensure_ascii=False)}
                if options.preserve_metadata
                else {}
            )
            CbfImage(data=cbf_data, header=header).write(str(path))
            return
        raise ValueError(f"Unsupported output format: {options.output_format}")

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
