"""Format Converter 的 framework-neutral 请求、结果和值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rules import compact_frame_summary


SUPPORTED_SUFFIXES = {".nxs": "NXS", ".cbf": "CBF", ".tif": "TIFF", ".tiff": "TIFF"}
OUTPUT_SUFFIXES = {"TIFF": ".tif", "CBF": ".cbf", "HDF5": ".h5", "NumPy": ".npy"}


@dataclass
class InputSource:
    path: str
    file_type: str
    frame_count: int = 1
    dataset_path: str | None = None
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


@dataclass
class ConversionOptions:
    output_format: str
    destination: str
    naming_template: str = "{source}_{frame:06d}"
    add_suffix: bool = True
    preserve_values: bool = True
    data_mode: str = "original"
    preserve_metadata: bool = True
    write_sidecar: bool = True
    single_metadata_file: bool = True
    container: bool = False

    @property
    def suffix(self) -> str:
        try:
            return OUTPUT_SUFFIXES[self.output_format]
        except KeyError as exc:
            raise ValueError(f"Unsupported output format: {self.output_format}") from exc


@dataclass(frozen=True)
class ConversionRequest:
    sources: tuple[InputSource, ...]
    options: ConversionOptions


@dataclass
class ConversionJob:
    source: InputSource
    frame_index: int
    output_path: str


@dataclass
class ConversionResult:
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


# 旧名称在迁移期间保留。
ConversionReport = ConversionResult
