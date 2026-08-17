"""Format Converter 旧导入路径的兼容门面。

新代码位于 :mod:`src.gimap.features.format_converter`。现有 GUI 和外部调用方
可以继续使用本模块中的名称，迁移期间不需要一次性修改全部 imports。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from src.gimap.features.format_converter.domain.models import (
    OUTPUT_SUFFIXES,
    SUPPORTED_SUFFIXES,
    ConversionJob,
    ConversionOptions,
    ConversionReport,
    ConversionRequest,
    InputSource,
)
from src.gimap.features.format_converter.domain.rules import (
    build_jobs as _build_jobs,
    compact_frame_summary,
    convert_dtype as _convert_dtype,
    parse_custom_frames,
    render_output_stem,
    safe_filename as _safe_filename,
)
from src.gimap.features.format_converter.infrastructure.adapters.local_files import (
    ConversionEngine,
    LocalSourceRepository,
    _json_safe,
)


_repository = LocalSourceRepository()


def inspect_source(path: str | Path) -> InputSource:
    """兼容旧 API：检查本地输入文件。"""
    return _repository.inspect_source(path)


def select_dataset(item: InputSource, dataset_path: str) -> None:
    """兼容旧 API：更新 NXS dataset 选择。"""
    _repository.select_dataset(item, dataset_path)


def scan_folder(
    folder: str | Path,
    *,
    include_cbf: bool = True,
    include_tiff: bool = True,
    include_nxs: bool = True,
    recursive: bool = False,
) -> list[str]:
    """兼容旧 API：扫描包含 detector images 的目录。"""
    return _repository.scan_folder(
        folder,
        include_cbf=include_cbf,
        include_tiff=include_tiff,
        include_nxs=include_nxs,
        recursive=recursive,
    )


def estimate_output(
    sources: Iterable[InputSource],
    options: ConversionOptions,
) -> tuple[int, int]:
    """兼容旧 API：估算输出 image 数量和字节数。"""
    source_list = list(sources)
    request = ConversionRequest(sources=tuple(source_list), options=options)
    return _repository.estimate_output(source_list, request)


def build_jobs(
    sources: Iterable[InputSource],
    options: ConversionOptions,
) -> list[ConversionJob]:
    """兼容旧 API：保留对磁盘已有输出文件的冲突检查。"""
    normalized = replace(
        options,
        destination=str(Path(options.destination).expanduser().resolve()),
    )
    return _build_jobs(
        sources,
        normalized,
        output_exists=lambda candidate: Path(candidate).exists(),
    )


__all__ = [
    "OUTPUT_SUFFIXES",
    "SUPPORTED_SUFFIXES",
    "ConversionEngine",
    "ConversionJob",
    "ConversionOptions",
    "ConversionReport",
    "InputSource",
    "_convert_dtype",
    "_json_safe",
    "_safe_filename",
    "build_jobs",
    "compact_frame_summary",
    "estimate_output",
    "inspect_source",
    "parse_custom_frames",
    "render_output_stem",
    "scan_folder",
    "select_dataset",
]
