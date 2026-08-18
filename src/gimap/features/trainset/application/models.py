"""Trainset application requests。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerateTrainsetRequest:
    config: dict[str, Any]
    sample_count: int
    mode: str = "full"
    output_dir: Path | None = None


@dataclass(frozen=True)
class GeneratedTrainset:
    value: Any = None
    files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ModelContractRequest:
    input_shape: tuple[int, int, int]
    output_size: int
    model_config: dict[str, Any]


@dataclass(frozen=True)
class ModelContractResult:
    static_summary: str
    output_shape: tuple[int, ...] | None = None
    trainable_weights: int | None = None
    runtime_error: str | None = None


@dataclass(frozen=True)
class TrainsetPreviewRequest:
    config: dict[str, Any]
    plugin: str
    key: str
    minimum: float
    maximum: float
    compared_text: str
    preview_count: int
    realization: int
    warnings: tuple[str, ...] = ()
    force: bool = False


@dataclass(frozen=True)
class TrainsetWhatIfRequest:
    config: dict[str, Any]
    sampled: dict[str, float]
    realization: int


@dataclass(frozen=True)
class PrepareTrainsetJobRequest:
    config: dict[str, Any]
    workspace: Path
    project_root: Path


@dataclass(frozen=True)
class TrainsetLocalProcessRequest:
    package_dir: Path
    python_executable: Path
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class RegisterTrainsetModelRequest:
    config: dict[str, Any]
    model_path: Path
    modules_root: Path


@dataclass(frozen=True)
class RegisteredTrainsetModel:
    module_name: str
    module_dir: Path


@dataclass(frozen=True)
class TrainsetRemoteJobStatus:
    job_id: str
    state: str
    elapsed: str = ""
    max_rss: str = ""
    exit_code: str = ""
    raw: str = ""
