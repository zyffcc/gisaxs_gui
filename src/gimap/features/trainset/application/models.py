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
