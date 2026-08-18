"""Prediction result text-export storage port."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class PredictionExportRepository(Protocol):
    """Persist already formatted prediction output without exposing filesystem APIs."""

    def write_text(self, path: Path, content: str) -> Path: ...

    def write_array(
        self,
        path: Path,
        values: Any,
        *,
        fmt: str,
        header: str,
        comments: str,
    ) -> Path: ...
