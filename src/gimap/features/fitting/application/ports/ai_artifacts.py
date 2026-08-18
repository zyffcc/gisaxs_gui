"""AI fitting output artifact port."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AiFittingArtifactRepository(Protocol):
    def has_output(self, output_dir: Path) -> bool: ...

    def append_log(self, output_dir: Path, text: str) -> Path: ...

    def export_output(
        self,
        output_dir: Path,
        parent_dir: Path,
        timestamp: str,
    ) -> Path: ...
