"""Local filesystem adapter for AI fitting output artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path


class LocalAiFittingArtifactRepository:
    def has_output(self, output_dir: Path) -> bool:
        source = Path(output_dir)
        return source.is_dir() and any(source.iterdir())

    def append_log(self, output_dir: Path, text: str) -> Path:
        target = Path(output_dir) / "gui_run.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(str(text).rstrip() + "\n")
        return target

    def export_output(
        self,
        output_dir: Path,
        parent_dir: Path,
        timestamp: str,
    ) -> Path:
        source = Path(output_dir)
        if not self.has_output(source):
            raise FileNotFoundError("No AI prediction output is available")
        parent = Path(parent_dir)
        destination = parent / f"ai_prediction_{timestamp}"
        suffix = 1
        while destination.exists():
            destination = parent / f"ai_prediction_{timestamp}_{suffix}"
            suffix += 1
        shutil.copytree(source, destination)
        return destination
