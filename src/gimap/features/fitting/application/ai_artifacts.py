"""AI fitting artifact commands."""

from __future__ import annotations

from pathlib import Path

from .ports.ai_artifacts import AiFittingArtifactRepository


class ManageAiFittingArtifacts:
    def __init__(self, repository: AiFittingArtifactRepository):
        self._repository = repository

    def has_output(self, output_dir: Path) -> bool:
        return self._repository.has_output(Path(output_dir))

    def append_log(self, output_dir: Path, text: str) -> Path:
        return self._repository.append_log(Path(output_dir), str(text))

    def export_output(
        self,
        output_dir: Path,
        parent_dir: Path,
        timestamp: str,
    ) -> Path:
        source = Path(output_dir).resolve()
        parent = Path(parent_dir).resolve()
        if parent == source or source in parent.parents:
            raise ValueError(
                "Choose a folder outside the reusable AI output directory."
            )
        return self._repository.export_output(source, parent, str(timestamp))
