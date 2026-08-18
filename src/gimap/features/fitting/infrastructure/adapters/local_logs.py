"""Local fitting log adapter."""

from __future__ import annotations

from pathlib import Path


class LocalFittingLogRepository:
    def save(self, path: Path, content: str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return target
