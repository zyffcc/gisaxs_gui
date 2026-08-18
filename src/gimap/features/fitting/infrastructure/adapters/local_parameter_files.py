"""Local filesystem adapter for fitting parameter snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class LocalFittingParameterFileRepository:
    def save_snapshot(self, path: Path, values) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(dict(values), handle, indent=4, ensure_ascii=False)
        return target

    def load_snapshot(self, path: Path) -> dict[str, object]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def copy(self, source: Path, destination: Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source), target)
        return target
