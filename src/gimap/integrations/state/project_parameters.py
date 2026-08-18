"""JSON adapter preserving the legacy project parameter file format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from src.gimap.shared.file_paths import normalize_path


class JsonProjectParametersRepository:
    def load(self, path: str | Path) -> dict:
        resolved = Path(normalize_path(path))
        with resolved.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("Project parameter file must contain a JSON object")
        return values

    def save(self, path: str | Path, values: Mapping) -> Path:
        resolved = Path(normalize_path(path))
        with resolved.open("w", encoding="utf-8") as stream:
            json.dump(dict(values), stream, indent=4, ensure_ascii=False)
        return resolved
