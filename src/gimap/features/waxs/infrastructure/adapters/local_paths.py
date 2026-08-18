"""Local filesystem adapter for WAXS workspace path operations."""

from __future__ import annotations

import os
from pathlib import Path

from src.gimap.shared.file_paths import normalize_path


class LocalWaxsPathAdapter:
    def normalize(self, path: str | Path) -> str:
        return normalize_path(path)

    def current_directory(self) -> str:
        return os.getcwd()

    def is_directory(self, path: str | Path) -> bool:
        return os.path.isdir(path)
