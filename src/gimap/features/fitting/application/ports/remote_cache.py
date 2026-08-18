"""Remote detector-file cache port."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class RemoteFileCachePort(Protocol):
    def default_directory(self) -> str: ...

    def display_directory(self, cache_dir: str) -> str: ...

    def resolve_directory(self, cache_dir: str) -> Path: ...

    def is_remote(self, path: str) -> bool: ...

    def target_path(self, source_path: str, cache_dir: str) -> Path: ...

    def prepare(
        self,
        source_path: str,
        cache_dir: str,
        max_gb: float,
        *,
        on_progress: Callable[[int, str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Path: ...

    def clear(self, cache_dir: str) -> int: ...
