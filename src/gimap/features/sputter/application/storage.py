"""The import caches of Analyze and how much room they may take.

Analyze imports the frames of a scan once per request label (``gui_<scan>_<hash>``) into a data folder of the analysis
package and reuses the import afterwards. An import is large (often 0.5–3 GB). Its cache is the label's folder, the
details of its steps beside it (``<label>.trace``) and GIMaP's marker (``<label>.gimap.json``, whose time says when the
cache was last used: every Analyze that uses it touches it).

The caches live in a local folder (``%LOCALAPPDATA%\\GIMaP`` on Windows, ``~/Library/Caches/GIMaP`` on macOS,
``~/.cache/gimap`` elsewhere); those GIMaP made before in the user data folder stay usable and are listed as such
(``place`` ``user data``). When the caches together exceed the limit (``DEFAULT_LIMIT_BYTES``, changeable; 0: none),
the least recently used are removed first — never the one in use. Results are never removed automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

GB = 1 << 30
DEFAULT_LIMIT_GB = 5
DEFAULT_LIMIT_BYTES = DEFAULT_LIMIT_GB * GB
PLACE_LOCAL, PLACE_USER = "local", "user data"
DAY_S = 86400.0


@dataclass(frozen=True)
class ImportCache:
    """The cache of one import: its label, the data folder that holds it, its size (bytes), when it was last used
    (seconds since the epoch) and where it is (``local`` or ``user data``); ``paths``: what belongs to it."""

    label: str
    folder: Path
    size: int = 0
    last_used: float = 0.0
    place: str = PLACE_LOCAL
    paths: tuple = field(default=(), compare=False)

    @property
    def key(self) -> str:
        return str(Path(self.folder) / self.label)


class CacheStore(Protocol):
    def caches(self) -> list:
        """Every import cache GIMaP made (``ImportCache``), local and in the user data folder."""
        ...

    def remove(self, cache: ImportCache) -> None:
        """Remove a cache (its folder, its details and its marker); raises ``OSError`` when it cannot."""
        ...

    def folder_size(self, folder: Path) -> int:
        """The size of everything in a folder (bytes; 0 when it is not there)."""
        ...


def total_size(caches) -> int:
    return sum(int(cache.size) for cache in caches)


def over_limit(caches, limit: Optional[int], keep=()) -> list:
    """The caches to remove so that the rest fit in ``limit`` bytes, the least recently used first, never a label in
    ``keep`` (empty when they fit, or without a limit: None or 0)."""
    if not limit or limit <= 0:
        return []
    kept = set(keep)
    total = total_size(caches)
    chosen = []
    for cache in sorted(caches, key=lambda item: item.last_used):
        if total <= limit:
            break
        if cache.label in kept:
            continue
        chosen.append(cache)
        total -= cache.size
    return chosen


def unused_for(caches, days: float, now: float, keep=()) -> list:
    """The caches not used for ``days`` days or more (never a label in ``keep``), the least recently used first."""
    kept = set(keep)
    return sorted((cache for cache in caches if cache.label not in kept and now - cache.last_used >= days * DAY_S),
                  key=lambda cache: cache.last_used)


def prune(store: CacheStore, limit: Optional[int], keep=()) -> list:
    """Remove the caches over the limit (``over_limit``); returns those removed (a cache that cannot be removed is
    skipped)."""
    removed = []
    for cache in over_limit(store.caches(), limit, keep):
        try:
            store.remove(cache)
        except OSError:
            continue
        removed.append(cache)
    return removed


__all__ = ["CacheStore", "DAY_S", "DEFAULT_LIMIT_BYTES", "DEFAULT_LIMIT_GB", "GB", "ImportCache", "PLACE_LOCAL",
           "PLACE_USER", "over_limit", "prune", "total_size", "unused_for"]
