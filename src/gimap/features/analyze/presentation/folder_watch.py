"""Report new detector frames in a folder once they are completely written."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence


class FolderWatch:
    """Poll a folder; a frame counts as new when its size is unchanged between polls.

    Detector servers write files incrementally, so a file is only reported
    after two consecutive polls saw the same non-zero size.  ``expand`` maps the
    folder to its frames (for example collapsing NXS module series).
    """

    def __init__(
        self,
        expand: Callable[[Sequence[Path]], list[Path]],
        size_of: Optional[Callable[[Path], int]] = None,
    ):
        self._expand = expand
        self._size_of = size_of or (lambda path: Path(path).stat().st_size)
        self.folder: Optional[Path] = None
        self._known: set[str] = set()
        self._sizes: dict[str, int] = {}

    @property
    def active(self) -> bool:
        return self.folder is not None

    def start(self, folder: Path, *, already_listed: Sequence[Path] = ()) -> None:
        """Watch ``folder``; frames already present or listed are not reported."""
        self.folder = Path(folder)
        self._sizes.clear()
        self._known = {self._key(path) for path in already_listed}
        self._known.update(self._key(path) for path in self._expand([self.folder]))

    def stop(self) -> None:
        self.folder = None
        self._sizes.clear()

    def poll(self) -> list[Path]:
        """New, completely written frames since the last poll (oldest name first)."""
        if self.folder is None or not self.folder.is_dir():
            return []
        ready: list[Path] = []
        for path in self._expand([self.folder]):
            key = self._key(path)
            if key in self._known:
                continue
            try:
                size = int(self._size_of(path))
            except OSError:
                continue
            if size > 0 and self._sizes.get(key) == size:
                self._known.add(key)
                self._sizes.pop(key, None)
                ready.append(path)
            else:
                self._sizes[key] = size
        return ready

    @staticmethod
    def _key(path: Path) -> str:
        return str(Path(path).resolve()).casefold()


__all__ = ["FolderWatch"]
