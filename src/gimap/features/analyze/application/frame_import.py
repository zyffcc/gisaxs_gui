"""Inspect detector sources before an import; pixel arrays are read only when a frame is opened."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportSource:
    path: Path
    frames: int


def prepare_frame_import(frames, paths, *, progress=lambda *args: None, cancelled=lambda: False):
    sources = frames.expand(paths)
    result = []
    for number, path in enumerate(sources, 1):
        if cancelled():
            return None
        try:
            count = max(1, int(frames.frame_count(path)))
        except (OSError, ValueError, RuntimeError):
            count = 1  # keep an explicitly named unreadable file: opening it explains the reader's error
        result.append(ImportSource(Path(path), count))
        progress(number, len(sources), Path(path).name)
    return None if cancelled() else result
