"""Inspect detector sources before an import; pixel arrays are read only when a frame is opened.

Frames added to a list that holds frames already (``probe=True``) have the first new source's first frame read, so
that frames of another detector (another frame size) are refused before they are listed (``other_detector``)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

OTHER_DETECTOR = ("These frames are {shape} ({detector}); the frames listed are {listed}. Frames of another detector "
                  "are not added to this list: choose Replace, or Analyze ▸ Clear first.")


@dataclass(frozen=True)
class ImportSource:
    path: Path
    frames: int
    shape: Optional[tuple] = None
    """``(rows, columns)`` of the first frame, when it was read (``probe``)."""
    detector: Optional[str] = None


def other_detector(sources, listed_shape, listed_detector: Optional[str] = None) -> Optional[str]:
    """Why ``sources`` may not join a list of frames of ``listed_shape`` (``None``: they may, or nothing is known)."""
    first = next((source for source in sources or () if source.shape is not None), None)
    if first is None or listed_shape is None:
        return None
    shape, listed = tuple(int(v) for v in first.shape[:2]), tuple(int(v) for v in listed_shape[:2])
    if shape == listed:
        return None
    return OTHER_DETECTOR.format(shape=f"{shape[0]}×{shape[1]}", detector=first.detector or "detector not named",
                                 listed=f"{listed[0]}×{listed[1]}" + (f" ({listed_detector})" if listed_detector else ""))


def prepare_frame_import(frames, paths, *, progress=lambda *args: None, cancelled=lambda: False, probe: bool = False):
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
    if probe and result and not cancelled():
        try:
            image = frames.load(result[0].path, 0)
            result[0] = ImportSource(result[0].path, result[0].frames, tuple(int(v) for v in image.data.shape[:2]),
                                     image.detector_name)
        except (OSError, ValueError, RuntimeError):
            pass  # unreadable: opening it explains the reader's error
    return None if cancelled() else result
