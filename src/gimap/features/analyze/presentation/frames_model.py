"""Frame part of the Analyze view model: which frames one analysis sums, batches, watching.

A multi-frame file (NeXus) contributes all its frames; a single-frame file
(CBF, TIFF) is one frame.  With "Sum N frames" the frame shown is added to
the next N−1 frames: the following frames of the same file, or the
following listed files of the same type.  Batch export and watching work
through the same sequence in groups of N.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ..application import AnalysisRequest, FrameRef

MAX_SUM = 10000


class FramesModelMixin:
    """Needs ``state`` (``files``, ``sum_count``), ``_frames`` and ``request_for``."""

    def _init_frames(self) -> None:
        self._frame_counts: dict[str, int] = {}
        self._watch_pending: list[FrameRef] = []

    def frame_count(self, path: Path) -> int:
        """Frames in a file (cached; a NeXus file is read once)."""
        key = str(path).casefold()
        if key not in self._frame_counts:
            try:
                self._frame_counts[key] = max(1, int(self._frames.frame_count(path)))
            except (OSError, ValueError, KeyError):
                self._frame_counts[key] = 1
        return self._frame_counts[key]

    def listed_frames(self) -> int:
        """Frames listed, without reading any file: known counts, one for a file not read yet."""
        return sum(self._frame_counts.get(str(path).casefold(), 1) for path in self.state.files)

    def remember_frame_count(self, path: Path, count: int) -> None:
        self._frame_counts[str(path).casefold()] = max(1, int(count))

    def _forget_frames(self, path: Path) -> None:
        """A file left the list: its frame count, and its frames waiting for a group of the folder watch."""
        key = str(path).casefold()
        self._frame_counts.pop(key, None)
        self._watch_pending = [frame for frame in self._watch_pending if str(frame[0]).casefold() != key]

    @property
    def sum_count(self) -> int:
        return self.state.sum_count

    def set_sum_count(self, count: int) -> None:
        """Sum this many consecutive frames into each analysis (1: no summing)."""
        self.state.sum_count = max(1, min(MAX_SUM, int(count)))

    def summed_frames_for(self, path: Path, frame_index: int = 0) -> tuple[FrameRef, ...]:
        """The frames added to ``(path, frame_index)`` for the current sum (maybe fewer at the end)."""
        extra = self.state.sum_count - 1
        if extra <= 0:
            return ()
        path = Path(path)
        count = self.frame_count(path)
        if count > 1:
            last = min(count, frame_index + 1 + extra)
            return tuple((path, index) for index in range(frame_index + 1, last))
        following: list[FrameRef] = []
        files = self.state.files
        keys = [str(item).casefold() for item in files]
        try:
            position = keys.index(str(path).casefold())
        except ValueError:
            return ()
        for candidate in files[position + 1 :]:
            if len(following) >= extra:
                break
            if candidate.suffix.lower() != path.suffix.lower() or self.frame_count(candidate) > 1:
                break
            following.append((Path(candidate), 0))
        return tuple(following)

    def all_frames(self, paths: Optional[Sequence[Path]] = None) -> list[FrameRef]:
        """Every frame of the listed files (or of ``paths``), in list order."""
        frames: list[FrameRef] = []
        for path in self.state.files if paths is None else paths:
            frames.extend((Path(path), index) for index in range(self.frame_count(Path(path))))
        return frames

    def _group_key(self, frame: FrameRef) -> tuple:
        """Frames sum only with frames of the same kind: one multi-frame file, or one file type."""
        path = Path(frame[0])
        return path.suffix.lower(), str(path).casefold() if self.frame_count(path) > 1 else None

    def _groups(
        self, frames: Sequence[FrameRef], *, keep_open_tail: bool = False
    ) -> tuple[list[list[FrameRef]], list[FrameRef]]:
        """Consecutive frames in groups of ``sum_count``; a change of kind closes a group early.

        With ``keep_open_tail`` a last, incomplete group is returned separately
        (more frames may still join it) instead of as a group.
        """
        size = self.state.sum_count
        groups: list[list[FrameRef]] = []
        current: list[FrameRef] = []
        key = None
        for frame in frames:
            frame_key = self._group_key(frame)
            if current and (frame_key != key or len(current) == size):
                groups.append(current)
                current = []
            key = frame_key
            current.append(frame)
        if current and not (keep_open_tail and len(current) < size):
            groups.append(current)
            current = []
        return groups, current

    def _requests(self, groups: Sequence[Sequence[FrameRef]]) -> list[AnalysisRequest]:
        return [
            self.request_for(group[0][0], group[0][1], summed=tuple(group[1:])) for group in groups
        ]

    def batch_requests(self) -> list[AnalysisRequest]:
        """One request per frame of every listed file, or per group of N frames when summing."""
        groups, _ = self._groups(self.all_frames())
        return self._requests(groups)

    def watch_requests(self, added: Sequence[Path]) -> list[AnalysisRequest]:
        """Requests for newly watched frames; with summing, only complete groups of N."""
        self._watch_pending.extend(self.all_frames(added))
        groups, self._watch_pending = self._groups(self._watch_pending, keep_open_tail=True)
        return self._requests(groups)

    def reset_watch_groups(self) -> None:
        self._watch_pending = []


__all__ = ["FramesModelMixin", "MAX_SUM"]
