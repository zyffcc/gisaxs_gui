"""Sources part of the Analyze view model: which detector sources are listed, and the folder watch.

A *source* is what one row of the file list stands for: a CBF / TIFF / EDF frame, or a NeXus / HDF5
source of one or many frames (one file, the module files of a detector, a master with its data files).
The frame source describes it (``FramesModelMixin.describe``); this mixin keeps the list right when
what is on disk changes:

* a source is one row, also when its files arrive one after the other. A module set first met without
  its first module takes the name of that module when it arrives, in the same row (``replaced``). A
  master met after its data files takes the row of the first of them, and the rows of the others leave
  the list (``dropped``): no frame is listed, counted or exported twice;
* every source is counted when it is listed, so the frame total is right before any frame is read; a
  file that cannot be read is listed all the same (one row, no count), and reading it says why;
* the folder watch reports new sources, the listed ones whose files changed (``watch_changed``) and the
  frames they gained (``watch_grown``). The frames go to ``watch_requests`` once each, and only frames
  that hold every part of the detector (``watch_held`` names the sources that wait):

  - a module set with a module file missing or not readable yet waits for it;
  - a set of module files that may still be arriving waits until it rests (``SET_RESTS``): a set
    without a gap in its numbering looks whole after every file, so nothing but time, and the other
    sets of its folder, say whether more follow;
* a source that changes at every look (a detector writing a frame a second for minutes) is followed
  while it grows: at each look the frames it already held at the look before are handed out (for a set
  of module files: those every module held). They have been in its files for a whole look; the newest
  ones, which may still be being written, follow at the next look, and the rest when the source rests.
  Meanwhile a series run takes of it only the frames handed out (``frames_to_run``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

from ..application import AnalysisRequest, FrameRef
from ..application.ports import MODULE_SET, module_files, parts_missing
from .frames_model import source_key
from .frame_selection import indices_between

SET_RESTS = 5
"""Looks of the folder watch (two seconds apart) for which a set of module files must stay as it is before
its frames are handed out: files copied one after the other (rsync, a slow disk) arrive seconds apart. A
set with as many module files as another set of its folder does not wait: it is whole."""
SET_RESTS_FEWER = 15
"""The same for a set with fewer module files than another set of its folder: more are expected, but a
detector may have had a module switched off, so in the end it is handed out as it is."""


class SourceStamp(NamedTuple):
    """What a NeXus / HDF5 source was at one look of the folder watch."""

    signature: tuple
    """``(file, time, size)`` of every file it consists of."""
    frames: int
    """Frames it held (a set of module files: those every module held)."""
    pending: int
    """Module files that could not be read."""


def source_stamp(frames) -> Optional[Callable[[Path], object]]:
    """What the folder watch compares besides the size of a source: for a NeXus / HDF5 source the stamp of
    every file it consists of and its frames (``None``: the frame source cannot describe its sources). The
    function only asks the frame source, so the watch may call it on another thread."""
    describe = getattr(frames, "describe", None)
    if describe is None:
        return None

    def stamp(path: Path):
        layout = describe(path)
        return None if layout is None else SourceStamp(layout.signature, int(layout.frames), len(layout.pending))

    return stamp


class SourcesModelMixin:
    """Needs ``state.files``, ``_frames``, ``watch`` and the frame part (``frames_model.py``)."""

    def _init_sources(self, *, keep_watch: bool = False) -> None:
        """``keep_watch``: the list was cleared while a folder is watched. What the watch handed out stays
        known, so a file that grows afterwards comes back with its new frames only."""
        self.already_listed: list[int] = []
        """After ``add_paths``: list positions of the sources asked for that were listed already."""
        self.replaced: list[tuple[int, Path, Path]] = []
        """After ``add_paths`` / ``poll_watch``: ``(row, path before, path now)`` of the rows whose source is
        now named by another of its files (``row``: its place in the list as it is now)."""
        self.dropped: list[tuple[int, Path]] = []
        """After ``add_paths`` / ``poll_watch``: ``(row, path)`` of the rows that left the list because their
        file belongs to another listed source (``row``: its place before, the highest first, so that the rows
        can be taken off a list in this order)."""
        self.watch_grown: list[tuple[Path, int]] = []
        """After ``poll_watch``: ``(source, frames gained)`` of the listed sources that grew."""
        self.watch_changed: list[Path] = []
        """After ``poll_watch``: the listed sources whose files changed and have rested since, whether they
        gained frames or not (a file written again, a module file that arrived, a writer that closed its file)."""
        self.watch_waiting: list[tuple[Path, str]] = []
        """After ``poll_watch``: ``(file, why)`` of what cannot be read yet (asked again at the next poll)."""
        self.watch_held: list[Path] = []
        """After ``poll_watch`` / ``watch_requests``: sources whose frames wait for a part of the detector."""
        self._watch_fresh: list[FrameRef] = []
        self._arriving: dict[str, tuple[Path, tuple, int]] = {}
        """Source → (its path, its module files, the look at which they were first as they are): the sets of
        module files that have not rested long enough yet."""
        self._had: dict[str, tuple[Path, tuple, int]] = {}
        """After a look, for the sources that do not rest: source → (its path, its files, the frames it held)
        at the look before."""
        if not keep_watch:
            self._watch_done: dict[str, tuple[tuple, int]] = {}
            """Source → (the files its frames were read from, frames handed to ``watch_requests`` or there
            before the watch began)."""
            self._set_sizes: dict[str, tuple[str, int]] = {}
            """Module set that was handed out → (its folder, how many module files it has)."""
            self._look = 0

    # -- the list ------------------------------------------------------------------------

    def add_paths(self, paths: Sequence[str | Path]) -> list[Path]:
        """Add frames from files/folders; returns the newly added ones.

        ``already_listed`` then holds the list positions of the frames among ``paths`` that were listed before;
        ``replaced`` and ``dropped`` the rows that changed because files now belong together.
        """
        added, self.already_listed = self._list_sources(self._frames.expand(paths))
        return added

    def add_import_sources(self, paths: Sequence[Path]) -> list[Path]:
        """Commit canonical sources already inspected by the import worker, without expanding again."""
        added, self.already_listed = self._list_sources(paths)
        return added

    def _list_sources(self, sources: Sequence[Path]) -> tuple[list[Path], list[int]]:
        """Append the sources not listed yet: ``(added, rows of those listed already)``. A listed file that is
        a part of one of them is that source: the first such row takes the source's name (``replaced``), and
        any further rows of its parts leave the list (``dropped``)."""
        files = self.state.files
        rows = {source_key(path): index for index, path in enumerate(files)}
        added: list[Path] = []
        listed: list[int] = []
        taken: dict[int, int] = {}  # the row of a part → the row of the source it belongs to
        self.replaced, self.dropped = [], []
        for path in map(Path, sources):
            key = source_key(path)
            row = rows.get(key)
            parts = sorted({rows[part] for part in map(source_key, self._parts(path)) if part in rows} - {row, *taken})
            if row is None and parts:
                row = parts.pop(0)
                restricted = any(self.import_selection(files[part]) is not None for part in [row, *parts])
                before, handed = files[row], self._handed_as_parts(path)
                if row == self.state.current_index:  # the frame shown stays the same frame of the source
                    self.state.frame_index = self._frame_within(path, before, self.state.frame_index)
                files[row] = path
                rows[key] = row
                self._renamed(before, path, handed, selection_restricted=restricted)
                self.replaced.append((row, before, path))
            elif row is not None and parts:
                handed = self._handed_as_parts(path)
                if handed is not None:
                    self._watch_done[key] = handed
            if row is None:
                rows[key] = len(files)
                files.append(path)
                added.append(path)
                continue
            taken.update((part, row) for part in parts)
            listed.append(row)
        if taken:
            listed = self._drop_rows(taken, listed)
        # Counted as it is listed (its layout was read to list it), and again when it is asked for again: it
        # may have grown, or been completed by another file, since.
        for path in [*added, *(files[row] for row in listed)]:
            self.frame_count(path)
        return added, listed

    def _parts(self, path: Path) -> tuple:
        """Every file ``path`` stands for (``FrameSource.parts``; for a source that cannot say: its layout)."""
        parts = getattr(self._frames, "parts", None)
        if parts is not None:
            try:
                return tuple(parts(path))
            except Exception:  # noqa: BLE001 - a source that cannot be asked has no parts
                return ()
        layout = self.describe(path)
        return () if layout is None else (*layout.members, *layout.pending)

    def _renamed(self, before: Path, now: Path, handed: Optional[tuple] = None,
                 *, selection_restricted: bool = False) -> None:
        """A row's source is named by another of its files: what was kept under the old name moves over
        (``handed``: what the new name has handed out already, when that is known better)."""
        old, new = source_key(before), source_key(now)
        self._migrate_import_selection(before, now, restrict=selection_restricted)
        self._frame_counts.pop(old, None)
        self.frame_count(now)
        moved = self._watch_done.pop(old, None)
        if handed is not None or moved is not None:
            self._watch_done[new] = handed if handed is not None else moved
        if old in self._arriving:
            self._arriving[new] = (now, *self._arriving.pop(old)[1:])
        self._set_sizes.pop(old, None)
        self._watch_pending = [frame for frame in self._watch_pending if source_key(frame[0]) != old]

    def _handed_as_parts(self, path: Path) -> Optional[tuple]:
        """For a master that takes the rows of its data files while a folder is watched: its files, and how
        many of its frames were handed out already under the names of those data files (its frames are counted
        through its data files in order, so these are its first frames; a frame still waiting for its group of
        a sum is not among them). ``None``: no master, or nothing is watched."""
        layout = self.describe(path)
        if not self.watch.active or layout is None or layout.kind != "chunked_master":
            return None
        handed = 0
        if len(layout.member_frames) == len(layout.members) - 1:
            for member, frames in zip(layout.members[1:], layout.member_frames):
                key = source_key(member)
                waiting = sum(source_key(frame[0]) == key for frame in self._watch_pending)
                done = self._watch_done.get(key, ((), 0))[1] - waiting
                handed += max(0, min(done, int(frames)))
                if done < frames:
                    break
        own = self._watch_done.get(source_key(path), ((), 0))[1]
        return tuple(str(member) for member in layout.members), max(own, handed)

    def _frame_within(self, source: Path, part: Path, frame_index: int) -> int:
        """The frame of ``source`` that frame ``frame_index`` of its file ``part`` is: the same number in a set
        of module files; in a master, counted on through the data files before ``part`` (its first frame when
        that cannot be told)."""
        layout = self.describe(source)
        if layout is not None and layout.kind == MODULE_SET:
            return min(int(frame_index), max(0, int(layout.frames) - 1))
        if layout is not None and len(layout.member_frames) == len(layout.members) - 1:
            before = 0
            for member, frames in zip(layout.members[1:], layout.member_frames):
                if source_key(member) == source_key(part):
                    return min(before + int(frame_index), max(0, int(layout.frames) - 1))
                before += int(frames)
        return 0

    def _drop_rows(self, taken: dict[int, int], listed: Sequence[int]) -> list[int]:
        """Take the rows ``taken`` (row → the row of the source its file belongs to) off the list; returns
        ``listed`` as the rows are numbered afterwards. When the file shown is one of them, its source becomes
        the file shown, at the same frame of the series (``analysis`` is ``None``: it was read from the part
        alone)."""
        state = self.state
        kept = [row for row in range(len(state.files)) if row not in taken]
        now = {row: place for place, row in enumerate(kept)}
        shown = state.current_index
        if shown in taken:
            state.frame_index = self._frame_within(state.files[taken[shown]], state.files[shown], state.frame_index)
        for row in sorted(taken, reverse=True):
            self._migrate_import_selection(state.files[row], state.files[taken[row]], merge=True)
            self._forget_source(state.files[row], for_good=True)
            self.dropped.append((row, state.files[row]))
        if shown in taken:
            state.current_index, state.analysis = now[taken[shown]], None
        elif state.current_index >= 0:
            state.current_index = now[state.current_index]
        state.files[:] = [state.files[row] for row in kept]
        self.replaced = [(now[row], before, path) for row, before, path in self.replaced]
        return [now[row] for row in listed]

    def _forget_source(self, path: Path, *, for_good: bool = False) -> None:
        """A file left the list: its frame count, its frames waiting for a group, its place among the sets
        that are arriving. What the folder watch handed out of it stays known while the folder is watched (it
        comes back with its new frames only when it grows), unless it left ``for_good`` (it is a part of
        another source now)."""
        key = source_key(path)
        self._forget_frames(path)
        self._arriving.pop(key, None)
        if for_good or not self.watch.active:
            self._watch_done.pop(key, None)
            self._set_sizes.pop(key, None)

    # -- the folder watch ------------------------------------------------------------------

    def start_watch(self, folder: Path) -> None:
        """Watch ``folder`` for frames written from now on (existing ones are listed too)."""
        self.add_paths([folder])
        self.reset_watch_groups()
        self._watch_fresh, self.watch_grown, self.watch_changed = [], [], []
        self.watch_waiting, self.watch_held = [], []
        self._watch_done, self._set_sizes, self._arriving, self._had, self._look = {}, {}, {}, {}, 0
        self.watch.start(Path(folder), already_listed=self.state.files)
        for path in self.state.files:  # what is there now is not new: only frames written from now on
            self._new_frames(path, there=True)

    def stop_watch(self) -> None:
        self.watch.stop()

    def poll_watch(self) -> list[Path]:
        """Append completely written new frames to the list and return them. ``watch_changed``, ``watch_grown``,
        ``watch_waiting`` and ``watch_held`` then say what else this look at the folder found; ``replaced``
        and ``dropped`` which rows changed because files now belong together."""
        return self.take_watch(self.watch.scan())

    def take_watch(self, scan) -> list[Path]:
        """``poll_watch`` for a look at the folder made beforehand (``watch.scan()``, on any thread)."""
        ready = self.watch.take(scan)
        self._look += 1
        # A source that does not rest is followed while it grows, as soon as it held a frame at the look before.
        # One that is not listed is taken up only when it gained a frame since: it grows by appending. (A file
        # that is written again in place at every look holds other frames now than then: it waits until it rests.)
        self._had = self._held_before()
        listed = {source_key(path) for path in self.state.files}
        following = [path for key, (path, _files, frames) in self._had.items()
                     if frames > 0 and (key in listed or self._frames_now(path) > frames)]
        # A known source that changed is listed like a new one when it is not listed (any more): the list was
        # cleared, or the file was taken off it, while the folder is watched.
        changed = list(self.watch.changed)
        added, _listed = self._list_sources([*ready, *changed, *following])
        for path in following:
            self.watch.follow(path)
        self.watch_waiting = list(self.watch.waiting)
        self.watch_grown, self.watch_held = [], []
        rows = {source_key(path): path for path in self.state.files}
        new = {source_key(path) for path in added}
        self.watch_changed = [rows[key] for key in dict.fromkeys(map(source_key, changed)) if key in rows and key not in new]
        again = [*self.watch_changed, *(now for _row, _before, now in self.replaced),
                 *(path for path, _files, _look in self._arriving.values()), *following]
        for key in dict.fromkeys(map(source_key, again)):
            if key not in rows or key in new:  # (the new ones are asked for by ``watch_requests(added)``)
                continue
            frames = self._new_frames(rows[key])
            if frames:
                self._watch_fresh.extend(frames)
                self.watch_grown.append((rows[key], len(frames)))
        return added

    def _held_before(self) -> dict[str, tuple[Path, tuple, int]]:
        """What the sources that changed again at this look (``watch.moving``) were at the look before: their
        files and the frames they held. Only sources whose frames are known then (NeXus / HDF5), and none
        that had a module file it could not read."""
        had: dict[str, tuple[Path, tuple, int]] = {}
        for path, state in self.watch.moving:
            stamp = state[1] if isinstance(state, tuple) and len(state) > 1 else None
            if isinstance(stamp, SourceStamp) and not stamp.pending:
                had[source_key(path)] = (Path(path), tuple(item[0] for item in stamp.signature), int(stamp.frames))
        return had

    def _frames_now(self, path: Path) -> int:
        layout = self.describe(path)
        return int(layout.frames) if layout is not None else 0

    def growing(self, path: Path) -> bool:
        """While a folder is watched: the source was seen changed at the last look, so its newest frames may
        still be being written (``followed_frame`` is the newest one known to be whole). ``False`` once it has
        rested for a look, and when nothing is watched."""
        return bool(self.watch.active) and not self.watch.rests(path)

    def followed_frame(self, path: Path) -> Optional[int]:
        """The newest frame of a watched source that is known to be whole: the last one the watch handed out
        (or that was there when the watch began). ``None``: none so far. While a source grows this is the frame
        to show; the newest frames of its file may still be being written."""
        _files, done = self._watch_done.get(source_key(path), ((), 0))
        return done - 1 if done > 0 else None

    def frames_to_run(self, path: Path) -> int:
        """Frames of a listed source that a series run started now may take (Batch Export, a Series map, Send
        Series): every frame it holds, but of a source that grows while the folder is watched only those the
        watch handed out (``followed_frame`` + 1; none so far: 0). Its newest frames may still be being written."""
        count = self.frame_count(path)
        if not self.growing(path):
            return count
        newest = self.followed_frame(path)
        return min(count, 0 if newest is None else newest + 1)

    def whole_frames(self, path: Path) -> int:
        """Frames of a source that hold every part of the detector: none while a module file of its set is
        missing or cannot be read yet, and none of a NeXus / HDF5 file that never could be read."""
        layout = self.describe(path)
        if layout is not None:
            return 0 if parts_missing(layout) else int(layout.frames)
        try:  # another format, or a frame source that does not describe its sources
            return max(0, int(self._frames.frame_count(path)))
        except Exception:  # noqa: BLE001 - a file that cannot be read holds no whole frame
            return 0

    def settling(self, path: Path) -> Optional[tuple[int, Optional[int]]]:
        """While the frames of a set of module files wait for the set to rest: ``(module files so far, how
        many the other sets of its folder have)``, the second ``None`` when no other set says. ``None``: the
        source does not wait for that."""
        known = self._arriving.get(source_key(path))
        if known is None:
            return None
        others = self._other_sets(Path(path))
        return len(known[1]), (max(others) if others else None)

    def _other_sets(self, path: Path) -> list[int]:
        """How many module files the other sets of the folder of ``path`` have (those handed out, or there
        when the watch began)."""
        key, folder = source_key(path), source_key(Path(path).parent)
        return [count for other, (place, count) in self._set_sizes.items() if other != key and place == folder]

    def _rested(self, path: Path, key: str, layout, files: tuple, *, there: bool) -> bool:
        """Whether a source may be handed out as far as its module files go: every source that is no set of
        module files, a set that was there when the watch began (``there``), one whose frames were handed out
        from these very files, and one that has stayed as it is for long enough (``SET_RESTS``; not at all when
        it has as many module files as another set of its folder, ``SET_RESTS_FEWER`` with fewer). A set with a
        part missing does not rest: it is looked at again at every look until the part is there. What has to
        stay as it is are the module FILES of the set, not what is in them: a set that grows in every module at
        every look is handed out as far as ``_new_frames`` lets it."""
        count = module_files(layout)
        if there or count == 0 or self._watch_done.get(key, ((), 0))[0] == files:
            self._arriving.pop(key, None)
            return True
        known = self._arriving.get(key)
        parts = tuple(sorted({*files, *(str(part) for part in layout.pending)}))
        if known is None or known[1] != parts:
            known = self._arriving[key] = (path, parts, self._look)
        if parts_missing(layout):
            return False
        others = self._other_sets(path)
        needed = SET_RESTS if not others else (0 if count >= max(others) else SET_RESTS_FEWER)
        if self._look - known[2] < needed:
            return False
        del self._arriving[key]
        return True

    def _new_frames(self, path: Path, *, there: bool = False) -> list[FrameRef]:
        """The whole frames of a watched source not handed out before (each frame once); none while the source
        waits for a part of the detector (``watch_held``). A set of module files that is made of other files
        than when its frames were handed out starts again: they were stitched without that part. So does a
        source that holds fewer frames than were handed out: it was written again from its start.

        A source that was seen changed at this look may still be being written: only the frames it already
        held at the look before are handed out (``_had``; none when it was made of other files then). Its
        newest frames follow at the next look, or when it rests."""
        path = Path(path)
        key = source_key(path)
        self.frame_count(path)  # the count everything else uses follows too
        layout = self.describe(path)
        files = tuple(str(member) for member in layout.members) if layout is not None else (str(path),)
        rested = self._rested(path, key, layout, files, there=there)
        if (layout is not None and parts_missing(layout)) or not rested:
            self.watch_held.append(path)  # (what was handed out before stays known: ``_watch_done`` is not touched)
            return []
        whole = int(layout.frames) if layout is not None else self.whole_frames(path)  # (no part is missing)
        before, done = self._watch_done.get(key, (files, 0))
        again = whole < done
        if (before != files and layout is not None and layout.kind == MODULE_SET) or again:
            done = 0
        if not there and not self.watch.rests(path):
            _path, files_then, frames_then = self._had.get(key, (path, files, done))
            held = frames_then if files_then == files and not again else 0
            whole = max(done, min(whole, held))
        self._watch_done[key] = (files, whole)
        if layout is not None and layout.kind == MODULE_SET:
            self._set_sizes[key] = (source_key(path.parent), len(files))
        if there:
            return []  # marking an existing million-frame source does not allocate its frame references
        selected = self.selected_frame_indices(path)
        return [(path, index) for index in indices_between(selected, done, whole)]

    def watch_requests(self, added: Sequence[Path] = (), *, wanted: bool = True) -> list[AnalysisRequest]:
        """Requests for the frames the folder watch found since the last call: the frames of the new sources
        ``added`` and those the listed sources gained (``poll_watch``); with summing, only complete groups of N.
        ``wanted=False`` takes the frames off the queue without asking for them (nothing is exported now, and
        they are not exported later either)."""
        frames, self._watch_fresh = self._watch_fresh, []
        for path in added:
            frames.extend(self._new_frames(Path(path)))
        return self._watch_groups(frames) if wanted else []


__all__ = ["SET_RESTS", "SET_RESTS_FEWER", "SourceStamp", "SourcesModelMixin", "source_stamp"]
