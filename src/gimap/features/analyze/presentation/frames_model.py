"""Frame part of the Analyze view model: which frames one analysis sums, batches, watching.

A multi-frame source (NeXus / HDF5: one file, a set of module files, a master
with its data files) contributes all its frames; a single-frame file (CBF, TIFF)
is one frame.  With "Sum N frames" the frame shown is added to
the next N−1 frames: the following frames of the same file, or the
following listed files of the same type.  Batch export and watching work
through the same sequence in groups of N.

How many frames a source holds is asked of the source every time (``frame_count``:
it may still grow; asking costs ``os.stat`` while nothing changed).  The last
count that could be read is kept for a source that cannot be read now and for
totals that must not touch the disk (``listed_frames``); a read that fails is
never remembered as a count.

A frame kept in memory is used again only while its files are as they were when
it was read (``files_stamp``, ``ReadStamps``): a module file that arrived since,
a file written again under the same name, and the frame is read again.

One cut for a series (``pin_series_cuts``, on unless turned off): where the cut of a
frame is automatic — the Yoneda rows and the centre columns of a GISAXS frame, the
strongest ring of a GIWAXS frame — every frame of a series run (``batch_requests``:
Series map, Batch Export, Send Series; ``_watch_groups``: a watched folder) is cut
where the frame on screen is (``series_cut``), so a row of a map differs from the next
by the sample and not by where it was cut. A cut the person set is the same for every
frame anyway and is never replaced.
"""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional, Sequence

from ..application import GISAXS, AnalysisRequest, FrameRef
from .frame_selection import ImportSelectionMixin
from .frame_sequences import BatchGroups, BatchRequests, FrameNumbers, FrameSequence, SumGroups
from .geometry_model import SECTION

MAX_SUM = 10000
PIN_KEY = "pin_series_cuts"
"""Settings ▸ ``analyze.pin_series_cuts``: one cut for every frame of a series (on unless turned off)."""
CUT_ROWS = "horizontal cut: rows {low}–{high} for every frame"
CUT_EACH = "horizontal cut: found again in each frame"
RING_Q = "I(χ) ring: q {low}–{high} Å⁻¹ for every frame"
RING_EACH = "I(χ) ring: the strongest ring of each frame"
CUT_BY_YOU = "cuts as set by you, the same for every frame"
CUT_TEXTS = (CUT_ROWS, CUT_EACH, RING_Q, RING_EACH, CUT_BY_YOU)
"""The sentences of ``series_cut_text`` (rows: the pixel rows the band covers, both ends included, as the Cuts
step counts them; ``series_cut`` keeps the edges of the band)."""
BY_USER, BY_AUTOMATIC = "user", "automatic"
"""Who set the Sum N in force (``sum_set_by``): the person, or Run Automatic Analysis for its own result."""


@dataclass(frozen=True)
class SumGroup:
    """One request of a series run: the frame it is named after and how many frames it sums."""

    first: FrameRef
    frames: int
    size: int
    """Frames a complete group has (the Sum N in force)."""

    @property
    def complete(self) -> bool:
        """``False`` for the last group of a source whose frames are no multiple of N: a sum of fewer frames."""
        return self.frames >= self.size


def _pair(values) -> Optional[tuple[float, float]]:
    try:
        low, high = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return (low, high)


def with_cut(request: AnalysisRequest, cut: Optional[dict]) -> AnalysisRequest:
    """``request`` cut where ``cut`` says (``series_cut``), wherever its own cut is automatic; the widths of
    the bands stay the request's (those the frame on screen was cut with). A request with a ``series_cut``
    field is also given ``cut`` itself, so that the frame's record can say where its cut comes from (the
    reduction alone calls any given row “manual”)."""
    if not cut:
        return request
    named = {"series_cut": dict(cut)} if any(item.name == "series_cut" for item in fields(request)) else {}
    if cut.get("technique") == GISAXS:
        gisaxs, rows, columns = request.gisaxs, cut.get("horizontal_rows"), cut.get("vertical_columns")
        if rows is not None and gisaxs.horizontal_row is None:
            gisaxs = replace(gisaxs, horizontal_row=0.5 * (rows[0] + rows[1]))
        if columns is not None and gisaxs.vertical_column is None:
            gisaxs = replace(gisaxs, vertical_column=0.5 * (columns[0] + columns[1]))
        return replace(request, gisaxs=gisaxs, **named)
    window = cut.get("chi_q_window")
    if window is not None and request.giwaxs.chi_q_window is None:
        return replace(request, giwaxs=replace(request.giwaxs, chi_q_window=(float(window[0]), float(window[1]))),
                       **named)
    return request


def band_rows(region_rows, band) -> Optional[tuple[int, int]]:
    """The pixel rows a horizontal band covers, both ends included (“202–206”), as the Cuts step names them:
    from the rows of the curve (``region["rows"]``, half-open) or else from the edges of the band (rows whose
    centre lies inside)."""
    if region_rows:
        start, stop = (int(value) for value in region_rows)
        return start, max(start, stop - 1)
    if band is None:
        return None
    low, high = band
    return int(math.ceil(low - 0.5)), max(int(math.ceil(low - 0.5)), int(math.floor(high - 0.5)))


def source_key(path) -> str:
    """How a listed source is recognised again (its path, whatever the case of its letters)."""
    return str(path).casefold()


class ReadStamps:
    """What the files of a frame were like when it was read (``files_stamp``), for as long as the frame is in
    memory. Nothing is held alive here (a frame of a large detector is a hundred megabytes); frames are
    known by identity, and an entry goes with its frame. Safe to use from the threads that analyse frames."""

    def __init__(self) -> None:
        self._kept: dict[int, tuple] = {}

    def keep(self, frame, stamp) -> None:
        key = id(frame)
        try:
            self._kept[key] = (weakref.ref(frame, lambda _gone, key=key: self._kept.pop(key, None)), stamp)
        except TypeError:  # (an object that cannot be referred to weakly: nothing is known of it)
            pass

    def of(self, frame, unknown=None):
        """The stamp kept for ``frame``; ``unknown`` for a frame that was never kept."""
        kept = self._kept.get(id(frame))
        return kept[1] if kept is not None and kept[0]() is frame else unknown


class FramesModelMixin(ImportSelectionMixin):
    """Needs ``state`` (``files``, ``sum_count``, ``gisaxs``, ``giwaxs``, ``analysis``), ``settings``,
    ``_read_setting``, ``_frames`` and ``request_for``."""

    sum_set_by = BY_USER
    _pin_unsaved: Optional[bool] = None
    """The pin of a session without a settings store."""

    def _init_frames(self) -> None:
        self._frame_counts: dict[str, int] = {}
        self._import_selections: dict = {}
        """The last count read of each source (never a failed read)."""
        self._watch_pending: list[FrameRef] = []
        self._read_stamps = ReadStamps()
        self._watch_cut: Optional[dict] = None
        """The cut the frames of the folder watch are given (taken once, from the frame then on screen)."""
        self._opened_cut: Optional[tuple] = None
        """``((source key, frame index), cut)`` of the frame shown from a row of a Series map (``show_with_cut``)."""
        self._frame_window: Optional[tuple[int, Optional[int]]] = None
        """``(first, last)`` frame numbers a series run is limited to (``only_frames``)."""

    def describe(self, path: Path):
        """What a NeXus / HDF5 source consists of (``NexusLayout``: files, frames, warnings, orientation), or
        ``None``: a frame of another format, a file never read so far, a frame source that cannot say."""
        describe = getattr(self._frames, "describe", None)
        if describe is None:
            return None
        try:
            return describe(path)
        except Exception:  # noqa: BLE001 - a source that cannot say what it is has no layout
            return None

    def files_stamp(self, path: Path, summed: Sequence[FrameRef] = ()) -> tuple:
        """What tells whether the files of a frame (and of the frames summed with it) changed since it was
        read: for a NeXus / HDF5 source the name, time and size of every file it consists of, for any other
        file its own time and size. Costs ``os.stat`` only."""
        return tuple(self._file_stamp(source) for source in dict.fromkeys([Path(path), *(Path(item[0]) for item in summed)]))

    def shown_changed(self) -> bool:
        """The files of the frame held in memory changed since it was read (``analyze`` reads it again)."""
        analysis = self.state.analysis
        if analysis is None:
            return False
        now = self.files_stamp(analysis.path, analysis.summed_frames)
        return self._read_stamps.of(analysis, now) != now

    def _file_stamp(self, path: Path):
        layout = self.describe(path)
        if layout is not None:
            return layout.signature
        try:
            status = path.stat()
        except OSError:
            return None  # (a frame source over something else than files)
        return status.st_mtime_ns, status.st_size

    def frame_count(self, path: Path) -> int:
        """Frames the source holds now (at least 1). A source that cannot be read keeps the last count read
        (1 when it never was read); that answer is not remembered, so the next call asks again. Never raises:
        one file that cannot be read must not stop a listing."""
        key = source_key(path)
        try:
            count = max(1, int(self._frames.frame_count(path)))
        except Exception:  # noqa: BLE001 - whatever a frame source raises for a file it cannot read
            return self._frame_counts.get(key, 1)
        self._frame_counts[key] = count
        return count

    def known_frame_count(self, path: Path) -> int:
        """The count last read, without touching the disk (asked of the source when it never was read)."""
        count = self._frame_counts.get(source_key(path))
        return self.frame_count(path) if count is None else count

    def listed_frames(self) -> int:
        """Frames listed, without reading any file: every source is counted when it is listed
        (``add_paths``, the folder watch), and again whenever one of its frames is shown or it grows."""
        return self.imported_frame_count()

    def remember_frame_count(self, path: Path, count: int) -> None:
        self._frame_counts[source_key(path)] = max(1, int(count))

    def _forget_frames(self, path: Path) -> None:
        """A file left the list: its frame count, and its frames waiting for a group of the folder watch."""
        key = source_key(path)
        self._frame_counts.pop(key, None)
        self._import_selections.pop(key, None)
        self._watch_pending = [frame for frame in self._watch_pending if source_key(frame[0]) != key]

    @property
    def sum_count(self) -> int:
        return self.state.sum_count

    def set_sum_count(self, count: int, *, by: str = BY_USER) -> None:
        """Sum this many consecutive frames into each analysis (1: no summing). ``by``: who asks —
        ``automatic`` when Run Automatic Analysis sums the frames of its own result, which a series run
        must not take for the person's choice (``sum_set_by``; no sum is nobody's)."""
        self.state.sum_count = max(1, min(MAX_SUM, int(count)))
        self.sum_set_by = by if by == BY_AUTOMATIC and self.state.sum_count > 1 else BY_USER

    def summed_frames_for(self, path: Path, frame_index: int = 0) -> tuple[FrameRef, ...]:
        """The frames added to ``(path, frame_index)`` for the current sum (maybe fewer at the end)."""
        extra = self.state.sum_count - 1
        if extra <= 0:
            return ()
        path = Path(path)
        count = self.frame_count(path)
        if count > 1:  # (of a source the watch follows as it grows, only frames handed out: ``_usable_frames``)
            last = min(max(self._usable_frames(path), frame_index + 1), frame_index + 1 + extra)
            selected = self.selected_frame_indices(path, usable=True)
            following = []
            for index in range(frame_index + 1, last):
                if index not in selected:
                    break
                following.append((path, index))
            return tuple(following)
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
            if (candidate.suffix.lower() != path.suffix.lower() or self.frame_count(candidate) > 1
                    or not self.frame_is_imported(candidate, 0)):
                break
            following.append((Path(candidate), 0))
        return tuple(following)

    def all_frames(self, paths: Optional[Sequence[Path]] = None) -> Sequence[FrameRef]:
        """Every frame of the listed files (or of ``paths``), in list order — of a source that grows while the
        folder is watched, only the frames handed out (``_usable_frames``)."""
        paths = self.state.files if paths is None else paths
        return FrameSequence((Path(path), self.selected_frame_indices(Path(path), usable=True)) for path in paths)

    def _usable_frames(self, path: Path) -> int:
        """Frames a series run may take of ``path``: all it holds, but of a source the folder watch follows as
        it grows only those already handed out (``sources_model.frames_to_run``: the newest may still be being
        written)."""
        usable = getattr(self, "frames_to_run", None)
        return self.frame_count(path) if usable is None else max(0, int(usable(path)))

    def _group_key(self, frame: FrameRef) -> tuple:
        """Frames sum only with frames of the same kind: one multi-frame file, or one file type."""
        path = Path(frame[0])
        return path.suffix.lower(), source_key(path) if self.known_frame_count(path) > 1 else None

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
        numbers = self.frame_numbers()
        for frame in frames:
            frame_key = self._group_key(frame)
            gap = bool(current and source_key(frame[0]) == source_key(current[-1][0])
                       and int(frame[1]) != int(current[-1][1]) + 1)
            if current:
                before = numbers.get((source_key(current[-1][0]), int(current[-1][1])))
                after = numbers.get((source_key(frame[0]), int(frame[1])))
                gap = gap or (before is not None and after is not None and after != before + 1)
            if current and (frame_key != key or len(current) == size or gap):
                groups.append(current)
                current = []
            key = frame_key
            current.append(frame)
        if current and not (keep_open_tail and len(current) < size):
            groups.append(current)
            current = []
        return groups, current

    def _requests(self, groups: Sequence[Sequence[FrameRef]], cut: Optional[dict] = None) -> list[AnalysisRequest]:
        return [
            with_cut(self.request_for(group[0][0], group[0][1], summed=tuple(group[1:])), cut) for group in groups
        ]

    def _run_frames(self) -> FrameSequence:
        """The frames of a series run: every listed frame, or those ``only_frames`` leaves."""
        frames = self.all_frames()
        if self._frame_window is None:
            return frames
        first, last = self._frame_window
        return frames.within(self.frame_numbers(), max(1, int(first)), last)

    @contextmanager
    def only_frames(self, first: int = 1, last: Optional[int] = None):
        """While it holds, a series run takes only the frames numbered ``first`` … ``last`` of the list (from 1,
        as the frame axis of a map counts; ``last`` empty: to the end) — a map of a part of a long series."""
        before, self._frame_window = self._frame_window, (max(1, int(first)), int(last) if last else None)
        try:
            yield
        finally:
            self._frame_window = before

    def _lazy_groups(self):
        frames = self._run_frames()
        counts = [(path, self.known_frame_count(path)) for path in self.state.files]
        return SumGroups(frames, self.state.sum_count, counts)

    def batch_requests(self) -> Sequence[AnalysisRequest]:
        """One request per frame of every listed file, or per group of N frames when summing — every one cut
        where the frame on screen is (``series_cut``) unless the pin is off."""
        prototype = self.request_for(Path("."), summed=())
        return BatchRequests(self._lazy_groups(), prototype, self.series_cut(), with_cut)

    def batch_groups(self) -> Sequence[SumGroup]:
        """What each request of ``batch_requests()`` sums, in the same order: with a Sum N, the last group of a
        source may hold fewer frames (``SumGroup.complete``)."""
        return BatchGroups(self._lazy_groups(), self.state.sum_count, SumGroup)

    def batch_group_summary(self) -> dict:
        """Frame/group totals and incomplete sizes without enumerating processing requests."""
        return self._lazy_groups().summary()

    def batch_tail(self) -> tuple[FrameRef, ...]:
        """Only the final physically consecutive unfinished sum, for watching its completion."""
        return self._lazy_groups().tail(self.state.sum_count)

    def batch_group_index(self, path, frame_index):
        """Processing-group index of a selected physical frame, or None outside the current run window."""
        return self._lazy_groups().index_for(path, frame_index)

    def frame_numbers(self) -> FrameNumbers:
        """``(source key, frame index)`` → the number of the frame in the listed series, from 1 (for one
        multi-frame source: the frame in the file) — what the frame axis of a Series map counts in."""
        return FrameNumbers((path, self.known_frame_count(path)) for path in self.state.files)

    def _watch_groups(self, frames: Sequence[FrameRef]) -> list[AnalysisRequest]:
        """Requests for frames the folder watch found (``sources_model.watch_requests``); with summing, only
        complete groups of N: the frames of a group still open wait for the next ones. One cut for the whole
        watch: that of the frame on screen when its first frames are asked for (the frame on screen then
        follows the newest one, whose own cut must not become the next frame's)."""
        self._watch_pending.extend(frame for frame in frames if self.frame_is_imported(*frame))
        groups, self._watch_pending = self._groups(self._watch_pending, keep_open_tail=True)
        if self._watch_cut is None:
            self._watch_cut = self.series_cut()
        return self._requests(groups, self._watch_cut if self.pin_series_cuts else None)

    def reset_watch_groups(self) -> None:
        self._watch_pending = []
        self._watch_cut = None

    # -- one cut for a series --------------------------------------------------------------------------

    @property
    def pin_series_cuts(self) -> bool:
        """Every frame of a series run is cut where the frame on screen is (on unless turned off)."""
        if self.settings is None and self._pin_unsaved is not None:
            return self._pin_unsaved
        return str(self._read_setting(SECTION, PIN_KEY, True)).strip().lower() not in ("false", "0", "no", "off")

    def set_pin_series_cuts(self, on: Optional[bool]) -> None:
        """Off: the cut is found again in every frame (remembered with Analyze's settings). ``None`` (a settings
        file that does not say): nothing changes."""
        if on is None:
            return
        self._pin_unsaved = bool(on)
        self._watch_cut = None
        if self.settings is not None:
            self.settings.set(SECTION, PIN_KEY, bool(on))
            self.settings.save()

    def _automatic_cuts(self) -> Optional[tuple[str, dict]]:
        """``(technique, {name: (low, high)})``: the cuts of the frame on screen that were found, not set by
        the person. ``None``: no frame with a reduction is shown."""
        analysis = self.state.analysis
        reduction = analysis.reduction if analysis is not None else None
        if reduction is None:
            return None
        markers = reduction.markers or {}
        if analysis.kind == GISAXS:
            settings = self.state.gisaxs
            return GISAXS, {
                "horizontal_rows": _pair(markers.get("horizontal_band")) if settings.horizontal_row is None else None,
                "vertical_columns": _pair(markers.get("vertical_band")) if settings.vertical_column is None else None,
            }
        window = _pair(markers.get("chi_q_window")) if self.state.giwaxs.chi_q_window is None else None
        return str(analysis.kind or "giwaxs"), {"chi_q_window": window}

    def series_cut(self) -> Optional[dict]:
        """What a series run is pinned to, from the frame on screen: ``{"technique": "gisaxs",
        "horizontal_rows": (low, high) | None, "vertical_columns": (low, high) | None, "frame": (path, index)}``
        (the edges of the bands, in pixels as the image's axes count) or ``{"technique": "giwaxs",
        "chi_q_window": (q_low, q_high) | None, "frame": …}``; a cut the person set is ``None`` there (it is
        the same for every frame already). ``None``: nothing is pinned — the pin is off, no frame is
        shown, or every cut is the person's."""
        found = self._automatic_cuts() if self.pin_series_cuts else None
        if found is None or all(value is None for value in found[1].values()):
            return None
        analysis = self.state.analysis
        return {"technique": found[0], **found[1], "frame": (Path(analysis.path), int(analysis.frame_index))}

    def series_cut_text(self) -> tuple[str, dict]:
        """``(template, values)``: one sentence for the interface (``CUT_TEXTS``; fill it with ``trf``) saying
        where the frames of a series run are cut. ``("", {})`` while no frame with a reduction is shown."""
        found = self._automatic_cuts()
        if found is None:
            return "", {}
        technique, cuts = found
        if technique == GISAXS:
            if cuts["horizontal_rows"] is None:
                return CUT_BY_YOU, {}
            if not self.pin_series_cuts:
                return CUT_EACH, {}
            curve = self.state.analysis.reduction.curve("horizontal")
            rows = band_rows((curve.region or {}).get("rows") if curve is not None else None, cuts["horizontal_rows"])
            return CUT_ROWS, {"low": f"{rows[0]}", "high": f"{rows[1]}"}
        window = cuts["chi_q_window"]
        if window is None:
            return (CUT_BY_YOU, {}) if self.state.giwaxs.chi_q_window is not None else ("", {})
        if not self.pin_series_cuts:
            return RING_EACH, {}
        return RING_Q, {"low": f"{window[0]:.3f}", "high": f"{window[1]:.3f}"}

    def show_with_cut(self, path: Path, frame_index: int, cut: Optional[dict]) -> None:
        """Series ▸ Open: the frame of a map row is shown cut as the map was (``cut``: the ``series_cut`` kept
        with the map; ``None``: as any frame). It holds for that frame only: another frame, or Reset of the
        cuts, and the cut is found in the frame again."""
        self._opened_cut = None if not cut else ((source_key(path), int(frame_index)), dict(cut))

    def held_cut(self, path: Path, frame_index: int) -> Optional[dict]:
        """The cut a frame is shown with because it was opened from a map row (``show_with_cut``), else ``None``."""
        held = self._opened_cut
        return dict(held[1]) if held is not None and held[0] == (source_key(path), int(frame_index)) else None

    def _shown_request(self, request: Optional[AnalysisRequest]) -> Optional[AnalysisRequest]:
        """``request`` of the frame on screen, with the cut of the map it was opened from (``show_with_cut``)."""
        held = self._opened_cut
        if request is None or held is None:
            return request
        if held[0] != (source_key(request.path), int(request.frame_index)):
            self._opened_cut = None  # the person moved on to another frame
            return request
        return with_cut(request, held[1])


__all__ = [
    "BY_AUTOMATIC", "BY_USER", "CUT_BY_YOU", "CUT_EACH", "CUT_ROWS", "CUT_TEXTS", "FramesModelMixin", "MAX_SUM",
    "PIN_KEY", "RING_EACH", "RING_Q", "ReadStamps", "SumGroup", "band_rows", "source_key", "with_cut",
]
