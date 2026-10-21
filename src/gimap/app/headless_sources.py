"""What the files given to a session without a window stand for (``tools/gimap_agent.py auto``).

A shell glob hands over every file of a folder, but a detector source may consist of several files:

- the module files of one detector (P03 Lambda 9M: ``…_m01.nxs`` … ``…_m11.nxs``) are ONE stitched series
  however many of them are given, and a data file of a master stands for the master. ``given_sources``
  maps the arguments to sources with the shared NeXus reader (the content decides, a name only
  proposes), so each source is analysed once and the report says what it consists of;
- a source with a part missing (a module file, a data file, unequal frame counts) is an open point of
  its report: the numbers lack that part;
- files that hold one frame each and are numbered alike may be one series stored as one frame per
  file. Nobody but the person knows (they may as well be separate samples), so by default each is
  analysed on its own and the summary says how to join them: ``series=True`` (``--series``) takes
  them as one series in the order of their numbers and analyses its final state like the frames of a
  multi-frame file (``series_block``).

Nothing here opens a window or writes a file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from ..features.assistant.application import batch_markdown
from ..features.assistant.application.pipeline import default_sum
from ..shared.detector_io import NexusLayout, cached_layout, group_sources, is_hdf5, scan_stem
from ..shared.detector_io import nexus_layout, nexus_modules

SEVERAL_FILES = ("module_set", "chunked_master")
PART_MISSING = (
    nexus_modules.MODULES_UNEQUAL, nexus_modules.MODULE_MISSING, nexus_modules.MODULES_MISSING,
    nexus_modules.MODULE_UNREADABLE, nexus_modules.MODULE_LEFT_UNFINISHED, nexus_modules.MODULES_LEFT_UNFINISHED,
    nexus_modules.STILL_WRITTEN, nexus_modules.LEFT_OPEN,
    nexus_modules.MODULES_FEWER, nexus_layout.DATA_FILE_MISSING, nexus_layout.DATA_FILES_MISSING,
)
"""Reader warnings that mean: the frames lack a part of the detector, or may lack one (another scan of the
folder has more module files), or the series is not all there."""
PART_MISSING_HINT = (
    "Check that every file of the scan is in the folder and complete (is the measurement still running?), "
    "then run again: until then the results lack that part."
)
SERIES_MINIMUM = 3
"""Fewer files numbered alike are not worth a remark about a series."""
_LAST_NUMBER = re.compile(r"^(?P<head>.*?)(?P<number>\d+)(?P<tail>\D*)$")


@dataclass(frozen=True)
class GivenSource:
    """One detector source of a command line."""

    path: str
    """What is analysed: the argument as it was given, or the file that stands for its source (the first
    module file of a set, the master of a data file)."""
    given: tuple[str, ...]
    """The arguments that are files of this source."""
    layout: Optional[NexusLayout] = None
    """What a NeXus / HDF5 source consists of (``None``: another format, or a file that cannot be read)."""

    @property
    def files(self) -> tuple[str, ...]:
        """The different files among ``given``, as first named: a file named twice counts once."""
        return _different_files(self.given)

    @property
    def stem(self) -> str:
        """The name for folders and labels: a set of module files is named by its scan, without ``_mNN``."""
        return scan_stem(self.layout, self.path) or Path(self.path).stem

    @property
    def frames(self) -> int:
        return 1 if self.layout is None else int(self.layout.frames)


def _natural_key(name: str) -> tuple:
    return tuple(int(token) if token.isdigit() else token.casefold() for token in re.split(r"(\d+)", name))


def _file_key(path: str | os.PathLike) -> str:
    """One key for a file however it is named: relative, through a linked folder, in another case on Windows."""
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _different_files(given: Sequence[str]) -> tuple[str, ...]:
    files: dict[str, str] = {}
    for text in given:
        files.setdefault(_file_key(text), text)
    return tuple(files.values())


def _as_named(stands_for: Path, text: str) -> str:
    """The file a given one stands for, in the folder as the person named it when it is there (the reader
    names it by its real path, which differs when the folder was reached through a link)."""
    beside = Path(text).parent / stands_for.name
    try:
        return str(beside if beside.samefile(stands_for) else stands_for)
    except OSError:
        return str(stands_for)


def given_sources(frames: Sequence[str | os.PathLike]) -> list[GivenSource]:
    """The sources the arguments stand for, in the order they were first named. A file named twice
    (overlapping globs) is one source, also when one of its names goes through a linked folder."""
    order: list[str] = []
    found: dict[str, tuple[str, list[str], Optional[NexusLayout]]] = {}
    for frame in frames:
        text, layout = os.fspath(frame), None
        path = text
        if is_hdf5(text) and os.path.isfile(text):
            try:
                rows = group_sources([text], keep_without_image=True)
                stands_for = rows[0] if rows else Path(text)
                layout = cached_layout(stands_for)
                if stands_for.name != Path(text).name:  # a module file of a set, a data file of a master
                    path = _as_named(stands_for, text)
            except Exception:  # noqa: BLE001 - a file that cannot be read: analysing it says why
                layout = None
        key = _file_key(path)
        if key not in found:
            order.append(key)
            found[key] = (path, [], layout)
        found[key][1].append(text)
    return [GivenSource(path, tuple(given), layout) for path, given, layout in (found[key] for key in order)]


def given_note(source: GivenSource) -> str:
    """What to say when several arguments were one source ('' for one argument): the files of a NeXus
    source, or one file named more than once (any format, also a file that cannot be read)."""
    files = source.files
    if len(files) > 1:
        of_what = f" ({source.layout.summary_text()})" if source.layout is not None else ""
        return f"{len(files)} files given for one source{of_what}: analysed once"
    return f"given {len(source.given)} times: analysed once" if len(source.given) > 1 else ""


def describe_source(report: dict, source: GivenSource) -> None:
    """Write into ``report`` what its NeXus / HDF5 source consists of: ``source`` (the record), a ``source``
    decision when it is more than one plain file or the reader had something to say, and an open point
    (``needs_attention``) for every part that is missing. A source of another format (or one that cannot
    be read) only gets ``given`` when it was named more than once."""
    layout = source.layout
    if layout is None or layout.kind == "no_image":
        if len(source.given) > 1:
            report["given"] = list(source.given)
        return
    summary = layout.summary_text()
    report["source"] = {
        "layout": layout.kind, "summary": summary, "name": source.stem, "frames": int(layout.frames),
        "files": [str(member) for member in layout.members], "given": list(source.given),
        "complete": bool(layout.complete), "warnings": list(layout.warning_texts()),
    }
    why, files = layout.decided_by, source.files
    if len(files) > 1:
        why += f"; {len(files)} of its files were given: they are one source, analysed once"
    elif Path(files[0]).name != Path(source.path).name:
        why += f"; {Path(files[0]).name} was given: it is a file of this source"
    elif len(source.given) > 1:
        why += f"; it was given {len(source.given)} times: analysed once"
    notes, missing = [], []
    for (template, _values), text in zip(layout.warnings, layout.warning_texts()):
        (missing if template in PART_MISSING else notes).append(text)
    if layout.kind in SEVERAL_FILES or len(source.given) > 1 or notes:
        if notes:
            why += "; " + "; ".join(notes)
        report.setdefault("decisions", []).insert(0, {"what": "source", "decision": summary, "why": why})
    for text in missing:
        item = {"item": "source", "why": f"{text}.", "option": None, "hint": PART_MISSING_HINT}
        if item not in report.setdefault("needs_attention", []):
            report["needs_attention"].append(item)


class _Analysed(NamedTuple):
    """A source as a report describes it: enough to see whether reports are of files numbered alike."""

    path: str
    stem: str
    frames: int


def _numbering(source) -> Optional[tuple]:
    """``(folder, suffix, name before the last number, name after it)`` of a one-frame source, else ``None``."""
    if source.frames != 1:
        return None
    numbered = _LAST_NUMBER.match(source.stem)
    if numbered is None:
        return None
    path = Path(source.path)
    return (os.path.normcase(str(path.parent)), path.suffix.casefold(), numbered["head"].casefold(), numbered["tail"].casefold())


def series_hints(sources: Sequence) -> list[list]:
    """Groups of at least ``SERIES_MINIMUM`` one-frame sources of one folder that are named alike up to
    their last number, each in the order of the numbers: possibly one series stored as one frame per file."""
    groups: dict[tuple, list] = {}
    for source in sources:
        key = _numbering(source)
        if key is not None:
            groups.setdefault(key, []).append(source)
    return [
        sorted(group, key=lambda source: _natural_key(Path(source.path).name))
        for group in groups.values() if len(group) >= SERIES_MINIMUM
    ]


def series_block(
    sources: Sequence[GivenSource], frame: Optional[int], sum_frames: Optional[int]
) -> tuple[list[GivenSource], int, int]:
    """``(the files in the order of their numbers, first file of the block (1-based), files summed)`` for
    files that are one series stored as one frame per file. ``frame`` (1-based, negative from the end;
    default: the last files) and ``sum_frames`` (default: ``default_sum``) choose the block exactly as
    they choose the frames of a multi-frame file; a block never runs past the ends. ``ValueError`` says
    why the files cannot be one such series."""
    if len(sources) < 2:
        raise ValueError("--series needs at least two files: the files of a series stored as one frame per file.")
    several = [Path(source.path).name for source in sources if source.frames != 1]
    if several:
        raise ValueError(
            "--series joins files that hold one frame each; "
            f"{several[0]} holds {next(source.frames for source in sources if source.frames != 1)} frames: "
            "give that file alone and choose its frames with --frame and --sum."
        )
    suffixes = sorted({Path(source.path).suffix.casefold() for source in sources})
    if len(suffixes) > 1:
        raise ValueError(f"--series needs files of one format; these are {', '.join(suffixes)}.")
    ordered = sorted(sources, key=lambda source: _natural_key(Path(source.path).name))
    total = len(ordered)
    count = max(1, min(int(sum_frames or default_sum(total)), total))
    number = -1 if frame is None else int(frame)
    if number < 0:
        number += total + 1
    return ordered, min(max(1, number), total - count + 1), count


def series_name(ordered: Sequence[GivenSource]) -> str:
    """A folder name for the report of a file series: what its files are called before their number."""
    numbered = _LAST_NUMBER.match(ordered[0].stem)
    head = numbered["head"].rstrip("_-. ") if numbered else ""
    return f"{head or Path(ordered[0].path).parent.name or 'files'}_series"


def source_notes(reports: Sequence[dict]) -> list[str]:
    """Lines under the summary table: files that were given for one source, and files that look like one
    series but were analysed one by one."""
    lines: list[str] = []
    for report in reports:
        source = report.get("source") or {}
        given = source.get("given") or report.get("given") or []
        files = _different_files(given)
        name = Path(report.get("frame") or (given[0] if given else "")).name
        if len(files) > 1:
            lines += [
                f"{len(files)} files were given for {name} "
                f"({source.get('summary')}): they are one source, analysed once.", "",
            ]
        elif len(given) > 1:
            lines += [f"{name} was given {len(given)} times: it was analysed once.", ""]
    analysed = []
    for report in reports:
        if not report.get("frame") or report.get("series"):
            continue
        source = report.get("source") or {}
        frames = source.get("frames") or (report.get("frames") or {}).get("total") or 1
        path = str(report["frame"])
        analysed.append(_Analysed(path, source.get("name") or Path(path).stem, int(frames)))
    for group in series_hints(analysed):
        first, last = Path(group[0].path).name, Path(group[-1].path).name
        lines += [
            f"{first} … {last} ({len(group)} files) look like one series stored as one frame per file: each was "
            "analysed on its own. `--series` analyses them as one series (its last files summed, like the "
            "frames of a multi-frame file; `--frame N --sum K` choose by position).", "",
        ]
    return lines


def batch_summary(reports: Sequence[dict]) -> str:
    """``summary.md`` of a batch: the tables of ``batch_markdown`` and what the files given stood for."""
    notes = source_notes(reports)
    return batch_markdown(list(reports)) + ("\n".join(notes) + "\n" if notes else "")


__all__ = [
    "GivenSource",
    "PART_MISSING",
    "batch_summary",
    "describe_source",
    "given_note",
    "given_sources",
    "series_block",
    "series_hints",
    "series_name",
    "source_notes",
]
