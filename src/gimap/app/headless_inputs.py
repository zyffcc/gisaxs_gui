"""What a command line of ``tools/gimap_agent.py auto`` names: its frames, and where each one's files go.

- **Frames.** A shell that expands globs hands over files; PowerShell and ``cmd`` hand over the pattern. An
  argument that is no file but holds ``*``, ``?`` or ``[`` is expanded here, in natural order (``f2`` before
  ``f10``), and a folder stands for every detector image in it (by suffix, subfolders not searched). A pattern
  or folder with nothing in it is an error that says so; the summary says how many files each one gave.
- **Folders.** Every frame has its own folder under the output folder, named by the frame. Frames of the same
  name in different folders get the shortest part of their paths that tells them apart (``<name>_<folder>``),
  else a number, so no frame overwrites another.
- **An earlier run.** An output folder that already holds a summary is not written into: files of two runs would
  mix and frame folders of the earlier one would stay beside the new summary. ``overwrite`` removes the frame
  folders the earlier ``summary.json`` lists (nothing else) and writes the new run.

Nothing here opens a window or reads a frame.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..shared.detector_io import DETECTOR_SUFFIXES
from .headless_sources import _file_key, _natural_key

SUMMARY_FILES = ("summary.md", "summary.json")
EARLIER_RUN = (
    "{out} already holds the summary of an earlier run ({found}). Give another --out folder, or --overwrite to "
    "replace that run: its frame folders listed in summary.json are removed first, nothing else."
)
NO_MATCH = "No file matches the pattern {pattern}."
EMPTY_FOLDER = (
    "No detector image in the folder {folder} (looked for {suffixes}; subfolders are not searched)."
)
_UNSAFE = re.compile(r"[^\w.\-]+")


def _natural(paths: Iterable[str]) -> list[str]:
    return sorted(paths, key=lambda text: _natural_key(os.path.normcase(text)))


def expand_frames(arguments: Sequence[str], *, exclude: Sequence[Optional[str]] = ()) -> tuple[list[str], list[str]]:
    """``(frames, notes)``: the files the arguments stand for, in the order given, and one sentence for the summary
    per pattern or folder. A plain name is passed on as it is (a missing file is that frame's error, the other
    frames are still analysed). ``exclude``: files that are no sample frames (the calibration, the mask, the flat
    field), left out of a folder's listing. ``FileNotFoundError`` when a pattern or a folder gives nothing."""
    skipped = {_file_key(text) for text in exclude if text}
    frames: list[str] = []
    notes: list[str] = []
    for text in (os.fspath(argument) for argument in arguments):
        if os.path.isdir(text):
            listed = [
                str(item) for item in Path(text).iterdir()
                if item.is_file() and item.suffix.lower() in DETECTOR_SUFFIXES
            ]
            found = _natural(item for item in listed if _file_key(item) not in skipped)
            if not found:
                raise FileNotFoundError(EMPTY_FOLDER.format(folder=text, suffixes=", ".join(DETECTOR_SUFFIXES)))
            left_out = len(listed) - len(found)
            notes.append(
                f"Folder {text}: {len(found)} detector image(s) found (subfolders are not searched"
                + (f"; {left_out} file(s) given as calibration, mask or flat field left out" if left_out else "") + ")."
            )
            frames += found
        elif not os.path.isfile(text) and glob.has_magic(text):
            found = _natural(item for item in glob.glob(text) if os.path.isfile(item))
            if not found:
                raise FileNotFoundError(NO_MATCH.format(pattern=text))
            notes.append(f"Pattern {text}: {len(found)} file(s) matched.")
            frames += found
        else:
            frames.append(text)
    return frames, notes


def _telling_apart(paths: Sequence[str]) -> list[str]:
    """For files of one name: the nearest folder name that differs between all of them ('' each when no single
    level of their paths does)."""
    parents = [list(reversed(Path(os.path.abspath(path)).parent.parts)) for path in paths]
    for depth in range(max(len(parts) for parts in parents)):
        names = [_UNSAFE.sub("_", parts[depth]).strip("_.") if depth < len(parts) else "" for parts in parents]
        if all(names) and len({name.casefold() for name in names}) == len(names):
            return names
    return [""] * len(paths)


def frame_folders(sources: Sequence) -> list[str]:
    """One folder name per source (``.stem`` and ``.path``), all different whatever the case: the stem, with
    what tells frames of the same name apart (``_telling_apart``), else with a number."""
    names = [source.stem for source in sources]
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(name.casefold(), []).append(index)
    for members in groups.values():
        if len(members) > 1:
            for index, part in zip(members, _telling_apart([sources[index].path for index in members])):
                names[index] = f"{names[index]}_{part}" if part else names[index]
    used: set[str] = set()
    for index, name in enumerate(names):
        number, unique = 1, name
        while unique.casefold() in used:
            number += 1
            unique = f"{name}_{number}"
        used.add(unique.casefold())
        names[index] = unique
    return names


def summary_reports(summary) -> list[dict]:
    """The frame reports of a ``summary.json``: its ``frames`` (an object, since 0.2.0), or the list it was
    before. ``ValueError`` when it is neither."""
    reports = summary.get("frames") if isinstance(summary, dict) else summary
    if not isinstance(reports, list) or not all(isinstance(report, dict) for report in reports):
        raise ValueError("not a summary of gimap_agent.py auto: no list of frame reports")
    return reports


def earlier_folders(out: Path) -> list[Path]:
    """The frame folders of the run whose ``summary.json`` lies in ``out``: only folders directly in ``out``
    that the summary lists (``folder``, or the folder of a report's first output for a summary written
    before 0.2.0)."""
    try:
        reports = summary_reports(json.loads((out / "summary.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []
    home, folders = out.resolve(), []
    for report in reports:
        outputs = report.get("outputs") or []
        name = report.get("folder") or (Path(str(outputs[0])).parent.name if outputs else "")
        folder = out / str(name)
        if name and Path(str(name)).name == str(name) and folder.is_dir() and folder.resolve().parent == home \
                and folder not in folders:
            folders.append(folder)
    return folders


def prepare_output(out: Path, overwrite: bool = False) -> list[Path]:
    """Make sure a run may write into ``out``. With the summary of an earlier run there: ``FileExistsError``
    unless ``overwrite``, which removes that run's frame folders (returned) and nothing else."""
    found = [name for name in SUMMARY_FILES if (out / name).is_file()]
    if not found:
        return []
    if not overwrite:
        raise FileExistsError(EARLIER_RUN.format(out=out, found=", ".join(found)))
    removed = earlier_folders(out)
    for folder in removed:
        shutil.rmtree(folder)
    return removed


__all__ = [
    "EARLIER_RUN", "EMPTY_FOLDER", "NO_MATCH", "SUMMARY_FILES", "earlier_folders", "expand_frames", "frame_folders",
    "prepare_output", "summary_reports",
]
