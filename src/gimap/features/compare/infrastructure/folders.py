"""How Analyze names what it writes, read back: the kinds of curves a folder holds (``FolderReader``).

* A Batch Export folder (``<batch>_batch.json`` and ``README.txt`` in it): per curve a table
  ``<batch>_<curve>_frames.<txt>``, a file per frame in ``curves/`` (``<frame>_<curve>.<txt>`` beside the
  frame's ``<frame>_analysis.json``) and the curve for Fitting in ``fit_input/``. The README lists the curves
  with their titles, in the order Analyze shows them.
* Its ``curves`` folder, or the folder single frames were exported into (``gimap_analysis``): the files per
  frame only; the record of every frame tells where the frame's name ends and the curve's begins.
* A ``fit_input`` folder: one kind, the curve Fitting reads.
* Any other folder: its curve files are one kind, as they are.

A series is named after the batch or the sample folder, not “curves” or “fit_input” — and not
“gimap_analysis” either, except for that folder itself when it is one series (its name of before).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional

from ..application import CURVE_SUFFIXES, CurveKind, FolderContent, curve_files, natural_key
from ..application.sources import FIT_INPUT_KIND as FIT_INPUT
from ..application.sources import OTHER_KIND as OTHER

BATCH_TITLE = "GIMaP Analyze batch export"
BATCH_END, RECORD_END, TABLE_END, FIT_INPUT_END = "_batch.json", "_analysis.json", "_frames", "_fit_input"
README = "README.txt"
README_CURVES = "Curves:"
"""The README's list of curves begins with this line; each curve is ``  <key>: <title>``."""
CURVES_FOLDER, FIT_INPUT_FOLDER, ANALYZE_FOLDER = "curves", "fit_input", "gimap_analysis"
SAME_AS_FIT_INPUT = ("radial", "horizontal")
"""The curves a batch also writes as fit inputs: with one of them listed, ``fit_input/`` is no kind of its own.
(The kinds ``fit_input`` and ``other`` carry no words of the application: the page says what they are.)"""


def batch_name(folder: Path) -> Optional[str]:
    """The name of the batch whose folder this is (its record or its README is here), else ``None``."""
    for record in sorted(folder.glob("*" + BATCH_END)):
        try:
            title = json.loads(record.read_text(encoding="utf-8")).get("title")
        except (OSError, ValueError, AttributeError):
            continue
        if str(title).startswith(BATCH_TITLE):
            return record.name[: -len(BATCH_END)]
    try:
        with (folder / README).open(encoding="utf-8", errors="replace") as stream:
            first = stream.readline().strip()
    except OSError:
        return None
    if first.startswith(BATCH_TITLE):  # “GIMaP Analyze batch export — <batch> — <date>”
        parts = [part.strip() for part in first.split(" — ")]
        return parts[1] if len(parts) > 1 and parts[1] else folder.name
    return None


def readme_curves(folder: Path) -> dict[str, str]:
    """``{key: title}`` of the curves a batch's README lists, in its order (empty without one)."""
    try:
        lines = (folder / README).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    curves: dict[str, str] = {}
    listing = False
    for line in lines:
        if line.strip() == README_CURVES:
            listing = True
        elif listing:
            if not line.startswith("  ") or ":" not in line:
                break
            key, title = line.strip().split(":", 1)
            curves[key.strip()] = title.strip()
    return curves


def holder_name(folder: Path, *, one_series: bool = True) -> str:
    """The folder a series is named after: the first one, going up, that Analyze did not name itself.
    ``gimap_analysis`` chosen itself keeps its name while it is one series (``one_series``), as it always
    did; with several kinds of curves in it, each is named after the sample folder above it."""
    if one_series and folder.name.lower() == ANALYZE_FOLDER:
        return folder.name
    for place in (folder, *folder.parents):
        if place.name and place.name.lower() not in (CURVES_FOLDER, FIT_INPUT_FOLDER, ANALYZE_FOLDER):
            return place.name
    return folder.name


def _stem(path: Path) -> str:
    return path.name[: -len(path.suffix)] if path.suffix else path.name


class AnalyzeFolders:
    """``content(folder)``: the kinds of curves in a folder Analyze wrote, or the one kind of any other."""

    def __init__(self, reader):
        self.reader = reader

    def content(self, folder: Path, pattern: str = "*") -> FolderContent:
        folder = Path(folder)
        batch = batch_name(folder)
        if batch is not None:
            return FolderContent(batch, tuple(self._batch_kinds(folder, batch, pattern)))
        above = batch_name(folder.parent) if folder.name.lower() in (CURVES_FOLDER, FIT_INPUT_FOLDER) else None
        files = curve_files(folder, pattern)
        groups = self._by_frame_record(folder, files)
        name = above or holder_name(folder, one_series=len(groups) < 2)
        if not groups:
            return FolderContent(name, (CurveKind("", files=tuple(files), frames=len(files)),) if files else ())
        titles = readme_curves(folder.parent) if above else {}
        return FolderContent(name, tuple(self._kind(key, titles.get(key, ""), found) for key, found in groups.items()))

    # -- the files of a folder, by curve ---------------------------------------------------------

    def _by_frame_record(self, folder: Path, files) -> dict[str, list[Path]]:
        """``{curve key: its files in counting order}`` of the files Analyze wrote per frame: what follows the
        name of a frame that has its record (``<frame>_analysis.json``) here is the curve, and a file ending
        in ``_fit_input`` is the curve for Fitting. Empty when the folder is none of Analyze's: no record of a
        frame, and not every file a curve for Fitting — its files are then one series as they are. No file
        is left out: beside Analyze's exports, the curve files that are none of them are a kind of their own."""
        frames = {path.name[: -len(RECORD_END)] for path in folder.glob("*" + RECORD_END)}
        groups: dict[str, list[Path]] = {}
        others: list[Path] = []
        for path in files:
            stem = _stem(path)
            if stem.endswith(FIT_INPUT_END):
                groups.setdefault(FIT_INPUT, []).append(path)
                continue
            cut = len(stem)
            while frames and (cut := stem.rfind("_", 0, cut)) > 0:  # the longest frame name that has a record
                if stem[:cut] in frames:
                    groups.setdefault(stem[cut + 1:], []).append(path)
                    break
            else:
                others.append(path)
        if others and not frames:
            return {}
        order = self._record_order(folder, frames)
        keys = [key for key in order if key in groups] + sorted(key for key in groups if key not in order and key != FIT_INPUT)
        if FIT_INPUT in groups and FIT_INPUT not in keys:
            keys.append(FIT_INPUT)
        found = {key: sorted(groups[key], key=natural_key) for key in keys}
        if others and found:
            found[OTHER] = sorted(others, key=natural_key)
        return found

    @staticmethod
    def _record_order(folder: Path, frames) -> list[str]:
        """The curves in the order one frame's record lists its files (as Analyze shows them)."""
        for frame in sorted(frames)[:1]:
            try:
                listed = json.loads((folder / f"{frame}{RECORD_END}").read_text(encoding="utf-8")).get("files") or []
            except (OSError, ValueError, AttributeError):
                return []
            return [_stem(Path(str(name)))[len(frame) + 1:] for name in listed if str(name).startswith(frame + "_")]
        return []

    def _kind(self, key: str, title: str, files=(), table: Optional[Path] = None) -> CurveKind:
        files = tuple(files)
        first = files[0] if files else table
        if not title and first is not None and key != OTHER:
            title = self.reader.title(first)
        frames = len(files) if files else (self.reader.table_frames(table) if table is not None else 0)
        return CurveKind(key, title, files, table, frames)

    # -- a Batch Export folder -------------------------------------------------------------------

    def _batch_kinds(self, root: Path, batch: str, pattern: str) -> list[CurveKind]:
        titles = readme_curves(root)
        per_frame = self._by_frame_record(root / CURVES_FOLDER, curve_files(root / CURVES_FOLDER, pattern))
        per_frame.pop(FIT_INPUT, None)
        tables = {}
        for path in sorted(root.glob(f"{glob.escape(batch)}_*{TABLE_END}.*")):
            if path.is_file() and path.suffix.lower() in CURVE_SUFFIXES:
                tables[_stem(path)[len(batch) + 1: -len(TABLE_END)]] = path
        listed = [key for key in titles if key in per_frame or key in tables]  # the README's order first
        kinds = [self._kind(key, titles.get(key, ""), per_frame.get(key, ()), tables.get(key))
                 for key in dict.fromkeys([*listed, *per_frame, *tables])]
        sent = curve_files(root / FIT_INPUT_FOLDER, pattern)
        if sent and not any(kind.key in SAME_AS_FIT_INPUT for kind in kinds):
            kinds.append(self._kind(FIT_INPUT, "", sent))
        return kinds


__all__ = ["AnalyzeFolders", "batch_name", "holder_name", "readme_curves"]
