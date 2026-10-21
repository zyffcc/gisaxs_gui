"""The folder a batch writes into: can it be written, and what does it hold already?

* ``cannot_write`` answers before the first frame is reduced — an hour of reduction into a folder that takes
  no file is an hour lost. ``may_be_written`` is the same question asked without touching the disk (for the
  dialog, while the folder is typed).
* ``earlier_export`` finds what an earlier batch left in the folder: the record of the same data
  (``<batch>_batch.json``) with the files it lists, and — when curves for Fitting are to be written — every
  curve already in ``fit_input/``, since Fitting ▸ In-situ series lists that whole folder: two sends into one
  folder would be fitted as one series.
* ``remove_export`` takes exactly those files away (*Replace*); ``new_subfolder`` names a folder beside them
  that does not exist yet (*Keep both*). Only files inside the folder are ever taken: a record may name
  ``../`` or a whole path, and a link in the folder may lead out of it.
* ``EarlierExport.not_written_again(choices)``: the files Replace would take that a batch with ``choices``
  does not write again (its tables, the curves of single frames, pictures …) — the question about the earlier
  export says them, and does not offer Replace as the answer of Enter then.

A Send Series to Fitting has a folder of its own (``folder_name(batch, for_fitting=True)``: ``<batch>_fitting``):
its README and record would otherwise take the place of those of a Batch Export of the same data.

A record written by this version (``RECORD_FORMAT``) lists every file of its export; an older one lists only
its tables, so the files of its single frames cannot be told from other files and are left where they are.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .batch_export import FOLDERS, BatchChoices
from .batch_fit import FIT_NONE

RECORD_FORMAT = "gimap_analyze_batch_2"
"""``"format"`` of a batch record whose ``"files"`` are every file of the export."""
RECORD_SUFFIX = "_batch.json"
README = "README.txt"
FIT_INPUT_FOLDER = "fit_input"
FIT_INPUT_PATTERN = "*_fit_input.dat"
SEND_SUFFIX = "_fitting"
"""The folder of a Send Series to Fitting is ``<batch>_fitting``."""
KINDS = ("tables", "per_frame", "fit_input", "q_map", "cake", "detector_image", "q_map_image", "frames", "fit",
         "fit_plot", "fit_curves", "readme", "record", "other")
"""The kinds of file of a batch folder (the keys of ``output_names``, and ``other``), in the order they are said."""
_KIND_OF_FOLDER = {folder: kind for kind, folder in FOLDERS.items() if kind not in ("cake", "q_map_image")}


def folder_name(batch: str, *, for_fitting: bool = False) -> str:
    """The subfolder a batch is written into: its name — for a Send Series to Fitting ``<batch>_fitting``, so
    that a send never shares its folder (README and record) with a Batch Export of the same data."""
    return f"{batch}{SEND_SUFFIX}" if for_fitting else batch


def file_kind(name: str, batch: str) -> str:
    """The kind of the file ``name`` (relative to the batch folder, ``/`` between folders) of the batch
    ``batch``: one of ``KINDS``, from its folder and the end of its name."""
    parts = str(name).replace("\\", "/").split("/")
    last = parts[-1]
    if len(parts) > 1:
        kind = _KIND_OF_FOLDER.get(parts[0], "other")
        if kind == "q_map" and "_cake." in last:
            return "cake"
        if kind == "detector_image" and "_qmap." in last:
            return "q_map_image"
        return kind
    if last == README:
        return "readme"
    if last == f"{batch}{RECORD_SUFFIX}":
        return "record"
    stem = last.rsplit(".", 1)[0]
    if stem.endswith(("_peak_fits", "_model_fits")):
        return "fit_plot" if last.endswith(".png") else "fit"
    return "tables" if stem.endswith("_frames") else "other"


def chosen_kinds(choices: BatchChoices) -> set[str]:
    """The kinds of file a batch with ``choices`` writes (``README.txt`` and the record always)."""
    kinds = {kind for kind in KINDS[:8] if getattr(choices, kind)} | {"readme", "record"}
    if choices.fit != FIT_NONE:
        kinds |= {"fit", "fit_plot"} | ({"fit_curves"} if choices.fit_curves else set())
    return kinds


def _inside(path: Path, folder: Path) -> bool:
    """``path`` is in ``folder`` also after every link on the way is followed."""
    try:
        return Path(folder).resolve() in Path(path).resolve().parents
    except (OSError, RuntimeError):
        return False


NOT_A_FOLDER = "a file of that name is in the way"
NO_SUCH_PLACE = "there is no such drive or folder"
NO_PERMISSION = "no permission to write there"
"""Why a folder will not take files, when it can be told without writing (``may_be_written``)."""


def _reason(error: OSError) -> str:
    return error.strerror or str(error)


def cannot_write(target: Path) -> Optional[str]:
    """Why ``target`` cannot be written (the system's words), or ``None``: the folder is made if need be and
    a file is created in it and taken away again — the one test that is right on every file system."""
    target = Path(target)
    try:
        target.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".gimap_write_test_", dir=target)
        os.close(handle)
        os.unlink(name)
    except OSError as exc:
        return _reason(exc)
    return None


def may_be_written(target: Path) -> Optional[str]:
    """Why ``target`` will not take files, as far as can be told without writing — a file is where the folder
    is to be, or the folder it would be made in (the nearest one that exists) does not allow it — else
    ``None``. One of ``NOT_A_FOLDER``, ``NO_SUCH_PLACE``, ``NO_PERMISSION``, or the system's own words."""
    holder = Path(target)
    try:
        while not holder.exists():
            if holder.parent == holder:
                return NO_SUCH_PLACE
            holder = holder.parent
        if not holder.is_dir():
            return NOT_A_FOLDER
        if not os.access(holder, os.W_OK | os.X_OK):
            return NO_PERMISSION
    except OSError as exc:
        return _reason(exc)
    return None


@dataclass
class EarlierExport:
    """What a folder holds of an earlier batch (``earlier_export``)."""

    folder: Path
    batch: str = ""
    record: Optional[Path] = None
    """``<batch>_batch.json`` of the same data, when there is one."""
    when: str = ""
    """When that batch was made (``YYYY-MM-DD HH:MM``)."""
    frames: int = 0
    files: list = field(default_factory=list)
    """The files its record lists that are still there."""
    complete: bool = False
    """The record lists every file of its export (``RECORD_FORMAT``)."""
    fit_inputs: list = field(default_factory=list)
    """Every curve for Fitting in ``fit_input/`` (asked for when this batch writes some)."""

    @property
    def to_remove(self) -> list[Path]:
        return list(dict.fromkeys([*self.files, *self.fit_inputs, *([self.record] if self.record is not None else [])]))

    def not_written_again(self, choices: BatchChoices) -> list[Path]:
        """The files of ``to_remove`` that a batch with ``choices`` does not write again: what *Replace* would
        take for good (tables, curves of single frames, pictures … of kinds that batch does not write)."""
        written = chosen_kinds(choices)
        return [path for path in self.to_remove if self._kind(Path(path)) not in written]

    def kinds_not_written_again(self, choices: BatchChoices) -> list[str]:
        """The kinds (``KINDS``, in that order) of ``not_written_again(choices)``."""
        found = {self._kind(Path(path)) for path in self.not_written_again(choices)}
        return [kind for kind in KINDS if kind in found]

    def _kind(self, path: Path) -> str:
        try:
            return file_kind(path.relative_to(self.folder).as_posix(), self.batch)
        except ValueError:  # a whole path in the record that names the folder another way
            return "other"


def earlier_export(target: Path, batch: str, *, fit_inputs: bool = False) -> Optional[EarlierExport]:
    """What an earlier export of ``batch`` left in ``target`` (module docstring), or ``None`` for a folder
    this batch can simply be written into. Only files inside ``target`` are listed, links followed."""
    target = Path(target)
    found = EarlierExport(folder=target, batch=batch)
    record = target / f"{batch}{RECORD_SUFFIX}"
    try:
        if record.is_file():
            found.record = record
            found.when = time.strftime("%Y-%m-%d %H:%M", time.localtime(record.stat().st_mtime))
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                found.complete = data.get("format") == RECORD_FORMAT
                found.when = str(data.get("started") or found.when)
                frames = data.get("frames")
                found.frames = len(frames) if isinstance(frames, list) else 0
                for name in data.get("files") or ():
                    path = target / str(name)
                    if path.is_file() and _inside(path, target) and path.resolve() != record.resolve():
                        found.files.append(path)
        if fit_inputs and (target / FIT_INPUT_FOLDER).is_dir():
            found.fit_inputs = sorted(path for path in (target / FIT_INPUT_FOLDER).glob(FIT_INPUT_PATTERN)
                                      if path.is_file() and _inside(path, target))
    except OSError:
        return None
    return found if found.record is not None or found.fit_inputs else None


def remove_export(found: EarlierExport) -> int:
    """Take the files of an earlier export away (``EarlierExport.to_remove``) and the folders that are empty
    afterwards; returns how many files went. ``OSError`` when one of them cannot be removed."""
    removed = 0
    folders = set()
    for path in found.to_remove:
        path = Path(path)
        if path.is_file():
            path.unlink()
            removed += 1
            folders.add(path.parent)
    for folder in folders:
        if folder != found.folder:
            try:
                folder.rmdir()  # only when nothing else is in it
            except OSError:
                pass
    return removed


def new_subfolder(destination: Path, batch: str) -> Path:
    """``<destination>/<batch>_2`` (``_3``, …): the first that does not exist — a folder of its own for an
    export that is not to touch the earlier one."""
    destination = Path(destination)
    number = 2
    while (destination / f"{batch}_{number}").exists():
        number += 1
    return destination / f"{batch}_{number}"


def other_records(target: Path, batch: str) -> list[str]:
    """The records of other batches in ``target`` (exports of other data into the same folder)."""
    try:
        return sorted(path.name for path in Path(target).glob(f"*{RECORD_SUFFIX}") if path.name != f"{batch}{RECORD_SUFFIX}")
    except OSError:
        return []


__all__ = [
    "EarlierExport", "FIT_INPUT_FOLDER", "KINDS", "NOT_A_FOLDER", "NO_PERMISSION", "NO_SUCH_PLACE", "README", "RECORD_FORMAT",
    "RECORD_SUFFIX", "SEND_SUFFIX", "cannot_write", "chosen_kinds", "earlier_export", "file_kind", "folder_name",
    "may_be_written", "new_subfolder", "other_records", "remove_export",
]
