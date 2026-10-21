"""What a NeXus / HDF5 source consists of: one inspection for every reader and every file list.

``inspect_nexus`` opens the file(s) once and says which layout they have:

* ``single_frame`` – one file, one frame;
* ``stack`` – one file holding the whole series;
* ``module_set`` – one file per detector module, each holding all frames (P03 Lambda 9M): the frames
  are stitched from the module positions;
* ``chunked_master`` – a master whose frames are in data files (Dectris ``data_NNNNNN`` links, a
  virtual dataset, one external link);
* ``no_image`` – an HDF5 file without a detector image.

The content decides, a file name only proposes: ``…_mNN`` siblings are a module set only when all of
them hold the same dataset and module shape and carry pairwise distinct module positions
(``<detector group>/translation/distance``; the decision is in ``nexus_modules``).  One file that does
not fit, or two files with the same position, settle that the files are no set: nothing more is opened,
and the verdict stands while those files are unchanged.  ``cached_layout`` repeats the inspection only
when a file of the source changed (modification time and size of every member; for module-like names
also the files next to it), so asking again costs ``os.stat`` calls.  A failed inspection is never stored.

What cannot be known from one source alone is said as a remark, never decided silently:

* a module set with fewer module files than another scan of its folder (a missing LAST module leaves no
  gap in the numbering) names the modules it may lack (``MODULES_FEWER``);
* a 3-D dataset that looks like ``(rows, columns, frames)`` is read as it always was and says so
  (``FRAMES_LAST``);
* a file that is marked as being written and has not changed for a while was left by a writer that
  ended (``LEFT_OPEN``; one that cannot be read at all: ``LeftUnfinished`` in ``nexus_files``, and as
  a module file of a set ``MODULE_LEFT_UNFINISHED``, both with how the file can be repaired).
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .nexus_datasets import (
    DATASET_IS_GROUP,
    DATASET_NOT_FRAMES,
    RANK_ASKED,
    RANK_AUXILIARY,
    ImageStack,
    image_stacks,
    stored_frames_last,
    tied,
)
from .nexus_files import (
    BEING_WRITTEN,
    FRAME_BEYOND,
    HDF5_SUFFIXES,
    LEFT_UNFINISHED,
    NO_IMAGE,
    BeingWritten,
    LeftUnfinished,
    _LISTINGS,
    _canonical,
    _folder_names,
    _signature,
    _stamp,
    is_hdf5,
    one_listing_per_folder,
    open_nexus,
)
from .nexus_modules import (
    LEFT_OPEN,
    MODULE_LEFT_UNFINISHED,
    MODULE_MISSING,
    MODULE_UNREADABLE,
    MODULES_FEWER,
    MODULES_LEFT_UNFINISHED,
    MODULES_MISSING,
    MODULES_UNEQUAL,
    NOT_A_MODULE,
    SAME_POSITION,
    STILL_WRITTEN,
    forget_facts,
    lambda_file,
    lambda_layout,
    larger_scans,
    largest_other_scan,
    module_like,
    module_position,
    module_set,
    proposed,
    remember_facts,
    scan_stem,
    still_written,
)

P03_LAMBDA, AS_STORED = "p03_lambda", "as_stored"

# English templates; presentation translates them with trf(template, **values).
SUMMARY_TEMPLATES = (
    "{modules} module files · {frames} frames · stitched",
    "{modules} module files · 1 frame · stitched",
    "master + {files} data files · {frames} frames",
    "master + {files} data files · 1 frame",
    "master + 1 data file · {frames} frames",
    "master + 1 data file · 1 frame",
    "1 file · {frames} frames",
    "1 file · 1 frame",
    "no detector image",
)
SEVERAL_DATASETS = ("{count} image datasets in this file: {dataset} is shown, not {others} "
                    "(Format Converter and Geometry Calibration let you choose the dataset)")
FRAMES_LAST = ("{dataset} is read as {frames} frames of {rows} × {columns} pixels — if these are {columns} frames "
               "stored last, this file is not supported yet")
_OTHERS_NAMED = 3  # the other datasets a remark names before it says "…"
ONLY_AUXILIARY = ("{dataset} is named like a mask or a correction, not like a detector image: "
                  "it is read because this file holds no other image")
DATASET_NOT_FOUND = "Dataset {wanted} is not in this file: {dataset} is read instead"
DATA_FILE_MISSING = "Data file {names} is missing: frames after frame {frames} cannot be read"
DATA_FILES_MISSING = "Data files {names} are missing: frames after frame {frames} cannot be read"
FRAME_GRID = "A {grid} grid of frames is read in storage order as {frames} frames"
WARNING_TEMPLATES = (MODULES_UNEQUAL, MODULE_MISSING, MODULES_MISSING, MODULE_UNREADABLE, MODULE_LEFT_UNFINISHED,
                     MODULES_LEFT_UNFINISHED, SAME_POSITION, NOT_A_MODULE, SEVERAL_DATASETS, ONLY_AUXILIARY,
                     DATASET_NOT_FOUND, DATA_FILE_MISSING, DATA_FILES_MISSING, FRAME_GRID, STILL_WRITTEN, LEFT_OPEN,
                     MODULES_FEWER, FRAMES_LAST)
# Sentences a read fails with: BeingWritten and LeftUnfinished, ValueError (no image; a dataset asked for that
# holds no frames) and IndexError (a frame the source does not have; {frame} is 1-based).
ERROR_TEMPLATES = (BEING_WRITTEN, LEFT_UNFINISHED, NO_IMAGE, FRAME_BEYOND, DATASET_NOT_FRAMES, DATASET_IS_GROUP)

_DATA_FILE = re.compile(r"^(?P<stem>.+)_data_\d{6}$", re.IGNORECASE)


@dataclass(frozen=True, eq=False)
class NexusLayout:
    """One detector source in NeXus / HDF5 files, as ``inspect_nexus`` found it."""

    kind: str                                   # single_frame | stack | module_set | chunked_master | no_image
    path: Path                                  # the file that stands for the source: first module / master / the file
    members: tuple[Path, ...]                   # every file it consists of
    dataset: str = ""                           # HDF5 path of the image dataset (the first chunk of a chunk family)
    chunks: tuple[tuple[str, int], ...] = ()    # ((dataset path, frames), …) in frame order, when the frames are chunked
    frames: int = 0                             # frames that can be read (every module has them)
    frame_shape: tuple[int, int] = (0, 0)       # stored (rows, columns) of one frame of one member
    member_frames: tuple[int, ...] = ()         # frames of every module / of every chunk
    alternatives: tuple[str, ...] = ()          # the other image datasets of the file
    decided_by: str = ""                        # one English sentence
    warnings: tuple[tuple[str, dict], ...] = ()  # (template, values) pairs
    complete: bool = True                       # False: modules or data files missing / unequal / still written
    orientation: str = AS_STORED                # p03_lambda | as_stored
    signature: tuple = ()                       # (path, mtime_ns, size) of every member
    lead_shape: tuple[int, ...] = ()            # dataset axes before the frame: () | (N,) | (points, frames)
    dtype: str = ""                             # stored dtype
    detector_group: str = ""                    # HDF5 group with the mask, module position and header values
    entry: str = ""                             # the NXentry (or top-level group) the dataset is in
    pending: tuple[Path, ...] = ()              # module files of the set that cannot be read yet
    behind: tuple[Path, ...] = ()               # data files of a master behind one that is missing: out of reach

    def summary(self) -> tuple[str, dict]:
        """``(template, values)`` saying what the source is, e.g. 11 module files · 240 frames · stitched."""
        frames = "1 frame" if self.frames == 1 else "{frames} frames"
        if self.kind == "module_set":
            return f"{{modules}} module files · {frames} · stitched", self._values(modules=len(self.members))
        if self.kind == "chunked_master":
            files = len(self.members) - 1
            head = "master + 1 data file" if files == 1 else "master + {files} data files"
            return f"{head} · {frames}", self._values(**({} if files == 1 else {"files": files}))
        if self.kind == "no_image":
            return "no detector image", {}
        return f"1 file · {frames}", self._values()

    def _values(self, **values: Any) -> dict:
        return values if self.frames == 1 else {**values, "frames": self.frames}

    def summary_text(self) -> str:
        template, values = self.summary()
        return template.format(**values)

    def warning_texts(self) -> tuple[str, ...]:
        return tuple(template.format(**values) for template, values in self.warnings)


@dataclass
class _Entry:
    layout: NexusLayout
    proposal: tuple[str, ...] = ()       # the sibling names the decision was made with
    watched: tuple = ()                  # signature of those siblings
    folder_stamp: Optional[int] = None   # folder mtime the proposal is known to be valid for
    missing: tuple[str, ...] = ()        # data files that were not there
    watch: bool = False                  # a new file next to it can change the layout
    expires: Optional[float] = None      # when a file marked as being written counts as left by its writer
    neighbours: int = 0                  # module set: files of the largest other scan of the folder (by name)


_CACHE: dict[tuple[str, str], _Entry] = {}
_LOCK = threading.Lock()
_REMEMBERED = 20_000  # files; beyond that the oldest quarter is forgotten (and simply inspected again)


def _cache_key(path: Path, dataset: Optional[str]) -> tuple[str, str]:
    return (os.path.normcase(str(path)), dataset or "")


def _missing_files(names: Iterable[str], frames: int) -> tuple[str, dict]:
    names = [os.path.basename(name) for name in names]
    return (DATA_FILES_MISSING if len(names) > 1 else DATA_FILE_MISSING), {"names": ", ".join(names), "frames": frames}


def _choose(stacks: list[ImageStack], dataset: Optional[str], warnings: list) -> Optional[ImageStack]:
    if not stacks:
        return None
    best = stacks[0]
    if best.rank == RANK_ASKED:
        return best
    if dataset:
        warnings.append((DATASET_NOT_FOUND, {"wanted": dataset, "dataset": best.path}))
    others = [stack.path for stack in tied(stacks)[1:]]
    if others:  # the file does not say which is the image: both are named, and where one can choose
        named = ", ".join(others[:_OTHERS_NAMED]) + (" …" if len(others) > _OTHERS_NAMED else "")
        warnings.append((SEVERAL_DATASETS, {"count": len(others) + 1, "dataset": best.path, "others": named}))
    if best.rank == RANK_AUXILIARY:
        warnings.append((ONLY_AUXILIARY, {"dataset": best.path}))
    return best


def _inspect(path: Path, dataset: Optional[str] = None) -> _Entry:
    folder = str(path.parent)
    own_stamp = _stamp(path)
    warnings: list[tuple[str, dict]] = []
    with open_nexus(path) as handle:
        stacks, lost, why_none = image_stacks(handle, wanted=dataset)
        writing = bool(handle.swmr_mode)
        chosen = _choose(stacks, dataset, warnings)
        if chosen is not None:
            group = handle.get(chosen.detector_group)
            own = {"frames": chosen.frames, "shape": chosen.frame_shape, "position": module_position(group),
                   "writing": writing}
            from_lambda = lambda_file(group)
            frames_last = not chosen.chunks and stored_frames_last(handle.get(chosen.path))
    if chosen is None:
        if lost:
            warnings.append(_missing_files(lost, 0))
        layout = NexusLayout("no_image", path, (path,), decided_by=why_none or "no image-shaped dataset in the file",
                             warnings=tuple(warnings), complete=not lost, signature=(own_stamp,))
        return _Entry(layout, missing=tuple(lost))
    # Masks and corrections are not "other images" of a file that has an image.
    alternatives = tuple(stack.path for stack in stacks
                         if stack is not chosen and (stack.rank < RANK_AUXILIARY or chosen.rank == RANK_AUXILIARY))
    plain = not chosen.chunks and not chosen.files and not chosen.missing and len(chosen.lead_shape) <= 1
    named_alike = plain and module_like(path)
    entry = _Entry(None, watch=named_alike and own["position"] is not None)  # type: ignore[arg-type]
    siblings: list = []
    if named_alike:
        remember_facts(path, chosen, own_stamp, own)  # the files named like this one need not open it again
    if named_alike and (entry.watch or from_lambda):
        names, entry.folder_stamp = _folder_names(folder)
        siblings = proposed(path, names)
        entry.proposal = tuple(name for _number, _label, name in siblings)
    if siblings and entry.watch:
        found = module_set(path, chosen, own, own_stamp, siblings, entry, warnings)
        if found is not None:
            members, counts, complete, pending = found
            entry.neighbours = largest_other_scan(path, names)
            _fewer_modules(path, names, siblings, chosen, len(members) + len(pending), entry.neighbours, warnings)
            layout = NexusLayout(
                "module_set", members[0], members, chosen.path, (), min(counts), chosen.frame_shape, counts,
                alternatives,
                f"{len(members)} files named alike up to the module number, each with its own module position "
                f"(translation) and a {chosen.frame_shape[0]}×{chosen.frame_shape[1]} module",
                tuple(warnings), complete, P03_LAMBDA, _signature_of(members, entry), (min(counts),), chosen.dtype,
                chosen.detector_group, chosen.entry, pending)
            entry.layout = layout
            return entry
    elif siblings:
        warnings.append((NOT_A_MODULE, {"name": path.name}))
    if chosen.missing:
        warnings.append(_missing_files(chosen.missing, chosen.frames))
    if len(chosen.lead_shape) > 1:
        warnings.append((FRAME_GRID, {"grid": "×".join(str(size) for size in chosen.lead_shape), "frames": chosen.frames}))
    if frames_last:
        warnings.append((FRAMES_LAST, {"dataset": chosen.path, "frames": chosen.frames, "rows": chosen.frame_shape[0],
                                       "columns": chosen.frame_shape[1]}))
    if writing:
        warnings.append((still_written([path], entry), {"name": path.name}))
    members = (path,) + tuple(Path(name) for name in chosen.files)
    kind = "chunked_master" if chosen.files or chosen.missing else ("single_frame" if chosen.frames == 1 else "stack")
    entry.missing = tuple(chosen.missing)
    incomplete = bool(chosen.missing) or writing or any(template == MODULE_UNREADABLE for template, _v in warnings)
    entry.layout = NexusLayout(
        kind, path, members, chosen.path, chosen.chunks, chosen.frames, chosen.frame_shape,
        tuple(count for _path, count in chosen.chunks) or (chosen.frames,), alternatives, chosen.how,
        tuple(warnings), not incomplete, P03_LAMBDA if from_lambda else AS_STORED,
        (own_stamp,) + _signature(members[1:]), chosen.lead_shape, chosen.dtype, chosen.detector_group, chosen.entry,
        behind=tuple(Path(name) for name in chosen.behind))
    return entry


def _fewer_modules(path: Path, names: tuple, siblings: list, chosen: ImageStack, count: int, largest: int,
                   warnings: list) -> None:
    """A last module that is missing leaves no gap in the numbering.  When another scan of the folder is a
    module set of the same detector (dataset, module shape) with more module files, say which ones this
    set lacks.  The other scan is asked through ``cached_layout``: its files are opened once all the same."""
    if largest <= count:
        return
    mine = {number for number, _label, _name in siblings}
    for files in larger_scans(path, names, count):
        try:
            other = cached_layout(path.parent / files[0][2], chosen.path if chosen.rank == RANK_ASKED else None)
        except Exception:  # noqa: BLE001 - a scan that cannot be read says nothing about this one
            continue
        total = len(other.members) + len(other.pending)
        if (other.kind != "module_set" or other.dataset != chosen.path or other.frame_shape != chosen.frame_shape
                or total <= count):
            continue
        lacking = [label for number, label, _name in files if number not in mine]
        if lacking:
            warnings.append((MODULES_FEWER, {"count": count, "other": scan_stem(other), "others": total,
                                             "names": ", ".join(lacking)}))
            return


def _signature_of(members: tuple[Path, ...], entry: _Entry) -> tuple:
    """The stamps taken before each member was opened (a change in between is seen at the next check)."""
    before = {stamp[0]: stamp for stamp in entry.watched}
    return tuple(before.get(str(member)) or _stamp(member) for member in members)


def inspect_nexus(path: str | Path, dataset: Optional[str] = None) -> NexusLayout:
    """Open the file (and the files it belongs with) and say what the source is.

    The layout is not remembered (``cached_layout`` does that).  ``dataset`` picks the image dataset by
    its HDF5 path, wherever it is; the first of a family of numbered datasets stands for the family.
    Raises the library's ``OSError`` for a file that cannot be opened, ``BeingWritten`` for one its
    writer holds, and ``ValueError`` for a ``dataset`` that is in the file but holds no frames.
    """
    return _inspect(_canonical(path), dataset).layout


def _still_valid(entry: _Entry, path: Path) -> bool:
    # Every member, and every file next to it the decision was made with: unchanged since they were opened.
    if any(_stamp(stamp[0]) != stamp for stamp in dict.fromkeys(entry.layout.signature + entry.watched)):
        return False
    if any(os.path.exists(name) for name in entry.missing):
        return False
    if entry.expires is not None and time.time() >= entry.expires:
        return False  # marked as being written, and unchanged for a while now: said differently
    if not entry.watch:
        return True
    folder = str(path.parent)
    try:
        if entry.folder_stamp is not None and os.stat(folder).st_mtime_ns == entry.folder_stamp:
            return True
    except OSError:
        return False
    names, settled = _folder_names(folder)
    if tuple(name for _number, _label, name in proposed(path, names)) != entry.proposal:
        return False
    if entry.layout.kind == "module_set" and largest_other_scan(path, names) != entry.neighbours:
        return False  # another scan of the folder has more (or fewer) files now: it may say a module is missing
    entry.folder_stamp = settled
    return True


def cached_layout(path: str | Path, dataset: Optional[str] = None) -> NexusLayout:
    """``inspect_nexus`` once per state of the files: nothing is opened while nothing changed."""
    path = _canonical(path)
    key = _cache_key(path, dataset)
    with _LOCK:
        entry = _CACHE.get(key)
    if entry is not None and _still_valid(entry, path):
        return entry.layout
    entry = _inspect(path, dataset)
    with _LOCK:
        if len(_CACHE) >= _REMEMBERED:
            for old in list(_CACHE)[: _REMEMBERED // 4]:
                del _CACHE[old]
        _CACHE[key] = entry
        if entry.layout.kind == "module_set":  # every module file stands for the same set
            for member in entry.layout.members:
                _CACHE[_cache_key(member, dataset)] = entry
    return entry.layout


def last_known_layout(path: str | Path, dataset: Optional[str] = None) -> Optional[NexusLayout]:
    """The layout of the last inspection that worked, even if the files changed since (``None``: never read)."""
    with _LOCK:
        entry = _CACHE.get(_cache_key(_canonical(path), dataset))
    return entry.layout if entry is not None else None


def forget_layout(path: Optional[str | Path] = None) -> None:
    """Drop what is remembered about ``path`` (about every file when ``None``)."""
    with _LOCK:
        forget_facts(None if path is None else _canonical(path))
        if path is None:
            _CACHE.clear()
            _LISTINGS.clear()
            return
        wanted = _canonical(path)
        name = os.path.normcase(str(wanted))
        for key in [key for key, entry in _CACHE.items() if key[0] == name or wanted in entry.layout.members]:
            del _CACHE[key]
        _LISTINGS.pop(str(wanted.parent), None)


def group_sources(paths: Iterable[str | Path], *, keep_without_image: bool = False) -> list[Path]:
    """One path per detector source, in the order given: what a file list shows.

    The files of a module set become its first module (also one that cannot be read yet), the data
    files of a master the master: every file it links to that is there, also one behind a data file
    that is missing (its frames are out of the master's reach; listed without its master it is a file
    of its own, and can be read).  An
    HDF5 file without a detector image (model weights, results) is left out unless
    ``keep_without_image``; one that cannot be opened now stays a row of its own.  Paths that are not
    HDF5 pass through as they are; HDF5 paths come back absolute.  Each file is opened at most once
    and each folder listed once.
    """
    rows: list[Path] = []
    listed: set[str] = set()
    stands_for: dict[str, Path] = {}
    out_of_reach: dict[str, Path] = {}  # a data file behind a missing one → its master
    folders: dict[str, str] = {}

    def add(path: Path) -> None:
        key = os.path.normcase(str(path))
        if key not in listed:
            listed.add(key)
            rows.append(path)

    def inspect(path: Path) -> Optional[NexusLayout]:
        try:
            layout = cached_layout(path)
        except Exception:  # noqa: BLE001 - unreadable now: reading it will say why
            return None
        if layout.kind != "no_image" or keep_without_image:
            stands_for.setdefault(os.path.normcase(str(layout.path)), layout.path)
            for member in layout.members + layout.pending:  # also a file that was first met on its own
                if member != layout.path:
                    stands_for[os.path.normcase(str(member))] = layout.path
            for member in layout.behind:
                out_of_reach[os.path.normcase(str(member))] = layout.path
        return layout

    with one_listing_per_folder():
        for item in paths:
            text = os.fspath(item)
            if not text.lower().endswith(HDF5_SUFFIXES):
                add(Path(item))
                continue
            folder, name = os.path.split(os.path.expanduser(text))
            if folder not in folders:
                folders[folder] = str(_canonical(os.path.join(folder, name)).parent)
            key = os.path.normcase(os.path.join(folders[folder], name))
            if key in stands_for:  # a module or data file of a source that is listed already
                add(stands_for[key])
                continue
            path = Path(folders[folder], name)
            chunk = _DATA_FILE.match(path.stem)
            if chunk:  # a data file listed before its master: ask the master first
                for other in _folder_names(folders[folder])[0]:
                    if other.rsplit("_master.", 1)[0] == chunk.group("stem") and "_master." in other and is_hdf5(other):
                        inspect(path.parent / other)
                if key in stands_for:
                    add(stands_for[key])
                    continue
            layout = inspect(path)
            if layout is None:
                add(path)
            elif layout.kind != "no_image" or keep_without_image:
                add(layout.path)
    # A data file that was listed before the master that holds it is not a row of its own; nor is one the
    # master cannot reach, when that master is listed.
    masters = {os.path.normcase(str(row)) for row in rows}
    return [row for row in rows if stands_for.get(os.path.normcase(str(row)), row) == row
            and os.path.normcase(str(out_of_reach.get(os.path.normcase(str(row)), ""))) not in masters]


__all__ = [
    "AS_STORED",
    "BeingWritten",
    "ERROR_TEMPLATES",
    "HDF5_SUFFIXES",
    "LeftUnfinished",
    "NexusLayout",
    "P03_LAMBDA",
    "SUMMARY_TEMPLATES",
    "WARNING_TEMPLATES",
    "cached_layout",
    "forget_layout",
    "group_sources",
    "inspect_nexus",
    "is_hdf5",
    "lambda_layout",
    "last_known_layout",
    "module_position",
    "open_nexus",
    "scan_stem",
]
