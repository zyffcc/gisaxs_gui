"""Are files named alike the modules of one detector?  (P03 Lambda 9M: one file per module.)

A file name (``…_mNN.nxs``, ``…mNN.nxs``) only proposes siblings.  They are a module set when every
proposed file holds the same dataset and module shape and they carry pairwise distinct module
positions (``<detector group>/translation/distance``).  Every proposed file has to fit, so the first
one that does not, or the first position held twice, settles the question: nothing more is opened,
and that verdict rests on those files alone, however many files named alike arrive later.  What a
file says is remembered per state of the file (modification time and size), so the files of a folder
are opened once each, not once per file that is named like them.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import h5py
import numpy as np

from .nexus_datasets import ImageStack, _text, image_stacks, natural_key
from .nexus_files import UNFINISHED_AFTER_S, BeingWritten, LeftUnfinished, _stamp, open_nexus

# English templates; presentation translates them with trf(template, **values).
MODULES_UNEQUAL = ("Module files hold different numbers of frames ({detail}; the others {full}): "
                   "frames after frame {frames} are left out")
MODULE_MISSING = "Module file {names} is missing: that part of the detector is left out"
MODULES_MISSING = "Module files {names} are missing: those parts of the detector are left out"
MODULE_UNREADABLE = "Module file {names} cannot be read yet: that part of the detector is left out"
MODULE_LEFT_UNFINISHED = ("Module file {names} was left open for writing and has not changed for a while: the program "
                          "that wrote it seems to have ended without closing it, and that part of the detector is left "
                          "out. Running h5clear -s on a copy of it makes readable what was stored in it")
MODULES_LEFT_UNFINISHED = ("Module files {names} were left open for writing and have not changed for a while: the "
                           "programs that wrote them seem to have ended without closing them, and those parts of the "
                           "detector are left out. Running h5clear -s on copies of them makes readable what was stored "
                           "in them")
SAME_POSITION = ("{first} and {second} are named like modules of one detector but hold the same module position: "
                 "each file is read on its own")
NOT_A_MODULE = ("{name} is named like a detector module of this scan but does not fit the set "
                "(dataset, module shape or module position): each file is read on its own")
STILL_WRITTEN = "{name} is still being written: it is read again when it changes"
LEFT_OPEN = ("{name} was not closed by the program that wrote it and has not changed for a while: it is read as it "
             "is, and its last frame may be incomplete")
MODULES_FEWER = ("{count} module files, but {other} in the same folder has {others}: {names} may be missing here "
                 "(that part of the detector is then left out)")

_MODULE = re.compile(r"^(?P<stem>.+)_m(?P<number>\d{1,3})$", re.IGNORECASE)
_LOOSE_MODULE = re.compile(r"^(?P<stem>.+?)m(?P<number>\d{1,3})$", re.IGNORECASE)
_PATTERNS = (_MODULE, _LOOSE_MODULE)
_MODULE_NUMBER = re.compile(r"^(?P<stem>.+?)_?m\d{1,3}$", re.IGNORECASE)
_CANDIDATES = 3  # other scans of a folder that are asked how many module files they have
_LAMBDA_LAYOUT = re.compile(r"\s*(\d+)\s*[xX]\s*(\d+)\s*")

# (file, dataset, detector group) -> (stamp, what the file says about being a module)
_FACTS: dict[tuple[str, str, str], tuple[tuple, Optional[dict]]] = {}
_FACTS_REMEMBERED = 40_000  # beyond that the oldest quarter is forgotten (and simply asked again)
_INDEXED: dict[int, tuple[tuple[str, ...], dict]] = {}  # id of a folder listing -> (the listing, its index)
_LOCK = threading.Lock()


def module_like(path: Path) -> bool:
    return any(pattern.match(path.stem) for pattern in _PATTERNS)


def scan_stem(source: Any, path: Any = None) -> str:
    """The name of a set of module files without its module number (``scan_00007_m01.nxs`` → ``scan_00007``);
    ``""`` for every other source, and for a source nothing is known of.

    ``source`` is a ``NexusLayout`` or the metadata of a frame read from one (its ``"layout"`` entry says what
    it was read from).  ``path`` is the file the set is named by when it is not the layout's own (any module
    file of a set gives the same name).
    """
    if isinstance(source, Mapping):
        kind, own = source.get("layout"), (source.get("module_files") or [None])[0]
    else:
        kind, own = getattr(source, "kind", None), getattr(source, "path", None)
    named = path if path is not None else own
    found = _MODULE_NUMBER.match(Path(named).stem) if kind == "module_set" and named is not None else None
    return found.group("stem") if found else ""


def _index(names: tuple[str, ...]) -> dict:
    """``{(pattern, stem, suffix): [(module number, label, file name), …]}`` of a folder listing, built once."""
    known = _INDEXED.get(id(names))
    if known is not None and known[0] is names:
        return known[1]
    index: dict = {}
    for name in names:
        stem, suffix = os.path.splitext(name)
        for which, pattern in enumerate(_PATTERNS):
            hit = pattern.match(stem)
            if hit:
                index.setdefault((which, hit.group("stem").casefold(), suffix.casefold()), []).append(
                    (int(hit.group("number")), "m" + hit.group("number"), name))
    sizes: dict = {}
    for (which, stem, suffix), found in index.items():
        found.sort(key=lambda item: (item[0], natural_key(item[2])))
        if len(found) > 1:
            sizes.setdefault((which, suffix), []).append((len(found), stem))
    for found in sizes.values():
        found.sort(key=lambda item: (-item[0], item[1]))
    index[None] = sizes  # {(pattern, suffix): [(files, stem), …]} of the groups of files named alike, the largest first
    if len(_INDEXED) >= 64:
        _INDEXED.clear()
    _INDEXED[id(names)] = (names, index)  # holding the listing keeps its id from being used again
    return index


def proposed(path: Path, names: Iterable[str]) -> list[tuple[int, str, str]]:
    """``(module number, label, file name)`` of the files whose names differ from ``path`` in the module number only.

    In module order; empty when no other file is named like ``path``.  The list is shared: read it only.
    """
    stem, suffix = os.path.splitext(path.name)
    index = _index(names if isinstance(names, tuple) else tuple(names))
    for which, pattern in enumerate(_PATTERNS):
        match = pattern.match(stem)
        found = index.get((which, match.group("stem").casefold(), suffix.casefold()), ()) if match else ()
        if len(found) > 1:
            return found
    return []


def _group_of(path: Path, names: Iterable[str]) -> tuple[dict, Optional[tuple]]:
    """``(index of the listing, the key of the group of files ``path`` is named like)``; no key: it has none."""
    stem, suffix = os.path.splitext(path.name)
    index = _index(names if isinstance(names, tuple) else tuple(names))
    for which, pattern in enumerate(_PATTERNS):
        match = pattern.match(stem)
        key = (which, match.group("stem").casefold(), suffix.casefold()) if match else None
        if key is not None and len(index.get(key, ())) > 1:
            return index, key
    return index, None


def larger_scans(path: Path, names: Iterable[str], count: int) -> list[list[tuple[int, str, str]]]:
    """The files of the other scans of the folder that are named like module files and are more than ``count``
    (the largest scans first, a few of them): what they are is for their content to say."""
    index, key = _group_of(path, names)
    if key is None:
        return []
    others = [stem for files, stem in index[None].get((key[0], key[2]), ()) if files > count and stem != key[1]]
    return [index[(key[0], stem, key[2])] for stem in others[:_CANDIDATES]]


def largest_other_scan(path: Path, names: Iterable[str]) -> int:
    """How many files the largest other scan of the folder has that is named like ``path``'s (0: there is none)."""
    index, key = _group_of(path, names)
    if key is None:
        return 0
    return next((files for files, stem in index[None].get((key[0], key[2]), ()) if stem != key[1]), 0)


def module_position(group: Any) -> Optional[tuple[int, int]]:
    """``(row, column)`` offset of a detector module on the stitched canvas (``translation/distance``)."""
    try:
        node = group.get("translation/distance") if group is not None else None
        values = np.asarray(node[()]).reshape(-1) if isinstance(node, h5py.Dataset) else np.empty(0)
        if values.size == 0:
            return None
        return (int(values[1]) if values.size > 1 else 0, int(values[0]))
    except (KeyError, OSError, ValueError, TypeError):
        return None


def lambda_layout(group: Any) -> Optional[tuple[int, int]]:
    """``(columns, rows)`` of a module as the Lambda server writes it in ``layout`` (``1556X516``)."""
    try:
        node = group.get("layout") if group is not None else None
        match = _LAMBDA_LAYOUT.fullmatch(_text(node[()])) if isinstance(node, h5py.Dataset) else None
        return (int(match.group(1)), int(match.group(2))) if match else None
    except (KeyError, OSError, ValueError, TypeError):
        return None


def lambda_file(group: Any) -> bool:
    """Written by the Lambda server: module position, ``layout`` or a name that says so."""
    if module_position(group) is not None or lambda_layout(group) is not None:
        return True
    for name in ("description", "type", "local_name"):
        try:
            node = group.get(name) if group is not None else None
            if isinstance(node, h5py.Dataset) and "lambda" in _text(node[()]).lower():
                return True
        except (KeyError, OSError, ValueError, TypeError):
            continue
    return False


def _module_facts(handle: h5py.File, chosen: ImageStack) -> Optional[dict]:
    """What one file says about being a module like ``chosen`` (``None``: it does not hold that dataset)."""
    node = handle.get(chosen.path)
    if not isinstance(node, h5py.Dataset) or node.ndim not in (2, 3):
        return None
    return {"frames": int(node.shape[0]) if node.ndim == 3 else 1,
            "shape": tuple(int(size) for size in node.shape[-2:]),
            "position": module_position(handle.get(chosen.detector_group)), "writing": bool(handle.swmr_mode)}


def remember_facts(member: Path, chosen: ImageStack, stamp: tuple, facts: Optional[dict]) -> None:
    """What ``member`` said when it was opened at ``stamp``: the files named like it need not open it again."""
    with _LOCK:
        if len(_FACTS) >= _FACTS_REMEMBERED:
            for old in list(_FACTS)[: _FACTS_REMEMBERED // 4]:
                del _FACTS[old]
        _FACTS[(os.path.normcase(str(member)), chosen.path, chosen.detector_group)] = (stamp, facts)


def forget_facts(path: Optional[Path] = None) -> None:
    with _LOCK:
        if path is None:
            _FACTS.clear()
            _INDEXED.clear()
            return
        name = os.path.normcase(str(path))
        for key in [key for key in _FACTS if key[0] == name]:
            del _FACTS[key]


def _sibling_facts(member: Path, chosen: ImageStack) -> tuple[tuple, Optional[dict]]:
    """``(stamp, module facts)`` of a file named like the one inspected; opened once per state of the file."""
    stamp = _stamp(member)
    with _LOCK:
        known = _FACTS.get((os.path.normcase(str(member)), chosen.path, chosen.detector_group))
    if known is not None and known[0] == stamp:
        return known
    with open_nexus(member) as handle:
        facts = _module_facts(handle, chosen)
    remember_facts(member, chosen, stamp, facts)
    return stamp, facts


def _reads(member: Path) -> bool:
    """The library can walk the file: one it cannot is damaged or half written, not a file of another kind."""
    try:
        with open_nexus(member) as handle:
            image_stacks(handle)
        return True
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001 - RuntimeError for metadata that contradict each other, KeyError, …
        return False


def still_written(members: Iterable[Path], entry: Any) -> str:
    """The remark for files that are marked as open for writing (in the mode in which others may read):
    ``STILL_WRITTEN``, or ``LEFT_OPEN`` once none of them has changed for a while (the mark stays in a file
    whose writer ended without closing it).  ``entry`` is told when to ask again (``expires``)."""
    now, ends = time.time(), []
    for member in members:
        try:
            ends.append(os.stat(member).st_mtime + UNFINISHED_AFTER_S)
        except OSError:
            continue
    if ends and max(ends) > now:
        _ask_again(entry, max(ends))
        return STILL_WRITTEN
    return LEFT_OPEN if ends else STILL_WRITTEN


def _ask_again(entry: Any, when: float) -> None:
    """The cached layout ``entry`` is to be asked again at ``when`` (the earliest such moment counts)."""
    entry.expires = when if entry.expires is None else min(entry.expires, when)


def _held(entry: Any, member: Path) -> None:
    """A module file its writer holds may turn out to be left by a writer that ended: the layout is asked again
    once the file has not changed for ``UNFINISHED_AFTER_S`` (nothing else would change it, and the set would
    say for ever that the file "cannot be read yet")."""
    try:
        _ask_again(entry, os.stat(member).st_mtime + UNFINISHED_AFTER_S)
    except OSError:
        pass


def _no_set(entry: Any, warnings: list, witnesses: tuple, template: str, values: dict) -> None:
    """The files are no module set whatever else is named like them: the verdict rests on ``witnesses`` alone."""
    entry.watched, entry.watch, entry.proposal = witnesses, False, ()
    warnings.append((template, values))


def module_set(path: Path, chosen: ImageStack, own: dict, own_stamp: tuple, siblings: list, entry: Any,
               warnings: list) -> Optional[tuple[tuple[Path, ...], tuple[int, ...], bool, tuple[Path, ...]]]:
    """``(members, frames of each, complete, unreadable files)`` when the proposed files are one detector.

    ``entry`` (the cache entry being built) is told which files the answer rests on: ``watched``
    (their stamps) and, for a refusal that no further file can change, ``watch = False``.  A proposed
    file that cannot be read (still being copied or written, or damaged: the library fails on it) is
    a part that is missing for now, never the reason to read every file on its own.  One whose writer
    ended without closing it (``LeftUnfinished``) is named as such, with how it can be repaired.
    """
    usable: list[tuple[int, str, Path, dict]] = []
    unread: list[str] = []
    left: list[str] = []  # files a writer that ended left unfinished
    pending: list[Path] = []
    stamps: list[tuple] = []
    held: dict[tuple[int, int], tuple[str, tuple]] = {}  # module position -> (file that holds it, its stamp)
    for number, label, name in siblings:
        member = path.parent / name
        if name == path.name:
            stamp, facts = own_stamp, own
        else:
            try:
                stamp, facts = _sibling_facts(member, chosen)
                fits = facts is not None and facts["shape"] == chosen.frame_shape and facts["position"] is not None
                if not fits and not _reads(member):  # damaged: what it says about itself counts for nothing
                    raise OSError(f"{name} cannot be read")
            except MemoryError:
                raise
            except Exception as error:  # noqa: BLE001 - still being copied or written, or damaged: left out, and said
                stamps.append(_stamp(member))
                (left if isinstance(error, LeftUnfinished) else unread).append(label)
                pending.append(member)
                if isinstance(error, BeingWritten) and not isinstance(error, LeftUnfinished):
                    _held(entry, member)
                continue
        stamps.append(stamp)
        if facts is None or facts["shape"] != chosen.frame_shape or facts["position"] is None:
            return _no_set(entry, warnings, (stamp,), NOT_A_MODULE, {"name": name})
        other = held.setdefault(facts["position"], (name, stamp))
        if other[0] != name:
            return _no_set(entry, warnings, (other[1], stamp), SAME_POSITION, {"first": other[0], "second": name})
        usable.append((number, label, member, facts))
    entry.watched = tuple(stamps)
    if unread:
        warnings.append((MODULE_UNREADABLE, {"names": ", ".join(unread)}))
    if left:
        warnings.append((MODULES_LEFT_UNFINISHED if len(left) > 1 else MODULE_LEFT_UNFINISHED, {"names": ", ".join(left)}))
    if len(usable) < 2:
        return None  # the others cannot be read yet: asked again when they change
    numbers = [number for number, _l, _m, _f in usable] + [int(label[1:]) for label in unread + left]
    width = len(usable[0][1]) - 1
    gaps = sorted(set(range(min(1, min(numbers)), max(numbers) + 1)) - set(numbers))
    if gaps:
        names = ", ".join(f"m{number:0{width}d}" for number in gaps)
        warnings.append((MODULES_MISSING if len(gaps) > 1 else MODULE_MISSING, {"names": names}))
    counts = tuple(facts["frames"] for _n, _l, _m, facts in usable)
    if len(set(counts)) > 1:
        short = ", ".join(f"{label}: {count}" for (_n, label, _m, _f), count in zip(usable, counts) if count < max(counts))
        warnings.append((MODULES_UNEQUAL, {"detail": short, "full": max(counts), "frames": min(counts)}))
    marked = [member for _n, _l, member, facts in usable if facts["writing"]]  # as open for writing
    writing = bool(marked)
    if writing:
        warnings.append((still_written(marked, entry), {"name": usable[0][2].name}))
    members = tuple(member for _n, _l, member, _f in usable)
    return members, counts, not (gaps or unread or left or writing or len(set(counts)) > 1), tuple(pending)
