"""Which datasets of an open NeXus / HDF5 file are detector images, best first.

The content decides, in this order: the dataset the caller names, the file's default plot (root
``@default`` → NXentry ``@default`` → NXdata ``@signal``), ``data`` of an NXdetector group, frames
stored as numbered datasets (Dectris ``data_NNNNNN`` links, GIMaP's own ``image_NNNNNN``) or a
virtual dataset, the signal of an NXdata group, any other image-shaped numeric dataset.  Soft and
external links are followed.

A dataset named like a mask, flat field, dark, variance or error is never chosen while the file
holds anything else that is an image: it comes last, and is read (with a remark) only from a file
that holds nothing else.  ``dark_subtracted`` or ``flatfield_corrected`` say what was done to an
image and are images.  Nothing under ``detectorSpecific`` is offered.  ``(H, W, 3|4)`` pictures,
line data, long tables and the weights of a neural-network model are not frames.  Numbered datasets
of one group are one series while their frames have one size; where the size changes another series
begins.  No pixel is read here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import h5py
import numpy as np

DEFAULT_DATASET_PATH = "/entry/instrument/detector/data"
# Frames stored as numbered datasets of one group: Dectris data_000001 …, Format Converter's image_000001 …
CHUNK_NAME = re.compile(r"^(?P<prefix>data|image)_(?P<number>\d{6})$")
RANK_ASKED, RANK_DEFAULT, RANK_DETECTOR, RANK_CHUNKS, RANK_SIGNAL, RANK_OTHER, RANK_AUXILIARY = 0, 1, 2, 3, 4, 5, 6

# English templates of the two refusals of a dataset the caller names (presentation translates them).
DATASET_NOT_FRAMES = "Dataset {dataset} cannot be read as detector frames: its shape is {shape} and its type {dtype}."
DATASET_IS_GROUP = "{dataset} is a group of the file, not a dataset."
MODEL_FILE = "the file holds the weights of a neural-network model (Keras / TensorFlow), not detector data"

_AUXILIARY = ("flatfield", "dark", "variance", "error", "sigma", "weight", "normalisation", "normalization")
# What was done to an image with a correction: dark_subtracted, flatfield_corrected, sigma_clipped, weighted_sum.
_APPLIED = ("subtracted", "corrected", "removed", "clipped", "normalized", "normalised", "weighted", "divided",
            "applied", "scaled", "compensated", "rejected", "filtered", "cleaned")
_DETECTOR_WORDS = ("detector", "pilatus", "eiger", "lambda", "lmbd", "image", "frame", "saxs", "waxs")
_MODEL_ATTRIBUTES = ("keras_version", "model_config", "training_config", "layer_names")
_MODEL_GROUPS = ("model_weights", "optimizer_weights")
_MODEL_VARIABLE = re.compile(r":\d+$|/vars/\d+$")  # kernel:0 (TensorFlow variables), layers/dense/vars/0 (Keras 3)
_MIN_SIDE = 4          # narrower than this is line data, also at a named place
_MIN_UNNAMED_SIDE = 16  # a dataset nothing points at must look like an image, not like a table
_MAX_UNNAMED_ASPECT = 16  # … and a table is long and narrow (500 samples × 16 parameters)
_DETECTOR_SIDE = 64    # fewer pixels along a side is no area detector (frames stored last: ``stored_frames_last``)
_MAX_DEPTH = 12
_NATURAL = re.compile(r"(\d+)")


class AmbiguousDatasetError(ValueError):
    def __init__(self, paths: list[str]):
        self.paths = paths
        super().__init__("Multiple detector image datasets are plausible: " + ", ".join(paths))


@dataclass(frozen=True)
class ImageStack:
    """One image dataset (or one family of ``data_NNNNNN`` chunks) of an open file."""

    path: str                       # HDF5 path as reached from the root; the first chunk of a family
    lead_shape: tuple[int, ...]     # () one image, (N,) a stack, (points, frames) a scan
    frame_shape: tuple[int, int]    # stored (rows, columns) of one frame
    dtype: str
    rank: int
    how: str                        # why this dataset, in words
    detector_group: str             # where its mask, module position and header values are looked up
    entry: str
    chunks: tuple[tuple[str, int], ...] = ()   # ((dataset path, frames), …) in frame order: a chunk family
    files: tuple[str, ...] = ()     # other files the pixels are in (external link targets, virtual sources)
    missing: tuple[str, ...] = ()   # data files that are not there
    frames: int = 0                 # frames that can be read (they stop before a missing data file)
    behind: tuple[str, ...] = ()    # data files that are there behind a missing one: linked to, but out of reach


@dataclass
class _Found:
    datasets: list
    detectors: list
    dangling: list


def _text(value: Any) -> str:
    """An attribute or a small dataset as text: the first element of an array, bytes decoded, "" for nothing."""
    if value is None or isinstance(value, h5py.Empty):
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):  # a numpy scalar is not an array to take apart again
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _marks_signal(value: Any) -> bool:
    """The mark NeXus used before 2014, on the dataset itself: ``signal = 1`` (a number, or the text "1")."""
    try:
        return float(_text(value).strip()) == 1.0
    except ValueError:
        return False


def _nx_class(obj: Any) -> str:
    try:
        return _text(obj.attrs.get("NX_class", ""))
    except (KeyError, OSError):
        return ""


def _key(obj: Any) -> tuple:
    """The same object reached through two links has one key."""
    try:
        info = h5py.h5o.get_info(obj.id)
        return (info.fileno, info.addr)
    except Exception:  # noqa: BLE001 - an HDF5 build without object addresses: the path identifies it
        return (obj.file.filename, obj.name)


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.casefold() for part in _NATURAL.split(text)]


def _walk(group: h5py.Group, prefix: str, seen: set, found: _Found, depth: int = 0) -> None:
    """Every dataset reachable from ``group``: links followed, each group once."""
    for name in group:
        path = f"{prefix}/{name}"
        try:
            obj = group[name]
        except (KeyError, OSError):  # a link whose target (file) is not there
            found.dangling.append((path, group.get(name, getlink=True)))
            continue
        if isinstance(obj, h5py.Dataset):
            found.datasets.append((path, obj, group))
        elif isinstance(obj, h5py.Group) and name.lower() != "detectorspecific" and depth < _MAX_DEPTH:
            key = _key(obj)
            if key in seen:
                continue
            seen.add(key)
            if _nx_class(obj) == "NXdetector" or path.lower().endswith("/instrument/detector"):
                found.detectors.append(path)
            _walk(obj, path, seen, found, depth + 1)


def _image_shaped(dataset: h5py.Dataset) -> bool:
    if dataset.dtype.kind not in "iuf" or not 2 <= dataset.ndim <= 4:
        return False
    rows, columns = dataset.shape[-2:]
    if min(rows, columns) < _MIN_SIDE:
        return False  # (N, 1, W): a line detector or a table of counters
    # (H, W, 3|4) is a colour picture, not H frames of W × 3 pixels.
    return not (dataset.ndim >= 3 and columns in (3, 4) and rows >= _MIN_UNNAMED_SIDE)


def _unnamed_image(dataset: h5py.Dataset) -> bool:
    """A dataset nothing points at looks like an image: both sides wide enough, and not a long table."""
    rows, columns = (int(size) for size in dataset.shape[-2:])
    return min(rows, columns) >= _MIN_UNNAMED_SIDE and max(rows, columns) <= _MAX_UNNAMED_ASPECT * min(rows, columns)


def _correction_words(words: list[str]) -> bool:
    return any(word.startswith(_AUXILIARY) or word.endswith(("mask", "masks")) for word in words)


def _auxiliary(path: str) -> bool:
    """Named like a mask, flat field, dark, variance or error dataset, not like an image.

    A name that says what was done (``dark_subtracted``, ``flatfield_corrected``, ``sigma_clipped``,
    ``masked_image``) is the image it was done to.
    """
    parts = path.lower().replace("flat_field", "flatfield").split("/")
    name = [word for word in re.split(r"[^a-z]+", parts[-1]) if word]
    if _correction_words(name) and not any(word in _APPLIED for word in name):
        return True
    # The group it is in ("dark/data"), unless that is the top-level group, which is named after the scan.
    group = [word for word in re.split(r"[^a-z]+", parts[-2]) if word] if len(parts) > 3 else []
    return _correction_words(group[:1]) and not any(word in _APPLIED for word in group)


def _model_file(handle: h5py.File) -> bool:
    """Weights of a neural network: their kernels are image-shaped numbers, the file says what it is."""
    try:
        return any(name in handle.attrs for name in _MODEL_ATTRIBUTES) or any(name in handle for name in _MODEL_GROUPS)
    except (KeyError, OSError):
        return False


def _same_size_runs(members: dict) -> list[dict]:
    """Numbered datasets in number order, cut where the frame size changes: each run is one series."""
    runs: list[dict] = []
    shape = None
    for number in sorted(members):
        size = tuple(int(side) for side in members[number][1].shape[-2:])
        if size != shape:
            runs.append({})
            shape = size
        runs[-1][number] = members[number]
    return runs or [{}]


def _asked_dataset(handle: h5py.File, wanted: Optional[str]) -> Optional[h5py.Dataset]:
    """The dataset the caller names, wherever it is; ``None`` when the file has no such name.

    ``ValueError`` naming it when it is there but holds no frames (a curve, a text, a group): another
    dataset is never read in its place.
    """
    if not wanted:
        return None
    try:
        node = handle.get(wanted)
    except (KeyError, OSError, ValueError, TypeError):  # a link whose target is not there
        node = None
    if node is None:
        return None
    if not isinstance(node, h5py.Dataset):
        raise ValueError(DATASET_IS_GROUP.format(dataset=wanted))
    if node.dtype.kind not in "iuf" or not 2 <= node.ndim <= 4:
        raise ValueError(DATASET_NOT_FRAMES.format(
            dataset=wanted, shape=tuple(int(size) for size in node.shape), dtype=node.dtype))
    return node


def _default_key(handle: h5py.File) -> Optional[tuple]:
    """The dataset NeXus says to plot: root ``@default`` → NXentry ``@default`` → NXdata ``@signal``."""
    try:
        entry = handle[_text(handle.attrs["default"])]
        data = entry[_text(entry.attrs["default"])]
        if isinstance(data, h5py.Group):
            data = data[_text(data.attrs["signal"])]
        return _key(data) if isinstance(data, h5py.Dataset) else None
    except (KeyError, OSError, ValueError, TypeError, AttributeError):
        return None


def _entry_of(handle: h5py.File, path: str) -> str:
    """The nearest NXentry above ``path``; without one, the top-level group it is in."""
    parts = path.strip("/").split("/")[:-1]
    for depth in range(len(parts), 0, -1):
        candidate = "/" + "/".join(parts[:depth])
        if _nx_class(handle[candidate]) == "NXentry":
            return candidate
    return "/" + parts[0] if parts else ""


def _detector_group(handle: h5py.File, aliases: list[str], detectors: list[str], entry: str) -> str:
    """Where the mask, the module position and the header values of the dataset are."""
    parents = [alias.rsplit("/", 1)[0] for alias in aliases]
    for parent in parents:
        if parent in detectors:
            return parent
    # The data sits in an NXdata group (Dectris, NXmx): the detector of the same entry describes it.
    in_entry = [group for group in detectors if group.startswith(entry + "/")]
    if len(in_entry) == 1:
        return in_entry[0]
    return parents[0]


def _owner(dataset: h5py.Dataset) -> str:
    return os.path.abspath(dataset.file.filename)


def _virtual_files(dataset: h5py.Dataset) -> tuple[list[str], list[str], int]:
    """(data files present, data files missing, frames before the first missing one) of a virtual dataset."""
    folder = os.path.dirname(_owner(dataset))
    total = 1
    for size in dataset.shape[:-2]:
        total *= int(size)
    present: list[str] = []
    missing: list[str] = []
    available = total
    for source in dataset.virtual_sources():
        if source.file_name in ("", "."):
            continue
        target = os.path.normpath(os.path.join(folder, source.file_name))
        if os.path.isfile(target):
            if target not in present:
                present.append(target)
            continue
        if target not in missing:
            missing.append(target)
        start = source.vspace.get_select_bounds()[0]
        available = min(available, int(start[0]) if dataset.ndim == 3 else 0)
    return present, missing, available


def _order(stack: ImageStack) -> tuple:
    """Best first within one rank: a detector-like name, an image-like shape, then the natural path order."""
    lowered = stack.path.lower()
    name = lowered.rsplit("/", 1)[-1]
    score = 4 * (stack.path == DEFAULT_DATASET_PATH) + 2 * name.startswith(("data", "image", "frame"))
    score += any(word in lowered for word in _DETECTOR_WORDS)
    rows, columns = stack.frame_shape
    return (stack.rank, -score, max(rows, columns) > 8 * min(rows, columns), natural_key(stack.path))


def image_stacks(handle: h5py.File, wanted: Optional[str] = None) -> tuple[list[ImageStack], list[str], str]:
    """``(image datasets best first, data files that links point at but are not there, why there is none)``.

    ``wanted`` is the dataset the caller asks for: it comes first whatever its name, size and place,
    also under ``detectorSpecific`` (``ValueError`` when it is there but cannot hold frames).  The
    first of a family of numbered datasets stands for the family, any other one is read on its own.
    The third value is "" unless the file says it is something else than detector data.
    """
    found = _Found([], [], [])
    _walk(handle, "", set(), found)
    default = _default_key(handle)
    node = _asked_dataset(handle, wanted)
    asked = _key(node) if node is not None else None
    wanted_path = "/" + str(wanted).strip("/") if wanted else ""
    model = variables = _model_file(handle)
    root = os.path.abspath(handle.filename)
    by_object: dict = {}
    families: dict = {}
    asked_at: list[str] = []
    for path, dataset, parent in found.datasets:
        if dataset.dtype.kind not in "iuf" or not 2 <= dataset.ndim <= 4:
            continue
        key = _key(dataset)
        if key == asked:
            asked_at.append(path)
        if model or _MODEL_VARIABLE.search(path):  # a kernel is not an image, whatever its shape
            variables = True
            continue
        if not _image_shaped(dataset):
            continue
        auxiliary = key != default and _auxiliary(path)
        group_path, name = path.rsplit("/", 1)
        chunk = CHUNK_NAME.match(name)
        if chunk and dataset.ndim in (2, 3) and not auxiliary:
            family = families.setdefault((group_path, chunk.group("prefix")), {})
            family[int(chunk.group("number"))] = (path, dataset, key)
            continue
        if key == asked:
            continue  # ranked below, as the dataset asked for
        if key == default:
            rank, how = RANK_DEFAULT, "the file's default plot (@default → NXentry → NXdata @signal)"
        elif auxiliary:
            if not _unnamed_image(dataset):
                continue
            rank, how = RANK_AUXILIARY, "an image-shaped dataset named like a mask or a correction"
        elif name == "data" and group_path in found.detectors:
            rank, how = RANK_DETECTOR, f"'data' of the detector group {group_path}"
        elif dataset.is_virtual:
            rank, how = RANK_CHUNKS, "the virtual dataset of the file"
        elif _text(parent.attrs.get("signal", "")) == name or _marks_signal(dataset.attrs.get("signal")):
            rank, how = RANK_SIGNAL, f"the signal of the NXdata group {group_path}"
        elif _unnamed_image(dataset):
            rank, how = RANK_OTHER, "an image-shaped dataset"
        else:
            continue
        by_object.setdefault(key, []).append((rank, len(path), path, how, dataset, path))
    series = [(name, run, index == 0) for name, members in families.items()
              for index, run in enumerate(_same_size_runs(members))]
    asked_series = None
    if node is not None:
        if CHUNK_NAME.match(wanted_path.rsplit("/", 1)[-1]):  # the name a series goes by is its first dataset
            asked_series = next((index for index, (_name, run, _first) in enumerate(series)
                                 if run[min(run)][2] == asked), None)
        if asked_series is None:
            by_object[asked] = [(RANK_ASKED, index, str(wanted), "the dataset asked for", node, alias)
                                for index, alias in enumerate(dict.fromkeys([wanted_path] + asked_at))]
    stacks: list[ImageStack] = []
    for reached in by_object.values():  # the same dataset reached through a link counts once
        rank, _length, path, how, dataset, _walked = min(reached, key=lambda item: item[:3])
        aliases = [path] + [item[5] for item in reached if item[5] != path]
        entry = _entry_of(handle, path)
        lead = tuple(int(size) for size in dataset.shape[:-2])
        frames = 1
        for size in lead:
            frames *= size
        files, missing = ([] if _owner(dataset) == root else [_owner(dataset)]), []
        if dataset.is_virtual:
            virtual, missing, frames = _virtual_files(dataset)
            files += [name for name in virtual if name != root and name not in files]
            if virtual or missing:
                how += f", a virtual dataset over {len(virtual) + len(missing)} data files"
        stacks.append(ImageStack(
            path, lead, tuple(int(size) for size in dataset.shape[-2:]), str(dataset.dtype), rank, how,
            _detector_group(handle, aliases, found.detectors, entry), entry,
            files=tuple(files), missing=tuple(missing), frames=frames))
    dangling = {path: link for path, link in found.dangling}
    for path in dangling:  # a master none of whose data files has arrived is a master all the same
        group_path, name = path.rsplit("/", 1)
        chunk = CHUNK_NAME.match(name)
        if chunk and not model and (group_path, chunk.group("prefix")) not in families:
            families[(group_path, chunk.group("prefix"))] = {}
            series.append(((group_path, chunk.group("prefix")), {}, True))
    for index, (name, run, first) in enumerate(series):
        rank = (RANK_ASKED if index == asked_series
                else RANK_DEFAULT if default in [key for _path, _dataset, key in run.values()] else RANK_CHUNKS)
        # A data file that has not arrived stops the series the file begins with.
        stacks.append(_family(handle, name, run, dangling if first else {}, found.detectors, root, rank))
    lost = sorted({os.path.normpath(os.path.join(os.path.dirname(root), link.filename))
                   for link in dangling.values() if isinstance(link, h5py.ExternalLink)})
    return sorted(stacks, key=_order), lost, (MODEL_FILE if variables else "")


def _family(handle: h5py.File, name: tuple[str, str], members: dict, dangling: dict, detectors: list[str],
            root: str, rank: int) -> ImageStack:
    """``data_000001``, ``data_000002`` … of one group as one series; it stops before a missing chunk."""
    group_path, prefix = name
    gone = {}
    for path, link in dangling.items():
        chunk = CHUNK_NAME.match(path.rsplit("/", 1)[1])
        if chunk and chunk.group("prefix") == prefix and path.rsplit("/", 1)[0] == group_path:
            gone[int(chunk.group("number"))] = link
    if members:
        first_path, first, _key_of_first = members[min(members)]
        frame_shape, dtype = tuple(int(size) for size in first.shape[-2:]), str(first.dtype)
    else:
        first_path, frame_shape, dtype = f"{group_path}/{prefix}_{min(gone):06d}", (0, 0), ""
    chunks: list[tuple[str, int]] = []
    files: list[str] = []
    behind: list[str] = []
    cut = False  # the series stops at a data file that is not there: what follows cannot be counted on
    for number in sorted(set(members) | set(gone)):
        if number in gone:
            cut = True
            continue
        path, dataset, _key_of_chunk = members[number]
        owner = _owner(dataset)
        if cut:  # linked to all the same: a file of this master, never a source of its own
            if owner != root and owner not in files and owner not in behind:
                behind.append(owner)
            continue
        if tuple(int(size) for size in dataset.shape[-2:]) != frame_shape:
            break
        chunks.append((path, int(dataset.shape[0]) if dataset.ndim == 3 else 1))
        if owner != root and owner not in files:
            files.append(owner)
    missing = [os.path.normpath(os.path.join(os.path.dirname(root), link.filename))
               for _number, link in sorted(gone.items()) if isinstance(link, h5py.ExternalLink)]
    entry = _entry_of(handle, first_path)
    frames = sum(count for _path, count in chunks)
    what = "data chunks" if prefix == "data" else "numbered datasets"
    return ImageStack(
        first_path, (frames,), frame_shape, dtype, rank,
        f"{len(members) + len(gone)} {what} ({first_path.rsplit('/', 1)[1]} …) in {group_path}",
        _detector_group(handle, [first_path], detectors, entry), entry,
        chunks=tuple(chunks), files=tuple(files), missing=tuple(missing), frames=frames, behind=tuple(behind))


def stored_frames_last(dataset: Any) -> bool:
    """A 3-D dataset that looks like ``(rows, columns, frames)``: the frames along the LAST axis.

    GIMaP reads the first axis as the frames.  What the file says counts first: the axis a dataset can grow
    along, and the way it is cut into chunks, are the frame axis.  Without either, the shape decides: two
    first sides like those of a detector and a last one that is small beside them.
    """
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
        return False
    first, second, last = (int(size) for size in dataset.shape)
    grows = dataset.maxshape
    if grows[0] is None or grows[2] is None:
        return grows[0] is not None
    chunks = dataset.chunks
    if chunks is not None and (chunks[0] == 1) != (chunks[2] == 1):
        return chunks[2] == 1
    narrow, wide = min(first, second), max(first, second)
    return last >= 2 and narrow >= _DETECTOR_SIDE and 4 * last <= narrow and wide <= 4 * narrow


def tied(stacks: list[ImageStack]) -> list[ImageStack]:
    """The datasets that share the best rank (more than one: the file does not say which is the image)."""
    return [stack for stack in stacks if stack.rank == stacks[0].rank] if stacks else []


def _dataset_candidates(handle: h5py.File) -> list[tuple[int, str]]:
    """``(score, path)`` of every dataset a caller may name, best first (kept for the dataset choosers).

    A series of numbered datasets is one candidate under the path of its first dataset; its other
    datasets follow, each readable on its own (score −5), then the datasets named like a mask or a
    correction (score −10).
    """
    stacks, _lost, _why = image_stacks(handle)
    images = [(10 * (RANK_OTHER - stack.rank), stack.path) for stack in stacks if stack.rank < RANK_AUXILIARY]
    parts = [(-5, path) for stack in stacks for path, _frames in stack.chunks[1:]]
    return images + parts + [(-10, stack.path) for stack in stacks if stack.rank == RANK_AUXILIARY]


def select_nxs_dataset(handle: h5py.File, dataset_path: Optional[str] = None) -> str:
    """The image dataset of the file; ``AmbiguousDatasetError`` when several are equally plausible."""
    if dataset_path and dataset_path in handle:
        return dataset_path
    stacks, _lost, _why = image_stacks(handle)
    if not stacks:
        raise ValueError("No usable detector image dataset was found in this NXS file.")
    best = tied(stacks)
    if len(best) > 1:
        raise AmbiguousDatasetError([stack.path for stack in best])
    return stacks[0].path
