"""Local NeXus / HDF5 and CBF series repository backed by the shared detector reader.

A NeXus / HDF5 series (kind ``nxs``) is, in this order:

* the frames of the chosen source, when it holds more than one: a stack in one file, a set of
  detector-module files (stitched per frame), a master with its data files;
* for a file holding ONE frame: the files of its folder that are named like it up to their last number
  and hold one frame of the same size each — a series stored as one frame per file (one file, or one
  set of module files, per angle);
* for a folder: every NeXus / HDF5 source in it, file after file.

What a source consists of is the shared reader's decision (``cached_layout``): a file name alone never
makes module files of unrelated files. A CBF series (kind ``cbf``) is the files of a folder that match
the pattern, one frame each.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.gimap.shared.detector_io import (
    HDF5_SUFFIXES,
    NexusLayout,
    cached_layout,
    group_sources,
    open_nexus,
    scan_stem,
)

from ...application import XrrDetectorFrame, XrrFrameRef, XrrSeriesLayout, XrrSeriesSpec
from ...application.models import (
    FILES_LEFT_OUT,
    FRAMES_IN_ALL,
    MORE_REMARKS,
    ONE_FRAME_EACH,
    SETS_ONE_FRAME_EACH,
)

NO_IMAGE = "{name} holds no detector image."
_LAST_NUMBER = re.compile(r"^(?P<head>.*?)(?P<number>\d+)(?P<tail>\D*)$")
_MODULE_SUFFIX = r"(?:_?m\d{1,3})?"


def _natural_key(path: Path) -> tuple:
    return tuple(
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", path.name)
    )


def _hdf5_files(folder: Path) -> list[Path]:
    try:
        return sorted(
            (path for path in folder.iterdir() if path.suffix.casefold() in HDF5_SUFFIXES and path.is_file()),
            key=_natural_key,
        )
    except OSError:
        return []


def _layout_of(path: Path) -> NexusLayout | None:
    try:
        return cached_layout(path)
    except Exception:  # noqa: BLE001 - a file that cannot be read now is no frame of the series
        return None


NAMED_LEFT_OUT = 3
"""Files left out of a series that are named in the remark; the others are counted."""


def series_layout(layouts: list[NexusLayout], left_out: tuple[str, ...] = ()) -> XrrSeriesLayout:
    """What a series of NeXus / HDF5 sources consists of, with what the reader had to say about them
    (one remark of each kind, and how many more files have one). ``left_out``: names of files numbered
    like the series that are no frame of it."""
    frames = sum(int(layout.frames) for layout in layouts)
    if len(layouts) == 1:
        summary = layouts[0].summary()
    elif all(layout.frames == 1 for layout in layouts):
        modules = {len(layout.members) for layout in layouts}
        if all(layout.kind == "module_set" for layout in layouts) and len(modules) == 1:
            summary = (SETS_ONE_FRAME_EACH, {"sets": len(layouts), "modules": modules.pop()})
        else:
            summary = (ONE_FRAME_EACH, {"files": len(layouts)})
    else:
        summary = (FRAMES_IN_ALL, {"files": len(layouts), "frames": frames})
    kinds: dict[str, tuple[str, dict]] = {}
    remarked = 0
    for layout in layouts:
        remarked += bool(layout.warnings)
        for template, values in layout.warnings:
            kinds.setdefault(template, (template, values))
    warnings = list(kinds.values())
    if remarked > 1:
        warnings.append((MORE_REMARKS, {"count": remarked - 1}))
    if left_out:
        names = ", ".join(left_out[:NAMED_LEFT_OUT]) + (" …" if len(left_out) > NAMED_LEFT_OUT else "")
        warnings.append((FILES_LEFT_OUT, {"count": len(left_out), "names": names}))
    members = tuple(str(member) for layout in layouts for member in layout.members)
    return XrrSeriesLayout(len(layouts), frames, summary, tuple(warnings), members)


class LocalXrrSeriesRepository:
    """Discover lightweight references and load only the requested frame."""

    def discover(self, spec: XrrSeriesSpec) -> tuple[XrrFrameRef, ...]:
        return self._series(spec)[0]

    def describe(self, spec: XrrSeriesSpec) -> XrrSeriesLayout:
        """What the frames of ``discover`` are stored in (asking again opens no file that did not change)."""
        return self._series(spec)[1]

    def series(self, spec: XrrSeriesSpec) -> tuple[tuple[XrrFrameRef, ...], XrrSeriesLayout]:
        """The frames and what they are stored in, from one look at the files."""
        return self._series(spec)

    def _series(self, spec: XrrSeriesSpec) -> tuple[tuple[XrrFrameRef, ...], XrrSeriesLayout]:
        source = Path(spec.source_path).expanduser().resolve()
        kind = self._kind(source, spec.source_kind, spec.pattern)
        if kind == "nxs":
            return self._discover_nxs(source, spec)
        refs = self._discover_cbf(source, spec.pattern)
        layout = XrrSeriesLayout(
            len(refs), len(refs), (ONE_FRAME_EACH, {"files": len(refs)}), members=tuple(str(ref.path) for ref in refs)
        )
        return refs, layout

    def load_frame(self, frame: XrrFrameRef) -> XrrDetectorFrame:
        from src.gimap.shared.detector_io import load_detector_image

        loaded = load_detector_image(frame.path, frame_idx=frame.frame_index)
        return XrrDetectorFrame(
            data=np.asarray(loaded.data, dtype=np.float32),
            invalid_mask=(
                None if loaded.mask is None else np.asarray(loaded.mask, dtype=bool)
            ),
            metadata=dict(loaded.metadata),
        )

    @classmethod
    def _kind(cls, source: Path, requested: str, pattern: str = "*.cbf") -> str:
        if requested in {"nxs", "cbf"}:
            return requested
        suffix = source.suffix.casefold()
        if suffix in HDF5_SUFFIXES and not source.is_dir():
            return "nxs"
        if suffix == ".cbf":
            return "cbf"
        if source.is_dir():
            # A folder is a CBF series when it holds CBF frames; else its NeXus / HDF5 files are the series.
            return "cbf" if cls._cbf_paths(source, pattern) or not _hdf5_files(source) else "nxs"
        raise ValueError("Select an NXS module file or a folder containing CBF files.")

    def _discover_nxs(
        self, source: Path, spec: XrrSeriesSpec
    ) -> tuple[tuple[XrrFrameRef, ...], XrrSeriesLayout]:
        layouts, left_out = self._nxs_sources(source)
        refs: list[XrrFrameRef] = []
        for layout in layouts:
            count = int(layout.frames)
            angles = self._nxs_angles(layout, spec.angle_dataset_path, count)
            name = scan_stem(layout)
            for index in range(count):
                refs.append(
                    XrrFrameRef(
                        path=layout.path,
                        frame_index=index,
                        sequence_index=len(refs),
                        theta_deg=None if angles is None else float(angles[index]),
                        name=name,
                        frames_in_file=count,
                    )
                )
        return tuple(refs), series_layout(layouts, left_out)

    def _nxs_sources(self, source: Path) -> tuple[list[NexusLayout], tuple[str, ...]]:
        """The sources of the series, in order (see the module's docstring), and the names of files
        numbered like a one-frame-per-file series that are no frame of it."""
        if source.is_dir():
            found = (_layout_of(path) for path in group_sources(_hdf5_files(source)))
            layouts = [layout for layout in found if layout is not None and layout.kind != "no_image"]
            if not layouts:
                raise ValueError(f"No NeXus / HDF5 detector files were found in {source}.")
            return layouts, ()
        if not source.is_file():
            raise FileNotFoundError(f"NXS source was not found: {source}")
        layout = cached_layout(source)
        if layout.kind == "no_image":
            raise ValueError(NO_IMAGE.format(name=source.name))
        if layout.frames != 1:
            return [layout], ()
        series, left_out = self._numbered_like(layout)
        return (series, left_out) if series else ([layout], ())

    @staticmethod
    def _numbered_like(layout: NexusLayout) -> tuple[list[NexusLayout], tuple[str, ...]]:
        """The one-frame sources of the folder that are named like ``layout`` up to their last number
        and hold a frame of the same size: a series stored as one frame per file (empty when the name
        holds no number). With them, the names of the files numbered alike that do not fit."""
        numbered = _LAST_NUMBER.match(scan_stem(layout) or layout.path.stem)
        if numbered is None:
            return [], ()
        alike = re.escape(numbered["head"]) + r"\d+" + re.escape(numbered["tail"])
        suffix = layout.path.suffix.casefold()
        candidates = [
            path
            for path in _hdf5_files(layout.path.parent)
            if path.suffix.casefold() == suffix
            and re.fullmatch(alike + _MODULE_SUFFIX, path.stem, re.IGNORECASE)
        ]
        series, left_out = [], []
        for path in group_sources(candidates):
            other = _layout_of(path)
            fits = (
                other is not None and other.frames == 1 and other.kind == layout.kind
                and other.frame_shape == layout.frame_shape and len(other.members) == len(layout.members)
            )
            if fits and re.fullmatch(alike, scan_stem(other) or other.path.stem, re.IGNORECASE):
                series.append(other)
            elif not fits and re.fullmatch(alike + _MODULE_SUFFIX, path.stem, re.IGNORECASE):
                left_out.append(path)
        left_out.sort(key=_natural_key)
        return sorted(series, key=lambda found: _natural_key(found.path)), tuple(path.name for path in left_out)

    @staticmethod
    def _nxs_angles(layout: NexusLayout, dataset_path: str, count: int) -> np.ndarray | None:
        if not dataset_path:
            return None
        with open_nexus(layout.path) as handle:
            if dataset_path not in handle:
                raise ValueError(f"NXS angle dataset was not found: {dataset_path}")
            dataset = handle[dataset_path]
            values = np.asarray(dataset[()], dtype=float).reshape(-1)
            raw_units = dataset.attrs.get("units", "deg")
            if isinstance(raw_units, bytes):
                raw_units = raw_units.decode("utf-8", errors="replace")
            units = str(raw_units).casefold()
        if count < values.size <= max(layout.member_frames or (count,)):
            # Module files with different frame counts: the frames every module has keep their angles
            # (frame k has angle k); the layout's warning says which frames are left out.
            values = values[:count]
        if values.size != count:
            raise ValueError(
                f"NXS angle dataset contains {values.size} values but the series has {count} frames."
            )
        if "mrad" in units:
            values = np.degrees(values / 1000.0)
        elif units in {"rad", "radian", "radians"}:
            values = np.degrees(values)
        return values

    @staticmethod
    def _cbf_paths(folder: Path, pattern: str) -> list[Path]:
        return sorted(
            (
                path
                for path in folder.glob(pattern.strip() or "*.cbf")
                if path.is_file() and path.suffix.casefold() == ".cbf"
            ),
            key=_natural_key,
        )

    @classmethod
    def _discover_cbf(cls, source: Path, pattern: str) -> tuple[XrrFrameRef, ...]:
        folder = source if source.is_dir() else source.parent
        if not folder.is_dir():
            raise FileNotFoundError(f"CBF series folder was not found: {folder}")
        selected_pattern = pattern.strip() or "*.cbf"
        paths = cls._cbf_paths(folder, selected_pattern)
        if not paths:
            raise ValueError(f"No CBF files match {selected_pattern!r} in {folder}.")
        return tuple(
            XrrFrameRef(path=path, frame_index=0, sequence_index=index)
            for index, path in enumerate(paths)
        )


__all__ = ["LocalXrrSeriesRepository"]
