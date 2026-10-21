"""Analyze application ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from src.gimap.shared.detector_io.models import DetectorImage
from src.gimap.shared.detector_io.nexus_layout import P03_LAMBDA, NexusLayout
from src.gimap.shared.detector_io.nexus_modules import (
    MODULE_LEFT_UNFINISHED,
    MODULE_MISSING,
    MODULE_UNREADABLE,
    MODULES_LEFT_UNFINISHED,
    MODULES_MISSING,
    NOT_A_MODULE,
    SAME_POSITION,
    module_like,
    scan_stem,
)
from src.gimap.shared.geometry import InstrumentProfile

from ..domain import Curve

FRAME_OF = "frame {i} of {n}"
"""A frame of a source as people count it, from 1 (file names keep the 0-based ``_frame0003``)."""
MODULE_SET = "module_set"
_PARTS_MISSING = (MODULE_MISSING, MODULES_MISSING, MODULE_UNREADABLE, MODULE_LEFT_UNFINISHED, MODULES_LEFT_UNFINISHED)
_READ_ALONE = (SAME_POSITION, NOT_A_MODULE)


class FrameSource(Protocol):
    def load(self, path: str | Path, frame_index: int = 0) -> DetectorImage: ...

    def frame_count(self, path: str | Path) -> int:
        """Frames the source holds now (it may grow); for a set of module files, those every module has.
        ``OSError`` or ``ValueError`` when it cannot be read and never could (never another kind of error)."""
        ...

    def describe(self, path: str | Path) -> Optional[NexusLayout]:
        """What a NeXus / HDF5 source consists of: files, frames, warnings (``None``: another format, or a
        file never read so far). Cheap to ask again; never raises. Sources made before this method lack it:
        call it through ``getattr(frames, "describe", None)``."""
        ...

    def parts(self, path: str | Path) -> tuple[Path, ...]:
        """Every file the source of ``path`` stands for, the one that names it first (module files, a master
        and its data files): none of them is a row of its own. Never raises. Sources made before this method
        lack it: call it through ``getattr(frames, "parts", None)``."""
        ...

    def settled_size(self, path: str | Path) -> int:
        """Bytes on disk for the frame (all files it consists of); ``OSError`` while it cannot be measured."""
        ...

    def expand(self, paths: Sequence[str | Path]) -> list[Path]:
        """Files and folders → the detector sources they contain, sorted: one path per source (the first
        module file of a set, the master of a master with data files)."""
        ...


def frame_label(frame_index: int, frame_count: int) -> str:
    """``frame 4 of 12`` for the 0-based ``frame_index`` (English: headers, records, picture titles)."""
    return FRAME_OF.format(i=int(frame_index) + 1, n=int(frame_count))


def _kind(source: Any) -> Optional[str]:
    if isinstance(source, Mapping):  # a frame's metadata
        return source.get("layout")
    return getattr(source, "kind", None)


def source_stem(path: str | Path, source: Any = None) -> str:
    """The name of a source without its suffix, for file names and labels. ``source`` says what the path
    stands for: its ``NexusLayout`` (``FrameSource.describe``) or the metadata of a frame read from it. A set
    of module files is named by its scan, without the module number (``scan_00007_m01.nxs`` → ``scan_00007``);
    everything else, and a path nothing is known of, by the stem of its file."""
    return scan_stem(source, path) or Path(path).stem


def source_name(path: str | Path, source: Any = None) -> str:
    """What a source is called on screen: the file name, or for a set of module files the scan they belong
    to (``source_stem``)."""
    return source_stem(path, source) if _kind(source) == MODULE_SET else Path(path).name


def parts_missing(layout: Optional[NexusLayout]) -> bool:
    """A module file of the set is missing, cannot be read yet, or was left unfinished by a writer that ended:
    every frame lacks that part of the
    detector, so nothing of the source is finished (``NexusLayout.complete`` is also false while modules
    only hold different numbers of frames; the frames every module has are whole then)."""
    if layout is None:
        return False
    return bool(layout.pending) or any(template in _PARTS_MISSING for template, _values in layout.warnings)


def module_files(layout: Optional[NexusLayout]) -> int:
    """How many module files a source that is, or may become, a set of them holds so far (0: it is no such
    source). A set counts the files it has, also those that cannot be read yet. One file of a Lambda detector
    that is named like a module counts as the first of its set: while a set arrives file by file, nothing in
    its first module says that more follow (no gap in the numbering)."""
    if layout is None:
        return 0
    if layout.kind == MODULE_SET:
        return len(layout.members) + len(layout.pending)
    alone = layout.kind in ("stack", "single_frame") and layout.orientation == P03_LAMBDA and module_like(layout.path)
    return int(alone and not any(template in _READ_ALONE for template, _values in layout.warnings))


class InstrumentProfileStore(Protocol):
    def load_all(self) -> list[InstrumentProfile]: ...

    def find(self, name: str) -> Optional[InstrumentProfile]: ...

    def save(self, profile: InstrumentProfile) -> None: ...

    def match(
        self, *, detector_name: Optional[str], shape: Optional[tuple[int, int]]
    ) -> Optional[InstrumentProfile]: ...


class CurveWriter(Protocol):
    def write_xy(self, curve: Curve, path: Path, metadata: Mapping[str, Any]) -> Path:
        """Whitespace columns ``x I sigma pixels`` for fitting programs.

        ``metadata["observation"]`` (when present) is written as a ``# observation:``
        JSON line: how the points were measured (see ``fit_input_observation``).
        """
        ...

    def write(
        self,
        curves: Sequence[Curve],
        destination: Path,
        stem: str,
        metadata: Mapping[str, Any],
        text_format: str = "csv",
    ) -> list[Path]:
        """One ``<stem>_<curve>.<text_format>`` per curve (csv: comma, txt: tab, dat: space) and the JSON record."""
        ...

    def write_table(self, path: Path, comments: Sequence[str], header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
        """A table: ``#`` comment lines, a header row, then the rows (NaN as blank cells); the separator
        follows the suffix (``.csv`` comma, ``.txt`` tab, ``.dat`` space)."""
        ...

    def write_map(
        self, image: Any, x_axis: Any, y_axis: Any, path: Path, labels: tuple[str, str], metadata: Mapping[str, Any]
    ) -> Path:
        """A 2D intensity table: first row the x axis, first column the y axis (``#`` provenance lines)."""
        ...

    def write_record(self, path: Path, record: Mapping[str, Any]) -> Path:
        """A JSON record (settings, frames, files written)."""
        ...


class FitPreviewCache(Protocol):
    """A bounded store of actual fit curves; overflow belongs in a private temporary cache."""

    closed: bool

    def remember(self, index: int, curves: Sequence[Any]) -> Sequence[int]:
        """Keep these curves and return the indices whose arrays were moved out of memory."""
        ...

    def read(self, index: int) -> Sequence[Any]: ...

    def close(self) -> None: ...


class FigureWriter(Protocol):
    """Publication figures (PNG/TIFF raster, SVG/PDF vector by suffix)."""

    def write_image(self, image: Any, path: Path, **display: Any) -> Path: ...

    def write_frame(self, data: Any, valid: Any, path: Path, **display: Any) -> Path:
        """A frame with automatic display levels."""
        ...

    def write_curves(
        self, curves: Sequence[tuple[str, Any, Any]], path: Path, **labels: Any
    ) -> Path: ...

    def write_panels(self, panels: Sequence[tuple[str, Sequence[tuple[str, Any, Any]]]], path: Path, **labels: Any) -> Path:
        """Stacked panels sharing x (a fit trend against frame)."""
        ...


__all__ = ["CurveWriter", "FRAME_OF", "FigureWriter", "FitPreviewCache", "FrameSource", "InstrumentProfileStore", "frame_label",
           "module_files", "parts_missing", "source_name", "source_stem"]
