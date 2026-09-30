"""Analyze application ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from src.gimap.shared.detector_io.models import DetectorImage
from src.gimap.shared.geometry import InstrumentProfile

from ..domain import Curve


class FrameSource(Protocol):
    def load(self, path: str | Path, frame_index: int = 0) -> DetectorImage: ...

    def frame_count(self, path: str | Path) -> int: ...

    def settled_size(self, path: str | Path) -> int:
        """Bytes on disk for the frame (all files it consists of)."""
        ...

    def expand(self, paths: Sequence[str | Path]) -> list[Path]:
        """Files and folders → the supported detector frames they contain, sorted."""
        ...


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


__all__ = ["CurveWriter", "FigureWriter", "FrameSource", "InstrumentProfileStore"]
