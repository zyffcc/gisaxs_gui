"""Explicit table-column selections and their portable, scientific source records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal


TableLayout = Literal["single", "shared_q"]


@dataclass(frozen=True)
class ColumnMapping:
    """Zero-based columns of one curve; names/width guard saved selections against table changes."""

    q: int
    intensity: int
    sigma: int | None = None
    column_names: tuple[str, ...] = ()
    column_count: int | None = None

    def __post_init__(self) -> None:
        indexes = (self.q, self.intensity) + (() if self.sigma is None else (self.sigma,))
        if any(type(index) is not int or index < 0 for index in indexes):
            raise ValueError("Curve column indexes must be non-negative integers.")
        if len(set(indexes)) != len(indexes):
            raise ValueError("q, intensity and sigma must use different columns.")
        if self.column_count is not None:
            if type(self.column_count) is not int or self.column_count <= max(indexes):
                raise ValueError("A selected curve column is missing from the table.")
        if not isinstance(self.column_names, tuple) or any(not isinstance(name, str) for name in self.column_names):
            raise ValueError("Curve column names must be a tuple of strings.")
        if self.column_names and len(self.column_names) <= max(indexes):
            raise ValueError("A selected curve column has no column name.")
        if self.column_names and self.column_count is not None and len(self.column_names) != self.column_count:
            raise ValueError("Curve column names do not match the table width.")

    def to_dict(self) -> dict:
        return {"q": self.q, "intensity": self.intensity, "sigma": self.sigma,
                "column_names": list(self.column_names), "column_count": self.column_count}

    @classmethod
    def from_dict(cls, value: dict) -> "ColumnMapping":
        if not isinstance(value, dict) or set(value) - {"q", "intensity", "sigma", "column_names", "column_count"}:
            raise ValueError("Invalid curve column mapping.")
        names = value.get("column_names", ())
        if not isinstance(names, (list, tuple)):
            raise ValueError("Invalid curve column names.")
        try:
            return cls(value["q"], value["intensity"], value.get("sigma"), tuple(names), value.get("column_count"))
        except KeyError as exc:
            raise ValueError("Invalid curve column mapping.") from exc

    @property
    def intensity_name(self) -> str:
        return self.column_names[self.intensity] if self.column_names else f"Column {self.intensity + 1}"


@dataclass(frozen=True)
class CurveTablePreview:
    """Only the header and first ten data rows are shown; no scientific array is modified."""

    path: Path
    column_names: tuple[str, ...]
    sample_rows: tuple[tuple[str, ...], ...]
    column_count: int
    suggested_layout: Literal["single", "shared_q", "ambiguous"]
    analyze_export: bool = False
    unit: str | None = None
    has_header: bool = False
    uncertainty_columns: tuple[int, ...] = ()
    intensity_columns: tuple[int, ...] = ()

    @property
    def allow_shared_q(self) -> bool:
        return self.column_count >= 3 and not self.analyze_export and not self.uncertainty_columns


def sources_from_table(preview: CurveTablePreview, layout: TableLayout):
    """Confirm legacy q/I/sigma, or independently select every I column sharing the first q column.

    An ambiguous table is never interpreted as multiple curves without this explicit layout choice.
    Analyze's pixel-count/error columns are protected even when the user requests a shared-q layout.
    """
    from .models import InSituSourceFrame

    if layout not in ("single", "shared_q"):
        raise ValueError("Unknown curve table layout.")
    if preview.column_count < 2:
        raise ValueError("A curve table needs at least two columns.")
    if layout == "shared_q" and not preview.allow_shared_q:
        if preview.analyze_export:
            raise ValueError("GIMaP Analyze exports contain one curve; sigma and pixels are not intensity columns.")
        if preview.uncertainty_columns:
            raise ValueError("Named uncertainty columns cannot be imported as separate intensities.")
        raise ValueError("Shared-q tables need at least three columns.")
    names = preview.column_names if preview.has_header else ()
    columns = range(1, preview.column_count) if layout == "shared_q" else (1,)
    sigma = 2 if layout == "single" and preview.column_count >= 3 else None
    sources = tuple(InSituSourceFrame(preview.path, columns=ColumnMapping(
        0, index, sigma, names, preview.column_count)) for index in columns)
    for source in sources:
        validate_columns(preview, source.columns)
    return sources


def validate_columns(preview: CurveTablePreview, columns: ColumnMapping) -> None:
    """Check a persisted mapping against a fresh table inspection; never silently remap columns."""
    if columns.column_count is not None and columns.column_count != preview.column_count:
        raise ValueError("The table column count changed; import it again to confirm the selected columns.")
    if columns.column_names and (not preview.has_header or columns.column_names != preview.column_names):
        raise ValueError("The table column names changed; import it again to confirm the selected columns.")
    if columns.column_count is not None and bool(columns.column_names) != preview.has_header:
        raise ValueError("The table column header changed; import it again to confirm the selected columns.")
    selected = (columns.q, columns.intensity) + (() if columns.sigma is None else (columns.sigma,))
    if max(selected) >= preview.column_count:
        raise ValueError("A selected curve column is missing from the table.")
    if preview.analyze_export and selected != (0, 1, 2):
        raise ValueError("GIMaP Analyze exports contain one curve; sigma and pixels are not intensity columns.")
    if columns.intensity in preview.uncertainty_columns:
        raise ValueError("Named uncertainty columns cannot be imported as separate intensities.")
    if columns.sigma is not None:
        if preview.uncertainty_columns and columns.sigma not in preview.uncertainty_columns:
            raise ValueError("Single-curve import requires q, I and optional sigma in that order; "
                             "the named uncertainty column is elsewhere.")
        if columns.sigma in preview.intensity_columns:
            raise ValueError("An intensity column cannot be used as sigma. Choose Shared q or correct the column names.")


def source_record(token) -> dict:
    """The physical file and explicit column selection, suitable for a JSON provenance record."""
    from .models import InSituSourceFrame

    source = InSituSourceFrame.from_token(str(token))
    return {"path": str(source.path), "frame_index": source.frame_index, "source_kind": source.source_kind,
            "columns": None if source.columns is None else source.columns.to_dict()}


def source_stem(token) -> str:
    """A portable output basename; display labels and encoded source tokens are never paths."""
    from .models import InSituSourceFrame

    source = InSituSourceFrame.from_token(str(token))
    if source.columns is None:
        return source.path.stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source.path.stem).rstrip(" .") or "curve"
    return f"{stem}_I{source.columns.intensity + 1}"
