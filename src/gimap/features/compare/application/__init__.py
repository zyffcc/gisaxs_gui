"""Compare: series from curve files or Analyze's Series map, the comparison, tables and records."""

from ..domain import END_FRAMES, Comparison, SeriesData, SeriesResult
from .sources import CurveKind, FolderContent, FolderReader, SeveralKinds
from .use_cases import (
    CURVE_SUFFIXES,
    ERRORS,
    METHOD,
    PROGRESS_EVERY,
    TOO_FEW_READ,
    UNREAD_FILE,
    CompareService,
    CompareSettings,
    CurveReader,
    Progress,
    TableWriter,
    curve_files,
    natural_key,
)

__all__ = [
    "CURVE_SUFFIXES", "END_FRAMES", "ERRORS", "METHOD", "PROGRESS_EVERY", "TOO_FEW_READ", "UNREAD_FILE",
    "CompareService", "CompareSettings", "Comparison", "CurveKind", "CurveReader", "FolderContent", "FolderReader",
    "Progress", "SeriesData", "SeriesResult", "SeveralKinds", "TableWriter", "curve_files", "natural_key",
]
