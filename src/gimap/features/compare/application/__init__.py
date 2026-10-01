"""Compare: series from curve files or Analyze's Series map, the comparison, tables and records."""

from ..domain import END_FRAMES, Comparison, SeriesData, SeriesResult
from .use_cases import (
    CURVE_SUFFIXES,
    METHOD,
    CompareService,
    CompareSettings,
    CurveReader,
    TableWriter,
    curve_files,
    natural_key,
)

__all__ = [
    "CURVE_SUFFIXES", "END_FRAMES", "METHOD", "CompareService", "CompareSettings", "Comparison", "CurveReader",
    "SeriesData", "SeriesResult", "TableWriter", "curve_files", "natural_key",
]
