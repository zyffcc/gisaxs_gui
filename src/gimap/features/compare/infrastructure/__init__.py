"""Compare's adapters: curve files in (and what a folder of them holds), CSV tables and JSON records out."""

from .files import CsvTableWriter, TextCurveReader
from .folders import AnalyzeFolders

__all__ = ["AnalyzeFolders", "CsvTableWriter", "TextCurveReader"]
