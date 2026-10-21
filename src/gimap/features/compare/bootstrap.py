"""Compose Compare: the service with its file adapters."""

from __future__ import annotations

from .application import CompareService
from .infrastructure import AnalyzeFolders, CsvTableWriter, TextCurveReader


def create_compare_service() -> CompareService:
    reader = TextCurveReader()
    return CompareService(reader, CsvTableWriter(), AnalyzeFolders(reader))


__all__ = ["create_compare_service"]
