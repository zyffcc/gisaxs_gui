"""Compose Compare: the service with its file adapters."""

from __future__ import annotations

from .application import CompareService
from .infrastructure import CsvTableWriter, TextCurveReader


def create_compare_service() -> CompareService:
    return CompareService(TextCurveReader(), CsvTableWriter())


__all__ = ["create_compare_service"]
