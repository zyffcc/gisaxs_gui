"""Compose Sputter: the service with the toolkit process and the result reader."""

from __future__ import annotations

from typing import Callable

from .application import SputterService
from .infrastructure import StructureFolderReader, SubprocessToolkit


def create_sputter_service(settings: Callable[[], tuple[str, str]]) -> SputterService:
    """``settings()`` gives (toolkit Python executable, toolkit folder), read when a run starts."""
    return SputterService(SubprocessToolkit(settings), StructureFolderReader())


__all__ = ["create_sputter_service"]
