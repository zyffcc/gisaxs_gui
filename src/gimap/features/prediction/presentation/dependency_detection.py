"""Lightweight optional-dependency discovery for presentation choices."""

from __future__ import annotations

import importlib.util


def dependency_available(module_name: str) -> bool:
    """Return whether an optional UI integration can be discovered safely."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False
