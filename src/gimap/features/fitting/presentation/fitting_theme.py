"""Centralized stylesheet loader for the Fitting presentation."""

from __future__ import annotations

from pathlib import Path


def fitting_stylesheet() -> str:
    return Path(__file__).with_name("fitting_theme.qss").read_text(encoding="utf-8")


__all__ = ["fitting_stylesheet"]
