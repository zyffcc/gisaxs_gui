"""Centralized QSS loader for shared workspace components。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PyQt5.QtWidgets import QWidget


@lru_cache(maxsize=1)
def design_system_stylesheet() -> str:
    path = Path(__file__).with_name("design_system.qss")
    return path.read_text(encoding="utf-8")


def apply_design_system(widget: QWidget) -> None:
    """Apply the shared component stylesheet without touching application data。"""
    widget.setStyleSheet(design_system_stylesheet())
