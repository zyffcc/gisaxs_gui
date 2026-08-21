"""Centralized stylesheet loader for the Prediction workbench."""

from pathlib import Path


def prediction_stylesheet() -> str:
    return Path(__file__).with_name("prediction_theme.qss").read_text(encoding="utf-8")


__all__ = ["prediction_stylesheet"]
