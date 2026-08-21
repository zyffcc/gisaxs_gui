"""Load the feature-owned WAXS stylesheet."""

from pathlib import Path


def waxs_stylesheet() -> str:
    return Path(__file__).with_name("waxs_theme.qss").read_text(encoding="utf-8")


__all__ = ["waxs_stylesheet"]
