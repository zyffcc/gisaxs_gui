"""QSS loading helpers for application widgets."""

from pathlib import Path


STYLE_DIR = Path(__file__).resolve().parent / "styles"
MAIN_WINDOW_QSS = STYLE_DIR / "main_window.qss"


def apply_main_window_styles(window) -> None:
    """Apply the main-window stylesheet if it is available."""
    if not MAIN_WINDOW_QSS.exists():
        return

    window.setStyleSheet(MAIN_WINDOW_QSS.read_text(encoding="utf-8"))
