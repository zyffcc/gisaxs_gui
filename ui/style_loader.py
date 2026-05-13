"""QSS loading helpers for application widgets."""

from pathlib import Path


STYLE_DIR = Path(__file__).resolve().parent / "styles"
MAIN_WINDOW_QSS = STYLE_DIR / "main_window.qss"


def apply_main_window_styles(window) -> None:
    """Apply the main-window stylesheet if it is available."""
    if not MAIN_WINDOW_QSS.exists():
        return

    stylesheet = MAIN_WINDOW_QSS.read_text(encoding="utf-8")
    chevron_path = (STYLE_DIR / "icons" / "chevron-down.svg").as_posix()
    chevron_up_path = (STYLE_DIR / "icons" / "chevron-up.svg").as_posix()
    stylesheet = stylesheet.replace("@CHEVRON_DOWN@", chevron_path)
    stylesheet = stylesheet.replace("@CHEVRON_UP@", chevron_up_path)
    window.setStyleSheet(stylesheet)
