"""QSS loading helpers for application widgets."""

from pathlib import Path

from core.user_settings import user_settings


STYLE_DIR = Path(__file__).resolve().parent / "styles"
MAIN_WINDOW_QSS = STYLE_DIR / "main_window.qss"


def _scaled_pt(base_size: float, scale_percent: int) -> str:
    size = max(7.0, base_size * scale_percent / 100.0)
    return f"{size:.1f}pt"


def apply_main_window_styles(window) -> None:
    """Apply the main-window stylesheet if it is available."""
    if not MAIN_WINDOW_QSS.exists():
        return

    scale_percent = user_settings.get_visual_font_scale()
    stylesheet = MAIN_WINDOW_QSS.read_text(encoding="utf-8")
    chevron_path = (STYLE_DIR / "icons" / "chevron-down.svg").as_posix()
    chevron_up_path = (STYLE_DIR / "icons" / "chevron-up.svg").as_posix()
    stylesheet = stylesheet.replace("@CHEVRON_DOWN@", chevron_path)
    stylesheet = stylesheet.replace("@CHEVRON_UP@", chevron_up_path)
    stylesheet = stylesheet.replace("@APP_FONT_SIZE@", _scaled_pt(9, scale_percent))
    stylesheet = stylesheet.replace("@CARD_TITLE_FONT_SIZE@", _scaled_pt(10, scale_percent))
    stylesheet = stylesheet.replace("@SECTION_TITLE_FONT_SIZE@", _scaled_pt(9, scale_percent))
    stylesheet = stylesheet.replace("@META_FONT_SIZE@", _scaled_pt(8, scale_percent))
    window.setStyleSheet(stylesheet)
