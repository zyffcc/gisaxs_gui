"""The trend figures of 2D Prediction find a Chinese font on every system the shared figures do."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.shared.figures import CJK_FONTS


def test_the_trend_figures_know_the_fonts_matplotlib_lists_on_a_mac() -> None:
    from src.gimap.features.prediction.presentation import trend_windows

    # Matplotlib knows a font collection by its first face: PingFang is "PingFang HK" there, never "PingFang SC".
    assert {"PingFang HK", "Hiragino Sans GB", "Heiti TC", "Songti SC", "Arial Unicode MS"} <= set(trend_windows._CJK_FONTS)
    assert set(CJK_FONTS) <= set(trend_windows._CJK_FONTS)


def test_in_chinese_the_figure_font_is_the_first_listed_font_that_is_installed() -> None:
    from matplotlib import font_manager

    from src.gimap.features.prediction.presentation import trend_windows

    QApplication.instance() or QApplication([])
    installed = {font.name for font in font_manager.fontManager.ttflist}
    expected = next((name for name in trend_windows._CJK_FONTS if name in installed), None)
    apply_language("zh")
    try:
        font = trend_windows.figure_font()
    finally:
        apply_language(DEFAULT_LANGUAGE)
    assert (font.get("fontfamily") or [None])[-1] == expected
    assert trend_windows.figure_font() == {}  # English: Matplotlib's own font
