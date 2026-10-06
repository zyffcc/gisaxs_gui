"""Saved figures write Chinese titles with a Chinese font, and leave English figures as they are."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def _names() -> set[str]:
    from matplotlib import font_manager

    return {font.name for font in font_manager.fontManager.ttflist}


def test_chinese_texts_get_a_chinese_font_and_english_ones_do_not(tmp_path: Path) -> None:
    from src.gimap.shared.figures import CJK_FONTS, cjk_fallback, publication_figure, save_figure

    if not any(name in _names() for name in CJK_FONTS):
        pytest.skip("no Chinese font on this machine")
    figure = publication_figure()
    axes = figure.add_subplot()
    axes.plot([0, 1], [1, 2])
    title = axes.set_title("水平切线 I(qy)")
    label = axes.set_xlabel("qy (Å⁻¹)")
    before = label.get_fontfamily()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # “Glyph … missing from font” would be a UserWarning
        written = save_figure(figure, tmp_path / "cut.png")
    assert written.is_file() and written.stat().st_size > 0
    assert any(name in CJK_FONTS for name in title.get_fontfamily())
    assert label.get_fontfamily() == before  # English and units keep Matplotlib's font

    english = publication_figure()
    english.add_subplot().set_title("Horizontal cut")
    assert cjk_fallback(english) == 0
