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


def _margins(path: Path) -> tuple[int, int, int, int, int, int]:
    """``(left, right, top, bottom, width, height)`` in pixels: the room around everything drawn, and the picture."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as picture:
        dark = np.asarray(picture.convert("L")) < 128
    height, width = dark.shape
    columns, rows = np.flatnonzero(dark.any(axis=0)), np.flatnonzero(dark.any(axis=1))
    return int(columns[0]), width - 1 - int(columns[-1]), int(rows[0]), height - 1 - int(rows[-1]), width, height


@pytest.mark.parametrize("shape", [(2048, 2048), (1065, 1030), (1679, 1475), (619, 487), (300, 1500), (1500, 300),
                                   (195, 1475), (516, 1556), (514, 1030), (790, 1000), (800, 1000), (1475, 195)])
def test_a_picture_of_a_frame_keeps_its_labels_inside_whatever_the_frame_shape(shape, tmp_path: Path) -> None:
    """A frame drawn with square pixels does not fill the room the layout gave its axes; the labels were then
    placed for the wrong box (a square frame lost half of “y (pixel)” at the left edge, a tall one sat at the
    right of an empty picture). Checked on the pixels of the saved picture."""
    import numpy as np

    from src.gimap.features.analyze.infrastructure.adapters import MatplotlibFigureWriter

    frame = np.random.default_rng(1).poisson(5, shape).astype(float)
    written = MatplotlibFigureWriter().write_frame(frame, None, tmp_path / "frame.png", title="frame_0001.tif")
    left, right, top, bottom, width, _height = _margins(written)
    assert min(left, right, top, bottom) >= 10, (left, right, top, bottom)  # nothing touches an edge (600 dpi)
    # … and it is centred: opposite margins within 1.7 mm of each other (5.6 mm apart for a Pilatus 2M frame before)
    assert abs(left - right) <= 40 and abs(top - bottom) <= 40, (left, right, top, bottom)
    assert width == pytest.approx(8.5 / 2.54 * 600, abs=2)  # one journal column, as before


def _longest_runs(mask):
    """The longest unbroken run of True in every column of ``mask``."""
    import numpy as np

    count = np.zeros(mask.shape[1], dtype=int)
    best = count.copy()
    for row in mask:
        count = np.where(row, count + 1, 0)
        best = np.maximum(best, count)
    return best


def _colour_bar(path: Path) -> tuple[int, int, int, int]:
    """``(length, width, height of the frame's box, height of the picture)`` in pixels, read from the saved file.

    A column of the colour bar or of the frame's box (its spines) is an unbroken run of ink; the strokes of the
    texts are shorter than 60 px at 600 dpi. The colour bar is the block of such columns furthest right."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as picture:
        inked = np.asarray(picture.convert("RGB")).min(axis=2) < 200
    runs = _longest_runs(inked)
    long = runs >= 100
    right = int(np.flatnonzero(long)[-1])
    left = right
    while left > 0 and long[left - 1]:
        left -= 1
    return int(runs[left:right + 1].max()), right - left + 1, int(runs[:left].max()), inked.shape[0]


@pytest.mark.parametrize("shape", [(195, 1475), (256, 1556), (516, 1556), (514, 1030), (600, 1000), (790, 1000),
                                   (800, 1000), (2048, 2048), (1043, 981), (1679, 1475), (619, 487), (1475, 195),
                                   (1500, 300)])
def test_the_colour_bar_of_a_frame_picture_can_be_read_whatever_the_frame_shape(shape, tmp_path: Path) -> None:
    """The scale next to the frame: long enough for its numbers, as wide as ever, and for a frame that is
    taller than wide exactly as tall as the frame. (Tied to the box of the frame it was 8 mm long with two
    numbers for a 195 x 1475 strip, and 0.6 mm wide and a fifth of the frame's height for a 1500 x 300 one.)"""
    import numpy as np

    from src.gimap.features.analyze.infrastructure.adapters import MatplotlibFigureWriter

    rows, columns = shape
    frame = np.random.default_rng(2).poisson(50, shape) * np.linspace(1.0, 300.0, columns)[None, :] + 1.0
    written = MatplotlibFigureWriter().write_frame(frame, None, tmp_path / "frame.png", title="frame_0001.tif")
    assert min(_margins(written)[:4]) >= 10, _margins(written)  # its numbers are inside the picture too
    length, width, box, height = _colour_bar(written)
    assert length >= 0.6 * height, (length, width, box, height)  # at least 43 mm of the 72 mm picture
    assert 50 <= width <= 90, (length, width, box, height)  # about 3 mm (67 px), as for a square frame
    if rows >= columns:
        assert abs(length - box) <= 0.03 * box, (length, width, box, height)
    else:
        assert length >= box - 2, (length, width, box, height)


@pytest.mark.parametrize("shape", [(3000, 100), (4439, 195), (100, 3000), (500, 1), (1, 500), (7, 7), (1, 1)])
def test_a_frame_of_any_shape_gives_a_picture(shape, tmp_path: Path, monkeypatch) -> None:
    """A long strip, one line, one pixel: a picture without a warning and with nothing at an edge.

    The colour bar's share of the frame's width is checked BEFORE Matplotlib gets it: a share near 1 leaves no
    room for the axes (“axes sizes collapsed to zero”), and with the share a 500 x 1 frame would ask for
    the renderer ends the program, which would end this test file instead of failing one test."""
    import numpy as np
    from matplotlib.figure import Figure

    from src.gimap.features.analyze.infrastructure.adapters import MatplotlibFigureWriter

    real = Figure.colorbar

    def checked(self, *args, **kwargs):
        assert kwargs["fraction"] + kwargs["pad"] <= 0.6, kwargs
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", checked)
    rows, columns = shape
    frame = np.random.default_rng(3).poisson(50, shape) * np.linspace(1.0, 300.0, columns)[None, :] + 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        written = MatplotlibFigureWriter().write_frame(frame, None, tmp_path / "frame.png", title="frame_0001.tif")
    left, right, top, bottom, width, height = _margins(written)
    assert min(left, right, top, bottom) >= 10, (left, right, top, bottom)
    assert abs(left - right) <= 40 and abs(top - bottom) <= 40, (left, right, top, bottom)
    assert (width, height) == (pytest.approx(8.5 / 2.54 * 600, abs=2), pytest.approx(8.5 * 0.85 / 2.54 * 600, abs=2))
