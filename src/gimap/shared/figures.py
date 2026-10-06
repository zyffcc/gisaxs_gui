"""Publication figures rendered with Matplotlib (no pyplot state).

The format follows the file suffix: PNG/TIFF (600 dpi raster), SVG or PDF
(vector).  Sizes default to one journal column (8.5 cm wide).  Used by the
Analyze and Fitting figure exports, so both look the same.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CM = 1.0 / 2.54
COLUMN_WIDTH_CM = 8.5
RASTER_DPI = 600
RASTER_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".jpg", ".jpeg"})
GAP_FACTOR = 5.0
"""A step this many times the local point spacing is a gap in the data, not a step of the curve."""
FIGURE_FILE_FILTER = "PNG image (*.png);;TIFF image (*.tif);;SVG vector (*.svg);;PDF (*.pdf)"
"""Save-dialog filter for the formats :func:`save_figure` writes."""


def publication_figure(width_cm: float = COLUMN_WIDTH_CM, height_cm: float = COLUMN_WIDTH_CM * 0.75):
    """A detached figure (Agg canvas, constrained layout) of the given size in cm."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width_cm * CM, height_cm * CM), constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def style_publication_axes(axes) -> None:
    """Inward ticks on all sides, thin spines and 7–8 pt text."""
    axes.tick_params(direction="in", top=True, right=True, labelsize=7, length=3, width=0.6)
    for spine in axes.spines.values():
        spine.set_linewidth(0.6)
    axes.xaxis.label.set_size(8)
    axes.yaxis.label.set_size(8)


def break_at_gaps(x, y) -> tuple[np.ndarray, np.ndarray]:
    """``x, y`` with a NaN point inserted wherever the data has a gap, so a line is not drawn across it.

    A curve has no point for a bin without pixels (a detector gap, a masked χ range, the missing
    wedge); joining its neighbours draws data that was never measured. A gap is a step more than
    ``GAP_FACTOR`` times the local spacing — the larger of the typical spacing and the smaller of
    the steps just before and after (so the wider spacing of merged sparse bins is not a gap).
    Only for drawing: the values are unchanged and no point is added or removed.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return x, y
    step = np.abs(np.diff(x))
    usable = np.isfinite(step) & (step > 0)
    if not usable.any():
        return x, y
    typical = float(np.median(step[usable]))
    before = np.concatenate([[np.inf], step[:-1]])
    after = np.concatenate([step[1:], [np.inf]])
    local = np.minimum(before, after)
    local = np.where(np.isfinite(local), local, typical)
    gaps = np.flatnonzero(usable & (step > GAP_FACTOR * np.maximum(local, typical)))
    if gaps.size == 0:
        return x, y
    return np.insert(x, gaps + 1, np.nan), np.insert(y, gaps + 1, np.nan)


CJK_FONTS = ("Microsoft YaHei", "Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei",
             "PingFang SC", "WenQuanYi Zen Hei")
"""Fonts with Chinese glyphs, in order of preference (Matplotlib's own font has none)."""


def _has_wide_text(text: str) -> bool:
    return any(ord(character) >= 0x2E80 for character in text)  # CJK and full-width punctuation


def cjk_fallback(figure) -> int:
    """Texts of ``figure`` with Chinese characters get a Chinese font after Matplotlib's own, so they are
    not drawn as empty boxes (Matplotlib falls back glyph by glyph; Latin letters keep their font).

    Other texts are left as they are, so an English figure is unchanged. Returns how many texts changed.
    """
    from matplotlib import font_manager
    from matplotlib.text import Text

    texts = [text for text in figure.findobj(Text) if _has_wide_text(text.get_text() or "")]
    if not texts:
        return 0
    names = {font.name for font in font_manager.fontManager.ttflist}
    fallback = next((name for name in CJK_FONTS if name in names), None)
    if fallback is None:
        return 0
    family = ["DejaVu Sans", fallback] if "DejaVu Sans" in names else [fallback]
    for text in texts:
        text.set_fontfamily(family)
    return len(texts)


def save_figure(figure, path: str | Path) -> Path:
    """Write ``figure``; raster formats at :data:`RASTER_DPI`, vector formats as vectors.

    Titles and labels in Chinese (the interface language) are written with a Chinese font."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raster = path.suffix.lower() in RASTER_SUFFIXES
    cjk_fallback(figure)
    figure.savefig(path, dpi=RASTER_DPI if raster else None)
    return path


__all__ = [
    "CJK_FONTS",
    "CM",
    "COLUMN_WIDTH_CM",
    "FIGURE_FILE_FILTER",
    "GAP_FACTOR",
    "RASTER_DPI",
    "break_at_gaps",
    "cjk_fallback",
    "publication_figure",
    "save_figure",
    "style_publication_axes",
]
