"""What the detector viewer draws: the image item, colour maps and the displayed array.

Display only: the scientific values are never changed here (``display_array`` returns a copy)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .levels import SAMPLE, auto_levels

FLOOR_PERCENTILE = 0.1
"""In log display, valid pixels at or below 0 are drawn at log₁₀ of this percentile of the positive values: at or
below every automatic lower limit, so they take the lowest colour of the map instead of none."""

_IMAGE_ITEM_CLASS = None


def _image_item_class():
    """``pg.ImageItem`` whose NaN mask follows the current down-sampling factor.

    pyqtgraph 0.13 caches the NaN positions of the first rendered (possibly
    down-sampled) image and reuses them after zooming changes the factor,
    which raises IndexError.  Dropping the cache before each render keeps NaN
    pixels transparent at every zoom level.
    """
    global _IMAGE_ITEM_CLASS
    if _IMAGE_ITEM_CLASS is None:
        import pyqtgraph as pg

        class NanSafeImageItem(pg.ImageItem):
            def render(self):
                self._imageNanLocations = None
                super().render()

        _IMAGE_ITEM_CLASS = NanSafeImageItem
    return _IMAGE_ITEM_CLASS


def _colormap(name: str):
    import pyqtgraph as pg

    if name == "gray":
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.getFromMatplotlib(name)


def display_levels(values: np.ndarray) -> Optional[tuple[float, float]]:
    """Robust colour limits (1–99.7 % of a sample of the finite displayed values; pass the sample of
    ``display_image`` so that in log display the pixels at or below 0 do not count)."""
    return auto_levels(values)


def _sample(array: np.ndarray) -> np.ndarray:
    """A strided sample of a large frame (as ``auto_levels`` takes one), so its percentiles are quick."""
    stride = max(1, int(np.ceil(np.sqrt(array.size / SAMPLE)))) if array.ndim == 2 else 1
    return array[::stride, ::stride] if array.ndim == 2 else array


def display_image(data: np.ndarray, valid: Optional[np.ndarray], log_scale: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(shown, level_sample)``: the float32 array to draw and the displayed values the automatic limits come from.

    Invalid (masked) pixels are NaN: drawn transparent. Linear: the values as they are, limits from all of them.
    Log: log₁₀ of the positive values; valid pixels at or below 0 (the noise of a dark- or background-subtracted
    frame) get the floor, ``FLOOR_PERCENTILE`` of the positive values: the lowest colour, not a hole (a down-sampled
    view would otherwise turn every block that holds one of them black). The limits come from the positive values
    only."""
    shown = np.array(np.asarray(data, dtype=np.float32), dtype=np.float32, copy=True)
    if valid is not None:
        shown[~np.asarray(valid, dtype=bool)] = np.nan
    if not log_scale:
        return shown, _sample(shown)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = shown > 0
        logged = np.where(positive, np.log10(shown), np.nan).astype(np.float32)
    sample = _sample(logged).copy()  # (a view would see the floor written below)
    finite = sample[np.isfinite(sample)]
    lower = finite if finite.size else logged[np.isfinite(logged)]
    if lower.size:
        logged[np.isfinite(shown) & ~positive] = np.float32(np.percentile(lower, FLOOR_PERCENTILE))
    return logged, sample


def display_array(data: np.ndarray, valid: Optional[np.ndarray], log_scale: bool) -> np.ndarray:
    """float32 array to draw (``display_image``): NaN where invalid (drawn transparent), log10 if asked, values at or
    below 0 at the lowest colour of a log display."""
    return display_image(data, valid, log_scale)[0]


__all__ = ["FLOOR_PERCENTILE", "display_array", "display_image", "display_levels"]
