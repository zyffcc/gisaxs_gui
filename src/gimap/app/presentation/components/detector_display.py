"""What the detector viewer draws: the image item, colour maps and the displayed array.

Display only: the scientific values are never changed here (``display_array`` returns a copy)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .levels import auto_levels

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
    """Robust colour limits (1–99.7 % of a sample of the finite displayed values)."""
    return auto_levels(values)


def display_array(data: np.ndarray, valid: Optional[np.ndarray], log_scale: bool) -> np.ndarray:
    """float32 array to draw: NaN where invalid (drawn transparent), log10 if asked."""
    array = np.asarray(data, dtype=np.float32)
    shown = np.array(array, dtype=np.float32, copy=True)
    if valid is not None:
        shown[~np.asarray(valid, dtype=bool)] = np.nan
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            shown = np.where(shown > 0, np.log10(shown), np.nan).astype(np.float32)
    return shown


__all__ = ["display_array", "display_levels"]
