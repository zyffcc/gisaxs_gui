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
            full_image = None
            """The whole picture when the item holds a part of it (``image_clip.ImageClip``)."""
            full_image_nans = False
            """The whole picture has NaN (pyqtgraph colours such a picture through another path)."""

            def render(self):
                self._imageNanLocations = None
                if self.full_image is not None and self._imageHasNans is None:
                    self._imageHasNans = self.full_image_nans  # (the part coloured as the whole picture is)
                super().render()

            def getHistogram(self, *args, **kwargs):  # noqa: N802 - pyqtgraph API
                """Of the whole picture, also when the item holds the part in view."""
                if self.full_image is None:
                    return super().getHistogram(*args, **kwargs)
                part, self.image = self.image, self.full_image
                try:
                    return super().getHistogram(*args, **kwargs)
                finally:
                    self.image = part

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


class GrowingDisplay:
    """``display_image`` of a picture that grows by rows (a live Series map: ``series_growing``), made only for the
    rows that are new: ``show(picture, same, log)`` gives what ``valid = np.isfinite(picture)`` and
    ``display_image(picture, valid, log)`` give, value for value (``tests/test_series_growing.py``).

    The rows made before stay in arrays that grow in place (``shared.growing_rows``). In log display the floor of
    the pixels at or below 0 comes from the sample of all rows; when it changes, it is written again into the rows
    before (the pixels that have it are kept). The sample is taken without the floor, as ``display_image`` takes it.
    One picture is shown at a time (the caller sees to it)."""

    def __init__(self) -> None:
        self._kept = None  # (log, columns, valid rows, shown rows, floor pixels or None, floor or None)

    def forget(self) -> None:
        self._kept = None

    def show(self, picture: np.ndarray, same: int, log_scale: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(valid, shown, level_sample)`` of ``picture``; ``same``: rows at its start that are the rows of the
        picture shown before (``0``: all are new)."""
        from src.gimap.shared.growing_rows import GrowingRows

        picture = np.asarray(picture)
        kept, self._kept = self._kept, None
        if (kept is None or kept[0] != bool(log_scale) or kept[1] != picture.shape[1:] or not 0 < same == kept[2].rows
                or picture.ndim != 2):
            same, kept = 0, None
        valid = np.isfinite(picture[same:])
        shown = np.array(np.asarray(picture[same:], dtype=np.float32), dtype=np.float32, copy=True)
        shown[~valid] = np.nan
        floor_at = None
        if log_scale:
            with np.errstate(divide="ignore", invalid="ignore"):
                positive = shown > 0
                floor_at = np.isfinite(shown) & ~positive
                shown = np.where(positive, np.log10(shown), np.nan).astype(np.float32)
        if kept is None:
            kept = [bool(log_scale), picture.shape[1:], GrowingRows(valid), GrowingRows(shown),
                    None if floor_at is None else GrowingRows(floor_at), None]
        else:
            kept[2].extend(valid)
            kept[3].extend(shown)
            if floor_at is not None:
                kept[4].extend(floor_at)
        all_valid, all_shown = kept[2].view(), kept[3].view()
        if not log_scale:
            self._kept = kept
            return all_valid, all_shown, _sample(all_shown)
        floors = kept[4].view()
        sample = _sample(all_shown).copy()
        sample[_sample(floors)] = np.nan  # (the sample of the logged values, without the floor)
        finite = sample[np.isfinite(sample)]
        lower = finite if finite.size else all_shown[np.isfinite(all_shown) & ~floors]
        floor = np.float32(np.percentile(lower, FLOOR_PERCENTILE)) if lower.size else None
        if floor != kept[5] or floor is None:
            np.putmask(all_shown, floors, np.nan if floor is None else floor)  # (all rows: the floor changed)
        else:
            np.putmask(all_shown[same:], floors[same:], floor)
        kept[5] = floor
        self._kept = kept
        return all_valid, all_shown, sample


def display_array(data: np.ndarray, valid: Optional[np.ndarray], log_scale: bool) -> np.ndarray:
    """float32 array to draw (``display_image``): NaN where invalid (drawn transparent), log10 if asked, values at or
    below 0 at the lowest colour of a log display."""
    return display_image(data, valid, log_scale)[0]


__all__ = ["FLOOR_PERCENTILE", "GrowingDisplay", "display_array", "display_image", "display_levels"]
