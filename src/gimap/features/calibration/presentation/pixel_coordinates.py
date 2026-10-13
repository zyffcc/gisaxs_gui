"""Pixel coordinates as the Geometry Calibration window shows and accepts them.

The fit works with pixel-centre indices (``CalibrationCandidate.center_x_px`` / ``center_y_px``: the centre of
the first pixel is 0.0, row 0 at the top), and an exported calibration stores those numbers. A person sees and
types canonical detector pixels, the ones of Analyze and of the instrument profile that Apply saves: pixel
corners on integers, so the pixel in row i and column j spans x = j…j+1 and y = i…i+1, and the centre of the
first pixel is (0.5, 0.5).

Every beam centre that goes to the screen or comes from it passes through ``shown_center`` or
``fitted_center``; the arithmetic itself is ``shared/geometry/conventions.py``. Distances and ring radii are
lengths: they are the same in both conventions.
"""

from __future__ import annotations

from src.gimap.shared.geometry.conventions import (
    canonical_from_index_center,
    index_center_from_canonical,
)


def shown_center(candidate) -> tuple[float, float]:
    """The beam centre of a fitted candidate as the window shows it (canonical pixels)."""
    return canonical_from_index_center(candidate.center_x_px, candidate.center_y_px)


def fitted_center(shown_x_px: float, shown_y_px: float) -> tuple[float, float]:
    """A beam centre typed or dragged in the window (canonical pixels) as the fit holds it."""
    return index_center_from_canonical(shown_x_px, shown_y_px)


def image_limits(width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """x and y limits that show the whole frame, row 0 at the top."""
    return (0.0, float(width)), (float(height), 0.0)


def image_extent(sampled_shape: tuple[int, int], stride: int = 1) -> tuple[float, float, float, float]:
    """``imshow`` extent (left, right, bottom, top) of a preview that keeps every ``stride``-th pixel.

    Sample k is pixel ``k * stride`` of the frame, whose centre is at ``k * stride + 0.5``: its block is drawn
    centred there. With every pixel kept (``stride`` 1) pixel j spans j…j+1.
    """
    rows, columns = int(sampled_shape[0]), int(sampled_shape[1])
    first_edge = 0.5 - 0.5 * stride
    return (
        first_edge,
        first_edge + columns * stride,
        first_edge + rows * stride,
        first_edge,
    )


__all__ = ["fitted_center", "image_extent", "image_limits", "shown_center"]
