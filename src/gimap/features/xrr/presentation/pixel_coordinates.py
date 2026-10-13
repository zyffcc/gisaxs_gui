"""Pixel coordinates as the XRR Series Extractor shows and accepts them.

The extraction works with pixel-centre indices (``SpecularGeometry.beam_center_x_px`` / ``beam_center_y_px``
and ``XrrPoint.roi_center_x_px`` / ``roi_center_y_px``: the centre of the first pixel is 0.0, x is the column
and y the row, row 0 at the top). The settings Geometry Calibration's Apply writes (``detector.beam_center_x``
/ ``beam_center_y``), the exported CSV and its JSON record hold those numbers too. A person sees and types
canonical detector pixels, the ones of Analyze and of the Geometry Calibration window: pixel corners on
integers, so the pixel in row i and column j spans x = j…j+1 and y = i…i+1, and the centre of the first pixel
is (0.5, 0.5).

Every position that goes to the screen or comes from it passes through this module; the arithmetic itself is
``shared/geometry/conventions.py``. The ROI radius and the distance are lengths: the same in both conventions.
"""

from __future__ import annotations

import math

from src.gimap.shared.geometry.conventions import (
    canonical_from_index_center,
    index_center_from_canonical,
)

# The sentence of the Geometry Calibration window, word for word: one entry in the Chinese table serves both.
PIXEL_NOTE = (
    "Direct-beam position in detector pixels, the same as in Analyze: pixel j spans [j, j+1] "
    "(the centre of the first pixel is 0.5), x from the left edge and y from the top edge."
)
ROI_PIXEL_NOTE = (
    "Centre of the ROI in detector pixels, the same as in Analyze: pixel j spans [j, j+1] (the centre of the "
    "first pixel is 0.5). The exported CSV counts pixel centres from 0: its roi_center values are 0.5 px lower."
)


def shown_coordinate(index_px: float) -> float:
    """One coordinate of a position the extraction holds, as the window shows it (x and y count alike)."""
    return canonical_from_index_center(index_px, index_px)[0]


def shown_roi_center(point) -> tuple[float, float]:
    """The centre of the ROI of an extracted point, as the table and the preview show it."""
    return canonical_from_index_center(point.roi_center_x_px, point.roi_center_y_px)


def shown_image_center(shape) -> tuple[float, float]:
    """The middle of a frame of ``shape`` (rows, columns), as the window shows it: (columns / 2, rows / 2)."""
    rows, columns = int(shape[0]), int(shape[1])
    return canonical_from_index_center((columns - 1) / 2.0, (rows - 1) / 2.0)


def extraction_coordinate(shown_px: float, decimals: int) -> float:
    """One coordinate typed or picked in the window, as the extraction holds it.

    A field holds a decimal number with ``decimals`` places. Half a pixel less is such a number too, and it is
    returned as the float nearest to it: the float the former window, which took indices, gave when that
    number was typed. (Without this the subtraction can be one unit in the last place off just below a power
    of two, e.g. 256.2 − 0.5.)
    """
    return round(index_center_from_canonical(shown_px, shown_px)[0], int(decimals))


def image_limits(full_shape) -> tuple[tuple[float, float], tuple[float, float]]:
    """x and y limits that show the whole frame of ``full_shape`` (rows, columns), row 0 at the top."""
    rows, columns = int(full_shape[0]), int(full_shape[1])
    return (0.0, float(columns)), (float(rows), 0.0)


def preview_stride(sampled_shape, full_shape) -> int:
    """How many pixels of the frame lie between two neighbours of a reduced preview.

    The preview keeps every ``stride``-th pixel, starting with the first, so that its longer side fits a
    bound: ``kept = ceil(size / stride)``. The smallest step that gives ``kept`` is ``ceil(size / kept)``;
    along the longer side, which fixed the step, that is the step itself.
    """
    steps = [
        math.ceil(int(size) / int(kept))
        for kept, size in zip(sampled_shape[:2], full_shape[:2])
        if int(kept) > 0
    ]
    return max([1, *steps])


def image_extent(sampled_shape, full_shape) -> tuple[float, float, float, float]:
    """``imshow`` extent (left, right, bottom, top) of the preview of a frame of ``full_shape``.

    Sample k is pixel ``k * stride`` of the frame, whose centre is at ``k * stride + 0.5``: its block is drawn
    centred there. With every pixel kept (``stride`` 1) pixel j spans j…j+1.
    """
    rows, columns = int(sampled_shape[0]), int(sampled_shape[1])
    stride = preview_stride(sampled_shape, full_shape)
    first_edge = 0.5 - 0.5 * stride
    return (
        first_edge,
        first_edge + columns * stride,
        first_edge + rows * stride,
        first_edge,
    )


__all__ = [
    "PIXEL_NOTE",
    "ROI_PIXEL_NOTE",
    "extraction_coordinate",
    "image_extent",
    "image_limits",
    "preview_stride",
    "shown_coordinate",
    "shown_image_center",
    "shown_roi_center",
]
