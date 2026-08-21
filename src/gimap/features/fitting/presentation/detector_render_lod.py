"""Screen-resolution rendering policy for detector images.

Scientific arrays stay at detector resolution.  This module only chooses a
display stride so Matplotlib does not have to paint substantially more mesh
cells than the current viewport can reveal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np


@dataclass(frozen=True)
class DetectorRenderLevel:
    """A shared stride for intensity and both detector-coordinate meshes."""

    source_shape: tuple[int, int]
    stride: int

    @property
    def rendered_shape(self) -> tuple[int, int]:
        rows, columns = self.source_shape
        return (ceil(rows / self.stride), ceil(columns / self.stride))

    @property
    def rendered_cells(self) -> int:
        rows, columns = self.rendered_shape
        return rows * columns

    def sample(self, array: np.ndarray) -> np.ndarray:
        """Return an aligned display-only view of a detector-shaped array."""

        if tuple(array.shape) != self.source_shape:
            raise ValueError(
                f"Detector render shape mismatch: {array.shape} != {self.source_shape}"
            )
        return array[:: self.stride, :: self.stride]


def detector_render_cell_budget(
    viewport_width: int,
    viewport_height: int,
    *,
    minimum_cells: int = 20_000,
    maximum_cells: int = 180_000,
    screen_pixels_per_cell: int = 6,
) -> int:
    """Choose a stable cell budget from the physical viewport size."""

    visible_pixels = max(1, int(viewport_width)) * max(1, int(viewport_height))
    screen_budget = visible_pixels // max(1, int(screen_pixels_per_cell))
    return max(int(minimum_cells), min(int(maximum_cells), screen_budget))


def choose_detector_render_level(
    image_shape: tuple[int, int],
    *,
    max_cells: int,
) -> DetectorRenderLevel:
    """Choose the smallest integer stride whose mesh fits ``max_cells``."""

    rows, columns = (int(image_shape[0]), int(image_shape[1]))
    if rows <= 0 or columns <= 0:
        raise ValueError("Detector image dimensions must be positive")
    budget = max(1, int(max_cells))
    stride = max(1, ceil(sqrt((rows * columns) / budget)))
    level = DetectorRenderLevel((rows, columns), stride)
    while level.rendered_cells > budget:
        stride += 1
        level = DetectorRenderLevel((rows, columns), stride)
    return level
