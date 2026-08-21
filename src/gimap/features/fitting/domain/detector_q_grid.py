"""Detector-aligned q-space grids and nearest-cell selection semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


HorizontalQAxis = Literal["qy", "qr"]


def normalize_horizontal_q_axis(value: object) -> HorizontalQAxis:
    """Return a supported horizontal q coordinate without leaking UI values."""

    return "qr" if str(value).strip().lower() == "qr" else "qy"


@dataclass(frozen=True)
class QGridPoint:
    """One detector cell expressed in both q and detector-pixel coordinates."""

    horizontal_q: float
    qz: float
    row: int
    column: int


@dataclass(frozen=True)
class QGridRegion:
    """A rectangular q request snapped to detector cells."""

    horizontal_min: float
    horizontal_max: float
    qz_min: float
    qz_max: float
    row_min: int
    row_max: int
    column_min: int
    column_max: int

    @property
    def center_horizontal(self) -> float:
        return (self.horizontal_min + self.horizontal_max) / 2.0

    @property
    def center_qz(self) -> float:
        return (self.qz_min + self.qz_max) / 2.0

    @property
    def width(self) -> float:
        return self.horizontal_max - self.horizontal_min

    @property
    def height(self) -> float:
        return self.qz_max - self.qz_min


@dataclass(frozen=True)
class DetectorQGrid:
    """Three q coordinates aligned with the canonical 2D analysis image.

    The stored matrices use analysis-array row order. ``display_meshes`` flips
    that order exactly once to match the detector preview's ``origin='lower'``
    projection. Scientific cut code continues to use the unflipped matrices.
    """

    qy: np.ndarray
    qz: np.ndarray
    qr: np.ndarray

    def __post_init__(self) -> None:
        qy = np.asarray(self.qy, dtype=float)
        qz = np.asarray(self.qz, dtype=float)
        qr = np.asarray(self.qr, dtype=float)
        if qy.ndim != 2 or qy.shape != qz.shape or qy.shape != qr.shape:
            raise ValueError("qy, qr, and qz grids must share one 2D detector shape")
        object.__setattr__(self, "qy", qy)
        object.__setattr__(self, "qz", qz)
        object.__setattr__(self, "qr", qr)

    def horizontal(self, axis: object) -> np.ndarray:
        """Return the selected in-plane coordinate in analysis-array order."""

        return self.qr if normalize_horizontal_q_axis(axis) == "qr" else self.qy

    def meshes(self, axis: object) -> tuple[np.ndarray, np.ndarray]:
        """Return horizontal-q and qz matrices for scientific calculations."""

        return self.horizontal(axis), self.qz

    def display_meshes(self, axis: object) -> tuple[np.ndarray, np.ndarray]:
        """Return matrices aligned with the vertically flipped preview image."""

        horizontal, qz = self.meshes(axis)
        return np.flipud(horizontal), np.flipud(qz)

    def nearest_point(self, horizontal_q: float, qz: float, axis: object) -> QGridPoint:
        """Snap a q coordinate to its nearest finite detector cell."""

        horizontal = self.horizontal(axis)
        finite = np.isfinite(horizontal) & np.isfinite(self.qz)
        if not np.any(finite):
            raise ValueError("Detector q grid has no finite cells")
        distance = np.full(horizontal.shape, np.inf, dtype=float)
        distance[finite] = np.hypot(
            horizontal[finite] - float(horizontal_q),
            self.qz[finite] - float(qz),
        )
        row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
        return QGridPoint(
            horizontal_q=float(horizontal[row, column]),
            qz=float(self.qz[row, column]),
            row=int(row),
            column=int(column),
        )

    def snap_region(
        self,
        horizontal_min: float,
        horizontal_max: float,
        qz_min: float,
        qz_max: float,
        axis: object,
    ) -> QGridRegion:
        """Snap a q rectangle to the detector cells it contains or touches."""

        horizontal = self.horizontal(axis)
        left, right = sorted((float(horizontal_min), float(horizontal_max)))
        bottom, top = sorted((float(qz_min), float(qz_max)))
        finite = np.isfinite(horizontal) & np.isfinite(self.qz)
        selected = (
            finite
            & (horizontal >= left)
            & (horizontal <= right)
            & (self.qz >= bottom)
            & (self.qz <= top)
        )
        if np.any(selected):
            rows, columns = np.where(selected)
        else:
            corners = (
                self.nearest_point(left, bottom, axis),
                self.nearest_point(left, top, axis),
                self.nearest_point(right, bottom, axis),
                self.nearest_point(right, top, axis),
            )
            rows = np.asarray([point.row for point in corners], dtype=int)
            columns = np.asarray([point.column for point in corners], dtype=int)

        row_min, row_max = int(rows.min()), int(rows.max())
        column_min, column_max = int(columns.min()), int(columns.max())
        cell_mask = selected.copy()
        if not np.any(cell_mask):
            cell_mask[row_min : row_max + 1, column_min : column_max + 1] = finite[
                row_min : row_max + 1, column_min : column_max + 1
            ]
        horizontal_values = horizontal[cell_mask]
        qz_values = self.qz[cell_mask]
        return QGridRegion(
            horizontal_min=float(np.min(horizontal_values)),
            horizontal_max=float(np.max(horizontal_values)),
            qz_min=float(np.min(qz_values)),
            qz_max=float(np.max(qz_values)),
            row_min=row_min,
            row_max=row_max,
            column_min=column_min,
            column_max=column_max,
        )

    def region_from_pixels(
        self,
        row_min: int,
        row_max: int,
        column_min: int,
        column_max: int,
        axis: object,
    ) -> QGridRegion:
        """Express one detector-cell rectangle in the selected q coordinates."""

        horizontal = self.horizontal(axis)
        rows = sorted(
            (
                int(np.clip(row_min, 0, horizontal.shape[0] - 1)),
                int(np.clip(row_max, 0, horizontal.shape[0] - 1)),
            )
        )
        columns = sorted(
            (
                int(np.clip(column_min, 0, horizontal.shape[1] - 1)),
                int(np.clip(column_max, 0, horizontal.shape[1] - 1)),
            )
        )
        row_slice = slice(rows[0], rows[1] + 1)
        column_slice = slice(columns[0], columns[1] + 1)
        horizontal_values = horizontal[row_slice, column_slice]
        qz_values = self.qz[row_slice, column_slice]
        finite = np.isfinite(horizontal_values) & np.isfinite(qz_values)
        if not np.any(finite):
            raise ValueError("Selected detector cells have no finite q coordinates")
        return QGridRegion(
            horizontal_min=float(np.min(horizontal_values[finite])),
            horizontal_max=float(np.max(horizontal_values[finite])),
            qz_min=float(np.min(qz_values[finite])),
            qz_max=float(np.max(qz_values[finite])),
            row_min=rows[0],
            row_max=rows[1],
            column_min=columns[0],
            column_max=columns[1],
        )


__all__ = [
    "DetectorQGrid",
    "HorizontalQAxis",
    "QGridPoint",
    "QGridRegion",
    "normalize_horizontal_q_axis",
]
