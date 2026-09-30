"""Value objects produced by the Analyze reductions (framework neutral)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

GISAXS = "gisaxs"
GIWAXS = "giwaxs"
MEASUREMENT_KINDS = (GISAXS, GIWAXS)


@dataclass(frozen=True)
class Curve:
    """One reduced 1D curve with the region it came from.

    ``x`` is in the unit named by ``x_label``; ``intensity`` is the mean pixel
    value per bin (detector counts unless the frame was normalised upstream);
    ``sigma`` is the Poisson standard error of that mean, ``pixels`` the number
    of pixels averaged in each bin.
    """

    key: str
    title: str
    x: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    pixels: np.ndarray
    x_label: str
    y_label: str = "I (counts/pixel)"
    region: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sizes = {len(self.x), len(self.intensity), len(self.sigma), len(self.pixels)}
        if len(sizes) != 1:
            raise ValueError(f"Curve {self.key!r} arrays differ in length: {sorted(sizes)}")

    @property
    def is_empty(self) -> bool:
        return len(self.x) == 0


@dataclass(frozen=True)
class YonedaEstimate:
    """Row of the Yoneda band found in the side bands next to the specular rod."""

    row: float
    """Continuous row coordinate (pixel centre, canonical frame)."""
    alpha_f_deg: float
    search_rows: tuple[int, int]
    side_columns: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class GisaxsCutSettings:
    """User adjustments of the automatic GISAXS cuts; ``None`` means automatic."""

    horizontal_row: float | None = None
    horizontal_half_height_px: float = 2.5
    vertical_column: float | None = None
    vertical_half_width_px: float = 5.0


@dataclass(frozen=True)
class Sector:
    """A user-defined χ range (and optional q range) of a GIWAXS pattern.

    χ follows the reduction convention: 0° along the surface normal, ±90° in
    plane, the sign that of q∥.
    """

    chi_min_deg: float
    chi_max_deg: float
    q_min: float | None = None
    q_max: float | None = None


@dataclass(frozen=True)
class QBox:
    """A rectangle in the q∥–qz plane (Å⁻¹), e.g. around a Bragg rod or peak."""

    q_parallel: tuple[float, float]
    qz: tuple[float, float]


X_AXIS_Q = "q"
X_AXIS_TWO_THETA = "two_theta"


@dataclass(frozen=True)
class GiwaxsSettings:
    """Sector widths and the q window of the azimuthal profile (``None`` = automatic)."""

    in_plane_half_width_deg: float = 10.0
    out_of_plane_half_width_deg: float = 10.0
    chi_q_window: tuple[float, float] | None = None
    bins: int | None = None
    """Radial bins of the I(q) curves; ``None`` chooses from the detector size."""
    x_axis: str = X_AXIS_Q
    """``q`` (Å⁻¹) or ``two_theta`` (degrees) for the I(q)-type curves."""
    sector: Sector | None = None
    box: QBox | None = None
    regions: tuple = ()
    """Cut regions the person drew (``CutRegion``): each gives I(q) and I(χ), keys ``region1``, ``region1_chi`` …"""


@dataclass(frozen=True)
class ReciprocalSpaceMap:
    """Intensity regridded onto a regular (q∥ or qy, qz) grid; row 0 is the largest qz.

    ``x_label`` names the horizontal axis: q∥ for GIWAXS, qy for GISAXS.
    """

    image: np.ndarray
    q_parallel_range: tuple[float, float]
    qz_range: tuple[float, float]
    x_label: str = "q∥ (Å⁻¹)"

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Bin centres: the horizontal axis (columns) and qz (rows, from the largest down)."""
        rows, columns = self.image.shape
        (x0, x1), (z0, z1) = self.q_parallel_range, self.qz_range
        x = x0 + (np.arange(columns) + 0.5) * (x1 - x0) / columns
        z = z0 + (np.arange(rows) + 0.5) * (z1 - z0) / rows
        return x, z[::-1]


@dataclass(frozen=True)
class Reduction:
    """Everything one frame reduction produced."""

    kind: str
    curves: tuple[Curve, ...]
    markers: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    reciprocal_space_map: ReciprocalSpaceMap | None = None

    def curve(self, key: str) -> Curve | None:
        return next((curve for curve in self.curves if curve.key == key), None)


__all__ = [
    "Curve",
    "GISAXS",
    "GIWAXS",
    "GisaxsCutSettings",
    "GiwaxsSettings",
    "MEASUREMENT_KINDS",
    "QBox",
    "ReciprocalSpaceMap",
    "Reduction",
    "Sector",
    "X_AXIS_Q",
    "X_AXIS_TWO_THETA",
    "YonedaEstimate",
]
