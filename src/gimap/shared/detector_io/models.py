"""Framework-neutral detector frame exchanged across feature boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetectorImage:
    """A detector frame in the application's established display orientation."""

    data: np.ndarray
    mask: np.ndarray | None
    source_path: Path
    detector_name: str | None = None
    pixel_size_x_m: float | None = None
    pixel_size_y_m: float | None = None
    energy_kev: float | None = None
    wavelength_angstrom: float | None = None
    distance_m: float | None = None
    beam_center_x_px: float | None = None
    beam_center_y_px: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
