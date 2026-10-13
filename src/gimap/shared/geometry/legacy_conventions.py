"""Historical pixel convention of the Trainset q maps.

Trainset keeps the convention its training data were generated with; the
function below reproduces it byte-for-byte on the displacement side and
delegates the physics to :func:`~.q_mapping.scattering_vectors`.  (The former
Cut & Fitting and WAXS pages had their own conventions; both pages are gone and
Analyze uses the canonical frame with the exact model.)

In the canonical frame of :class:`~.detector_geometry.DetectorGeometry` (pixel
centre at ``j + 0.5``), Trainset puts pixel centres at 0-based indices, the
beam centre in the same indices (row 0 at the top), uses the model
``SUBTRACT_INCIDENCE`` and returns q in nm⁻¹.

Switching a feature to the exact model and the canonical pixel convention is a
scientific decision; see ``docs/architecture/geometry.md``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .q_mapping import ExitAngleModel, scattering_vectors


def trainset_q_vectors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Trainset detector grids ``qx, qy, qz, qr`` in nm⁻¹ from a project config."""
    detector = config["detector"]
    beam = config["beam"]
    nx, ny = int(detector["pixels_x"]), int(detector["pixels_y"])
    px, py = float(detector["pixel_size_x_mm"]), float(detector["pixel_size_y_mm"])
    distance = float(detector["distance_mm"])
    theta_in = np.deg2rad(float(beam["grazing_angle_deg"]))
    wavelength = float(beam["wavelength_nm"])
    center_x = float(detector["beam_center_x_px"])
    center_y = float(detector["beam_center_y_px"])
    x = (np.arange(nx, dtype=np.float64) - center_x) * px
    # Row zero is the detector top: vertical displacement is center_y - row.
    y = (center_y - np.arange(ny, dtype=np.float64)) * py
    xx, yy = np.meshgrid(x, y)
    q = scattering_vectors(
        xx,
        yy,
        distance,
        theta_in,
        2.0 * np.pi / wavelength,
        ExitAngleModel.SUBTRACT_INCIDENCE,
    )
    return {"qx": q.qx, "qy": q.qy, "qz": q.qz, "qr": q.q_parallel}
