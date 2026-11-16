"""1σ of the beam centre and the distance of a calibration: the scatter between the rings (jackknife over rings).

The points along one ring move together when the ring is slightly off, so they are not independent; the error is
taken from fits that leave one ring out each. With two rings only the least-squares covariance is left (a lower
bound)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.gimap.features.calibration.domain.engine import CalibrationEngine
from src.gimap.features.calibration.domain.geometry_model import q_to_ring_radius_px
from src.gimap.features.calibration.domain.models import CalibrationCandidate, DetectorImage
from src.gimap.features.calibration.domain.optimizer import estimate_uncertainty
from src.gimap.features.calibration.domain.standards import STANDARDS
from tests.test_calibration import synthetic_rings

ENERGY, DISTANCE, PIXEL, CENTRE = 12.0, 1000.0, 172e-6, (442.3, 467.8)


def _calibrate(rings: int):
    radii = q_to_ring_radius_px(STANDARDS["agbh"].q_values_inv_angstrom[:rings], ENERGY, DISTANCE, PIXEL)
    data = synthetic_rings(shape=(900, 900), center=CENTRE, radii=tuple(float(r) for r in radii))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "standard.npy"
        path.write_bytes(b"synthetic calibration source")
        image = DetectorImage(data, ~np.isfinite(data), path, pixel_size_x_m=PIXEL, pixel_size_y_m=PIXEL)
        return CalibrationEngine().calibrate(image, energy_kev=ENERGY, standard_key="agbh", estimated_distance_mm=950,
                                             distance_range_mm=(500, 1600))


def test_five_rings_give_a_jackknife_error_that_covers_the_true_centre() -> None:
    best = _calibrate(5).selected_candidate
    assert best.uncertainty_method == "ring jackknife"
    assert 0 < best.center_x_sigma_px < 0.2 and 0 < best.center_y_sigma_px < 0.2 and 0 < best.distance_sigma_mm < 1.0
    # the synthetic ring centre counts pixel centres from 0 like the engine (``center_convention`` "index")
    assert abs(best.center_x_px - CENTRE[0]) < 3 * best.center_x_sigma_px + 0.02
    assert abs(best.center_y_px - CENTRE[1]) < 3 * best.center_y_sigma_px + 0.02
    record = _calibrate(5).to_dict()["selected_candidate"]
    assert record["uncertainty_method"] == "ring jackknife" and record["distance_sigma_mm"] > 0


def test_one_ring_gives_no_error_and_an_older_record_reads_without_one() -> None:
    candidate = CalibrationCandidate("agbh", 10.0, 10.0, 1000.0, matched_ring_count=1)
    assert estimate_uncertainty(None, candidate, 1.0, PIXEL, PIXEL).uncertainty_method == ""  # type: ignore[arg-type]
    old = {key: value for key, value in candidate.to_dict().items()
           if key not in ("center_x_sigma_px", "center_y_sigma_px", "distance_sigma_mm", "uncertainty_method")}
    restored = CalibrationCandidate.from_dict(old)
    assert restored.center_x_sigma_px is None and restored.uncertainty_method == ""
