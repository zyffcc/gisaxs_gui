"""Geometry Calibration refines the detector tilt (pyFAI's rot1, rot2) with the centre and the distance.

The tilt stage (``calibration/domain/tilt_fit.py``) fits five parameters to points on the rings of the standard, with
each point's scattering angle from GIMaP's one pixel→q mapping. Checked here: the tilt of synthetic rings is found
within 0.05° (ring points of the public Xeuss 6° geometry; a whole synthetic LaB6 frame through the engine), an
untilted geometry keeps the q of every pixel to 1e-12, the record keeps the tilt (and stays readable by an older
GIMaP when there is none), and the public AgBh frame gives a smaller ring residual and the .poni's beam centre.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.calibration.domain.engine import CalibrationEngine
from src.gimap.features.calibration.domain.instrument_profile import calibrated_geometry, calibration_poni_text
from src.gimap.features.calibration.domain.models import CalibrationCandidate, CalibrationResult, DetectorImage
from src.gimap.features.calibration.domain.standards import STANDARDS
from src.gimap.features.calibration.domain.tilt_fit import fit_tilted_geometry, tilt_residual
from src.gimap.shared.detector_io.radiation import energy_to_wavelength
from src.gimap.shared.geometry import DetectorGeometry, transmission_map
from src.gimap.shared.geometry.poni import read_poni

EXTERNAL = Path(__file__).resolve().parent / "data" / "external"
XEUSS_PONI = EXTERNAL / "giwaxs_xeuss_pbte" / "AgBe_250403_XENOCS.poni"
AGBH_FRAME = EXTERNAL / "saxs_agbh_pyfai" / "Pilatus1M.edf"
TILT_TOLERANCE_DEG = 0.05


def _ring_points(geometry: DetectorGeometry, shape, q_values, width=0.0015):
    """Pixel centres (engine indices) within ``width`` of each q of a geometry, and their q."""
    q = transmission_map(shape, geometry).q
    xs, ys, qs = [], [], []
    for value in q_values:
        rows, columns = np.nonzero(np.abs(q - value) < width)
        if rows.size >= 30:
            xs.append(columns.astype(float))
            ys.append(rows.astype(float))
            qs.append(np.full(rows.size, value))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(qs)


@pytest.mark.skipif(not XEUSS_PONI.exists(), reason="public test data")
def test_ring_points_of_the_xeuss_six_degree_geometry_give_its_tilt() -> None:
    poni = read_poni(XEUSS_PONI.read_text(encoding="utf-8"))
    truth = poni.geometry()  # direct beam, distance to it, rot1 −5.10°, rot2 −3.27°
    assert truth.tilt_deg == pytest.approx(6.06, abs=0.01)
    xs, ys, qs = _ring_points(truth, (619, 487), STANDARDS["agbh"].q_values_inv_angstrom)
    cx, cy = truth.beam_center_x_px - 0.5, truth.beam_center_y_px - 0.5  # engine indices count centres from 0
    start = [cx + 5.0, cy - 4.0, truth.distance_m * 1e3 * 1.03, 0.0, 0.0]  # an untilted guess, 3 % off in distance
    fit = fit_tilted_geometry(xs, ys, qs, truth.wavelength_angstrom, truth.pixel_size_x_m, truth.pixel_size_y_m, start)
    assert math.degrees(fit[3]) == pytest.approx(math.degrees(truth.rot1_rad), abs=TILT_TOLERANCE_DEG)
    assert math.degrees(fit[4]) == pytest.approx(math.degrees(truth.rot2_rad), abs=TILT_TOLERANCE_DEG)
    assert fit[0] == pytest.approx(cx, abs=0.2) and fit[1] == pytest.approx(cy, abs=0.2)
    assert fit[2] == pytest.approx(truth.distance_m * 1e3, rel=1e-3)


def test_the_residual_of_an_untilted_detector_is_the_circle_fit() -> None:
    """Without a tilt the residual is the radial pixel distance of the circle fit (``optimizer._ring_residual``)."""
    from src.gimap.features.calibration.domain.optimizer import _ring_residual

    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(0, 400, 200), rng.uniform(0, 400, 200)
    qs = np.full(200, 0.3)
    flat = _ring_residual(xs, ys, qs, 1.0, 172e-6, 172e-6)(np.array([200.0, 210.0, 900.0]))
    tilted = tilt_residual(xs, ys, qs, 1.0, 172e-6, 172e-6)(np.array([200.0, 210.0, 900.0, 0.0, 0.0]))
    assert np.allclose(flat, tilted, rtol=0, atol=1e-9)


def _tilted_lab6(rot1_deg: float, rot2_deg: float):
    energy, distance_mm, pixel, shape, centre = 12.0, 200.0, 172e-6, (1200, 1200), (590.3, 610.7)
    geometry = DetectorGeometry(pixel, pixel, distance_mm * 1e-3, centre[0], centre[1], energy_to_wavelength(energy),
                                rot1_rad=math.radians(rot1_deg), rot2_rad=math.radians(rot2_deg))
    q = transmission_map(shape, geometry).q
    data = np.zeros(shape, np.float32)
    for order, value in enumerate(STANDARDS["lab6"].q_values_inv_angstrom[:8]):
        data += 300.0 / (1 + 0.3 * order) * np.exp(-0.5 * ((q - value) / 0.0025) ** 2)
    data += np.random.default_rng(3).poisson(2.0, shape).astype(np.float32)
    return data, geometry, energy


def _calibrate(data, energy, *, tilt="auto", rot3_deg=0.0):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "lab6_synthetic.npy"
        path.write_bytes(b"synthetic standard")
        image = DetectorImage(data, np.zeros(data.shape, bool), path, pixel_size_x_m=172e-6, pixel_size_y_m=172e-6)
        return CalibrationEngine().calibrate(image, energy_kev=energy, standard_key="lab6", estimated_distance_mm=190,
                                             distance_range_mm=(150, 600), tilt=tilt, rot3_deg=rot3_deg,
                                             energy_source="entered by user")


def test_the_engine_finds_the_tilt_of_a_synthetic_lab6_frame_and_records_it() -> None:
    data, truth, energy = _tilted_lab6(2.0, -1.5)
    result = _calibrate(data, energy, rot3_deg=7.0)
    best = result.selected_candidate
    assert best.rot3_deg == pytest.approx(7.0)
    assert best.tilt_refined and best.matched_ring_count >= 3
    assert best.rot1_deg == pytest.approx(2.0, abs=TILT_TOLERANCE_DEG)
    assert best.rot2_deg == pytest.approx(-1.5, abs=TILT_TOLERANCE_DEG)
    assert best.rot1_sigma_deg is not None and 0 < best.rot1_sigma_deg < 0.05  # 1σ, as for the centre and distance
    assert best.uncertainty_method == "ring jackknife" and best.distance_sigma_mm > 0
    assert best.point_rms_px < best.untilted_point_rms_px
    geometry = calibrated_geometry(result)  # the profile Analyze gets: tilted, centre and distance of the direct beam
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == pytest.approx(
        (truth.beam_center_x_px, truth.beam_center_y_px), abs=0.05)
    assert geometry.distance_m == pytest.approx(truth.distance_m, rel=2e-4)
    assert math.degrees(geometry.rot1_rad) == pytest.approx(best.rot1_deg)
    assert result.metadata["energy_source"] == "entered by user" and result.metadata["tilt_mode"] == "auto"
    record = result.to_dict()
    assert record["selected_candidate"]["rot1_deg"] == pytest.approx(best.rot1_deg)
    again = CalibrationResult.from_dict(record)
    assert again.selected_candidate.rot2_deg == pytest.approx(best.rot2_deg) and again.selected_candidate.tilt_refined
    poni = read_poni(calibration_poni_text(result))  # Export Calibration… as .poni gives the same geometry back
    back = poni.geometry()
    assert (back.beam_center_x_px, back.beam_center_y_px, back.distance_m) == pytest.approx(
        (geometry.beam_center_x_px, geometry.beam_center_y_px, geometry.distance_m), rel=1e-12)
    assert (back.rot1_rad, back.rot2_rad) == pytest.approx((geometry.rot1_rad, geometry.rot2_rad), abs=1e-15)
    assert poni.shape == (1200, 1200) and poni.wavelength_angstrom == pytest.approx(geometry.wavelength_angstrom)


def test_fixed_rotation_survives_when_ring_tilt_is_disabled():
    data, _, energy = _tilted_lab6(0.0, 0.0)
    result = _calibrate(data, energy, tilt="off", rot3_deg=-3.5)
    assert all(c.rot3_deg == -3.5 and not c.tilt_refined for c in result.candidates)
    assert read_poni(calibration_poni_text(result)).geometry().rot3_rad == pytest.approx(math.radians(-3.5))


def test_without_the_tilt_the_record_is_the_one_an_older_gimap_reads() -> None:
    candidate = CalibrationCandidate("agbh", 10.0, 12.0, 900.0, matched_ring_count=4)
    record = candidate.to_dict()
    assert "rot1_deg" not in record and "tilt_refined" not in record  # an older reader takes it key by key
    assert CalibrationCandidate.from_dict(record).rot1_deg == 0.0
    few = CalibrationCandidate("agbh", 10.0, 12.0, 900.0, matched_ring_count=2)
    from src.gimap.features.calibration.domain.tilt_fit import NOT_REFINED_FEW_RINGS, refine_tilt

    assert not refine_tilt(None, few, 1.0, 172e-6, 172e-6).tilt_refined  # type: ignore[arg-type]
    assert NOT_REFINED_FEW_RINGS in few.warnings


def test_an_untilted_calibration_keeps_the_q_of_every_pixel() -> None:
    """The tilt fields of a record and the .poni round trip change nothing for a detector that is not tilted."""
    shape = (300, 280)
    plain = DetectorGeometry(172e-6, 172e-6, 1.2, 140.25, 151.75, 1.0332)
    candidate = CalibrationCandidate("agbh", 139.75, 151.25, 1200.0, matched_ring_count=4)
    result = CalibrationResult("x.edf", 1, 1, "0", 12.0, 1.0332, None, 172e-6, 172e-6, candidate, [candidate], "now",
                               metadata={"image_shape": list(shape)})
    calibrated = calibrated_geometry(result)
    assert not calibrated.tilted
    reference = transmission_map(shape, plain).q
    assert np.max(np.abs(transmission_map(shape, calibrated).q - reference)) <= 1e-12
    back = read_poni(calibration_poni_text(result)).geometry()
    assert np.max(np.abs(transmission_map(shape, back).q - reference)) <= 1e-12


@pytest.mark.skipif(not AGBH_FRAME.exists(), reason="public test data")
def test_the_public_agbh_frame_is_fitted_better_with_the_tilt() -> None:
    import fabio

    data = fabio.open(str(AGBH_FRAME)).data.astype(np.float32)
    image = DetectorImage(data, data < 0, AGBH_FRAME, pixel_size_x_m=172e-6, pixel_size_y_m=172e-6)
    best = CalibrationEngine().calibrate(image, energy_kev=12.0, standard_key="agbh").selected_candidate
    assert best.tilt_refined and best.matched_ring_count >= 3
    assert best.point_rms_px < 0.5 * best.untilted_point_rms_px
    # the .poni's direct beam (179.98, 263.86) px; the engine counts pixel centres from 0
    assert best.center_x_px + 0.5 == pytest.approx(179.98, abs=0.5)
    assert best.center_y_px + 0.5 == pytest.approx(263.86, abs=0.5)
    assert abs(best.rot1_deg) < 1.0 and abs(best.rot2_deg) < 1.0  # a small tilt (the .poni: 0.37°, 0.43°)
    assert best.rot2_deg == pytest.approx(0.433, abs=3 * best.rot2_sigma_deg)
