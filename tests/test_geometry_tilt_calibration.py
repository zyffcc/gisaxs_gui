"""Geometry Calibration refines the detector tilt (pyFAI's rot1, rot2) with the centre and the distance.

The tilt stage (``calibration/domain/tilt_fit.py``) fits five parameters to points on the rings of the standard, with
each point's scattering angle from GIMaP's one pixel→q mapping. Checked here: the tilt of synthetic rings is found
within 0.05° (ring points of the public Xeuss 6° geometry; a whole synthetic LaB6 frame through the engine), an
untilted geometry keeps the q of every pixel to 1e-12, the record keeps the tilt (and stays readable by an older
GIMaP when there is none), and the public AgBh frame gives five rings and agrees with pyFAI's fit of the same frame.
"""

from __future__ import annotations

import json
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
from src.gimap.shared.detector_io.radiation import HC_KEV_ANGSTROM, energy_to_wavelength
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
    fixed = fit_tilted_geometry(xs, ys, qs, truth.wavelength_angstrom, truth.pixel_size_x_m, truth.pixel_size_y_m,
                                start[:3], fixed_tilt=(truth.rot1_rad, truth.rot2_rad))
    assert fixed[:2] == pytest.approx((cx, cy), abs=0.2)
    assert fixed[2] == pytest.approx(truth.distance_m * 1e3, rel=1e-3)


def test_the_residual_of_an_untilted_detector_is_the_circle_fit() -> None:
    """Without a tilt the residual is the radial pixel distance of the circle fit (``optimizer._ring_residual``)."""
    from src.gimap.features.calibration.domain.optimizer import _ring_residual

    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(0, 400, 200), rng.uniform(0, 400, 200)
    qs = np.full(200, 0.3)
    flat = _ring_residual(xs, ys, qs, 1.0, 172e-6, 172e-6)(np.array([200.0, 210.0, 900.0]))
    tilted = tilt_residual(xs, ys, qs, 1.0, 172e-6, 172e-6)(np.array([200.0, 210.0, 900.0, 0.0, 0.0]))
    assert np.allclose(flat, tilted, rtol=0, atol=1e-9)


def test_one_sided_masked_peaks_do_not_supply_tilt_centroids():
    from src.gimap.features.calibration.domain.models import MatchedRing
    from src.gimap.features.calibration.domain.preprocessing import AnalysisImage
    from src.gimap.features.calibration.domain.tilt_fit import tilted_ring_points

    cx, cy, distance_mm, pixel_m, radius = 120.0, 116.0, 150.0, 172e-6, 70.0
    rows, columns = np.indices((250, 250))
    offset = np.hypot(columns - cx, rows - cy) - radius
    angle = np.arctan2(cy - rows, columns - cx)
    counts = 12.0 + 200.0 * np.exp(-0.5 * offset**2)
    valid = ~((np.abs(angle) < 0.3) & (offset > 0.3))
    analysis = AnalysisImage(counts, valid, valid.mean())
    two_theta = np.arctan(radius * pixel_m / (distance_mm * 1e-3))
    q = 4 * np.pi * np.sin(two_theta / 2)  # wavelength 1 Å
    candidate = CalibrationCandidate("agbh", cx, cy, distance_mm,
                                     matched_rings=[MatchedRing(0, q, radius, radius, 0.0)])

    xs, ys, _ = tilted_ring_points(analysis, candidate, (cx, cy, distance_mm), 1.0, pixel_m, pixel_m)
    measured_angles = np.abs(np.arctan2(cy - ys, xs - cx))
    assert xs.size > 100
    assert np.all(measured_angles > 0.25)
    assert np.any(measured_angles < 0.4)


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
    few = CalibrationCandidate("agbh", 10.0, 12.0, 900.0, matched_ring_count=1)
    from src.gimap.features.calibration.domain.tilt_fit import NOT_REFINED_FEW_RINGS, refine_ring_geometry

    assert refine_ring_geometry(None, few, 1.0, 172e-6, 172e-6, tilt="auto") is None  # type: ignore[arg-type]
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


def test_rms_before_and_after_use_the_points_of_the_last_pass(monkeypatch):
    from types import SimpleNamespace
    from src.gimap.features.calibration.domain import tilt_fit

    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    qs = np.repeat([0.11, 0.22, 0.33], angles.size)
    radii = np.tan(2 * np.arcsin(qs / (4 * np.pi))) / 172e-6
    xs = 150 + radii * np.tile(np.cos(angles), 3)
    ys = 110 + radii * np.tile(np.sin(angles), 3)
    initial = (155, 110, 1000, 0, 0)
    candidate = CalibrationCandidate("agbh", *initial[:3], matched_ring_count=3)

    def sample(*args, widen):
        # The wider first-pass band samples different centroids, just as it can on an image.
        return xs + (2 if widen > 1 else 0), ys, qs

    monkeypatch.setattr(tilt_fit, "tilted_ring_points", sample)
    points = tilt_fit.refine_ring_geometry(SimpleNamespace(signal=np.empty((1, 1))), candidate,
                                           1, 172e-6, 172e-6, tilt="auto")
    assert np.array_equal(points[0], xs)
    assert candidate.initial_point_rms_px == pytest.approx(tilt_fit.point_rms_px(*points, initial, 1, 172e-6, 172e-6))
    assert candidate.point_rms_px < 1e-7 and candidate.untilted_point_rms_px < 1e-7


@pytest.mark.parametrize("mode,refined", [("auto", False), ("on", True)])
def test_only_rings_with_points_in_the_final_pass_constrain_tilt(monkeypatch, mode, refined):
    from types import SimpleNamespace
    from src.gimap.features.calibration.domain import tilt_fit
    from src.gimap.features.calibration.domain.optimizer import estimate_uncertainty

    angles = np.tile(np.linspace(0, 2 * np.pi, 90, endpoint=False), 3)
    qs = np.repeat([0.11, 0.22, 0.33], 90)
    radii = np.tan(2 * np.arcsin(qs / (4 * np.pi))) / 172e-6
    xs, ys = 150 + radii * np.cos(angles), 110 + radii * np.sin(angles)
    # The third band has usable points in the broad initial extraction only.
    def sample(*args, widen):
        keep = (qs < 0.33) | (widen > 1.0)
        return xs[keep], ys[keep], qs[keep]

    monkeypatch.setattr(tilt_fit, "tilted_ring_points", sample)
    candidate = CalibrationCandidate("agbh", 155, 110, 1000, matched_ring_count=3, tilt_refined=True,
                                     rot1_deg=0.5, rot2_deg=1.0, rot1_sigma_deg=3.0, rot2_sigma_deg=4.0,
                                     untilted_point_rms_px=7, warnings=[tilt_fit.NOT_REFINED_FEW_RINGS])
    points = tilt_fit.refine_ring_geometry(SimpleNamespace(signal=np.empty((1, 1))), candidate,
                                           1, 172e-6, 172e-6, tilt=mode)
    assert np.unique(points[2]).size == 2
    assert candidate.tilt_refined is refined
    assert candidate.rot1_sigma_deg is None and candidate.rot2_sigma_deg is None
    assert (candidate.untilted_point_rms_px is not None) is refined
    assert (tilt_fit.NOT_REFINED_CONTRIBUTING_RINGS in candidate.warnings) is (not refined)
    assert tilt_fit.NOT_REFINED_FEW_RINGS not in candidate.warnings
    estimate_uncertainty(points, candidate, 1, 172e-6, 172e-6)
    assert candidate.distance_sigma_mm is not None and candidate.uncertainty_method == "least squares"
    assert (candidate.rot1_sigma_deg is not None and candidate.rot2_sigma_deg is not None) is refined
    if not refined:
        assert (candidate.rot1_deg, candidate.rot2_deg) == (0, 0)


@pytest.mark.skipif(not AGBH_FRAME.exists(), reason="public test data")
def test_the_public_agbh_frame_is_fitted_better_with_the_tilt() -> None:
    from src.gimap.features.calibration.application import LoadCalibrationImage
    from src.gimap.features.calibration.infrastructure.adapters import LocalCalibrationImageAdapter

    image = LoadCalibrationImage(LocalCalibrationImageAdapter(ask_dataset=False))(AGBH_FRAME)
    options = dict(energy_kev=HC_KEV_ANGSTROM, standard_key="agbh", pixel_size_x_m=172e-6, pixel_size_y_m=172e-6)
    flat_result = CalibrationEngine().calibrate(image, **options, tilt="off")
    flat = flat_result.selected_candidate
    result = CalibrationEngine().calibrate(image, **options)
    best = result.selected_candidate
    assert not flat.tilt_refined and (flat.rot1_deg, flat.rot2_deg) == (0, 0)
    assert np.hypot(flat.center_x_px - best.center_x_px, flat.center_y_px - best.center_y_px) < 0.5
    assert flat.point_rms_px < flat.initial_point_rms_px and flat.point_rms_px < 1.0
    assert flat.uncertainty_method == best.uncertainty_method == "ring jackknife"
    assert 0 < flat.center_x_sigma_px < 0.5 and 0 < flat.distance_sigma_mm < 1
    from src.gimap.features.calibration.application.headless import describe_calibration

    record = flat_result.to_dict()
    # A version 1 reader constructs candidates with strict keyword arguments and ignores new metadata.
    for legacy_fields in (record["selected_candidate"], *record["candidates"]):
        assert {"initial_point_rms_px", "point_rms_px", "rot1_deg"}.isdisjoint(legacy_fields)
        CalibrationCandidate(**legacy_fields)
    restored = CalibrationResult.from_dict(record).selected_candidate
    assert (restored.initial_point_rms_px, restored.point_rms_px) == (flat.initial_point_rms_px, flat.point_rms_px)
    report = describe_calibration(flat_result)
    assert (report["initial_point_rms_px"], report["point_rms_px"]) == (flat.initial_point_rms_px, flat.point_rms_px)
    assert best.tilt_refined and best.matched_ring_count == 5
    assert [ring.theoretical_index + 1 for ring in best.matched_rings] == [1, 2, 3, 4, 5]
    assert best.point_rms_px < best.untilted_point_rms_px < 1.0
    assert best.point_rms_px < best.initial_point_rms_px
    benchmark = Path(__file__).resolve().parents[1] / "tools/benchmarks/calibration_check.json"
    pyfai = json.loads(benchmark.read_text(encoding="utf-8"))["real_frame"]["pyfai"]["values_gimap_convention"]
    geometry = calibrated_geometry(result)
    beam_error_px = math.hypot(geometry.beam_center_x_px - pyfai["beam_x_px"],
                               geometry.beam_center_y_px - pyfai["beam_y_px"])
    assert beam_error_px < 0.5
    assert best.distance_mm == pytest.approx(pyfai["distance_mm"], abs=1.0)
    assert best.rot1_deg == pytest.approx(pyfai["rot1_deg"], abs=3 * best.rot1_sigma_deg)
    assert best.rot2_deg == pytest.approx(pyfai["rot2_deg"], abs=3 * best.rot2_sigma_deg)


def test_the_window_shows_one_rms_residual_and_says_which_it_is() -> None:
    """A fitted geometry gives before and after on its ring points. Old files may only have ring radii."""
    from src.gimap.features.calibration.domain.models import CalibrationCandidate
    from src.gimap.features.calibration.presentation.residual_text import rms_text, rms_tip, rms_title

    tilted = CalibrationCandidate("agbh", 179.3, 263.5, 1634.0, rms_residual_px=1.056, tilt_refined=True,
                                  point_rms_px=0.45, untilted_point_rms_px=0.46, initial_point_rms_px=1.80)
    assert (rms_title(tilted), rms_text(tilted), rms_text(tilted, 2)) == ("Ring-point RMS:", "0.450 px", "0.45 px")
    assert "Before refinement" in rms_tip(tilted) and "1.800 px" in rms_tip(tilted)
    assert "same points with zero tilt gives 0.460 px" in rms_tip(tilted)
    current = CalibrationCandidate("agbh", 179.4, 263.6, 1634.5, point_rms_px=0.47, initial_point_rms_px=1.84)
    assert (rms_title(current), rms_text(current)) == ("Ring-point RMS:", "0.470 px")
    assert "1.840 px" in rms_tip(current) and "zero tilt" not in rms_tip(current)
    flat = CalibrationCandidate("agbh", 179.3, 263.5, 1634.0, rms_residual_px=1.056)
    assert (rms_title(flat), rms_text(flat)) == ("Ring-radius RMS:", "1.056 px")
    assert "azimuthal average" in rms_tip(flat)
