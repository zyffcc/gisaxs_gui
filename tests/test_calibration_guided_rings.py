import numpy as np

from src.gimap.features.calibration.domain.engine import CalibrationEngine, _rings
from src.gimap.features.calibration.domain.geometry_model import q_to_ring_radius_m
from src.gimap.features.calibration.domain.guided_rings import search_rings
from src.gimap.features.calibration.domain.models import CalibrationCandidate, DetectorImage
from src.gimap.features.calibration.domain.preprocessing import preprocess_detector_image
from src.gimap.features.calibration.domain.standards import STANDARDS
from src.gimap.shared.detector_io.radiation import HC_KEV_ANGSTROM


def test_guided_search_recovers_a_weak_third_arc(tmp_path):
    standard = STANDARDS["agbh"]
    wavelength, pixel_m, distance_mm = 1.0, 100e-6, 387.0
    cx, cy = 224.1, 226.4
    radii = q_to_ring_radius_m(standard.q_values_inv_angstrom[:3], wavelength, distance_mm) / pixel_m
    y, x = np.indices((460, 458))
    radius, angle = np.hypot(x - cx, y - cy), np.arctan2(y - cy, x - cx)
    arc = (angle > -0.15) & (angle < 1.45)
    counts = np.full(radius.shape, 80.0)
    for height, r in zip((600.0, 200.0, 14.0 * arc), radii):
        counts += height * (1 + 0.09 * np.cos(2 * angle)) * np.exp(-0.5 * ((radius - r) / 1.7) ** 2)
    path = tmp_path / "weak_arc.npy"
    np.save(path, counts)
    image = DetectorImage(counts, None, path, beam_center_x_px=cx, beam_center_y_px=cy)
    analysis = preprocess_detector_image(image)

    coarse, _, _ = _rings(analysis, cx, cy)
    np.testing.assert_allclose([peak.radius_px for peak in coarse], radii[:2], atol=1)
    seed = CalibrationCandidate("agbh", cx, cy, distance_mm)
    evidence = search_rings(analysis, seed, standard, wavelength, pixel_m, pixel_m)
    third = next(ring for ring in evidence if ring.theoretical_index == 2)
    assert third.match and 14 <= third.supporting_sectors <= 22
    assert all(line.match is None for line in evidence if line.theoretical_index > 2)

    result = CalibrationEngine().calibrate(
        image, standard_key="agbh", energy_kev=HC_KEV_ANGSTROM / wavelength,
        pixel_size_x_m=pixel_m, pixel_size_y_m=pixel_m,
        estimated_distance_mm=distance_mm, distance_range_mm=(370, 410),
    )
    best = result.selected_candidate
    assert [ring.theoretical_index for ring in best.matched_rings] == [0, 1, 2]
    assert best.tilt_refined and best.distance_sigma_mm is not None
    assert abs(best.distance_mm - distance_mm) < 1
    assert np.hypot(best.center_x_px - cx, best.center_y_px - cy) < 1


def test_a_bright_detector_corner_does_not_supply_an_extra_order(tmp_path):
    standard = STANDARDS["agbh"]
    pixel, wavelength, distance = 100e-6, 1.0, 255.5
    cx, cy = 112.0, 72.0
    radii = q_to_ring_radius_m(standard.q_values_inv_angstrom[:5], wavelength, distance) / pixel
    y, x = np.indices((256, 256))
    radius = np.hypot(x - cx, y - cy)
    counts = np.full(radius.shape, 20.0)
    for r in radii[:3]:
        counts += 280 * np.exp(-0.5 * ((radius - r) / 1.4) ** 2)
    # A short bright feature meets the fifth-order radius at the far corner.
    counts += 120 * np.exp(-0.5 * ((radius - radii[4]) / 1.4) ** 2)
    path = tmp_path / "corner.npy"
    np.save(path, counts)
    image = DetectorImage(counts, None, path, beam_center_x_px=cx, beam_center_y_px=cy)
    analysis = preprocess_detector_image(image)
    peaks, _, _ = _rings(analysis, cx, cy)
    assert any(abs(p.radius_px - radii[4]) < 2 for p in peaks)
    result = CalibrationEngine().calibrate(
        image, standard_key="agbh", energy_kev=HC_KEV_ANGSTROM,
        pixel_size_x_m=pixel, pixel_size_y_m=pixel, estimated_distance_mm=distance,
        distance_range_mm=(250, 275),
    )
    assert [r.theoretical_index + 1 for r in result.selected_candidate.matched_rings] == [1, 2, 3]
