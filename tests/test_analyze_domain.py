"""Analyze reductions on synthetic frames built from the exact grazing-incidence model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gimap.features.analyze.domain import (
    GISAXS,
    GIWAXS,
    BinnedMean,
    GisaxsCutSettings,
    GiwaxsSettings,
    binned_mean_2d,
    classify_measurement,
    giwaxs_maps,
    locate_yoneda,
    reduce_gisaxs,
    reduce_giwaxs,
    valid_pixels,
)
from src.gimap.shared.geometry import (
    DetectorGeometry,
    grazing_q_map,
    grazing_q_region,
    pixel_center_displacements,
    region_displacements,
)


def _gisaxs_geometry(**changes) -> DetectorGeometry:
    values = dict(
        pixel_size_x_m=172e-6,
        pixel_size_y_m=172e-6,
        distance_m=4.2,
        beam_center_x_px=240.5,
        beam_center_y_px=420.5,
        wavelength_angstrom=1.0332,
        incidence_deg=0.4,
    )
    values.update(changes)
    return DetectorGeometry(**values)


def _giwaxs_geometry() -> DetectorGeometry:
    return DetectorGeometry(
        pixel_size_x_m=172e-6,
        pixel_size_y_m=172e-6,
        distance_m=0.15,
        beam_center_x_px=250.5,
        beam_center_y_px=470.5,
        wavelength_angstrom=1.0332,
        incidence_deg=0.15,
    )


def test_binned_mean_reports_mean_poisson_error_and_pixel_count() -> None:
    accumulator = BinnedMean(np.array([0.0, 1.0, 2.0, 3.0]))
    accumulator.add(np.array([0.2, 0.8, 1.5, 3.0, 7.0, np.nan]), np.array([4, 16, 9, 1, 5, 5]))
    centres, mean, sigma, pixels = accumulator.result()

    np.testing.assert_allclose(centres, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(mean, [10.0, 9.0, 1.0])  # 3.0 lands in the last bin
    np.testing.assert_allclose(sigma, [math.sqrt(20) / 2, 3.0, 1.0])
    np.testing.assert_array_equal(pixels, [2, 1, 1])


def test_binned_mean_2d_places_low_y_in_row_zero_and_leaves_empty_cells_nan() -> None:
    grid = binned_mean_2d(
        np.array([0.1, 0.9, 0.9]),
        np.array([0.1, 0.9, 0.9]),
        np.array([1.0, 2.0, 4.0]),
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        shape=(2, 2),
    )
    assert grid[0, 0] == 1.0 and grid[1, 1] == 3.0
    assert np.isnan(grid[0, 1]) and np.isnan(grid[1, 0])


def test_region_helpers_equal_slices_of_the_full_frame() -> None:
    geometry = _gisaxs_geometry()
    shape = (60, 80)
    x_full, y_full = pixel_center_displacements(shape, geometry)
    x_part, y_part = region_displacements(geometry, (10, 25), (30, 71))
    np.testing.assert_array_equal(x_part, x_full[10:25, 30:71])
    np.testing.assert_array_equal(y_part, y_full[10:25, 30:71])

    full = grazing_q_map(shape, geometry)
    part = grazing_q_region(geometry, (10, 25), (30, 71))
    for name in ("qx", "qy", "qz", "q_parallel"):
        np.testing.assert_array_equal(getattr(part, name), getattr(full, name)[10:25, 30:71])


def test_valid_pixels_rejects_negative_nan_and_masked() -> None:
    data = np.array([[1.0, -1.0], [np.nan, 5.0]])
    mask = np.array([[False, False], [False, True]])
    np.testing.assert_array_equal(valid_pixels(data, mask), [[True, False], [False, False]])


def test_classification_uses_the_largest_scattering_angle() -> None:
    assert classify_measurement((500, 480), _gisaxs_geometry()) == GISAXS
    assert classify_measurement((500, 500), _giwaxs_geometry()) == GIWAXS
    # A short GISAXS set-up with a large detector (P03, Pilatus 2M at 1.46 m) reaches
    # about 10° at the far corner and is still GISAXS.
    close_gisaxs = DetectorGeometry(172e-6, 172e-6, 1.4567, 797.98, 1307.75, 1.033, 0.4)
    assert classify_measurement((1679, 1475), close_gisaxs) == GISAXS


def _synthetic_gisaxs(geometry: DetectorGeometry, shape=(500, 480), *, yoneda_deg=0.25, rod_qy=0.02):
    q = grazing_q_map(shape, geometry)
    k = geometry.wavevector_inv_angstrom
    alpha_f = np.degrees(np.arcsin(np.clip(q.qz / k - math.sin(geometry.incidence_rad), -1, 1)))
    yoneda = 50.0 * np.exp(-0.5 * ((alpha_f - yoneda_deg) / 0.02) ** 2) * (alpha_f > 0)
    rods = 30.0 * np.exp(-0.5 * ((np.abs(q.qy) - rod_qy) / 0.0008) ** 2) * (alpha_f > 0)
    image = 1.0 + yoneda + rods
    # A specular spot far brighter than the Yoneda band, on the beam-centre column.
    specular = np.hypot(q.qy / 0.0005, (alpha_f - geometry.incidence_deg) / 0.01)
    image += 5000.0 * np.exp(-0.5 * specular**2)
    return image.astype(np.float32)


def test_yoneda_is_found_beside_the_specular_rod() -> None:
    geometry = _gisaxs_geometry()
    image = _synthetic_gisaxs(geometry)
    estimate = locate_yoneda(image, valid_pixels(image), geometry)

    expected_row = geometry.row_for_exit_angle(0.25)
    assert estimate is not None
    assert abs(estimate.row - expected_row) <= 1.0
    assert estimate.alpha_f_deg == pytest.approx(0.25, abs=0.01)


def test_gisaxs_reduction_needs_no_input_and_resolves_rods_and_yoneda() -> None:
    geometry = _gisaxs_geometry()
    image = _synthetic_gisaxs(geometry)
    reduction = reduce_gisaxs(image, valid_pixels(image), geometry)

    assert reduction.kind == GISAXS
    assert reduction.markers["horizontal_source"] == "yoneda"
    horizontal = reduction.curve("horizontal")
    vertical = reduction.curve("vertical")
    assert horizontal.x_label == "qy (Å⁻¹)" and vertical.x_label == "qz (Å⁻¹)"
    # Rods at ±0.02 Å⁻¹ are the maxima of I(qy) on each side of the specular column.
    left = horizontal.x < -0.005
    right = horizontal.x > 0.005
    assert horizontal.x[left][np.argmax(horizontal.intensity[left])] == pytest.approx(-0.02, abs=4e-4)
    assert horizontal.x[right][np.argmax(horizontal.intensity[right])] == pytest.approx(0.02, abs=4e-4)
    # I(qz) away from the specular spot peaks at the Yoneda qz = k(sin αf + sin αi).
    k = geometry.wavevector_inv_angstrom
    yoneda_qz = k * (math.sin(math.radians(0.25)) + math.sin(geometry.incidence_rad))
    specular_qz = 2 * k * math.sin(geometry.incidence_rad)
    away = np.abs(vertical.x - specular_qz) > 0.004
    assert vertical.x[away][np.argmax(vertical.intensity[away])] == pytest.approx(yoneda_qz, abs=3e-4)
    assert np.all(np.diff(vertical.x) > 0)
    assert vertical.x.min() >= k * math.sin(geometry.incidence_rad) - 1e-3


def test_manual_cut_positions_override_the_automatic_ones() -> None:
    geometry = _gisaxs_geometry()
    image = _synthetic_gisaxs(geometry)
    settings = GisaxsCutSettings(horizontal_row=100.0, vertical_column=300.0)
    reduction = reduce_gisaxs(image, valid_pixels(image), geometry, settings)

    assert reduction.markers["horizontal_source"] == "manual"
    assert reduction.curve("horizontal").region["rows"] == (97, 103)
    assert reduction.curve("vertical").region["columns"] == (295, 305)


def test_zero_incidence_is_reported_instead_of_silently_mislabelled() -> None:
    geometry = _gisaxs_geometry(incidence_deg=0.0)
    image = _synthetic_gisaxs(_gisaxs_geometry())
    reduction = reduce_gisaxs(image, valid_pixels(image), geometry)
    assert any("Incidence angle is 0" in warning for warning in reduction.warnings)


def _synthetic_giwaxs(geometry: DetectorGeometry, shape=(500, 500)):
    maps = giwaxs_maps(shape, geometry)
    q = maps.q.astype(np.float64)
    chi = maps.chi_deg.astype(np.float64)
    ring = 100.0 * np.exp(-0.5 * ((q - 1.5) / 0.01) ** 2)
    arc = 60.0 * np.exp(-0.5 * ((q - 0.8) / 0.01) ** 2) * (np.abs(chi) < 8.0)
    return (5.0 + ring + arc).astype(np.float32), maps


def test_giwaxs_reduction_finds_ring_sectors_and_azimuthal_profile() -> None:
    geometry = _giwaxs_geometry()
    image, maps = _synthetic_giwaxs(geometry)
    reduction = reduce_giwaxs(image, valid_pixels(image), geometry, maps=maps)

    assert reduction.kind == GIWAXS
    radial = reduction.curve("radial")
    assert radial.x[np.argmax(radial.intensity)] == pytest.approx(1.5, abs=0.01)
    out_of_plane = reduction.curve("out_of_plane")
    in_plane = reduction.curve("in_plane")
    near_arc = np.abs(out_of_plane.x - 0.8) < 0.01
    assert out_of_plane.intensity[near_arc].max() > 40.0
    in_near_arc = np.abs(in_plane.x - 0.8) < 0.01
    assert in_plane.intensity[in_near_arc].max() < 10.0
    azimuthal = reduction.curve("azimuthal")
    low, high = reduction.markers["chi_q_window"]
    assert low < 1.5 < high
    assert azimuthal.x.min() >= -90.0 and azimuthal.x.max() <= 90.0
    # The isotropic ring is flat in χ within the sampled range.
    assert np.ptp(azimuthal.intensity) < 0.25 * azimuthal.intensity.mean()
    rsm = reduction.reciprocal_space_map
    assert rsm.image.shape == (800, 800)
    assert rsm.qz_range[0] >= 0.0


def test_giwaxs_manual_chi_window_is_used_verbatim() -> None:
    geometry = _giwaxs_geometry()
    image, maps = _synthetic_giwaxs(geometry)
    reduction = reduce_giwaxs(
        image, valid_pixels(image), geometry, GiwaxsSettings(chi_q_window=(0.78, 0.82)), maps=maps
    )
    azimuthal = reduction.curve("azimuthal")
    assert azimuthal.region["q_window"] == (0.78, 0.82)
    inside = np.abs(azimuthal.x) < 5.0
    outside = np.abs(azimuthal.x) > 20.0
    assert azimuthal.intensity[inside].mean() > 5 * azimuthal.intensity[outside].mean()


def test_yoneda_search_ignores_a_hot_detector_row() -> None:
    geometry = _gisaxs_geometry()
    image = _synthetic_gisaxs(geometry)
    hot_row = int(geometry.row_for_exit_angle(1.0))
    image[hot_row, ::3] = 1e6  # hot pixels along a module edge
    estimate = locate_yoneda(image, valid_pixels(image), geometry)
    assert abs(estimate.row - geometry.row_for_exit_angle(0.25)) <= 1.0


def test_yoneda_is_found_when_the_geometry_places_the_horizon_too_high() -> None:
    true_geometry = _gisaxs_geometry()
    image = _synthetic_gisaxs(true_geometry, yoneda_deg=0.05)
    # A beam centre 0.1° too high moves the computed horizon above the real Yoneda row.
    shift_px = true_geometry.distance_m * math.tan(math.radians(0.1)) / true_geometry.pixel_size_y_m
    wrong = true_geometry.with_beam_center(
        true_geometry.beam_center_x_px, true_geometry.beam_center_y_px - shift_px
    )
    estimate = locate_yoneda(image, valid_pixels(image), wrong)
    assert abs(estimate.row - true_geometry.row_for_exit_angle(0.05)) <= 1.0


def test_strongest_ring_ignores_single_bin_spikes() -> None:
    from src.gimap.features.analyze.domain import Curve, strongest_ring

    x = np.linspace(0.1, 4.0, 1200)
    y = 100.0 * np.exp(-x) + 40.0 * np.exp(-0.5 * ((x - 1.5) / 0.02) ** 2)
    y[1100] *= 50.0  # a hot pixel lands in one bin
    curve = Curve("radial", "I(q)", x, y, np.ones_like(y), np.ones_like(y, dtype=int), "q")
    low, high = strongest_ring(curve)
    assert low < 1.5 < high
