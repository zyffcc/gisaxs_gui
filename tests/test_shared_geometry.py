"""Physics and convention tests for the canonical ``shared.geometry`` mapping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gimap.shared.geometry import (
    DetectorGeometry,
    ExitAngleModel,
    exit_directions,
    grazing_q_map,
    pixel_center_displacements,
    scattering_vectors,
    transmission_map,
)


def _geometry(**overrides) -> DetectorGeometry:
    values = dict(
        pixel_size_x_m=172e-6,
        pixel_size_y_m=172e-6,
        distance_m=4.2,
        beam_center_x_px=737.5,
        beam_center_y_px=1400.5,
        wavelength_angstrom=1.0332,
        incidence_deg=0.4,
    )
    values.update(overrides)
    return DetectorGeometry(**values)


def test_beam_center_is_expressed_in_pixel_corner_coordinates() -> None:
    geometry = _geometry(beam_center_x_px=3.5, beam_center_y_px=2.5)
    x, y = pixel_center_displacements((5, 7), geometry)

    assert x[2, 3] == 0.0 and y[2, 3] == 0.0  # the centre of pixel (row 2, column 3)
    assert x[2, 4] == pytest.approx(172e-6)  # one pixel to the right
    assert y[1, 3] == pytest.approx(172e-6)  # one row up is positive y


def test_every_pixel_is_elastic() -> None:
    geometry = _geometry()
    q = grazing_q_map((64, 48), geometry.with_beam_center(20.25, 40.75))
    k = geometry.wavevector_inv_angstrom
    alpha = geometry.incidence_rad
    k_out = np.sqrt((q.qx + k * math.cos(alpha)) ** 2 + q.qy**2 + (q.qz - k * math.sin(alpha)) ** 2)
    np.testing.assert_allclose(k_out, k, rtol=0, atol=1e-13)


def test_direct_beam_specular_and_horizon_land_where_physics_says() -> None:
    geometry = _geometry()
    k = geometry.wavevector_inv_angstrom
    alpha = geometry.incidence_rad
    d = geometry.distance_m
    x = np.zeros(3)
    y = np.array([0.0, d * math.tan(alpha), d * math.tan(2 * alpha)])
    q = scattering_vectors(x, y, d, alpha, k)

    # Direct beam: q = 0.  Horizon: k_f is parallel to the surface, so
    # q = k(1 − cos αi, 0, sin αi).  Specular: q is purely along the normal.
    np.testing.assert_allclose(q.qx, [0.0, k * (1 - math.cos(alpha)), 0.0], atol=1e-15)
    np.testing.assert_allclose(q.qy, 0.0, atol=1e-15)
    np.testing.assert_allclose(q.qz, [0.0, k * math.sin(alpha), 2 * k * math.sin(alpha)], atol=1e-15)


def test_characteristic_rows_follow_the_exact_model() -> None:
    geometry = _geometry(beam_center_x_px=10.5)
    q = grazing_q_map((1600, 21), geometry)
    k, alpha = geometry.wavevector_inv_angstrom, geometry.incidence_rad

    specular = geometry.specular_row()
    horizon = geometry.horizon_row()
    assert specular < horizon < geometry.beam_center_y_px  # above the beam, rows grow downward

    # Linear interpolation of qz along the centre column at the continuous rows.
    rows = np.arange(q.qz.shape[0]) + 0.5
    qz_centre = q.qz[:, 10]
    assert np.interp(horizon, rows, qz_centre) == pytest.approx(k * math.sin(alpha), rel=1e-6)
    assert np.interp(specular, rows, qz_centre) == pytest.approx(2 * k * math.sin(alpha), rel=1e-6)


def test_exact_model_reduces_to_transmission_rings_without_tilt() -> None:
    geometry = _geometry(incidence_deg=0.0, beam_center_x_px=31.2, beam_center_y_px=17.9)
    grazing = grazing_q_map((40, 60), geometry)
    rings = transmission_map((40, 60), geometry)

    np.testing.assert_allclose(grazing.q, rings.q, rtol=1e-12, atol=1e-15)
    two_theta = np.radians(rings.two_theta_deg)
    np.testing.assert_allclose(rings.q, 4 * math.pi / geometry.wavelength_angstrom * np.sin(two_theta / 2))


def test_transmission_azimuth_is_zero_right_and_ninety_up() -> None:
    geometry = _geometry(beam_center_x_px=5.5, beam_center_y_px=5.5)
    chi = transmission_map((11, 11), geometry).chi_deg
    assert chi[5, 8] == pytest.approx(0.0)
    assert chi[2, 5] == pytest.approx(90.0)
    assert chi[5, 2] == pytest.approx(180.0)
    assert chi[8, 5] == pytest.approx(-90.0)


def test_approximate_models_agree_with_exact_on_the_plane_of_incidence() -> None:
    d, alpha = 4200.0, math.radians(0.4)
    y = np.linspace(-50.0, 300.0, 41)
    x = np.zeros_like(y)
    exact = np.stack(exit_directions(x, y, d, alpha, ExitAngleModel.EXACT))
    subtract = np.stack(exit_directions(x, y, d, alpha, ExitAngleModel.SUBTRACT_INCIDENCE))
    np.testing.assert_allclose(subtract, exact, rtol=0, atol=1e-15)


def test_legacy_models_deviate_from_exact_by_documented_amounts() -> None:
    """Quantifies the approximations the older pages use on a Pilatus 2M at 4.2 m."""
    geometry = _geometry()
    shape = (1679, 1475)
    exact = grazing_q_map(shape, geometry)
    k = geometry.wavevector_inv_angstrom
    deviations = {}
    for model in (ExitAngleModel.HORIZON_SHIFT, ExitAngleModel.SUBTRACT_INCIDENCE):
        other = grazing_q_map(shape, geometry, model)
        deviations[model] = float(np.max(np.abs(other.qz - exact.qz))) / k
    # Both flat-detector approximations stay below 1e-4·k (≈ 6e-4 Å⁻¹ here)
    # across the whole detector; see docs/architecture/geometry.md.
    assert all(value < 1e-4 for value in deviations.values())
    # Ignoring alpha_i in the exit angle (WAXS legacy) shifts qz by ~k·sin(alpha_i).
    waxs = grazing_q_map(shape, geometry, ExitAngleModel.NO_INCIDENCE_OFFSET)
    shift = float(np.median(waxs.qz - exact.qz)) / k
    assert shift == pytest.approx(math.sin(geometry.incidence_rad), rel=0.05)


def test_cropping_keeps_the_physical_pixels_q() -> None:
    geometry = _geometry(beam_center_x_px=40.3, beam_center_y_px=70.8)
    full = grazing_q_map((100, 80), geometry)
    crop = grazing_q_map((30, 25), geometry.cropped(left_px=12, top_px=33))
    np.testing.assert_array_equal(crop.qz, full.qz[33:63, 12:37])
    np.testing.assert_array_equal(crop.q_parallel, full.q_parallel[33:63, 12:37])


@pytest.mark.parametrize(
    "field,value",
    [
        ("pixel_size_x_m", 0.0),
        ("distance_m", -1.0),
        ("wavelength_angstrom", float("nan")),
        ("beam_center_x_px", float("inf")),
        ("incidence_deg", 90.0),
    ],
)
def test_invalid_geometry_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        _geometry(**{field: value})
