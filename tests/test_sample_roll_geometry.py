"""The measured sample frame, its inverse, and compatibility with the upright reduction."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application import detector_point, q_at_position
from src.gimap.features.analyze.application.indexing import detector_positions
from src.gimap.features.analyze.application.export_record import recorded_region
from src.gimap.features.analyze.application.symmetry_center import refine_center_by_symmetry
from src.gimap.features.analyze.domain.gisaxs import gisaxs_maps, reduce_gisaxs
from src.gimap.features.analyze.domain.models import GisaxsCutSettings
from src.gimap.features.analyze.domain.sources import curve_source_mask
from src.gimap.features.analyze.domain.transmission import transmission_maps
from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import positions_of_q, q_of_positions
from src.gimap.shared.geometry.q_mapping import ExitAngleModel, exit_directions, geometry_rays, grazing_q_map, transmission_map


FLAT = DetectorGeometry(1.3e-4, 1.1e-4, 0.16, 85.75, 107.3, 1.2, 0.55)
SHAPE = (121, 177)


@pytest.mark.parametrize("model", list(ExitAngleModel))
def test_each_exit_angle_model_uses_the_rolled_transverse_ray(model):
    psi = math.radians(3.)
    x, y = np.array([-.02, .03]), np.array([.01, .04])
    across = x * math.cos(psi) - y * math.sin(psi)
    up = x * math.sin(psi) + y * math.cos(psi)
    expected = exit_directions(across, up, FLAT.distance_m, FLAT.incidence_rad, model)
    rolled = exit_directions(x, y, FLAT.distance_m, FLAT.incidence_rad, model, sample_roll_rad=psi)
    np.testing.assert_allclose(rolled, expected, rtol=2e-15)


@pytest.mark.parametrize("geometry", [FLAT, replace(FLAT, orientation=2, rot1_rad=0.12, rot3_rad=-0.18)])
@pytest.mark.parametrize("critical", [0.0, 0.2])
def test_zero_roll_reproduces_previous_exact_arithmetic_bit_for_bit(geometry, critical):
    geometry = geometry.with_critical_angle(critical)
    along, x, y = geometry_rays(geometry, (0, SHAPE[0]), (0, SHAPE[1]))
    radius = np.sqrt(along**2 + x**2 + y**2)
    ci, si = np.cos(geometry.incidence_rad), np.sin(geometry.incidence_rad)
    k = geometry.wavevector_inv_angstrom
    normal = (y * ci - along * si) / radius
    qx, qy = k * ((along * ci + y * si) / radius - ci), k * (x / radius)
    if critical:
        sc2 = math.sin(math.radians(critical))**2
        film = np.where(normal > 0, np.sqrt(np.maximum(normal * normal - sc2, 0)), 0)
        qz = k * (film + math.sqrt(max(math.sin(geometry.incidence_rad)**2 - sc2, 0)))
    else:
        qz = k * (normal + si)
    qpar = np.copysign(np.sqrt(qx**2 + qy**2), qy)
    for explicit in (geometry, geometry.with_sample_roll(0.0)):
        q = grazing_q_map(SHAPE, explicit)
        for actual, previous in zip((q.qx, q.qy, q.qz, q.q_parallel, q.sin_exit), (qx, qy, qz, qpar, normal)):
            assert actual.tobytes() == previous.tobytes()


@pytest.mark.parametrize("psi", [-8.0, 2.3])
@pytest.mark.parametrize("orientation", [1, 3, 4])
def test_rolled_q_matches_sputter_basis_and_overlay_inverse(psi, orientation):
    geometry = replace(FLAT, rot1_rad=0.07, rot2_rad=-0.04, rot3_rad=0.10,
                       sample_roll_deg=psi, orientation=orientation)
    i, j = 53, 96
    ray = np.array(geometry_rays(geometry, (i, i+1), (j, j+1))).ravel()
    unit = ray / np.linalg.norm(ray)
    a, p = np.radians([geometry.incidence_deg, psi])
    normal = np.array([-np.sin(a), np.cos(a)*np.sin(p), np.cos(a)*np.cos(p)])
    beam = np.array([np.cos(a), np.sin(a)*np.sin(p), np.sin(a)*np.cos(p)])
    transverse = np.cross(normal, beam)
    vector = geometry.wavevector_inv_angstrom * (unit - np.array([1, 0, 0]))
    x, y = j + 0.5, i + 0.5
    point = q_at_position(geometry, x, y)
    assert [point[name] for name in ("qx", "qy", "qz")] == pytest.approx([vector @ axis for axis in (beam, transverse, normal)])
    assert geometry.exit_angle_at(x, y) == pytest.approx(math.asin(unit @ normal))
    assert positions_of_q(geometry, point["qy"], point["qz"]) == pytest.approx((x, y))
    assert detector_positions(geometry, point["q_parallel"], point["qz"]) == pytest.approx((x, y))
    readout = detector_point(geometry, x, y, "giwaxs")
    assert readout["chi_deg"] == pytest.approx(math.degrees(math.atan2(point["q_parallel"], point["qz"])))
    # Laboratory transmission azimuth and 2θ do not involve the sample.
    upright = transmission_map(SHAPE, geometry.with_sample_roll(0))
    rolled = transmission_map(SHAPE, geometry)
    assert rolled.q.tobytes() == upright.q.tobytes()
    assert rolled.chi_deg.tobytes() == upright.chi_deg.tobytes()
    reduction = transmission_maps(SHAPE, geometry)
    assert reduction.chi_deg[i, j] == pytest.approx(upright.chi_deg[i, j])


def test_rolled_yoneda_and_sources_follow_constant_sample_q():
    g = FLAT.with_sample_roll(7.0)
    maps = gisaxs_maps(SHAPE, g)
    # Increasing canonical row to the right is the Sputter positive sign.
    assert g.horizon_row(150) > g.horizon_row(15)
    for col in [0.5, 50.5, 175.5]:
        assert abs(g.exit_angle_at(col, g.horizon_row(col))) < 1e-13
    band_row = 74.5
    centre = q_of_positions(g, g.beam_center_x_px, band_row)[1]
    image = 1 + 60 * np.exp(-0.5 * ((maps.qz - centre) / 0.003)**2)
    valid = np.ones(SHAPE, dtype=bool)
    result = reduce_gisaxs(image, valid, g, maps=maps)
    assert result.markers["yoneda"].row == pytest.approx(band_row, abs=1)
    for key, axis in (("horizontal", "qz"), ("vertical", "qy")):
        curve = result.curve(key)
        lower, upper = curve.region[axis + "_edges"]
        field = getattr(maps, axis)
        expected = maps.above_horizon & (field >= lower) & (field <= upper)
        assert np.array_equal(curve_source_mask(curve, valid, maps), expected)
        assert expected.sum() == curve.pixels.sum() > 0
        if key == "horizontal":
            assert curve.intensity.min() > 15
            assert np.where(expected[:, -1])[0].mean() > np.where(expected[:, 0])[0].mean() + 15


def test_rolled_refraction_uses_the_surface_exit_angle_and_preserves_lateral_q():
    geometry = FLAT.with_sample_roll(-5.0)
    film = geometry.with_critical_angle(0.18)
    x, y = 121.5, 48.5
    vacuum_y, _ = q_of_positions(geometry, x, y)
    qy, qz = q_of_positions(film, x, y)
    si, sf = math.sin(film.incidence_rad), math.sin(film.exit_angle_at(x, y))
    sc = math.sin(film.critical_rad)
    assert qy == vacuum_y
    assert qz == pytest.approx(film.wavevector_inv_angstrom * (math.sqrt(si**2 - sc**2) + math.sqrt(sf**2 - sc**2)))
    assert positions_of_q(film, qy, qz) == pytest.approx((x, y))


def test_rolled_cut_refines_the_direct_beam_and_records_the_sample_band():
    shape = (145, 443)
    g = replace(FLAT, distance_m=0.38, beam_center_x_px=211.8, beam_center_y_px=127.5, sample_roll_deg=6)
    maps = gisaxs_maps(shape, g)
    image = 12 + 3000 / (1 + (maps.qy / 0.009)**2)**2
    valid = np.ones(shape, bool)
    wrong = g.with_beam_center(g.beam_center_x_px + 3, g.beam_center_y_px)
    reduction = reduce_gisaxs(image, valid, wrong, GisaxsCutSettings(horizontal_row=71.5))
    frame = SimpleNamespace(data=image, valid=valid, shape=shape, reduction=reduction, geometry=wrong, kind="gisaxs")
    found = refine_center_by_symmetry(frame)
    assert found.x_px == pytest.approx(g.beam_center_x_px, abs=0.35)
    region = recorded_region(frame, reduction.curve("horizontal"))
    assert region["qz_edges"] == region["qz"] and region["pixel_rows"] != region["rows"]
    assert "detector_corners_qy_qz_A_inv" not in region
