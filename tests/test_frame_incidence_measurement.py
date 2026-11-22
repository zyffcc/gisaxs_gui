"""Absolute incidence from independent forward simulations of calibrated laboratory rays."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.domain.frame_incidence import measure_incidence
from src.gimap.shared.geometry.detector_geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import lab_rays


def test_unresolved_automatic_roll_compares_incidence_on_the_restored_profile_axes():
    from src.gimap.features.analyze.application.sample_roll import measure_frame_angles

    data = np.zeros((64, 64))
    geometry = DetectorGeometry(.0001, .0001, .3, 32., 52., 1., .2, sample_roll_deg=3.)
    frame = SimpleNamespace(data=data, valid=np.ones_like(data, dtype=bool), geometry=geometry,
                            kind="gisaxs", corrections=None)
    roll, incidence = measure_frame_angles(frame, None, fallback_roll_deg=0.)
    assert not roll.significant and incidence.roll_deg == 0.


def _frame(*, horizon=None, spot=None, band=None, roll=2.2, orientation=3, seed=369):
    g = DetectorGeometry(.00018, .00020, .55, 203.6, 272.2, 1.05, 1.89, .035, -.016, .008,
                         orientation=orientation)
    row, col = np.ogrid[:390, :430]
    x = (col+.5-g.beam_center_x_px)*g.pixel_size_x_m
    y = (g.beam_center_y_px-row-.5)*g.pixel_size_y_m
    along, right, up = lab_rays(g, x, y)
    psi = np.radians(roll)
    z = right*np.sin(psi)+up*np.cos(psi)
    across = right*np.cos(psi)-up*np.sin(psi)
    radius = np.sqrt(along**2+right**2+up**2)
    light = np.full((390, 430), 6.)
    if horizon is not None:
        inc = np.radians(horizon)
        signed = (z*np.cos(inc)-along*np.sin(inc))/radius
        light += 55./(1.+np.exp(-np.clip(signed/.00038, -80, 80)))
    if spot is not None:
        angle = np.arctan2(z, along)
        light += 1000*np.exp(-.5*((angle-2*np.radians(spot))/.00076)**2-.5*(across/radius/.00061)**2)
    if band is not None:
        incidence, critical = np.radians(band)
        sin_exit = (z*np.cos(incidence)-along*np.sin(incidence))/radius
        light += 350*np.exp(-.5*((sin_exit-np.sin(critical))/.00069)**2)
    data = np.random.default_rng(seed).poisson(light)
    valid = np.ones_like(data, dtype=bool)
    valid[:, 71:75] = False
    valid[340:344] = False
    data[~valid] = 1_000_000
    return data, valid, g


@pytest.mark.parametrize("inc,orientation", [(.31, 4), (.52, 3), (.45, 2)])
def test_horizon_and_spot_recover_absolute_incidence(inc, orientation):
    data, valid, g = _frame(horizon=inc, spot=inc, orientation=orientation)
    measure = lambda geometry: measure_incidence(data, geometry, valid=valid, roll_deg=2.2)
    result = measure(g)
    assert result == measure(g.with_incidence(.04))
    assert result.method == "combined" and result.used_estimators == ("horizon", "specular")
    for estimate in (result.horizon, result.specular, result):
        assert abs(estimate.deg-inc) <= estimate.error_deg


@pytest.mark.parametrize("feature", ["horizon", "specular"])
def test_individual_feature_survives_without_the_other(feature):
    data, valid, g = _frame(**{"spot" if feature == "specular" else "horizon": .43})
    result = measure_incidence(data, g, valid=valid, roll_deg=2.2)
    assert result.method == feature
    assert abs(result.deg-.43) <= result.error_deg


def test_incompatible_horizon_and_spot_are_reported_without_a_selected_value():
    data, valid, g = _frame(horizon=.27, spot=.81)
    result = measure_incidence(data, g, valid=valid, roll_deg=2.2)
    assert result.method == "disagreement"
    assert result.horizon and result.specular and result.deg is None


def test_negative_horizon_and_positive_specular_signal_a_geometry_disagreement():
    data, valid, g = _frame(horizon=-.28, spot=.43)
    result = measure_incidence(data, g, valid=valid, roll_deg=2.2)
    assert result.horizon.deg < -3*result.horizon.error_deg
    assert result.specular.deg > 0.
    assert result.method == "disagreement" and result.deg is None


@pytest.mark.parametrize("masked", [False, True])
def test_noise_or_missing_pixels_do_not_supply_an_incidence(masked):
    data, valid, g = _frame()
    if masked:
        valid[:] = False
    result = measure_incidence(data, g, valid=valid, roll_deg=0.)
    assert result.deg is None and result.method == "unavailable"


def test_beamstop_single_hot_pixel_and_off_axis_spot_cannot_supply_an_angle():
    data, valid, g = _frame(roll=0.)
    row, col = np.ogrid[:data.shape[0], :data.shape[1]]
    beamstop = (col-g.beam_center_x_px)**2+(row-g.beam_center_y_px+13)**2 < 13**2
    data[beamstop] = 0
    data[208, 204] = 100000
    data += (5000*np.exp(-.5*((row-181)/2)**2-.5*((col-221)/1.8)**2)).astype(int)
    result = measure_incidence(data, g, valid=valid, roll_deg=0.)
    assert result.specular is None and result.deg is None


def test_textured_bragg_arc_crossing_the_specular_axis_is_not_a_spot():
    from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, WAVELENGTH, giwaxs_frame

    data = giwaxs_frame()
    geometry = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, *CENTER, WAVELENGTH, .2)
    result = measure_incidence(data, geometry, valid=np.ones(data.shape, dtype=bool))
    assert result.specular is None and result.deg is None


def test_diffuse_band_requires_known_critical_angle_and_is_not_counted_twice():
    data, valid, g = _frame(band=(.38, .21), roll=0.)
    kwargs = dict(valid=valid, technique="gisaxs")
    unknown = measure_incidence(data, g, **kwargs)
    assert unknown.yoneda is None and unknown.horizon is None and unknown.deg is None
    known = measure_incidence(data, g, critical_angle_deg=.21, **kwargs)
    assert known.method == "yoneda"
    assert abs(known.deg-.38) <= known.error_deg
    assert known.yoneda.details["critical_angle_deg"] == .21


@pytest.mark.parametrize("name", ["gisaxs_galaxi", "giwaxs_xeuss_pbte"])
def test_public_incidence_vs_published_angle(name):
    from tools.benchmarks.sample_roll import DATA, measure_public

    if not (DATA / name).is_dir():
        pytest.skip("Public frame unavailable.")
    record = measure_public(name)
    result = record["incidence"]
    if name == "gisaxs_galaxi":
        assert result["used_estimators"] == ("specular", "yoneda")
        assert abs(result["deg"]-record["published_incidence_deg"]) <= result["error_deg"]
        assert result["horizon"] is None
    else:
        assert result["deg"] is None and result["specular"] is None
        edge = result["horizon"]
        assert abs(edge["deg"]-record["published_incidence_deg"]) <= 3*edge["error_deg"]
