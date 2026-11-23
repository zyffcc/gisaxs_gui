"""Recovery of the absolute sample roll from independently simulated lab-ray intensities."""
from __future__ import annotations

import numpy as np
import pytest

from src.gimap.shared.geometry.detector_geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import lab_rays
from src.gimap.shared.geometry.sample_roll import measure_sample_roll


def _frame(psi, *, technique="giwaxs", edge_angle=None, orientation=3, shadow=True):
    shape = (500, 730)
    geometry = DetectorGeometry(.00018, .00020, .115, 333., 395., 1.08, .32,
                                .014, -.019, .006, orientation=orientation)
    row, col = np.ogrid[:shape[0], :shape[1]]
    along, right, up = lab_rays(geometry, (col+.5-geometry.beam_center_x_px)*geometry.pixel_size_x_m,
                                (geometry.beam_center_y_px-row-.5)*geometry.pixel_size_y_m)
    r = np.sqrt(along**2 + right**2 + up**2)
    ux, uy, uz = along/r, right/r, up/r
    si, ci = np.sin(geometry.incidence_rad), np.cos(geometry.incidence_rad)
    # Dot the lab momentum transfer with a known unit sample normal. This forward
    # simulation does not call the production sample rotation or either estimator.
    def normal(angle):
        return ci*(uy*np.sin(np.radians(angle))+uz*np.cos(np.radians(angle)))-si*ux
    nf = normal(psi)
    q_over_k = np.sqrt((ux-1)**2 + uy**2 + uz**2)
    polar = np.degrees(np.arccos(np.clip((nf+si)/q_over_k, -1., 1.)))
    visibility = 1/(1+np.exp(-np.clip(normal(psi if edge_angle is None else edge_angle)/.0018, -100, 100)))
    light = np.full(shape, 32.)
    if technique == "giwaxs":
        texture = 1 + 3*np.exp(-.5*((polar-40)/5.5)**2) + 2*np.exp(-.5*((polar-69)/4)**2)
        for radius, scale in ((.2, 120.), (.32, 160.), (.39, 60.)):
            light += scale*np.exp(-.5*((q_over_k-radius)/.004)**2)*texture
    else:
        rod = uy*np.cos(np.radians(psi))-uz*np.sin(np.radians(psi))
        light += 200*np.exp(-.5*(rod/.0023)**2)
        if shadow:
            light += 170*np.exp(-.5*((nf-.022)/.003)**2)
    rng = np.random.default_rng(1542)
    pixels = rng.poisson(4. + (visibility if shadow else 1.)*light)
    valid = np.ones(shape, dtype=bool)
    valid[210:215] = False
    valid[:, 145:152] = False
    pixels[~valid] = 2000000
    return pixels, valid, geometry


@pytest.mark.parametrize("angle", [-3.7, 0., 2.3])
def test_fibre_and_shadow_recover_roll_within_reported_error(angle):
    data, valid, geometry = _frame(angle)
    result = measure_sample_roll(data, geometry, valid=valid)
    assert result.method == "combined"
    for estimate in (result.horizon, result.symmetry, result):
        assert abs(estimate.deg-angle) <= estimate.error_deg
    assert result.significant == (angle != 0.)


def test_specular_rod_and_yoneda_band():
    data, valid, geometry = _frame(-2.15, technique="gisaxs")
    result = measure_sample_roll(data, geometry, valid=valid, technique="gisaxs")
    assert result.method == "combined"
    for estimate in (result.horizon, result.symmetry, result):
        assert abs(estimate.deg+2.15) <= estimate.error_deg


def test_conflicting_texture_and_shadow_are_reported():
    data, valid, geometry = _frame(3., edge_angle=-4.)
    result = measure_sample_roll(data, geometry, valid=valid)
    assert result.method == "disagreement"
    assert result.deg is None
    assert not result.significant


def test_inclined_fibre_without_a_shadow_cannot_set_sample_roll():
    data, valid, geometry = _frame(3., shadow=False)
    result = measure_sample_roll(data, geometry, valid=valid)
    assert result.horizon is None and result.symmetry.deg == pytest.approx(3., abs=.5)
    assert result.deg is None and not result.significant
    assert result.method == "unsupported fibre tilt"
    assumed = measure_sample_roll(data, geometry, valid=valid, assume_fibre_normal=True)
    assert assumed.significant and assumed.deg == result.symmetry.deg
    assert assumed.assume_fibre_normal


@pytest.mark.parametrize("masked", [True, False])
def test_uniform_or_unmeasured_frame_cannot_set_roll(masked):
    data, valid, geometry = _frame(2.)
    data[:] = 12
    if masked:
        valid[:] = False
    result = measure_sample_roll(data, geometry, valid=valid)
    assert result.method == "unavailable"
    assert not result.significant


def test_p08_ring_symmetry_and_conditional_fibre_normal_reduction():
    from tools.benchmarks.sample_roll import DATA, measure_public

    if not (DATA / "giwaxs_p08_mapi").is_dir():
        pytest.skip("Public P08 frame is unavailable.")
    result = measure_public("giwaxs_p08_mapi")
    assert result["method"] == "unsupported fibre tilt" and not result["significant"]
    assert abs(result["horizon"]["deg"]) < 3 * result["horizon"]["error_deg"]
    assert result["incidence"]["roll_deg"] == 0.
    result = measure_public("giwaxs_p08_mapi", assume_fibre_normal=True)
    assert result["significant"] and result["method"] == "fibre assumption"
    assert -2. < result["deg"] < -.7
    incidence = result["incidence"]
    assert incidence["used_estimators"] == ("horizon", "specular")
    assert incidence["dominant_estimator"] == "specular"
    assert incidence["estimator_weights"]["specular"] > .99
    assert abs(incidence["deg"]-.50) <= incidence["error_deg"] < .03
    assert incidence["deg"]-result["published_incidence_deg"] > 3*incidence["error_deg"]
    # The full frame and the nearest half of strips have different apparent edges.
    assert incidence["horizon"]["error_deg"] > .2
    profile = result["strongest_ring"]
    before = profile["counts_at_abs_chi_84_to_86_before"]
    after = profile["counts_at_abs_chi_84_to_86_after"]
    assert abs(after[1]-after[0]) < .5*abs(before[1]-before[0])
    assert profile["common_one_degree_bins"] >= 70
    assert profile["rms_signed_difference_after"] < profile["rms_signed_difference_before"]


def test_featureless_poisson_background_and_gaps_cannot_set_roll():
    data, valid, geometry = _frame(0.)
    data = np.random.default_rng(1420).poisson(75., size=data.shape)
    data[~valid] = 1000000
    result = measure_sample_roll(data, geometry, valid=valid)
    assert not result.significant


def test_isolated_rod_after_manual_roll_on_reversed_detector():
    data, valid, geometry = _frame(2.5, technique="gisaxs", orientation=4, shadow=False)
    result = measure_sample_roll(data, geometry.with_sample_roll(-1.), valid=valid, technique="gisaxs")
    assert result.horizon is None and result.method == "symmetry"
    assert abs(result.deg-2.5) <= result.error_deg


def test_measure_before_film_lorentz_and_fill_retains_observed_pixel_corrections(tmp_path, monkeypatch):
    from dataclasses import replace

    from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, Corrections
    from src.gimap.features.analyze.application import sample_roll
    from src.gimap.features.analyze.application.models import GeometryResolution
    from tests.test_analyze_frames_and_center import FakeFrames

    data, valid, geometry = _frame(2.)
    frames = FakeFrames({"sample.cbf": [data], "dark.cbf": [np.full(data.shape, 2.)]})
    (tmp_path / "dark.cbf").touch()  # the in-memory adapter supplies the pixels, the cache uses file timestamps
    flat = np.broadcast_to(np.linspace(80., 120., data.shape[1]), data.shape)
    np.save(tmp_path / "flat.npy", flat)
    np.save(tmp_path / "mask.npy", ~valid)
    settings = Corrections(dark_path=str(tmp_path / "dark.cbf"), flat_field_path=str(tmp_path / "flat.npy"),
                           mask_path=str(tmp_path / "mask.npy"))
    request = AnalysisRequest(tmp_path / "sample.cbf", mode="giwaxs", with_map=False, with_record=False,
                              corrections=settings)
    reducer = AnalyzeFrame(frames, None)
    resolution = GeometryResolution(geometry.with_sample_roll(-3.), None, "chosen")
    reference = reducer(request, resolution=resolution)
    dependent = replace(settings, film_thickness_nm=110., attenuation_length_um=4., lorentz=True, mirror_fill=True)
    corrected = reducer(replace(request, corrections=dependent), resolution=resolution)
    assert np.any(reference.valid & ~corrected.valid)
    assert corrected.filled_pixels.any()

    def capture(data, g, *, valid, technique):
        np.testing.assert_array_equal(valid, reference.valid)
        np.testing.assert_array_equal(data[valid], reference.data[valid])
        assert technique == "giwaxs" and g == resolution.geometry

    monkeypatch.setattr(sample_roll, "measure_sample_roll", capture)
    sample_roll.measure_frame_roll(corrected, frames)
