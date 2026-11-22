"""GIWAXS intensity corrections: solid angle, polarisation, film absorption, and their errors."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.gimap.features.analyze.application import (
    AnalysisRequest,
    AnalyzeFrame,
    AnalyzeSettings,
    Corrections,
    analysis_metadata,
    fit_input_observation,
    settings_from_record,
    settings_record,
)
from src.gimap.features.analyze.domain import BinnedMean, giwaxs_maps, intensity_factor, reduce_giwaxs
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_assistant_calibration import SHAPE
from tests.test_giwaxs_workspace import GEOMETRY

ON = dict(solid_angle=True, polarization=0.99)


def _two_theta(geometry: DetectorGeometry, row: int, column: int) -> tuple[float, float, float]:
    x = (column + 0.5 - geometry.beam_center_x_px) * geometry.pixel_size_x_m
    y = (geometry.beam_center_y_px - (row + 0.5)) * geometry.pixel_size_y_m
    return x, y, math.atan2(math.hypot(x, y), geometry.distance_m)


def test_solid_angle_and_polarisation_follow_their_formulas() -> None:
    geometry = GEOMETRY
    solid = intensity_factor(SHAPE, geometry, solid_angle=True)
    horizontal = intensity_factor(SHAPE, geometry, polarization=1.0)
    unpolarised = intensity_factor(SHAPE, geometry, polarization=0.0)
    for row, column in ((250, 399), (0, 200), (10, 20), (249, 200)):
        x, y, two_theta = _two_theta(geometry, row, column)
        assert solid[row, column] == pytest.approx(math.cos(two_theta) ** 3, rel=1e-5)
        cos_2phi = (x * x - y * y) / (x * x + y * y)
        expected = 0.5 * (1 + math.cos(two_theta) ** 2 - cos_2phi * math.sin(two_theta) ** 2)
        assert horizontal[row, column] == pytest.approx(expected, rel=1e-5)
        assert unpolarised[row, column] == pytest.approx(0.5 * (1 + math.cos(two_theta) ** 2), rel=1e-5)
    x, _y, two_theta = _two_theta(geometry, 249, 399)  # in the horizontal plane: cos² 2θ
    assert horizontal[249, 399] == pytest.approx(math.cos(two_theta) ** 2, rel=1e-3)
    assert float(solid[249, 199]) == pytest.approx(1.0, abs=1e-6)  # at the beam


def test_film_absorption_limits() -> None:
    geometry = GEOMETRY  # αi = 0.2°
    thick = intensity_factor(SHAPE, geometry, film_thickness_nm=1e6, attenuation_length_um=0.01)  # t ≫ L
    thin = intensity_factor(SHAPE, geometry, film_thickness_nm=0.1, attenuation_length_um=1e6)  # t ≪ L
    maps = giwaxs_maps(SHAPE, geometry)
    above = maps.above_horizon
    assert np.all(np.isnan(thick[~above])) and np.all(np.isfinite(thick[above]))
    np.testing.assert_allclose(thin[above], 1.0, rtol=1e-4)  # a thin film absorbs nothing either way
    sin_i = math.sin(geometry.incidence_rad)
    row = int(np.flatnonzero(above[:, 200])[-1])  # just above the horizon (αf < αi): long path out, less signal
    x, y = (200 + 0.5 - geometry.beam_center_x_px) * 1e-4, (geometry.beam_center_y_px - row - 0.5) * 1e-4
    distance = geometry.distance_m
    radius = math.sqrt(distance**2 + x**2 + y**2)
    sin_f = (y * math.cos(geometry.incidence_rad) - distance * sin_i) / radius
    expected = (2.0 / sin_i) / (1.0 / sin_i + 1.0 / sin_f)  # thick film: 1/s relative to the specular
    assert 0 < sin_f < sin_i
    assert thick[row, 200] == pytest.approx(expected, rel=1e-4) and thick[row, 200] < 1.0
    high = int(np.flatnonzero(above[:, 200])[0])  # far above: a short path out, more signal than at the specular
    assert thick[high, 200] > 1.0


def test_scaled_counts_carry_their_poisson_variance() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    plain = BinnedMean(np.array([0.0, 0.25, 0.5]))
    plain.add(x, values)
    same = BinnedMean(np.array([0.0, 0.25, 0.5]))
    same.add(x, values, np.ones(4))
    np.testing.assert_allclose(plain.result()[2], same.result()[2])
    scaled = BinnedMean(np.array([0.0, 0.25, 0.5]))
    scaled.add(x[:2], values[:2])  # counts first, then corrected counts
    scaled.add(x[2:], values[2:], np.array([2.0, 4.0]))
    _centres, _mean, sigma, _pixels = scaled.result()
    assert sigma[0] == pytest.approx(math.sqrt(30.0) / 2)
    assert sigma[1] == pytest.approx(math.sqrt(2 * 30.0 + 4 * 40.0) / 2)


def _counts_frame(geometry: DetectorGeometry, seed: int, *, level: float = 400.0) -> np.ndarray:
    """A flat isotropic scatterer as a detector sees it (solid angle × polarisation), in photon counts."""
    factor = intensity_factor(SHAPE, geometry, **ON)
    return np.random.default_rng(seed).poisson(level * factor).astype(np.int32)


def test_a_flat_scatterer_is_flat_once_corrected_with_honest_errors() -> None:
    geometry = GEOMETRY
    frame = _counts_frame(geometry, 1)
    valid = np.ones(SHAPE, dtype=bool)
    maps = giwaxs_maps(SHAPE, geometry)
    raw = reduce_giwaxs(frame.astype(np.float32), valid, geometry, maps=maps, with_map=False).curve("radial")
    factor = intensity_factor(SHAPE, geometry, **ON)
    corrected = reduce_giwaxs(
        (frame / factor).astype(np.float32), valid, geometry, maps=maps, with_map=False, counts=(1.0 / factor),
    ).curve("radial")
    assert raw.intensity[-5:].mean() < 0.9 * raw.intensity[:5].mean()  # measured: falls off with 2θ
    residual = (corrected.intensity - 400.0) / corrected.sigma
    assert abs(float(np.mean(corrected.intensity)) - 400.0) < 2.0
    assert 0.7 < float(np.std(residual)) < 1.3  # σ from the propagated variance matches the scatter
    azimuthal = reduce_giwaxs(
        (frame / factor).astype(np.float32), valid, geometry, maps=maps, with_map=False, counts=(1.0 / factor),
        settings=None,
    ).curve("azimuthal")
    assert azimuthal is None or np.nanstd(azimuthal.intensity) / np.nanmean(azimuthal.intensity) < 0.05


def test_the_analysis_applies_them_for_giwaxs_and_records_them(tmp_path: Path) -> None:
    path = tmp_path / "flat_001.tif"
    Image.fromarray(_counts_frame(GEOMETRY, 2), mode="I").save(path)
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    analyze = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    request = AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic")
    plain = analyze(request)
    corrected = analyze(replace(request, corrections=Corrections(**ON)))
    assert plain.intensity_scale is None and corrected.intensity_scale is not None
    assert corrected.raw_data is not None and np.array_equal(corrected.raw_data, plain.data)  # the raw frame kept
    flat = corrected.reduction.curve("radial").intensity
    assert np.std(flat) / np.mean(flat) < np.std(plain.reduction.curve("radial").intensity) / np.mean(
        plain.reduction.curve("radial").intensity)
    record = analysis_metadata(corrected)
    assert record["intensity_corrections"]["polarization"]["factor"] == pytest.approx(0.99)
    assert "solid_angle" in record["intensity_corrections"]
    assert analysis_metadata(plain)["intensity_corrections"] is None
    assert not fit_input_observation(corrected)["counting_model_valid"]
    again = analyze(replace(request, corrections=Corrections(**ON)), loaded=corrected)  # re-analysis from the raw frame
    np.testing.assert_allclose(again.reduction.curve("radial").intensity, flat)
    gisaxs = analyze(replace(request, mode="gisaxs", corrections=Corrections(**ON)))
    assert gisaxs.intensity_scale is None  # GIWAXS only


def test_settings_files_keep_the_intensity_corrections() -> None:
    corrections = Corrections(solid_angle=True, polarization=0.95, film_thickness_nm=80.0, attenuation_length_um=300.0)
    record = settings_record(AnalyzeSettings(mode="giwaxs", corrections=corrections))
    back = settings_from_record(json.loads(json.dumps(record)))
    assert back.corrections == corrections
    old = settings_record(AnalyzeSettings(mode="giwaxs"))
    for key in ("solid_angle", "polarization", "film_thickness_nm", "attenuation_length_um"):
        old["corrections"].pop(key)
    assert settings_from_record(old).corrections == Corrections()  # files written before: nothing corrected


@pytest.mark.parametrize("thickness", [140., None])
def test_reduction_recovers_a_ring_at_subcritical_exit_angles(tmp_path, thickness):
    from src.gimap.shared.geometry.q_mapping import exit_directions, geometry_rays

    geometry = replace(GEOMETRY, incidence_deg=.52, sample_roll_deg=-1.3)
    delta, beta = 2.6e-6, 1.6e-7
    along, x, y = geometry_rays(geometry, (0, SHAPE[0]), (0, SHAPE[1]))
    _, _, sin_f = exit_directions(x, y, along, geometry.incidence_rad, sample_roll_rad=geometry.sample_roll_rad)
    sin_i = np.sin(geometry.incidence_rad)
    k = 20 * np.pi / geometry.wavelength_angstrom  # nm^-1
    kz_i = k * np.sqrt(sin_i**2 - 2*delta + 2j*beta)

    def depth_weight(sine):
        kz = k * np.sqrt(sine**2 - 2*delta + 2j*beta)
        decay = 2 * (kz_i.imag + kz.imag)
        field = (2*k*sin_i / (k*sin_i + kz_i)) * (2*k*sine / (k*sine + kz))
        return abs(field)**2 * (1. if thickness is None else -np.expm1(-decay*thickness)) / decay

    maps = giwaxs_maps(SHAPE, geometry)
    signal = 50. + 500.*np.exp(-.5*((maps.q - 1.1)/.025)**2)
    above = sin_f > 0.
    raw = np.zeros(SHAPE, np.float32)
    raw[above] = signal[above] * depth_weight(sin_f[above]) / depth_weight(sin_i)
    path = tmp_path / "ring.tif"
    Image.fromarray(raw).save(path)
    profile = InstrumentProfile("film test", geometry, None, SHAPE)
    reduce = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    corrections = Corrections(film_delta=delta, film_beta=beta, film_thickness_nm=thickness, bad_pixels=False)
    analysis = reduce(AnalysisRequest(path=path, mode="giwaxs", profile_name=profile.name, corrections=corrections))
    subcritical = above & (sin_f < np.sqrt(2*delta)) & (np.abs(maps.q - 1.1) < .025)
    assert subcritical.any() and not np.any(analysis.valid[~above]) and np.all(analysis.valid[above])
    np.testing.assert_allclose(analysis.data[above], signal[above], rtol=2e-5)
    meta = analysis_metadata(analysis)["intensity_corrections"]["film_absorption"]
    assert meta["model"] == "refracted waves" and meta["delta"] == delta and meta["beta"] == beta
    assert meta["thickness_nm"] == thickness and meta["thick_limit"] == (thickness is None)
    assert settings_from_record(settings_record(AnalyzeSettings(corrections=corrections))).corrections == corrections


def test_film_cif_is_resolved_at_the_frame_wavelength_and_recorded(tmp_path):
    from src.gimap.features.analyze.application.indexing import phase_from_cif
    from src.gimap.features.analyze.domain.crystal.film_optics import film_optics

    cif = tmp_path / "film.cif"
    cif.write_text("""data_film
_chemical_formula_sum 'Si O2'
_cell_formula_units_Z 2
_cell_length_a 4.5
_cell_length_b 4.5
_cell_length_c 4.5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
""")
    path = tmp_path / "film.tif"
    Image.fromarray(np.full(SHAPE, 100, dtype=np.uint16)).save(path)
    corrections = Corrections(film_cif_path=str(cif), bad_pixels=False, lorentz=True)
    phase = phase_from_cif(cif)
    assert phase.positions_only and not phase.phase().atoms
    for wavelength in (.70, 1.0):
        profile = InstrumentProfile("film test", replace(GEOMETRY, wavelength_angstrom=wavelength), None, SHAPE)
        reducer = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
        analysis = reducer(AnalysisRequest(path=path, mode="giwaxs", profile_name=profile.name, corrections=corrections))
        record = analysis_metadata(analysis)
        optic = record["intensity_corrections"]["film_absorption"]
        expected = film_optics(phase.phase(), wavelength)
        assert optic["delta"] == expected["delta"] and optic["beta"] == expected["beta"]
        assert optic["source"] == str(cif)
        assert {"file": cif.name, "sha256": phase.source_sha256} in record["input_sha256"]
    # Restore the measured rays using the constants at reduction time, even if the CIF moved.
    cif.rename(tmp_path / "moved.cif")
    snapshot = (vars(analysis), analysis.corrections, analysis.geometry, analysis.raw_data, analysis.raw_valid, path.name)
    data, valid, _ = reducer.measured_orientation_pixels(*snapshot)
    assert valid.any() and np.isfinite(data[valid]).all()


@pytest.mark.parametrize("mode,key", [("transmission", "radial"), ("gisaxs", "horizontal")])
def test_inactive_film_cif_is_not_required_to_reproduce_other_modes(tmp_path, mode, key):
    from src.gimap.features.analyze.infrastructure.reproduce import replay_reduction

    path = tmp_path / "sample.tif"
    Image.fromarray(np.full(SHAPE, 20, dtype=np.uint16)).save(path)
    profile = InstrumentProfile("film test", GEOMETRY, None, SHAPE)
    reducer = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    absent = tmp_path / "unused.cif"
    analysis = reducer(AnalysisRequest(path=path, mode=mode, profile_name=profile.name,
                                       corrections=Corrections(film_cif_path=str(absent))))
    record = analysis_metadata(analysis, curve_keys=[key])
    assert record["intensity_corrections"] is None and str(absent) not in record["reproduction"]["inputs"]
    replayed = replay_reduction(record, tmp_path)
    np.testing.assert_array_equal(replayed[key]["intensity"], analysis.reduction.curve(key).intensity)


def test_film_controls_apply_only_complete_inputs_and_restore_them(qt_application, monkeypatch, tmp_path):
    from src.gimap.features.analyze.presentation.views.film_controls import FilmControls
    from PyQt5.QtWidgets import QFileDialog

    widget = FilmControls()
    widget.check.setChecked(True)
    assert widget.values()["attenuation_length_um"] == 100.
    widget.model.setCurrentIndex(widget.model.findData("typed"))
    widget.delta.setText("2.25")
    assert not any(widget.values().values())
    widget.beta.setText("0.13")
    assert widget.values()["film_beta"] == pytest.approx(1.3e-7) and widget.values()["film_thickness_nm"] is None
    widget.thick.setChecked(False)
    widget.thickness.setValue(125.)
    saved = Corrections(**widget.values())
    widget.restore(Corrections())
    assert not widget.check.isChecked()
    widget.restore(saved)
    assert Corrections(**widget.values()) == saved and widget.model.currentData() == "typed"
    path = str(tmp_path / "test.cif")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (path, ""))
    widget.model.setCurrentIndex(widget.model.findData("cif"))
    widget._choose()
    assert widget.values()["film_cif_path"] == path and widget.values()["film_delta"] is None
    widget.close()
