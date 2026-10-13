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
