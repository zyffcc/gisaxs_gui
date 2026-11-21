"""Normalisation, dark frame and Lorentz factor of Analyze: the factors, their σ, the records, the defaults.

- with every new option off the curves and their σ are those of the release before them (two public frames);
- ``I_norm = A · I / (t · M · T · d)``: one constant ``c`` per frame; σ is ``c`` times what it was, for photon counts
  (Poisson carried through ``c``) and for the pixel scatter alike; each term says where it came from;
- the exposure time and monitors come from the header (summed over summed frames) or are typed; a header that lacks
  the value asked for is said, not guessed;
- a dark frame is subtracted before the background and makes σ the pixel scatter;
- the Lorentz factor of a fibre texture is GIXSGUI's ``1 / (cos αi cos αf |sin 2θf|)``; the Hermans factor of a
  Lorentz-corrected I(χ) is not weighted by sin χ again.
"""

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
    ExportAnalysis,
    Normalisation,
    analysis_metadata,
    fit_input_observation,
    settings_from_record,
    settings_record,
)
from src.gimap.features.analyze.application.export_record import error_model, provenance_lines
from src.gimap.features.analyze.domain import CutRegion, header_values, intensity_factor, summed_header_values
from src.gimap.features.analyze.domain.normalisation import NO_EXPOSURE, NO_MONITOR, normalisation_use
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource
from src.gimap.features.assistant.domain import parse_poni
from src.gimap.features.assistant.domain.orientation import ring_orientation
from src.gimap.features.assistant.domain.profiles import Profile
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.q_mapping import exit_directions, geometry_rays
from tests.test_assistant_calibration import SHAPE
from tests.test_giwaxs_workspace import GEOMETRY

DATA = Path(__file__).resolve().parent / "data" / "external"
GISAXS_SHAPE = (160, 240)
GISAXS_GEOMETRY = DetectorGeometry(172e-6, 172e-6, 2.5, 120.0, 140.0, 1.0, 0.3)
LEVEL = 400.0

MAIN_CURVES = {  # (points, Σx, ΣI, Σσ, Σpixels) of each curve with Corrections(), as the release before computed them
    "galaxi": {"horizontal": (974, -48.3395814206, 121654.0, 3153.87629673, 4870),
               "vertical": (638, 119.013875328, 278261.0, 1236.95412207, 7018)},
    "p08": {"radial": (2050, 4901.63819899, 41555.2331458, 1343.12795251, 3731451),
            "in_plane": (1396, 2363.36253493, 13451.4440037, 2930.53001397, 216834),
            "out_of_plane": (1384, 2265.20326733, 40319.2098257, 2859.11223685, 254936),
            "azimuthal": (174, 0.0, 25913.5746985, 2341.33099952, 18712)},
}


def _analyze(profile: InstrumentProfile) -> AnalyzeFrame:
    return AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))


def _save_int(path: Path, data: np.ndarray) -> Path:
    Image.fromarray(np.asarray(data, dtype=np.int32), mode="I").save(path)
    return path


def _save_edf(path: Path, data: np.ndarray, header: dict) -> Path:
    import fabio

    fabio.edfimage.EdfImage(data=np.asarray(data, dtype=np.int32), header=header).write(str(path))
    return path


def _counts_frame(tmp_path: Path, name: str = "film_001.tif", seed: int = 1) -> Path:
    return _save_int(tmp_path / name, np.random.default_rng(seed).poisson(LEVEL, GISAXS_SHAPE))


def _gisaxs(tmp_path: Path):
    return _analyze(InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE))


# -- the defaults ----------------------------------------------------------------------------------------------------


def test_with_every_new_option_off_the_curves_and_errors_are_those_of_the_release_before() -> None:
    poni = parse_poni((DATA / "giwaxs_p08_mapi/LaB6_2021_12_DESY_P08.poni").read_text(encoding="utf-8"))
    cases = {
        "galaxi": ("gisaxs_galaxi/galaxi_data.tif", "gisaxs",
                   DetectorGeometry(172e-6, 172e-6, 1.730, 597.1, 719.6, 1.34, 0.463), (1043, 981)),
        "p08": ("giwaxs_p08_mapi/S121_MAI_A2_00841.tif", "giwaxs",
                DetectorGeometry(poni.pixel_size_x_m, poni.pixel_size_y_m, poni.distance_mm / 1e3,
                                 poni.beam_center_x_px, poni.beam_center_y_px, poni.wavelength_angstrom, 0.075),
                (2048, 2048)),
    }
    for name, (frame, mode, geometry, shape) in cases.items():
        analysis = _analyze(InstrumentProfile(name, geometry, None, shape))(AnalysisRequest(
            path=DATA / frame, mode=mode, profile_name=name, corrections=Corrections()))
        assert analysis.normalisation is None and analysis.intensity_scale is None
        record = analysis_metadata(analysis)
        assert record["normalisation"] is None and not any("normalised" in line or "dark frame" in line
                                                           for line in provenance_lines(record, "Analyze"))
        curves = {curve.key: curve for curve in analysis.reduction.curves}
        assert set(curves) == set(MAIN_CURVES[name])
        for key, (points, x, intensity, sigma, pixels) in MAIN_CURVES[name].items():
            curve = curves[key]
            assert len(curve.x) == points and int(np.sum(curve.pixels)) == pixels, key
            assert curve.y_label == "I (counts/pixel)"
            np.testing.assert_allclose([np.sum(curve.x), np.sum(curve.intensity), np.sum(curve.sigma)],
                                       [x, intensity, sigma], rtol=1e-9, atol=1e-9, err_msg=f"{name} {key}")


# -- the normalisation -----------------------------------------------------------------------------------------------


def test_the_constant_scales_intensities_and_sigma_and_says_each_term_and_its_source(tmp_path: Path) -> None:
    path = _counts_frame(tmp_path)
    analyze = _gisaxs(tmp_path)
    request = AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic", corrections=Corrections(bad_pixels=False))
    plain = analyze(request)
    norm = Normalisation(exposure="value", exposure_s=4.0, transmission=0.5, thickness_mm=0.25, absolute_factor=3.0)
    scaled = analyze(replace(request, corrections=Corrections(bad_pixels=False, normalisation=norm)))
    factor = 3.0 / (4.0 * 0.5 * 0.25)
    assert scaled.normalisation.factor == pytest.approx(factor)
    for key in ("horizontal", "vertical"):
        before, after = plain.reduction.curve(key), scaled.reduction.curve(key)
        np.testing.assert_allclose(after.intensity, factor * before.intensity, rtol=1e-6)
        np.testing.assert_allclose(after.sigma, factor * before.sigma, rtol=1e-5)  # Poisson carried through c
        np.testing.assert_array_equal(after.x, before.x)
        assert after.y_label == "I (counts/(pixel·s·mm) × 3)"
    assert error_model(plain) == "poisson" and error_model(scaled) == "scaled_poisson"
    assert np.array_equal(scaled.raw_data, plain.data)  # the frame as loaded is kept for re-analysis
    record = analysis_metadata(scaled)
    terms = record["normalisation"]
    assert terms["factor"] == pytest.approx(factor) and terms["intensity_unit"] == "counts/(pixel·s·mm) × 3"
    assert terms["exposure_time_s"] == {"value": 4.0, "source": "typed"}
    assert terms["transmission"] == {"value": 0.5, "source": "typed"}
    assert record["corrections"]["normalisation"]["absolute_factor"] == 3.0
    observation = fit_input_observation(scaled)
    assert observation["intensity_unit"] == "counts/(pixel·s·mm) × 3" and not observation["counting_model_valid"]
    written = ExportAnalysis(CsvCurveWriter())(scaled, tmp_path / "out")
    text = next(path for path in written if path.name.endswith("_horizontal.csv")).read_text(encoding="utf-8")
    assert "# intensity: counts/(pixel*s*mm) x 3; normalised I = A*I/(t*M*T*d), factor 6" in text  # (export_text)
    assert "exposure_time_s 4 (typed)" in text and "sigma scaled by the same factor" in text
    assert "I (counts/(pixel*s*mm) x 3),sigma,pixels" in text


def test_a_frame_of_pixel_scatter_keeps_that_error_model_scaled(tmp_path: Path) -> None:
    frame = np.random.default_rng(2).normal(50.0, 7.0, GISAXS_SHAPE).astype(np.float32)
    path = tmp_path / "dark_subtracted.tif"
    Image.fromarray(frame, mode="F").save(path)
    analyze = _gisaxs(tmp_path)
    request = AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic", corrections=Corrections(bad_pixels=False))
    plain = analyze(request)
    scaled = analyze(replace(request, corrections=Corrections(
        bad_pixels=False, normalisation=Normalisation(monitor="value", monitor_value=2.0e5))))
    assert error_model(scaled) == "pixel_scatter" and scaled.intensity_scale is None
    curve, before = scaled.reduction.curve("horizontal"), plain.reduction.curve("horizontal")
    np.testing.assert_allclose(curve.intensity, before.intensity / 2.0e5, rtol=1e-6)
    np.testing.assert_allclose(curve.sigma, before.sigma / 2.0e5, rtol=1e-5)
    assert curve.y_label == "I (counts/(pixel·monitor))"


def test_exposure_and_monitor_come_from_the_header_and_a_sum_is_divided_by_their_sums(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    first = _save_edf(tmp_path / "scan_0001.edf", rng.poisson(LEVEL, GISAXS_SHAPE),
                      {"ExposureTime": "20.0", "Monitor": "5000", "Title": "film"})
    second = _save_edf(tmp_path / "scan_0002.edf", rng.poisson(LEVEL, GISAXS_SHAPE),
                       {"ExposureTime": "10.0", "Monitor": "2500"})
    offered = header_values(DetectorIoFrameSource().load(first).metadata)
    assert offered == {"exposure_time_s": 20.0, "monitors": {"Monitor": 5000.0}}
    assert summed_header_values([offered, {"exposure_time_s": 10.0, "monitors": {"Monitor": 2500.0}}]) == {
        "exposure_time_s": 30.0, "monitors": {"Monitor": 7500.0}}
    analyze = _gisaxs(tmp_path)
    norm = Normalisation(exposure="header", monitor="header", monitor_key="Monitor")
    request = AnalysisRequest(path=first, mode="gisaxs", profile_name="synthetic",
                              corrections=Corrections(bad_pixels=False, normalisation=norm))
    single = analyze(request)
    assert single.normalisation.factor == pytest.approx(1.0 / (20.0 * 5000.0))
    assert single.normalisation.term("monitor") == (5000.0, "header (Monitor)")
    summed = analyze(replace(request, summed_frames=((second, 0),)))
    assert summed.normalisation.factor == pytest.approx(1.0 / (30.0 * 7500.0))
    assert summed.header_values == {"exposure_time_s": 30.0, "monitors": {"Monitor": 7500.0}}
    assert analysis_metadata(summed)["normalisation"]["exposure_time_s"] == {
        "value": 30.0, "source": "header (exposure_time_s)"}


def test_a_header_without_the_value_asked_for_is_said_not_guessed(tmp_path: Path) -> None:
    path = _counts_frame(tmp_path, "film_002.tif")  # a TIFF: no header at all
    analyze = _gisaxs(tmp_path)
    with pytest.raises(ValueError) as exposure:
        analyze(AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic", corrections=Corrections(
            normalisation=Normalisation(exposure="header"))))
    assert str(exposure.value) == NO_EXPOSURE.format(name="film_002.tif")
    with pytest.raises(ValueError) as monitor:
        normalisation_use(Normalisation(monitor="header", monitor_key="I0"), {"monitors": {"Monitor": 1.0}}, "f.edf")
    assert str(monitor.value) == NO_MONITOR.format(name="f.edf", key="I0")
    for bad in (dict(transmission=0.0), dict(transmission=1.5), dict(thickness_mm=-1.0), dict(exposure="value")):
        with pytest.raises(ValueError):
            Normalisation(**bad)


def test_monitor_names_of_cbf_and_edf_headers_are_offered() -> None:
    cbf = {"header": {"_array_data.header_contents": "# Detector: PILATUS\r\n# Exposure_time 1.0 s\r\n"
                                                     "# Flux 1.5e+12\r\n# Tau = 0 s\r\n"}, "exposure_time_s": 1.0}
    assert header_values(cbf) == {"exposure_time_s": 1.0, "monitors": {"Flux": 1.5e12}}
    edf = {"header": {"Intensity0": "123456.0", "I0": "7", "Dummy": "-1", "Title": "x"}}
    assert header_values(edf)["monitors"] == {"Intensity0": 123456.0, "I0": 7.0}
    nexus = {"monitor": 9.5e6, "monitor_source": "/entry/monitor/data", "exposure_time_s": 0.1}
    assert header_values(nexus) == {"exposure_time_s": 0.1, "monitors": {"/entry/monitor/data": 9.5e6}}


def test_settings_files_keep_the_dark_frame_the_normalisation_and_the_lorentz_factor(tmp_path: Path) -> None:
    corrections = Corrections(dark_path=str(tmp_path / "dark.tif"), dark_frame=2, dark_scale=0.5, lorentz=True,
                              normalisation=Normalisation(exposure="header", monitor="header", monitor_key="I0",
                                                          transmission=0.8, absolute_factor=1.5e-3))
    record = json.loads(json.dumps(settings_record(AnalyzeSettings(corrections=corrections))))
    assert settings_from_record(record).corrections == corrections
    old = settings_record(AnalyzeSettings())
    for key in ("dark_path", "dark_frame", "dark_scale", "normalisation", "lorentz"):
        old["corrections"].pop(key)
    assert settings_from_record(old).corrections == Corrections()  # a file of the release before: all off


# -- the dark frame --------------------------------------------------------------------------------------------------


def test_the_dark_frame_is_subtracted_before_the_background_and_sigma_is_the_pixel_scatter(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    frame = _save_int(tmp_path / "film.tif", rng.poisson(LEVEL, GISAXS_SHAPE) + 100)
    dark = _save_int(tmp_path / "dark.tif", np.full(GISAXS_SHAPE, 50))
    background = _save_int(tmp_path / "substrate.tif", np.full(GISAXS_SHAPE, 30))
    analyze = _gisaxs(tmp_path)
    corrections = Corrections(bad_pixels=False, dark_path=str(dark), dark_scale=2.0, background_path=str(background),
                              background_scale=0.5)
    analysis = analyze(AnalysisRequest(path=frame, mode="gisaxs", profile_name="synthetic", corrections=corrections))
    raw = analysis.raw_data
    np.testing.assert_allclose(analysis.data, raw - 2.0 * 50 - 0.5 * 30, rtol=1e-6)
    assert error_model(analysis) == "pixel_scatter" and analysis.intensity_scale is None
    assert not fit_input_observation(analysis)["counting_model_valid"]
    lines = provenance_lines(analysis_metadata(analysis), "Analyze")
    assert "dark frame: dark.tif (index 0) x 2, subtracted before the background" in lines
    wrong = _save_int(tmp_path / "dark_small.tif", np.zeros((10, 10)))
    with pytest.raises(ValueError, match="The dark frame is 10×10"):
        analyze(AnalysisRequest(path=frame, mode="gisaxs", profile_name="synthetic",
                                corrections=Corrections(dark_path=str(wrong))))


# -- the Lorentz factor ----------------------------------------------------------------------------------------------


def test_the_lorentz_factor_is_gixsgui_s_fibre_texture_factor() -> None:
    factor = intensity_factor(SHAPE, GEOMETRY, lorentz=True)
    along, x, y = geometry_rays(GEOMETRY, (0, SHAPE[0]), (0, SHAPE[1]))
    _along_beam, across, sin_exit = exit_directions(x, y, along, GEOMETRY.incidence_rad)
    cos_exit = np.sqrt(1.0 - sin_exit**2)
    sin_2theta_f = np.abs(across) / cos_exit  # across = cos αf sin 2θf
    expected = 1.0 / (math.cos(GEOMETRY.incidence_rad) * cos_exit * sin_2theta_f)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(factor[finite], expected[finite], rtol=1e-5)
    # Near the beam 1/L ≈ (q/k) sin χ: the sin χ weight of the Hermans factor, made exact.
    k = 2 * math.pi / GEOMETRY.wavelength_angstrom
    from src.gimap.features.analyze.domain import giwaxs as giwaxs_domain

    maps = giwaxs_domain.giwaxs_maps(SHAPE, GEOMETRY)
    approx = (maps.q / k) * np.abs(np.sin(np.radians(maps.chi_deg)))
    near = maps.above_horizon & (maps.q < 0.3 * maps.q.max()) & (np.abs(maps.chi_deg) > 20)
    np.testing.assert_allclose(1.0 / factor[near], approx[near], rtol=0.05)


def test_lorentz_corrected_rings_count_crystallites_and_record_the_formula(tmp_path: Path) -> None:
    path = _save_int(tmp_path / "ring.tif", np.random.default_rng(5).poisson(LEVEL, SHAPE))
    analyze = _analyze(InstrumentProfile("synthetic", GEOMETRY, None, SHAPE))
    request = AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic", corrections=Corrections(bad_pixels=False))
    plain = analyze(request)
    corrected = analyze(replace(request, corrections=Corrections(bad_pixels=False, lorentz=True)))
    factor = intensity_factor(SHAPE, GEOMETRY, lorentz=True)
    valid = corrected.valid
    np.testing.assert_allclose(corrected.data[valid], (plain.data / factor)[valid], rtol=1e-5)
    np.testing.assert_allclose(corrected.intensity_scale[valid], (1.0 / factor)[valid], rtol=1e-5)
    record = analysis_metadata(corrected)["intensity_corrections"]
    assert "GIXSGUI" in record["lorentz"]["formula"] and "not weighted by sin(chi) again" in record["lorentz"]["orientation"]
    assert error_model(corrected) == "scaled_poisson"


def test_the_hermans_factor_of_a_lorentz_corrected_ring_is_not_weighted_twice() -> None:
    chi = np.arange(-89.5, 90.0, 1.0)
    density = 1.0 + 4.0 * np.exp(-0.5 * (chi / 15.0) ** 2)  # pole density: oriented along the normal
    sigma = np.full(chi.shape, 0.02)
    measured = ring_orientation(Profile.of(chi, 100.0 * density, sigma, np.ones_like(chi)), q_window=(1.0, 1.1))
    corrected = ring_orientation(Profile.of(chi, 100.0 * density * np.abs(np.sin(np.radians(chi))), sigma,
                                            np.ones_like(chi)), q_window=(1.0, 1.1), lorentz=True)
    assert corrected.herman == pytest.approx(measured.herman, abs=0.01)  # one sin χ, not two
    twice = ring_orientation(Profile.of(chi, 100.0 * density * np.abs(np.sin(np.radians(chi))), sigma,
                                        np.ones_like(chi)), q_window=(1.0, 1.1))
    assert twice.herman < measured.herman - 0.1  # what weighting it again would have given
    assert corrected.herman_isotropic == pytest.approx(measured.herman_isotropic)
    assert "not weighted by sin χ again" in corrected.notes[0]


# -- exports ---------------------------------------------------------------------------------------------------------


def test_region_curves_are_written_under_their_names(tmp_path: Path) -> None:
    path = _save_int(tmp_path / "ring.tif", np.random.default_rng(6).poisson(LEVEL, SHAPE))
    analyze = _analyze(InstrumentProfile("synthetic", GEOMETRY, None, SHAPE))
    from src.gimap.features.analyze.domain import GiwaxsSettings

    regions = (CutRegion("ring q1.0", (0.3, 0.5)), CutRegion("radial", (0.6, 0.8)), CutRegion("ring q1.0", (0.9, 1.0)),
               CutRegion("spot 60°", (0.4, 0.6), (50.0, 70.0)))
    analysis = analyze(AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic",
                                       corrections=Corrections(bad_pixels=False), giwaxs=GiwaxsSettings(regions=regions)))
    names = {Path(item).name for item in ExportAnalysis(CsvCurveWriter())(analysis, tmp_path / "out")}
    assert {"ring_spot_60.csv", "ring_spot_60_chi.csv",  # a name of its own: just the name
            "ring_ring_q1_0_region1.csv", "ring_ring_q1_0_region1_chi.csv", "ring_ring_q1_0_region3.csv",  # shared
            "ring_radial.csv", "ring_radial_region2.csv", "ring_radial_region2_chi.csv"} <= names  # a curve's key
    assert not {"ring_region1.csv", "ring_region4.csv"} & names
