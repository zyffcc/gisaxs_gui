"""Flat field (detector response) of Analyze: the factor, its variance, the pixels it leaves out, the record.

- ``F_ff`` is the flat field divided by its mean over the pixels it keeps (``F_ff > 0.05``, finite, not masked by
  its reader); the corrected value is ``I / F_ff``; pixels it does not keep are left out and counted;
- photon counts divided by it have ``Var(I / F) = I/F² + I² Var(F)/F⁴`` (``scaled_poisson``), in
  GISAXS cuts and GIWAXS bins, and composed with the GIWAXS intensity corrections (``F = F_Ω F_P F_A F_ff``);
- a floating-point frame keeps the pixel-scatter σ; a flat field of another shape is refused, never resampled;
- the record's ``corrections`` block names its file, SHA-256, normalisation mean and the pixels it leaves out;
  settings files keep it; the same file serves every frame of a batch or series (read once).
"""

from __future__ import annotations

import hashlib
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
from src.gimap.features.analyze.application.export_record import error_model
from src.gimap.features.analyze.domain import intensity_factor, normalise_flat_field
from src.gimap.features.analyze.domain.flat_field import FLOOR
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_assistant_calibration import SHAPE
from tests.test_giwaxs_workspace import GEOMETRY

GISAXS_SHAPE = (160, 240)
GISAXS_GEOMETRY = DetectorGeometry(172e-6, 172e-6, 2.5, 120.0, 140.0, 1.0, 0.3)
LEVEL = 400.0


def _flat(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """An invented detector response between 0.3 and 1.7 (smooth modules plus pixel noise), in the units of a
    flat-field file, with dead pixels (0), a NaN, one pixel just below the floor and one just above it."""
    rows, columns = np.indices(shape)
    response = 1.0 + 0.35 * np.sin(columns / 9.0) + 0.15 * np.cos(rows / 13.0)
    response += np.random.default_rng(seed).normal(0.0, 0.1, shape)
    raw = 7000.0 * np.clip(response, 0.3, 1.7)
    raw[3, 5] = raw[7, 11] = raw[20, 30] = 0.0
    raw[10, 10] = np.nan
    raw[12, 40] = -5.0
    return raw


def _counts(level: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).poisson(level).astype(np.int32)


def _save_int(path: Path, data: np.ndarray) -> Path:
    Image.fromarray(np.asarray(data, dtype=np.int32), mode="I").save(path)
    return path


def _analyze(profile: InstrumentProfile) -> AnalyzeFrame:
    return AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))


# -- the factor ------------------------------------------------------------------------------------------------------


def test_the_flat_field_is_normalised_to_one_over_the_pixels_it_keeps_and_counts_the_others() -> None:
    raw = _flat(GISAXS_SHAPE)
    mean_before = float(np.nanmean(raw[raw > 0]))
    raw[50, 50] = 0.04 * mean_before  # does not respond: at or below 0.05 of the mean
    raw[50, 52] = 0.2 * mean_before  # a weak pixel, still used
    flat = normalise_flat_field(raw, name="flat.npy")
    kept = flat.kept
    assert float(np.mean(flat.factor[kept])) == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_allclose(flat.factor[kept], raw[kept] / flat.mean, rtol=1e-6)
    assert np.all(flat.factor[kept] > FLOOR) and np.all(np.isnan(flat.factor[~kept]))
    # Self-consistent: every finite pixel above 0 that is not kept is at or below the floor of the final mean.
    dropped = np.isfinite(raw) & (raw > 0) & ~kept
    assert np.all(raw[dropped] <= FLOOR * flat.mean)
    assert not kept[50, 50] and kept[50, 52]
    for row, column in ((3, 5), (7, 11), (20, 30), (10, 10), (12, 40)):
        assert not kept[row, column]
    assert flat.masked_pixels == 6 == int((~kept).sum())
    # The reader's mask is left out too; an image without any response is refused.
    reader = np.ones(GISAXS_SHAPE, dtype=bool)
    reader[0, :] = False
    assert normalise_flat_field(raw, reader).masked_pixels == 6 + GISAXS_SHAPE[1] - int((~kept[0]).sum())
    with pytest.raises(ValueError, match="has no pixel above 0"):
        normalise_flat_field(np.zeros((4, 4)), name="dark.npy")
    with pytest.raises(ValueError, match="not one 2D image"):
        normalise_flat_field(np.ones((2, 4, 4)), name="stack.npy")


# -- the variance of corrected photon counts ----------------------------------------------------------------------


def test_gisaxs_counts_divided_by_the_flat_field_are_flat_with_scaled_poisson_errors(tmp_path: Path) -> None:
    raw = _flat(GISAXS_SHAPE, seed=3)
    np.save(tmp_path / "flat.npy", raw)
    factor = normalise_flat_field(raw).factor
    frame = _counts(LEVEL * np.nan_to_num(factor), seed=4)  # a uniform scatterer as this detector records it
    path = _save_int(tmp_path / "film_001.tif", frame)
    profile = InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE)
    analyze = _analyze(profile)
    request = AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic",
                              corrections=Corrections(bad_pixels=False))
    plain = analyze(request)
    corrected = analyze(replace(request, corrections=Corrections(bad_pixels=False,
                                                                 flat_field_path=str(tmp_path / "flat.npy"))))
    assert error_model(plain) == "poisson" and error_model(corrected) == "scaled_poisson"
    np.testing.assert_allclose(corrected.intensity_scale[corrected.valid], 1.0 / factor[corrected.valid], rtol=1e-6)
    assert np.array_equal(corrected.raw_data, plain.data)  # the raw frame is kept for re-analysis
    curve = corrected.reduction.curve("horizontal")
    measured = plain.reduction.curve("horizontal")
    assert np.std(measured.intensity) / np.mean(measured.intensity) > 0.15  # the detector's response, measured
    residual = (curve.intensity - LEVEL) / curve.sigma
    assert abs(float(np.mean(curve.intensity)) - LEVEL) < 3.0 and 0.8 < float(np.std(residual)) < 1.2
    # The normalisation mean is fixed. For an uncorrected Poisson flat, Var(F) = F / mean(D_ff).
    start, stop = curve.region["rows"]
    data, valid = corrected.data[start:stop], corrected.valid[start:stop]
    columns = [column for column in range(GISAXS_SHAPE[1]) if valid[:, column].any()]
    expected = []
    for column in columns:
        inside = valid[:, column]
        values, f = data[inside, column].astype(float), factor[start:stop][inside, column].astype(float)
        n = int(inside.sum())
        shot = max(np.sum(values / f), np.mean(1.0 / f)**2)
        reference = np.sum(values**2 / (f * corrected.flat_field.mean))
        expected.append(math.sqrt(shot + reference) / n)
    np.testing.assert_allclose(curve.sigma, expected, rtol=1e-5)


def test_giwaxs_composes_the_flat_field_with_the_intensity_corrections(tmp_path: Path) -> None:
    raw = _flat(SHAPE, seed=5)
    np.save(tmp_path / "flat.npy", raw)
    flat = normalise_flat_field(raw).factor
    geometric = intensity_factor(SHAPE, GEOMETRY, solid_angle=True, polarization=0.99)
    path = _save_int(tmp_path / "ring_001.tif", _counts(LEVEL * np.nan_to_num(geometric * flat), seed=6))
    analyze = _analyze(InstrumentProfile("synthetic", GEOMETRY, None, SHAPE))
    corrections = Corrections(bad_pixels=False, solid_angle=True, polarization=0.99,
                              flat_field_path=str(tmp_path / "flat.npy"))
    analysis = analyze(AnalysisRequest(path=path, mode="giwaxs", profile_name="synthetic", corrections=corrections))
    valid = analysis.valid
    np.testing.assert_allclose(analysis.intensity_scale[valid], 1.0 / (geometric * flat)[valid], rtol=1e-5)
    radial = analysis.reduction.curve("radial")
    residual = (radial.intensity - LEVEL) / radial.sigma
    assert abs(float(np.median(radial.intensity)) - LEVEL) < 4.0 and 0.7 < float(np.std(residual)) < 1.3
    assert error_model(analysis) == "scaled_poisson"
    assert not fit_input_observation(analysis)["counting_model_valid"]


def test_a_floating_point_frame_keeps_the_pixel_scatter_errors(tmp_path: Path) -> None:
    raw = _flat(GISAXS_SHAPE)
    np.save(tmp_path / "flat.npy", raw)
    frame = (LEVEL * np.nan_to_num(normalise_flat_field(raw).factor)).astype(np.float32)
    path = tmp_path / "dark_subtracted.tif"
    Image.fromarray(frame, mode="F").save(path)
    analysis = _analyze(InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE))(AnalysisRequest(
        path=path, mode="gisaxs", profile_name="synthetic",
        corrections=Corrections(bad_pixels=False, flat_field_path=str(tmp_path / "flat.npy"))))
    assert error_model(analysis) == "pixel_scatter" and analysis.intensity_scale is None
    np.testing.assert_allclose(analysis.data[analysis.valid], LEVEL, rtol=1e-5)


# -- the pixels it leaves out, its shape, its file -------------------------------------------------------------------


def test_pixels_the_flat_field_does_not_keep_are_left_out_of_the_frame_and_the_gap_guard_grows_around_them(
        tmp_path: Path) -> None:
    raw = np.full(GISAXS_SHAPE, 5000.0)
    raw[80, 100] = 0.0  # dead
    raw[90, 150] = 100.0  # 0.02 of the mean: does not respond
    raw[95, 160] = np.nan
    np.save(tmp_path / "flat.npy", raw)
    path = _save_int(tmp_path / "film_001.tif", np.full(GISAXS_SHAPE, 50))
    analyze = _analyze(InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE))
    corrections = Corrections(bad_pixels=False, flat_field_path=str(tmp_path / "flat.npy"))
    analysis = analyze(AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic", corrections=corrections))
    for pixel in ((80, 100), (90, 150), (95, 160)):
        assert not analysis.valid[pixel]
    assert int((~analysis.valid).sum()) == 3 == analysis.flat_field.masked_pixels
    guarded = analyze(AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic",
                                      corrections=replace(corrections, gap_guard_px=1)))
    assert int((~guarded.valid).sum()) == 27  # 3 × 3 around each
    assert analysis.reduction is not None
    np.testing.assert_allclose(analysis.data[analysis.valid], 50.0, rtol=1e-6)


def test_a_flat_field_of_another_shape_is_refused_not_resampled(tmp_path: Path) -> None:
    np.save(tmp_path / "small.npy", np.ones((10, 12)))
    path = _save_int(tmp_path / "film_001.tif", np.full(GISAXS_SHAPE, 50))
    analyze = _analyze(InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE))
    message = f"The flat field small.npy is 10×12; the frame is {GISAXS_SHAPE[0]}×{GISAXS_SHAPE[1]}."
    with pytest.raises(ValueError) as refused:
        analyze(AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic",
                                corrections=Corrections(flat_field_path=str(tmp_path / "small.npy"))))
    assert str(refused.value) == message
    with pytest.raises(ValueError) as checked:
        analyze.check_flat_field_file(tmp_path / "small.npy", GISAXS_SHAPE)
    assert str(checked.value) == message


def test_a_flat_field_is_read_from_any_frame_format_or_npy_once_for_every_frame(tmp_path: Path) -> None:
    raw = np.nan_to_num(_flat(GISAXS_SHAPE), nan=0.0).astype(np.float32)
    np.save(tmp_path / "flat.npy", raw)
    Image.fromarray(raw, mode="F").save(tmp_path / "flat.tif")
    source = DetectorIoFrameSource()
    loads: list[str] = []
    original = source.load
    source.load = lambda path, index=0: loads.append(Path(path).name) or original(path, index)
    analyze = AnalyzeFrame(source, InMemoryInstrumentProfileRepository(
        [InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE)]))
    from_npy = analyze.check_flat_field_file(tmp_path / "flat.npy", GISAXS_SHAPE)
    from_tif = analyze.check_flat_field_file(tmp_path / "flat.tif", GISAXS_SHAPE)
    assert from_npy.mean == pytest.approx(from_tif.mean) and from_npy.masked_pixels == from_tif.masked_pixels
    frames = [_save_int(tmp_path / f"film_{index:03d}.tif", np.full(GISAXS_SHAPE, 40 + index)) for index in range(3)]
    corrections = Corrections(bad_pixels=False, flat_field_path=str(tmp_path / "flat.tif"))
    records = [analysis_metadata(analyze(AnalysisRequest(path=frame, mode="gisaxs", profile_name="synthetic",
                                                         corrections=corrections))) for frame in frames]
    assert loads.count("flat.tif") == 1  # one read for the series: the same flat field for every frame
    assert len({json.dumps(record["corrections"]["flat_field"]) for record in records}) == 1


# -- the record and the settings -----------------------------------------------------------------------------------


def test_the_record_names_the_flat_field_and_settings_files_keep_it(tmp_path: Path) -> None:
    raw = _flat(GISAXS_SHAPE)
    np.save(tmp_path / "flat.npy", raw)
    path = _save_int(tmp_path / "film_001.tif", _counts(np.full(GISAXS_SHAPE, LEVEL), seed=7))
    analyze = _analyze(InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE))
    request = AnalysisRequest(path=path, mode="gisaxs", profile_name="synthetic")
    corrections = Corrections(flat_field_path=str(tmp_path / "flat.npy"))
    record = analysis_metadata(analyze(replace(request, corrections=corrections)))
    flat = record["corrections"]["flat_field"]
    expected = normalise_flat_field(raw)
    assert record["corrections"]["flat_field_path"] == str(tmp_path / "flat.npy")
    assert flat["file"] == "flat.npy" and flat["path"] == str(tmp_path / "flat.npy")
    assert flat["sha256"] == hashlib.sha256((tmp_path / "flat.npy").read_bytes()).hexdigest()
    assert flat["normalisation_mean"] == pytest.approx(expected.mean) and flat["masked_pixels"] == expected.masked_pixels
    assert flat["floor"] == FLOOR and flat["shape"] == list(GISAXS_SHAPE) and "I / F_ff" in flat["applied_as"]
    assert record["error_model"] == "scaled_poisson"
    json.dumps(record)
    plain = analysis_metadata(analyze(request))
    assert plain["corrections"]["flat_field"] is None and plain["corrections"]["flat_field_path"] is None
    back = settings_from_record(json.loads(json.dumps(settings_record(AnalyzeSettings(corrections=corrections)))))
    assert back.corrections == corrections
    old = settings_record(AnalyzeSettings())
    old["corrections"].pop("flat_field_path")
    assert settings_from_record(old).corrections.flat_field_path is None  # files written before: no flat field


def test_a_batch_export_divides_every_frame_by_the_same_flat_field_and_records_it(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import BatchChoices, ExportAnalysis, FrameJob, FrameTools, run_frame_job
    from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter

    raw = _flat(GISAXS_SHAPE, seed=8)
    np.save(tmp_path / "flat.npy", raw)
    factor = np.nan_to_num(normalise_flat_field(raw).factor)
    frames = [_save_int(tmp_path / f"film_{index:03d}.tif", _counts(LEVEL * factor, seed=10 + index)) for index in range(3)]
    tools = FrameTools(_analyze(InstrumentProfile("synthetic", GISAXS_GEOMETRY, None, GISAXS_SHAPE)),
                       ExportAnalysis(CsvCurveWriter()))
    corrections = Corrections(bad_pixels=False, flat_field_path=str(tmp_path / "flat.npy"))
    outcomes = [run_frame_job(tools, FrameJob(
        index=index, request=AnalysisRequest(path=frame, mode="gisaxs", profile_name="synthetic", corrections=corrections),
        choices=BatchChoices(tables=False, per_frame=True), destination=tmp_path / "out", live_key="horizontal"))
        for index, frame in enumerate(frames)]
    digest = hashlib.sha256((tmp_path / "flat.npy").read_bytes()).hexdigest()
    for outcome in outcomes:
        assert outcome.metadata["corrections"]["flat_field"]["sha256"] == digest
        assert outcome.metadata["error_model"] == "scaled_poisson"
        assert abs(float(np.mean(outcome.live_row[1])) - LEVEL) < 5.0  # the detector's pattern divided out
    records = sorted((tmp_path / "out").rglob("*_analysis.json"))
    assert len(records) == 3
    assert all(json.loads(path.read_text(encoding="utf-8"))["corrections"]["flat_field"]["sha256"] == digest
               for path in records)
