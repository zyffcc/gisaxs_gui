"""Calibration through detector and text-file inputs, including the measurement's provenance."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, Corrections, ExportAnalysis, Normalisation
from src.gimap.features.analyze.application.absolute_scale import CalibrateStandard, StandardRequest
from src.gimap.features.analyze.application.batch_export import BatchChoices, CurveTables
from src.gimap.features.analyze.application.batch_job import FrameOutcome
from src.gimap.features.analyze.application.batch_report import BatchRun, batch_record, readme_text
from src.gimap.features.analyze.application.export_record import analysis_metadata, fit_input_observation
from src.gimap.features.analyze.application.nxcansas_export import sas_curves
from src.gimap.features.analyze.application.series import CorrectSeriesFrame
from src.gimap.features.analyze.application.standard_curves import read_standard_curve
from src.gimap.features.analyze.domain import SeriesCorrection
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource
from src.gimap.features.analyze.presentation.batch_model import BatchModelMixin
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile


def _profile():
    geometry = DetectorGeometry(0.00012, 0.00015, 0.15, 43.25, 61.75, 1.02, 0.2)
    return InstrumentProfile("Test bench", geometry, None, (96, 112))


def _worker(corrections=Corrections(bad_pixels=False)):
    profile = _profile()
    return CalibrateStandard(DetectorIoFrameSource(), profile, profile.geometry, profile.detector_shape,
                             corrections, "transmission", bins=24)


def test_detector_standard_reuses_mask_background_and_flat_field_before_its_own_normalisation(tmp_path):
    profile = _profile()
    shape = profile.detector_shape
    data = np.random.default_rng(71).poisson(430, shape).astype(np.int32)
    data[5:18, 9:20] += 5000
    mask = np.zeros(shape, np.uint8)
    mask[5:18, 9:20] = 1
    measured, mask_path, background = (tmp_path / name for name in ("standard.tif", "mask.tif", "empty.tif"))
    for array, path in ((data, measured), (mask, mask_path), (np.full(shape, 31, np.int32), background)):
        Image.fromarray(array).save(path)
    flat_path = tmp_path / "response.npy"
    flat = np.ones(shape, np.float32)
    flat[:, :40] *= 1.09
    np.save(flat_path, flat)
    corrections = Corrections(mask_path=str(mask_path), background_path=str(background), background_scale=2.0,
                              flat_field_path=str(flat_path), bad_pixels=False)
    sample = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    cut = AnalysisRequest(measured, mode="transmission", profile_name=profile.name, corrections=corrections)
    truth = sample(replace(cut, giwaxs=replace(cut.giwaxs, bins=24), with_map=False)).reduction.curve("radial")
    scale = 0.0032
    reference = tmp_path / "test_reference.csv"
    # t = 8 s, M = 2, T = 0.5, d = 2.5 mm = 0.25 cm. Reference σ is in cm⁻¹.
    np.savetxt(reference, np.column_stack((truth.x, truth.intensity * scale / 2, truth.sigma * scale / 2)),
               delimiter=",", header="q,I,sigma")
    previous = replace(corrections, normalisation=Normalisation(exposure="value", exposure_s=5, absolute_factor=97))
    request = StandardRequest("frame", measured, exposure_s=8, monitor=2, transmission=0.5, thickness_mm=2.5,
                              reference_path=reference)
    result = _worker(previous)(request)
    assert result.fit.factor == pytest.approx(scale, rel=1e-6)
    assert previous.normalisation.absolute_factor == 97
    record = result.record
    assert record["geometry"]["beam_center_x_px"] == profile.geometry.beam_center_x_px
    assert record["frame_index"] == 0 and record["instrument_profile"] == profile.name
    assert record["corrections"]["background_scale"] == 2 and record["reference"]["file"]["sha256"]
    assert record["input_sha256"][0]["file"] == measured.name
    assert record["standard_normalisation"]["thickness_cm"] == 0.25
    assert {item["file"] for item in record["correction_files"]} == {mask_path.name, background.name, flat_path.name}
    assert record["q_range_A_inv"] == pytest.approx([truth.x[0], truth.x[-1]])


def test_reduced_curve_and_reference_q_units_are_independent(tmp_path):
    source, ref = tmp_path / "calibration.dat", tmp_path / "table.csv"
    q = np.linspace(0.12, 0.9, 12)
    level = 20.0 / (1 + q ** 4 * 23)
    np.savetxt(source, np.column_stack((q * 10, level)), header="q I", comments="")
    np.savetxt(ref, np.column_stack((q, 0.002 * level, np.full(q.shape, 0.0001))), delimiter=",",
               header="q,dSigma/dOmega,sigma", comments="")
    request = StandardRequest("curve", source, source_q_unit="nm⁻¹", thickness_mm=10, reference_path=ref,
                              q_range=(0.2, 0.8))
    result = _worker()(request)
    assert result.fit.factor == pytest.approx(0.002)
    assert result.record["standard"] == ref.name
    assert result.record["requested_q_range_A_inv"] == [0.2, 0.8]
    assert result.record["source_file"]["sha256"]
    assert result.record["date_utc"].endswith("+00:00")
    assert min(result.fit.q) > 0.2 and max(result.fit.q) < 0.8


def test_normalised_gimap_curve_and_missing_reference_sigma_are_refused(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text("# GIMaP Analyze export (q in 1/A, angles in deg)\n"
                    "# intensity: counts/s; normalised I = A*I/(t*M*T*d), factor 2\n"
                    "q (1/A),I,sigma,pixels\n0.1,0.02,0.001,30\n")
    with pytest.raises(ValueError, match="normalisation off"):
        read_standard_curve(path, "Å⁻¹")
    path.write_text("q,I\n0.02,2.8\n")
    with pytest.raises(ValueError, match="three columns"):
        read_standard_curve(path, "Å⁻¹", reference=True)


def test_laboratory_q_export_and_pixel_count_and_shape_validation(tmp_path):
    path = tmp_path / "radial.txt"
    path.write_text("# GIMaP Analyze export (q in 1/A, angles in deg)\n"
                    "q (1/A)\tI (counts/pixel)\tsigma\tpixels\n0.1\t50\t2\t25\n0.2\tnan\tnan\t0\n")
    q, level, sigma = read_standard_curve(path, "Å⁻¹")
    assert q.tolist() == [0.1] and level.tolist() == [50] and sigma.tolist() == [2]
    with pytest.raises(ValueError, match="GIMaP radial curves"):
        read_standard_curve(path, "nm⁻¹")
    frame = tmp_path / "short.tif"
    Image.fromarray(np.ones((12, 12), np.uint8)).save(frame)
    with pytest.raises(ValueError, match="detector shape"):
        _worker()(StandardRequest("frame", frame))
    with pytest.raises(ValueError, match="frame number"):
        _worker()(StandardRequest("frame", frame, frame_index=1))


def test_analyze_radial_exports_round_trip_as_measured_curves(tmp_path):
    profile = _profile()
    frame = tmp_path / "detector image.tif"
    data = np.random.default_rng(34).poisson(49, profile.detector_shape).astype(np.int32)
    Image.fromarray(data).save(frame)
    sample = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    request = AnalysisRequest(frame, profile_name=profile.name, mode="transmission", with_map=False)
    analysis = sample(request)
    export = ExportAnalysis(CsvCurveWriter())
    radial = analysis.reduction.curve("radial")
    for fmt in ("csv", "txt", "dat"):
        curve_path, _record = export.curves(analysis, ("radial",), tmp_path, text_format=fmt)
        q, intensity, sigma = read_standard_curve(curve_path, "Å⁻¹")
        np.testing.assert_allclose(q, radial.x, rtol=1e-7)
        np.testing.assert_allclose(intensity, radial.intensity, rtol=1e-7)
        np.testing.assert_allclose(sigma, radial.sigma, rtol=1e-7)
    fitted_file = export.fit_input(analysis, tmp_path)
    np.testing.assert_allclose(read_standard_curve(fitted_file, "Å⁻¹")[1], radial.intensity, rtol=1e-7)
    norm = Normalisation(exposure="value", exposure_s=5)
    scaled = sample(replace(request, corrections=Corrections(normalisation=norm)))
    with pytest.raises(ValueError, match="normalisation off"):
        read_standard_curve(export.fit_input(scaled, tmp_path), "Å⁻¹")


@pytest.mark.parametrize("peak_scaled", [False, True])
def test_batch_records_curve_units_after_standard_and_reference_peak(tmp_path, peak_scaled):
    profile = _profile()
    frame = tmp_path / "frame.tif"
    Image.fromarray(np.random.default_rng(8).poisson(310, profile.detector_shape).astype(np.int32)).save(frame)
    source = tmp_path / "water.txt"
    np.savetxt(source, [[0.05, 210], [0.1, 230]])
    standard = _worker()(StandardRequest("curve", source, thickness_mm=1)).record
    norm = Normalisation(thickness_mm=2.5, absolute_factor=standard["factor"], standard_calibration=standard)
    sample = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))
    request = AnalysisRequest(frame, profile_name=profile.name, mode="giwaxs",
                              corrections=Corrections(normalisation=norm))
    analysis, _ = CorrectSeriesFrame(sample)(request, SeriesCorrection(normalize=peak_scaled), factor=7)
    radial = analysis.reduction.curve("radial")
    unit = "relative" if peak_scaled else "cm⁻¹"
    assert radial.y_label == f"I ({unit})"
    record = analysis_metadata(analysis)
    assert record["output_intensity_unit"] == unit and record["normalisation"]["standard_calibration"] == standard
    if peak_scaled:
        assert record["reference_peak_scale"] == 7
    assert fit_input_observation(analysis)["intensity_unit"] == unit
    sas, = sas_curves([radial], record, run="frame")
    assert sas.intensity_unit == ("arbitrary" if peak_scaled else "1/cm")
    writer = CsvCurveWriter()
    export = ExportAnalysis(writer)
    fit_input = export.fit_input(analysis, tmp_path).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="normalisation off"):
        read_standard_curve(tmp_path / "frame_fit_input.dat", "Å⁻¹")
    tables = CurveTables()
    tables.add("frame.tif", [radial])
    table, = tables.write(writer, tmp_path, "run")
    choices = BatchChoices()
    run = BatchRun(choices, tmp_path, "run")
    BatchModelMixin().fold_outcome(run, FrameOutcome(0, "frame", "frame", metadata=record))
    batch = batch_record(run, [table])
    assert batch["settings_of_first_frame"]["normalisation"]["standard_calibration"] == standard
    entry, = batch["frame_normalisations"]
    assert entry["output_intensity_unit"] == unit and entry["normalisation"]["thickness_cm"]["value"] == 0.25
    assert entry["normalisation"]["standard_calibration"]["$ref"].endswith("/normalisation/standard_calibration")
    if peak_scaled:
        assert entry["reference_peak_scale"] == 7
    readme = readme_text(choices, batch="run", frame="frame", frames=1, curves=[], run=run)
    for text in (fit_input, table.read_text(encoding="utf-8"), readme):
        assert ("relative" if peak_scaled else "cm^-1") in text
    if peak_scaled:
        cake = export.cake(analysis, sample.cake(analysis), tmp_path / "cake.csv")
        assert "# intensity: cm^-1;" in cake.read_text(encoding="utf-8")
