"""Records and report of the automatic analysis: the Record section (release, input SHA-256, calibration file and
instrument profile, corrections), ± in the peak table, the same ``#`` lines in the window's CSVs as in the command
line's, sizes with a unit in their key, the geometry at full precision, the profile's ``tilt`` key, and the report's
wording about fitted peaks, series hints and poor fits."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.features.analyze.application.export_record import provenance_lines
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.features.assistant.application import PipelineOptions, StandardPipeline, pipeline_markdown
from src.gimap.features.assistant.application.giwaxs_decisions import peak_cap_decision, uncertainty_decision
from src.gimap.features.assistant.application.run_record import (
    exact_calibration,
    profile_source,
    record_lines,
    run_record,
)
from src.gimap.features.assistant.domain import Profile, find_peaks, scherrer_size
from src.gimap.features.assistant.domain.peaks import POOR_FIT_CHI2, _series_hints
from src.gimap.shared.file_digest import sha256_of
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.detector_tilt import DetectorTilt
from src.gimap.shared.version import SOFTWARE
from tests.test_assistant_pipeline import setup

GEOMETRY = {
    "pixel_size_x_m": 1.72e-4, "pixel_size_y_m": 1.72e-4, "distance_m": 0.14758232012185601,
    "beam_center_x_px": 7.047881178006987, "beam_center_y_px": 589.4898129471951, "wavelength_angstrom": 1.54,
    "incidence_deg": 0.235, "rot1_rad": -0.089, "rot2_rad": -0.057, "rot3_rad": 0.0, "critical_angle_deg": 0.0,
}


def _files(folder: Path) -> tuple[Path, Path]:
    frame, poni = folder / "film_00028.edf", folder / "standard.poni"
    frame.write_bytes(b"frame bytes")
    poni.write_text("Distance: 0.1476\n", encoding="utf-8")
    return frame, poni


def _analysis_record(frame: Path, source: str) -> dict:
    return {
        "source_file": str(frame), "frame": "frame 1 of 1", "geometry": dict(GEOMETRY),
        "instrument_profile": {"name": "Detector 619×487", "source": source, "updated_at": "2026-01-02T03:04:05+00:00"},
        "corrections": {"background_path": None, "gap_guard_px": 3, "mask_path": None, "mask_shapes": [],
                        "mirror_fill": False, "bad_pixels": True},
        "intensity_corrections": None, "bad_pixels_left_out": {"hot": 2, "dead": 0, "defective_line_pixels": 0},
        "error_model": "poisson", "input_sha256": [{"file": frame.name, "sha256": sha256_of(frame)}],
    }


def _report(frame: Path, poni: Path) -> dict:
    peak = {"q": 0.2212, "d_A": 28.41, "fwhm": 0.021, "snr": 11.3, "flags": ["poor_fit"], "caveat": "",
            "orientation": "mainly out-of-plane", "size_nm": 26.9, "size_is_lower_bound": True, "size_note": "",
            "fit": {"q_err": 0.00077, "d_err_A": 0.098, "fwhm_err": 0.00002, "reduced_chi2": 2731.0}}
    return {
        "ok": True, "procedure": "giwaxs", "frame": str(frame), "needs_attention": [], "decisions": [], "steps": [],
        "geometry": {"source": "instrument profile 'Detector 619×487'", "distance_mm": 147.58232012185601,
                     "beam_center_px": [7.05, 589.49], "wavelength_A": 1.54, "incidence_deg": 0.235},
        "calibration": {"source_image": str(poni), "from_file": True}, "peaks": [peak] * 9, "rings": [],
        "peak_search": {"truncated": True, "max_peaks": 12, "candidates": 21},
    }


def test_a_saved_report_names_the_release_the_input_and_the_calibration(tmp_path: Path) -> None:
    frame, poni = _files(tmp_path)
    source = profile_source("automatic analysis", poni.name, str(poni))
    assert source == f"automatic analysis: {poni.name} [{poni}]"  # the profile keeps the path, not only the name
    report = _report(frame, poni)
    assert record_lines(report) == []  # no record attached yet: no section
    report.update(run_record(report, _analysis_record(frame, source)))
    text = pipeline_markdown(report)
    section = text[text.index("## Record"):text.index("## Peaks")]
    assert f"- written by {SOFTWARE}" in section
    assert f"- input: `{frame}` (SHA-256 {sha256_of(frame)}, 11 bytes)" in section
    assert f"- calibration file: `{poni}` (SHA-256 {sha256_of(poni)}" in section
    assert f"- instrument profile: 'Detector 619×487' (source: {source}" in section
    assert "no background subtracted" in section and "intensity corrections: none" in section
    assert "- mask applied: none" in text and "2 hot" in text
    assert "## 记录" in pipeline_markdown(report, language="zh")


def test_the_window_attaches_the_record_and_the_page_names_the_release(tmp_path: Path) -> None:
    from src.gimap.features.assistant.presentation.guided_report import attach_record, report_html, report_markdown

    frame, poni = _files(tmp_path)
    report = _report(frame, poni)

    class Automation:
        def analysis_record(self):
            return _analysis_record(frame, "automatic analysis: standard.poni")

    attach_record(report, Automation())
    assert report["software_version"] == SOFTWARE and report["inputs"]["frame"][0]["sha256"] == sha256_of(frame)
    assert report["inputs"]["calibration"]["path"] == str(poni) and report["mask"]["applied"] is False
    markdown = report_markdown(report)
    assert sha256_of(frame) in markdown and "instrument profile: 'Detector 619×487'" in markdown
    page = report_html(report, markdown)
    assert SOFTWARE in page and sha256_of(frame) in page
    untouched = {"frame": str(frame), "geometry": {}}
    attach_record(untouched, object())  # no frame record: the report stays as it is
    assert "software_version" not in untouched


def test_the_peak_table_gives_the_errors_and_says_how_many_peaks_passed(tmp_path: Path) -> None:
    text = pipeline_markdown(_report(*_files(tmp_path)))
    row = next(line for line in text.splitlines() if line.startswith("| 0.2212"))
    assert row.startswith("| 0.22120(77) | 28.410(98) | 0.021000(20) | 11.3 |")
    assert len(row) < 100 and row.endswith("| [1] |")
    assert "- [1] poor fit (χ²_r above 10)" in text
    assert "The 12 most prominent of 21 candidate peaks were fitted; 9 of them gave a peak and are in the table" in text
    assert "Only the 12" not in text

    class Search:
        candidates, max_peaks, peaks = 21, 12, (object(),) * 9

    assert peak_cap_decision(Search())["decision"] == (
        "21 candidate peaks found; the 12 most prominent were fitted, 9 of them gave a peak")


def test_a_lamellar_hint_needs_consecutive_orders() -> None:
    assert _series_hints([0.4524, 0.8989, 2.225]) == ()  # ratios 1 : 1.99 : 4.92: orders 3 and 4 are missing
    hints = _series_hints([0.2, 0.4, 0.6])
    assert len(hints) == 1 and "lamellar (1:2:3:4)" in hints[0] and "0.2, 0.4, 0.6" in hints[0]
    assert _series_hints([0.2, 0.4, 0.6, 1.0]) == hints  # a fifth order is no part of a 1:2:3:4 series
    hexagonal = _series_hints([0.2, 0.2 * 3 ** 0.5, 0.4])
    assert len(hexagonal) == 1 and "hexagonal" in hexagonal[0]  # named only with an order 1:2:3 does not have


def test_a_peak_far_from_a_gaussian_is_flagged_as_a_poor_fit() -> None:
    q = np.linspace(0.1, 0.5, 200)
    lorentz = 5.0e4 / (1.0 + ((q - 0.3) / 0.01) ** 2) + 2.0e4 / (1.0 + ((q - 0.325) / 0.004) ** 2)
    counts = 500.0 + lorentz
    poor = find_peaks(Profile.of(q, counts, np.sqrt(counts / 400.0), np.full_like(q, 400.0)))
    worst = max(poor.peaks, key=lambda peak: peak.reduced_chi2 or 0.0)
    assert worst.reduced_chi2 > POOR_FIT_CHI2 and "poor_fit" in worst.flags
    rng = np.random.default_rng(3)
    clean = 500.0 + 300.0 * np.exp(-0.5 * ((q - 0.3) / 0.008) ** 2)
    noisy = clean + rng.normal(0.0, np.sqrt(clean / 400.0))
    good = find_peaks(Profile.of(q, noisy, np.sqrt(clean / 400.0), np.full_like(q, 400.0)))
    assert good.peaks and all("poor_fit" not in peak.flags for peak in good.peaks)
    why = uncertainty_decision("poisson")["why"]
    assert "multiplied by √χ²_r" in why and "χ²_r = 10" in why  # what peaks._fit_peak does, said in Decisions


def test_every_size_of_the_record_names_its_unit() -> None:
    record = scherrer_size(1.0, 0.02, 0.001).record()
    assert "size" not in record and "size_err" not in record
    assert record["size_nm"] == pytest.approx(record["size_A"] / 10.0)
    assert record["size_err_nm"] == pytest.approx(record["size_err_A"] / 10.0)


def test_the_profile_file_calls_the_tilt_tilt_and_still_reads_the_old_key() -> None:
    tilt = DetectorTilt(distance_m=0.1468, poni1_m=0.1098, poni2_m=-0.0119, rot1_rad=-0.089, rot2_rad=-0.057,
                        rot3_rad=0.0, pixel1_m=1.72e-4, pixel2_m=1.72e-4)
    geometry = DetectorGeometry(1.72e-4, 1.72e-4, 0.1476, 7.0, 589.5, 1.54, 0.2)
    saved = InstrumentProfile("tilted", geometry, None, (619, 487), dropped_tilt=tilt).to_dict()
    assert "tilt" in saved and "dropped_tilt" not in saved
    old = {**{key: value for key, value in saved.items() if key != "tilt"}, "dropped_tilt": saved["tilt"]}
    for values in (saved, old):
        read = InstrumentProfile.from_dict(values)
        assert read.dropped_tilt == tilt and read.geometry.rot1_rad == pytest.approx(-0.089)


def test_the_windows_csv_starts_with_the_command_lines_lines(tmp_path: Path) -> None:
    frame, _poni = _files(tmp_path)
    record = _analysis_record(frame, "by hand")
    lines = provenance_lines(record, "command line", "report.json")
    assert lines[0] == f"{SOFTWARE} (command line); settings and decisions in report.json"
    assert lines[1] == f"input: {frame.name} sha256 {sha256_of(frame)}" and lines[2] == "frame: frame 1 of 1"
    assert lines[3].startswith("geometry: distance 147.5823 mm, direct beam (7.048, 589.490) px, wavelength 1.54 A, "
                               "alpha_i 0.235 deg, sample roll 0 deg, detector tilt 6.05")
    path = CsvCurveWriter().write_map(
        np.ones((2, 2)), [0.0, 1.0], [0.0, 1.0], tmp_path / "map.csv", ("q_par (Å⁻¹)", "qz (Å⁻¹)"), record)
    header = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert f"# {SOFTWARE} (Analyze)" in header
    assert f"# {lines[1]}" in header and f"# {lines[3]}" in header
    assert sum(line.startswith("# frame") for line in header) == 1  # its own frame line, not a second one


def test_the_record_keeps_the_geometry_as_applied(tmp_path: Path) -> None:
    catalog, workbench, _calibrator = setup(tmp_path, frames=1)
    answered = workbench.status

    def status() -> dict:
        values = answered()
        if values.get("geometry"):
            values["geometry"] = {**values["geometry"], "distance_mm": 147.58232012185601, "tilt_deg": 6.05813304651471}
        return values

    workbench.status = status
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, incidence_deg=0.2)).run()
    assert report["geometry"]["distance_mm"] == 147.582320121856  # 15 significant digits, not 147.6
    assert report["geometry"]["tilt_deg"] == pytest.approx(6.05813304651471, abs=1e-12)
    assert "distance 148 mm" in pipeline_markdown(report)  # the text rounds for reading

    rounded = {"from_file": True, "distance_mm": 147.6, "beam_center_px": [7.048, 589.5], "wavelength_angstrom": 1.54,
               "detector_tilt": {"rot1_rad": -0.08902}}
    used = {"distance_mm": 147.58232012185601, "beam_center_x_px": 7.047881178006987,
            "beam_center_y_px": 589.4898129471951, "wavelength_angstrom": 1.54,
            "detector_tilt": {"rot1_rad": -0.08901621237574575}}
    exact = exact_calibration(rounded, used)
    assert exact["distance_mm"] == used["distance_mm"] and exact["beam_center_px"][1] == used["beam_center_y_px"]
    assert exact["detector_tilt"]["rot1_rad"] == used["detector_tilt"]["rot1_rad"]
    assert exact_calibration({**rounded, "distance_mm": 370.0}, used)["distance_mm"] == 370.0  # another geometry


def test_report_reuses_applied_angle_precision_and_preserves_peak_csv(tmp_path):
    from src.gimap.features.assistant.application.peak_export import peak_table_csv

    report = _report(*_files(tmp_path))
    report["geometry"].update(incidence_deg=0.42738, sample_roll_deg=-1.2849)
    report["incidence_measurement"] = dict(reference_deg=0.12, reference_source="header", deg=0.4274,
                                          error_deg=0.0136, decision="disagreement", method="specular")
    report["sample_roll_measurement"] = dict(deg=-1.2849, error_deg=0.179, significant=True,
                                            origin="automatic", method="horizon")
    report["peaks"][0]["q"] = 0.2212013549
    text = pipeline_markdown(report)
    assert "measured αi 0.427 ± 0.014° differ by more than 3σ. Used 0.427°." in text
    assert "sample roll ψ applied: -1.28°" in text
    assert "selected estimate (horizon): -1.28 ± 0.18°" in text
    assert "0.22120(77)" in text and "0.2212013549,0.00077" in peak_table_csv(report)
    assert report["geometry"]["incidence_deg"] == 0.42738
    report["geometry"]["incidence_deg"] = 0.12
    report["incidence_measurement"]["decision"] = "user_override"
    assert "Used 0.12°." in pipeline_markdown(report)


def test_compact_cell_and_strain_table_keeps_the_q_shift_and_residual_precision():
    from src.gimap.features.assistant.application.pipeline_cell import cell_lines

    phase = {"ok": True, "phase": "Synthetic cell", "system": "cubic", "values": {"a": 5.432789},
             "errors": {"a": 0.001071}, "strain": {"a": 0.0043997}, "strain_errors": {"a": 0.0001945},
             "reference": {"a": 5.409}, "zero_shift": None, "zero_shift_error": None, "chi2_red": 1.8034, "dof": 3,
             "residuals": [{"q": 1.15775, "sigma": 0.0116, "residual": -0.00388, "hkl": [1, 0, 0]}]}
    lines = cell_lines([phase])
    assert "| a (Å) | 5.4328(11) | 5.409 | 0.440(19) |" in lines
    assert "| 1 0 0 | 1.158(12) | -0.004 |" in lines
    assert all(len(line) < 100 for line in lines if line.startswith("|"))
