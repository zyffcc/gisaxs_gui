"""A detector tilt GIMaP's flat geometry leaves out is measured and said, not folded away in silence.

GIMaP's geometry is a flat detector perpendicular to the beam; a ``.poni`` tilt (rot1, rot2) is converted into the
direct beam and its distance, then dropped. ``shared/geometry/detector_tilt`` compares |q| of the two geometries
pixel by pixel. The hand-computed case: rot1 = 60°, rot2 = rot3 = 0, a pyFAI distance L = 0.05 m, so the direct beam
is D = L / cos rot1 = 0.1 m from the sample. A pixel centre on the beam's row, u = 0.1 m from the beam along the
detector, is at 2θ = atan(u cos rot1 / (D − u sin rot1)) = atan(2 + √3) = 75° on the tilted detector, and at
2θ = atan(u / D) = 45° in GIMaP's flat geometry. So |q| is off by sin 22.5° / sin 37.5° − 1 = −0.37137.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.domain.validity import valid_pixels
from src.gimap.features.assistant.domain import parse_poni
from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.detector_tilt import (
    Q_ERROR_LIMIT,
    TILT_LIMIT_DEG,
    DetectorTilt,
    flat_detector_q_error,
    tilted_cos_two_theta,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "tests" / "data" / "external"
ROT1 = math.radians(60.0)
L_PONI = 0.05
PIXEL = 0.1
"""One pixel of 0.1 m: its centre is the point of the hand-computed case."""


def _poni_text(rot1: float, rot2: float = 0.0, rot3: float = 0.0, *, poni1: float, poni2: float, distance: float,
               pixel: float) -> str:
    return (
        f"poni_version: 2\nDetector: Detector\nDetector_config: {{\"pixel1\": {pixel}, \"pixel2\": {pixel}}}\n"
        f"Distance: {distance}\nPoni1: {poni1}\nPoni2: {poni2}\nRot1: {rot1}\nRot2: {rot2}\nRot3: {rot3}\n"
        "Wavelength: 1e-10\n"
    )


def _hand_case(rot3: float = 0.0):
    """One 0.1 m pixel: its centre (0.05, 0.05) m from the corner lies on the beam's row, u = 0.1 m from the beam."""
    poni2 = 0.05 - (-L_PONI * math.tan(ROT1) + 0.1)  # p2 − poni2 = −L tan rot1 + u
    flat = parse_poni(_poni_text(ROT1, rot3=rot3, poni1=0.05, poni2=poni2, distance=L_PONI, pixel=PIXEL))
    geometry = DetectorGeometry(PIXEL, PIXEL, flat.distance_mm * 1e-3, flat.beam_center_x_px, flat.beam_center_y_px,
                                flat.wavelength_angstrom, 0.0)
    return flat, geometry


def test_the_flat_geometry_of_a_tilted_poni_is_off_by_the_hand_computed_amount() -> None:
    flat, geometry = _hand_case()
    assert flat.distance_mm == pytest.approx(100.0)  # D = L / cos 60°
    assert (flat.beam_center_x_px, flat.beam_center_y_px) == pytest.approx((-0.5, 0.5))
    assert flat.tilt is not None and flat.tilt.tilt_deg == pytest.approx(60.0)
    assert math.degrees(math.acos(tilted_cos_two_theta(flat.tilt, np.array([0.0]), np.array([0.0]))[0])) == pytest.approx(75.0)
    error = flat_detector_q_error(flat.tilt, geometry, (1, 1), near_beam_px=0.0)
    expected = math.sin(math.radians(22.5)) / math.sin(math.radians(37.5)) - 1.0
    assert expected == pytest.approx(-0.371374, abs=1e-6)
    assert error.q_error_max == pytest.approx(abs(expected), abs=1e-6) and error.pixels == 1  # a float32 map
    assert error.needs_attention and error.to_dict()["dropped_rotations_deg"]["rot1"] == pytest.approx(60.0)


def test_a_rotation_about_the_beam_leaves_q_as_it_is() -> None:
    flat, geometry = _hand_case(rot3=0.4)
    assert flat_detector_q_error(flat.tilt, geometry, (1, 1), near_beam_px=0.0).q_error_max == pytest.approx(0.371374, abs=1e-6)


def test_an_untilted_poni_costs_nothing_and_a_valid_mask_and_the_beam_are_respected() -> None:
    tilt = DetectorTilt(0.2, 0.01, 0.02, 0.0, 0.0, 0.0, 1e-4, 1e-4)
    geometry = DetectorGeometry(1e-4, 1e-4, 0.2, 200.0, 100.0, 1.0, 0.1)  # the PONI is the direct beam
    assert flat_detector_q_error(tilt, geometry, (300, 400)).q_error_max == pytest.approx(0.0, abs=1e-12)
    assert parse_poni(_poni_text(0.0, poni1=0.01, poni2=0.02, distance=0.2, pixel=1e-4)).tilt is None
    tilted = DetectorTilt(0.2, 0.01, 0.02, 0.01, 0.0, 0.0, 1e-4, 1e-4)  # 0.57°: below the limit of 1°
    nothing = flat_detector_q_error(tilted, geometry, (300, 400), np.zeros((300, 400), dtype=bool))
    assert nothing.pixels == 0 and nothing.q_error_max is None and not nothing.needs_attention
    assert flat_detector_q_error(DetectorTilt(0.2, 0.01, 0.02, 0.02, 0.0, 0.0, 1e-4, 1e-4), geometry, (300, 400),
                                 np.zeros((300, 400), dtype=bool)).needs_attention  # 1.15°: the tilt alone says so
    near = np.zeros((300, 400), dtype=bool)
    near[100, 200:205] = True  # within 10 px of the direct beam: left out
    assert flat_detector_q_error(tilted, geometry, (300, 400), near).pixels == 0


def test_a_profile_keeps_the_tilt_it_was_read_with_and_a_new_geometry_drops_it() -> None:
    tilt = DetectorTilt(0.2, 0.01, 0.02, 0.02, -0.01, 0.0, 1e-4, 1e-4)
    geometry = DetectorGeometry(1e-4, 1e-4, 0.2, 200.0, 100.0, 1.0, 0.1)
    profile = InstrumentProfile("Detector 300×400", geometry, detector_shape=(300, 400), dropped_tilt=tilt)
    stored = json.loads(json.dumps(profile.to_dict()))
    assert InstrumentProfile.from_dict(stored).dropped_tilt == tilt
    assert "dropped_tilt" not in InstrumentProfile("plain", geometry).to_dict()  # older files stay as they were
    assert InstrumentProfile.from_dict({**stored, "dropped_tilt": None}).dropped_tilt is None
    assert profile.updated(geometry, source="calibration").dropped_tilt is None
    assert profile.updated(geometry, source="beam centre", dropped_tilt=tilt).dropped_tilt == tilt


@pytest.mark.parametrize("key, tilt, largest, attention", [
    ("giwaxs_xeuss_pbte", 6.06, 0.0625, True),  # a lab GIWAXS set-up: the detector tilted on purpose
    ("giwaxs_p08_mapi", 0.31, 0.0020, False),
])
def test_the_public_poni_files_cost_what_the_full_pyfai_geometry_says(key, tilt, largest, attention) -> None:
    manifest = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))
    flat = parse_poni((EXTERNAL / manifest[key]["poni"]).read_text(encoding="utf-8"))
    frame = load_detector_image(EXTERNAL / manifest[key]["frame"])
    shape = frame.data.shape[:2]
    valid = valid_pixels(frame.data, frame.mask)
    geometry = DetectorGeometry(flat.pixel_size_x_m, flat.pixel_size_y_m, flat.distance_mm * 1e-3, flat.beam_center_x_px,
                                flat.beam_center_y_px, flat.wavelength_angstrom, 0.1)
    error = flat_detector_q_error(flat.tilt, geometry, shape, valid)
    assert error.tilt_deg == pytest.approx(tilt, abs=0.01)
    assert error.q_error_max == pytest.approx(largest, rel=0.05)
    assert error.needs_attention is attention
    assert (error.tilt_deg > TILT_LIMIT_DEG or error.q_error_max > Q_ERROR_LIMIT) is attention


def test_a_centre_moved_after_the_poni_is_no_part_of_what_the_tilt_costs() -> None:
    """The public AgBh .poni (0.57° of tilt): its own flat geometry is off by 0.08 % in |q|. The GISAXS symmetry axis
    then moved the centre by −0.39 px, and compared with that geometry in use the "tilt error" was 4.1 % — the
    centre shift, not the tilt. The estimate compares the .poni with the flat geometry made from it."""
    from dataclasses import replace

    from src.gimap.features.analyze.application.flat_detector import flat_detector_error

    manifest = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))
    flat = parse_poni((EXTERNAL / manifest["saxs_agbh_pyfai"]["poni"]).read_text(encoding="utf-8"))
    made = flat.tilt.flat_geometry()  # the same numbers parse_poni gives
    assert (made.beam_center_x_px, made.beam_center_y_px, made.distance_m * 1e3) == pytest.approx(
        (flat.beam_center_x_px, flat.beam_center_y_px, flat.distance_mm), abs=1e-9)
    frame = load_detector_image(EXTERNAL / manifest["saxs_agbh_pyfai"]["frame"])
    shape, valid = frame.data.shape[:2], valid_pixels(frame.data, frame.mask)
    geometry = DetectorGeometry(flat.pixel_size_x_m, flat.pixel_size_y_m, flat.distance_mm * 1e-3, flat.beam_center_x_px,
                                flat.beam_center_y_px, flat.wavelength_angstrom, 0.0)
    profile = InstrumentProfile("AgBh-like", geometry, detector_shape=shape, dropped_tilt=flat.tilt)
    before = flat_detector_error(profile, geometry, shape, valid)
    refined = geometry.with_beam_center(geometry.beam_center_x_px - 0.39, geometry.beam_center_y_px)
    after = flat_detector_error(profile, replace(refined, distance_m=1.6), shape, valid)
    assert before.q_error_max == pytest.approx(0.0008, rel=0.05) and not before.needs_attention
    assert after.q_error_max == pytest.approx(before.q_error_max) and after.needs_attention is before.needs_attention
    assert after.q_error_max == pytest.approx(flat_detector_q_error(flat.tilt, made, shape, valid).q_error_max, rel=1e-6)
    assert flat_detector_q_error(flat.tilt, refined, shape, valid).q_error_max > 0.04  # the centre shift, not the tilt


def test_the_gisaxs_run_on_a_tilted_poni_says_one_estimate_everywhere(tmp_path) -> None:
    """After the symmetry axis moved the centre, Decisions, the Geometry section, report.json and the frame's record
    give the same estimate and the same verdict (it was 0.08 % in Decisions, 4.1 % and an open point elsewhere)."""
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.analyze.application.export_record import analysis_metadata
    from src.gimap.features.assistant.application import PipelineOptions
    from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown

    for item in (EXTERNAL / "saxs_agbh_pyfai").iterdir():
        if item.is_file():
            shutil.copy(item, tmp_path / item.name)
    session = open_headless_session(tmp_path / "Pilatus1M.edf", saved_profiles=False, out_dir=tmp_path / "out")
    try:
        report = session.run_pipeline(PipelineOptions(calibration=str(tmp_path / "Pilatus1M.poni"), energy_kev=12.398,
                                                      incidence_deg=0.0, technique="gisaxs", fit=False))
        record = json.loads(json.dumps(analysis_metadata(session.page.view_model.state.analysis)))
    finally:
        session.close()
    symmetry = report["gisaxs"]["symmetry"]
    assert abs(symmetry["x_px"] - symmetry["initial_x_px"]) > 0.1  # the centre moved after the .poni was applied
    assert report["geometry"]["beam_center_px"][0] == pytest.approx(symmetry["x_px"], abs=1e-6)
    flat = report["geometry"]["flat_detector"]
    assert flat["q_error_max"] == pytest.approx(0.0008, rel=0.05) and not flat["needs_attention"]
    assert not any(item["item"] == "detector tilt" for item in report["needs_attention"])
    decision = next(item for item in report["decisions"] if item["what"] == "detector tilt")
    assert decision["decision"] == "0.57° left out" and "by at most 0.08 % over" in decision["why"]
    assert "|q| off by up to 0.08 % over the valid detector area" in pipeline_markdown(report)
    assert record["geometry"]["flat_detector_q_error_max"] == pytest.approx(flat["q_error_max"], rel=1e-3)
    assert not any("tilted by" in message for message in record["messages"])


def test_the_tilt_is_said_again_from_the_status_the_report_describes() -> None:
    """Frames summed after the geometry (or a mirrored half filled) change the valid pixels: the report's tilt line
    is the estimate of the status it reports, in the place it was first said; above a limit it moves to the points."""
    from types import SimpleNamespace

    from src.gimap.features.assistant.application.pipeline_geometry import GeometryStepsMixin

    def status(error: float, attention: bool) -> dict:
        return {"geometry": {"flat_detector": {"dropped_tilt_deg": 0.9, "q_error_max": error, "needs_attention": attention,
                                               "limits": {"tilt_deg": 1.0, "q_error": 0.005}}}}

    run = SimpleNamespace(decisions=[{"what": "geometry", "decision": "from a.poni", "why": ""}], attention=[])
    pipeline = SimpleNamespace(catalog=SimpleNamespace(results=SimpleNamespace(status=status(0.004, False))), _run=run)
    GeometryStepsMixin._flat_detector(pipeline)
    run.decisions.append({"what": "technique", "decision": "GISAXS", "why": ""})
    pipeline.catalog.results.status = status(0.0041, False)
    GeometryStepsMixin._flat_detector_as_reported(pipeline)
    assert [item["what"] for item in run.decisions] == ["geometry", "detector tilt", "technique"]
    assert "by at most 0.41 %" in run.decisions[1]["why"] and not run.attention
    pipeline.catalog.results.status = status(0.006, True)
    GeometryStepsMixin._flat_detector_as_reported(pipeline)
    assert [item["what"] for item in run.decisions] == ["geometry", "technique"]
    assert [item["item"] for item in run.attention] == ["detector tilt"] and "up to 0.6 %" in run.attention[0]["why"]


@pytest.fixture(scope="module")
def xeuss_run(tmp_path_factory):
    """The public Xeuss frame (copied with its .poni and mask), analysed as the command line does."""
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application import PipelineOptions

    folder = tmp_path_factory.mktemp("xeuss")
    for item in (EXTERNAL / "giwaxs_xeuss_pbte").iterdir():
        shutil.copy(item, folder / item.name)
    session = open_headless_session(folder / "PbTe_3010_r1_PT_1_10_0_00028.edf", saved_profiles=False,
                                    out_dir=folder / "out")
    report = session.run_pipeline(PipelineOptions(calibration=str(folder / "AgBe_250403_XENOCS.poni"), incidence_deg=0.235))
    yield session, report
    session.close()


def test_a_tilted_poni_is_a_point_to_check_with_exit_code_2_and_in_the_record(xeuss_run) -> None:
    import tools.gimap_agent as agent
    from src.gimap.features.analyze.application.export_record import analysis_metadata
    from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown

    session, report = xeuss_run
    item = next(entry for entry in report["needs_attention"] if entry["item"] == "detector tilt")
    assert "tilted by 6.06°" in item["why"] and "up to 6.3 %" in item["why"] and item["option"] is None
    assert agent.exit_code([report]) == 2
    markdown = pipeline_markdown(report)
    assert "Status: **NEEDS INPUT**" in markdown and "detector tilt left out: 6.06°" in markdown
    flat = report["geometry"]["flat_detector"]
    assert flat["needs_attention"] and flat["q_error_max"] == pytest.approx(0.0625, rel=0.02)

    analysis = session.page.view_model.state.analysis
    record = json.loads(json.dumps(analysis_metadata(analysis)))
    geometry = record["geometry"]
    assert geometry["dropped_tilt_deg"] == pytest.approx(6.058, abs=1e-3)
    assert geometry["flat_detector_q_error_max"] == pytest.approx(0.0625, rel=0.02)
    assert geometry["dropped_rotations_deg"]["rot3"] == 0.0
    assert "flat detector perpendicular to the beam" in record["geometry_convention"]
    assert record["instrument_profile"]["source"] == "automatic analysis: AgBe_250403_XENOCS.poni"
    assert any("tilted by 6.06°" in message for message in record["messages"])  # Analyze's status line says it
    assert "tilt 6.06° left out" in session.page.summary_label.text()
    assert "tilt 6.06° left out" in session.page.step_intro["geometry"].text()  # the Geometry step warns
    assert datetime.fromisoformat(record["created"]).utcoffset().total_seconds() == 0
    assert record["error_model"] == "poisson"
    binning = record["binning"]
    assert binning["radial_bins"] >= binning["radial_points"] > 100 and binning["bin_width"] > 0
    assert binning["bin_width"] == pytest.approx((binning["q_max"] - binning["q_min"]) / binning["radial_bins"])


def test_the_tilt_texts_are_shown_in_chinese_with_their_numbers() -> None:
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.analyze.application.flat_detector import FLAT_DETECTOR_MESSAGE
    from src.gimap.features.analyze.presentation.texts import message_text
    from src.gimap.features.assistant.application.pipeline_geometry import TILT_HINT, TILT_SMALL, TILT_WHY
    from src.gimap.features.assistant.presentation.guided_words import words

    apply_language("zh")
    try:
        shown = message_text(FLAT_DETECTOR_MESSAGE.format(tilt="6.06", error="6.3"))
        assert "6.06°" in shown and "6.3 %" in shown and "倾斜" in shown
        point = words(TILT_WHY.format(tilt="6.06", error="6.3"))
        assert "6.06°" in point and "有效区域" in point
        assert "pyFAI" in words(TILT_HINT) and words(TILT_HINT) != TILT_HINT
        decision = words(TILT_SMALL.format(error="0.2", limit_deg="1", limit_error="0.5"))
        assert "0.2 %" in decision and "1°" in decision and decision.startswith("GIMaP 的几何")
        assert words("0.31° left out") == "略去 0.31°"
    finally:
        apply_language(DEFAULT_LANGUAGE)
