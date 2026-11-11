"""Synthetic .poni rotations: a correct |q| magnitude does not establish its direction."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.shared.geometry.detector_tilt import DetectorTilt, flat_detector_q_error, tilted_rays


def _estimate(rotation: float, *, tilt: float = 0.0, valid=None):
    detector = DetectorTilt(0.2, 0.004, 0.004, math.radians(tilt), 0.0,
                            math.radians(rotation), 1e-4, 1e-4)
    return flat_detector_q_error(detector, detector.flat_geometry(), (80, 80), valid)


def test_pure_rot3_warns_about_direction_without_inventing_a_q_error():
    detector = DetectorTilt(0.2, 0.004, 0.004, 0.0, 0.0, math.pi / 6, 1e-4, 1e-4)
    unrotated = DetectorTilt(0.2, 0.004, 0.004, 0.0, 0.0, 0.0, 1e-4, 1e-4)
    before = tilted_rays(unrotated, np.array([20]), np.array([60]))
    after = tilted_rays(detector, np.array([20]), np.array([60]))
    angle = math.degrees(float(np.arctan2(after[1], after[0])[0] - np.arctan2(before[1], before[0])[0]))
    assert angle == pytest.approx(30.0)
    estimate = _estimate(30.0)
    assert estimate.q_error_max == pytest.approx(0.0, abs=1e-12)
    assert estimate.needs_attention, "A 30° direction error must not pass because |q| is unchanged"
    assert not estimate.q_needs_attention and estimate.direction_needs_attention
    stored = estimate.to_dict()
    assert stored["dropped_rotations_deg"] == pytest.approx({"rot1": 0.0, "rot2": 0.0, "rot3": 30.0})
    assert stored["dropped_beam_axis_rotation_deg"] == pytest.approx(30.0)


@pytest.mark.parametrize("rotation, normalized, attention", [
    (0.0, 0.0, False), (360.0, 0.0, False), (-720.0, 0.0, False),
    (390.0, 30.0, True), (-390.0, -30.0, True), (180.0, 180.0, True),
    (360.0 + 1e-12, 0.0, False), (2e-6, 2e-6, False), (0.5, 0.5, False),
    (1.0, 1.0, False), (1.01, 1.01, True),
])
def test_rot3_is_periodic_and_small_rotations_remain_recorded(rotation, normalized, attention):
    estimate = _estimate(rotation)
    assert estimate.beam_axis_rotation_deg == pytest.approx(normalized)
    assert estimate.direction_needs_attention is attention
    assert estimate.to_dict()["dropped_rotations_deg"]["rot3"] == pytest.approx(rotation)


def test_direction_is_known_even_without_valid_pixels_and_combines_with_normal_tilt():
    empty = _estimate(-30.0, valid=np.zeros((80, 80), dtype=bool))
    assert empty.q_error_max is None and empty.direction_needs_attention
    both = _estimate(30.0, tilt=5.0)
    normal_only = _estimate(0.0, tilt=5.0)
    assert both.q_needs_attention and both.direction_needs_attention
    assert both.q_error_max == pytest.approx(normal_only.q_error_max, abs=1e-12)


def test_direction_and_q_review_entries_refresh_independently():
    from src.gimap.features.assistant.application.pipeline_geometry import (
        GeometryStepsMixin, ROTATION_ITEM, TILT_ITEM,
    )

    run = SimpleNamespace(decisions=[], attention=[])
    status = {"geometry": {"flat_detector": _estimate(30.0, tilt=5.0).to_dict()}}
    pipeline = SimpleNamespace(catalog=SimpleNamespace(results=SimpleNamespace(status=status)), _run=run)
    GeometryStepsMixin._flat_detector(pipeline)
    assert [entry["item"] for entry in run.attention] == [TILT_ITEM, ROTATION_ITEM]
    status["geometry"]["flat_detector"] = _estimate(30.0).to_dict()
    GeometryStepsMixin._flat_detector_as_reported(pipeline)
    assert [entry["item"] for entry in run.attention] == [ROTATION_ITEM]
    assert [entry["what"] for entry in run.decisions] == [TILT_ITEM]
    status["geometry"]["flat_detector"] = _estimate(0.5).to_dict()
    GeometryStepsMixin._flat_detector_as_reported(pipeline)
    assert not run.attention
    assert [entry["what"] for entry in run.decisions] == [TILT_ITEM, ROTATION_ITEM]
    assert "Rot3 = 0.5°" in run.decisions[1]["why"]


def test_synthetic_rot3_is_visible_in_analyze_and_all_report_exports(tmp_path):
    from PIL import Image

    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.analyze.application.export_record import analysis_metadata
    from src.gimap.features.assistant.application import PipelineOptions
    from src.gimap.features.assistant.application.pipeline_geometry import ROTATION_ITEM, TILT_ITEM
    from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown
    from tools.gimap_agent import exit_code

    frame = tmp_path / "synthetic_ring.tif"
    rows, columns = np.indices((80, 80))
    counts = 10 + 200 * np.exp(-((np.hypot(rows - 39.5, columns - 39.5) - 25) / 2)**2)
    Image.fromarray(counts.astype(np.float32)).save(frame)
    calibration = tmp_path / "synthetic_geometry.poni"
    calibration.write_text(
        'poni_version: 2\nDetector: Detector\nDetector_config: {"pixel1": 0.0001, "pixel2": 0.0001}\n'
        f"Distance: 0.2\nPoni1: 0.004\nPoni2: 0.004\nRot1: 0\nRot2: 0\nRot3: {math.pi / 6}\n"
        "Wavelength: 1e-10\n", encoding="utf-8",
    )
    session = open_headless_session(frame, saved_profiles=False, out_dir=tmp_path / "out")
    try:
        report = session.run_pipeline(PipelineOptions(calibration=str(calibration), incidence_deg=0.2,
                                                      technique="giwaxs", stop_after_geometry=True))
        record = analysis_metadata(session.page.view_model.state.analysis)
        summary = session.page.summary_label.text()
        step = session.page.step_rail.detail("geometry")
        assert session.page.step_rail.state("geometry") == "warn"
    finally:
        session.close()
    assert "Rot3 30°" in summary and "χ and qy/qz directions affected" in summary
    assert "Rot3 30°" in step and "|q| up to" not in step
    assert report["geometry"]["flat_detector"]["q_error_max"] == pytest.approx(0.0, abs=1e-12)
    assert any(entry["item"] == ROTATION_ITEM for entry in report["needs_attention"])
    assert not any(entry["item"] == TILT_ITEM for entry in report["needs_attention"])
    assert exit_code([report]) == 2
    markdown = pipeline_markdown(report)
    assert "Rot3 = 30°" in markdown and "Rot3 alone does not change |q|" in markdown
    geometry = record["geometry"]
    assert geometry["dropped_rotations_deg"] == pytest.approx({"rot1": 0.0, "rot2": 0.0, "rot3": 30.0})
    assert geometry["dropped_beam_axis_rotation_deg"] == pytest.approx(30.0)
    assert geometry["flat_detector_direction_needs_attention"] and not geometry["flat_detector_q_needs_attention"]
    assert any("in-plane/out-of-plane" in message for message in record["messages"])
    assert not any("tilted by 0.00°" in message for message in record["messages"])
