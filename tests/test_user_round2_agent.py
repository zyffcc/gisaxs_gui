"""``tools/gimap_agent.py auto`` against what a user test found: values no analysis can use are refused before
anything is read, and a calibration that is not the frame's is neither adopted silently nor reported as OK."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import PipelineOptions
from src.gimap.features.assistant.application.calibration_match import (
    SEARCHED_ITEM,
    detector_shape,
    shape_mismatch,
)

DATA = Path(__file__).parent / "data" / "external"
GALAXI = DATA / "gisaxs_galaxi"
XEUSS_PONI = DATA / "giwaxs_xeuss_pbte" / "AgBe_250403_XENOCS.poni"


def _refused(monkeypatch, capsys, *arguments: str) -> str:
    from src.gimap.app import headless_assistant
    from tools import gimap_agent

    called = []
    monkeypatch.setattr(headless_assistant, "analyse_frames", lambda *a, **k: called.append(a) or [])
    code = gimap_agent.main(["auto", str(GALAXI / "galaxi_data.tif"), "--quiet", *arguments])
    assert code == 1 and not called
    return capsys.readouterr().err


@pytest.mark.parametrize("value", ["-0.4", "0", "90", "95", "nan"])
def test_an_incidence_angle_outside_0_to_90_is_an_argument_error(monkeypatch, capsys, tmp_path, value) -> None:
    error = _refused(monkeypatch, capsys, f"--incidence-deg={value}", "--out", str(tmp_path / "out"))
    assert "--incidence-deg" in error and "above 0° and below 90°" in error
    assert not (tmp_path / "out").exists()


def test_a_calibration_file_that_does_not_exist_is_an_argument_error(monkeypatch, capsys, tmp_path) -> None:
    missing = tmp_path / "nope.poni"
    error = _refused(monkeypatch, capsys, "--calibration", str(missing), "--out", str(tmp_path / "out"))
    assert "No such calibration file" in error and "nope.poni" in error


def test_a_valid_incidence_angle_is_passed_on(monkeypatch, tmp_path) -> None:
    from src.gimap.app import headless_assistant
    from tools import gimap_agent

    seen = {}
    monkeypatch.setattr(headless_assistant, "analyse_frames",
                        lambda frames, options, out, **kwargs: seen.update(options=options) or [])
    gimap_agent.main(["auto", str(GALAXI / "galaxi_data.tif"), "--quiet", "--incidence-deg", "0.463",
                      "--calibration", str(GALAXI / "galaxi_bornagain.poni"), "--out", str(tmp_path / "out")])
    assert seen["options"].incidence_deg == 0.463


def test_a_poni_that_names_another_detector_states_its_shape() -> None:
    assert detector_shape("Pilatus300k") == (619, 487) and detector_shape("Pilatus 1M") == (1043, 981)
    assert detector_shape("Eiger2 CdTe 1M") == (1062, 1028) and detector_shape("Perkin") is None
    frame = {"shape": [1043, 981], "path": "frame.tif"}
    why = shape_mismatch("x.poni", {"detector": "Pilatus300k"}, frame, XEUSS_PONI.read_text(encoding="utf-8"))
    assert why.startswith("the calibration does not match this frame") and "[619, 487]" in why and "Pilatus300k" in why
    assert shape_mismatch("x.poni", {"detector": "Pilatus1M"}, frame, "") is None
    assert shape_mismatch("x.poni", {"detector": "Pilatus1M"}, {"shape": [981, 1043]}, "") is None  # either orientation
    assert shape_mismatch("x.poni", {"detector": "Detector"}, frame, "") is None  # states nothing: not refused


def _alone(tmp_path: Path, poni: Path, name: str = "found.poni") -> Path:
    """The public GALAXI frame alone in a folder, a calibration file in the folder above it."""
    folder = tmp_path / "beamtime" / "run1"
    folder.mkdir(parents=True)
    shutil.copy(poni, folder.parent / name)
    return Path(shutil.copy(GALAXI / "galaxi_data.tif", folder / "galaxi_data.tif"))


def _run(frame: Path, out: Path, **options) -> dict:
    from src.gimap.app.headless_assistant import analyse_frames

    options = PipelineOptions(incidence_deg=0.463, fit=False, **options)
    [report] = analyse_frames([str(frame)], options, out, saved_profiles=False)
    return report


def test_a_searched_calibration_of_another_detector_is_not_used(tmp_path: Path) -> None:
    frame = _alone(tmp_path, XEUSS_PONI)
    report = _run(frame, tmp_path / "out")
    skipped = [item for item in report["decisions"] if item["decision"] == "skipped found.poni"]
    assert skipped and "[619, 487]" in skipped[0]["why"] and "[1043, 981]" in skipped[0]["why"]
    assert "found.poni" not in str(report.get("geometry_source") or "")
    assert report["needs_attention"]  # never plain OK


def test_a_given_calibration_of_another_detector_is_said_not_to_match(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    frame = Path(shutil.copy(GALAXI / "galaxi_data.tif", folder / "galaxi_data.tif"))
    report = _run(frame, tmp_path / "out", calibration=str(XEUSS_PONI))
    whys = " ".join(item["why"] for item in report["needs_attention"])
    assert "does not match this frame" in whys and "Pilatus300k" in whys
    assert XEUSS_PONI.name not in str(report.get("geometry_source") or "")


def test_a_calibration_found_in_the_parent_folder_is_an_open_item_with_its_path(tmp_path: Path) -> None:
    from tools import gimap_agent

    frame = _alone(tmp_path, GALAXI / "galaxi_bornagain.poni")
    report = _run(frame, tmp_path / "out", technique="gisaxs")
    found = frame.parent.parent / "found.poni"
    [item] = [item for item in report["needs_attention"] if item["item"] == SEARCHED_ITEM]
    assert str(found) in item["why"] and item["option"] == "calibration"
    assert gimap_agent.exit_code([report]) == 2
    text = (tmp_path / "out" / "galaxi_data" / "report.md").read_text(encoding="utf-8")
    assert "Status: **OK**" not in text and str(found) in text.split("## Geometry")[0]


def test_a_calibration_beside_the_frame_or_given_is_no_open_item(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    frame = Path(shutil.copy(GALAXI / "galaxi_data.tif", folder / "galaxi_data.tif"))
    report = _run(frame, tmp_path / "out", technique="gisaxs", calibration=str(GALAXI / "galaxi_bornagain.poni"))
    assert not [item for item in report["needs_attention"] if item["item"] == SEARCHED_ITEM]
    assert report["gisaxs"]["symmetry"].get("not_applied") is None


def test_a_symmetry_axis_hundreds_of_pixels_away_is_not_applied(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    frame = Path(shutil.copy(GALAXI / "galaxi_data.tif", folder / "galaxi_data.tif"))
    text = (GALAXI / "galaxi_bornagain.poni").read_text(encoding="utf-8")
    assert "Poni2: 0.1027012" in text
    shifted = tmp_path / "shifted.poni"  # the beam-centre column 300 px to the left of the documented one
    shifted.write_text(text.replace("Poni2: 0.1027012", "Poni2: 0.0511012"), encoding="utf-8")
    report = _run(frame, tmp_path / "out", technique="gisaxs", calibration=str(shifted))
    symmetry = report["gisaxs"]["symmetry"]
    assert symmetry["not_applied"] is True and abs(symmetry["shift_px"]) > 100
    assert abs(report["geometry"]["beam_center_px"][0] - 297.1) < 0.01  # the calibrated centre, not the refined one
    whys = [item["why"] for item in report["needs_attention"] if item["item"] == "calibration"]
    assert any(why.startswith("The calibration does not match this frame") for why in whys)
    markdown = (tmp_path / "out" / "galaxi_data" / "report.md").read_text(encoding="utf-8")
    assert "**not applied**" in markdown and "Status: **OK**" not in markdown
