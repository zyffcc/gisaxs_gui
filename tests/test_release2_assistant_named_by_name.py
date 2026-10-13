"""A calibration the notes name by its file name alone is the one used (release 2).

Notes often say ``calibration LaB6_take2.poni`` without a path. That file used to be left to the automatic
search, which ranks ready calibration files by how close in time they are to the frame: in a folder that was
copied or cloned all files are a few milliseconds apart, and a calibration of a neighbouring data set could
win. A name is now looked up in the frame's folder and in the folders the notes name, nowhere else.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import tools.eval_giwaxs_agent as tool
from src.gimap.features.assistant.application import PipelineOptions, StandardPipeline
from src.gimap.features.assistant.application.pipeline_geometry import NAMED, SEARCHED
from src.gimap.features.assistant.domain import file_name_mentions
from tests.test_agent_tooling_pipeline import _decisions, _series
from tests.test_assistant_calibration import POOR_AGBH, touch

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "tests" / "data" / "external"


@pytest.mark.parametrize("text, names", [
    ("GIWAXS, alpha_i = 0.075 deg, calibration abc_standard_take2.poni", ["abc_standard_take2.poni"]),
    ("12.4 keV, αi 0.2°; use geometry_take2.poni. Then sum 10 frames", ["geometry_take2.poni"]),
    ("标定文件是abc_redone.poni，能量 12.4 keV", ["abc_redone.poni"]),
    (r"Geometry: D:\beamtime\calib\final.poni and calib/agbh_0001.tif", ["final.poni", "agbh_0001.tif"]),
    ("E = 11.5 keV, 0.15 deg, pixel 55.0 um, v1.2.3, see notes.txt, old.json.bak", []),
    ("AgBH.TIF, AgBH.tif and agbh_(2).cbf", ["AgBH.TIF", "agbh_(2).cbf"]),
    ("", []),
])
def test_file_names_written_in_notes(text: str, names: list[str]) -> None:
    assert file_name_mentions(text) == names


def test_a_calibration_named_by_its_name_is_taken_from_the_frames_folder(tmp_path: Path) -> None:
    notes = "12.4 keV, alpha_i = 0.2 deg, calibration geometry_take2.poni"
    catalog, workbench, calibrator = _series(tmp_path, notes)
    frame = Path(workbench.status()["path"])
    named = touch(frame.parent / "geometry_take2.poni", POOR_AGBH)
    other = touch(frame.parent.parent / "calib" / "another_sample.poni", POOR_AGBH)
    # The search would prefer the other one: written just before the frame, the named one long after.
    moment = frame.stat().st_mtime
    os.utime(other, (moment - 1, moment - 1))
    os.utime(named, (moment + 86400, moment + 86400))

    report = StandardPipeline(catalog, PipelineOptions(notes=notes, follow_detection=True)).run()

    assert report["ok"] and report["needs_attention"] == [], report["needs_attention"]
    assert _decisions(report, "calibration candidates") == [
        {"what": "calibration candidates", "decision": "geometry_take2.poni", "why": NAMED}]
    assert _decisions(report, "geometry")[0]["decision"] == "from geometry_take2.poni"
    assert not _decisions(report, SEARCHED)  # nobody had to search: the person named it
    assert "find_calibration_files" not in [step["tool"] for step in report["steps"]] and calibrator.calls == []


def test_a_name_is_also_looked_up_in_a_folder_the_notes_name(tmp_path: Path) -> None:
    folder = tmp_path / "elsewhere" / "final"
    touch(folder / "geometry_take2.poni", POOR_AGBH)
    notes = f"12.4 keV, alpha_i = 0.2 deg. Calibrations are in {folder} - use geometry_take2.poni"
    catalog, _workbench, _calibrator = _series(tmp_path, notes)
    assert catalog.named_calibration_files(notes) == [str(folder / "geometry_take2.poni")]
    report = StandardPipeline(catalog, PipelineOptions(notes=notes, follow_detection=True)).run()
    assert _decisions(report, "geometry")[0]["decision"] == "from geometry_take2.poni"
    assert not _decisions(report, SEARCHED)


def test_a_name_starts_no_search_and_names_nothing_but_calibrations(tmp_path: Path) -> None:
    catalog, workbench, _calibrator = _series(tmp_path)
    frame = Path(workbench.status()["path"])
    # A file of that name exists, but neither in the frame's folder nor in a folder the notes name.
    touch(frame.parent.parent / "calib" / "geometry_take2.poni", POOR_AGBH)
    assert catalog.named_calibration_files("use geometry_take2.poni") == []
    # The open frame and an ordinary frame beside it are no calibration, however they are written.
    touch(frame.parent / "film_00002_m01.nxs", "x")
    assert catalog.named_calibration_files(f"frames {frame.name} and film_00002_m01.nxs") == []
    # Written once with its path and once by its name: one candidate.
    here = touch(frame.parent / "geometry_take2.poni", POOR_AGBH)
    assert catalog.named_calibration_files(f"Geometry: {here}; that is geometry_take2.poni") == [str(here)]
    assert catalog.named_calibration_files("GEOMETRY_TAKE2.PONI")[0].lower() == str(here).lower()


def test_the_named_calibration_wins_where_file_times_say_nothing(tmp_path: Path, capsys) -> None:
    """The public P08 frame beside the public Xeuss set, as a clone or a copy leaves them: all files written
    within a moment, the calibration of the other set the nearest in time."""
    if not (EXTERNAL / "giwaxs_p08_mapi" / "S121_MAI_A2_00841.tif").is_file():
        pytest.skip("the public GIWAXS frames are not in this checkout")
    data = tmp_path / "data"
    for name in ("giwaxs_p08_mapi", "giwaxs_xeuss_pbte"):
        shutil.copytree(EXTERNAL / name, data / name)
    frame = data / "giwaxs_p08_mapi" / "S121_MAI_A2_00841.tif"
    moment = frame.stat().st_mtime
    os.utime(data / "giwaxs_p08_mapi" / "LaB6_2021_12_DESY_P08.poni", (moment - 0.09, moment - 0.09))
    os.utime(data / "giwaxs_xeuss_pbte" / "AgBe_250403_XENOCS.poni", (moment + 0.003, moment + 0.003))
    case = tmp_path / "public-frame.json"
    case.write_text(json.dumps({
        "name": "Public: the P08 frame beside another data set",
        "data_root": str(data / "giwaxs_p08_mapi"),
        "frames": [frame.name],
        "notes": "GIWAXS, alpha_i = 0.075 deg, calibration LaB6_2021_12_DESY_P08.poni",
        "baseline": {"geometry_source_contains": "LaB6_2021_12_DESY_P08.poni"},
    }), encoding="utf-8")
    tool.main(["--case", str(case), "baseline", "--out", str(tmp_path / "out")])
    printed = capsys.readouterr().out
    assert "PASS  calibration file: LaB6_2021_12_DESY_P08.poni" in printed, printed[-600:]
    report = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))[0]
    assert not [item for item in report["decisions"] if item["what"] == SEARCHED]
