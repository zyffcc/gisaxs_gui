"""The automatic search ranks ready calibration files by folder first, then by file time.

Help ▸ Example Data copies each example into a folder of its own beside the others
(``Documents/GIMaP examples/<example>/``), and a copy, a download or a clone leaves file times that say
nothing. Ranked by time alone, the GISAXS frame of one example took the GIWAXS ``.poni`` of the example
next to it and was analysed as GIWAXS. A calibration in the frame's own folder now comes first, then one in a
subfolder, the parent folder, a sibling folder, and further up; the file time decides within each.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import PipelineOptions, RunResults, StandardPipeline, ToolCatalog
from src.gimap.features.assistant.application.pipeline_geometry import SEARCHED
from src.gimap.features.assistant.domain import FAR_FOLDER, folder_place
from src.gimap.features.assistant.infrastructure import LocalFileExplorer
from tests.test_agent_tooling_pipeline import _decisions, _goals
from tests.assistant_fakes import call
from tests.test_assistant_calibration import CalibrationWorkbench, FakeCalibrator, payload, touch

DATA = Path(__file__).parent / "data" / "external"


def _poni(distance_m: float) -> str:
    return "\n".join([
        "poni_version: 2", "Detector: Detector",
        'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [400, 400]}',
        f"Distance: {distance_m}", "Poni1: 0.025", "Poni2: 0.02", "Rot1: 0", "Rot2: 0", "Rot3: 0",
        "Wavelength: 1e-10",
    ])


def _at(path: Path, moment: float) -> Path:
    os.utime(path, (moment, moment))
    return path


@pytest.mark.parametrize("path, place", [
    ("D:/runs/sample_a/cal.poni", (0, 0)),
    ("D:/runs/sample_a/calib/old/cal.poni", (0, 1)),
    ("D:/runs/cal.poni", (1, 0)),
    ("D:/runs/sample_b/cal.poni", (1, 1)),
    ("D:/runs/sample_b/deeper/cal.poni", (1, 1)),
    ("D:/cal.poni", (2, 0)),
    ("D:/calib/cal.poni", (2, 1)),
    ("E:/elsewhere/cal.poni", FAR_FOLDER),
])
def test_folder_place_orders_the_folders_around_the_frame(path: str, place: tuple) -> None:
    assert folder_place("D:/runs/sample_a", path) == place
    assert folder_place("D:\\runs\\sample_a", path.replace("/", "\\")) == place


def _examples(tmp_path: Path) -> Path:
    """``Documents/GIMaP examples`` with nothing else within three folders of it."""
    return tmp_path / "home" / "Documents" / "GIMaP examples"


def test_a_calibration_beside_the_frame_comes_before_the_neighbours_however_old(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    frame = touch(examples / "sample_a" / "film_0001.tif", "x")
    moment = frame.stat().st_mtime
    # The neighbour's file is the nearest in time, the frame's own the furthest (as a copy may leave them).
    own = _at(touch(examples / "sample_a" / "film_geometry.poni", _poni(0.1)), moment + 86400)
    later_own = _at(touch(examples / "sample_a" / "film_geometry_2.poni", _poni(0.1)), moment + 2 * 86400)
    sub = _at(touch(examples / "sample_a" / "calib" / "sub.poni", _poni(0.1)), moment + 3600)
    parent = _at(touch(examples / "shared.poni", _poni(0.1)), moment - 60)
    sibling = _at(touch(examples / "sample_b" / "LaB6_neighbour.poni", _poni(0.2)), moment - 1)
    catalog = ToolCatalog(CalibrationWorkbench(frame), _goals(), RunResults(), explorer=LocalFileExplorer())
    catalog.execute(call("get_status"))

    found = payload(catalog.execute(call("find_calibration_files")))

    assert [Path(item["path"]).name for item in found["calibration_results"]] == [
        own.name, later_own.name, sub.name, parent.name, sibling.name]
    assert found["calibration_results"][-1]["folder"] == "sibling folder sample_b"
    assert found["ranking"].startswith("calibration files are ranked by folder")


def test_two_sibling_examples_each_get_their_own_calibration(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    frames, ponis = {}, {}
    for name, distance in (("gisaxs_example", 0.1), ("giwaxs_example", 0.2)):
        frames[name] = touch(examples / name / f"{name}_frame.tif", "x")
        ponis[name] = touch(examples / name / f"{name}.poni", _poni(distance))
    # Each frame finds the other example's calibration the nearest in time.
    for name, other in (("gisaxs_example", "giwaxs_example"), ("giwaxs_example", "gisaxs_example")):
        moment = frames[name].stat().st_mtime
        _at(ponis[name], moment + 7200)
        _at(ponis[other], moment - 1)
        workbench = CalibrationWorkbench(frames[name])
        catalog = ToolCatalog(workbench, _goals(), RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator())

        report = StandardPipeline(catalog, PipelineOptions(follow_detection=True, stop_after_geometry=True)).run()

        assert _decisions(report, "geometry")[0]["decision"] == f"from {name}.poni", report["decisions"]
        (searched,) = _decisions(report, SEARCHED)
        assert searched["decision"] == str(ponis[name]) and searched["why"].startswith("same folder")
        _tool, values, _profile, _source = workbench.calls[-1]
        assert values["distance_mm"] == pytest.approx(1000 * (0.1 if name == "gisaxs_example" else 0.2))


@pytest.mark.parametrize("example, poni, measurement", [
    ("gisaxs_galaxi", "galaxi_bornagain.poni", "gisaxs"),
    ("giwaxs_p08_mapi", "LaB6_2021_12_DESY_P08.poni", "giwaxs"),
])
def test_the_public_examples_side_by_side_each_take_their_own_poni(tmp_path: Path, example, poni, measurement) -> None:
    """GALAXI (GISAXS, Pilatus 1M) and P08 (GIWAXS) copied side by side, as Help ▸ Example Data does: neither
    TIFF states a pixel size, so only the folder tells the two .poni files apart."""
    from src.gimap.app.headless_assistant import open_headless_session

    sources = {name: DATA / name for name in ("gisaxs_galaxi", "giwaxs_p08_mapi")}
    if not all(folder.is_dir() for folder in sources.values()):
        pytest.skip("the public examples are not in this checkout")
    examples = _examples(tmp_path)
    for name, folder in sources.items():
        (examples / name).mkdir(parents=True)
        for path in folder.iterdir():
            if path.is_file():
                shutil.copy2(path, examples / name / path.name)
    frame = next(path for path in (examples / example).iterdir() if path.suffix == ".tif")
    moment = frame.stat().st_mtime
    other = next(path for name in sources if name != example for path in (examples / name).glob("*.poni"))
    _at(other, moment - 1)  # the neighbour's calibration the nearest in time
    _at(examples / example / poni, moment + 86400)

    session = open_headless_session(frame, saved_profiles=False, out_dir=tmp_path / "out")
    try:
        report = session.run_pipeline(PipelineOptions(stop_after_geometry=True))
    finally:
        session.close()

    assert report["geometry"]["source"] == poni, report["decisions"]
    assert report["measurement"] == measurement
    (searched,) = _decisions(report, SEARCHED)
    assert Path(searched["decision"]).name == poni and searched["why"].startswith("same folder")
