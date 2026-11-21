"""One list of readable image endings: a standard image in ``.edf`` (which the Calibration window fits) is fitted
by the tools and the command line too."""

from __future__ import annotations

from pathlib import Path

from src.gimap.features.assistant.application import READABLE_IMAGE_SUFFIXES
from src.gimap.features.assistant.application import calibration_tools
from src.gimap.features.assistant.domain.calibration_files import IMAGE_SUFFIXES
from tests.assistant_fakes import call
from tests.test_assistant_calibration import ENERGY_KEV, beamtime, catalog_for, payload, touch


def test_there_is_one_list_of_readable_image_endings() -> None:
    assert READABLE_IMAGE_SUFFIXES is IMAGE_SUFFIXES
    assert not hasattr(calibration_tools, "IMAGE_ENDINGS")  # the tools' own, shorter list is gone
    assert ".edf" in READABLE_IMAGE_SUFFIXES


def test_a_standard_image_in_edf_is_calibrated_from(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    standard = touch(paths["frame"].parent.parent / "calib" / "AgBh_00003.edf", "x")
    catalog, _workbench, calibrator, _results = catalog_for(paths)
    listed = payload(catalog.execute(call("find_calibration_files")))
    entry = next(item for item in listed["standard_images"] if str(item.get("path", "")).endswith(standard.name))
    assert entry["gimap_can_fit"] is True
    fitted = payload(catalog.execute(call("calibrate_geometry", path=str(standard), standard="agbh", energy_kev=ENERGY_KEV)))
    assert calibrator.calls[-1][0] == standard.name and fitted
