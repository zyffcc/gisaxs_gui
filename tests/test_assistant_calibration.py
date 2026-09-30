"""A frame without geometry: the assistant finds, checks and uses calibration material.

Covers the name and file recognition, the bounded folder search, the tools
(whitelist, confirmation, choices), GIMaP's real calibration engine on a
synthetic AgBH image, and a whole run in Analyze that starts without any
instrument profile.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

from src.gimap.features.assistant.application import (
    PERMISSION_AUTO,
    PERMISSION_CONFIRM,
    AnalysisGoals,
    RunResults,
    ToolCatalog,
)
from src.gimap.features.assistant.domain import classify, parse_poni, standard_from_name, text_hints
from src.gimap.features.assistant.infrastructure import LocalFileExplorer
from tests.assistant_fakes import FakeWorkbench, call

SHAPE = (400, 400)
PIXEL = 100e-6
DISTANCE_M = 0.100
CENTER = (200.0, 250.0)  # canonical pixels
WAVELENGTH = 1.0
ENERGY_KEV = 12.398419843320026 / WAVELENGTH
AGBH_Q1 = 2 * math.pi / 58.38


def agbh_image(seed: int = 3) -> np.ndarray:
    rows, columns = np.indices(SHAPE)
    radius = np.hypot((columns + 0.5 - CENTER[0]) * PIXEL, (rows + 0.5 - CENTER[1]) * PIXEL)
    q = 4 * math.pi * np.sin(np.arctan2(radius, DISTANCE_M) / 2) / WAVELENGTH
    image = 20.0 + 400.0 * np.exp(-q / 0.05)
    for order in range(1, 16):
        image += (600.0 / order) * np.exp(-0.5 * ((q - AGBH_Q1 * order) / 0.004) ** 2)
    return np.random.default_rng(seed).poisson(image).astype(np.float32)


def save_tiff(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data.astype(np.float32), mode="F").save(path)
    return path


def touch(path: Path, text: str = "", hours: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    moment = time.time() + hours * 3600
    os.utime(path, (moment, moment))
    return path


POOR_AGBH = "\n".join([
    "poni_version: 2", "Detector: Detector",
    'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [400, 400]}',
    "Distance: 0.1", "Poni1: 0.025", "Poni2: 0.02", "Rot1: 0", "Rot2: 0", "Rot3: 0", "Wavelength: 1e-10",
])


def beamtime(tmp_path: Path) -> dict:
    """raw/sample_A/frame (the frame), raw/calib/{agbh,lab6}, raw/logs, processed/agbh.poni, a .fio two levels up."""
    root = tmp_path / "beamtime"
    frame = touch(root / "raw" / "sample_A" / "P3HT_00012.tif", "x")
    touch(root / "raw" / "sample_A" / "P3HT_00013.tif", "x")
    paths = {
        "frame": frame,
        "agbh_before": touch(root / "raw" / "calib" / "AgBH_00001.tif", "x", hours=-3),
        "agbh_old": touch(root / "raw" / "calib" / "agbe_2025_00002.tif", "x", hours=-400),
        "lab6": touch(root / "raw" / "calib" / "LaB6_00001.tif", "x", hours=-2),
        "log": touch(root / "raw" / "logs" / "beamtime.log", "energy = 12.4 keV\nsample to detector distance: 100 mm\nfilter 3\n"),
        "poni": touch(root / "processed" / "agbh.poni", POOR_AGBH),
        "fio": touch(root / "eh1_scan_00012.fio", "%p\nom = 0.2\n"),
        "unrelated": touch(root / "raw" / "other" / "sample_00001.tif", "x"),
    }
    return paths


def test_file_names_say_which_standard_and_what_kind_of_file() -> None:
    assert standard_from_name("raw/calib/AgBH_00001.tif") == "agbh"
    assert standard_from_name("silver_behenate_det2.cbf") == "agbh"
    assert standard_from_name("LaB6_scan.cbf") == "lab6" and standard_from_name("CeO2.tif") == "ceo2"
    assert standard_from_name("P3HT_on_Si_0p2deg.cbf") is None  # a substrate, not a standard
    assert standard_from_name("Si_powder_std.cbf") == "si"
    assert standard_from_name("lab_notes.txt") is None
    assert classify("processed/agbh.poni") == "poni"
    assert classify("calibration_2025.json") == "gimap_calibration"
    assert classify("raw/calib/img_00001.cbf") == "standard_image"
    assert classify("raw/sample/img_00001.cbf") is None
    assert classify("eh1_scan_00012.fio") == "log" and classify("beamtime.log") == "log"


def test_poni_files_give_the_direct_beam_geometry() -> None:
    flat = parse_poni(POOR_AGBH)
    assert flat.distance_mm == pytest.approx(100.0)
    assert (flat.beam_center_x_px, flat.beam_center_y_px) == pytest.approx((200.0, 250.0))
    assert flat.wavelength_angstrom == pytest.approx(1.0) and flat.tilt_deg == pytest.approx(0.0)
    tilted = parse_poni(POOR_AGBH.replace("Rot1: 0", "Rot1: 0.02"))
    # pyFAI Geometry.getFit2D: the direct beam is at poni2 − L·tan(rot1) (and poni1 + L·tan(rot2)/cos(rot1)).
    assert tilted.beam_center_x_px == pytest.approx(200.0 - 0.1 * math.tan(0.02) / 1e-4)
    assert tilted.distance_mm > 100.0 and tilted.notes
    both = parse_poni(POOR_AGBH.replace("Rot1: 0", "Rot1: 0.02").replace("Rot2: 0", "Rot2: 0.01"))
    assert both.beam_center_y_px == pytest.approx(250.0 + 0.1 * math.tan(0.01) / math.cos(0.02) / 1e-4)
    named = parse_poni(POOR_AGBH.replace('Detector: Detector', "Detector: Pilatus1M").replace(
        'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [400, 400]}', "Detector_config: {}"))
    assert named.pixel_size_x_m == pytest.approx(172e-6)  # a named detector brings its pixel size
    with pytest.raises(ValueError):
        parse_poni("Distance: 0.1")


def test_log_lines_about_the_geometry_are_picked_out() -> None:
    lines = text_hints("filter 3\nEnergy = 12.4 keV\nsample-detector distance 1.2 m\ncomment: nice\nom = 0.15\n")
    assert lines == ["Energy = 12.4 keV", "sample-detector distance 1.2 m", "om = 0.15"]


def test_the_search_looks_around_the_frame_but_not_everywhere(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    listing = LocalFileExplorer().related_files(str(paths["frame"]))
    found = {Path(entry["path"]).name: entry["relation"] for entry in listing["entries"]}
    assert found["AgBH_00001.tif"] == "sibling folder calib"
    assert found["beamtime.log"] == "sibling folder logs"
    assert found["agbh.poni"] == "folder processed two levels up"
    assert found["eh1_scan_00012.fio"] == "two levels up"
    assert "sample_00001.tif" not in found and "P3HT_00013.tif" not in found
    assert listing["frame"]["path"] == str(paths["frame"].resolve()) and not listing["truncated"]
    small = LocalFileExplorer(max_entries=3).related_files(str(paths["frame"]))
    assert small["truncated"]


class CalibrationWorkbench(FakeWorkbench):
    """No geometry until ``use_geometry``."""

    def __init__(self, frame: Path):
        super().__init__(measurement=None)
        self.frame = frame
        self.geometry = None

    def status(self) -> dict:
        status = {
            "file": self.frame.name, "path": str(self.frame), "measurement": self.measurement,
            "shape": list(SHAPE), "geometry": self.geometry, "header": {"pixel_size_um": [None, None]}, "curves": [],
        }
        if self.geometry is not None:
            status.update(super().status(), path=str(self.frame), shape=list(SHAPE))
        return status

    def use_geometry(self, values, name, source):
        self.calls.append(("use_geometry", values, name, source))
        self.geometry = values
        self.measurement = "giwaxs"
        return self.status()


class FakeCalibrator:
    def __init__(self):
        self.calls = []

    def standards(self):
        return {"agbh": "AgBH"}

    def calibrate(self, path, *, standard, energy_kev, distance_mm=None, pixel_size_m=None, cancelled=None):
        self.calls.append((Path(path).name, standard, energy_kev, distance_mm, pixel_size_m))
        return {
            "standard": standard, "energy_kev": energy_kev, "wavelength_angstrom": 12.398419843320026 / energy_kev,
            "distance_mm": 100.2, "beam_center_px": [200.1, 249.8], "pixel_size_um": [100.0, 100.0],
            "matched_rings": 12, "rms_residual_px": 0.4, "confidence": "High", "source_image": path, "alternatives": [],
        }

    def read_result(self, path):
        raise AssertionError("not used")


class Chooser:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def choose(self, question, options, allow_text):
        self.asked.append((question, [option["label"] for option in options], allow_text))
        return self.answer


class Confirmer:
    def __init__(self, answer=True):
        self.answer = answer
        self.questions = []

    def confirm(self, title, text):
        self.questions.append(text)
        return self.answer


def catalog_for(paths: dict, *, permission=PERMISSION_AUTO, chooser=None, confirmer=None):
    workbench = CalibrationWorkbench(paths["frame"])
    calibrator = FakeCalibrator()
    results = RunResults()
    catalog = ToolCatalog(
        workbench, AnalysisGoals(goals=("peaks",), permission=permission), results,
        explorer=LocalFileExplorer(), calibrator=calibrator, chooser=chooser, confirmer=confirmer,
    )
    catalog.execute(call("get_status"))
    return catalog, workbench, calibrator, results


def payload(outcome) -> dict:
    assert not outcome.is_error, outcome.content
    return json.loads(outcome.content)


def test_the_tools_find_rank_read_and_use_calibration_material(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    confirmer = Confirmer()
    catalog, workbench, calibrator, results = catalog_for(paths, permission=PERMISSION_CONFIRM, confirmer=confirmer)

    found = payload(catalog.execute(call("find_calibration_files")))
    images = [Path(item["path"]).name for item in found["standard_images"]]
    assert images[:2] == ["LaB6_00001.tif", "AgBH_00001.tif"]  # closest in time first
    assert images[-1] == "agbe_2025_00002.tif"
    agbh = found["standard_images"][1]
    assert agbh["standard_key"] == "agbh" and agbh["gimap_can_fit"] and agbh["hours_from_frame"] == pytest.approx(-3, abs=0.1)
    assert agbh["folder"] == "sibling folder calib" and agbh["modified"]
    assert [Path(item["path"]).name for item in found["calibration_results"]] == ["agbh.poni"]
    assert {Path(item["path"]).name for item in found["logs"]} == {"beamtime.log", "eh1_scan_00012.fio"}
    assert payload(catalog.execute(call("inspect_file", path=str(paths["log"]))))["lines"] == [
        "energy = 12.4 keV", "sample to detector distance: 100 mm",
    ]
    missing = catalog.execute(call("inspect_file", path=str(tmp_path / "secret.txt")))
    assert missing.is_error and "There is no file" in missing.content

    # A calibration image is fitted, then used after the person approves it.
    fitted = payload(catalog.execute(call("calibrate_geometry", path=str(paths["agbh_before"]), standard="agbh",
                                          energy_kev=12.4, distance_mm=100, pixel_size_um=100)))
    assert calibrator.calls == [("AgBH_00001.tif", "agbh", 12.4, 100.0, pytest.approx(100e-6))]
    assert fitted["calibration_index"] == 0 and fitted["assessment"].startswith("good")
    used = catalog.execute(call("use_geometry", source="calibration", incidence_deg=0.2))
    assert not used.is_error
    assert "distance 100.2 mm" in confirmer.questions[0] and "(200.1, 249.8) px" in confirmer.questions[0]
    _name, values, profile, source = workbench.calls[-1]
    assert values["distance_mm"] == 100.2 and values["incidence_deg"] == 0.2 and profile is None
    assert "agbh calibration of AgBH_00001.tif (12 rings" in source
    assert results.geometry_used["source"].startswith("agbh calibration")

    # A .poni file is used only after it was read.
    early = catalog.execute(call("use_geometry", source="file", path=str(paths["poni"])))
    assert early.is_error and "Inspect" in early.content
    poni = payload(catalog.execute(call("inspect_file", path=str(paths["poni"]))))
    assert poni["distance_mm"] == pytest.approx(100.0) and poni["energy_kev"] == pytest.approx(ENERGY_KEV, rel=1e-3)
    assert not catalog.execute(call("use_geometry", source="file", path=str(paths["poni"]))).is_error
    assert workbench.calls[-1][1]["beam_center_x_px"] == pytest.approx(200.0)

    # Explicit values need all of them.
    missing = catalog.execute(call("use_geometry", source="values", distance_mm=100))
    assert missing.is_error and "beam_center_x_px" in missing.content and "energy_kev" in missing.content


def test_files_near_the_frame_are_readable_and_others_need_the_persons_consent(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    far = touch(tmp_path.parent / f"{tmp_path.name}_elsewhere" / "cal" / "LaB6_run7.txt", "LaB6 at 12.4 keV, distance 1470 mm\n")
    hidden = touch(tmp_path / "beamtime" / "raw" / ".cache" / "notes.txt", "energy 12 keV\n")
    program = touch(tmp_path / "beamtime" / "raw" / "calib" / "tool.exe", "x")
    confirmer = Confirmer(False)
    catalog, *_rest = catalog_for(paths, permission=PERMISSION_CONFIRM, confirmer=confirmer)

    # The frame's own tree is open without a search first.
    assert payload(catalog.execute(call("inspect_file", path=str(paths["log"]))))["lines"]
    refused = catalog.execute(call("inspect_file", path=str(far)))
    assert refused.is_error and "did not allow" in refused.content and "elsewhere" in confirmer.questions[0]
    assert "never read" in catalog.execute(call("inspect_file", path=str(hidden))).content
    assert "not .exe" in catalog.execute(call("inspect_file", path=str(program))).content
    confirmer.answer = True
    assert payload(catalog.execute(call("inspect_file", path=str(far))))["lines"] == ["LaB6 at 12.4 keV, distance 1470 mm"]

    # Searching up from the frame needs no consent; a name filter finds any readable file.
    confirmer.questions.clear()
    found = payload(catalog.execute(call("search_files", folder=str(tmp_path), name_contains="lab6")))
    assert {Path(item["path"]).name for item in found["matches"]} == {"LaB6_00001.tif"}
    assert not confirmer.questions
    automatic, *_rest = catalog_for(paths, permission=PERMISSION_AUTO)
    assert not automatic.execute(call("inspect_file", path=str(far))).is_error  # logged, not asked


def test_paths_the_person_names_are_read_and_searched_directly(tmp_path: Path) -> None:
    from src.gimap.features.assistant.application import RunAssistantTask
    from tests.assistant_fakes import ScriptedLlm, report, turn

    paths = beamtime(tmp_path)
    given = save_tiff(tmp_path / "other drive" / "img_0005.tif", agbh_image())
    folder = (tmp_path / "other drive").resolve()
    notes = f"The calibration is {given} 是 LaB6 或 CeO2 的定标图；logs are in “{folder}”."
    goals = AnalysisGoals(goals=("peaks",), permission=PERMISSION_CONFIRM, instructions=notes)
    confirmer = Confirmer(False)
    catalog = ToolCatalog(CalibrationWorkbench(paths["frame"]), goals, RunResults(), explorer=LocalFileExplorer(), confirmer=confirmer)
    catalog.execute(call("get_status"))
    assert set(catalog.access_notes()) == {
        f"file named by the user, readable and searchable: {given}",
        f"folder named by the user, readable and searchable: {folder}",
    }
    header = payload(catalog.execute(call("inspect_file", path=str(given))))
    assert header["shape"] == list(SHAPE) and header["standard"] is None and not confirmer.questions
    listed = payload(catalog.execute(call("search_files", folder=str(folder), name_contains="img")))
    assert [Path(item["path"]).name for item in listed["matches"]] == ["img_0005.tif"]
    # The notes reach the model together with what became readable.
    llm = ScriptedLlm([turn(report(("peaks", "not_available")))])
    RunAssistantTask(llm, CalibrationWorkbench(paths["frame"]), explorer=LocalFileExplorer())(goals)
    message = llm.requests[0]["messages"][0]["content"]
    assert f"file named by the user, readable and searchable: {given}" in message


class RankingCalibrator(FakeCalibrator):
    """Every standard fits, some much better than others."""

    def __init__(self, quality: dict):
        super().__init__()
        self.quality = quality

    def calibrate(self, path, *, standard, **options):
        result = super().calibrate(path, standard=standard, **options)
        if standard not in self.quality:
            raise RuntimeError("No calibration-standard match was found.")
        rings, rms, distance = self.quality[standard]
        return {**result, "matched_rings": rings, "rms_residual_px": rms, "distance_mm": distance}


def test_an_unknown_standard_is_found_by_comparing_fits(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    image = str(paths["agbh_before"])

    def compare(quality, **arguments):
        catalog, _workbench, _calibrator, results = catalog_for(paths)
        catalog.calibrator = RankingCalibrator(quality)
        catalog.execute(call("find_calibration_files"))
        outcome = payload(catalog.execute(call("calibrate_geometry", path=image, standard="compare", energy_kev=12.4, **arguments)))
        return outcome, results

    clear, results = compare({"agbh": (13, 0.3, 100.0), "lab6": (4, 1.1, 34.4), "ceo2": (3, 1.9, 60.0)})
    assert [row["standard"] for row in clear["comparison"]] == ["agbh", "lab6", "ceo2"]
    assert clear["verdict"].startswith("clear: agbh") and clear["best_calibration_index"] == 0
    assert len(results.calibrations) == 3
    close, _results = compare({"agbh": (6, 0.5, 100.0), "lab6": (6, 0.55, 1470.0)})
    assert close["verdict"].startswith("ambiguous")
    by_distance, _results = compare({"agbh": (6, 0.5, 100.0), "lab6": (6, 0.55, 1470.0)}, distance_mm=1450)
    assert by_distance["verdict"].startswith("probably lab6")
    failed, _results = compare({"ceo2": (2, 3.0, 60.0)})
    assert failed["verdict"].startswith("none fits well")
    assert {item["standard"] for item in failed["failed"]} == {"agbh", "lab6", "lab6_ceo2"}


def test_comparing_standards_on_a_real_agbh_image_picks_agbh(tmp_path: Path) -> None:
    from src.gimap.features.calibration.bootstrap import create_headless_calibration

    paths = beamtime(tmp_path)
    image = save_tiff(tmp_path / "beamtime" / "raw" / "calib" / "img_0005.tif", agbh_image())
    catalog, _workbench, _calibrator, _results = catalog_for(paths)
    catalog.calibrator = create_headless_calibration()
    outcome = payload(catalog.execute(call("calibrate_geometry", path=str(image), standard="compare", energy_kev=ENERGY_KEV, pixel_size_um=100)))
    assert outcome["verdict"].startswith("clear: agbh"), outcome
    best = outcome["comparison"][0]
    assert best["standard"] == "agbh" and best["distance_mm"] == pytest.approx(100.0, rel=0.01)


def test_a_declined_geometry_is_not_saved(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    catalog, workbench, _calibrator, results = catalog_for(paths, permission=PERMISSION_CONFIRM, confirmer=Confirmer(False))
    catalog.execute(call("find_calibration_files"))
    catalog.execute(call("calibrate_geometry", path=str(paths["agbh_before"]), standard="agbh", energy_kev=12.4))
    declined = payload(catalog.execute(call("use_geometry", source="calibration")))
    assert declined["declined"] and not any(entry[0] == "use_geometry" for entry in workbench.calls)
    assert results.geometry_used is None


def test_the_person_can_be_asked_to_choose_or_to_type(tmp_path: Path) -> None:
    paths = beamtime(tmp_path)
    chooser = Chooser({"index": 1, "text": ""})
    catalog, *_rest = catalog_for(paths, chooser=chooser)
    options = [{"label": "AgBH_00001.tif", "detail": "calib · 3 h before"}, {"label": "agbe_2025_00002.tif", "detail": "400 h before"}]
    answer = payload(catalog.execute(call("ask_user", question="Which calibration?", options=options)))
    assert answer["answered"] and answer["option"]["label"] == "agbe_2025_00002.tif" and answer["index"] == 1
    assert chooser.asked == [("Which calibration?", ["AgBH_00001.tif", "agbe_2025_00002.tif"], False)]
    typed = Chooser({"index": None, "text": "12.4 keV"})
    catalog, *_rest = catalog_for(paths, chooser=typed)
    answer = payload(catalog.execute(call("ask_user", question="Energy?", options=[], allow_text=True)))
    assert answer["answered"] and answer["option"] is None and answer["text"] == "12.4 keV"
    catalog, *_rest = catalog_for(paths, chooser=Chooser(None))
    assert payload(catalog.execute(call("ask_user", question="Energy?", options=[])))["answered"] is False


def test_gimaps_calibration_recovers_a_known_geometry_from_agbh(tmp_path: Path) -> None:
    from src.gimap.features.calibration.bootstrap import create_headless_calibration

    path = save_tiff(tmp_path / "AgBH_00001.tif", agbh_image())
    result = create_headless_calibration().calibrate(str(path), standard="agbh", energy_kev=ENERGY_KEV, pixel_size_m=PIXEL)
    assert result["standard"] == "agbh" and result["matched_rings"] >= 8
    assert result["distance_mm"] == pytest.approx(100.0, rel=0.01)
    assert result["beam_center_px"] == pytest.approx(list(CENTER), abs=1.0)
    assert result["rms_residual_px"] < 1.0 and result["shape"] == list(SHAPE)


def giwaxs_frame(seed: int = 5) -> np.ndarray:
    """Lamellar 0.40 / 0.80 Å⁻¹ along the surface normal and an isotropic ring at 1.10 Å⁻¹."""
    from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps
    from src.gimap.shared.geometry import DetectorGeometry

    maps = giwaxs_maps(SHAPE, DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2))
    q, chi = maps.q.astype(float), np.abs(maps.chi_deg.astype(float))
    normal = np.exp(-(chi**2) / (2 * 15.0**2))

    def ring(center, fwhm):
        return np.exp(-4 * np.log(2) * (q - center) ** 2 / fwhm**2)

    image = 30.0 + 300.0 * np.exp(-q / 0.25) + 600 * ring(0.40, 0.03) * normal + 150 * ring(0.80, 0.035) * normal
    image += 120 * ring(1.10, 0.04)
    return np.random.default_rng(seed).poisson(image).astype(np.float32)


def test_a_frame_without_geometry_is_calibrated_from_files_nearby(tmp_path: Path) -> None:
    from PyQt5.QtWidgets import QMainWindow

    from src.gimap.app import AppContext
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.features.assistant.application import BACKEND_API, RUN_COMPLETED
    from src.gimap.features.assistant.presentation import AssistantController
    from src.gimap.features.calibration.bootstrap import create_headless_calibration
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from tests.assistant_fakes import ScriptedLlm, report, turn
    from tests.test_assistant_gui import _app, _services, _wait

    _app()
    root = tmp_path / "beamtime" / "raw"
    frame = save_tiff(root / "P3HT_film" / "P3HT_00012.tif", giwaxs_frame())
    agbh = save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    log = touch(root / "logs" / "beamtime.log", "Energy: 12.3984 keV\nSDD approx 100 mm\nalpha_i = 0.2 deg\n")
    profiles = InMemoryInstrumentProfileRepository([])
    context = AppContext(
        settings=InMemorySettingsRepository({}), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(), instrument_profiles=profiles,
    )
    page = AnalyzePage(create_analyze_view_model(context))
    window = QMainWindow()
    window.setCentralWidget(page)
    page.add_paths([frame])
    assert page.tasks.wait(60)
    assert page.view_model.state.analysis.geometry is None  # nothing calibrated yet

    llm = ScriptedLlm([
        turn(call("find_calibration_files"), text="No geometry: looking for calibration files."),
        turn(call("inspect_file", path=str(log))),
        turn(call("calibrate_geometry", path=str(agbh), standard="agbh", energy_kev=12.3984, distance_mm=100, pixel_size_um=100)),
        turn(call("use_geometry", source="calibration", incidence_deg=0.2)),
        turn(call("set_measurement_mode", mode="giwaxs")),
        turn(call("find_peaks", curve="radial")),
        turn(report(("peaks", "done"))),
    ])
    services = _services(tmp_path, llm)
    services = type(services)(**{**services.__dict__, "explorer": LocalFileExplorer(), "calibrator": create_headless_calibration()})
    context.settings.set("assistant", "backend", BACKEND_API)
    controller = AssistantController(window, services, settings=context.settings, automation=page.automation)
    assert controller.run(AnalysisGoals(goals=("peaks",), permission=PERMISSION_AUTO))
    _wait(lambda: controller.outcome is not None, 120)

    outcome = controller.outcome
    assert outcome.state == RUN_COMPLETED, outcome.message
    assert all(step.ok for step in outcome.steps), [(step.tool, step.summary) for step in outcome.steps]
    fit = outcome.results.calibrations[0]
    assert fit["matched_rings"] >= 8 and fit["distance_mm"] == pytest.approx(100.0, rel=0.01)
    saved = profiles.load_all()
    assert len(saved) == 1 and saved[0].detector_shape == SHAPE
    geometry = saved[0].geometry
    assert geometry.distance_m == pytest.approx(0.100, rel=0.01)
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == pytest.approx(CENTER, abs=1.0)
    assert geometry.incidence_deg == pytest.approx(0.2) and "AgBH_00001.tif" in saved[0].source
    analysis = page.view_model.state.analysis
    assert analysis.geometry is not None and analysis.kind == "giwaxs"
    found = [peak.q for peak in outcome.results.peak_searches["radial"].peaks]
    for expected in (0.40, 0.80, 1.10):
        assert min(abs(q - expected) for q in found) < 0.02, found
    text = controller.panel.report_view.toPlainText()
    assert "Geometry used for this frame" in text and "Calibration fits" in text
    page.tasks.wait(60)
    window.close()


def test_the_choice_dialog_returns_the_option_or_the_typed_text() -> None:
    from tests.test_assistant_gui import _app

    from src.gimap.features.assistant.presentation import ChoiceDialog

    _app()
    options = [{"label": "AgBH_00001.cbf", "detail": "calib · 3 h before"}, {"label": "LaB6_00001.cbf", "detail": ""}]
    dialog = ChoiceDialog("Which calibration?", options, False)
    assert dialog.use_button.isEnabled() and dialog.answer() == {"index": 0, "text": ""}
    dialog.option_list.setCurrentRow(1)
    assert dialog.answer()["index"] == 1 and not dialog.text_edit.isVisibleTo(dialog)
    typed = ChoiceDialog("Energy?", [], True)
    assert not typed.use_button.isEnabled()
    typed.text_edit.setText("12.4 keV")
    assert typed.use_button.isEnabled() and typed.answer() == {"index": None, "text": "12.4 keV"}


def test_a_calibration_named_in_the_notes_is_used_even_far_away(tmp_path: Path, monkeypatch) -> None:
    """The user's case: 'the calibration is <path>' — no search range, no standard in the name."""
    from PyQt5.QtWidgets import QMainWindow, QMessageBox

    asked: list[str] = []

    def answer_yes(_parent, _title, text, *_args):
        asked.append(text)
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer_yes))

    from src.gimap.app import AppContext
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.features.assistant.application import BACKEND_API, RUN_COMPLETED
    from src.gimap.features.assistant.presentation import AssistantController
    from src.gimap.features.calibration.bootstrap import create_headless_calibration
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from tests.assistant_fakes import ScriptedLlm, report, turn
    from tests.test_assistant_gui import _app, _services, _wait

    _app()
    frame = save_tiff(tmp_path / "beamtime" / "raw" / "P3HT_film" / "P3HT_00012.tif", giwaxs_frame())
    far = save_tiff(tmp_path.parent / f"{tmp_path.name}_other_disk" / "2026" / "img_0005.tif", agbh_image())
    profiles = InMemoryInstrumentProfileRepository([])
    context = AppContext(
        settings=InMemorySettingsRepository({}), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(), instrument_profiles=profiles,
    )
    page = AnalyzePage(create_analyze_view_model(context))
    window = QMainWindow()
    window.setCentralWidget(page)
    page.add_paths([frame])
    assert page.tasks.wait(60)

    def use_best(messages):
        # Take the comparison's best fit, as the model is told to.
        results = [block for block in messages[-1]["content"] if block.get("type") == "tool_result"]
        best = json.loads(results[0]["content"])["best_calibration_index"]
        return turn(call("use_geometry", source="calibration", calibration_index=best, incidence_deg=0.2))

    llm = ScriptedLlm([
        turn(call("inspect_file", path=str(far))),
        turn(call("calibrate_geometry", path=str(far), standard="compare", energy_kev=12.3984, pixel_size_um=100)),
        use_best,
        turn(call("set_measurement_mode", mode="giwaxs")),
        turn(call("find_peaks", curve="radial")),
        turn(report(("peaks", "done"))),
    ])
    services = _services(tmp_path, llm)
    services = type(services)(**{**services.__dict__, "explorer": LocalFileExplorer(), "calibrator": create_headless_calibration()})
    context.settings.set("assistant", "backend", BACKEND_API)
    controller = AssistantController(window, services, settings=context.settings, automation=page.automation)
    notes = f"定标文件是 {far} ，能量 12.3984 keV"
    assert controller.run(AnalysisGoals(goals=("peaks",), permission=PERMISSION_CONFIRM, instructions=notes))
    _wait(lambda: controller.outcome is not None, 180)

    outcome = controller.outcome
    assert outcome.state == RUN_COMPLETED, outcome.message
    assert f"file named by the user, readable and searchable: {far}" in llm.requests[0]["messages"][0]["content"]
    inspected, compared = outcome.steps[1], outcome.steps[2]
    assert inspected.ok and compared.ok, (inspected.summary, compared.summary)
    assert "clear: agbh" in compared.summary
    assert len(asked) == 1 and "Save the geometry" in asked[0]  # reading the named file needed no question
    geometry = profiles.load_all()[0].geometry
    assert geometry.distance_m == pytest.approx(0.100, rel=0.01)
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == pytest.approx(CENTER, abs=1.0)
    found = [peak.q for peak in outcome.results.peak_searches["radial"].peaks]
    assert min(abs(q - 0.40) for q in found) < 0.02
    page.tasks.wait(60)
    window.close()
