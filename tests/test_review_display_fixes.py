"""First-user review, display round: a q map no finer than the detector samples q and a view fitted to its measured
cells, the header's distance against the calibration, a calibration of another folder that contradicts the frame,
operations numbered per run, a bounded coverage histogram, the legend of Geometry Calibration in the theme's text
colour, and the batch's fit time said as the command line says it."""

from __future__ import annotations

import json
import os
import re
import tracemalloc
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation.i18n import ZH

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "tests" / "data" / "external"
MANIFEST = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8")) if (EXTERNAL / "manifest.json").exists() else {}
public = pytest.mark.skipif(not MANIFEST, reason="public test data not downloaded (tests/data/external)")


def _fields(text: str) -> set:
    return set(re.findall(r"\{\w+\}", text))


# -- K: the coverage histogram ---------------------------------------------------------------------------------


def test_the_coverage_histogram_is_bounded_by_the_points() -> None:
    """A step of 1e-12 over a range of 1 asked for 10¹² edges (a MemoryError on a 401-point cut)."""
    from src.gimap.shared import signed_cut

    values = np.linspace(0.01, 1.0, 200)
    tracemalloc.start()
    share = signed_cut._coverage(values, 0.01, 1.0, 1e-12)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 5e6 and 0.0 < share <= 1.0 / signed_cut.COVERAGE_BINS_PER_POINT + 1e-9
    assert signed_cut._coverage(values, 0.01, 1.0, 1.5 * (values[1] - values[0])) == pytest.approx(1.0)  # as before


def test_a_cut_whose_halves_share_their_q_values_chooses_its_halves() -> None:
    """Both halves on the same |qy| grid: the median spacing of the union is 0, the step was 1e-6 or less."""
    from src.gimap.shared.signed_cut import choose_halves

    half = np.linspace(1e-3, 2.0, 200)
    q = np.r_[-half[::-1], half]
    intensity = 1e3 / (1.0 + (np.abs(q) * 20) ** 2) + 1.0
    choice = choose_halves(q, intensity)
    assert choice.side in ("mean", "both_abs", "positive", "negative") and set(choice.coverage) == {"negative", "positive"}


# -- K: operations are numbered per run ------------------------------------------------------------------------


def test_operations_are_numbered_within_their_own_list() -> None:
    from src.gimap.features.assistant.application.operations import Operation, add_operation

    Operation("set_halves", {"side": "mean"}, "spent ids")  # an earlier frame of the batch advanced the counter
    first, second = [], []
    for operations in (first, second):
        for side in ("mean", "positive"):
            add_operation(operations, Operation("set_halves", {"side": side}, side))
    assert [item.id for item in first] == [item.id for item in second] == ["op1", "op2"]
    assert [item.record()["id"] for item in second] == ["op1", "op2"]


def test_the_tools_that_record_operations_number_them_per_run() -> None:
    source = (ROOT / "src/gimap/features/assistant/application/operation_tools.py").read_text(encoding="utf-8")
    assert "operations.append(" not in source and source.count("add_operation(self.results.operations, operation)") == 2


# -- K: a calibration the search found in another folder --------------------------------------------------------


def test_a_searched_calibration_of_another_folder_is_not_adopted_against_its_own_shape_or_energy(tmp_path) -> None:
    from src.gimap.features.assistant.application.calibration_match import (
        poni_shape,
        searched_mismatch,
        searched_other_energy,
    )

    (tmp_path / "sample").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "sample" / "setup.poni").write_text("", encoding="utf-8")
    frame = {"path": str(tmp_path / "sample" / "frame_0001.tif"), "shape": [2048, 2048]}
    text = 'Detector_config: {"pixel1": 0.000172, "pixel2": 0.000172, "max_shape": [1043, 981]}\nDistance: 1.7\n'
    assert poni_shape(text) == (1043, 981) and poni_shape("Detector_config: {}") is None
    sibling, beside = str(tmp_path / "other" / "setup.poni"), str(tmp_path / "sample" / "setup.poni")
    why = searched_mismatch(sibling, {"energy_kev": 9.25}, frame, text)
    assert why is not None and "another folder" in why and "[1043, 981]" in why and "[2048, 2048]" in why
    assert searched_mismatch(beside, {"energy_kev": 9.25}, frame, text) is None  # beside the frame: the person's
    assert searched_mismatch(sibling, {"energy_kev": 9.25}, {**frame, "shape": [981, 1043]}, text) is None
    assert searched_mismatch(sibling, {"energy_kev": 9.25}, frame, "") is None  # the file states no shape
    # Another energy: the distance and centre hold, so it is used, but said as a point to check.
    said = searched_other_energy(sibling, {"energy_kev": 9.25}, frame, 18.0)
    assert said is not None and "9.25 keV" in said and "18 keV" in said and "another folder" in said
    assert searched_other_energy(sibling, {"energy_kev": 18.01}, frame, 18.0) is None
    assert searched_other_energy(beside, {"energy_kev": 9.25}, frame, 18.0) is None
    assert searched_other_energy(sibling, {"energy_kev": 9.25}, frame, None) is None


def test_candidates_of_other_folders_are_listed_with_their_folder_and_ranked_after_the_frames_own() -> None:
    from src.gimap.features.assistant.application.pipeline_geometry import _placed
    from src.gimap.features.assistant.domain.calibration_files import folder_place

    assert _placed({"path": "/data/s1/a.poni", "folder": "same folder"}) == "a.poni"
    assert _placed({"path": "/data/s1/cal/a.poni", "folder": "same folder, subfolder s1/cal"}) == "a.poni"
    assert _placed({"path": "/data/s2/b.poni", "folder": "sibling folder s2"}) == "b.poni (sibling folder s2)"
    assert folder_place("/data/s1", "/data/s1/a.poni") < folder_place("/data/s1", "/data/s2/b.poni")


@public
def test_the_run_skips_a_sibling_folders_poni_made_for_another_detector(tmp_path, monkeypatch) -> None:
    """The MAPbI3 frame (2048×2048) with only the GALAXI calibration (1043×981) next door: not adopted silently."""
    import shutil

    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application import PipelineOptions

    monkeypatch.setenv("GIMAP_HOME", str(tmp_path / "home"))
    root = tmp_path / "a" / "b" / "c" / "d"  # deep enough that the search sees no folder of another test
    (root / "sample").mkdir(parents=True)
    shutil.copy(EXTERNAL / MANIFEST["giwaxs_p08_mapi"]["frame"], root / "sample")
    (root / "gisaxs_galaxi").mkdir()
    shutil.copy(EXTERNAL / "gisaxs_galaxi" / "galaxi_bornagain.poni", root / "gisaxs_galaxi")
    session = open_headless_session(root / "sample" / "S121_MAI_A2_00841.tif", saved_profiles=False,
                                    out_dir=tmp_path / "out")
    try:
        report = session.run_pipeline(PipelineOptions(stop_after_geometry=True))
    finally:
        session.close()
    listed = next(item for item in report["decisions"] if item["what"] == "calibration candidates")
    assert listed["decision"].startswith("galaxi_bornagain.poni (sibling folder gisaxs_galaxi)")
    skipped = next(item for item in report["decisions"] if item["decision"] == "skipped galaxi_bornagain.poni")
    assert "another folder" in skipped["why"] and "[1043, 981]" in skipped["why"]
    assert not any(item["what"] == "geometry" and "galaxi" in item["decision"] for item in report["decisions"])


# -- K: the batch's fit time ------------------------------------------------------------------------------------


def test_batch_export_says_the_fit_time_as_the_command_line_does() -> None:
    from src.gimap.features.analyze.presentation.batch_texts import MODEL_LINES
    from src.gimap.features.assistant.application.pipeline_progress import FIT_TAKES

    for line in MODEL_LINES.values():
        assert "25 s" not in line and f"from {FIT_TAKES} per frame" in line and "几秒到几分钟" in ZH[line]
    for name in ("User_Manual.md", "User_Manual_zh.md"):
        page = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "25 s per frame" not in page and "约 15 秒一帧" not in page and "每帧约 25 s" not in page, name
    assert "about 25 s per frame" not in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


# -- J(i): the q map --------------------------------------------------------------------------------------------


def _blank_share(grid: np.ndarray) -> float:
    """Share of blank cells between the first and the last filled cell of each row."""
    filled = np.isfinite(grid)
    total = blank = 0
    for row in filled:
        index = np.flatnonzero(row)
        if index.size > 1:
            total += index[-1] - index[0] + 1
            blank += index[-1] - index[0] + 1 - index.size
    return blank / max(total, 1)


def test_the_q_map_grid_is_not_finer_than_the_detector() -> None:
    from src.gimap.features.analyze.domain.binning import binned_mean_2d, map_grid_shape

    rows, columns = np.indices((120, 90), dtype=np.float64)
    x, y = 0.01 * columns, 0.02 * rows  # one pixel: 0.01 in x, 0.02 in y
    usable = np.ones(x.shape, dtype=bool)
    ny, nx = map_grid_shape(x, y, usable, float(np.ptp(x)), float(np.ptp(y)), (800, 800))
    assert ny in (118, 119) and nx in (88, 89)  # one cell per pixel step, not 800 × 800
    grid = binned_mean_2d(x[usable], y[usable], np.ones(usable.sum()), x_range=(0.0, x.max() + 1e-12),
                          y_range=(0.0, y.max() + 1e-12), shape=(ny, nx))
    assert np.isfinite(grid).all()
    assert map_grid_shape(x, y, usable, float(np.ptp(x)), float(np.ptp(y)), (50, 40)) == (50, 40)  # the cap holds
    fine = np.indices((3000, 20), dtype=np.float64)
    assert map_grid_shape(1e-4 * fine[1], 1e-4 * fine[0], np.ones((3000, 20), bool), 19e-4, 0.2999, (800, 800)) == (800, 32)


@public
def test_the_q_map_of_the_tilted_public_frame_has_no_holes_between_its_pixels() -> None:
    """619×487 pixels on an 800×800 grid left 48 % of the cells between the first and last filled one blank. What
    stays blank now is what a three times coarser grid leaves blank too: the gaps and the missing wedge. The curves
    are reduced from the pixels and are the same with and without the map."""
    from src.gimap.features.analyze.domain.giwaxs import MAP_SHAPE, reduce_giwaxs
    from src.gimap.features.analyze.domain.binning import binned_mean_2d
    from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps
    from src.gimap.features.analyze.domain.validity import valid_pixels
    from src.gimap.features.assistant.domain import parse_poni
    from src.gimap.shared.detector_io import load_detector_image
    from src.gimap.shared.geometry import DetectorGeometry

    info = MANIFEST["giwaxs_xeuss_pbte"]
    flat = parse_poni((EXTERNAL / info["poni"]).read_text(encoding="utf-8"))
    frame = load_detector_image(EXTERNAL / info["frame"])
    valid = valid_pixels(frame.data, frame.mask)
    geometry = DetectorGeometry(flat.pixel_size_x_m, flat.pixel_size_y_m, flat.distance_mm * 1e-3, flat.beam_center_x_px,
                                flat.beam_center_y_px, flat.wavelength_angstrom, info["incidence_deg"])
    maps = giwaxs_maps(frame.data.shape, geometry)
    reduction = reduce_giwaxs(frame.data, valid, geometry, maps=maps)
    picture = reduction.reciprocal_space_map.image
    ny, nx = picture.shape
    assert ny < MAP_SHAPE[0] and nx < MAP_SHAPE[1] and ny <= 1.1 * frame.data.shape[0] and nx <= 1.3 * frame.data.shape[1]
    usable = valid & maps.above_horizon
    q_par, qz = maps.q_parallel[usable], maps.qz[usable]
    ranges = {"x_range": (float(q_par.min()), float(q_par.max()) + 1e-12), "y_range": (float(qz.min()), float(qz.max()) + 1e-12)}
    coarse = binned_mean_2d(q_par, qz, frame.data[usable], shape=(ny // 3, nx // 3), **ranges)
    old = binned_mean_2d(q_par, qz, frame.data[usable], shape=MAP_SHAPE, **ranges)
    assert _blank_share(picture) < _blank_share(coarse) + 0.03 < _blank_share(old) - 0.1
    without = reduce_giwaxs(frame.data, valid, geometry, maps=maps, with_map=False)
    for mapped, plain in zip(reduction.curves, without.curves):
        assert np.array_equal(mapped.x, plain.x) and np.array_equal(mapped.intensity, plain.intensity, equal_nan=True)


def test_the_view_is_fitted_to_the_measured_cells_not_to_a_stray_streak() -> None:
    from src.gimap.app.presentation.components.image_extent import filled_extent

    picture = np.full((100, 200), np.nan)
    picture[10:90, 80:200] = 1.0  # the data: columns 80–199, rows 10–89
    picture[50, 0:3] = 1.0        # three cells far to the left
    x, y, width, height = filled_extent(picture, (-1.0, 0.0, 2.0, 1.0))
    assert x == pytest.approx(-1.0 + 2.0 * 80 / 200, abs=0.02) and x + width == pytest.approx(1.0, abs=0.02)
    assert y == pytest.approx(0.10, abs=0.02) and y + height == pytest.approx(0.90, abs=0.02)
    assert filled_extent(np.full((4, 4), np.nan), (0, 0, 1, 1)) is None


def test_the_detector_view_shows_the_part_it_is_told_to_fit() -> None:
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app.presentation.components.detector_view import DetectorView

    app = QApplication.instance() or QApplication([])
    view = DetectorView()
    try:
        view.resize(600, 600)
        view.set_aspect_locked(False)
        picture = np.full((50, 50), np.nan)
        picture[:, 25:] = 1.0
        view.set_image(picture, rect=(-1.0, 0.0, 2.0, 1.0), y_down=False, fit_rect=(0.0, 0.0, 1.0, 1.0))
        app.processEvents()
        (x0, x1), _ = view.view_box.viewRange()
        assert -0.1 < x0 < 0.05 and 0.95 < x1 < 1.1
        view.set_image(picture, rect=(-1.0, 0.0, 2.0, 1.0), y_down=False)  # no part named: the whole picture again
        app.processEvents()
        assert view.view_box.viewRange()[0][0] < -0.9
    finally:
        view.dispose()
        view.deleteLater()


# -- J(ii): the header's distance against the calibration --------------------------------------------------------


def test_a_header_distance_far_from_the_calibration_is_said_and_a_close_one_is_not() -> None:
    from src.gimap.features.assistant.presentation.guided_words import SENTENCES
    from src.gimap.shared.geometry.header_check import (
        HEADER_DISTANCE_HINT,
        HEADER_DISTANCE_WHY,
        header_distance_disagreement,
    )

    assert header_distance_disagreement(370.0, 147.6) == {"header": "370", "used": "147.6", "percent": "151"}
    assert header_distance_disagreement(1730.0, 1700.0) is None and header_distance_disagreement(1.0, 1.06) is not None
    assert header_distance_disagreement(None, 147.6) is None and header_distance_disagreement(370.0, None) is None
    assert header_distance_disagreement(0.0, 147.6) is None
    assert HEADER_DISTANCE_WHY in SENTENCES and _fields(ZH[HEADER_DISTANCE_WHY]) == _fields(HEADER_DISTANCE_WHY)
    assert HEADER_DISTANCE_HINT in ZH


@public
def test_the_public_xeuss_frame_says_its_header_distance_in_the_report_and_in_the_data_step(tmp_path, monkeypatch) -> None:
    """Header 370 mm, calibration 147.6 mm: a point to check in the report, a line in Analyze's Data step."""
    import shutil

    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application import PipelineOptions
    from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown

    monkeypatch.setenv("GIMAP_HOME", str(tmp_path / "home"))
    folder = tmp_path / "pbte"
    shutil.copytree(EXTERNAL / "giwaxs_xeuss_pbte", folder)
    session = open_headless_session(folder / "PbTe_3010_r1_PT_1_10_0_00028.edf", saved_profiles=False, out_dir=tmp_path / "out")
    try:
        report = session.run_pipeline(PipelineOptions(calibration=str(folder / "AgBe_250403_XENOCS.poni"), incidence_deg=0.235))
        card = session.page.data_info_label.text()
    finally:
        session.close()
    point = next(item for item in report["needs_attention"] if item["item"] == "detector distance")
    assert point["why"] == "The file header gives a detector distance of 370 mm, the calibration in use 147.6 mm (151 % apart)."
    assert "**detector distance**" in pipeline_markdown(report)
    assert "370 mm" in card and "147.6 mm" in card and "⚠" in card


# -- J(iii): a calibrant frame in transmission ---------------------------------------------------------------------


def test_the_run_looks_for_rings_only_without_an_incidence_angle() -> None:
    from src.gimap.features.assistant.application.ring_frames import (
        TRANSMISSION_ATTENTION,
        TRANSMISSION_DECISION,
        TRANSMISSION_WHY,
        transmission_rings,
    )
    from src.gimap.features.assistant.presentation.guided_words import SENTENCES

    class Bench:
        asked = 0

        def ring_frame(self):
            self.asked += 1
            return {"complete_rings": 5, "is_ring_frame": True}

    bench, no_angle = Bench(), {"geometry": {"incidence_deg": 0.0}}
    assert transmission_rings(bench, no_angle, False) == 5
    assert transmission_rings(bench, no_angle, True) is None  # αi given: the person's statement
    assert transmission_rings(bench, {"geometry": {"incidence_deg": 0.2}}, False) is None  # αi of the profile
    assert transmission_rings(bench, {}, False) is None and bench.asked == 1
    assert transmission_rings(object(), no_angle, False) is None  # a workbench that cannot look

    class Few(Bench):
        def ring_frame(self):
            return {"complete_rings": 2, "is_ring_frame": False}

    assert transmission_rings(Few(), no_angle, False) is None
    for sentence in (TRANSMISSION_ATTENTION, TRANSMISSION_DECISION, TRANSMISSION_WHY):
        assert sentence in SENTENCES and _fields(ZH[sentence]) == _fields(sentence) == {"{rings}"}


@public
def test_the_public_agbh_frame_is_not_fitted_with_particles(tmp_path, monkeypatch) -> None:
    """Auto calls it GISAXS (small angles); with no αi and five complete rings the run fits no particle model."""
    import shutil

    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application import PipelineOptions

    monkeypatch.setenv("GIMAP_HOME", str(tmp_path / "home"))
    folder = tmp_path / "agbh"
    shutil.copytree(EXTERNAL / "saxs_agbh_pyfai", folder)
    session = open_headless_session(folder / "Pilatus1M.edf", saved_profiles=False, out_dir=tmp_path / "out")
    fitted = []
    session.catalog.fitter = lambda *args, **kwargs: fitted.append(1) or []
    try:
        report = session.run_pipeline(PipelineOptions(calibration=str(folder / "Pilatus1M.poni")))
    finally:
        session.close()
    decisions = {(item["what"], item["decision"]) for item in report["decisions"]}
    assert ("technique", "a transmission frame of a calibrant or powder (5 complete rings about the beam centre)") in decisions
    assert ("fit", "not run") in decisions and not fitted
    assert any(item["item"] == "technique" and "no particle model was fitted" in item["why"] for item in report["needs_attention"])


# -- J(iv): the legend of Geometry Calibration ---------------------------------------------------------------------


def test_the_calibration_legend_words_take_the_themes_text_colour() -> None:
    from PyQt5.QtWidgets import QApplication

    from tests.test_release_tools_calibration import _context, _dispose
    from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
    from src.gimap.features.calibration.presentation.preview_style import LEGEND_SAMPLE_BACKGROUND, MATCHED_RING_COLOR

    QApplication.instance() or QApplication([])
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        legend = dialog.overlay_legend.text()
        assert "#f8fafc" not in legend and dialog.overlay_legend.property("gimapRole") != "overlay"
        words = [word.strip() for word in re.findall(r"</span> ([^<&]+)", legend)]
        assert words == ["Center", "Detected", "Matched", "Other theoretical"]  # outside every coloured span
        assert f"color:{MATCHED_RING_COLOR}; background-color:{LEGEND_SAMPLE_BACKGROUND}" in legend
    finally:
        _dispose(dialog)
