"""The GIWAXS workspace: regions in the list, on the cake and in the plots; masks; mirror filling; peak tracking."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_workspace import _app, _context
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, giwaxs_frame, save_tiff

GEOMETRY = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2)


def _settle(page, condition=lambda: True, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        page.tasks.wait(0.2)
        _app().processEvents()
        if condition() or time.monotonic() > deadline:
            break
    assert condition()


@pytest.fixture
def page(tmp_path: Path):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    _app()
    image = giwaxs_frame()
    image[:, 60:66] = np.nan  # a module gap (a floating-point frame: NaN, since negative values are data)
    frame = save_tiff(tmp_path / "film.tif", image)
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    page.resize(1500, 950)
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(frame)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    yield page
    page.tasks.wait(30)
    page.dispose()
    page.close()


def test_regions_are_listed_plotted_and_coloured_the_same_everywhere(page) -> None:
    from src.gimap.features.analyze.presentation.bindings.regions import REGION_COLORS

    assert [row.key for row in page._region_rows] == ["full", "in_plane", "out_of_plane", "ring"]
    assert not page.regions_panel.isHidden() and page.gisaxs_cuts.isHidden()
    page._add_region("ring")
    _settle(page, lambda: len(page._region_rows) == 5)
    assert page.region_list.currentRow() == 4 and page.region_editor.currentWidget() is page.region_editor_pages["generic"]
    region = page.view_model.state.giwaxs.regions[0]
    assert region.q_range[0] < 1.10 < region.q_range[1] or region.q_range[0] < 0.40 < region.q_range[1]
    assert "region1" in page._plot_keys["top"] and "region1_chi" in page._plot_keys["bottom"]
    top_colors = dict(zip(page._plot_keys["top"], page.top_plot.curve_colors()))
    bottom_colors = dict(zip(page._plot_keys["bottom"], page.bottom_plot.curve_colors()))
    assert top_colors["region1"] == bottom_colors["region1_chi"] == REGION_COLORS[0]
    # The editor changes the region; its curves follow.
    page.region_all_chi_check.setChecked(False)  # χ as centre ± half width: 15 ± 15° = 0–30°
    page.region_chi_center.setValue(15.0)
    page.region_chi_half.setValue(15.0)
    _settle(page, lambda: page.view_model.state.giwaxs.regions[0].chi_range == (0.0, 30.0))
    curve = page.view_model.state.analysis.reduction.curve("region1_chi")
    _settle(page, lambda: page.view_model.state.analysis.reduction.curve("region1_chi").x.max() <= 30.0)
    assert curve is not None
    # Unticking hides its curves and its outline.
    item = page.region_list.item(4)
    item.setCheckState(Qt.Unchecked)
    assert "region1" not in page._plot_keys["top"] and "region1_chi" not in page._plot_keys["bottom"]
    page.region_list.item(4).setCheckState(Qt.Checked)
    assert "region1" in page._plot_keys["top"]
    page.region_list.setCurrentRow(4)
    page.region_remove_button.click()
    _settle(page, lambda: len(page._region_rows) == 4)


def test_the_cake_shows_regions_as_rectangles_that_can_be_dragged(page) -> None:
    page._add_region("out_of_plane")
    _settle(page, lambda: len(page._region_rows) == 5)
    page.set_view(2)
    _settle(page, lambda: getattr(page, "_cake", None) is not None and page.detector_view.has_image()
            and page.detector_view._data.shape == (360, 800))
    keys = page.shape_layer.rect_keys()
    assert {"region1", "region1~", "ring", "in_plane", "out_of_plane"} <= set(keys)
    page.shape_layer.rectChanged.emit("region1~", 0.3, -25.0, 0.5, -5.0)  # the mirrored rectangle, below χ = 0
    _settle(page, lambda: page.view_model.state.giwaxs.regions[0].q_range == pytest.approx((0.3, 0.5)))
    assert page.view_model.state.giwaxs.regions[0].chi_range == pytest.approx((5.0, 25.0))
    page.shape_layer.rectChanged.emit("ring", 0.78, -90.0, 0.82, 90.0)  # the ring of I(χ) moves to 0.80
    _settle(page, lambda: page.view_model.state.giwaxs.chi_q_window == pytest.approx((0.78, 0.82)))
    page.set_view(1)
    assert page.shape_layer._outlines  # the regions outlined on the q map


def test_masks_are_drawn_saved_loaded_and_filled_from_the_mirror(page, tmp_path: Path) -> None:
    page.shape_layer.shapeDrawn.emit("rectangle", [(300.0, 20.0), (340.0, 60.0)])
    _settle(page, lambda: page.view_model.state.analysis.drawn_mask is not None)
    analysis = page.view_model.state.analysis
    assert analysis.drawn_mask[20:60, 300:340].all() and not analysis.valid[20:60, 300:340].any()
    assert page.mask_list.count() == 1 and "Rectangle" in page.mask_list.item(0).text()
    assert page.shape_layer._outlines  # outlined on the detector
    saved = page.save_masks(tmp_path / "masks.json")
    assert json.loads(saved.read_text())["masks"][0]["kind"] == "rectangle"
    page.mask_clear_button.click()
    _settle(page, lambda: page.view_model.state.analysis.drawn_mask is None)
    assert page.load_masks(saved) and page.mask_list.count() == 1
    mask_image = np.zeros(SHAPE, dtype=np.float32)
    mask_image[100:110, 100:110] = 1
    assert page.load_masks(save_tiff(tmp_path / "mask.tif", mask_image))
    _settle(page, lambda: page.view_model.state.analysis.drawn_mask is not None
            and page.view_model.state.analysis.drawn_mask[100:110, 100:110].all())
    assert "Mask file: mask.tif" in [page.mask_list.item(index).text() for index in range(page.mask_list.count())]
    page.mirror_fill_check.setChecked(True)
    _settle(page, lambda: page.view_model.state.analysis.filled_pixels is not None)
    filled = page.view_model.state.analysis.filled_pixels
    assert filled[:, 60:66].mean() > 0.5  # the module gap takes the mirror side
    assert "filled from the mirror side" in page.mask_summary_label.text()


def test_each_plot_saves_its_curves_and_the_cake_saves_as_data(page, tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *args, **kwargs: str(tmp_path / "upper")))
    page.save_plot_data("upper")
    names = sorted(path.name for path in (tmp_path / "upper").iterdir())
    assert any(name.endswith("_radial.csv") for name in names) and any(name.endswith("_analysis.json") for name in names)
    page.set_view(2)
    _settle(page, lambda: getattr(page, "_cake", None) is not None)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(tmp_path / "cake.csv"), "")))
    page.save_view_data()
    lines = (tmp_path / "cake.csv").read_text(encoding="ascii").splitlines()
    assert lines[0].startswith("# GIMaP Analyze cake") and any(line.startswith("chi (deg) \\ q") or "\\ q" in line for line in lines)


def test_the_series_tracks_a_peak_frame_by_frame(tmp_path: Path) -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from tests.test_series_map import _ring_frame

    _app()
    paths = [save_tiff(tmp_path / f"t{index:02d}.tif", _ring_frame(300.0, q0=1.05 + 0.01 * index, seed=index)) for index in range(6)]
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        page.tasks.wait(60)
        page.build_series_map()
        _settle(page, lambda: not page._series_queue and page._series_map is not None and page._series_map.rows == 6)
        page._series_pick_q(0.98, 1.18)
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("position"))
        positions = page.series_trace_plot.figure_state()["curves"][0][2]
        assert np.all(np.diff(positions) > 0) and positions[0] == pytest.approx(1.05, abs=0.01)
        written = page.export_series_track(tmp_path / "peaks.csv")
        rows = [line for line in written.read_text(encoding="ascii").splitlines() if not line.startswith("#")]
        assert rows[0] == "frame,file,position,fwhm,area,height" and len(rows) == 7 and rows[1].startswith("1,t00.tif,")
    finally:
        page.tasks.wait(30)
        page.dispose()


def test_the_ai_sets_cut_regions_that_can_be_undone(page) -> None:
    from src.gimap.features.assistant.application import (
        GOALS, PERMISSION_AUTO, AnalysisGoals, RunResults, ToolCall, ToolCatalog, inverse_arguments,
    )
    from src.gimap.features.assistant.presentation import GuiBridge, GuiWorkbench

    catalog = ToolCatalog(GuiWorkbench(page.automation(), GuiBridge()), AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO), RunResults())
    before = catalog.workbench.status()
    assert before["giwaxs"]["regions"] == []
    region = {"name": "PbI2", "q_min": 0.88, "q_max": 0.92, "chi_min_deg": 0, "chi_max_deg": 90, "both_sides": True}
    outcome = catalog.execute(ToolCall("t1", "set_cut_regions", {"regions": [region]}))
    assert not outcome.is_error and "PbI2" in outcome.summary
    _settle(page, lambda: any(row.name == "PbI2" for row in page._region_rows))
    status = catalog.workbench.status()
    assert status["giwaxs"]["regions"][0]["q_min"] == pytest.approx(0.88)
    assert any(item["key"] == "region1_chi" for item in status.get("curves") or [])
    assert inverse_arguments("set_cut_regions", before) == {"regions": []}
    bad = catalog.execute(ToolCall("t2", "set_cut_regions", {"regions": [{"name": "x", "chi_min_deg": -10, "chi_max_deg": 10}]}))
    assert bad.is_error and "both_sides" in bad.content
