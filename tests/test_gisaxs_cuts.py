"""The person's GISAXS cuts: bands, boxes, sectors and lines defined in q, their curves and pixels, typing them in q,
exit angles or pixels, the table of the Cuts step, the q map and the detector view, the export record, and the
question when a frame of another set-up comes. The automatic Yoneda cut that Fitting takes stays exactly as it was."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from src.gimap.features.analyze.application import (
    AnalyzeFrame,
    ExportAnalysis,
    GisaxsCutSettings,
    SaveInstrumentProfile,
    gisaxs_cut_tools as tools,
    settings_from_record,
    settings_record,
)
from src.gimap.features.analyze.application.settings_file import AnalyzeSettings
from src.gimap.features.analyze.domain import (
    GisaxsCut,
    curve_source_mask,
    cut_mask,
    gisaxs_maps,
    horizontal_cut,
    reduce_gisaxs,
    valid_pixels,
    vertical_cut,
)
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.features.analyze.presentation import AnalyzeViewModel
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository, InMemorySettingsRepository
from src.gimap.shared.detector_io import load_detector_image
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.q_inverse import (
    alpha_f_of_qz,
    positions_of_q,
    q_of_positions,
    qz_of_alpha_f,
    two_theta_f_of_qy,
)
from tests.test_analyze_frames_and_center import FakeFrames

DATA = Path(__file__).resolve().parent / "data" / "external"
SHAPE = (200, 300)
GEOMETRY = DetectorGeometry(100e-6, 100e-6, 2.0, 150.3, 190.0, 1.0, 0.3)
"""A small GISAXS set-up: the horizon near row 85, q from about −0.05 to 0.04 Å⁻¹ across."""


def _frame(seed: int = 3) -> np.ndarray:
    rows, columns = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    rng = np.random.default_rng(seed)
    return rng.poisson(40.0 + 0.3 * columns + 0.5 * rows).astype(np.float32)


def _galaxi():
    info = json.loads((DATA / "manifest.json").read_text())["gisaxs_galaxi"]
    image = load_detector_image(DATA / info["frame"])
    data = np.asarray(image.data, dtype=np.float32)
    geometry = DetectorGeometry(info["pixel_size_m"], info["pixel_size_m"], info["distance_m"], *info["beam_center_px"],
                                info["wavelength_angstrom"], info["incidence_deg"])
    return data, valid_pixels(data, image.mask), geometry


CUTS = (
    GisaxsCut("Band", "horizontal", qz=(0.045, 0.048)),
    GisaxsCut("Rod", "vertical", qy=(0.01, 0.012)),
    GisaxsCut("Box", "box", qy=(-0.02, 0.02), qz=(0.04, 0.055), along="qz"),
    GisaxsCut("Sector", "sector", chi_range=(20.0, 40.0)),
    GisaxsCut("Line", "line", start=(-0.02, 0.04), end=(0.02, 0.058), half_width=0.001),
)
"""Above the horizon of ``GEOMETRY`` q_z runs from 0.033 (α_f = 0) to 0.060 Å⁻¹ (row 0)."""


# -- the automatic cut does not change ------------------------------------------------------------


def test_the_yoneda_cut_fitting_takes_is_the_same_with_and_without_cuts_of_the_person() -> None:
    data, valid, geometry = _galaxi()
    alone = reduce_gisaxs(data, valid, geometry)
    with_cuts = reduce_gisaxs(data, valid, geometry, GisaxsCutSettings(regions=CUTS))
    for key in ("horizontal", "vertical"):
        before, after = alone.curve(key), with_cuts.curve(key)
        for name in ("x", "intensity", "sigma", "pixels"):
            assert np.array_equal(getattr(before, name), getattr(after, name)), (key, name)
    # The values of release 218cbe2 (before the cuts of the person existed) on this public frame.
    horizontal, vertical = alone.curve("horizontal"), alone.curve("vertical")
    assert horizontal.region["rows"] == (605, 610) and len(horizontal.x) == 974 and int(horizontal.pixels.sum()) == 4870
    assert horizontal.x[0] == pytest.approx(-0.2776202470340491, rel=1e-12)
    assert horizontal.x[-1] == pytest.approx(0.17859417810246023, rel=1e-12)
    assert float(horizontal.intensity.sum()) == pytest.approx(121654.0, rel=1e-12)
    assert float(horizontal.sigma.sum()) == pytest.approx(3153.876296729858, rel=1e-12)
    assert vertical.region["columns"][0] - geometry.beam_center_x_px >= 15
    manual = reduce_gisaxs(data, valid, geometry, GisaxsCutSettings(vertical_column=geometry.beam_center_x_px)).curve("vertical")
    assert manual.region["columns"] == (592, 603) and len(manual.x) == 638 and int(manual.pixels.sum()) == 7018
    assert float(manual.intensity.sum()) == pytest.approx(278261.0, rel=1e-12)
    # … and as the band functions alone give it (the same rows, the same sampling).
    again = horizontal_cut(data, valid, geometry, center_row=607.5, half_height_px=2.5)
    assert np.array_equal(again.intensity, horizontal.intensity) and np.array_equal(again.sigma, horizontal.sigma)
    beam = vertical_cut(data, valid, geometry, center_column=geometry.beam_center_x_px, half_width_px=5.0)
    assert np.array_equal(beam.intensity, manual.intensity)
    # Where the bands lie in q, for the record of an export: the q_z and α_f of the row edges on the beam column.
    qz = horizontal.region["qz_edges"]
    assert qz[0] < qz[1] and horizontal.region["alpha_f_edges_deg"][0] == pytest.approx(alpha_f_of_qz(geometry, qz[0]))
    assert vertical.region["qy_edges"][0] > 0 and manual.region["qy_edges"][0] < 0 < manual.region["qy_edges"][1]


@pytest.mark.parametrize("geometry", [GEOMETRY, replace(GEOMETRY, rot3_rad=0.22),
                                       replace(GEOMETRY, orientation=4, rot3_rad=-0.12),
                                       GEOMETRY.with_beam_center(SHAPE[1] - 8, GEOMETRY.beam_center_y_px)])
def test_automatic_vertical_band_clears_entire_projected_rod_and_keeps_explicit_column(geometry):
    from src.gimap.features.analyze.application.both_cuts import cut_geometry

    maps = gisaxs_maps(SHAPE, geometry)
    step = geometry.wavevector_inv_angstrom * geometry.pixel_size_x_m / geometry.distance_m
    # A synthetic specular ridge with a one-pixel half-width and a uniform diffuse background.
    image = np.where(np.abs(maps.qy) < step, 10000., 2.)
    valid = np.ones_like(image, dtype=bool)
    auto = reduce_gisaxs(image, valid, geometry)
    assert auto.markers["vertical_source"] == "automatic"
    curve = auto.curve("vertical")
    assert len(curve.x) > 3 and np.all(curve.intensity == 2)
    left, right = curve.region["columns"]
    assert np.min(np.abs(maps.qy[:, left:right][maps.above_horizon[:, left:right]])) > 13 * step
    record = cut_geometry(SimpleNamespace(reduction=auto, geometry=geometry, kind="gisaxs"))
    assert not record["band_crosses_specular"]
    # Place the band at the ridge of an illuminated row. On a rolled detector this is away from the beam column.
    row = 20
    x = float(np.argmin(np.abs(maps.qy[row]))) + 0.5
    manual = reduce_gisaxs(image, valid, geometry, GisaxsCutSettings(vertical_column=x))
    assert manual.curve("vertical").region["center_column"] == x
    assert manual.curve("vertical").intensity.max() > 2
    assert cut_geometry(SimpleNamespace(reduction=manual, geometry=geometry, kind="gisaxs"))["band_crosses_specular"]


def test_a_narrow_frame_reports_limited_clearance_and_add_handles_one_sided_q_ranges():
    from src.gimap.features.analyze.domain.vertical_band import LIMITED_CLEARANCE

    image = np.ones((SHAPE[0], 26))
    reduction = reduce_gisaxs(image, image.astype(bool), GEOMETRY.with_beam_center(10, 190))
    assert LIMITED_CLEARANCE in reduction.warnings
    assert reduction.curve("vertical").region["columns"] == (16, 26)
    for q_range in ((-0.035, -0.03), (-0.001, 0.0005), (-0.04, 0.0001), (0.05, 0.06)):
        cut = tools.new_cut("vertical", GEOMETRY, (q_range, (0.03, 0.05)), name="Band")
        assert q_range[0] <= cut.qy[0] < cut.qy[1] <= q_range[1]
        assert cut.qy[0] * cut.qy[1] > 0
    point = tools.new_cut("vertical", GEOMETRY, ((-0.01, -0.01), (0.03, 0.05)), name="One column")
    assert point.qy[0] < -0.01 < point.qy[1] < 0


# -- the geometry back from q ------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", [
    GEOMETRY,
    replace(GEOMETRY, rot1_rad=0.08, rot2_rad=-0.05, rot3_rad=0.3),
    GEOMETRY.with_critical_angle(0.1),
])
def test_a_q_point_goes_back_to_the_pixel_it_came_from(geometry) -> None:
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, SHAPE[1], 500), rng.uniform(0, 45, 500)  # (with refraction: above α_c, row 50)
    qy, qz = q_of_positions(geometry, x, y)
    back_x, back_y = positions_of_q(geometry, qy, qz)
    assert np.allclose(back_x, x, atol=1e-6) and np.allclose(back_y, y, atol=1e-6)
    assert alpha_f_of_qz(geometry, qz_of_alpha_f(geometry, 0.4)) == pytest.approx(0.4)
    assert np.isnan(positions_of_q(geometry, 50.0, 0.01)[0])  # no ray has this q


@pytest.mark.parametrize("geometry", [GEOMETRY, replace(GEOMETRY, rot1_rad=0.08, rot2_rad=-0.05)])
@pytest.mark.parametrize("unit", tools.UNITS)
def test_a_value_typed_in_any_unit_comes_back_as_typed(geometry, unit) -> None:
    for axis, values in (("qz", [0.02, 0.031, 0.05]), ("qy", [-0.03, 0.0, 0.02])):
        shown = tools.axis_to_unit(geometry, unit, axis, values)
        assert np.allclose(tools.axis_from_unit(geometry, unit, axis, shown), values, atol=1e-9)
    # Pixels: a q_z is the row of the beam-centre column, a q_y the column of the horizon row.
    row = tools.axis_to_unit(geometry, tools.UNIT_PIXEL, "qz", 0.03)[0]
    assert q_of_positions(geometry, geometry.beam_center_x_px, row)[1] == pytest.approx(0.03, abs=1e-9)
    assert tools.width_from_unit(geometry, unit, tools.width_to_unit(geometry, unit, 0.002)) == pytest.approx(0.002)
    assert two_theta_f_of_qy(geometry, 0.0) == 0.0


# -- the cuts' pixels and curves -------------------------------------------------------------------


def _maps_and_usable(geometry=GEOMETRY, frame=None):
    frame = _frame() if frame is None else frame
    maps = gisaxs_maps(SHAPE, geometry)
    return frame, maps, np.ones(SHAPE, dtype=bool) & maps.above_horizon


def test_a_band_averages_the_whole_pixels_whose_centre_q_is_inside_one_point_per_column() -> None:
    frame, maps, usable = _maps_and_usable()
    reduction = reduce_gisaxs(frame, np.ones(SHAPE, dtype=bool), GEOMETRY, GisaxsCutSettings(regions=CUTS))
    band = reduction.curve("cut1")
    inside = usable & (maps.qz >= 0.045) & (maps.qz <= 0.048)
    assert band.x_label == "qy (Å⁻¹)" and band.region["gisaxs_cut"] == "Band" and band.region["qz"] == [0.045, 0.048]
    assert int(band.pixels.sum()) == int(inside.sum()) == band.region["pixels_in_cut"]
    column = int(np.nonzero(inside.any(axis=0))[0][7])
    pixels = frame[:, column][inside[:, column]].astype(np.float64)
    point = int(np.argmin(np.abs(band.x - maps.qy[:, column][inside[:, column]].mean())))
    assert band.intensity[point] == pytest.approx(pixels.mean())
    assert band.sigma[point] == pytest.approx(math.sqrt(pixels.sum()) / pixels.size)  # Poisson: √Σ/n
    assert band.region["alpha_f_deg"] == pytest.approx(list(alpha_f_of_qz(GEOMETRY, np.array([0.045, 0.048]))))
    rod = reduction.curve("cut2")
    assert rod.x_label == "qz (Å⁻¹)" and int(rod.pixels.sum()) == int((usable & (maps.qy >= 0.01) & (maps.qy <= 0.012)).sum())
    box = reduction.curve("cut3")
    assert box.x_label == "qz (Å⁻¹)" and np.all(np.diff(box.x) < 0) | np.all(np.diff(box.x) > 0)


@pytest.mark.parametrize("key, cut", [("cut4", CUTS[3]), ("cut5", CUTS[4])])
def test_sectors_and_lines_bin_whole_pixels_and_keep_their_mean(key, cut) -> None:
    frame, maps, usable = _maps_and_usable()
    curve = reduce_gisaxs(frame, np.ones(SHAPE, dtype=bool), GEOMETRY, GisaxsCutSettings(regions=CUTS)).curve(key)
    inside = cut_mask(cut, maps, usable)
    assert inside.sum() > 50 and int(curve.pixels.sum()) == int(inside.sum())
    weighted = float((curve.intensity * curve.pixels).sum() / curve.pixels.sum())
    assert weighted == pytest.approx(float(frame[inside].astype(np.float64).mean()), rel=1e-9)
    step = np.diff(curve.x)
    assert np.all(step > 0) and float(np.median(step)) >= tools.q_per_pixel(GEOMETRY) * 0.99
    if cut.kind == "line":
        assert curve.x_label == "s (Å⁻¹)" and curve.x.max() <= math.hypot(0.04, 0.018) + 1e-9
    else:
        assert curve.x_label == "qyz (Å⁻¹)"


def test_the_errors_follow_the_frame_s_error_model() -> None:
    frame, maps, usable = _maps_and_usable()
    valid = np.ones(SHAPE, dtype=bool)
    settings = GisaxsCutSettings(regions=CUTS[3:4])
    counted = reduce_gisaxs(frame, valid, GEOMETRY, settings).curve("cut1")
    scatter = reduce_gisaxs(frame, valid, GEOMETRY, settings, counts=False).curve("cut1")
    doubled = reduce_gisaxs(frame, valid, GEOMETRY, settings, counts=np.full(SHAPE, 4.0)).curve("cut1")
    assert np.array_equal(counted.intensity, scatter.intensity) and not np.allclose(counted.sigma, scatter.sigma)
    assert np.allclose(doubled.sigma, 2.0 * counted.sigma)  # a pixel's variance is factor × value


def test_a_cut_shows_its_own_pixels_under_sources() -> None:
    frame, maps, usable = _maps_and_usable()
    valid = np.ones(SHAPE, dtype=bool)
    reduction = reduce_gisaxs(frame, valid, GEOMETRY, GisaxsCutSettings(regions=CUTS))
    for index in range(len(CUTS)):
        curve = reduction.curve(f"cut{index + 1}")
        assert int(curve_source_mask(curve, valid, maps).sum()) == int(curve.pixels.sum())


def test_a_cut_without_pixels_says_so_and_bad_cuts_are_refused() -> None:
    above_frame = GisaxsCut("High", "horizontal", qz=(5.0, 5.1))
    reduction = reduce_gisaxs(_frame(), np.ones(SHAPE, dtype=bool), GEOMETRY, GisaxsCutSettings(regions=(above_frame,)))
    assert reduction.curve("cut1").is_empty and "Cut “High”: no valid pixels above the horizon in it." in reduction.warnings
    with pytest.raises(ValueError, match="the qz range is empty"):
        GisaxsCut("Flat", "horizontal", qz=(0.03, 0.03))
    with pytest.raises(ValueError, match="no length"):
        GisaxsCut("Dot", "line", start=(0.0, 0.03), end=(0.0, 0.03), half_width=0.001)


def test_outlines_lie_where_the_pixels_are_on_the_detector_and_the_q_map() -> None:
    extent = ((-0.05, 0.04), (0.033, 0.06))
    for cut in CUTS:
        for qy, qz in tools.q_outlines(cut, extent):
            assert qy.size >= 2 and not np.any((np.diff(qy) == 0) & (np.diff(qz) == 0))  # no zero-length step
        for x, y in tools.detector_outlines(cut, GEOMETRY, extent):
            back_qy, back_qz = q_of_positions(GEOMETRY, x, y)
            assert np.all(np.isfinite(back_qy)) and x.size >= 2
    band = tools.detector_outlines(CUTS[0], GEOMETRY, extent)[0]
    _qy, qz = q_of_positions(GEOMETRY, *band)
    assert qz.min() == pytest.approx(0.045, abs=1e-9) and qz.max() == pytest.approx(0.048, abs=1e-9)


def test_the_cuts_live_in_a_settings_file_and_a_set_up_change_is_named() -> None:
    record = settings_record(AnalyzeSettings(gisaxs=GisaxsCutSettings(horizontal_row=40.0, regions=CUTS)))
    assert record["gisaxs"]["cuts"][4] == {"name": "Line", "kind": "line", "start": [-0.02, 0.04], "end": [0.02, 0.058],
                                           "half_width": 0.001}
    assert settings_from_record(json.loads(json.dumps(record))).gisaxs.regions == CUTS
    analysis = SimpleNamespace(geometry=GEOMETRY, detector_name="Pilatus", shape=SHAPE)
    setup = tools.cut_setup(analysis)
    assert tools.setup_changes(setup, tools.cut_setup(analysis)) == []
    moved = SimpleNamespace(geometry=GEOMETRY.with_beam_center(160.0, 190.0).with_incidence(0.5), detector_name="Eiger",
                            shape=SHAPE)
    assert tools.setup_changes(setup, tools.cut_setup(moved)) == ["detector", "beam centre"]  # αi is the sample's


# -- the Cuts step ---------------------------------------------------------------------------------


class TwoDetectors(FakeFrames):
    """``b.cbf`` comes from another detector."""

    def load(self, path, frame_index=0):
        image = super().load(path, frame_index)
        return SimpleNamespace(**{**vars(image), "detector_name": "Other" if Path(path).name == "b.cbf" else "Test"})


@pytest.fixture
def page(tmp_path, qt_application, monkeypatch):
    del qt_application
    from src.gimap.features.analyze.presentation import AnalyzePage
    from src.gimap.features.analyze.presentation.bindings import gisaxs_cuts

    monkeypatch.setattr(gisaxs_cuts, "show_toast", lambda *_args, **_options: None)
    frames = TwoDetectors({"a.cbf": [_frame(1)], "b.cbf": [_frame(2)]})
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("Test", GEOMETRY, "Test", SHAPE)])
    settings = InMemorySettingsRepository({})
    model = AnalyzeViewModel(
        analyze_frame=AnalyzeFrame(frames, profiles), export_analysis=ExportAnalysis(CsvCurveWriter()),
        save_profile=SaveInstrumentProfile(profiles), frames=frames, profiles=profiles, read_setting=settings.get,
        settings=settings,
    )
    model.set_mode("gisaxs")
    model.set_profile("Test")
    widget = AnalyzePage(model)
    widget.resize(1500, 900)
    widget.add_paths([tmp_path / "a.cbf", tmp_path / "b.cbf"])
    assert widget.tasks.wait(20)
    widget.show_step("cuts")
    yield widget
    widget.tasks.wait(20)
    widget.dispose()
    widget.close()


def _settle(page) -> None:
    assert page.tasks.wait(20)
    page.tasks.wait(0.2)


def test_the_table_starts_with_the_automatic_cuts_and_adds_typed_cuts(page) -> None:
    assert [row.key for row in page._cut_rows] == ["horizontal", "vertical"]
    assert page.cut_table.item(0, 0).text() == "Yoneda cut" and not page.cut_remove_button.isEnabled()
    assert page._plot_keys == {"top": ["horizontal"], "bottom": ["vertical"]}
    for kind in ("horizontal", "vertical", "box", "sector", "line"):
        page.add_gisaxs_cut(kind)
        _settle(page)
    reduction = page.view_model.state.analysis.reduction
    assert [curve.key for curve in reduction.curves] == ["horizontal", "vertical", "cut1", "cut2", "cut3", "cut4", "cut5"]
    assert page.cut_table.rowCount() == 7 and page.cut_table.currentRow() == 6  # the cut added last is selected
    assert page._plot_keys["top"] == ["horizontal", "cut1", "cut3"] and page._plot_keys["bottom"] == ["vertical", "cut2"]
    assert not page.lower_choice.isHidden()  # sectors and lines are in the lower plot's list
    page.lower_choice.setCurrentIndex(page.lower_choice.findData("g_sector"))
    page._lower_profile_chosen(page.lower_choice.currentIndex())
    assert page._plot_keys["bottom"] == ["cut4"] and page.bottom_plot.plot.getAxis("bottom").labelText.startswith("qyz")


def test_a_band_is_typed_in_pixels_and_kept_in_q(page) -> None:
    page.add_gisaxs_cut("horizontal")
    _settle(page)
    page.cut_unit_combo.setCurrentIndex(page.cut_unit_combo.findData("pixel"))
    assert page.cut_row_units[0].text() == "px" and page.cut_row_captions[0].text() == "y"
    page.cut_spins[1].setValue(3.0)  # the half width first, then the centre (each change analyses again)
    _settle(page)
    page.cut_spins[0].setValue(40.0)
    _settle(page)
    cut = page.view_model.gisaxs_cuts[0]
    rows = tools.range_to_unit(GEOMETRY, tools.UNIT_PIXEL, "qz", cut.qz)
    assert rows == pytest.approx((37.0, 43.0), abs=1e-6)
    _qy, qz = q_of_positions(GEOMETRY, [GEOMETRY.beam_center_x_px] * 2, [43.0, 37.0])
    assert cut.qz == pytest.approx(tuple(qz), abs=1e-12)
    page.cut_unit_combo.setCurrentIndex(page.cut_unit_combo.findData("angle"))
    assert page.cut_row_captions[0].text() == "αf"
    assert page.cut_spins[0].value() == pytest.approx(float(alpha_f_of_qz(GEOMETRY, sum(cut.qz) / 2)), abs=1e-4)


def test_the_automatic_yoneda_cut_is_typed_as_its_rows_and_duplicated_into_a_cut_in_q(page) -> None:
    page.cut_table.selectRow(0)
    page.cut_unit_combo.setCurrentIndex(page.cut_unit_combo.findData("pixel"))
    page.cut_spins[0].setValue(50.5)
    _settle(page)
    assert page.view_model.state.gisaxs.horizontal_row == pytest.approx(50.5)
    assert page.view_model.state.analysis.reduction.curve("horizontal").region["rows"] == (48, 53)
    page.cut_table.selectRow(0)
    page.duplicate_gisaxs_cut()
    _settle(page)
    copy = page.view_model.gisaxs_cuts[0]
    assert copy.kind == "horizontal" and copy.name == "Yoneda cut (2)"
    assert tools.range_to_unit(GEOMETRY, tools.UNIT_PIXEL, "qz", copy.qz) == pytest.approx((48.0, 53.0), abs=1e-6)


def test_rename_hide_remove_and_undo(page) -> None:
    page.add_gisaxs_cut("vertical")
    _settle(page)
    page.cut_table.item(2, 0).setText("Bragg rod")
    _settle(page)
    assert page.view_model.gisaxs_cuts[0].name == "Bragg rod"
    assert page.view_model.state.analysis.reduction.curve("cut1").region["gisaxs_cut"] == "Bragg rod"
    page.cut_table.item(2, 0).setCheckState(Qt.Unchecked)
    assert page._plot_keys["bottom"] == ["vertical"]
    page.cut_table.item(0, 0).setCheckState(Qt.Unchecked)  # the automatic band too: its curve and its band
    assert page._plot_keys["top"] == [] and not page.detector_view.horizontal_band.isVisible()
    page.cut_table.item(0, 0).setCheckState(Qt.Checked)
    page.cut_table.selectRow(2)
    page.remove_gisaxs_cut()
    _settle(page)
    assert page.view_model.gisaxs_cuts == () and page.cut_table.rowCount() == 2
    assert page.undo_setup()
    _settle(page)
    assert [cut.name for cut in page.view_model.gisaxs_cuts] == ["Bragg rod"]


def test_cuts_are_drawn_on_the_detector_and_the_q_map_and_drawn_by_clicks(page) -> None:
    page.set_view(1)  # the q map: points are (qy, qz)
    page.draw_gisaxs_cut("box")
    assert page.cut_draw_buttons["box"].isChecked()
    page.shape_layer._finish([(-0.01, 0.04), (0.01, 0.05)])
    _settle(page)
    box = page.view_model.gisaxs_cuts[0]
    assert box.kind == "box" and box.qy == pytest.approx((-0.01, 0.01)) and box.qz == pytest.approx((0.04, 0.05))
    assert not page.cut_draw_buttons["box"].isChecked()
    assert "g:cut1" in page.shape_layer.rect_keys()  # draggable on the q map
    page._cut_rect_moved("g:cut1", -0.02, 0.04, 0.0, 0.055)
    _settle(page)
    assert page.view_model.gisaxs_cuts[0].qz == pytest.approx((0.04, 0.055))
    page.set_view(0)  # the detector: a click at a pixel is the q there
    page.draw_gisaxs_cut("line")
    page.shape_layer._finish([(100.0, 60.0), (220.0, 20.0)])
    _settle(page)
    line = page.view_model.gisaxs_cuts[1]
    assert line.start == pytest.approx(q_of_positions(GEOMETRY, 100.0, 60.0)) and line.kind == "line"
    colors = {item.opts["pen"].color().name() for item in page.shape_layer._outlines}
    assert {row.color for row in page._cut_rows[2:]} <= colors


def test_the_export_states_each_cut_in_q(page, tmp_path: Path) -> None:
    page.add_gisaxs_cut("vertical")
    _settle(page)
    written = page.view_model._export(page.view_model.state.analysis, tmp_path)
    cut = next(path for path in written if path.name.endswith("_cut1.csv"))
    region = json.loads(next(line for line in cut.read_text(encoding="utf-8").splitlines()
                              if line.startswith("# region: "))[len("# region: "):])
    assert region["gisaxs_cut"] == "Vertical 1" and region["kind"] == "vertical" and region["q_unit"] == "1/A"
    assert region["qy"] == pytest.approx(list(page.view_model.gisaxs_cuts[0].qy)) and "two_theta_f_deg" in region
    horizontal = next(path for path in written if path.name.endswith("_horizontal.csv")).read_text(encoding="utf-8")
    assert '"qz_edges"' in horizontal and '"alpha_f_edges_deg"' in horizontal


def test_a_frame_of_another_detector_asks_before_the_cuts_are_kept(page) -> None:
    page.add_gisaxs_cut("horizontal")
    _settle(page)
    assert page.cut_notice.isHidden()
    page.file_list.setCurrentRow(1)  # b.cbf: another detector
    _settle(page)
    assert not page.cut_notice.isHidden() and "detector" in page.cut_notice_label.text()
    assert page.view_model.state.analysis.reduction.curve("cut1") is not None  # in q, meanwhile applied
    page.cut_drop_button.click()
    _settle(page)
    assert page.cut_notice.isHidden() and page.view_model.gisaxs_cuts == ()
    page.add_gisaxs_cut("vertical")  # made on b's set-up …
    _settle(page)
    page.file_list.setCurrentRow(0)  # … asked again on a's
    _settle(page)
    assert not page.cut_notice.isHidden()
    page.cut_keep_button.click()
    assert page.cut_notice.isHidden() and len(page.view_model.gisaxs_cuts) == 1
    page.file_list.setCurrentRow(1)
    _settle(page)
    assert not page.cut_notice.isHidden()  # b is another set-up than the one the cuts were kept for


def test_every_text_of_the_cuts_panel_has_its_chinese(page) -> None:
    from PyQt5.QtWidgets import QAbstractButton, QComboBox, QLabel

    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.analyze.application.gisaxs_cuts import CUT_TEMPLATES
    from src.gimap.features.analyze.presentation.bindings import gisaxs_cut_display as display
    from src.gimap.features.analyze.presentation.bindings import gisaxs_cuts as cuts

    texts = set()
    for root in (page.gisaxs_cuts_panel, page.cut_notice):
        for widget in [root, *root.findChildren(object)]:
            if isinstance(widget, (QLabel, QAbstractButton)):
                texts.update((widget.text(), widget.toolTip()))
            if isinstance(widget, QComboBox):
                texts.update(widget.itemText(index) for index in range(widget.count()))
    texts.update(action.text() for action in page.cut_add_actions.values())
    texts.update(page.cut_table.horizontalHeaderItem(column).text() for column in range(2))
    texts.update((*cuts.KIND_WORDS.values(), *cuts.AUTO_NAMES.values(), *display.DRAW_HINTS.values()))
    texts.update((*(title for _key, title in display.LOWER_GISAXS), *display.LOWER_TITLES.values(), *CUT_TEMPLATES))
    plain = {"", "±", "q (Å⁻¹)", "Å⁻¹", "qy", "qz", "Add", "Cuts", "Name", "Sector", "Remove", "Half width"}
    missing = sorted(text for text in texts if text not in plain and not set(text) <= set("0123456789.±–, ()")
                     and text not in ZH and not text.startswith("= "))
    assert missing == []
