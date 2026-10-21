"""One cut for a whole series (decision D4), and the frames a series run works on.

When the cut is automatic — the Yoneda band of a GISAXS frame, the strongest ring of a GIWAXS frame — every
frame of a Series map, a Batch Export and a watched folder uses the cut of the frame on screen, unless “Find the
cut again in every frame” is ticked. The Sum N that Run Automatic Analysis sets does not leak into series work,
a last sum of fewer frames is known as such, and the halves the automatic analysis chooses for one sample are
not remembered for the next.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import settings_from_record, settings_record
from src.gimap.features.analyze.application.settings_file import AnalyzeSettings
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.features.analyze.presentation import frames_model
from src.gimap.features.analyze.presentation.frames_model import SumGroup
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.detector_io import forget_layout
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_nexus_sources import _ring_stack
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, save_tiff
from tests.test_review_analyze_run_and_clear import _profiles
from tests.test_series_map import _wait

GISAXS_SHAPE = (300, 240)
GISAXS_GEOMETRY = DetectorGeometry(172e-6, 172e-6, 1.0, 120.0, 260.0, 1.0, 0.4)
BANDS = (204, 204, 208, 200, 206)
"""The detector row of the brightest diffuse band in each frame of the synthetic GISAXS series."""


@pytest.fixture(autouse=True)
def _english():
    forget_layout()
    apply_language("en")
    yield
    apply_language("en")
    forget_layout()


def _gisaxs_frame(band: int, seed: int = 0) -> np.ndarray:
    rows, columns = np.indices(GISAXS_SHAPE)
    image = 4.0 + 500.0 * np.exp(-np.abs(columns - 120) / 50.0) * np.exp(-0.5 * ((rows - band) / 2.5) ** 2)
    return np.random.default_rng(seed).poisson(image).astype(np.float32)


def _gisaxs_profiles() -> InMemoryInstrumentProfileRepository:
    return InMemoryInstrumentProfileRepository([InstrumentProfile("small GISAXS", GISAXS_GEOMETRY, None, GISAXS_SHAPE)])


def _gisaxs_files(folder: Path, bands=BANDS) -> list[Path]:
    return [save_tiff(folder / f"film_{index:02d}.tif", _gisaxs_frame(band, seed=index)) for index, band in enumerate(bands)]


def _two_ring_frame(second: float, seed: int = 0) -> np.ndarray:
    """A ring at q = 1.1 of constant height and one at q = 1.5 of height ``second``."""
    import math

    rows, columns = np.indices(SHAPE)
    radius = np.hypot((columns + 0.5 - CENTER[0]) * PIXEL, (rows + 0.5 - CENTER[1]) * PIXEL)
    q = 4 * math.pi * np.sin(np.arctan2(radius, DISTANCE_M) / 2) / WAVELENGTH
    image = 30.0 + 300.0 * np.exp(-4 * np.log(2) * (q - 1.1) ** 2 / 0.04**2) + second * np.exp(-4 * np.log(2) * (q - 1.5) ** 2 / 0.04**2)
    return np.random.default_rng(seed).poisson(image).astype(np.float32)


def _shown(model, mode: str, paths, index: int = 0):
    """The view model with ``paths`` listed and frame ``index`` analysed, as after it was opened."""
    model.set_mode(mode)
    model.add_paths([str(path) for path in paths])
    model.select(index)
    analysis = model.analyze(model.request())
    model.accept(analysis)
    return analysis


def _rows_of(model, requests) -> list[tuple]:
    return [tuple(model.analyze(request).reduction.curve("horizontal").region["rows"]) for request in requests]


# -- the pin of the view model ----------------------------------------------------------------------------------------


def test_every_frame_of_a_gisaxs_series_is_cut_where_the_frame_on_screen_is(tmp_path: Path) -> None:
    context = _context(_gisaxs_profiles())
    model = create_analyze_view_model(context)
    shown = _shown(model, "gisaxs", _gisaxs_files(tmp_path))
    low, high = shown.reduction.markers["horizontal_band"]
    assert (low, high) == (202.0, 207.0) and shown.reduction.markers["horizontal_source"] == "yoneda"
    assert model.pin_series_cuts is True
    cut = model.series_cut()
    assert cut == {"technique": "gisaxs", "horizontal_rows": (low, high),
                   "vertical_columns": tuple(shown.reduction.markers["vertical_band"]), "frame": (shown.path, 0)}
    # The rows as the Cuts step counts them: the band 202.0 … 207.0 covers the pixel rows 202–206.
    assert model.series_cut_text() == ("horizontal cut: rows {low}–{high} for every frame", {"low": "202", "high": "206"})
    requests = model.batch_requests()
    assert len(requests) == 5 and {request.gisaxs.horizontal_row for request in requests} == {0.5 * (low + high)}
    assert {request.gisaxs.vertical_column for request in requests} == {120.0}
    assert set(_rows_of(model, requests)) == {(202, 207)}  # one cut: the rows of the frame on screen

    model.set_pin_series_cuts(False)  # “Find the cut again in every frame”
    assert model.pin_series_cuts is False and model.series_cut() is None
    assert model.series_cut_text() == ("horizontal cut: found again in each frame", {})
    again = model.batch_requests()
    assert {request.gisaxs.horizontal_row for request in again} == {None}
    assert _rows_of(model, again) == [(band - 2, band + 3) for band in BANDS]  # each frame its own Yoneda rows
    assert context.settings.get("analyze", "pin_series_cuts") is False  # saved with Analyze's settings
    assert create_analyze_view_model(context).pin_series_cuts is False
    model.set_pin_series_cuts(True)
    assert context.settings.get("analyze", "pin_series_cuts") is True and model.series_cut() == cut


def test_a_cut_the_person_set_is_kept_and_said_as_theirs(tmp_path: Path) -> None:
    model = create_analyze_view_model(_context(_gisaxs_profiles()))
    _shown(model, "gisaxs", _gisaxs_files(tmp_path))
    model.set_horizontal_band(190.0, 196.0)
    model.accept(model.analyze(model.request()))
    cut = model.series_cut()  # the vertical cut is still automatic: only that one is pinned
    assert cut["horizontal_rows"] is None and cut["vertical_columns"] == (115.0, 125.0)
    assert model.series_cut_text() == ("cuts as set by you, the same for every frame", {})
    assert {request.gisaxs.horizontal_row for request in model.batch_requests()} == {193.0}
    model.set_vertical_band(100.0, 110.0)
    model.accept(model.analyze(model.request()))
    assert model.series_cut() is None  # every cut is the person's: nothing left to pin
    assert model.series_cut_text() == ("cuts as set by you, the same for every frame", {})
    empty = create_analyze_view_model(_context(_gisaxs_profiles()))
    assert empty.series_cut() is None and empty.series_cut_text() == ("", {})  # no frame on screen: nothing to say


def test_the_ring_of_a_giwaxs_series_does_not_change_when_another_ring_overtakes_it(tmp_path: Path) -> None:
    paths = [save_tiff(tmp_path / f"ring_{index}.tif", _two_ring_frame(height, seed=index))
             for index, height in enumerate((60.0, 150.0, 700.0, 900.0))]
    model = create_analyze_view_model(_context(_profiles()))
    shown = _shown(model, "giwaxs", paths)
    first = tuple(shown.reduction.markers["chi_q_window"])
    assert first[0] < 1.1 < first[1]
    cut = model.series_cut()
    assert cut == {"technique": "giwaxs", "chi_q_window": first, "frame": (shown.path, 0)}
    template, values = model.series_cut_text()
    assert template == "I(χ) ring: q {low}–{high} Å⁻¹ for every frame"
    assert values == {"low": f"{first[0]:.3f}", "high": f"{first[1]:.3f}"}
    pinned = [model.analyze(request).reduction.markers["chi_q_window"] for request in model.batch_requests()]
    assert {tuple(window) for window in pinned} == {first}
    assert model.analyze(model.batch_requests()[-1]).reduction.curve("azimuthal").title == shown.reduction.curve("azimuthal").title

    model.set_pin_series_cuts(False)
    assert model.series_cut_text() == ("I(χ) ring: the strongest ring of each frame", {})
    free = [model.analyze(request).reduction.markers["chi_q_window"] for request in model.batch_requests()]
    assert free[0][0] < 1.1 < free[0][1] and free[-1][0] < 1.5 < free[-1][1]  # the ring changes along the series
    model.set_pin_series_cuts(True)
    model.set_chi_window(1.45, 1.55)
    model.accept(model.analyze(model.request()))
    assert model.series_cut() is None and model.series_cut_text() == ("cuts as set by you, the same for every frame", {})


def test_the_frames_a_watched_folder_brings_are_cut_like_the_frame_on_screen(tmp_path: Path) -> None:
    paths = _gisaxs_files(tmp_path / "run")
    later = tmp_path / "later"
    model = create_analyze_view_model(_context(_gisaxs_profiles()))
    _shown(model, "gisaxs", paths[:1])
    model.start_watch(tmp_path / "run")
    centre = 0.5 * sum(model.series_cut()["horizontal_rows"])
    new = [save_tiff(tmp_path / "run" / f"film_{index:02d}.tif", _gisaxs_frame(band, seed=index))
           for index, band in ((7, 210), (8, 199))]
    requests = model._watch_groups([(path, 0) for path in new])
    assert [request.gisaxs.horizontal_row for request in requests] == [centre, centre]
    # The frame on screen follows the newest one (its own automatic cut): the watch keeps the cut it began with.
    model.select(len(model.state.files) - 1)
    model.accept(model.analyze(model.request()))
    more = model._watch_groups([(save_tiff(later / "film_09.tif", _gisaxs_frame(215, seed=9)), 0)])
    assert [request.gisaxs.horizontal_row for request in more] == [centre]
    model.reset_watch_groups()  # a watch begun anew takes the cut of the frame then on screen
    assert model._watch_groups([(new[0], 0)])[0].gisaxs.horizontal_row != centre


def test_the_pin_is_part_of_a_settings_file() -> None:
    """A settings file (Save Settings, a project, the memory of the session) says the pin in force, loading it
    sets the pin, and a file that does not say it — written before, or by something that does not know the
    pin — leaves the pin as it is (review: the record said ``true`` while the pin was off)."""
    assert AnalyzeSettings().pin_series_cuts is None and "pin_series_cuts" not in settings_record(AnalyzeSettings())
    record = settings_record(AnalyzeSettings(pin_series_cuts=False))
    assert record["pin_series_cuts"] is False and settings_from_record(record).pin_series_cuts is False
    del record["pin_series_cuts"]
    assert settings_from_record(record).pin_series_cuts is None

    context = _context(_gisaxs_profiles())
    model = create_analyze_view_model(context)
    model.set_pin_series_cuts(False)
    saved = settings_record(model.current_settings())
    assert saved["pin_series_cuts"] is False
    model.set_pin_series_cuts(True)
    model.apply_settings(settings_from_record(saved))
    assert model.pin_series_cuts is False and context.settings.get("analyze", "pin_series_cuts") is False
    del saved["pin_series_cuts"]
    model.apply_settings(settings_from_record(saved))  # (said nothing of it: as it was)
    assert model.pin_series_cuts is False
    model.set_pin_series_cuts(None)  # (nothing said: nothing changed)
    assert model.pin_series_cuts is False
    model.set_pin_series_cuts(True)


def test_a_changed_geometry_of_the_profile_marks_a_map_as_built_with_other_settings(tmp_path: Path) -> None:
    """Geometry ▸ a profile entered by hand under the same name: every q of a new map scales, so the map is no
    longer what Build Map would give. Saving the same geometry again (only its time and source change) is no
    change."""
    from dataclasses import replace

    model = create_analyze_view_model(_context(_gisaxs_profiles()))
    shown = _shown(model, "gisaxs", _gisaxs_files(tmp_path))
    assert shown.resolution.profile.name == "small GISAXS"
    built = model.series_built_with()
    assert not model.series_settings_changed(built)
    model.save_profile("small GISAXS", replace(GISAXS_GEOMETRY, distance_m=1.5), source="entered by hand")
    assert model.series_settings_changed(built)  # at once, before the frame on screen is analysed again
    model.accept(model.analyze(model.request()))
    assert model.analyze(model.request()).geometry.distance_m == 1.5 and model.series_settings_changed(built)
    built = model.series_built_with()
    model.save_profile("small GISAXS", replace(GISAXS_GEOMETRY, distance_m=1.5), source="saved again")
    assert not model.series_settings_changed(built)


def test_a_request_that_can_carry_the_cut_of_the_series_is_given_it() -> None:
    """``with_cut`` names where a pinned cut comes from in a request that has room for it (``series_cut``), so
    a frame's record can say “the cut of the series” instead of “set by hand” once the analysis passes it on."""
    from dataclasses import dataclass, field

    from src.gimap.features.analyze.application import GisaxsCutSettings

    @dataclass(frozen=True)
    class Request:
        gisaxs: GisaxsCutSettings = field(default_factory=GisaxsCutSettings)
        series_cut: object = None

    cut = {"technique": "gisaxs", "horizontal_rows": (202.0, 207.0), "vertical_columns": None, "frame": (Path("a"), 0)}
    pinned = frames_model.with_cut(Request(), cut)
    assert pinned.gisaxs.horizontal_row == 204.5 and pinned.series_cut == cut
    assert frames_model.with_cut(Request(), None).series_cut is None


def test_a_series_run_takes_only_the_frames_a_growing_source_has_handed_out(tmp_path: Path) -> None:
    """While a watched source grows (live following), its newest frames may still be being written: a series
    run started now takes the frames ``frames_to_run`` allows, and a sum on screen sums no frame beyond them."""
    model = create_analyze_view_model(_context(_profiles()))
    model.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
    path = model.state.files[0]
    assert len(model.batch_requests()) == 12
    model.frames_to_run = lambda source: 7  # (as for a source the watch follows: frames 0–6 handed out)
    assert [request.frame_index for request in model.batch_requests()] == list(range(7))
    model.set_sum_count(5)
    assert [index for _path, index in model.summed_frames_for(path, 5)] == [6]
    assert model.summed_frames_for(path, 8) == ()  # a frame not handed out yet sums nothing further
    assert [group.frames for group in model.batch_groups()] == [5, 2]


# -- sums ---------------------------------------------------------------------------------------------------------------


def test_the_groups_of_a_sum_say_how_many_frames_each_has(tmp_path: Path) -> None:
    model = create_analyze_view_model(_context(_profiles()))
    model.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
    assert [group.frames for group in model.batch_groups()] == [1] * 12 and model.sum_set_by == "user"
    model.set_sum_count(5)
    groups = model.batch_groups()
    assert [(group.first[1], group.frames, group.size, group.complete) for group in groups] == [
        (0, 5, 5, True), (5, 5, 5, True), (10, 2, 5, False)]
    assert isinstance(groups[0], SumGroup) and len(model.batch_requests()) == len(groups) == 3
    numbers = model.frame_numbers()
    assert numbers[frames_model.source_key(model.state.files[0]), 10] == 11 and len(numbers) == 12


def test_a_sum_of_the_automatic_analysis_is_known_as_such(tmp_path: Path) -> None:
    model = create_analyze_view_model(_context(_profiles()))
    model.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
    model.set_sum_count(10, by="automatic")
    assert (model.sum_count, model.sum_set_by) == (10, "automatic")
    model.set_sum_count(3)  # the Sum box
    assert (model.sum_count, model.sum_set_by) == (3, "user")
    model.set_sum_count(10, by="automatic")
    model.set_sum_count(1, by="automatic")  # no sum is nobody's
    assert model.sum_set_by == "user"


# -- the halves the automatic analysis chooses (critic 14) ----------------------------------------------------------------


def test_halves_chosen_for_one_sample_are_not_remembered_for_the_next(tmp_path: Path) -> None:
    context = _context(_gisaxs_profiles())
    model = create_analyze_view_model(context)
    paths = _gisaxs_files(tmp_path)
    _shown(model, "gisaxs", paths)
    assert model.fit_side == "both_abs"
    model.set_fit_side("positive", remember=False)  # as the automatic analysis does for the frame it works on
    assert model.fit_side == "positive" and context.settings.get("analyze", "fit_side", "both_abs") == "both_abs"
    model.select(1)  # another file: the stored choice again
    assert model.fit_side == "both_abs"
    model.set_fit_side("mean", remember=False)
    model.clear_files()
    assert model.fit_side == "both_abs"
    model.set_fit_side("negative")  # the person's own choice is kept, as before
    assert context.settings.get("analyze", "fit_side") == "negative" and model.fit_side == "negative"
    model.set_fit_side("mean", remember=False)
    model.set_fit_side("positive")  # … and replaces what an analysis chose for this frame
    assert model.fit_side == "positive"


# -- the page ---------------------------------------------------------------------------------------------------------


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _gisaxs_page(tmp_path: Path, bands=BANDS):
    page = _page(_context(_gisaxs_profiles()))
    page.set_mode_choice("gisaxs")
    page.add_paths([str(path) for path in _gisaxs_files(tmp_path, bands)])
    _done(page)
    return page


def _build(page, rows: int) -> None:
    page.build_series_map()
    _wait(page, lambda: page._series_map is not None and page._series_map.rows == rows and not page.batch_running())
    _done(page)


def test_a_map_is_built_on_one_cut_and_says_which(tmp_path: Path) -> None:
    page = _gisaxs_page(tmp_path)
    try:
        assert page.series_curve_combo.currentData() == "horizontal"
        assert "horizontal cut: rows 202–206 for every frame" in page.series_info_label.text()
        _build(page, 5)
        series = page._series_map
        assert np.isfinite(series.image).all() and series.x.size == 240  # every row on the same columns
        assert series.built_with["cut"]["horizontal_rows"] == (202.0, 207.0)
        assert "horizontal cut: rows 202–206 for every frame" in page.series_title_label.text()
        # Open shows the frame of a row with the cut of the map, not with the one found in it.
        page._series_pick_row(2.5)
        assert page.open_series_row()
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.path.name == "film_02.tif"
        assert tuple(analysis.reduction.curve("horizontal").region["rows"]) == (202, 207)
        page.file_list.setCurrentRow(3)  # … a frame chosen in the list is cut where it is found, as ever
        _done(page)
        assert tuple(page.view_model.state.analysis.reduction.curve("horizontal").region["rows"]) == (198, 203)
    finally:
        _close(page)


def test_the_check_box_finds_the_cut_again_in_every_frame(tmp_path: Path) -> None:
    page = _gisaxs_page(tmp_path)
    try:
        box = page.series_each_cut_check
        assert box.text() == "Find the cut again in every frame" and not box.isChecked()
        assert page.series_options.is_expanded() is False  # folded: one cut is the usual way
        box.setChecked(True)
        assert page.view_model.pin_series_cuts is False
        assert "horizontal cut: found again in each frame" in page.series_info_label.text()
        _build(page, 5)
        assert "found again in each frame" in page.series_title_label.text()
        assert page._series_map.built_with["cut"] is None
        box.setChecked(False)
        assert page.view_model.pin_series_cuts is True
    finally:
        page.view_model.set_pin_series_cuts(True)
        _close(page)


def test_the_check_box_marks_the_map_as_built_with_other_settings_at_once(tmp_path: Path) -> None:
    page = _gisaxs_page(tmp_path)
    try:
        _build(page, 5)
        stale = page.series_stale_label
        assert stale.isHidden()
        page.series_each_cut_check.setChecked(True)
        assert not stale.isHidden() and stale.text() == "Built with other settings — Build Map again."
        page.series_each_cut_check.setChecked(False)
        assert stale.isHidden()
    finally:
        page.view_model.set_pin_series_cuts(True)
        _close(page)


def test_open_shows_a_row_cut_as_the_map_it_belongs_to(tmp_path: Path) -> None:
    """A row of a map built with one cut opens at that cut; a row of a map built afterwards with the cut found
    in each frame opens at its own cut — also when it is the frame on screen already (review: it kept the cut
    of the map opened before)."""
    page = _gisaxs_page(tmp_path)
    try:
        _build(page, 5)
        page._series_pick_row(2.5)
        assert page.open_series_row()
        _done(page)
        horizontal = page.view_model.state.analysis.reduction.curve("horizontal")
        assert page.view_model.state.analysis.path.name == "film_02.tif" and tuple(horizontal.region["rows"]) == (202, 207)
        page.series_each_cut_check.setChecked(True)
        _build(page, 5)
        page._series_pick_row(2.5)
        assert page.open_series_row()
        _done(page)
        analysis = page.view_model.state.analysis
        assert tuple(analysis.reduction.curve("horizontal").region["rows"]) == (206, 211)  # its own Yoneda rows
        assert analysis.reduction.markers["horizontal_source"] == "yoneda"
        shown = []
        page.show_frame = lambda *args: shown.append(args)
        assert page.open_series_row() and shown == []  # the same row again, as it is shown: nothing to read again
    finally:
        page.view_model.set_pin_series_cuts(True)
        _close(page)


def test_the_sum_of_the_automatic_analysis_is_switched_off_for_a_map_and_said(tmp_path: Path) -> None:
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
        _done(page)
        from src.gimap.features.analyze.presentation.automation import AnalyzeAutomation

        done = []
        AnalyzeAutomation(page).set_frame(3, 10, lambda ok, text: done.append(ok))
        _wait(page, lambda: bool(done))
        _done(page)
        assert done == [True] and (page.view_model.sum_count, page.view_model.sum_set_by) == (10, "automatic")
        page.build_series_map()
        assert page.sum_spin.value() == 1 and page.view_model.sum_count == 1
        assert "Sum 10 of the automatic analysis was switched off for the series" in page.status_text()
        _wait(page, lambda: page._series_map is not None and not page.batch_running())
        _done(page)
        assert page._series_map.rows == 12
    finally:
        _close(page)


def test_a_sum_the_person_set_is_said_in_groups_and_its_short_last_group_is_marked(tmp_path: Path) -> None:
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
        _done(page)
        page.sum_spin.setValue(5)
        _done(page)
        _app().processEvents()
        page._refresh_series_curves(page.view_model.state.analysis)
        info = page.series_info_label.text()
        assert "12 frames in 3 sums of 5" in info and "the last sums only 2" in info
        _build(page, 3)
        series = page._series_map
        # Rows that are sums are called rows, not frames (review: “3 frames × …”, “Series map: 3 frames of …”).
        assert page.series_info_label.text().startswith("3 rows (sums of 5 frames) × ")
        assert page.status_text() == "Series map: 3 rows (sums of 5 frames) of I(q)"
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text() == "序列热图：I(q)，3 行（每行 5 帧之和）"
        apply_language("en", [page])
        page.refresh_language()
        assert [fact.incomplete for fact in series.facts] == [False, False, True] and series.numbers == (1, 6, 11)
        assert series.labels[2] == "stack.h5 frame 11 of 12 (+1 summed)"
        assert "the last row sums only 2 of 5 frames" in page.series_title_label.text()
        marks = page._series_sum_marks  # … and on the map itself, on that row
        assert len(marks) == 1 and marks[0].toPlainText() == "sum of 2 of 5" and marks[0].pos().y() == 2.5
        assert "sums" in page.series_map_view.marks.present()
        page._series_pick_row(2.5)
        assert "sum of 2 of 5" in page.series_profile_plot.title_label.text()
        names = [curve[0] for curve in page.series_trace_plot.figure_state()["curves"]]
        assert names[-1] == "sum of 2 of 5" and len(page.series_trace_plot.figure_state()["curves"][0][1]) == 2
        # Files of the row are named as Batch Export names the frame: 0-based, with the frames summed.
        assert page._series_frame_name() == "stack_frame0010_sum2_radial"
        page._series_pick_row(0.5)
        assert page._series_frame_name() == "stack_frame0000_sum5_radial"
        page.sum_spin.setValue(7)  # one complete row and a short one: the complete one alone is a dot, not no line
        _done(page)
        _build(page, 2)
        assert page.series_trace_plot._markers[:2] == [True, "x"]
    finally:
        page.sum_spin.setValue(1)
        _close(page)
