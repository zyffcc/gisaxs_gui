"""Cross-area review items owned by Analyze.

The set-up offer names αi; the region list is as tall as its rows; the Series stages speak of the map's
own axis; the upper plot's |qy| tooltip names the second shade; the Results step follows the frames of a
series the run analysed; ``batch_kind``, ``last_folder``, ``set_mode(remember=False)`` and the cut's pixel
ranges in ``status()`` are the public hooks other areas use; a language switch composes the run-time texts
again (``refresh_language``); the image and the curves share the room.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import ZH, apply_language, tr, trf
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from tests.test_analyze_workspace import (
    GALAXI,
    P08,
    _context,
    _done,
    _galaxi_profile,
    _p08_profile,
    _page,
    _public_page,
    requires_public,
)


@pytest.fixture
def english():
    """Every test leaves the interface in English, whatever it switched to."""
    yield
    apply_language("en")


def _series(path: Path, frames: int = 12) -> Path:
    import h5py

    data = np.random.default_rng(0).poisson(5, (frames, 64, 64)).astype(np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data", data=data)
    return path


def _show(page, frame_index: int, summed: int) -> None:
    page.view_model.set_sum_count(summed)
    page.view_model.set_frame(frame_index)
    page.run_analysis()
    _done(page, 60)


# -- the set-up of the last session names αi (analyze[0]) -------------------------------------------------


def test_the_last_setup_offer_names_the_incidence_angle() -> None:
    from src.gimap.features.analyze.application import AnalyzeSettings
    from src.gimap.features.analyze.presentation.bindings.session_memory import _setup_summary

    assert "αi 0.4°" in _setup_summary(AnalyzeSettings(mode="giwaxs", incidence_deg=0.4)).split(" · ")
    assert "αi" not in _setup_summary(AnalyzeSettings(mode="giwaxs"))  # the profile's αi: nothing to say


# -- the region list is as tall as its rows (analyze[1]) --------------------------------------------------


@requires_public
def test_every_region_row_shows_without_scrolling() -> None:
    page = _public_page()
    page.add_paths([str(P08)])
    _done(page)
    listing = page.region_list
    assert listing.count() >= 4  # full ring, in-plane, out-of-plane, the ring of I(χ)
    rows = sum(listing.sizeHintForRow(row) for row in range(listing.count()))
    assert rows > 40 * listing.count()  # two lines each: the former 40 px a row cut the last one
    assert listing.height() >= rows + 2 * listing.frameWidth()
    page.dispose()
    page.close()


# -- the stages speak of the map's own axis (compare[2]) -------------------------------------------------


def test_series_stages_name_the_axis_of_the_map() -> None:
    from src.gimap.features.analyze.presentation.bindings.series_stages import SeriesStagesMixin

    change = SimpleNamespace(stage=2, rises=[(42.0, 30.0)], falls=[(-12.0, -20.0)])
    odd = SimpleNamespace(row=3, narrow=True, q=12.5, z=9.0)
    stages = SimpleNamespace(
        stage_representatives=lambda: [0, 6], ranges=lambda: [(0, 4), (5, 9)], stage_changes=lambda: [change],
        half_row=None, count=2, ninety_row=None, q=np.linspace(-90.0, 90.0, 50), odd=[odd],
    )
    series = SimpleNamespace(x_label="χ (°)", labels=[f"frame {n}" for n in range(1, 11)])
    text = SeriesStagesMixin._stages_text(None, series, stages)
    assert "grows most at χ 42" in text and "falls most at χ -12" in text
    assert "differs only near χ 12.5" in text
    assert "at q " not in text and "near q " not in text  # the changes and odd frames are not put on q


# -- the upper plot's |qy| tooltip (components[2]) --------------------------------------------------------


def test_the_folded_halves_tooltip_names_the_second_shade() -> None:
    from src.gimap.features.analyze.presentation.views.analyze_page_view import TOP_SIDE_TIPS

    assert TOP_SIDE_TIPS[3] == "Both halves on |qy|, the negative one dashed in a second shade (display only)"
    assert TOP_SIDE_TIPS[3] in ZH


# -- the Results step follows the frames of a series (assistant[1]) ---------------------------------------


def test_the_results_step_follows_the_frames_the_run_analysed(tmp_path: Path) -> None:
    from src.gimap.features.analyze.presentation.bindings.results_state import other_frames

    page = _page(_context())
    series = _series(tmp_path / "insitu_series.nxs")
    page.add_paths([str(series)])
    _done(page)
    _show(page, 2, 10)  # frames 3–12 summed: what the standard procedure analyses of a series of 12
    page.automatic_started("Automatic analysis …")
    page.automatic_finished("ok", "12 peaks", frames={"first": 3, "summed": 10, "total": 12})
    assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "12 peaks"

    _show(page, 0, 1)  # frame 1 of the same file: the results are not this frame's
    assert page.step_rail.state("results") == "pending"
    assert page.step_rail.detail("results") == "Results are for frames 3–12; run again for this frame"
    assert page.step_intro["results"].text() == page.step_rail.detail("results")
    assert page.current_right() == "results"  # the report stays in the tab, saying the same
    page.run_analysis()  # a re-reduction of frame 1 says the same
    _done(page)
    assert page.step_rail.state("results") == "pending"

    _show(page, 2, 10)  # the run's frames again: the step comes back
    assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "12 peaks"
    assert page.step_intro["results"].text() == "12 peaks"

    # A run that ends while other frames are shown keeps its results for its frames.
    _show(page, 5, 1)
    page.automatic_started("Automatic analysis …")
    page.automatic_finished("warn", "1 question", frames={"first": 3, "summed": 10, "total": 12})
    assert page.step_rail.state("results") == "pending" and "3–12" in page.step_rail.detail("results")
    _show(page, 2, 10)
    assert page.step_rail.state("results") == "warn" and page.step_rail.detail("results") == "1 question"

    # Without frames (a single frame, or a caller that does not say): the file counts, as before.
    page.automatic_started("Automatic analysis …")
    page.automatic_finished("ok", "whole file")
    _show(page, 0, 1)
    assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "whole file"
    analysis = SimpleNamespace(frame_index=0, frame_total=1)
    assert other_frames({"first": 1, "summed": 1, "total": 1}, analysis) is None  # no series
    assert other_frames({"first": 1, "summed": 1, "total": 5}, analysis) is None  # the run's frame
    assert other_frames({"first": 2, "summed": 3, "total": 5}, analysis) == (2, 4)
    assert other_frames({"first": "x"}, analysis) is None and other_frames(None, analysis) is None
    page.dispose()
    page.close()


# -- public hooks for the shell (shell[3], analyze[2]) and the assistant (assistant[0], [2]) ---------------


def test_batch_kind_and_last_folder(tmp_path: Path) -> None:
    page = _page(_context())
    page._batch_map_only = True
    assert page.batch_kind() == "series_map"
    page._batch_map_only = False
    assert page.batch_kind() == "export"

    assert page.last_folder == ""  # nothing opened yet
    frame = tmp_path / "data" / "film.tif"
    frame.parent.mkdir()
    frame.write_bytes(b"")
    page.remember_recent([str(frame)])
    assert page.last_folder == str(frame.parent)  # a drop or Recent: the folder of the newest item
    page._remember(last_folder=str(tmp_path))
    assert page.last_folder == str(tmp_path)  # Open… / Open Folder… win
    with pytest.raises(AttributeError):
        page.last_folder = "elsewhere"  # read-only
    page.dispose()
    page.close()


def test_a_mode_switch_of_the_run_is_not_saved_as_the_persons_mode() -> None:
    page = _page(_context())
    assert page.view_model.settings is not None

    def saved():
        return page.view_model.settings.get("analyze", "mode")

    page.choose_mode("auto")
    assert saved() == "auto"
    outcomes = []
    page.automation().set_mode("gisaxs", lambda ok, message: outcomes.append(ok), remember=False)
    assert page.view_model.state.mode == "gisaxs" and page.mode_combo.currentData() == "gisaxs"
    assert saved() == "auto"
    page._remember(last_folder="C:/data")  # anything else remembered later keeps the person's mode
    assert saved() == "auto"
    page.automation().set_mode("giwaxs", lambda ok, message: outcomes.append(ok), remember=False)
    assert saved() == "auto"  # still the person's, not the first switch
    page.choose_mode("gisaxs")  # the person's own choice
    assert saved() == "gisaxs"
    page.automation().set_mode("giwaxs", lambda ok, message: outcomes.append(ok))  # remembered by default
    assert saved() == "giwaxs"
    assert outcomes == [False, False, False]  # no frame open: each says so
    page.dispose()
    page.close()


@requires_public
def test_the_status_names_the_pixels_of_each_cut_both_ends_included() -> None:
    from src.gimap.features.analyze.presentation.bindings.display import cut_lines

    page = _public_page()
    page.add_paths([str(GALAXI)])
    _done(page)
    cuts = page.automation().status()["gisaxs"]
    reduction = page.view_model.state.analysis.reduction
    start, stop = reduction.curve("horizontal").region["rows"]
    assert cuts["horizontal_pixel_rows"] == [int(start), int(stop) - 1]
    left, right = reduction.curve("vertical").region["columns"]
    assert cuts["vertical_pixel_columns"] == [int(left), int(right) - 1]
    first, last = cuts["horizontal_pixel_rows"]
    assert f"rows {first}–{last}" in cut_lines(page.view_model.state.analysis)[0]  # as the Cuts step says
    page.dispose()
    page.close()


# -- a language switch composes the run-time texts again (components[1]) ---------------------------------


@requires_public
def test_refresh_language_composes_the_texts_of_the_frame_shown(monkeypatch, english) -> None:
    from src.gimap.features.analyze.presentation.bindings.display import _horizontal_where, pixel_range

    profiles = InMemoryInstrumentProfileRepository([_galaxi_profile(), _p08_profile()])
    page = _page(_context(profiles))
    page.add_paths([str(GALAXI)])
    _done(page)
    english_title = page.top_plot.title_label.text()
    english_status = page.status_text()
    assert page.step_rail.detail("cuts") == "Yoneda cut" and english_status.startswith("galaxi_data.tif: ")
    # Texts whose Chinese the integrator merges from this area's entries.
    for key, value in (("{name}: {curves} curves in {seconds} s", "{name}：{curves} 条曲线，用时 {seconds} s"),
                       ("Beam {x}, {y} px · profile", "光束 {x}, {y} px · 配置")):
        monkeypatch.setitem(i18n.ZH, key, value)
    box = page.top_plot.plot.getViewBox()
    box.setRange(xRange=(0.01, 0.05), yRange=(0.5, 1.5), padding=0)  # the person zoomed in
    zoomed = box.viewRange()

    apply_language("zh", [page])
    page.refresh_language()
    assert page.step_rail.detail("cuts") == ZH["Yoneda cut"]
    reduction = page.view_model.state.analysis.reduction
    title = trf("I(qy) · rows {rows} · {where}", rows=pixel_range(*reduction.curve("horizontal").region["rows"]),
                where=_horizontal_where(reduction, short=True))
    assert title != english_title and page.top_plot.title_label.text() == title
    assert "条曲线" in page.status_text()
    assert page.center_button.text().startswith("光束 ") and page.center_button.text().endswith("配置")
    assert np.allclose(box.viewRange(), zoomed)  # the zoom stays

    apply_language("en", [page])
    page.refresh_language()
    assert page.step_rail.detail("cuts") == "Yoneda cut"
    assert page.top_plot.title_label.text() == english_title
    assert page.status_text() == english_status
    assert page.center_button.text().endswith("· profile")
    page.dispose()
    page.close()


def test_refresh_language_says_which_frames_the_results_are_for(tmp_path: Path, english) -> None:
    from src.gimap.features.analyze.presentation.bindings.results_state import RESULTS_OTHER_FRAMES

    page = _page(_context())
    page.add_paths([str(_series(tmp_path / "insitu.nxs"))])
    _done(page)
    _show(page, 2, 10)
    page.automatic_started("Automatic analysis …")
    page.automatic_finished("ok", "12 peaks", frames={"first": 3, "summed": 10, "total": 12})
    _show(page, 0, 1)
    apply_language("zh", [page])
    page.refresh_language()
    assert RESULTS_OTHER_FRAMES in ZH
    assert page.step_rail.detail("results") == ZH[RESULTS_OTHER_FRAMES].format(a=3, b=12)
    apply_language("en", [page])
    page.refresh_language()
    assert page.step_rail.detail("results") == "Results are for frames 3–12; run again for this frame"
    assert tr("Ready") == "Ready"
    page.dispose()
    page.close()


# -- the image and the curves share the room (assistant[3]) ------------------------------------------------


def _laid_out(page, width: int) -> list[int]:
    page.resize(width, 700)
    for _ in range(3):  # the resize, then the balance posted after it
        QApplication.processEvents()
    return page.splitter.sizes()


def test_the_image_and_the_curves_share_the_room_the_steps_leave() -> None:
    from src.gimap.features.analyze.presentation.bindings.workspace import IMAGE_KEEPS

    page = _page(_context())
    page.show()
    steps, image, curves = _laid_out(page, 1100)  # the page of a 1280 px window
    assert steps >= page.process_panel.minimumWidth()
    assert image == IMAGE_KEEPS and curves >= 350  # the image keeps its two toolbar lines, the rest is the curves'
    _steps, image, curves = _laid_out(page, 1500)  # a wide window: half each
    assert abs(image - curves) <= 1 and curves > 500

    # Once the person moves a handle, their split stays (not balanced again); the splitter shares more room.
    page.splitter.setSizes([steps, 700, image + curves - 700])
    page.splitter.splitterMoved.emit(steps + 700, 2)
    before = page.splitter.sizes()
    after = _laid_out(page, 1700)
    assert page.even_split.by_hand and after[1] - after[2] >= before[1] - before[2] - 2
    page.dispose()
    page.close()
