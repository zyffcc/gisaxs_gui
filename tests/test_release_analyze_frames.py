"""Release fixes of Analyze's frames and masks (R9, R26, R33).

A mask image of another detector size is refused when it is loaded (it used to be kept, and every later
analysis failed while the curves of the analysis before stayed on screen); a masks file that is no masks file
says so. A frame that cannot be read takes the frame shown before off the screen: its image, curves and green
steps must not stay under the new file's name, and nothing of it can be exported or sent to Fitting. With no
file, and for such a frame, both curve plots show a sentence instead of an empty grid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.presentation.bindings.display import NO_GEOMETRY_TEXT, NOT_ANALYSED_TEXT
from src.gimap.features.analyze.presentation.views.analyze_page_view import NO_FRAME_TEXT
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_assistant_calibration import SHAPE, save_tiff
from tests.test_review_analyze_run_and_clear import _frames, _profiles


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


def _open(tmp_path: Path, count: int = 1, profiles=True):
    paths = _frames(tmp_path, count)
    page = _page(_context(_profiles() if profiles else None))
    page.add_paths([str(path) for path in paths])
    _done(page)
    return page, paths


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _empty_texts(page) -> list[str]:
    return [plot.empty_overlay.text for plot in (page.top_plot, page.bottom_plot)]


# -- masks (R9) ---------------------------------------------------------------------------------------------


def test_a_mask_image_of_another_size_is_refused_when_it_is_loaded(tmp_path: Path) -> None:
    page, _paths = _open(tmp_path)
    try:
        before = page.view_model.state.analysis
        assert before.reduction is not None
        other = save_tiff(tmp_path / "masks" / "mask_other_detector.tif", np.ones((120, 100), dtype=np.float32))
        assert page.load_masks(other) is False
        assert page.view_model.state.corrections.mask_path is None and page.mask_list.count() == 0
        assert page.status_level() == "error"
        rows, columns = SHAPE
        assert page.status_text() == (
            f"Mask not loaded: The mask mask_other_detector.tif is 120×100; the frame is {rows}×{columns}.")
        _done(page)
        assert page.view_model.state.analysis is before  # nothing was analysed again, the curves are this frame's
        assert page.step_rail.state("data") == "ok"

        mask = np.zeros(SHAPE, dtype=np.float32)
        mask[:40, :] = 1.0
        fitting = save_tiff(tmp_path / "masks" / "mask_this_detector.tif", mask)
        assert page.load_masks(fitting) is True
        assert page.mask_list.count() == 1 and page.mask_list.item(0).text() == "Mask file: mask_this_detector.tif"
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis is not before and not analysis.valid[:40, :].any() and analysis.valid[60:, :].any()

        missing = tmp_path / "masks" / "not_there.tif"
        assert page.load_masks(missing) is False and page.status_level() == "error"
        assert page.status_text().startswith("Could not read the masks: ")
        assert Path(page.view_model.state.corrections.mask_path) == fitting  # the mask in use stays
    finally:
        _close(page)


def test_a_mask_image_is_kept_unchecked_only_while_no_frame_is_shown(tmp_path: Path) -> None:
    """Without a frame there is no size to compare with: the first analysis says it (as before)."""
    page = _page(_context(_profiles()))
    try:
        other = save_tiff(tmp_path / "mask_other_detector.tif", np.ones((120, 100), dtype=np.float32))
        assert page.load_masks(other) is True and page.view_model.state.corrections.mask_path == str(other)
    finally:
        _close(page)


def test_a_file_that_is_no_masks_file_says_what_is_wrong(tmp_path: Path) -> None:
    page, _paths = _open(tmp_path)
    try:
        settings = tmp_path / "analyze_settings.json"  # another JSON of GIMaP: it has no "masks"
        settings.write_text(json.dumps({"software": "GIMaP Analyze", "mode": "auto"}), encoding="utf-8")
        assert page.load_masks(settings) is False and page.status_level() == "error"
        assert page.status_text() == "analyze_settings.json is not a GIMaP masks file (it has no masks list)."
        for name, text in (("list.json", "[1, 2, 3]"), ("items.json", '{"masks": [{"kind": "rectangle"}]}'),
                           ("points.json", '{"masks": [{"kind": "rectangle", "points": 5}]}')):
            (tmp_path / name).write_text(text, encoding="utf-8")
            assert page.load_masks(tmp_path / name) is False
            assert page.status_text() == f"{name} is not a GIMaP masks file (it has no masks list)."
        broken = tmp_path / "half_written.json"
        broken.write_text('{"masks": [', encoding="utf-8")
        assert page.load_masks(broken) is False
        assert page.status_text() == "half_written.json is not a GIMaP masks file (it is not valid JSON)."
        triangle = tmp_path / "two_points.json"  # a masks file whose mask cannot be used: the domain says why
        triangle.write_text(json.dumps({"masks": [{"kind": "polygon", "points": [[0, 0], [5, 5]]}]}), encoding="utf-8")
        assert page.load_masks(triangle) is False
        assert page.status_text() == "Mask not loaded: A polygon mask needs at least three vertices."
        assert page.view_model.state.corrections.mask_shapes == () and page.mask_list.count() == 0

        good = tmp_path / "masks.json"  # … and a real one still loads
        good.write_text(json.dumps({"masks": [{"kind": "rectangle", "points": [[10, 20], [30, 40]]}]}), encoding="utf-8")
        assert page.load_masks(good) is True and len(page.view_model.state.corrections.mask_shapes) == 1
        _done(page)
    finally:
        _close(page)


# -- a frame that cannot be read (R33) --------------------------------------------------------------------------


def _broken(folder: Path, name: str = "broken_frame.cbf") -> Path:
    path = folder / name
    path.write_bytes(b"not a detector")  # 14 bytes: no CBF
    return path


def test_a_frame_that_cannot_be_read_takes_the_frame_before_off_the_screen(tmp_path: Path) -> None:
    sent = []
    page, paths = _open(tmp_path)
    page._send_to_fitting = lambda path, side="both_abs": sent.append(path)
    failed = []
    page.analysisFailed.connect(failed.append)
    try:
        good = page.view_model.state.analysis
        assert good.reduction is not None and page.detector_view.has_image() and page.top_plot.has_curves()
        assert [page.step_rail.state(key) for key in ("data", "geometry", "mask", "cuts")] == ["ok"] * 4
        page.fit_button.setEnabled(True)

        broken = _broken(tmp_path)
        page.add_paths([str(broken)])
        _done(page)
        assert page.view_model.current_path == broken.resolve()
        assert len(failed) == 1 and failed[0].startswith("broken_frame.cbf ")  # English, for the assistant
        # Nothing of the frame before under the new file's name.
        assert page.view_model.state.analysis is None
        assert not page.detector_view.has_image()
        assert not page.top_plot.has_curves() and not page.bottom_plot.has_curves()
        assert _empty_texts(page) == [NOT_ANALYSED_TEXT, NOT_ANALYSED_TEXT]
        assert page.top_plot.title_label.text() == "" and page._plot_keys == {"top": [], "bottom": []}
        assert page.step_rail.state("data") == "error"
        assert [page.step_rail.state(key) for key in ("geometry", "mask", "cuts")] == ["pending"] * 3
        assert page.file_chip.full_text() == "broken_frame.cbf" and page.file_meta.text() == "could not be read"
        assert page.status_level() == "error" and page.status_text().startswith(
            "Could not analyse the frame: broken_frame.cbf ")
        assert page.data_info_label.text().startswith("broken_frame.cbf ")  # not the frame before's card
        assert "film_0.tif" not in page.step_intro["data"].text()
        assert not page.fit_button.isEnabled() and not page.center_button.isEnabled()
        assert page.banner.isHidden()
        # … and nothing of it is written: Export has no frame.
        page.export_current()
        assert page.status_text() == "Export failed: Open a frame first." and not (tmp_path / "gimap_analysis").exists()
        page.fit_current()
        assert sent == []

        # A language switch does not bring the frame before back (it used to rewrite the name and the steps).
        page.refresh_language()
        assert page.file_meta.text() == "could not be read" and page.step_rail.state("data") == "error"
        assert page.view_model.state.analysis is None

        page.file_list.setCurrentRow(0)  # the good frame again: shown as before
        _done(page)
        again = page.view_model.state.analysis
        assert again is not None and again.path == paths[0].resolve() and again.reduction is not None
        assert page.detector_view.has_image() and page.top_plot.has_curves()
        assert _empty_texts(page) == ["", ""] and page.step_rail.state("data") == "ok"
        assert page.fit_button.isEnabled()
    finally:
        _close(page)


def _axis_names(page) -> list[str]:
    return [plot.plot.getAxis(side).labelText for plot in (page.top_plot, page.bottom_plot) for side in ("bottom", "left")]


def test_no_axis_name_or_incidence_of_the_frame_before_is_left(tmp_path: Path) -> None:
    """With the curves go their axis names (“qy (Å⁻¹)” under a file that is no frame) and the αi the field
    showed for the frame before; Clear leaves neither, and nothing to send to Fitting."""
    page, _paths = _open(tmp_path)
    page._send_to_fitting = lambda path, side="both_abs": None
    try:
        assert all(_axis_names(page)) and page.incidence_spin.text() != "αi from profile"
        shown = page.incidence_spin.text()
        page.add_paths([str(_broken(tmp_path))])
        _done(page)
        assert _axis_names(page) == [""] * 4
        assert page.incidence_spin.text() == "αi from profile" and page.view_model.state.incidence_deg is None
        page.file_list.setCurrentRow(0)  # the good frame again: its axes and its αi
        _done(page)
        assert all(_axis_names(page)) and page.incidence_spin.text() == shown
        page.fit_button.setEnabled(True)
        page.clear_files()
        assert _axis_names(page) == [""] * 4 and page.incidence_spin.text() == "αi from profile"
        assert not page.fit_button.isEnabled()
    finally:
        _close(page)


def test_an_incidence_set_by_hand_stays_when_a_frame_cannot_be_read(tmp_path: Path) -> None:
    page, _paths = _open(tmp_path)
    try:
        page.incidence_spin.setValue(0.31)  # the person's value, for every frame
        _done(page)
        page.add_paths([str(_broken(tmp_path))])
        _done(page)
        assert page.incidence_spin.value() == pytest.approx(0.31) and page.incidence_spin.property("override") is True
        assert page.view_model.state.incidence_deg == pytest.approx(0.31)
        page.clear_files()
        assert page.incidence_spin.value() == pytest.approx(0.31) and page.view_model.state.incidence_deg == pytest.approx(0.31)
    finally:
        page.view_model.set_incidence(None)
        page._remember()
        _close(page)


def test_no_automatic_analysis_starts_for_a_frame_that_could_not_be_read(tmp_path: Path) -> None:
    """Its outcome would be kept as another file's (“Results are for broken_frame.cbf; run again for this
    frame”) on the very frame it was run for: the status line says again why the frame is not shown."""
    page, _paths = _open(tmp_path)
    called = []
    page.set_automatic_analysis(run=lambda: called.append("run"), find_geometry=lambda: called.append("find"))
    try:
        page.add_paths([str(_broken(tmp_path))])
        _done(page)
        for button in (page.run_pipeline_button, page.find_geometry_button):
            page._status("Ready")
            button.click()
            assert called == []
            assert page.status_level() == "error"
            assert page.status_text().startswith("Could not analyse the frame: broken_frame.cbf ")
        assert page.step_rail.state("results") == "pending" and page.step_rail.state("data") == "error"

        page.file_list.setCurrentRow(0)  # a frame that can be read: while it is read, and once it is shown
        page.run_pipeline_button.click()
        assert called == ["run"]
        _done(page)
        page.find_geometry_button.click()
        assert called == ["run", "find"]
    finally:
        _close(page)


def test_a_setting_that_cannot_be_applied_keeps_the_frame_shown(tmp_path: Path) -> None:
    """The same frame fails with a new setting (a background of another size): its last good picture stays."""
    page, _paths = _open(tmp_path)
    try:
        good = page.view_model.state.analysis
        other = save_tiff(tmp_path / "background_other.tif", np.ones((120, 100), dtype=np.float32))
        page.view_model.set_background(other)
        page.run_analysis()
        _done(page)
        assert page.status_level() == "error" and page.step_rail.state("data") == "error"
        assert page.view_model.state.analysis is good  # kept: it is this frame
        assert page.detector_view.has_image() and page.top_plot.has_curves()
        assert [page.step_rail.state(key) for key in ("geometry", "mask", "cuts")] == ["ok"] * 3
        page.view_model.set_background(None)
        page.run_analysis()
        _done(page)
        assert page.step_rail.state("data") == "ok" and page.status_level() == "ok"
    finally:
        _close(page)


def test_results_kept_for_the_frame_before_are_named_as_its_own(tmp_path: Path) -> None:
    page, paths = _open(tmp_path)
    try:
        page.automatic_finished("ok", "GIWAXS — 3 peaks", path=paths[0])
        assert page.step_rail.state("results") == "ok"
        page.add_paths([str(_broken(tmp_path))])
        _done(page)
        assert page.step_rail.state("results") == "pending"
        assert page.step_intro["results"].text() == "Results are for film_0.tif; run again for this frame"
        assert page.current_right() == "curves"
        page.file_list.setCurrentRow(0)
        _done(page)
        assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "GIWAXS — 3 peaks"
    finally:
        _close(page)


def test_a_broken_first_file_and_a_removed_file_leave_no_picture_behind(tmp_path: Path) -> None:
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(_broken(tmp_path))])  # the first file opened cannot be read
        _done(page)
        assert page.view_model.state.analysis is None and page.step_rail.state("data") == "error"
        assert _empty_texts(page) == [NOT_ANALYSED_TEXT, NOT_ANALYSED_TEXT]
        good = _frames(tmp_path, 1)
        page.add_paths([str(good[0])])
        _done(page)
        assert page.detector_view.has_image() and page.file_list.currentRow() == 1
        # The file shown is removed and the one that takes its place cannot be read: no picture of the removed one.
        assert page.remove_file(1) is True
        _done(page)
        assert page.view_model.current_path.name == "broken_frame.cbf"
        assert not page.detector_view.has_image() and not page.top_plot.has_curves()
        assert page.step_rail.state("data") == "error"
    finally:
        _close(page)


# -- what the curve plots say with nothing to draw (R26) ---------------------------------------------------------------


def test_the_curve_plots_say_a_sentence_instead_of_showing_an_empty_grid(tmp_path: Path) -> None:
    page = _page(_context())  # no instrument profile: a frame has no geometry
    try:
        assert _empty_texts(page) == [NO_FRAME_TEXT, NO_FRAME_TEXT] == ["Open a detector frame"] * 2
        for plot in (page.top_plot, page.bottom_plot):  # before the first file: no ticks that mean nothing
            assert plot.empty_overlay.bare and not plot.empty_overlay.label.isHidden()
            assert plot.empty_overlay.label.text() == "Open a detector frame"
        paths = _frames(tmp_path, 1)
        page.add_paths([str(paths[0])])
        _done(page)
        assert page.view_model.state.analysis.reduction is None
        assert _empty_texts(page) == [NO_GEOMETRY_TEXT, NO_GEOMETRY_TEXT]
        page.clear_files()
        assert _empty_texts(page) == [NO_FRAME_TEXT, NO_FRAME_TEXT]
        assert not page.top_plot.has_curves() and page.top_plot.empty_overlay.bare
    finally:
        _close(page)

    page, _paths = _open(tmp_path)  # with a geometry: the curves, and no sentence left over them
    try:
        assert _empty_texts(page) == ["", ""] and page.top_plot.has_curves() and page.bottom_plot.has_curves()
        assert not page.top_plot.empty_overlay.bare and page.top_plot.empty_overlay.label.isHidden()
        page.clear_files()
        assert _empty_texts(page) == [NO_FRAME_TEXT, NO_FRAME_TEXT]
    finally:
        _close(page)


def test_the_views_are_made_with_the_sentence() -> None:
    _app()
    page = _page(_context())
    try:
        assert page.detector_view.readout_label.text() == "Open a detector frame"  # the image says the same
    finally:
        _close(page)
