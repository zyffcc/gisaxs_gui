"""What the series work left between its parts: each part said it, none of them could do it in its own files.

* A frame cut at the cut of the series (one frame's automatic cut, held for every frame of a Series map, a
  Batch Export, a watched folder, a map row opened) is recorded and shown as the cut of the series, not as
  “set by hand”.
* A saved Series map of a few rows has whole frame numbers on its axis.
* A last session written before the pin of the series cut existed is not offered again for that alone.
* A Batch Export dialog that waits for the frame on screen to be read again stops waiting when that fails.
* Rows of a series are named “frame 4 of 12”.
* Export (the frame on screen, and while watching) goes into the session's folder when the folder of the data
  takes no file; pictures and tables are offered under the name of the frame.
* An In-situ series run that ends with poor fits says so.
* On a GIWAXS frame the Fitting button and Send Series say what Fitting fits and where peaks are fitted; the
  Send Series dialog opens with its options in view.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import analysis_metadata, frame_cuts
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.shared.detector_io import forget_layout
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_assistant_calibration import save_tiff
from tests.test_review_analyze_run_and_clear import _profiles
from tests.test_series_cut import (
    _build,
    _gisaxs_files,
    _gisaxs_frame,
    _gisaxs_page,
    _gisaxs_profiles,
    _shown,
    _two_ring_frame,
)

ZH = i18n.ZH
SERIES_SOURCE = "Cut of the series (the cut of one frame, held for every frame)"


@pytest.fixture(autouse=True)
def _english():
    forget_layout()
    apply_language("en")
    yield
    apply_language("en")
    forget_layout()


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _settle(seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


# -- 1. the cut of the series in the record and on screen ---------------------------------------------------------


def test_a_frame_cut_at_the_cut_of_the_series_says_so_in_its_record(tmp_path: Path) -> None:
    model = create_analyze_view_model(_context(_gisaxs_profiles()))
    shown = _shown(model, "gisaxs", _gisaxs_files(tmp_path))
    assert shown.reduction.markers["horizontal_source"] == "yoneda" and "series_cut" not in shown.reduction.markers
    held = model.series_cut()
    requests = model.batch_requests()
    assert [request.series_cut for request in requests] == [held] * 5
    other = model.analyze(requests[3])
    markers = other.reduction.markers
    assert markers["horizontal_source"] == "series"  # not “manual”: nobody set it by hand
    assert markers["series_cut"] == {"frame": (str(shown.path), 0), "horizontal_rows": held["horizontal_rows"],
                                     "vertical_columns": held["vertical_columns"]}
    record = json.loads(json.dumps(analysis_metadata(other)))  # (what the _analysis.json beside its curves says)
    cuts = frame_cuts(record)
    assert cuts["horizontal_source"] == "series"
    assert cuts["series_cut"] == {"frame": [str(shown.path), 0], "horizontal_rows": list(held["horizontal_rows"]),
                                  "vertical_columns": list(held["vertical_columns"])}
    model.set_pin_series_cuts(False)  # found again in every frame: each frame its own Yoneda cut, no cut of a series
    free = model.analyze(model.batch_requests()[3])
    assert free.reduction.markers["horizontal_source"] == "yoneda" and "series_cut" not in frame_cuts(analysis_metadata(free))
    model.set_pin_series_cuts(True)


def test_a_ring_held_for_a_giwaxs_series_is_recorded_with_the_frame_it_comes_from(tmp_path: Path) -> None:
    paths = [save_tiff(tmp_path / f"ring_{index}.tif", _two_ring_frame(height, seed=index))
             for index, height in enumerate((60.0, 150.0, 700.0, 900.0))]
    model = create_analyze_view_model(_context(_profiles()))
    shown = _shown(model, "giwaxs", paths)
    window = tuple(float(value) for value in shown.reduction.markers["chi_q_window"])
    last = model.analyze(model.batch_requests()[-1])
    assert last.reduction.markers["series_cut"] == {"frame": (str(shown.path), 0), "chi_q_window": window}
    cuts = frame_cuts(json.loads(json.dumps(analysis_metadata(last))))
    assert cuts["chi_q_window"] == list(window) and cuts["series_cut"] == {"frame": [str(shown.path), 0],
                                                                          "chi_q_window": list(window)}


def test_a_cut_the_person_set_stays_theirs_in_the_record(tmp_path: Path) -> None:
    model = create_analyze_view_model(_context(_gisaxs_profiles()))
    _shown(model, "gisaxs", _gisaxs_files(tmp_path))
    model.set_horizontal_band(190.0, 196.0)  # the person's rows; the vertical cut is still the automatic one
    shown = model.analyze(model.request())
    model.accept(shown)
    other = model.analyze(model.batch_requests()[2])
    markers = other.reduction.markers
    assert markers["horizontal_source"] == "manual"
    assert markers["series_cut"] == {"frame": (str(shown.path), 0), "vertical_columns": (115.0, 125.0)}
    assert frame_cuts(analysis_metadata(other))["horizontal_source"] == "manual"


def test_a_row_opened_from_a_map_shows_the_cut_of_the_series_in_the_cuts_step(tmp_path: Path) -> None:
    from src.gimap.features.analyze.presentation.bindings.display import cut_lines

    page = _gisaxs_page(tmp_path, bands=(204, 204, 208, 200, 206))
    try:
        assert page.step_rail.detail("cuts") == "Yoneda cut"
        _build(page, 5)
        page._series_pick_row(2.5)
        assert page.open_series_row()
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.path.name == "film_02.tif" and analysis.reduction.markers["horizontal_source"] == "series"
        assert page.step_rail.detail("cuts") == SERIES_SOURCE
        assert cut_lines(analysis)[0] == "Horizontal cut I(qy) over rows 202–206, the cut of the series."
        assert page.top_plot.title_label.text() == "I(qy) · rows 202–206 · cut of the series"
        apply_language("zh", [page])
        page.refresh_language()
        assert page.step_rail.detail("cuts") == ZH[SERIES_SOURCE]
        assert ZH["the cut of the series"] in page.cuts_info_label.text()
        apply_language("en", [page])
        page.refresh_language()
        assert page.step_rail.detail("cuts") == SERIES_SOURCE
    finally:
        apply_language("en")
        _close(page)


def test_a_vertical_cut_held_from_another_frame_is_not_called_set_by_hand() -> None:
    from types import SimpleNamespace

    from src.gimap.features.analyze.domain import Curve, Reduction
    from src.gimap.features.analyze.presentation.bindings.display import cut_lines
    from src.gimap.shared.geometry import DetectorGeometry

    def curve(key, region):
        return Curve(key, key, np.array([0.1, 0.2]), np.array([1.0, 2.0]), np.array([0.1, 0.1]), np.array([5, 5]),
                     "q (Å⁻¹)", "I", region)

    markers = {"horizontal_source": "series", "yoneda": None,
               "series_cut": {"frame": ("a.tif", 0), "horizontal_rows": (202.0, 207.0), "vertical_columns": (110.0, 120.0)}}
    reduction = Reduction("gisaxs", (curve("horizontal", {"rows": (202, 207)}),
                                     curve("vertical", {"columns": (110, 121), "center_column": 115.0})), markers)
    analysis = SimpleNamespace(reduction=reduction, geometry=DetectorGeometry(172e-6, 172e-6, 1.0, 120.0, 260.0, 1.0, 0.4))
    assert cut_lines(analysis) == ["Horizontal cut I(qy) over rows 202–206, the cut of the series.",
                                   "Vertical cut I(qz) over columns 110–120 (the cut of the series)."]
    for key in ("the cut of the series", "cut of the series", "Vertical cut I(qz) over columns {columns} (the cut of the series).",
                SERIES_SOURCE):
        assert ZH.get(key) and ZH[key] != key


# -- 2. the saved map: whole frame numbers -------------------------------------------------------------------------


def _drawn(monkeypatch):
    from src.gimap.features.analyze.infrastructure.adapters import figures

    kept = []
    monkeypatch.setattr(figures, "save_figure", lambda figure, path: (kept.append(figure), Path(path))[1])
    return figures.MatplotlibFigureWriter(), kept


def _y_labels(figure) -> list[str]:
    axes = figure.axes[0]
    figure.canvas.draw()
    low, high = sorted(axes.get_ylim())
    return [label.get_text() for tick, label in zip(axes.get_yticks(), axes.get_yticklabels()) if low <= tick <= high]


def test_a_saved_map_of_few_rows_has_whole_frame_numbers(tmp_path: Path, monkeypatch) -> None:
    writer, kept = _drawn(monkeypatch)
    image = np.arange(30.0).reshape(3, 10)
    writer.write_image(image, tmp_path / "map.png", extent=(0.0, 1.0, 3.5, 0.5), x_label="q (Å⁻¹)", y_label="frame")
    labels = _y_labels(kept[-1])
    assert labels and all(label.lstrip("−-").isdigit() for label in labels), labels  # 1, 2, 3 — not 1.0, 1.5, 2.0 …
    writer.write_image(image, tmp_path / "rows.png", extent=(0.0, 1.0, 3.5, 0.5), y_label="frame",
                       y_ticks=[(1.0, "4"), (2.0, "7"), (3.0, "10")])  # rows left out: the numbers of the rows
    assert _y_labels(kept[-1]) == ["4", "7", "10"]
    writer.write_image(image, tmp_path / "qmap.png", extent=(-0.1, 0.1, 0.0, 0.3), y_label="qz (Å⁻¹)")
    assert any("." in label for label in _y_labels(kept[-1]))  # a q axis keeps its decimals


# -- 3. the last set-up of a session written before the pin existed ------------------------------------------------


def test_a_last_setup_without_the_pin_is_not_offered_for_the_pin_alone(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.analyze.application import settings_record
    from src.gimap.features.analyze.presentation.bindings import session_memory

    offers = []
    monkeypatch.setattr(session_memory, "show_toast", lambda _parent, text, **options: offers.append(text))
    page = _page(_context(_gisaxs_profiles()))
    try:
        record = settings_record(page.view_model.current_settings())
        assert record.pop("pin_series_cuts") is True  # the record of now says the pin …
        setup = tmp_path / "last_analyze_setup.json"
        setup.write_text(json.dumps(record), encoding="utf-8")  # … one of before did not know it
        page.view_model.last_setup_path = setup
        page._offer_last_setup()
        assert offers == []  # nothing would change
        record["mode"] = "giwaxs"
        setup.write_text(json.dumps(record), encoding="utf-8")
        page._setup_offered = False
        page._offer_last_setup()
        assert len(offers) == 1 and "last session" in offers[0]
    finally:
        _close(page)


# -- 4. a Batch Export dialog waiting for the frame on screen ------------------------------------------------------


def test_a_failed_read_lets_the_batch_dialog_that_waits_for_it_go(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import BatchChoices
    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    page = _page(_context(_gisaxs_profiles()))
    dialog = BatchExportDialog(frames=3, files=3, stem="run", settings_text="GISAXS", choices=BatchChoices(),
                               destination=tmp_path, subfolder=True, series=SeriesCorrection(),
                               curve_options=[("horizontal", "Horizontal cut I(qy)")], gisaxs=True,
                               cut_text=lambda: "horizontal cut: rows 202–206 for every frame", reading=True)
    try:
        assert dialog.reading() and not dialog.export_button.isEnabled()
        page._batch_dialog = dialog
        page._analysis_failed("The file could not be read.", "")
        assert not dialog.reading() and dialog.export_button.isEnabled()
        assert dialog.cut_label.text() == "horizontal cut: rows 202–206 for every frame"
        assert page.status_level() == "error"  # what failed is said as before
    finally:
        page._batch_dialog = None
        dialog.done(0)
        dialog.deleteLater()
        _close(page)


# -- 5. the rows of a series ---------------------------------------------------------------------------------------


def test_rows_of_a_multi_frame_source_are_named_frame_n_of_m(tmp_path: Path) -> None:
    from tests.test_analyze_nexus_sources import _ring_stack

    model = create_analyze_view_model(_context(_profiles()))
    model.set_mode("giwaxs")
    model.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
    request = model.batch_requests()[3]
    row = model.series_row(request, "radial")  # a row of the Series map, and of the live map of a batch
    assert str(row[3]) == "stack.h5 frame 4 of 12" and "#" not in str(row[3])  # (the file, as in a batch's tables)


# -- 6. export when the folder of the data takes no file; the names offered ----------------------------------------


def _blocked(tmp_path: Path) -> Path:
    """A folder that cannot be made: its parent is a file."""
    blocker = tmp_path / "read_only.txt"
    blocker.write_text("x", encoding="utf-8")
    return blocker / "gimap_analysis"


def _ask(monkeypatch, answer) -> list:
    asked = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda _parent, title, *_args: (asked.append(title), str(answer))[1]))
    return asked


def test_export_from_a_folder_that_takes_no_file_asks_once_for_the_session(tmp_path: Path, monkeypatch) -> None:
    page = _gisaxs_page(tmp_path / "raw")
    out = tmp_path / "mine"
    out.mkdir()
    try:
        monkeypatch.setattr(page.view_model, "default_export_dir", lambda: _blocked(tmp_path))
        asked = _ask(monkeypatch, out)
        page.export_current()
        assert len(asked) == 1 and asked[0].startswith("Cannot write next to the data (")
        assert (out / "film_00_horizontal.csv").is_file() and page.session_export_dir() == out
        assert page.status_level() == "ok" and page.status_text().endswith(str(out))
        page.show_frame(1)
        _done(page)
        page.export_current()  # the next one: the folder of the session, no question
        assert len(asked) == 1 and (out / "film_01_horizontal.csv").is_file()
    finally:
        _close(page)
    page = _gisaxs_page(tmp_path / "raw")  # a new session, and the question is cancelled
    try:
        monkeypatch.setattr(page.view_model, "default_export_dir", lambda: _blocked(tmp_path))
        asked = _ask(monkeypatch, "")
        page.export_current()
        assert len(asked) == 1 and page.status_level() == "error" and page.status_text().startswith("Export failed: ")
    finally:
        _close(page)


def _watched(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    page = _gisaxs_page(run, bands=(204, 205))
    page.start_watch(run)
    page.auto_export_check.setChecked(True)
    monkeypatch.setattr(page.view_model, "default_export_dir", lambda: _blocked(tmp_path))
    assert page.poll_watch() == []
    return page, run


def _arrive(page, path: Path, band: int, seed: int) -> None:
    save_tiff(path, _gisaxs_frame(band, seed=seed))
    assert page.poll_watch() == []  # size seen once
    assert [item.name for item in page.poll_watch()] == [path.name]
    _done(page)


def test_auto_export_while_watching_goes_into_the_session_folder(tmp_path: Path, monkeypatch) -> None:
    page, run = _watched(tmp_path, monkeypatch)
    out = tmp_path / "mine"
    out.mkdir()
    try:
        asked = _ask(monkeypatch, out)
        _arrive(page, run / "film_05.tif", 206, 5)
        assert len(asked) == 1 and (out / "film_05_horizontal.csv").is_file()
        _arrive(page, run / "film_06.tif", 207, 6)
        assert len(asked) == 1 and (out / "film_06_horizontal.csv").is_file()  # asked once for the session
        assert page.auto_export_check.isChecked()
    finally:
        page.stop_watch()
        _close(page)


def test_auto_export_is_switched_off_when_no_folder_takes_the_files(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.analyze.presentation.bindings.batch_watch import AUTO_EXPORT_OFF

    from src.gimap.features.analyze.presentation.bindings import batch_watch

    page, run = _watched(tmp_path, monkeypatch)
    said = []
    monkeypatch.setattr(batch_watch, "show_toast", lambda _parent, text, **options: said.append((text, options.get("level"))))
    try:
        asked = _ask(monkeypatch, "")
        _arrive(page, run / "film_05.tif", 206, 5)
        assert len(asked) == 1 and not page.auto_export_check.isChecked()
        assert len(said) == 1 and said[0][1] == "warning" and said[0][0].startswith(AUTO_EXPORT_OFF.split("(")[0])
        assert not list(tmp_path.rglob("film_05_horizontal.csv"))
        _arrive(page, run / "film_06.tif", 207, 6)
        assert len(asked) == 1  # not asked at every look
        assert ZH.get(AUTO_EXPORT_OFF)
    finally:
        page.stop_watch()
        _close(page)


def test_saved_pictures_and_tables_are_offered_under_the_name_of_the_frame(tmp_path: Path, monkeypatch) -> None:
    from tests.test_analyze_nexus_sources import _ring_stack

    page = _page(_context(_profiles()))
    offered = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda _parent, _title, path, *_args: (offered.append(Path(path).name), ("", ""))[1]))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(_ring_stack(tmp_path / "stack.h5", 12))])
        _done(page)
        page.show_frame(0, 3)
        _done(page)
        page.save_image()
        page.save_plot(page.top_plot, "upper")
        page.save_map_data()
        assert offered == ["stack_frame0003_detector.png", "stack_frame0003_upper.png", "stack_frame0003_qmap.csv"]
    finally:
        _close(page)


# -- 8. an In-situ series of poor fits -----------------------------------------------------------------------------


def test_an_in_situ_series_with_poor_fits_ends_with_a_warning_that_counts_them(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.fitting.application.series_fit import POOR_CHI2
    from src.gimap.features.fitting.presentation.single import series_page
    from tests.test_fit_series import _model, _pages, _series, _wait
    from tests.test_fit_series_table import _ripple

    said = []
    monkeypatch.setattr(series_page, "show_toast", lambda _parent, text, **options: said.append((text, options.get("level"))))
    single, series = _pages()
    try:
        folder = _series(tmp_path)
        _ripple(folder, "run_00002_fit_input.dat")
        _ripple(folder, "run_00004_fit_input.dat")
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        series.open_series(folder)
        series.start_same.setChecked(True)
        series.start()
        _wait(series)
        poor = len(series._poor_frames())
        assert poor == 2
        done = "Series done: {count} frames fitted, {poor} of them poor fits (χ²ᵣ above {limit}), {failed} failed."
        assert series.status_label.text() == done.format(count=5, poor=2, limit=f"{POOR_CHI2:g}", failed=0)
        assert said[-1] == (series.status_label.text(), "warning")
        series._stop.set()
        series._finish()
        stopped = "Series stopped: {count} frames fitted, {poor} of them poor fits (χ²ᵣ above {limit}), {failed} failed."
        assert series.status_label.text() == stopped.format(count=5, poor=2, limit="10", failed=0)
        apply_language("zh", [series])
        series.refresh_language()
        assert series.status_label.text() == ZH[stopped].format(count=5, poor=2, limit="10", failed=0)
        assert ZH.get(done) and ZH[done] != done
    finally:
        apply_language("en")
        series.dispose()
        single.dispose()


# -- 9. a GIWAXS frame: what Fitting fits ---------------------------------------------------------------------------


GIWAXS_TIP = ("Open the I(q) in Fitting (q in Å⁻¹). Fitting's models are particle form factors for the horizontal cut of "
              "a GISAXS frame: the peaks of a GIWAXS frame are fitted in Batch Export ▸ Peaks of the ring regions.")
SENT_GIWAXS = ("Fitting ▸ In-situ series fits particle form factors, made for the horizontal cut of a GISAXS frame: they do "
               "not describe diffraction peaks. To follow the peaks of this series use Batch Export ▸ Peaks of the ring "
               "regions.")


def test_on_a_giwaxs_frame_the_fitting_button_says_where_its_peaks_are_fitted(tmp_path: Path) -> None:
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    _app()
    page = AnalyzePage(create_analyze_view_model(_context(_profiles())), send_to_fitting=lambda path, side: None)
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(save_tiff(tmp_path / "ring_0.tif", _two_ring_frame(300.0)))])
        _done(page)
        assert page.fit_button.toolTip() == GIWAXS_TIP
        apply_language("zh", [page])
        page.refresh_language()
        assert page.fit_button.toolTip() == ZH[GIWAXS_TIP]
        apply_language("en", [page])
        page.refresh_language()
        page.set_mode_choice("gisaxs")  # a GISAXS cut: the halves, as before
        page.run_analysis()
        _done(page)
        assert page.fit_button.toolTip().startswith("Open the curve in Fitting (q in Å⁻¹). GISAXS: the horizontal cut, ")
    finally:
        apply_language("en")
        _close(page)


def _send_dialog(destination: Path, *, gisaxs: bool = False):
    from src.gimap.features.analyze.application import BatchChoices
    from src.gimap.features.analyze.domain import SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    _app()
    return BatchExportDialog(frames=8, files=1, stem="scan", settings_text="GIWAXS · profile “P08”", choices=BatchChoices(),
                             destination=destination, subfolder=True, series=SeriesCorrection(),
                             curve_options=[("radial", "I(q)")], gisaxs=gisaxs, for_fitting=True,
                             frame_stem="scan_frame0000", cut_text=lambda: "I(χ) ring: q 0.981–1.011 Å⁻¹ for every frame",
                             pin_cuts=True, set_pin_cuts=lambda _on: None)


def test_send_series_on_a_giwaxs_frame_says_what_fitting_fits(tmp_path: Path) -> None:
    dialog = _send_dialog(tmp_path)
    try:
        dialog.show()
        assert dialog.halves_note.isVisible() and dialog.halves_note.text() == SENT_GIWAXS
        assert dialog.halves_combo.isHidden()  # an I(q) has no halves
        apply_language("zh", [dialog])
        dialog._sync()
        assert dialog.halves_note.text() == ZH[SENT_GIWAXS]
    finally:
        apply_language("en")
        dialog.done(0)
        dialog.deleteLater()
    gisaxs = _send_dialog(tmp_path, gisaxs=True)
    try:
        gisaxs.show()
        assert gisaxs.halves_note.text() != SENT_GIWAXS and not gisaxs.halves_combo.isHidden()
    finally:
        gisaxs.done(0)
        gisaxs.deleteLater()


def test_send_series_opens_with_its_options_in_view_when_an_earlier_export_is_named(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application.batch_folder import RECORD_FORMAT

    earlier = tmp_path / "scan_fitting"
    files = ["fit_input/scan_frame0000_fit_input.dat", "README.txt"]
    for name in files:
        (earlier / name).parent.mkdir(parents=True, exist_ok=True)
        (earlier / name).write_text("x", encoding="utf-8")
    (earlier / "scan_batch.json").write_text(json.dumps({"format": RECORD_FORMAT, "frames": ["a"] * 8, "started": "2026-10-14",
                                                         "files": [*files, "scan_batch.json"]}), encoding="utf-8")
    dialog = _send_dialog(tmp_path)
    try:
        dialog.show()
        _settle()
        assert dialog.folder_note.isVisible()  # the bottom has two more lines: the earlier export
        for start in (300, 400):  # however short it was before
            dialog.resize(dialog.width(), start)
            _settle()
            dialog.fit_to_screen(room=700)
            _settle()
            scroll = dialog.options_scroll
            assert scroll.verticalScrollBar().maximum() == 0, (start, scroll.viewport().height(), scroll.widget().height())
    finally:
        dialog.done(0)
        dialog.deleteLater()


# -- 10. the Chinese table of Compare -------------------------------------------------------------------------------


def test_the_chinese_of_compares_former_folder_tooltip_is_gone() -> None:
    from src.gimap.app.presentation.i18n.zh_compare import COMPARE_ZH

    assert "Every curve file of a folder (q, I columns), in counting order: one series" not in COMPARE_ZH
