"""The new-user path: the Start page, the guided GIWAXS page (no model), and how the shell connects them."""

from __future__ import annotations

import gc
import time
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QLabel

from src.gimap.app.presentation.home_page import HomePage
from src.gimap.features.assistant.presentation import GuidedAnalysis
from src.gimap.features.assistant.presentation.guided_details import PeakDetails
from src.gimap.features.assistant.presentation.guided_results import GuidedResultsPanel
from tests.test_assistant_calibration import agbh_image, giwaxs_frame, save_tiff
from tests.test_assistant_gui import _app, _wait, analyze  # noqa: F401 - the fixture


def test_the_start_page_asks_what_you_want_to_know() -> None:
    _app()
    page = HomePage()
    tasks, asked, dropped, opened = [], [], [], []
    page.taskRequested.connect(tasks.append)
    page.askRequested.connect(asked.append)
    page.filesDropped.connect(dropped.append)
    page.openRequested.connect(lambda: opened.append(True))
    assert list(page.cards) == ["giwaxs", "gisaxs", "series", "calibrate"]
    page.cards["giwaxs"].click()
    page.cards["calibrate"].click()
    page.ask_edit.setText("Cu on PEO, αi 0.4° — how does it crystallise?")
    page.ask_button.click()
    page.open_button.click()
    page.drop_zone.filesDropped.emit(["C:/data/frame_00001_m01.nxs"])
    assert tasks == ["giwaxs", "calibrate"] and asked == ["Cu on PEO, αi 0.4° — how does it crystallise?"]
    assert opened == [True] and dropped == [["C:/data/frame_00001_m01.nxs"]]


def _attach(page, guided) -> None:
    """Wire the automatic analysis into the workspace the way the main window does."""
    from src.gimap.app.main_window import _automatic_outcome

    page.add_step_panel("results", guided.controls)
    page.add_result_panel(guided.results)
    page.set_automatic_analysis(run=guided.run, find_geometry=guided.find_geometry)
    guided.started.connect(page.automatic_started)
    guided.finished.connect(lambda report: page.automatic_finished(
        *_automatic_outcome(report), show_results=report.get("procedure") != "geometry"))


def test_one_click_gives_results_with_their_evidence_without_any_model(analyze) -> None:  # noqa: F811
    window, page, context = analyze
    GuidedResultsPanel.details_open = False
    guided = GuidedAnalysis(page.automation, parent=page)
    _attach(page, guided)
    assert not page.run_pipeline_button.isHidden() and not page.find_geometry_button.isHidden()
    guided.notes_edit.setPlainText("GIWAXS, alpha_i = 0.2 deg, 12.4 keV")
    assert guided.notes_found.text() == "Found in the notes: αi = 0.2°, energy = 12.4 keV"
    progress: list[str] = []
    guided.progressed.connect(progress.append)
    page.run_pipeline_button.click()
    assert page.step_rail.state("results") == "busy"
    _wait(lambda: guided.report is not None, 120)
    report = guided.report
    assert report["ok"] and report["needs_attention"] == [] and report["procedure"] == "giwaxs"
    assert page.step_rail.state("results") == "ok" and page.current_right() == "results"
    # The status during the run says what is done in words, never "find_peaks: …".
    assert progress and all("_" not in line.split(" — ")[0] for line in progress), progress
    assert any(line.startswith("Finding the peaks of I(q) — ") for line in progress), progress
    assert guided.status_label.text() == "Done — see the Results tab."
    results = guided.results
    badge = results.findChildren(QLabel)[0]
    assert badge.text().startswith("GIWAXS — ") and badge.property("gimapRole") == "success" and not badge.styleSheet()
    # Nothing in the Results tab asks for more width than a narrow panel has (the q map shrinks with it).
    assert results.minimumSizeHint().width() <= 360, results.minimumSizeHint()
    peaks = results.peak_table
    assert peaks.rowCount() == len(report["peaks"]) > 0
    found = [float(peaks.item(row, 0).text()) for row in range(peaks.rowCount())]
    for expected in (0.40, 0.80):  # the synthetic film's lamellar orders
        assert min(abs(q - expected) for q in found) < 0.02, found
    assert results.q_map is not None and not results.q_map.pixmap().isNull()  # see what was and was not measured
    assert "GIWAXS" in results.report_view.toPlainText()
    # Each peak can be seen on the map, and every column says how it was obtained.
    assert "2π / q" in peaks.horizontalHeaderItem(1).toolTip() and "Scherrer" in peaks.horizontalHeaderItem(4).toolTip()
    assert not results.peak_details.is_expanded() and results.peak_map.pixmap() is None  # closed: nothing drawn
    results.peak_details.set_expanded(True)  # Fit details: the fit, its numbers and where the peak is
    shown = results.peak_map.pixmap().cacheKey()
    peaks.selectRow(1 if peaks.currentRow() != 1 else 0)
    assert not results.peak_map.pixmap().isNull() and results.peak_map.pixmap().cacheKey() != shown
    details = results.peak_details.findChild(PeakDetails)
    assert details.plot.curve_count() == 3  # the points fitted, the Gaussian + background, the background
    shown_values = [item.text() for item in details.values.findChildren(QLabel)]
    assert "q" in shown_values and any("±" in text and "Å⁻¹" in text for text in shown_values)
    assert "SNIP" in details.method.text() and "√χ²ᵣ" in details.method.text()
    GuidedResultsPanel.details_open = False
    # Running again replaces the whole panel (nested layouts too).
    guided.report = None
    page.run_pipeline_button.click()
    _wait(lambda: guided.report is not None, 120)
    distance = [item for item in results.findChildren(QLabel) if item.text() == "Distance"]
    assert len(distance) == 1
    # The report to keep or send: one web page with the q map and I(q) as pictures.
    page_html = guided.report_page()
    assert page_html.startswith("<!DOCTYPE html>") and page_html.count("data:image/png;base64,") == 2
    assert "film_giwaxs.tif" in page_html and "Peaks" in page_html
    assert "<h1>GIWAXS report</h1>" in page_html and "Automatic analysis, no AI" in page_html


def test_the_check_picture_shows_where_each_ring_was_measured() -> None:
    import numpy as np
    from matplotlib.figure import Figure

    from src.gimap.features.analyze.presentation.automation import _draw_rings
    from src.gimap.features.assistant.application import ring_overlays

    image = np.full((200, 200), np.nan)  # q∥ −4…4 (columns), qz 0…4 (rows, top = 4)
    image[:, 105:] = 50.0  # only q∥ > 0.2 measured
    rings = ring_overlays({"rings": [{"q": 3.0, "shadowed": [[60.0, 90.0]]}, {"q": None}]})
    assert rings == [{"q": 3.0, "shadowed": [[60.0, 90.0]], "label": "3"}]
    axes = Figure().add_subplot()
    _draw_rings(axes, image, (-4.0, 4.0, 0.0, 4.0), rings)
    parts = {}
    for line in axes.get_lines():
        chi = np.degrees(np.arctan2(line.get_xdata(), line.get_ydata()))
        parts.setdefault(line.get_color(), []).append((round(chi.min()), round(chi.max())))
    assert parts["white"] == [(4, 60)]  # measured
    assert parts["#ff9f1c"] == [(60, 90)]  # in the shadow
    assert (-90, 4) in parts["#e63946"]  # not measured: no pixels at q∥ < 0.2
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["not measured", "measured", "in a shadow"]
    assert [text.get_text() for text in axes.texts] == ["3"]


def test_the_check_step_says_once_where_the_rings_are_shadowed_or_unmeasured() -> None:
    from src.gimap.features.assistant.presentation.guided_text import coverage_notes

    rings = [  # the real P03 frame
        {"q": 2.56, "missing": [[0.0, 11.0], [67.0, 90.0]], "shadowed": [[67.0, 90.0]],
         "notes": ["|χ| 67–90° is shadowed: …", "|χ| < 11° is not measured (missing wedge); …"]},
        {"q": 2.942, "missing": [[0.0, 13.0], [56.0, 90.0]], "shadowed": [[59.0, 90.0]], "notes": []},
        {"q": 1.0, "missing": [[40.0, 45.0]], "shadowed": [], "notes": []},
    ]
    (shadow, shadow_detail), (wedge, _detail) = coverage_notes(rings)
    assert shadow.startswith("Shadow (orange on the map): |χ| 67–90° at q 2.56, 59–90° at q 2.94 Å⁻¹.")
    assert shadow_detail == "|χ| 67–90° is shadowed: …"
    assert wedge.startswith("Missing wedge (red dashed next to qz): |χ| < 11° at q 2.56, < 13° at q 2.94 Å⁻¹")
    assert coverage_notes([{"q": 1.0, "missing": [], "shadowed": []}]) == []


def test_a_ring_says_in_one_line_why_its_orientation_is_not_determined() -> None:
    from src.gimap.features.assistant.presentation.guided_text import ring_summary

    reason = ("Only 57% of the orientation range Herman's f weighs (sin χ) is measured at this q (|χ| 11–36°, "
              "38–67°); f and the orientation distribution need most of it. Within the measured part the ring is "
              "strongest at |χ| ≈ 16°.")
    text, detail = ring_summary({  # the real 2.56 Å⁻¹ ring
        "q": 2.56, "herman": None, "weighted_coverage": 0.57, "reason": reason,
        "missing": [[0.0, 11.0], [36.0, 38.0], [67.0, 90.0]], "shadowed": [[67.0, 90.0]], "notes": ["|χ| 67–90° is shadowed"],
    })
    assert text == ("Ring q ≈ 2.56 Å⁻¹: orientation not determined — only 57% of the range Herman's f needs is measured "
                    "(in a shadow at |χ| 67–90°; missing wedge below 11°). Hover for details.")
    assert detail.startswith(reason) and "shadowed" in detail
    text, _detail = ring_summary({"q": 0.36, "herman": 0.52, "herman_isotropic": -0.03,
                                  "texture": "oriented along the surface normal (out-of-plane, χ ≈ 0°)"})
    assert text == ("Ring q ≈ 0.36 Å⁻¹: f = 0.52 (a random ring would give -0.03 here) — oriented along the surface "
                    "normal (out-of-plane, χ ≈ 0°)")


def test_a_peak_at_the_end_of_the_data_is_not_called_reliable() -> None:
    from src.gimap.features.assistant.presentation.guided_text import PEAK_COLUMNS, peak_rows

    # The verdict comes right after q, d and FWHM; the long orientation words last (the column that stretches).
    assert [header for header, _tip in PEAK_COLUMNS][3:] == ["trust", "size (nm)", "in-/out-of-plane"]
    # The real 1.655 Å⁻¹ "peak" is a shoulder on the rising edge of I(q), 0.01 Å⁻¹ from where the data start.
    edge, inner = peak_rows([
        {"q": 1.655, "d_A": 3.797, "fwhm": 0.0123, "flags": ["at_edge", "overlap"], "caveat": ""},
        {"q": 2.942, "d_A": 2.136, "fwhm": 0.0793, "flags": [], "caveat": "", "size_nm": 7.14, "size_is_lower_bound": True},
    ])
    assert edge[3][0] == "at the end of the data: check" and "cut off" in edge[3][1]
    assert inner[3] == ("reliable", "") and inner[4][0] == "≥ 7.14" and inner[5][0] == "—"


def test_only_questions_with_a_field_are_called_questions() -> None:
    from src.gimap.features.assistant.presentation import automatic_outcome

    _app()
    report = {
        "ok": True, "procedure": "giwaxs", "frame": "C:/data/film_00001.tif", "peaks": [], "rings": [],
        "needs_attention": [
            {"item": "incidence angle αi", "why": "No αi anywhere.", "option": "incidence_deg", "hint": "The notes."},
            {"item": "beam centre", "why": "The axis is 25 px off.", "option": None, "hint": "Look at the cut."},
        ],
    }
    state, line = automatic_outcome(report)
    assert state == "warn" and line.startswith("GIWAXS — 0 peaks") and line.endswith("; 1 question(s).")  # Results step shown
    guided = GuidedAnalysis(lambda: None)
    guided._finished(report)
    assert list(guided.question_fields) == ["incidence_deg"] and not guided.questions.isHidden()
    assert guided.status_label.text() == "Done — see the Results tab. 1 question(s) below."
    texts = [item.text() for item in guided.results.findChildren(QLabel)]
    assert texts[0].startswith("GIWAXS — 0 peaks (0 reliable), 0 rings analysed. 1 question(s) in the Results step.")
    assert "1 point(s) to check — see below" in texts and "Beam centre: The axis is 25 px off." in texts
    # Advice alone (no field to fill in) is a point to check, not a question.
    advice = dict(report, needs_attention=report["needs_attention"][1:])
    assert automatic_outcome(advice) == ("ok", "GIWAXS — 0 peaks (0 reliable), 0 rings analysed; 1 point(s) to check.")
    guided._finished(advice)
    assert not guided.question_fields and guided.questions.isHidden()
    assert guided.status_label.text() == "Done — see the Results tab. 1 point(s) to check there."
    # The pipeline's own lines ("tool: summary", also used by the command line) are shown in words.
    guided._stepped({"state": "start", "tool": "set_halves", "arguments": {"side": "both_abs"}})
    guided._progress("set_halves: halves: both_abs")
    assert guided.status_label.text() == "Choosing the halves of the cut — Both halves on |qy|"
    guided._progress("fit_horizontal_cut: fitting spheres and cylinders to I(qy) (about half a minute)…")
    assert guided.status_label.text().startswith("Fitting form-factor models to the horizontal cut — fitting spheres")
    guided.results.deleteLater()


def test_two_points_about_one_value_are_one_question() -> None:
    # GISAXS without αi: the incidence angle and the Yoneda band are both answered by αi — one field, one question.
    from src.gimap.features.assistant.presentation import automatic_outcome

    _app()
    report = {
        "ok": True, "procedure": "gisaxs", "frame": "C:/data/galaxi_data.tif", "peaks": [], "rings": [], "gisaxs": {},
        "needs_attention": [
            {"item": "incidence angle αi", "why": "No αi anywhere.", "option": "incidence_deg", "hint": "The notes."},
            {"item": "Yoneda band", "why": "No Yoneda band above the horizon.", "option": "incidence_deg", "hint": "Check αi."},
        ],
    }
    state, line = automatic_outcome(report)
    assert state == "warn" and line.startswith("GISAXS — ") and line.endswith("; 1 question(s).")
    guided = GuidedAnalysis(lambda: None)
    guided._finished(report)
    assert list(guided.question_fields) == ["incidence_deg"]
    assert guided.status_label.text() == "Done — see the Results tab. 1 question(s) below."
    tip = guided.question_fields["incidence_deg"].toolTip()
    assert "No αi anywhere." in tip and "No Yoneda band above the horizon." in tip  # both reasons on the one field
    texts = [item.text() for item in guided.results.findChildren(QLabel)]
    assert texts[0].endswith(" 1 question(s) in the Results step.")
    guided.results.deleteLater()


def test_find_geometry_is_reported_as_geometry_not_as_giwaxs() -> None:
    import re

    from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
    from PyQt5.QtGui import QImage

    from src.gimap.features.assistant.presentation import automatic_outcome
    from src.gimap.features.assistant.presentation.guided_report import report_page

    _app()
    report = {  # Find Geometry on a GISAXS frame, αi unknown
        "ok": True, "procedure": "geometry", "measurement": "gisaxs", "frame": "C:/data/galaxi_data.tif",
        "peaks": [], "rings": [], "decisions": [], "steps": [],
        "geometry": {"distance_mm": 1730.0, "beam_center_px": [500.0, 900.0], "wavelength_A": 1.34},
        "calibration_quality": {"assessment": "good: 7 lines of the standard land within 0.05% of their q on average"},
        "needs_attention": [{"item": "incidence angle αi", "why": "No αi anywhere.", "option": "incidence_deg", "hint": ""}],
    }
    state, line = automatic_outcome(report)
    assert state == "warn"  # the αi field waits in the Results step: the workspace shows it
    assert line == "Geometry — good: 7 lines of the standard land within 0.05% of their q on average; 1 question(s)."
    assert automatic_outcome(dict(report, needs_attention=[]))[0] == "ok"
    guided = GuidedAnalysis(lambda: None)
    guided._finished(report)
    assert list(guided.question_fields) == ["incidence_deg"]
    texts = [item.text() for item in guided.results.findChildren(QLabel)]
    assert texts[0].startswith("Geometry — good: 7 lines") and not any("GIWAXS" in text or "peaks" in text for text in texts)
    assert "Checks" not in texts  # no rings were analysed: nothing to check against them
    assert guided.report_markdown().startswith("# Geometry — galaxi_data.tif")
    assert "Scherrer" not in guided.report_markdown()

    class Page:
        @staticmethod
        def preview_png(width, rings=()):
            image, data = QImage(40, 20, QImage.Format_RGB32), QByteArray()
            image.fill(0)
            buffer = QBuffer(data)
            buffer.open(QIODevice.WriteOnly)
            image.save(buffer, "PNG")
            return bytes(data)

    page = report_page(report, None, Page)
    assert "<h1>Geometry report</h1>" in page and "GIWAXS" not in page and page.count("data:image/png;base64,") == 1
    captions = re.findall(r"<figcaption>(.*?)</figcaption>", page)
    assert captions == ["The q map of the frame with the geometry found."]
    guided.results.deleteLater()


def test_a_frame_shown_before_the_end_of_a_run_is_handled_is_looked_at_after_it() -> None:
    # The worker thread has ended but its result still waits in the event queue: the run is not over yet.
    import threading

    _app()
    guided = GuidedAnalysis(lambda: None)
    frame_a, frame_b = "C:/data/film_a.tif", "C:/data/film_b.tif"
    report = {"ok": True, "procedure": "giwaxs", "frame": frame_a, "peaks": [], "rings": [], "needs_attention": []}
    guided.frame_shown(frame_a)
    guided._finished(report)
    guided._thread = threading.Thread(target=lambda: None, daemon=True)
    guided._thread.start()
    guided._thread.join()
    assert not guided.running()
    guided.frame_shown(frame_b)  # handled before the finished signal
    assert guided.report is report and not guided.results.isHidden()
    rerun = dict(report)
    guided._finished(rerun)  # the queued result of the run on A arrives
    assert guided.report is None and guided.results.isHidden()
    assert guided.status_label.text() == "Results are for film_a.tif; run again for this frame"
    guided.frame_shown(frame_a)
    assert guided.report is rerun and not guided.results.isHidden()
    guided.results.deleteLater()


def test_saves_start_next_to_the_data(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.assistant.presentation import guided_text

    monkeypatch.setattr(guided_text, "_SAVE_FOLDERS", {})
    frame = tmp_path / "beamtime" / "P3HT_00012.tif"
    frame.parent.mkdir()
    assert guided_text.proposed_save_path(frame, "report.html") == str(frame.parent / "P3HT_00012_report.html")
    (frame.parent / "gimap_analysis").mkdir()  # where Analyze's exports go: used when it exists
    assert guided_text.proposed_save_path(frame, "report.html") == str(frame.parent / "gimap_analysis" / "P3HT_00012_report.html")
    guided_text.remember_save_folder(frame, str(tmp_path / "results" / "x.html"))
    (tmp_path / "results").mkdir()
    assert guided_text.proposed_save_path(frame, "fit.png") == str(tmp_path / "results" / "P3HT_00012_fit.png")


def test_a_picture_shrinks_with_its_column() -> None:
    from PyQt5.QtGui import QPixmap

    from src.gimap.features.assistant.presentation.guided_text import ScaledPixmapLabel

    _app()
    picture = ScaledPixmapLabel()
    source = QPixmap(500, 400)
    picture.setPixmap(source)
    assert picture.minimumSizeHint().width() == 0 and picture.sizeHint().width() == 500
    assert picture.hasHeightForWidth() and picture.heightForWidth(250) == 200 and picture.heightForWidth(900) == 400
    picture.show()  # a hidden widget gets its resize event only when it is shown
    picture.resize(250, 200)
    assert picture.pixmap().width() == 250 and picture.sizeHint().width() == 500  # the hint never follows the shown size
    picture.resize(800, 400)
    assert picture.pixmap().width() == 500  # never larger than the picture itself
    picture.setText("No q map for this frame.")
    assert picture.source_pixmap() is None and not picture.hasHeightForWidth()
    picture.close()


def test_questions_only_a_person_can_answer_become_fields(tmp_path: Path) -> None:
    _app()
    from PyQt5.QtWidgets import QMainWindow

    from src.gimap.app import AppContext
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.features.assistant.infrastructure import LocalFileExplorer
    from src.gimap.features.calibration.bootstrap import create_headless_calibration
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    root = tmp_path / "beamtime" / "raw"
    frame = save_tiff(root / "film" / "P3HT_00012.tif", giwaxs_frame())
    save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    context = AppContext(
        settings=InMemorySettingsRepository({}), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(), instrument_profiles=InMemoryInstrumentProfileRepository([]),
    )
    page = AnalyzePage(create_analyze_view_model(context))
    window = QMainWindow()
    window.setCentralWidget(page)
    page.add_paths([frame])
    assert page.tasks.wait(60)
    guided = GuidedAnalysis(page.automation, explorer=LocalFileExplorer(), calibrator=create_headless_calibration(), parent=page)
    _attach(page, guided)
    guided.run()
    _wait(lambda: guided.report is not None, 120)
    first = guided.report
    assert not first["ok"]  # a TIFF has neither energy nor pixel size
    asked = set(guided.question_fields)
    assert "energy_kev" in asked and guided.questions.isVisibleTo(page)
    assert page.current_step() == "results" and page.step_rail.state("results") == "warn"
    guided.question_fields["energy_kev"].setText("12.3984")
    guided.report = None
    guided.again_button.click()
    _wait(lambda: guided.report is not None, 120)
    second = guided.report
    assert "pixel_size_um" in guided.question_fields  # the next thing only the person knows
    guided.question_fields["pixel_size_um"].setText("100")
    guided.notes_edit.setPlainText("alpha_i = 0.2 deg")
    guided.report = None
    guided.again_button.click()
    _wait(lambda: guided.report is not None, 120)
    third = guided.report
    assert third["ok"] and third["needs_attention"] == [] and not guided.questions.isVisibleTo(page)
    assert guided._answers == {"energy_kev": "12.3984", "pixel_size_um": "100"}  # every earlier answer was kept
    assert second is not third and guided.options().energy_kev == 12.3984
    page.tasks.wait(60)
    window.close()


def test_the_shell_starts_on_the_start_page_and_routes_tasks(monkeypatch) -> None:
    app = _app()
    from PyQt5.QtWidgets import QFileDialog

    from main import MainWindow

    # With nothing listed a Start card may also offer to open frames: the file dialog is cancelled here.
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *args, **kwargs: asked.append(1) or ([], "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *args, **kwargs: asked.append(1) or ""))
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    window = MainWindow(AppContext(
        settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
        instrument_profiles=InMemoryInstrumentProfileRepository([]),
    ))
    window.show()
    deadline = time.monotonic() + 15
    while not getattr(window, "_initialization_completed", False) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    components = window.components
    assert components.current_page_key() == "home"
    assert "guided" not in components.pages  # one workspace: the automatic analysis is its Results step
    components.home_page.cards["giwaxs"].click()
    assert components.current_page_key() == "analyze"
    assert components.analyze_page.mode_combo.currentData() == "giwaxs"  # the task preselects the technique
    components.show_page("home")
    components.home_page.cards["gisaxs"].click()
    assert components.current_page_key() == "analyze" and components.analyze_page.mode_combo.currentData() == "gisaxs"
    assert components.analyze_page.run_pipeline_button.isVisibleTo(components.analyze_page)
    window.close()
    QApplication.processEvents()
    gc.collect()


def test_the_start_and_the_end_of_a_series_are_compared() -> None:
    from src.gimap.features.assistant.application import series_changes

    # Sample 117: 2.56 and 3.90 are there before the deposition, 2.94 and 3.41 appear during it.
    start = {"peaks": [{"q": 2.557, "caveat": ""}, {"q": 3.899, "caveat": ""}, {"q": 4.893, "caveat": "a spike …"}]}
    end = {"peaks": [{"q": 2.560, "caveat": ""}, {"q": 2.942, "caveat": ""}, {"q": 3.408, "caveat": ""},
                     {"q": 4.893, "caveat": "a spike …"}]}
    rows = [(round(row["q"], 3), row["change"]) for row in series_changes(start, end)]
    assert rows == [(2.56, "present at both"), (2.942, "appeared"), (3.408, "appeared"), (3.899, "disappeared")]
    # The real series: a peak weak at the start grows; 3.899 → 3.884 (0.4 %, more than the sharp peaks'
    # width) may be one line moving: one row says so instead of "disappeared" plus "appeared".
    start = {"peaks": [{"q": 1.649, "fwhm": 0.02, "caveat": "weak (3–5σ): tentative"}, {"q": 1.884, "fwhm": 0.04, "caveat": ""},
                       {"q": 2.562, "fwhm": 0.12, "caveat": ""}, {"q": 3.899, "fwhm": 0.012, "caveat": ""},
                       {"q": 4.3, "fwhm": 0.05, "caveat": ""}]}
    end = {"peaks": [{"q": 1.655, "fwhm": 0.012, "caveat": ""}, {"q": 1.881, "fwhm": 0.045, "caveat": ""},
                     {"q": 2.54, "fwhm": 0.13, "caveat": ""}, {"q": 3.884, "fwhm": 0.011, "caveat": ""},
                     {"q": 4.3, "fwhm": 0.05, "caveat": "weak (3–5σ): tentative"}]}
    rows = {round(row["q"], 3): (row["start"], row["end"], row["change"]) for row in series_changes(start, end)}
    assert rows == {
        1.655: ("weak", "yes", "grew (weak at the start)"),
        1.881: ("yes", "yes", "present at both"),
        2.54: ("yes", "yes", "shifted -0.022 Å⁻¹ (-0.9%) from 2.562"),  # inside half its width: the same line
        3.884: ("yes", "yes", "moved? 3.899 → 3.884 (-0.4%): one line shifting further than its width, or one "
                              "line replacing another"),
        4.3: ("yes", "weak", "faded (weak at the end)"),
    }
    from src.gimap.features.assistant.application import series_markdown

    text = series_markdown({**start, "ok": True, "frames": {"first": 1, "summed": 10}},
                           {**end, "ok": True, "frames": {"first": 394, "summed": 10}})
    assert "Frames 1–10 compared with frames 394–403" in text and "| 1.655 | weak | yes | grew (weak at the start) |" in text
    assert series_markdown({"ok": False}, end) == ""
