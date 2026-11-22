"""The normalisation, the dark frame, the Lorentz factor and the display in the Analyze window.

- Mask & corrections ▸ Normalisation is off by default and always says which header values the frame shown offers;
  a header value chosen divides the frame and the section says by what, each term with its source; the dark frame
  and the Lorentz factor are set from their controls; every new text has its Chinese;
- log display: valid pixels at or below 0 take the lowest colour (not a hole), the automatic limits come from the
  positive values; linear display: percentile limits and a bar around them;
- Find Calibration does not switch the right panel to Results; the Results geometry says the αi in use;
- Export Curves… asks for its folder (starting in the last one) and remembers it; the image is the widest panel and
  the side panels can be hidden.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

from src.gimap.app.presentation.components.detector_display import display_image
from src.gimap.app.presentation.i18n import ZH, apply_language
from src.gimap.app.presentation.i18n.zh_normalisation import NORMALISATION_ZH
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_assistant_calibration import SHAPE
from tests.test_review_analyze_run_and_clear import _frames, _profiles
from tests.test_series_map import _ring_frame

ANALYZE = Path(__file__).resolve().parents[1] / "src/gimap/features/analyze"


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _edf(path: Path, data: np.ndarray, header: dict) -> Path:
    import fabio

    fabio.edfimage.EdfImage(data=np.asarray(data, dtype=np.int32), header=header).write(str(path))
    return path


def test_the_normalisation_section_says_the_header_values_and_by_what_the_frame_was_multiplied(tmp_path: Path) -> None:
    apply_language("en")
    path = _edf(tmp_path / "film_0001.edf", _ring_frame(200.0, seed=1), {"ExposureTime": "20", "Monitor": "5000"})
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(path)])
        _done(page)
        plain = page.view_model.state.analysis
        assert plain.normalisation is None and page.normalisation_status.isHidden()  # off by default
        assert page.header_values_label.text() == "In the file header: exposure 20 s; Monitor = 5000"
        items = [page.monitor_combo.itemText(index) for index in range(page.monitor_combo.count())]
        assert items == ["Off", "Typed", "Header: Monitor = 5000"]
        page.exposure_combo.setCurrentIndex(page.exposure_combo.findData("header"))
        page.exposure_combo.activated.emit(page.exposure_combo.currentIndex())
        page.monitor_combo.setCurrentIndex(page.monitor_combo.findData("header:Monitor"))
        page.monitor_combo.activated.emit(page.monitor_combo.currentIndex())
        page.transmission_check.setChecked(True)
        page.transmission_spin.setValue(0.5)
        page._options_timer.stop()
        page.run_analysis()
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.normalisation.factor == np.float64(1.0 / (20.0 * 5000.0 * 0.5))
        assert page.normalisation_status.text() == (
            "Multiplied by 2e-05: t = 20 s (file header), M = 5000 (file header), T = 0.5 (typed); "
            "intensity in counts/(pixel·s·monitor)")
        radial, before = analysis.reduction.curve("horizontal"), plain.reduction.curve("horizontal")
        np.testing.assert_allclose(radial.intensity, before.intensity * 2e-5, rtol=1e-5)
        assert radial.y_label == "I (counts/(pixel·s·monitor))" and "normalised" in page.settings_summary()
        apply_language("zh", [page])
        page._show_normalisation()
        assert page.header_values_label.text() == "文件头中：曝光 20 s; Monitor = 5000"
        assert page.normalisation_status.text().startswith("已乘以 2e-05：t = 20 s (文件头)")
        assert page.normalisation_section.toggle_button.text() == "归一化"
        apply_language("en", [page])
        # A typed value that is not a number is refused with the reason, and nothing changes.
        page.absolute_check.setChecked(True)
        page.absolute_edit.setText("")
        page.absolute_edit.editingFinished.emit()
        assert page.status_text() == "Normalisation not changed: The absolute-scale factor must be a positive number."
    finally:
        apply_language("en")
        _close(page)


def test_the_dark_frame_and_the_lorentz_factor_are_set_from_their_controls(tmp_path: Path, monkeypatch) -> None:
    apply_language("en")
    paths = _frames(tmp_path, 1)
    dark = tmp_path / "dark.tif"
    from tests.test_assistant_calibration import save_tiff

    save_tiff(dark, np.full(SHAPE, 3.0))
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(paths[0])])
        _done(page)
        assert page.dark_label.text() == "No dark frame" and not page.lorentz_check.isChecked()
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *_args, **_kw: (str(dark), "")))
        page.dark_button.click()
        page.lorentz_check.setChecked(True)
        page.lorentz_band_spin.setValue(4.5)
        page.orientation_exit_spin.setValue(0.35)
        page.orientation_polar_check.setChecked(False)
        page._options_timer.stop()
        page.run_analysis()
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.corrections.dark_path == str(dark) and analysis.corrections.lorentz
        assert analysis.corrections.lorentz_band_px == 4.5 and analysis.corrections.orientation_min_exit_deg == 0.35
        assert not analysis.corrections.orientation_polarization
        np.testing.assert_allclose(np.nanmax(analysis.raw_data - 3.0 - np.where(analysis.valid, 0, np.nan)),
                                   np.nanmax(analysis.raw_data) - 3.0)
        assert page.dark_label.text() == "dark.tif" and "dark dark.tif" in page.settings_summary()
        page.dark_clear_button.click()
        assert page.view_model.state.corrections.dark_path is None
    finally:
        _close(page)


def test_every_new_text_has_its_chinese() -> None:
    sources = [ANALYZE / name for name in (
        "presentation/views/normalisation_view.py", "presentation/bindings/normalisation.py",
        "presentation/bindings/batch_export.py", "presentation/bindings/batch_watch.py",
        "presentation/views/analyze_page_view.py", "presentation/texts.py", "domain/normalisation.py",
        "presentation/setup_history.py", "presentation/bindings/workspace.py", "presentation/views/analyze_steps_view.py")]
    sources += [ANALYZE.parents[0] / name for name in (
        "assistant/presentation/guided_results.py", "assistant/domain/orientation.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for english, chinese in NORMALISATION_ZH.items():
        assert ZH[english] == chinese, english
        assert english.split("{")[0][:28] in text, english


# -- the display -----------------------------------------------------------------------------------------------------


def test_log_display_draws_values_at_or_below_zero_at_the_lowest_colour_and_levels_from_positive_values() -> None:
    rng = np.random.default_rng(0)
    frame = rng.normal(20.0, 15.0, (300, 300)).astype(np.float32)  # a dark-subtracted frame: many values ≤ 0
    valid = np.ones(frame.shape, dtype=bool)
    valid[:, :5] = False  # a gap
    shown, sample = display_image(frame, valid, True)
    non_positive = valid & (frame <= 0)
    assert non_positive.mean() > 0.05
    assert np.isnan(shown[~valid]).all()  # masked: no colour
    floor = np.log10(np.percentile(frame[valid & (frame > 0)], 0.1))
    np.testing.assert_allclose(shown[non_positive], floor, rtol=1e-3)  # not a hole: the lowest colour
    assert np.isfinite(shown[valid]).all()
    assert np.isnan(sample[non_positive[::1, ::1]]).all()  # the limits come from the positive values only
    linear, linear_sample = display_image(frame, valid, False)
    np.testing.assert_array_equal(linear[valid], frame[valid]) and np.isnan(linear[~valid]).all()


def test_the_detector_view_takes_its_limits_from_the_positive_values_and_frames_the_bar_around_them() -> None:
    from src.gimap.app.presentation.components.detector_view import DetectorView

    QApplication.instance() or QApplication([])
    view = DetectorView()
    try:
        frame = np.random.default_rng(1).normal(20.0, 15.0, (200, 200)).astype(np.float32)
        frame[100, 100] = 1.0e6  # one bright pixel
        view.set_image(frame)
        low, high = view.color_bar.levels()
        positive = frame[frame > 0]
        assert 10 ** low == np.float32(np.percentile(positive, 1.0)) or abs(10 ** low / np.percentile(positive, 1.0) - 1) < 0.05
        view.log_check.setChecked(False)  # linear: percentile limits of every value, and a bar around them
        low, high = view.color_bar.levels()
        assert abs(low - np.percentile(frame, 1.0)) < 2.0 and abs(high - np.percentile(frame, 99.7)) < 2.0
        x, _counts = view.color_bar.plot.getData()
        assert x.max() < 5 * high  # not stretched to the one bright pixel
    finally:
        view.dispose()


# -- the panels, the Results, Export Curves… -------------------------------------------------------------------------


def test_the_image_is_the_widest_panel_and_the_side_panels_can_be_hidden() -> None:
    page = _page(_context())
    try:
        page.resize(1600, 900)
        page.show()
        for _ in range(3):
            QApplication.processEvents()
        steps, image, curves = page.splitter.sizes()
        assert image > curves > 0 and image > steps
        page.show_curves_action.setChecked(False)
        QApplication.processEvents()
        assert page.splitter.widget(2).isHidden() and page.splitter.sizes()[1] > image
        page.show_steps_action.setChecked(False)
        assert page.process_panel.isHidden()
        page.show_steps_action.setChecked(True)
        page.show_curves_action.setChecked(True)
        for _ in range(3):
            QApplication.processEvents()
        assert not page.process_panel.isHidden() and not page.splitter.widget(2).isHidden()
    finally:
        _close(page)


def test_find_calibration_keeps_the_tab_the_person_chose(tmp_path: Path) -> None:
    paths = _frames(tmp_path, 1)
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(paths[0])])
        _done(page)
        found = []
        page.set_automatic_analysis(run=lambda: found.append("run"), find_geometry=lambda: found.append("geometry"))
        page.show_right("curves")
        page._find_geometry_clicked()
        page.automatic_started("Looking for a calibration …")
        assert found == ["geometry"] and page.current_right() == "curves"
        page.automatic_finished("ok", "", show_results=False)
        assert page.current_right() == "curves"
        page._run_pipeline_clicked()  # the automatic analysis: its progress is at the top of Results
        page.automatic_started("Running …")
        assert page.current_right() == "results"
        page.automatic_finished("ok", "")
    finally:
        _close(page)


def test_the_results_geometry_says_the_incidence_angle_in_use() -> None:
    from src.gimap.features.assistant.presentation.guided_results import GuidedResultsPanel

    QApplication.instance() or QApplication([])
    status = {"path": "C:/data/film.tif", "geometry": {"incidence_deg": 0.0}}
    panel = GuidedResultsPanel(lambda: SimpleNamespace(status=lambda: status))
    try:
        report = {"procedure": "geometry", "ok": True, "frame": "C:/data/film.tif",
                  "geometry": {"distance_mm": 1730.0, "beam_center_px": [597.1, 719.6], "wavelength_A": 1.34,
                               "incidence_deg": 0.0}}
        panel.show_report(report)
        assert panel._incidence_label.text() == "0°"
        status["geometry"]["incidence_deg"] = 0.463  # typed in the command bar after the run
        panel.refresh_incidence()
        assert panel._incidence_label.text() == "0.463° (the run used 0°)"
        status["path"] = "C:/data/other.tif"  # another frame shown: the run's value
        panel.refresh_incidence()
        assert panel._incidence_label.text() == "0°"
    finally:
        panel.deleteLater()


def test_export_curves_asks_for_its_folder_and_starts_in_the_last_one(tmp_path: Path, monkeypatch) -> None:
    paths = _frames(tmp_path / "raw", 1)
    out = tmp_path / "results"
    out.mkdir()
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(paths[0])])
        _done(page)
        asked = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda _parent, title, start, *_a: (asked.append((title, start)), str(out))[1]))
        page.export_curves_button.click()
        assert asked == [("Export Curves: choose a folder", str(paths[0].parent / "gimap_analysis"))]
        assert (out / f"{paths[0].stem}_horizontal.csv").is_file() and not (paths[0].parent / "gimap_analysis").exists()
        assert page.status_text().endswith(str(out))
        page.export_curves_button.click()
        assert asked[-1][1] == str(out)  # the folder chosen last
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *_a: ""))
        assert page.export_curves() is None  # cancelled: nothing written, nothing said wrong
    finally:
        _close(page)
