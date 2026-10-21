"""The remaining hand-over contracts after Analyze's source and Series-map work."""

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QMessageBox

from src.gimap.app.handover import HandoverMixin
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.presentation.bindings.series_exports import SeriesExportsMixin
from src.gimap.features.assistant.application.gisaxs_procedure import GisaxsProcedureMixin
from src.gimap.features.assistant.application.gisaxs_report import gisaxs_sections
from src.gimap.features.assistant.presentation.guided_words import words
from src.gimap.features.compare.presentation.render import CompareRenderMixin


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


@pytest.mark.parametrize("label, en, zh", [
    ("stack.h5 frame 4 of 12", "frame 4", "第 4 帧"),
    ("stack.h5 frames 6–10 of 12 (+4 summed)", "frames 6–10", "第 6–10 帧"),
    ("stack.h5 #4", "frame 4", "第 4 帧"),
    ("film_0003.tif", "film_0003", "film_0003"),
])
def test_compare_keeps_the_frame_number_of_the_new_series_labels(label, en, zh):
    page = SimpleNamespace(_compared=[SimpleNamespace(labels=(label,))])
    assert CompareRenderMixin._label(page, 0, 0) == en
    apply_language("zh")
    assert CompareRenderMixin._label(page, 0, 0) == zh


def test_assistant_reports_a_held_cut_as_the_cut_of_the_series():
    cuts = {"horizontal_source": "series", "horizontal_rows": (20.0, 25.0), "vertical_columns": (100.0, 104.0),
            "series_cut": {"frame": ("first.tif", 0), "horizontal_rows": (20.0, 25.0)}}
    status = {"shape": (200, 300)}
    report = gisaxs_sections({"gisaxs": {"cuts": cuts}, "tables": {"status": status}})
    assert "at the cut of the series" in report[2] and "rows 20–24" in report[2]
    assert "set by hand" not in report[2]
    decisions = []
    procedure = SimpleNamespace(catalog=SimpleNamespace(results=SimpleNamespace(status=status)),
                                _decide=lambda *args: decisions.append(args))
    GisaxsProcedureMixin._horizontal_cut(procedure, cuts)
    assert decisions == [("horizontal cut", "cut of the series (rows 20–24)",
                          "the cut of one frame, held for every frame")]
    apply_language("zh")
    assert words(decisions[0][1]) == "序列的切线（第 20–24 行）"
    assert words(decisions[0][2]) == "某一帧的切线，用于每一帧"


class _MapExport(SeriesExportsMixin):
    def __init__(self, values, axis="frame"):
        self._series_map = SimpleNamespace(rows=len(values), view="both", x_label="qy (Å⁻¹)")
        self.values, self.axis, self.saved = np.asarray(values), axis, {}
        self.data = np.arange(len(values) * 4.0).reshape(len(values), 4)
        self.series_map_view = SimpleNamespace(display_state=lambda: {
            "image": self.data, "extent": (0.01, 0.04, len(values) + 0.5, 0.5),
            "title": "Horizontal cut", "origin_upper": True,
        })
        self.view_model = SimpleNamespace(save_image_figure=self.save)

    def _series_axis_values(self, series):
        return self.values

    def _series_axis_label(self, series):
        return self.axis

    def save(self, target, image, **state):
        self.saved.update(state, image=image)
        return target

    def _series_saved(self, write, path, failed):
        return write(path)


@pytest.mark.parametrize("values, axis, labels", [
    ([4, 7, 11], "frame", ["4", "7", "11"]),
    ([0.0, 0.125, 0.4], "time (s)", ["0", "0.125", "0.4"]),
])
def test_a_saved_irregular_map_names_the_real_frame_or_time_of_each_row(tmp_path, values, axis, labels):
    page = _MapExport(values, axis)
    assert page.export_series_figure(tmp_path / "map.png") == tmp_path / "map.png"
    assert page.saved["extent"] == (0.01, 0.04, 3.5, 0.5)
    assert page.saved["y_label"] == axis
    assert page.saved["y_ticks"] == list(zip((1.0, 2.0, 3.0), labels))
    np.testing.assert_array_equal(page.saved["image"], page.data)


def test_irregular_map_ticks_stay_readable_for_a_long_series(tmp_path):
    values = np.arange(240, dtype=float) ** 2 + 1
    page = _MapExport(values)
    page.export_series_figure(tmp_path / "map.png")
    ticks = page.saved["y_ticks"]
    assert 2 <= len(ticks) <= 6
    assert ticks[0] == (1.0, "1") and ticks[-1] == (240.0, "57122")
    for row, label in ticks:
        assert float(label) == values[int(row) - 1]


@pytest.mark.parametrize("accepted", [False, True])
def test_series_handover_reports_the_receivers_outcome(tmp_path, monkeypatch, accepted):
    from src.gimap.app import handover

    calls, notices = [], []
    window = HandoverMixin()
    window.ui = SimpleNamespace(runtime=SimpleNamespace(navigate=lambda key: calls.append(key)))
    window.fitting_workspace = SimpleNamespace(open_series=lambda folder, pattern: accepted)
    monkeypatch.setattr(handover, "show_toast", lambda host, text, **options: notices.append((text, options)))
    assert window._send_series_to_fitting(tmp_path) is accepted
    assert calls == ["fitting"]
    if accepted:
        assert not notices
    else:
        assert len(notices) == 1
        assert "Series not opened in Fitting" in notices[0][0]
        assert notices[0][1]["level"] == "warning" and notices[0][1]["action"][0] == "Open Folder"


def test_cancelling_series_handover_keeps_unsaved_results_and_says_where_the_exports_are(tmp_path, monkeypatch):
    from tests.test_handover import _fit_input, _settle, _window
    from src.gimap.app.presentation.components import visible_toasts

    window = _window()
    try:
        components = window.components
        page = components.fitting_workspace.series_page
        old, new = tmp_path / "old" / "fit_input", tmp_path / "new" / "fit_input"
        for folder in (old, new):
            for frame in range(3):
                _fit_input(folder / f"cut_{frame:04d}_fit_input.dat")
        assert page.open_series(old)
        retained = {0: SimpleNamespace(fitted=True, ok=True)}
        page.fits, page._unsaved = retained, True
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_args: QMessageBox.Cancel))
        components.show_page("analyze")
        assert components._send_series_to_fitting(new) is False
        assert page.folder == str(old) and page.fits is retained and page._unsaved
        assert any("Series not opened in Fitting" in toast.text() for toast in visible_toasts(window))
    finally:
        page._unsaved = False
        window.close()
        _settle()


@pytest.mark.parametrize("profile", ["pseudo_voigt", "gaussian", "lorentzian"])
@pytest.mark.parametrize("for_fitting", [False, True])
def test_reopening_batch_with_remembered_formats_initializes_before_slots_run(tmp_path, monkeypatch, profile, for_fitting):
    """A first export remembers HDF5 frames; opening its next dialog used to call _sync without folder state."""
    import sys
    from src.gimap.features.analyze.application import BatchChoices, SeriesCorrection
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, traceback: errors.append((kind, str(error))))
    common = dict(frames=12, files=1, stem="stack", settings_text="GIWAXS", destination=tmp_path,
                  subfolder=False, series=SeriesCorrection(), curve_options=[("radial", "I(q)")],
                  fit_targets=("Ring A",), fit_keys=("region1",), screen_scale="0–100", beside=tmp_path,
                  cut_text=lambda: "I(χ) ring: q 1–2 Å⁻¹ for every frame", pin_cuts=True,
                  set_pin_cuts=lambda _value: None)
    first = second = None
    try:
        first = BatchExportDialog(choices=BatchChoices(), **common)
        first._set_choices(BatchChoices(fit="peaks", fit_profile=profile, fit_curves=True, frames=True,
                                       frame_format="hdf5", text_format="dat", image_format="pdf", every=2))
        remembered = first.choices()
        assert not errors and remembered.frame_format == "hdf5" and remembered.fit_profile == profile
        second = BatchExportDialog(choices=remembered, for_fitting=for_fitting, **common)
        assert not errors, errors
        assert second.frame_format_combo.currentData() == "hdf5"
        assert second.text_format_combo.currentData() == "dat" and second.image_format_combo.currentData() == "pdf"
        assert second.fit_profile_combo.currentData() == profile
        assert second.frame_count() == 6 and second.export_button.isEnabled()
        assert second.target() == tmp_path
        if not for_fitting:
            assert second.choices() == remembered
        else:
            assert second.choices().fit_input
    finally:
        for dialog in (first, second):
            if dialog is not None:
                dialog.done(0)
                dialog.deleteLater()


def test_watch_does_not_mix_uncorrected_rows_into_a_corrected_batch_map(tmp_path):
    """Series corrections belong to the batch; ordinary watch exports must not contaminate its map."""
    from src.gimap.features.analyze.application import BatchChoices, SeriesCorrection
    from tests.test_analyze_workspace import _context, _done, _page
    from tests.test_analyze_nexus_sources import _ring_stack, _grow_rings
    from tests.test_review_analyze_run_and_clear import _profiles
    from tests.test_finish_live_map import _look
    from tests.test_live_following import _close

    path = _ring_stack(tmp_path / "data" / "live.h5", 4)
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path)])
        _done(page)
        correction = SeriesCorrection(reference_q=1.1, half_width=0.03, normalize=True,
                                      target_intensity=17.0, per_frame=True)
        page.run_batch(tmp_path / "batch", BatchChoices(tables=True), series=correction, live_key="radial")
        _done(page)
        assert page._series_map.rows == 4 and not page._series_run.series.is_identity
        before = np.array(page._series_map.image, copy=True)
        page.start_watch(path.parent)
        page._watch_timer.stop()
        page.auto_export_check.setChecked(True)
        _grow_rings(path, 6)
        _look(page)
        assert page._series_map.rows == 4
        np.testing.assert_array_equal(page._series_map.image, before)
        assert len(list((path.parent / "gimap_analysis").glob("*_radial.csv"))) == 2
        assert "batch series corrections" in page.series_info_label.text()
        assert "Build Map again" in page.series_info_label.text()
    finally:
        page.stop_watch()
        _close(page)
