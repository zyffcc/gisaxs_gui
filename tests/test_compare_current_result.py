"""Only results of the current series and settings may be shown or exported."""

import pytest

from tests.test_compare import _close, _idle, _map, _shown_page, _wait


def _ready():
    page = _shown_page()
    page.add_map(_map(1), "synthetic")
    _wait(page, lambda: _idle(page) and page.comparison is not None)
    return page


@pytest.mark.parametrize("setting", ["end", "shape", "range"])
def test_setting_change_immediately_blocks_table_and_plot_exports(setting, tmp_path):
    page = _ready()
    try:
        if setting == "end":
            page.end_spin.setValue(1)
        elif setting == "shape":
            page.shape_check.setChecked(False)
        else:
            page.q_low_spin.stepUp()
        # Before the debounce timer or worker has even started.
        assert not page.comparison_current()
        assert not page.save_button.isEnabled()
        assert page.step_rail.detail("results") == "Comparing …"
        assert page.save_series_table(tmp_path / "stale.csv") is None
        assert page.save_frames_table(tmp_path / "frames.csv") is None
        assert page._save_plot(page.end_plot, "end", tmp_path / "stale.png") is None
        assert list(tmp_path.iterdir()) == []
        _wait(page, lambda: _idle(page) and page.comparison_current())
        assert page.save_button.isEnabled()
        assert page.save_series_table(tmp_path / "current.csv").exists()
    finally:
        _close(page)


@pytest.mark.parametrize("failed", [False, True])
def test_completion_before_debounce_cannot_install_old_settings(failed, monkeypatch):
    page = _ready()
    pending = []
    try:
        previous = page.comparison
        monkeypatch.setattr(page.tasks, "submit", lambda key, fn, **callbacks: pending.append((fn, callbacks)))
        page.end_spin.setValue(2)
        page._timer.stop()
        page.recompute()
        fn, callbacks = pending.pop()
        result = fn()
        page.end_spin.setValue(3)  # debounce has not called recompute / set _again yet
        if failed:
            callbacks["on_error"]("obsolete failure", "")
        else:
            callbacks["on_done"](result)
        assert page.comparison is previous
        assert not page.comparison_current()
        assert not page.failure_reason()
        assert not page.save_button.isEnabled()
    finally:
        _close(page)


def test_returning_to_previous_settings_restores_the_exportable_plot():
    page = _ready()
    try:
        original = page.end_plot.figure_state()["curves"]
        assert original
        page.shape_check.setChecked(False)
        assert not page.comparison_current()
        page.shape_check.setChecked(True)
        assert page.comparison_current()
        assert page.save_button.isEnabled()
        assert page.plot_stack.currentWidget() is page.plots_page
        assert page.end_plot.figure_state()["curves"]
    finally:
        _close(page)


def test_invalid_range_cannot_export_the_last_valid_result(tmp_path):
    page = _ready()
    try:
        page.q_low_spin.setValue(page.q_high_spin.value())
        assert not page.comparison_current()
        assert not page.save_button.isEnabled()
        assert page.save_series_table(tmp_path / "stale.csv") is None
        _wait(page, lambda: _idle(page))
        assert page.comparison is None and page.failure_reason()
    finally:
        _close(page)
