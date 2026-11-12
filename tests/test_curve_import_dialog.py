"""Explicit column choices and cancellable, nonblocking table previews use only synthetic tables."""

from pathlib import Path
from threading import Event, get_ident
import gc
import time

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.app.presentation.theme import apply_theme
from src.gimap.features.fitting.application.curve_sources import CurveTablePreview
from src.gimap.features.fitting.presentation.curve_import import CurveImportDialog


def _preview(path, names=("q", "I", "sigma"), *, layout="single", header=True, analyze=False, unit=None,
             uncertainty=(), intensities=()):
    rows = tuple(tuple(str(row + column + 1) for column in range(len(names))) for row in range(12))
    return CurveTablePreview(Path(path), names, rows, len(names), layout, analyze, unit, header, uncertainty, intensities)


def _until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


@pytest.fixture
def dialogs(qt_application):
    tasks = TaskRunner(max_threads=1)
    created = []

    def build(paths, inspect, **kwargs):
        dialog = CurveImportDialog(paths, inspect, task_runner=tasks, **kwargs)
        created.append(dialog)
        return dialog

    yield build
    for dialog in created:
        if not sip.isdeleted(dialog):
            dialog.dispose()
            dialog.close()
            dialog.deleteLater()
    assert tasks.wait(10)
    tasks.shutdown()
    tasks.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    apply_language("en")
    apply_theme("light", 9)


@pytest.mark.parametrize("names,layout,header,analyze,count,sigma", [
    (("q", "I", "sigma"), "single", True, False, 1, 2),
    (("q", "I1", "I2"), "shared_q", True, False, 2, None),
    (("Column 1", "Column 2", "Column 3"), "ambiguous", False, False, 1, 2),
    (("q", "I", "sigma", "pixels"), "single", True, True, 1, 2),
])
def test_preview_defaults_roles_and_explicit_import_plan(dialogs, tmp_path, names, layout, header, analyze, count, sigma):
    path = tmp_path / "synthetic.dat"
    preview = _preview(path, names, layout=layout, header=header, analyze=analyze)
    dialog = dialogs([path], lambda _path: preview)
    _until(lambda: dialog._complete)
    assert dialog.preview_table.rowCount() == 10
    assert dialog.preview_table.item(0, 0).text() == "1"
    assert dialog.import_button.text() == ("Import 1 curve" if count == 1 else f"Import {count} curves")
    assert dialog.preview_table.horizontalHeaderItem(0).text().endswith("Role: q")
    if layout == "ambiguous":
        assert "column 3 as σ" in dialog.layout_note.text()
        assert "values are not used to guess" in dialog.layout_note.text()
    if analyze:
        assert "pixel counts" in dialog.layout_note.text()
        assert not dialog.layout_combo.model().item(1).isEnabled()
        # Even programmatic selection cannot override the scientific protection.
        dialog.layout_combo.setCurrentIndex(1)
        assert dialog.layout_combo.currentData() == "single"
        assert dialog.preview_table.horizontalHeaderItem(3).text().endswith("Role: Ignored")
    dialog.accept()
    plan = dialog.import_plan
    assert plan is not None and plan.independent
    assert len(plan.sources) == count
    assert all(source.path == path for source in plan.sources)
    assert all(source.columns.sigma == sigma for source in plan.sources)
    assert [source.columns.intensity for source in plan.sources] == list(range(1, count + 1))


def test_apply_all_preserves_file_and_column_order_and_explicit_time_mode(dialogs, tmp_path):
    paths = [tmp_path / "second.dat", tmp_path / "first.dat"]
    dialog = dialogs(paths, lambda path: _preview(path, ("Column 1", "Column 2", "Column 3"),
                                                 layout="ambiguous", header=False), q_unit="nm")
    _until(lambda: dialog._complete)
    assert dialog.apply_all_button.isEnabled()
    dialog.layout_combo.setCurrentIndex(1)
    assert dialog.import_button.text() == "Import 3 curves"
    dialog.apply_all_button.click()
    assert dialog.import_button.text() == "Import 4 curves"
    dialog.mode_combo.setCurrentIndex(1)
    dialog.accept()
    plan = dialog.import_plan
    assert plan is not None and not plan.independent and plan.q_unit == "nm^-1"
    assert [(source.path, source.columns.intensity) for source in plan.sources] == [
        (paths[0], 1), (paths[0], 2), (paths[1], 1), (paths[1], 2)]
    assert all(source.columns.sigma is None for source in plan.sources)


@pytest.mark.parametrize("names,uncertainty", [(("q", "I", "sigma"), (2,)),
                                              (("q", "I1", "sigma1", "I2", "sigma2"), (2, 4))])
def test_named_uncertainties_cannot_be_selected_as_separate_intensity_curves(dialogs, tmp_path, names, uncertainty):
    path = tmp_path / "with_errors.dat"
    dialog = dialogs([path], lambda item: _preview(item, names, uncertainty=uncertainty))
    _until(lambda: dialog._complete)
    assert "separate q, I, σ files" in dialog.layout_note.text()
    assert not dialog.layout_combo.model().item(1).isEnabled()
    dialog.layout_combo.setCurrentIndex(1)
    assert dialog.layout_combo.currentData() == "single"
    dialog.accept()
    assert len(dialog.import_plan.sources) == 1
    assert dialog.import_plan.sources[0].columns.sigma == 2


def test_misplaced_named_uncertainty_is_visible_and_cannot_be_imported(dialogs, tmp_path):
    path = tmp_path / "misordered.dat"
    dialog = dialogs([path], lambda item: _preview(item, ("q", "I1", "I2", "sigma"), uncertainty=(3,)))
    _until(lambda: dialog._complete)
    assert "named uncertainty column is elsewhere" in dialog.error_details.toPlainText()
    assert "Could not import this table" in dialog.layout_note.text()
    assert dialog.inspect_job.message_label.text() == "Checked 1 of 1 files"
    assert dialog.preview_table.horizontalHeaderItem(3).text().startswith("sigma")
    assert not dialog.import_button.isEnabled() and not dialog.layout_combo.isEnabled()
    dialog.accept()
    assert dialog.import_plan is None


def test_confirmation_validates_every_source_before_publishing_plan(dialogs, tmp_path, monkeypatch):
    from src.gimap.features.fitting.presentation import curve_import

    path = tmp_path / "synthetic.dat"
    dialog = dialogs([path], _preview)
    _until(lambda: dialog._complete)

    def invalid(*_args):
        raise ValueError("A selected curve column is missing from the table.")

    monkeypatch.setattr(curve_import, "sources_from_table", invalid)
    dialog.accept()
    assert dialog.import_plan is None
    assert "A selected curve column is missing" in dialog.error_details.toPlainText()
    assert not dialog.import_button.isEnabled()


def test_mixed_columns_keep_per_file_choices_and_header_units_take_priority(dialogs, tmp_path):
    paths = [tmp_path / "with_error.dat", tmp_path / "multiple.dat"]
    previews = {paths[0]: _preview(paths[0], unit="angstrom"),
                paths[1]: _preview(paths[1], ("q", "I1", "I2"), layout="shared_q", unit="nm", intensities=(1, 2))}
    dialog = dialogs(paths, previews.__getitem__)
    _until(lambda: dialog._complete)
    assert not dialog.apply_all_button.isEnabled()
    assert not dialog.q_unit_combo.isEnabled()
    assert "Å⁻¹" in dialog.unit_note.text()
    assert "different columns" in dialog.apply_all_button.toolTip()
    dialog.file_list.setCurrentRow(1)
    assert dialog.layout_combo.currentData() == "shared_q"
    assert "nm⁻¹" in dialog.unit_note.text()
    dialog.layout_combo.setCurrentIndex(0)
    assert dialog.layout_combo.currentData() == "shared_q"
    assert "Layout not changed" in dialog.layout_note.text()
    assert "cannot be used as sigma" in dialog.layout_note.text()
    assert not dialog._errors
    assert dialog.import_button.isEnabled()
    dialog.file_list.setCurrentRow(0)
    assert dialog.layout_combo.currentData() == "single"
    assert dialog.import_button.text() == "Import 3 curves"
    dialog.accept()
    assert len(dialog.import_plan.sources) == 3
    assert dialog.import_plan.sources[0].columns.sigma == 2
    assert all(source.columns.sigma is None for source in dialog.import_plan.sources[1:])


def test_inspection_is_off_gui_thread_and_cancel_discards_pending_files(dialogs, tmp_path):
    entered, release = Event(), Event()
    paths = [tmp_path / "slow.dat", tmp_path / "queued.dat"]
    calls, threads, heartbeats = [], [], []
    main_thread = get_ident()

    def inspect(path):
        calls.append(path)
        threads.append(get_ident())
        entered.set()
        assert release.wait(10)
        return _preview(path)

    dialog = dialogs(paths, inspect)
    heartbeat = QTimer(dialog)
    heartbeat.timeout.connect(lambda: heartbeats.append(True))
    heartbeat.start(1)
    try:
        _until(entered.is_set)
        _until(lambda: bool(heartbeats))
        assert threads == [threads[0]] and threads[0] != main_thread
        assert not dialog.import_button.isEnabled()
        dialog.inspect_job.cancelRequested.emit()
        assert dialog._lifetime.token.is_set()
        assert dialog.import_plan is None
    finally:
        release.set()
    assert dialog.tasks.wait(10)
    assert calls == paths[:1]
    assert dialog._previews == [None, None]


def test_parent_deletion_cancels_worker_without_late_widget_callback(dialogs, tmp_path):
    entered, release = Event(), Event()
    owner = QWidget()
    calls = []

    def inspect(path):
        calls.append(path)
        entered.set()
        assert release.wait(10)
        return _preview(path)

    dialog = dialogs([tmp_path / "active.dat", tmp_path / "queued.dat"], inspect, parent=owner)
    try:
        _until(entered.is_set)
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(dialog)
        assert dialog._lifetime.token.is_set()
        gc.collect()
    finally:
        release.set()
    assert dialog.tasks.wait(10)
    assert len(calls) == 1 and dialog._previews == [None, None]


def test_owned_runner_outlives_deleted_dialog_until_active_inspection_returns(tmp_path):
    entered, release = Event(), Event()
    owner = QWidget()

    def inspect(path):
        entered.set()
        assert release.wait(10)
        return _preview(path)

    dialog = CurveImportDialog([tmp_path / "active.dat"], inspect, parent=owner)
    tasks = dialog.tasks
    try:
        _until(entered.is_set)
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(dialog)
        assert not sip.isdeleted(tasks)
        assert dialog._lifetime.token.is_set()
    finally:
        release.set()
    _until(lambda: sip.isdeleted(tasks))
    assert dialog.import_plan is None


def test_any_file_error_is_visible_and_prevents_partial_import(dialogs, tmp_path):
    paths = [tmp_path / "bad.dat", tmp_path / "good.dat", tmp_path / "other_bad.dat"]
    calls = []

    def inspect(path):
        calls.append(path)
        if path.name != "good.dat":
            raise ValueError("A curve table needs at least two columns.")
        return _preview(path)

    dialog = dialogs(paths, inspect)
    _until(lambda: dialog._complete)
    assert calls == paths
    assert "bad.dat" in dialog.error_details.toPlainText()
    assert "other_bad.dat" in dialog.error_details.toPlainText()
    assert "2 files could not be checked" in dialog.summary_label.text()
    assert not dialog.import_button.isEnabled()
    dialog.accept()
    assert dialog.import_plan is None
    dialog.file_list.setCurrentRow(1)
    assert dialog.preview_table.rowCount() == 10
    assert "Nothing will be imported" in dialog.summary_label.text()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_live_language_and_theme_keep_column_values_and_selection(dialogs, tmp_path, theme):
    path = tmp_path / "synthetic.dat"
    dialog = dialogs([path], lambda item: _preview(item, layout="ambiguous", header=False))
    _until(lambda: dialog._complete)
    apply_theme(theme, 9)
    dialog.show()
    apply_language("zh")
    assert dialog.windowTitle() == "导入曲线表格"
    assert dialog.import_button.text() == "导入 1 条曲线"
    assert "用途：不确定度" in dialog.preview_table.horizontalHeaderItem(2).text()
    assert "默认把第 3 列作为 σ" in dialog.layout_note.text()
    assert dialog.layout_combo.currentData() == "single"
    assert dialog.preview_table.item(0, 0).text() == "1"
    apply_language("en")
    assert dialog.windowTitle() == "Import Curve Tables"
    assert "Role: Uncertainty" in dialog.preview_table.horizontalHeaderItem(2).text()
    assert dialog.import_button.text() == "Import 1 curve"
