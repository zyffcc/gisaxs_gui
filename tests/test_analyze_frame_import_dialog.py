"""Frame choices are made from metadata, preserve source indices, and never decode images."""
from pathlib import Path

import h5py
import pytest
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QLabel

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application.frame_import import ImportSource
from src.gimap.features.analyze.presentation.frame_import_dialog import FrameImportDialog
from src.gimap.features.analyze.presentation.frame_selection import FrameSelection, IndexRanges
from tests.test_analyze_frame_import import _sparse
from tests.test_analyze_workspace import _app, _context, _done, _page


@pytest.fixture(autouse=True)
def _english():
    _app()
    apply_language("en")
    yield
    apply_language("en")


def _dialog(tmp_path, *, selections=None, counts=(1_000_000,)):
    sources = [ImportSource(tmp_path / f"source_{index}.nxs", count) for index, count in enumerate(counts)]
    return FrameImportDialog(sources, selections=selections), sources


def _mode(dialog, key):
    dialog.import_mode_combo.setCurrentIndex(dialog.import_mode_combo.findData(key))


def _ok(dialog):
    return dialog.import_buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_all_million_frame_choices_need_no_image_or_file_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The choice dialog must not reopen detector files")

    monkeypatch.setattr(h5py, "File", forbidden)
    dialog, sources = _dialog(tmp_path)
    try:
        assert dialog.import_mode_combo.currentData() == "all" and _ok(dialog)
        assert dialog.selections()[str(sources[0].path)] is None
        assert "1000000" in dialog.import_summary_label.text()
        assert not dialog.import_numbers_edit.isEnabled()
    finally:
        dialog.close()


def test_range_and_specific_frames_are_zero_based_in_the_returned_choice(tmp_path):
    dialog, sources = _dialog(tmp_path)
    key = str(sources[0].path)
    try:
        _mode(dialog, "range")
        dialog.import_first_spin.setValue(2)
        dialog.import_last_spin.setValue(11)
        dialog.import_step_spin.setValue(3)
        assert _ok(dialog) and list(dialog.selections()[key].indices(1_000_000)) == [1, 4, 7, 10]
        _mode(dialog, "specific")
        dialog.import_numbers_edit.setText("1000000，1, 500000, 500000")
        assert _ok(dialog)
        assert list(dialog.selections()[key].indices(1_000_000)) == [0, 499_999, 999_999]
    finally:
        dialog.close()


def test_each_source_keeps_its_own_choice_when_switching_the_source_combo(tmp_path):
    dialog, sources = _dialog(tmp_path, counts=(12, 20))
    try:
        _mode(dialog, "range")
        dialog.import_first_spin.setValue(2)
        dialog.import_last_spin.setValue(10)
        dialog.import_step_spin.setValue(3)
        dialog.source_combo.setCurrentIndex(1)
        assert dialog.import_mode_combo.currentData() == "all"
        _mode(dialog, "specific")
        dialog.import_numbers_edit.setText("1, 5")
        dialog.source_combo.setCurrentIndex(0)
        assert dialog.import_mode_combo.currentData() == "range"
        assert dialog.import_first_spin.value() == 2 and dialog.import_step_spin.value() == 3
        choices = dialog.selections()
        assert list(choices[str(sources[0].path)].indices(12)) == [1, 4, 7]
        assert list(choices[str(sources[1].path)].indices(20)) == [0, 4]
        assert dialog.import_summary_label.text() == "5 frames will be imported from 2 sources."
    finally:
        dialog.close()


def test_keep_current_selection_preserves_large_disjoint_ranges_without_enumerating_them(tmp_path, monkeypatch):
    selection = FrameSelection((range(0, 400_000, 2), range(600_000, 1_000_000, 3)))
    path = tmp_path / "source_0.nxs"

    def no_enumeration(self):
        raise AssertionError("A huge existing selection must remain compressed in the dialog")

    monkeypatch.setattr(IndexRanges, "__iter__", no_enumeration)
    dialog, sources = _dialog(tmp_path, selections={str(path): selection})
    try:
        assert dialog.import_mode_combo.currentData() == "current" and _ok(dialog)
        assert dialog.import_numbers_edit.text() == ""
        assert dialog.selections()[str(sources[0].path)] == selection
        assert "333334" in dialog.import_summary_label.text()
        apply_language("zh", [dialog])
        assert dialog.selections()[str(sources[0].path)] == selection
        assert _ok(dialog)
    finally:
        dialog.close()


@pytest.mark.parametrize("text", ["", "0", "13", "1,,5", "1, x", "-1"])
def test_invalid_specific_frame_numbers_disable_import_instead_of_silently_changing_scope(tmp_path, text):
    dialog, _sources = _dialog(tmp_path, counts=(12,))
    try:
        _mode(dialog, "specific")
        dialog.import_numbers_edit.setText(text)
        assert not _ok(dialog) and dialog.import_error_label.text()
    finally:
        dialog.close()


def test_an_inverted_range_is_refused(tmp_path):
    dialog, _sources = _dialog(tmp_path, counts=(12,))
    try:
        _mode(dialog, "range")
        dialog.import_first_spin.setValue(8)
        dialog.import_last_spin.setValue(2)
        assert not _ok(dialog)
        assert dialog.import_error_label.text() == "The first frame must not be after the last frame."
    finally:
        dialog.close()


def test_cancelling_the_interactive_choice_imports_no_rows_and_reads_no_detector_pixels(tmp_path, monkeypatch):
    path = _sparse(tmp_path / "cancelled.nxs")
    reads = []
    getitem = h5py.Dataset.__getitem__

    def recorded(dataset, selection):
        if dataset.name == "/entry/instrument/detector/data":
            reads.append(selection)
        return getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recorded)
    monkeypatch.setattr(FrameImportDialog, "exec_", lambda self: QDialog.Rejected)
    page = _page(_context())
    try:
        page.import_paths([path], interactive=True)
        _done(page)
        assert page.frame_list.model().rowCount() == 0 and page.view_model.state.files == []
        assert reads == [] and page.status_text() == "Frame import cancelled."
    finally:
        page.dispose()
        page.close()


def test_fresh_chinese_dialog_translates_import_labels_and_preserves_frame_choices(tmp_path):
    apply_language("zh")
    dialog, sources = _dialog(tmp_path, selections={str(tmp_path / "source_0.nxs"): FrameSelection.explicit((0, 9))})
    try:
        apply_language("zh", [dialog])
        assert dialog.windowTitle() == "导入帧"
        assert dialog.import_buttons.button(QDialogButtonBox.Ok).text() == "导入"
        assert not any(label.text() == "Import" for label in dialog.findChildren(QLabel))
        assert "指定帧" == dialog.import_mode_combo.currentText()
        assert list(dialog.selections()[str(sources[0].path)].indices(1_000_000)) == [0, 9]
    finally:
        dialog.close()


def test_an_invalid_explicit_import_keeps_the_existing_list_and_shown_frame(tmp_path):
    existing = _sparse(tmp_path / "existing.nxs", count=3)
    rejected = _sparse(tmp_path / "rejected.nxs", count=3)
    page = _page(_context())
    try:
        page.import_paths([existing], interactive=False,
                          selections={str(existing): FrameSelection.explicit((1,))})
        _done(page)
        before = page.view_model.state.analysis
        page.import_paths([rejected], interactive=False,
                          selections={str(rejected): FrameSelection.explicit((9,))})
        _done(page)
        assert page.view_model.state.files == [existing]
        assert page.frame_list.model().rowCount() == 1 and page.frame_list.model().ref(0) == (existing, 1)
        assert page.view_model.state.analysis is before
        assert "within this source" in page.status_text()
    finally:
        page.dispose()
        page.close()


@pytest.mark.parametrize("accepted", [False, True])
def test_batch_from_folder_opens_only_after_the_user_accepts_the_imported_subset(tmp_path, monkeypatch, accepted):
    from tests.test_analyze_nexus_sources import _ring_stack
    from tests.test_review_analyze_run_and_clear import _profiles

    path = _ring_stack(tmp_path / "raw" / "rings.nxs", 5).resolve()
    page = _page(_context(_profiles()))
    opened = []

    def choose(dialog):
        _mode(dialog, "range")
        dialog.import_first_spin.setValue(2)
        dialog.import_last_spin.setValue(4)
        dialog.import_step_spin.setValue(2)
        return QDialog.Accepted if accepted else QDialog.Rejected

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(path.parent))
    monkeypatch.setattr(FrameImportDialog, "exec_", choose)
    monkeypatch.setattr(page, "batch_export_dialog", lambda: opened.append(
        [request.frame_index for request in page.view_model.batch_requests()]))
    try:
        page.set_mode_choice("giwaxs")
        page.batch_from_folder()
        _done(page)
        if accepted:
            assert opened == [[1, 3]] and page.frame_list.model().rowCount() == 2
        else:
            assert opened == [] and page.frame_list.model().rowCount() == 0
            # A later ordinary open cannot accidentally trigger the cancelled Batch Export.
            page.import_paths([path], interactive=False)
            _done(page)
            assert opened == []
        assert not page._batch_pending
    finally:
        page.dispose()
        page.close()
