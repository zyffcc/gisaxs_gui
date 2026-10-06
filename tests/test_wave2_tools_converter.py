"""Format Converter (wave 2): one thumbnail for one frame, statistics first, log + viridis display
with raw statistics; the destination next to the data; the remembered input folder; drops."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog

from src.gimap.app import AppContext
from src.gimap.features.format_converter.presentation.dialog import (
    FolderImportDialog,
    FormatConverterDialog,
)
from src.gimap.features.format_converter.presentation.display_formatting import (
    _array_pixmap,
    _viridis_lut,
    display_levels,
)
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIFF = PROJECT_ROOT / "tests/data/external/gisaxs_galaxi/galaxi_data.tif"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dispose(dialog) -> None:
    """Close the window and delete it now, by the event loop: a parentless window left to the
    garbage collector is deleted inside a collection, which can take the interpreter down."""
    for name in ("_preview_thread", "_conversion_thread", "_load_thread", "_cal_thread",
                 "_inspect_thread", "_extract_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _context(preferences=None) -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=preferences or InMemoryUserPreferencesRepository(),
    )


def _write_nxs(path: Path, frames: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.arange(frames * 8 * 6, dtype=np.uint16).reshape(frames, 8, 6)
        handle.create_dataset("/entry/instrument/detector/data", data=data)
    return path


def _visible(dialog, widgets) -> list[bool]:
    return [widget.isVisibleTo(dialog) for widget in widgets]


# -- preview ------------------------------------------------------------------------------------


def test_thumbnails_are_log_scaled_viridis_and_never_change_the_values():
    data = np.array([[0.0, 1.0, 10.0], [100.0, 1000.0, -2.0], [np.nan, 5.0, 50.0]], dtype=np.float32)
    original = data.copy()
    levels = display_levels(data)
    assert np.array_equal(data, original, equal_nan=True)
    # Negative and NaN pixels show as the lowest level; the order of the others is kept.
    assert levels[1, 2] == levels[2, 0] == levels[0, 0] == 0.0
    assert levels[0, 1] < levels[0, 2] < levels[1, 0] <= levels[1, 1]
    _app()
    image = _array_pixmap(np.tile(np.arange(64, dtype=np.float32), (48, 1)) ** 3, 64, 48).toImage()
    low, high = QColor(image.pixel(0, 24)), QColor(image.pixel(63, 24))
    lut = _viridis_lut()
    assert (low.red(), low.green(), low.blue()) == tuple(int(value) for value in lut[0])
    assert (high.red(), high.green(), high.blue()) == pytest.approx(tuple(int(v) for v in lut[255]), abs=8)


@pytest.mark.skipif(not TIFF.is_file(), reason="GALAXI TIFF not available")
def test_a_single_image_shows_one_thumbnail_below_the_statistics(tmp_path):
    _app()
    source_path = tmp_path / "data" / "galaxi_data.tif"
    source_path.parent.mkdir()
    shutil.copy(TIFF, source_path)
    dialog = FormatConverterDialog(app_context=_context())
    dialog.show()
    try:
        dialog.add_paths([str(source_path)])
        source = dialog.sources[0]
        payload = dialog.view_model.load_preview(source)
        assert len(payload) == 1  # read once, not three times
        dialog._preview_ready(dialog._preview_request, payload)

        assert _visible(dialog.stack, dialog.preview_labels) == [False, False, False]  # page 1 is shown
        dialog.stack.setCurrentIndex(1)
        assert _visible(dialog, dialog.preview_labels) == [True, False, False]
        assert _visible(dialog, dialog.preview_captions) == [True, False, False]
        assert dialog.preview_captions[0].text() == "Frame 1"
        assert not dialog.preview_labels[0].pixmap().isNull()
        layout = dialog.formatPreviewContentLayout
        assert layout.indexOf(dialog.preview_stats) < layout.indexOf(dialog.first_preview_caption)
        stats = dialog.preview_stats.text()
        raw = np.asarray(payload[0]["data"])
        finite = raw[np.isfinite(raw)]
        assert f"Min / max: {float(finite.min()):.6g} / {float(finite.max()):.6g}" in stats
        assert stats.startswith("Image size:")
    finally:
        _dispose(dialog)


def test_a_series_shows_first_and_last_once_each(tmp_path):
    _app()
    path = _write_nxs(tmp_path / "scan.nxs", frames=2)
    dialog = FormatConverterDialog(app_context=_context())
    dialog.show()
    try:
        dialog.add_paths([str(path)])
        dialog.stack.setCurrentIndex(1)
        payload = dialog.view_model.load_preview(dialog.sources[0])
        assert [item["label"] for item in payload] == ["First", "Last"]
        dialog._preview_ready(dialog._preview_request, payload)
        assert _visible(dialog, dialog.preview_labels) == [True, True, False]
        assert [caption.text() for caption in dialog.preview_captions[:2]] == [
            "First · frame 1",
            "Last · frame 2",
        ]
        assert dialog.preview_stats.text().startswith("Statistics of frame 1:")
        # Three frames or more: first, middle, last.
        dialog.sources[0].selected_frames = [0, 1, 1]
        assert len(dialog.view_model.load_preview(dialog.sources[0])) == 2
    finally:
        _dispose(dialog)


# -- destination and folders ---------------------------------------------------------------------


def test_the_destination_follows_the_first_input_until_the_user_sets_one(tmp_path):
    _app()
    first = _write_nxs(tmp_path / "beamtime" / "a" / "scan_a.nxs")
    second = _write_nxs(tmp_path / "beamtime" / "b" / "scan_b.nxs")
    dialog = FormatConverterDialog(app_context=_context())
    try:
        dialog.add_paths([str(first)])
        assert dialog.destination_edit.text() == str(first.parent / "converted")
        dialog.add_paths([str(second)])  # the first input stays first
        assert dialog.destination_edit.text() == str(first.parent / "converted")
        dialog.view_model.remove_indices([0])  # as Remove selected does
        dialog._follow_default_destination()
        assert dialog.destination_edit.text() == str(second.parent / "converted")

        chosen = str(tmp_path / "elsewhere")
        dialog.destination_edit.setText(chosen)
        dialog.add_paths([str(first)])
        assert dialog.destination_edit.text() == chosen
    finally:
        _dispose(dialog)


def test_without_inputs_the_destination_is_next_to_the_remembered_folder(tmp_path):
    _app()
    preferences = InMemoryUserPreferencesRepository({"format_converter.last_folder": str(tmp_path)})
    dialog = FormatConverterDialog(app_context=_context(preferences))
    try:
        assert dialog.destination_edit.text() == str(tmp_path / "converted")
    finally:
        _dispose(dialog)


def test_add_files_and_add_folder_start_in_the_remembered_folder(tmp_path, monkeypatch):
    _app()
    remembered = tmp_path / "earlier"
    remembered.mkdir()
    preferences = InMemoryUserPreferencesRepository({"format_converter.last_folder": str(remembered)})
    dialog = FormatConverterDialog(app_context=_context(preferences))
    try:
        chosen = _write_nxs(tmp_path / "new" / "scan.nxs")
        starts = []

        def open_names(_parent, _title, start, _filter):
            starts.append(start)
            return [str(chosen)], ""

        monkeypatch.setattr(QFileDialog, "getOpenFileNames", open_names)
        dialog._choose_files()
        assert starts == [str(remembered)]
        assert preferences.get("format_converter.last_folder") == str(chosen.parent)
        assert len(dialog.sources) == 1

        shown = []

        def exec_(folder_dialog):
            shown.append(folder_dialog.path_edit.text())
            return QDialog.Rejected

        monkeypatch.setattr(FolderImportDialog, "exec_", exec_)
        dialog._choose_folder()
        assert shown == [str(chosen.parent)]
    finally:
        _dispose(dialog)


# -- drag and drop ------------------------------------------------------------------------------


def _mime(*paths: Path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def test_dropped_files_and_folders_become_inputs(tmp_path):
    _app()
    preferences = InMemoryUserPreferencesRepository()
    dialog = FormatConverterDialog(app_context=_context(preferences))
    try:
        assert dialog.acceptDrops()
        folder = tmp_path / "series"
        _write_nxs(folder / "s1.nxs")
        _write_nxs(folder / "s2.nxs")
        _write_nxs(folder / "deeper" / "s3.nxs")  # a subfolder is not read
        single = _write_nxs(tmp_path / "single.nxs")
        note = tmp_path / "notes.txt"
        note.write_text("x", encoding="utf-8")

        text_mime = _mime(note)
        refused = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, text_mime, Qt.LeftButton, Qt.NoModifier)
        dialog.dragEnterEvent(refused)
        assert not refused.isAccepted()

        mime = _mime(folder, single, note)
        enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        dialog.dragEnterEvent(enter)
        assert enter.isAccepted()
        dialog.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))

        assert sorted(Path(source.path).name for source in dialog.sources) == ["s1.nxs", "s2.nxs", "single.nxs"]
        assert preferences.get("format_converter.last_folder") == str(folder)
        assert dialog.input_tree.topLevelItemCount() == 3
    finally:
        _dispose(dialog)
