"""Format Converter presentation ownership and Qt boundary regression tests."""

from __future__ import annotations

import ast
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QHeaderView

from src.gimap.app import AppContext
from src.gimap.features.format_converter.bootstrap import (
    create_format_converter_view_model,
)
from src.gimap.features.format_converter.domain import InputSource
from src.gimap.features.format_converter.presentation.dialog import (
    ConversionProgressDialog,
    FolderImportDialog,
    FormatConverterDialog,
)
from src.gimap.features.format_converter.presentation.views import (
    ConversionProgressDialogView,
    FolderImportDialogView,
    FormatConverterDialogView,
)
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def test_menu_opens_converter_through_feature_owned_module() -> None:
    menu_source = (PROJECT_ROOT / "src/gimap/app/menus.py").read_text(encoding="utf-8")

    assert "src.gimap.features.format_converter.presentation.dialog" in menu_source
    assert "from ui.format_converter_dialog" not in menu_source


def test_feature_dialog_preserves_workspace_controls_and_defaults() -> None:
    _app()
    dialog = FormatConverterDialog(app_context=_context())

    assert dialog.windowTitle() == "Format Converter"
    assert dialog.minimumWidth() == 920
    assert dialog.minimumHeight() == 650
    assert dialog.stack.count() == 3
    assert [dialog.frame_mode.itemText(index) for index in range(dialog.frame_mode.count())] == [
        "All",
        "Current frame",
        "Frame range",
        "Custom",
        "Every Nth frame",
    ]
    assert set(dialog.format_buttons) == {"TIFF", "CBF", "HDF5", "NumPy"}
    assert dialog.format_buttons["TIFF"].isChecked()
    assert dialog.back_button.shortcut().toString() == ""
    assert dialog.next_button.shortcut().toString() == ""
    assert dialog.cancel_button.shortcut().toString() == ""
    for attribute in (
        "input_tree",
        "dataset_combo",
        "selection_table",
        "frame_mode",
        "preview_labels",
        "destination_edit",
        "naming_combo",
        "output_summary",
    ):
        assert hasattr(dialog, attribute)
    dialog.close()


def test_main_dialog_layout_is_owned_by_feature_python_view() -> None:
    view = (
        PROJECT_ROOT
        / "src/gimap/features/format_converter/presentation/views"
        / "format_converter_dialog_view.py"
    )
    dialog_source = (
        PROJECT_ROOT
        / "src/gimap/features/format_converter/presentation/dialog.py"
    ).read_text(encoding="utf-8")

    assert view.is_file()
    assert issubclass(FormatConverterDialog, FormatConverterDialogView)
    assert "def _build_ui(" not in dialog_source
    assert "def _build_input_page(" not in dialog_source
    assert "def _build_selection_page(" not in dialog_source
    assert "def _build_output_page(" not in dialog_source


def test_auxiliary_dialog_layouts_are_owned_by_feature_python_views() -> None:
    _app()
    folder = FolderImportDialog(view_model=create_format_converter_view_model(_context()))
    progress = ConversionProgressDialog("/tmp")

    assert isinstance(folder, FolderImportDialogView)
    assert isinstance(progress, ConversionProgressDialogView)
    assert folder.cbf.isChecked() and folder.tiff.isChecked() and folder.nxs.isChecked()
    assert folder.recursive.isChecked() is False
    assert progress.job_status.objectName() == "job_status"
    assert progress.minimumWidth() == 570

    folder.close()
    progress.running = False
    progress.close()


def test_view_model_owns_frame_and_output_format_commands_without_qapplication() -> None:
    view_model = create_format_converter_view_model(_context())
    nxs = InputSource(path="/tmp/scan.nxs", file_type="NXS", frame_count=12)
    tiff = InputSource(path="/tmp/image.tif", file_type="TIFF")
    view_model.sources.extend((nxs, tiff))

    view_model.apply_frame_selection(
        [0, 1],
        "Custom",
        custom_frames="1, 5, 8–10",
    )

    assert nxs.selected_frames == [0, 4, 7, 8, 9]
    assert tiff.selected_frames == [0]
    visibility = view_model.output_format_visibility()
    assert all(visibility.values())


def test_presentation_has_no_conversion_or_file_adapter_implementation() -> None:
    dialog_source = (
        PROJECT_ROOT
        / "src"
        / "gimap"
        / "features"
        / "format_converter"
        / "presentation"
        / "dialog.py"
    ).read_text(encoding="utf-8")
    view_model_source = (
        PROJECT_ROOT
        / "src"
        / "gimap"
        / "features"
        / "format_converter"
        / "presentation"
        / "view_model.py"
    ).read_text(encoding="utf-8")

    assert "utils.format_converter" not in dialog_source
    assert "parse_custom_frames" not in dialog_source
    assert ".is_dir(" not in dialog_source
    assert "LocalSourceRepository" not in dialog_source
    assert "LocalConversionExecutor" not in dialog_source
    assert "ConvertFile" not in dialog_source
    imported_modules = []
    for node in ast.walk(ast.parse(view_model_source)):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not any(module.casefold().startswith("pyqt") for module in imported_modules)
    assert "QWidget" not in view_model_source
    assert "QMessageBox" not in view_model_source
    assert "QFileDialog" not in view_model_source


# -- Tool-window fixes: frame modes in Chinese, step button, source table, Esc ------------------


def _settle(app, seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)


def test_frame_modes_use_english_keys_in_the_chinese_interface(tmp_path) -> None:
    from src.gimap.app.presentation.i18n import apply_language

    _app()
    dialog = FormatConverterDialog(app_context=_context())
    dialog._start_preview = lambda _source: None  # no preview thread in this test
    source = InputSource(path=str(tmp_path / "scan.nxs"), file_type="NXS", frame_count=100)
    dialog.view_model.sources.append(source)
    apply_language("zh", [dialog])
    try:
        assert dialog.frame_mode.itemText(0) == "全部"
        assert dialog.frame_mode.itemText(3) == "自定义"
        dialog._refresh_selection_table()
        dialog.selection_table.selectRow(0)
        dialog.frame_mode.setCurrentIndex(0)
        dialog._apply_frame_selection()
        assert source.selected_frames == list(range(100))
        dialog.frame_mode.setCurrentIndex(3)
        assert dialog.custom_frames.isEnabled()
        assert not dialog.nth_frame.isEnabled()
        dialog.frame_mode.setCurrentIndex(4)
        assert dialog.nth_frame.isEnabled()
        assert [dialog.frame_mode.itemData(i) for i in range(dialog.frame_mode.count())] == [
            "All",
            "Current frame",
            "Frame range",
            "Custom",
            "Every Nth frame",
        ]
    finally:
        apply_language("en", [dialog])
        dialog.close()


def test_step_button_is_the_primary_action_without_a_mnemonic(monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    _app()
    dialog = FormatConverterDialog(app_context=_context())
    dialog.stack.setCurrentIndex(2)
    dialog._update_step_header()
    text = dialog.next_button.text()
    assert text == "Review && Convert"
    assert text.replace("&&", "&") == "Review & Convert"
    assert QKeySequence.mnemonic(text).isEmpty()
    assert dialog.next_button.property("gimapRole") == "primary"
    assert not dialog.next_button.isDefault()

    monkeypatch.setitem(i18n.ZH, "Next", "下一步")
    monkeypatch.setitem(i18n.ZH, "Review && Convert", "检查并转换")
    i18n.apply_language("zh")
    try:
        for step, expected in ((0, "下一步"), (2, "检查并转换"), (1, "下一步")):
            dialog.stack.setCurrentIndex(step)
            dialog._update_step_header()
            assert dialog.next_button.text() == expected
    finally:
        i18n.apply_language("en")
        dialog.close()


def test_source_table_shows_distinct_series_names_and_full_paths(tmp_path) -> None:
    app = _app()
    dialog = FormatConverterDialog(app_context=_context())
    dialog._start_preview = lambda _source: None
    stem = "jg_gisaxs_4nm_old_3ml_insitu_ds03_00001"
    paths = [tmp_path / f"{stem}_{suffix}.cbf" for suffix in ("00005", "00033", "00045")]
    dialog.view_model.sources.extend(InputSource(path=str(path), file_type="CBF") for path in paths)
    dialog.resize(920, 650)
    dialog.show()
    dialog._next()
    dialog.selection_splitter.setSizes([560, 320])  # narrow enough that the names elide
    _settle(app, 0.2)
    table = dialog.selection_table
    header = table.horizontalHeader()

    assert table.textElideMode() == Qt.ElideMiddle
    assert dialog.input_tree.textElideMode() == Qt.ElideMiddle
    assert not table.verticalHeader().isVisible()
    assert header.sectionResizeMode(1) == QHeaderView.Stretch
    for column in (0, 2, 3, 4):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeToContents
    frames_title = table.horizontalHeaderItem(3).text()
    assert header.sectionSize(3) >= header.fontMetrics().horizontalAdvance(frames_title)
    width = header.sectionSize(1) - 12
    assert table.fontMetrics().horizontalAdvance(table.item(0, 1).text()) > width
    shown = [
        table.fontMetrics().elidedText(table.item(row, 1).text(), table.textElideMode(), width)
        for row in range(3)
    ]
    assert len(set(shown)) == 3
    for row, suffix in enumerate(("00005", "00033", "00045")):
        assert shown[row].endswith(f"{suffix}.cbf")
        assert table.item(row, 1).toolTip() == str(paths[row])
    dialog.close()


def test_escape_while_a_preview_loads_keeps_the_dialog_until_the_thread_ends(tmp_path) -> None:
    app = _app()
    dialog = FormatConverterDialog(app_context=_context())
    release = threading.Event()

    def slow_preview(_source):
        release.wait(10)
        return []

    dialog.view_model.load_preview = slow_preview
    dialog.show()
    events = []
    dialog.destroyed.connect(lambda *_: events.append("destroyed"))
    dialog._start_preview(InputSource(path=str(tmp_path / "frame.tif"), file_type="TIFF"))
    thread = dialog._preview_thread
    assert thread is not None and thread.isRunning()
    thread.finished.connect(lambda: events.append("finished"))

    QTest.keyClick(dialog, Qt.Key_Escape)
    _settle(app, 0.2)
    assert events == []
    assert thread.isRunning()

    release.set()
    end = time.monotonic() + 10
    while "destroyed" not in events and time.monotonic() < end:
        _settle(app, 0.05)
    assert events == ["finished", "destroyed"]
