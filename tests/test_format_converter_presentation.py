"""Format Converter presentation ownership and Qt boundary regression tests."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

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
from ui.format_converter_dialog import (
    ConversionProgressDialog as LegacyConversionProgressDialog,
)
from ui.format_converter_dialog import FolderImportDialog as LegacyFolderImportDialog
from ui.format_converter_dialog import FormatConverterDialog as LegacyFormatConverterDialog


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


def test_legacy_dialog_entry_reexports_feature_owned_classes() -> None:
    assert LegacyFormatConverterDialog is FormatConverterDialog
    assert LegacyFolderImportDialog is FolderImportDialog
    assert LegacyConversionProgressDialog is ConversionProgressDialog

    legacy_source = (PROJECT_ROOT / "ui" / "format_converter_dialog.py").read_text(encoding="utf-8")
    assert "class FormatConverterDialog" not in legacy_source
    assert len(legacy_source.splitlines()) <= 12


def test_menu_opens_converter_through_feature_owned_module() -> None:
    menu_source = (PROJECT_ROOT / "src/gimap/app/menu_manager.py").read_text(encoding="utf-8")

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
