"""WAXS presentation ownership and offscreen compatibility tests."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QMainWindow

from src.gimap.app import AppContext
from src.gimap.features.waxs.infrastructure import (
    load_image_matrix as infrastructure_load_image_matrix,
)
from src.gimap.features.waxs.presentation.page import (
    ImageLoadResult,
    InSituProcessingWidget,
    ScatteringImageViewer,
)
from src.gimap.features.waxs.presentation.views import WaxsPageView
from src.gimap.features.waxs.standalone import WaxsStandaloneWindow
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.waxs_page import (
    InSituProcessingWidget as LegacyInSituProcessingWidget,
)
from ui.waxs_page import (
    ScatteringImageViewer as LegacyScatteringImageViewer,
)
from ui.waxs_page import load_image_matrix as legacy_load_image_matrix
from ui.main_window import Ui_MainWindow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_APP = None


class _ViewModel:
    def working_directory(self) -> str:
        return "/tmp"

    def normalize_path(self, path) -> str:
        return str(path)

    def is_directory(self, path) -> bool:
        return Path(path).is_dir()


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_legacy_waxs_entry_reexports_feature_owned_page_and_loader() -> None:
    assert LegacyInSituProcessingWidget is InSituProcessingWidget
    assert LegacyScatteringImageViewer is ScatteringImageViewer
    assert legacy_load_image_matrix is infrastructure_load_image_matrix

    legacy_source = (PROJECT_ROOT / "ui" / "waxs_page.py").read_text(encoding="utf-8")
    assert "class InSituProcessingWidget" not in legacy_source
    assert "def load_image_matrix" not in legacy_source
    assert len(legacy_source.splitlines()) <= 35


def test_waxs_image_load_result_remains_a_dataclass_after_module_split() -> None:
    image = np.ones((2, 3), dtype=np.float32)
    result = ImageLoadResult("scan.nxs", 1, 4, image)

    assert result.frame_index == 1
    assert result.frame_count == 4
    assert result.image is image


def test_feature_page_preserves_sections_controls_signals_and_job_status_offscreen() -> None:
    _app()
    page = InSituProcessingWidget(view_model=_ViewModel())
    statuses: list[str] = []
    page.statusChanged.connect(statuses.append)

    assert page.objectName() == "waxsEmbeddedPage"
    assert page.batch_output_edit.text() == "/tmp"
    for object_name in (
        "waxsInputSection",
        "waxsContentSplitter",
        "waxsViewTabs",
        "waxsControlTabs",
        "waxsAdvancedControlTabs",
        "waxsConfigureSection",
        "waxsAdvancedSection",
        "waxsRunSection",
        "waxsWorkflowTabs",
        "waxsControlsScrollArea",
        "waxsAdvancedScrollArea",
        "waxsBatchScrollArea",
        "waxsPreviewPanel",
        "waxsResultsSection",
        "waxsExportSection",
    ):
        assert page.findChild(QObject, object_name) is not None

    assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == [
        "ROI / Cut",
        "1D Integration",
    ]
    assert [page.advanced_tabs.tabText(index) for index in range(page.advanced_tabs.count())] == [
        "Display",
        "Mask",
        "Geometry",
    ]
    assert page.cut_type_combo.currentText() == "Q Range"
    assert page.integration_mode.currentText() == "Radial"
    assert page.bin_spin.value() == 500
    assert page.qr_min_spin.value() == -121.0
    assert page.distance_spin.value() == 2000.0
    assert page.pixel_x_spin.value() == 75.0
    assert page.wavelength_spin.value() == 1.0332
    assert page.batch_pattern_edit.text() == "*.tif"
    assert [
        page.waxs_workflow_tabs.tabText(index)
        for index in range(page.waxs_workflow_tabs.count())
    ] == ["1  Cut + integrate", "2  Advanced", "3  Batch"]
    assert page.waxs_advanced_section.is_expanded() is True
    assert page.waxsAdvancedToggle.isHidden()
    assert page.open_button.property("waxsPrimaryAction") is True
    assert page.integrate_button.property("waxsPrimaryAction") is True
    assert page.batch_start_button.property("waxsPrimaryAction") is True
    for button in (
        page.open_button,
        page.reload_button,
        page.export_button,
        page.integrate_button,
        page.export_1d_button,
        page.batch_start_button,
        page.batch_pause_button,
        page.batch_stop_button,
    ):
        assert button.shortcut().toString() == ""

    page.set_job_state("running", "Processing frame", progress=35)
    assert page.status_label is page.waxs_job_status.message_label
    assert page.progress is page.waxs_job_status.progress_bar
    assert page.progress.maximum() == 100
    assert page.progress.value() == 35
    assert page.waxs_job_status.state_label.text() == "RUNNING"
    assert statuses == ["Processing frame"]
    page.close()


def test_feature_page_keeps_file_implementations_outside_presentation() -> None:
    page_source = (PROJECT_ROOT / "src/gimap/features/waxs/presentation/page.py").read_text(
        encoding="utf-8"
    )
    view_model_source = (
        PROJECT_ROOT / "src/gimap/features/waxs/presentation/view_model.py"
    ).read_text(encoding="utf-8")

    assert "utils.path_utils" not in page_source
    assert "calibration.image_loader" not in page_source
    assert "os.path" not in page_source
    assert "os.getcwd" not in page_source
    assert "def load_image_matrix" not in page_source

    imported_modules: list[str] = []
    for node in ast.walk(ast.parse(view_model_source)):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not any(module.casefold().startswith("pyqt") for module in imported_modules)
    for qt_name in ("QWidget", "QMessageBox", "QFileDialog"):
        assert qt_name not in view_model_source


def test_waxs_page_static_layout_is_owned_by_python_views() -> None:
    page_source = (PROJECT_ROOT / "src/gimap/features/waxs/presentation/page.py").read_text(
        encoding="utf-8"
    )
    views = PROJECT_ROOT / "src/gimap/features/waxs/presentation/views"

    assert issubclass(InSituProcessingWidget, WaxsPageView)
    for removed_builder in (
        "def _build_ui",
        "def _build_toolbar",
        "def _display_tab",
        "def _mask_tab",
        "def _geometry_tab",
        "def _roi_tab",
        "def _integration_tab",
        "def _batch_tab",
    ):
        assert removed_builder not in page_source
    assert {path.name for path in views.glob("*_view.py")} == {
        "advanced_panel_view.py",
        "batch_panel_view.py",
        "configure_panel_view.py",
        "integration_panel_view.py",
        "page_view.py",
        "preview_panel_view.py",
        "roi_panel_view.py",
        "toolbar_view.py",
    }


def test_application_shell_reserves_stable_waxs_page_slot() -> None:
    app = _app()
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.mainWindowWidget.count() == 5
    assert ui.mainWindowWidget.indexOf(ui.waxsPageHost) == 4
    assert ui.waxsPageHost.objectName() == "waxsPageHost"

    window.close()
    app.processEvents()


def test_standalone_window_hosts_feature_owned_page_offscreen() -> None:
    app = _app()
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window = WaxsStandaloneWindow(context)

    assert window.centralWidget() is window.page
    assert isinstance(window.page, InSituProcessingWidget)
    assert window.windowTitle() == "In-situ Data Processing"
    assert window.page.view_model is not None

    window.close()
    app.processEvents()
