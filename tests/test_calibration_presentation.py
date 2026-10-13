"""Geometry Calibration presentation ownership and Qt boundary tests."""

from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QScrollArea

from src.gimap.app import AppContext
from src.gimap.features.calibration.bootstrap import create_calibration_view_model
from src.gimap.features.calibration.domain import (
    CalibrationCandidate,
    CalibrationResult,
    DetectorImage,
    energy_to_wavelength,
)
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.features.calibration.presentation.views import (
    GeometryCalibrationDialogView,
)
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dialog_source() -> str:
    presentation = (
        PROJECT_ROOT / "src/gimap/features/calibration/presentation"
    )
    paths = [presentation / "dialog.py", presentation / "workers.py"]
    paths.extend(sorted((presentation / "bindings").glob("*.py")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)
_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def _context(settings=None) -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(settings or {}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def _result(source_path: Path) -> CalibrationResult:
    candidate = CalibrationCandidate(
        "agbh",
        123.5,
        234.5,
        1456.7,
        matched_ring_count=3,
    )
    return CalibrationResult(
        str(source_path),
        10,
        20,
        "abc",
        12.0,
        energy_to_wavelength(12.0),
        "Detector",
        172e-6,
        172e-6,
        candidate,
        [candidate],
        datetime.now(timezone.utc).isoformat(),
    )


def test_menu_opens_calibration_through_feature_owned_module() -> None:
    menu_source = (PROJECT_ROOT / "src/gimap/app/menus.py").read_text(encoding="utf-8")

    assert "src.gimap.features.calibration.presentation.dialog" in menu_source
    assert "from ui.geometry_calibration_dialog" not in menu_source


def test_feature_dialog_preserves_object_names_flags_controls_and_shortcuts() -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())

    assert dialog.windowTitle() == "Geometry Calibration"
    assert dialog.minimumWidth() == 900
    assert dialog.minimumHeight() == 560
    assert dialog.windowFlags() & Qt.WindowMaximizeButtonHint
    assert dialog.calibrate_button.objectName() == "primaryCalibrationButton"
    assert dialog.manual_refine_button.objectName() == "manualRefineButton"
    assert dialog.fit_image_button.objectName() == "previewActionButton"
    assert dialog.manual_group.objectName() == "manualRefinementGroup"
    assert dialog.manual_hint.objectName() == "manualHint"
    assert dialog.preview_info_label.objectName() == "previewInfo"
    assert dialog.overlay_legend.objectName() == "overlayLegend"
    assert dialog.findChild(QScrollArea, "calibrationControlsScroll") is not None
    for button in (
        dialog.open_button,
        dialog.calibrate_button,
        dialog.cancel_button,
        dialog.import_button,
        dialog.export_button,
        dialog.apply_button,
        dialog.close_button,
    ):
        assert button.shortcut().toString() == ""
    assert hasattr(dialog, "calibrationApplied")
    dialog.close()


def test_calibration_layout_is_owned_by_feature_python_view() -> None:
    view = (
        PROJECT_ROOT
        / "src/gimap/features/calibration/presentation/views"
        / "geometry_calibration_dialog_view.py"
    )
    dialog_source = _dialog_source()

    assert view.is_file()
    assert issubclass(GeometryCalibrationDialog, GeometryCalibrationDialogView)
    assert "def _build_ui(" not in dialog_source
    assert "FigureCanvas(self.figure)" in dialog_source


def test_view_model_owns_calibration_presentation_commands_without_qapplication(
    tmp_path: Path,
) -> None:
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    context = _context()
    context.instrument_profiles = InMemoryInstrumentProfileRepository()
    view_model = create_calibration_view_model(context)
    view_model.result = _result(tmp_path / "scan_agbh.cbf")
    view_model.apply_result()

    assert view_model.detected_standard_keys("scan_agbh.cbf") == ("agbh",)
    assert view_model.standard_display_name("agbh") == "Silver behenate (AgBH)"
    preview = view_model.display_candidate(
        manual_enabled=True,
        center_x_px=130.0,
        center_y_px=240.0,
        distance_mm=1500.0,
    )
    assert preview is not view_model.result.selected_candidate
    assert preview.center_x_px == 130.0
    assert view_model.result.selected_candidate.center_x_px == 123.5
    view_model.commit_manual_refinement(
        manual_enabled=True,
        center_x_px=130.0,
        center_y_px=240.0,
        distance_mm=1500.0,
    )
    assert view_model.result.selected_candidate.center_x_px == 130.0
    # A few pixels and 3 % in distance: not a significant change of the saved profile.
    assert view_model.significantly_changed_profile() is None
    assert view_model.default_export_path("scan.cbf") == "scan.gimap-calibration.json"


def test_feature_dialog_has_no_migrated_business_or_file_implementation() -> None:
    presentation_root = PROJECT_ROOT / "src" / "gimap" / "features" / "calibration" / "presentation"
    dialog_source = _dialog_source()
    view_model_source = (presentation_root / "view_model.py").read_text(encoding="utf-8")

    assert "ui.geometry_calibration_dialog" not in dialog_source
    assert "utils.path_utils" not in dialog_source
    assert "STANDARDS" not in dialog_source
    assert "q_to_ring_radius_m" not in dialog_source
    assert "distance_from_ring_radius" not in dialog_source
    assert "copy.deepcopy" not in dialog_source
    imported_modules = []
    for node in ast.walk(ast.parse(view_model_source)):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not any(module.casefold().startswith("pyqt") for module in imported_modules)
    for qt_name in ("QWidget", "QMessageBox", "QFileDialog"):
        assert qt_name not in view_model_source


def test_feature_dialog_accepts_loaded_image_offscreen() -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    image = DetectorImage(
        np.ones((32, 40), dtype=np.float32),
        np.zeros((32, 40), dtype=bool),
        Path("unknown_detector.cbf"),
    )

    dialog._image_loaded(image)

    assert dialog.path_edit.text() == ""
    assert dialog.detector_combo.currentIndex() == 0
    assert "choose a detector model" in dialog.detector_label.text()
    assert dialog.calibrate_button.isEnabled()
    dialog.close()
