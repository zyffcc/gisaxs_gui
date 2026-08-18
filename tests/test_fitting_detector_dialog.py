import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app import AppContext
from src.gimap.features.fitting.bootstrap import create_fitting_view_model
from src.gimap.features.fitting.presentation.detector_parameters_dialog import (
    DetectorParametersDialog,
)
from src.gimap.features.fitting.presentation.views import DetectorParametersDialogView
from src.gimap.integrations.jobs import LocalProcessJobRunner
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from ui.detector_parameters_dialog import (
    DetectorParametersDialog as LegacyDetectorParametersDialog,
)


_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_detector_dialog_uses_injected_view_model_and_legacy_path_reexports() -> None:
    app = _app()
    settings = InMemorySettingsRepository(
        {"fitting": {"detector": {"distance": 1450.0}}, "beam": {"wavelength": 0.02}}
    )
    context = AppContext(
        settings=settings,
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    view_model = create_fitting_view_model(context)

    dialog = DetectorParametersDialog(view_model=view_model)

    assert LegacyDetectorParametersDialog is DetectorParametersDialog
    assert isinstance(dialog, DetectorParametersDialogView)
    assert dialog.objectName() == "DetectorParametersDialog"
    assert dialog.distance_spinbox.value() == 1450.0
    assert dialog.wavelength_spinbox.value() == 0.02

    dialog.distance_spinbox.setValue(1600.0)
    dialog._save_parameters()
    assert settings.get("fitting", "detector.distance") == 1600.0

    dialog.close()
    app.processEvents()
