"""Debounced fitting parameter persistence through SettingsRepository。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDoubleSpinBox

from src.gimap.integrations.state import InMemorySettingsRepository
from src.gimap.features.fitting.presentation.parameter_trigger import (
    UniversalParameterTriggerManager as FeatureParameterTriggerManager,
)
from utils.universal_parameter_trigger_manager import UniversalParameterTriggerManager


_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_meta_parameter_persists_through_injected_settings_repository():
    _app()
    settings = InMemorySettingsRepository()
    manager = UniversalParameterTriggerManager(settings_repository=settings)
    widget = QDoubleSpinBox()
    manager.register_parameter_widget(
        widget,
        "beam-center-x",
        "fitting",
        lambda _value: None,
        meta={
            "persist": "settings",
            "key_path": ("fitting", "detector.beam_center_x"),
            "connect_mode": "external",
        },
    )
    widget.setValue(37.5)

    manager._commit_meta_widget("beam-center-x")

    assert settings.get("fitting", "detector.beam_center_x") == 37.5


def test_parameter_trigger_module_has_no_global_params_import():
    import inspect

    source = inspect.getsource(UniversalParameterTriggerManager)
    assert "from core.global_params" not in source


def test_legacy_parameter_trigger_path_reexports_feature_owner():
    assert UniversalParameterTriggerManager is FeatureParameterTriggerManager


def test_split_trigger_keeps_legacy_callback_and_diagnostics_contract():
    _app()
    manager = UniversalParameterTriggerManager()
    widget = QDoubleSpinBox()
    immediate = []
    delayed = []
    manager.register_parameter_widget(
        widget,
        "legacy-value",
        "fitting",
        immediate.append,
        delayed.append,
        connect_signals=False,
    )

    manager._on_immediate_trigger("legacy-value", 2.5)
    manager._on_delayed_trigger("legacy-value", 3.5)
    manager.register_parameter_widget(
        widget,
        "meta-value",
        "fitting",
        lambda _value: None,
        meta={"connect_mode": "external"},
    )

    assert immediate == [2.5]
    assert delayed == [3.5]
    assert manager.get_meta_entry("meta-value")["widget_id"] == "meta-value"
    assert manager.debug_dump_meta()["meta-value"]["current"] == widget.value()
