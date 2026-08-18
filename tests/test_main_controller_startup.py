"""ApplicationRuntime startup sequencing regression tests."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT_ROOT / "src/gimap/app/runtime.py"


def _method(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method not found: {name}")


def _binding_initialize_calls(method: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "initialize" or not isinstance(node.func.value, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner.value, ast.Name) and owner.value.id == "self":
            names.append(owner.attr)
    return names


def test_delayed_startup_initializes_each_feature_binding_once() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert _binding_initialize_calls(_method(tree, "_initialize_ui")) == []
    assert _binding_initialize_calls(_method(tree, "_delayed_feature_initialization")) == [
        "trainset",
        "fitting",
        "classification",
        "prediction",
    ]


def test_main_composition_root_injects_bornagain_adapter() -> None:
    controller_source = RUNTIME.read_text(encoding="utf-8")
    main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

    assert "src.gimap.integrations.bornagain" not in controller_source
    assert "BornAgainSimulator(" not in controller_source
    assert "simulation_port=BornAgainSimulator(runner=self.app_context.jobs)" in main_source


def test_application_runtime_uses_app_context_without_global_registry() -> None:
    source = RUNTIME.read_text(encoding="utf-8")

    assert "core.global_params" not in source
    assert "global_params" not in source
    assert "self.settings = self.app_context.settings" in source
    assert "def _register_controllers" not in source
    assert "def _register_ui_controls" not in source


def test_legacy_main_controller_path_is_a_thin_reexport() -> None:
    source = (PROJECT_ROOT / "controllers/main_controller.py").read_text(encoding="utf-8")

    assert "src.gimap.app.runtime" in source
    assert len(source.splitlines()) <= 7

    app_legacy = (PROJECT_ROOT / "src/gimap/app/legacy_controller.py").read_text(
        encoding="utf-8"
    )
    assert "from .runtime import ApplicationRuntime" in app_legacy
    assert len(app_legacy.splitlines()) <= 9


def test_legacy_main_window_and_settings_paths_are_thin_reexports() -> None:
    main_window = (PROJECT_ROOT / "ui/main_window.py").read_text(encoding="utf-8")
    settings = (PROJECT_ROOT / "ui/settings_dialog.py").read_text(encoding="utf-8")
    production_main = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

    assert "class Ui_MainWindow" not in main_window
    assert "class SettingsDialog" not in settings
    assert len(main_window.splitlines()) <= 7
    assert len(settings.splitlines()) <= 7
    assert "src.gimap.app.window_view" in production_main
    assert "src.gimap.app.presentation.settings_dialog" in production_main
    assert "from ui.main_window" not in production_main
    assert "from ui.settings_dialog" not in production_main
