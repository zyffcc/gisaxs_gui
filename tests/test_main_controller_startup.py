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


def test_main_entry_imports_feature_owned_shell_modules() -> None:
    production_main = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

    assert "src.gimap.app.window_view" in production_main
    assert "src.gimap.app.main_window" in production_main
    assert "src.gimap.app.menus" in production_main
    # Qt's own high-DPI scaling replaces the removed per-resolution profiles.
    assert "HighDpiScaleFactorRoundingPolicy.PassThrough" in production_main
    for alias_root in ("ui", "controllers", "calibration", "trainset"):
        assert f"from {alias_root}." not in production_main
        assert f"import {alias_root}." not in production_main
