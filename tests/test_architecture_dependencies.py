"""面向新 ``src/gimap`` 代码的依赖方向守护测试。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIMAP_ROOT = PROJECT_ROOT / "src" / "gimap"

DOMAIN_FORBIDDEN_MODULES = frozenset(
    {
        "bornagain",
        "keras",
        "pyqt5",
        "pyqt6",
        "pyside2",
        "pyside6",
        "tensorflow",
    }
)

ALLOWED_LAYER_DEPENDENCIES = {
    "presentation": frozenset({"application", "domain"}),
    "application": frozenset({"domain"}),
    "domain": frozenset(),
    "infrastructure": frozenset({"application", "domain"}),
}


def _imported_names(source: str) -> list[tuple[str, int]]:
    """返回源码中的 import 名称及其行号，包括相对导入的目标名称。"""
    imports: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
            imports.extend((alias.name, node.lineno) for alias in node.names)
    return imports


def _python_files_in_layer(layer: str) -> list[Path]:
    """查找 ``src/gimap`` 下属于指定架构层的 Python 文件。"""
    return sorted(
        file_path
        for file_path in GIMAP_ROOT.rglob("*.py")
        if layer in file_path.relative_to(GIMAP_ROOT).parts
    )


def _module_segments(module_name: str) -> set[str]:
    return {segment.casefold() for segment in module_name.split(".")}


def _domain_module_is_allowed(module_name: str) -> bool:
    root_module = module_name.split(".", maxsplit=1)[0].casefold()
    return root_module not in DOMAIN_FORBIDDEN_MODULES


def _layer_dependency_is_allowed(source_layer: str, target_layer: str) -> bool:
    return target_layer in ALLOWED_LAYER_DEPENDENCIES[source_layer]


@pytest.mark.parametrize("module_name", ["PyQt5", "tensorflow.keras", "bornagain"])
def test_domain_policy_rejects_gui_ml_and_simulation_runtimes(module_name: str) -> None:
    assert not _domain_module_is_allowed(module_name)


@pytest.mark.parametrize("module_name", ["numpy", "scipy.optimize"])
def test_domain_policy_allows_approved_numerical_libraries(module_name: str) -> None:
    assert _domain_module_is_allowed(module_name)


def test_domain_files_do_not_import_forbidden_runtimes() -> None:
    violations: list[str] = []
    for file_path in _python_files_in_layer("domain"):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _imported_names(source):
            if not _domain_module_is_allowed(module_name):
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Domain 包含禁止的外部依赖：\n" + "\n".join(violations)


def test_presentation_can_depend_on_application() -> None:
    assert _layer_dependency_is_allowed("presentation", "application")


def test_application_cannot_depend_on_presentation() -> None:
    assert not _layer_dependency_is_allowed("application", "presentation")

    violations: list[str] = []
    for file_path in _python_files_in_layer("application"):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _imported_names(source):
            if "presentation" in _module_segments(module_name):
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Application 不得依赖 presentation：\n" + "\n".join(violations)
