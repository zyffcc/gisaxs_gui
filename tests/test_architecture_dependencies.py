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
APPLICATION_FORBIDDEN_SEGMENTS = frozenset(
    {
        "bornagain",
        "infrastructure",
        "keras",
        "presentation",
        "pyqt5",
        "pyqt6",
        "pyside2",
        "pyside6",
        "tensorflow",
    }
)

ALLOWED_LAYER_DEPENDENCIES = {
    "presentation": frozenset({"application"}),
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


def _absolute_imported_names(source: str) -> list[tuple[str, int]]:
    """返回真正的绝对 import；不把 ``from .calibration`` 误判为顶层包。"""
    imports: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.module, node.lineno))
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


def test_presentation_does_not_import_domain_directly() -> None:
    violations: list[str] = []
    for file_path in _python_files_in_layer("presentation"):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _imported_names(source):
            if "domain" in _module_segments(module_name):
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Presentation 只能通过 application API 使用 domain：\n" + "\n".join(
        violations
    )


def test_presentation_files_do_not_import_concrete_adapters() -> None:
    violations: list[str] = []
    for file_path in _python_files_in_layer("presentation"):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _imported_names(source):
            segments = _module_segments(module_name)
            if "infrastructure" in segments or "adapters" in segments:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Presentation 不得直接构造具体 adapter：\n" + "\n".join(
        violations
    )


def test_presentation_files_do_not_import_legacy_utils_package() -> None:
    violations: list[str] = []
    for file_path in _python_files_in_layer("presentation"):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _absolute_imported_names(source):
            if module_name.split(".", maxsplit=1)[0].casefold() == "utils":
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Presentation 不得依赖 legacy utils package：\n" + "\n".join(
        violations
    )


def test_qt_file_and_message_dialogs_are_confined_to_presentation() -> None:
    violations: list[str] = []
    for file_path in sorted(GIMAP_ROOT.rglob("*.py")):
        relative_parts = file_path.relative_to(GIMAP_ROOT).parts
        if "presentation" in relative_parts:
            continue
        for module_name, line_number in _imported_names(
            file_path.read_text(encoding="utf-8")
        ):
            if module_name in {"QFileDialog", "QMessageBox"}:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "QFileDialog/QMessageBox 只能位于 presentation：\n" + "\n".join(
        violations
    )


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


def test_application_files_do_not_import_gui_runtimes_or_concrete_adapters() -> None:
    violations: list[str] = []
    for file_path in _python_files_in_layer("application"):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _imported_names(source):
            forbidden = _module_segments(module_name) & APPLICATION_FORBIDDEN_SEGMENTS
            if forbidden:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Application 包含禁止依赖：\n" + "\n".join(violations)


def test_new_source_does_not_depend_on_legacy_compatibility_packages() -> None:
    violations: list[str] = []
    for file_path in sorted(GIMAP_ROOT.rglob("*.py")):
        source = file_path.read_text(encoding="utf-8")
        for module_name, line_number in _absolute_imported_names(source):
            root = module_name.split(".", maxsplit=1)[0].casefold()
            if root in {"calibration", "controllers", "trainset", "ui", "utils", "waxs"}:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "新架构源码不得反向依赖 legacy compatibility package：\n" + "\n".join(violations)


def test_production_source_does_not_use_internal_legacy_bridges() -> None:
    violations: list[str] = []
    for file_path in sorted(GIMAP_ROOT.rglob("*.py")):
        if file_path.name in {"legacy_bridge.py", "legacy_controller.py"}:
            continue
        for module_name, line_number in _imported_names(
            file_path.read_text(encoding="utf-8")
        ):
            if module_name.endswith(
                ("legacy_bridge", "legacy_controller")
            ):
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "生产源码不得通过内部 legacy bridge 装配：\n" + "\n".join(
        violations
    )


def test_features_do_not_import_other_feature_internals() -> None:
    violations: list[str] = []
    features_root = GIMAP_ROOT / "features"
    for file_path in sorted(features_root.rglob("*.py")):
        relative = file_path.relative_to(features_root)
        owner = relative.parts[0]
        for module_name, line_number in _imported_names(file_path.read_text(encoding="utf-8")):
            prefix = "src.gimap.features."
            if not module_name.startswith(prefix):
                continue
            target = module_name.removeprefix(prefix).split(".", maxsplit=1)[0]
            if target != owner:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Feature 不得导入其他 feature 内部实现：\n" + "\n".join(violations)


def test_shared_does_not_import_feature_implementations() -> None:
    violations: list[str] = []
    shared_root = GIMAP_ROOT / "shared"
    for file_path in sorted(shared_root.rglob("*.py")):
        for module_name, line_number in _imported_names(file_path.read_text(encoding="utf-8")):
            if module_name.startswith("src.gimap.features"):
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number}: {module_name}")

    assert not violations, "Shared 不得反向依赖 feature 实现：\n" + "\n".join(violations)


def test_global_params_singleton_is_confined_to_app_composition_root() -> None:
    allowed = {GIMAP_ROOT / "app" / "bootstrap.py"}
    violations: list[str] = []
    for file_path in sorted(GIMAP_ROOT.rglob("*.py")):
        if file_path in allowed:
            continue
        source = file_path.read_text(encoding="utf-8")
        if "core.global_params" in source:
            violations.append(str(file_path.relative_to(PROJECT_ROOT)))

    assert not violations, "global_params 只能在 AppContext composition root 适配：\n" + "\n".join(
        violations
    )


def test_legacy_user_preferences_singleton_is_confined_to_app_composition_root() -> None:
    allowed = {GIMAP_ROOT / "app" / "bootstrap.py"}
    violations: list[str] = []
    for file_path in sorted(GIMAP_ROOT.rglob("*.py")):
        if file_path in allowed:
            continue
        source = file_path.read_text(encoding="utf-8")
        if "core.user_settings" in source:
            violations.append(str(file_path.relative_to(PROJECT_ROOT)))

    assert not violations, "user_settings 只能在 AppContext composition adapter 适配：\n" + "\n".join(
        violations
    )


def test_runtime_python_modules_do_not_regrow_into_monoliths() -> None:
    """Require an explicit architecture review before a runtime module exceeds 600 lines."""

    runtime_roots = ("src", "controllers", "ui", "trainset", "utils")
    violations: list[str] = []
    for root_name in runtime_roots:
        for file_path in sorted((PROJECT_ROOT / root_name).rglob("*.py")):
            line_count = len(file_path.read_text(encoding="utf-8-sig").splitlines())
            if line_count > 600:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}: {line_count} lines")

    assert not violations, "Runtime Python module 超过 600 行 monolith 门禁：\n" + "\n".join(
        violations
    )


def test_presentation_composition_entries_do_not_regrow_into_monoliths() -> None:
    """Keep stable entry points small while focused modules hold the behavior."""

    entry_names = {"dialog.py", "multifile_results.py", "page.py", "view_binding.py"}
    violations: list[str] = []
    features_root = GIMAP_ROOT / "features"
    for file_path in sorted(features_root.rglob("*.py")):
        if file_path.name not in entry_names or "presentation" not in file_path.parts:
            continue
        line_count = len(file_path.read_text(encoding="utf-8").splitlines())
        if line_count > 600:
            relative_path = file_path.relative_to(PROJECT_ROOT)
            violations.append(f"{relative_path}: {line_count} lines")

    assert not violations, "Presentation 聚合入口超过 600 行 review 阈值：\n" + "\n".join(
        violations
    )


def test_feature_view_models_stay_within_review_threshold() -> None:
    violations: list[str] = []
    for file_path in sorted((GIMAP_ROOT / "features").rglob("view_model.py")):
        line_count = len(file_path.read_text(encoding="utf-8").splitlines())
        if line_count > 300:
            relative_path = file_path.relative_to(PROJECT_ROOT)
            violations.append(f"{relative_path}: {line_count} lines")

    assert not violations, "ViewModel 超过 300 行 review 阈值：\n" + "\n".join(violations)
