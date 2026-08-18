"""Architecture tests for feature-owned, hand-maintained Python Views."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "gimap"

EXPECTED_VIEWS_BY_OWNER = {
    "app": {"main_window_view.py", "settings_dialog_view.py"},
    "calibration": {"geometry_calibration_dialog_view.py"},
    "classification": {
        "classification_dataset_panel_view.py",
        "classification_experiment_panel_view.py",
        "classification_inspection_panel_view.py",
        "classification_page_view.py",
        "classification_preprocessing_panel_view.py",
        "classification_results_panel_view.py",
    },
    "fitting": {
        "detector_parameters_dialog_view.py",
        "fitting_page_view.py",
        "fitting_workspace_view.py",
        "independent_fit_window_view.py",
        "independent_image_window_view.py",
    },
    "format_converter": {
        "conversion_progress_dialog_view.py",
        "folder_import_dialog_view.py",
        "format_converter_dialog_view.py",
    },
    "prediction": {
        "distribution_heatmap_dialog_view.py",
        "export_dialog_view.py",
        "multifile_results_widget_view.py",
        "parameter_trend_dialog_view.py",
        "prediction_page_view.py",
        "prediction_workspace_view.py",
    },
    "trainset": {
        "dataset_page_view.py",
        "model_page_view.py",
        "monitor_page_view.py",
        "page_view.py",
        "preview_page_view.py",
        "run_page_view.py",
    },
    "waxs": {
        "advanced_panel_view.py",
        "batch_panel_view.py",
        "configure_panel_view.py",
        "integration_panel_view.py",
        "page_view.py",
        "preview_panel_view.py",
        "roi_panel_view.py",
        "toolbar_view.py",
    },
}


def _owned_views() -> dict[str, set[str]]:
    actual = {
        "app": {
            path.name
            for path in (SOURCE_ROOT / "app/presentation/views").glob("*_view.py")
        }
    }
    for views_dir in (SOURCE_ROOT / "features").glob("*/presentation/views"):
        actual[views_dir.parents[1].name] = {
            path.name for path in views_dir.glob("*_view.py")
        }
    return actual


def _all_view_paths() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*_view.py")
        if "presentation" in path.parts and "views" in path.parts
    )


def test_python_view_inventory_is_explicit_and_feature_owned() -> None:
    assert _owned_views() == EXPECTED_VIEWS_BY_OWNER


def test_qt_designer_sources_and_generated_modules_are_retired() -> None:
    assert list(SOURCE_ROOT.rglob("*.ui")) == []
    assert [path for path in SOURCE_ROOT.rglob("*.py") if "_generated" in path.parts] == []
    assert not (PROJECT_ROOT / "tools/compile_ui.py").exists()


def test_python_views_do_not_import_workflows_or_external_runtimes() -> None:
    forbidden_fragments = (
        ".application",
        ".domain",
        ".infrastructure",
        "controllers",
        "core.global_params",
        "tensorflow",
        "keras",
        "bornagain",
    )
    violations: list[str] = []
    for path in _all_view_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if any(fragment in module.casefold() for fragment in forbidden_fragments):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {module}")
    assert not violations, "Python View dependency violations:\n" + "\n".join(violations)


def test_python_views_are_hand_maintained_sources() -> None:
    forbidden_markers = (
        "Form implementation generated from reading ui file",
        "Any manual changes made to this file will be lost",
    )
    for path in _all_view_paths():
        source = path.read_text(encoding="utf-8")
        assert not any(marker in source for marker in forbidden_markers), path


def test_application_shell_view_contains_hosts_not_feature_controls() -> None:
    source = (
        SOURCE_ROOT / "app/presentation/views/main_window_view.py"
    ).read_text(encoding="utf-8")
    for host in (
        "trainsetBuildPage",
        "gisaxsPredictPageHost",
        "gisaxsFittingPageHost",
        "classificationPage",
        "waxsPageHost",
    ):
        assert host in source
    assert "gisaxsPredictPredictButton" not in source
    assert "FittingManualFittingButton" not in source
