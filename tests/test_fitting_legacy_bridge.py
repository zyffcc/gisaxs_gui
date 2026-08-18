"""Guard the legacy fitting bridge while callers are migrated incrementally."""

from __future__ import annotations

import ast
from pathlib import Path


BINDING = (
    Path(__file__).resolve().parents[1]
    / "src/gimap/features/fitting/presentation/view_binding.py"
)


def _imports():
    tree = ast.parse(BINDING.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.casefold() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.casefold()


def test_view_binding_does_not_load_heavy_runtime_integrations():
    imported = tuple(_imports())

    assert not any(name == "bornagain" or name.startswith("bornagain.") for name in imported)
    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in imported)
    assert not any(name == "keras" or name.startswith("keras.") for name in imported)
    assert "core.global_params" not in imported
    assert "core.user_settings" not in imported
    assert not any("fitting.infrastructure" in name for name in imported)


def test_view_binding_no_longer_owns_ai_process_or_pipeline():
    source = BINDING.read_text(encoding="utf-8")
    imported = tuple(_imports())

    assert "QProcess" not in source
    assert "utils.ai_fitting_pipeline" not in imported
    assert "import fabio" not in source
    assert "csv.DictWriter" not in source
    assert "insitu_current_session.jsonl\").open" not in source
    assert "shutil.copy2(self.model_params_manager" not in source
    assert "json.dump(self._build_fitting_parameter_snapshot" not in source
    assert "gui_run.log\").open" not in source
    assert "shutil.copytree" not in source
    assert "with open(filepath, 'w'" not in source
    assert "importlib.util" not in source
    assert "from scipy" not in source
    assert "utils.fitting" not in source
    assert "utils.q_space_calculator" not in source
    assert "config.model_parameters_manager" not in source
    assert "._parameters" not in source
    assert "utils.ai_fitting_models" not in source
    assert "utils.ai_fitting_profiles" not in source


def test_view_binding_routes_scientific_work_through_view_model():
    source = BINDING.read_text(encoding="utf-8")
    imported = tuple(_imports())

    assert not any("fitting.domain" in name for name in imported)
    assert "self.compute_cut.execute(self.payload)" in source
    assert "science.insitu_cut" in source
    assert "def _sort_filter_pairs" not in source
    assert "def _interpolate_series(x, y, x_new" not in source


def test_legacy_controller_path_is_a_thin_compatibility_entry():
    path = Path(__file__).resolve().parents[1] / "controllers/fitting_controller.py"
    source = path.read_text(encoding="utf-8")

    assert "src.gimap.features.fitting.presentation.legacy_bridge" in source
    assert len(source.splitlines()) <= 25

    feature_legacy = BINDING.parent / "legacy_bridge.py"
    legacy_source = feature_legacy.read_text(encoding="utf-8")
    assert "view_binding as _implementation" in legacy_source
    assert len(legacy_source.splitlines()) <= 30


def test_dynamic_particle_stack_receives_injected_fitting_view_model():
    tree = ast.parse(BINDING.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CurrentPageHeightStackedWidget"
    ]

    assert calls
    assert all(
        any(keyword.arg == "fitting_view_model" for keyword in call.keywords)
        for call in calls
    )


def test_fitting_view_models_remain_below_architecture_review_threshold():
    root = BINDING.parent

    assert len((root / "view_model.py").read_text(encoding="utf-8").splitlines()) <= 300
    assert len(
        (root / "storage_view_model.py").read_text(encoding="utf-8").splitlines()
    ) <= 300
    assert len(
        (root / "insitu_view_model.py").read_text(encoding="utf-8").splitlines()
    ) <= 300
    assert len(
        (root / "scientific_view_model.py").read_text(encoding="utf-8").splitlines()
    ) <= 300
