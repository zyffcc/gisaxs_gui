import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_classification_presentation_does_not_load_ml_or_model_storage_runtime():
    for relative in (
        "src/gimap/features/classification/presentation/view_binding.py",
        "src/gimap/features/classification/presentation/workers.py",
    ):
        imports = _imports(ROOT / relative)
        assert "joblib" not in imports
        assert not any(name == "sklearn" or name.startswith("sklearn.") for name in imports)
        assert not any(name == "umap" or name.startswith("umap.") for name in imports)
    assert "core.global_params" not in _imports(
        ROOT / "src/gimap/features/classification/presentation/view_binding.py"
    )


def test_classification_training_worker_delegates_to_view_model():
    source = (
        ROOT / "src/gimap/features/classification/presentation/workers.py"
    ).read_text(encoding="utf-8")

    assert "self.view_model.train(" in source
    assert "ClassificationTrainingService" not in source
    assert "self.view_model.compute_embedding(" in source


def test_feature_infrastructure_does_not_import_legacy_controllers():
    infrastructure = ROOT / "src/gimap/features/classification/infrastructure"
    for path in infrastructure.rglob("*.py"):
        assert not any(name.startswith("controllers") for name in _imports(path)), path


