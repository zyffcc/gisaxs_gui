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


def test_legacy_presentation_does_not_load_ml_or_model_storage_runtime():
    for relative in (
        "controllers/classification_controller.py",
        "controllers/classification_workers.py",
    ):
        imports = _imports(ROOT / relative)
        assert "joblib" not in imports
        assert not any(name == "sklearn" or name.startswith("sklearn.") for name in imports)
        assert not any(name == "umap" or name.startswith("umap.") for name in imports)
    assert "core.global_params" not in _imports(
        ROOT / "controllers/classification_controller.py"
    )


def test_classification_training_worker_delegates_to_view_model():
    source = (ROOT / "controllers/classification_workers.py").read_text(encoding="utf-8")

    assert "self.view_model.train(" in source
    assert "ClassificationTrainingService" not in source
    assert "self.view_model.compute_embedding(" in source
