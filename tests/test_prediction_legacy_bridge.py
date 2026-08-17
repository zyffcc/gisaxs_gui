import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "controllers" / "gisaxs_predict_controller.py"


def _imports(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_prediction_controller_is_a_legacy_presentation_bridge_only():
    source = CONTROLLER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = _imports(tree)

    assert "core.global_params" not in imports
    assert "controllers.fitting_controller" not in imports
    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in imports)
    assert not any(name == "keras" or name.startswith("keras.") for name in imports)
    assert "PredictionViewModel" in source
    assert "PredictionImageLoader" in source


def test_main_controller_injects_prediction_view_model():
    source = (ROOT / "controllers" / "main_controller.py").read_text(encoding="utf-8")

    assert "create_prediction_view_model(self.app_context)" in source
    assert "prediction_view_model=prediction_view_model" in source
