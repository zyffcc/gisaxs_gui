import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gimap.features.prediction.presentation.legacy_bridge import (
    GisaxsPredictController,
)
from src.gimap.features.prediction.presentation.view_binding import (
    PredictionViewBinding,
)


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "src/gimap/features/prediction/presentation/view_binding.py"


def _imports(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_prediction_view_binding_has_only_presentation_dependencies():
    source = BINDING.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = _imports(tree)

    assert "core.global_params" not in imports
    assert "controllers.fitting_controller" not in imports
    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in imports)
    assert not any(name == "keras" or name.startswith("keras.") for name in imports)
    assert "PredictionViewModel" in source
    assert "PredictionImageLoader" in source
    assert "os.listdir" not in source
    assert "_extract_spec_from_dict" not in source
    assert "_extract_spec_fallback" not in source
    assert "discover_numbered_files" in source
    assert "discover_files" in source
    assert "exec_module" not in source
    assert "utils.tools.Preprocessing" not in source
    assert ".predict(inp" not in source
    assert "np.load(" not in source
    assert "np.savetxt(" not in source
    assert not any("prediction.infrastructure" in name for name in imports)


def test_prediction_bridge_delegates_preprocessing_to_view_model():
    image = np.ones((4, 4), dtype=np.float32)
    typed_module = object()
    prepared = SimpleNamespace(
        values=np.ones((1, 4, 4, 1), dtype=np.float32),
        steps=({"label": "normalized"},),
    )
    calls = []
    messages = []
    bridge = SimpleNamespace(
        _current_module={"_prediction_module": typed_module},
        _modules_by_name={},
        current_parameters={},
        prediction_view_model=SimpleNamespace(
            state=SimpleNamespace(error_message=None),
            prepare_input=lambda values, module: calls.append((values, module)) or prepared,
        ),
        _append_status_message=lambda message, **kwargs: messages.append((message, kwargs)),
    )

    result = PredictionViewBinding._preprocess_for_module(bridge, image)

    assert result is prepared.values
    assert calls == [(image, typed_module)]
    assert bridge._latest_preprocess_steps == list(prepared.steps)
    assert messages[-1][0] == "Module preprocess output shape (1, 4, 4, 1)"


def test_prediction_bridge_delegates_prediction_to_view_model():
    typed_module = object()
    prepared = np.ones((1, 4, 4, 1), dtype=np.float32)
    outputs = {"parameters": np.array([0.25], dtype=np.float32)}
    calls = []
    bridge = SimpleNamespace(
        _current_model=object(),
        _current_module={"_prediction_module": typed_module},
        current_parameters={"module_model_path": "/tmp/model.keras"},
        _latest_preprocess_steps=({"label": "normalized"},),
        prediction_view_model=SimpleNamespace(
            state=SimpleNamespace(error_message=None),
            predict_prepared=lambda *args: calls.append(args)
            or SimpleNamespace(outputs=outputs),
        ),
        _append_status_message=lambda *args, **kwargs: None,
    )

    result = PredictionViewBinding._predict_with_current_model(bridge, prepared)

    assert result == outputs
    assert calls == [
        (
            prepared,
            typed_module,
            Path("/tmp/model.keras"),
            bridge._latest_preprocess_steps,
        )
    ]


def test_application_runtime_injects_prediction_view_model():
    source = (ROOT / "src/gimap/app/runtime.py").read_text(encoding="utf-8")

    assert "create_prediction_view_model(self.app_context)" in source
    assert "prediction_view_model=create_prediction_view_model(self.app_context)" in source


def test_legacy_prediction_controller_path_is_a_thin_reexport():
    assert GisaxsPredictController is PredictionViewBinding
    source = (ROOT / "controllers/gisaxs_predict_controller.py").read_text(encoding="utf-8")

    assert "src.gimap.features.prediction.presentation.legacy_bridge" in source
    assert len(source.splitlines()) <= 9

    feature_legacy = (
        ROOT / "src/gimap/features/prediction/presentation/legacy_bridge.py"
    ).read_text(encoding="utf-8")
    assert "from .view_binding import PredictionViewBinding" in feature_legacy
    assert len(feature_legacy.splitlines()) <= 8
