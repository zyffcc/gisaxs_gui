from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.gimap.app.jobs import JobError, JobResult
from src.gimap.features.prediction.application import RunPrediction
from src.gimap.features.prediction.domain import (
    ModelRuntimeInfo,
    PredictionRequest,
    PredictionResult,
)
from src.gimap.integrations.jobs import encode_numpy_tree
from src.gimap.integrations.tensorflow import (
    TensorFlowModelProxy,
    TensorFlowModelError,
    TensorFlowPredictor,
    TensorFlowWorkerError,
    discover_tensorflow_artifacts,
    validate_model_manifest,
)


class FakePredictor:
    def inspect(self, model_path: Path, allow_unsafe_lambda: bool = False):
        del allow_unsafe_lambda
        return ModelRuntimeInfo(model_path, "fake", input_names=("image",))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        return PredictionResult(outputs=np.asarray(request.inputs) * 2)


class RecordingRunner:
    def __init__(self, crash: bool = False):
        self.crash = crash
        self.requests = []

    def run(self, request, on_progress=None):
        del on_progress
        self.requests.append(request)
        if self.crash:
            return JobResult(
                job_id=request.job_id,
                status="failed",
                error=JobError(code="worker_crash", message="native worker exited"),
            )
        runtime = {
            "artifact_path": request.payload["artifact_path"],
            "runtime_name": "tensorflow:test",
            "runtime_version": "test",
            "input_names": ["image"],
            "output_names": ["score"],
            "input_shape": [None, 2],
            "output_shape": [None, 1],
        }
        value = runtime
        if request.handler.endswith(":predict_tensorflow_model"):
            value = {
                "runtime": runtime,
                "outputs": encode_numpy_tree({"score": np.array([[0.75]], dtype=np.float32)}),
            }
        return JobResult(job_id=request.job_id, status="succeeded", value=value)

    def cancel(self, job_id):
        del job_id
        return False

    def shutdown(self):
        return None


def _fake_keras_model(tmp_path: Path, manifest: dict | None = None) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.keras").write_bytes(b"not-a-real-model")
    if manifest is not None:
        (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model_dir


def test_prediction_application_uses_fake_predictor_without_tensorflow_or_qapplication():
    use_case = RunPrediction(FakePredictor())
    result = use_case.execute(
        PredictionRequest(model_path=Path("unused.keras"), inputs=np.array([1.0, 2.0]))
    )
    np.testing.assert_array_equal(result.outputs, np.array([2.0, 4.0]))


def test_legacy_model_proxy_exposes_numpy_tensor_surface():
    model = TensorFlowModelProxy(Path("unused.keras"), predictor=FakePredictor())
    outputs = model(np.array([1.0, 2.0]), training=False)
    np.testing.assert_array_equal(outputs.numpy(), np.array([2.0, 4.0]))


def test_manifest_discovery_and_validation_do_not_import_tensorflow(tmp_path):
    tensorflow_was_loaded = "tensorflow" in sys.modules
    model_dir = _fake_keras_model(tmp_path, {"required_inputs": ["image"]})
    artifact = discover_tensorflow_artifacts(model_dir)[0]
    validate_model_manifest(artifact)
    assert artifact.path.name == "model.keras"
    assert ("tensorflow" in sys.modules) is tensorflow_was_loaded


def test_invalid_manifest_is_reported_before_runtime_loading(tmp_path):
    model_dir = _fake_keras_model(tmp_path, {"required_inputs": "image"})
    artifact = discover_tensorflow_artifacts(model_dir)[0]
    with pytest.raises(TensorFlowModelError, match="required_inputs"):
        validate_model_manifest(artifact)


def test_tensorflow_predictor_transfers_only_serializable_worker_data(tmp_path):
    model_dir = _fake_keras_model(
        tmp_path,
        {"required_inputs": ["image"], "required_outputs": ["score"]},
    )
    runner = RecordingRunner()
    predictor = TensorFlowPredictor(runner=runner)
    runtime = predictor.inspect(model_dir)
    result = predictor.predict(
        PredictionRequest(model_path=model_dir, inputs=np.ones((1, 2), dtype=np.float32))
    )
    assert runtime.input_shape == (None, 2)
    np.testing.assert_array_equal(result.outputs["score"], np.array([[0.75]], dtype=np.float32))
    json.dumps(runner.requests[-1].payload, allow_nan=False)


def test_tensorflow_worker_crash_becomes_adapter_error(tmp_path):
    model_dir = _fake_keras_model(tmp_path)
    predictor = TensorFlowPredictor(runner=RecordingRunner(crash=True))
    with pytest.raises(TensorFlowWorkerError, match="native worker exited"):
        predictor.inspect(model_dir)


def test_existing_keras_model_is_accepted_by_isolated_worker(tmp_path):
    model_dir = (
        Path(__file__).resolve().parents[1]
        / "modules"
        / "Fitting_1D_Model"
        / "k1_k2_k3_k4_phys"
    )
    if not (model_dir / "model.keras").is_file():
        pytest.skip("Repository TensorFlow compatibility artifact is not available.")
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow runtime is not installed.")
    predictor = TensorFlowPredictor(load_timeout_seconds=60)
    runtime = predictor.inspect(
        model_dir,
        allow_unsafe_lambda=True,
    )
    assert runtime.runtime_version
    assert "x" in runtime.input_names
    assert "exist_logit" in runtime.output_names

    inputs = {
        "x": np.zeros((1, 1000, 3), dtype=np.float32),
        "point_mask": np.ones((1, 1000), dtype=bool),
        "global_features": np.zeros((1, 5), dtype=np.float32),
        "type_allowed": np.ones((1, 4, 4), dtype=np.float32),
        "param_low_norm": np.zeros((1, 4, 4, 6), dtype=np.float32),
        "param_high_norm": np.ones((1, 4, 4, 6), dtype=np.float32),
        "param_range_mask": np.zeros((1, 4, 4, 6), dtype=np.float32),
        "force_exist": np.full((1, 4), -1.0, dtype=np.float32),
        "global_low_norm": np.zeros((1, 5), dtype=np.float32),
        "global_high_norm": np.ones((1, 5), dtype=np.float32),
        "global_range_mask": np.zeros((1, 5), dtype=np.float32),
        "d_allowed": np.ones((1, 4, 2), dtype=np.float32),
        "d_spacing_rule": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    }
    isolated = predictor.predict(
        PredictionRequest(
            model_path=runtime.artifact_path,
            inputs=inputs,
            allow_unsafe_lambda=True,
            timeout_seconds=60,
        )
    ).outputs

    input_path = tmp_path / "direct-inputs.npz"
    output_path = tmp_path / "direct-outputs.npz"
    np.savez(input_path, **inputs)
    direct_script = """
import sys
import numpy as np
import tensorflow as tf

model = tf.saved_model.load(sys.argv[1])
function = model.signatures["serving_default"]
with np.load(sys.argv[2]) as archive:
    values = {key: archive[key] for key in archive.files}
outputs = {key: value.numpy() for key, value in function(**values).items()}
np.savez(sys.argv[3], **outputs)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            direct_script,
            str(runtime.artifact_path),
            str(input_path),
            str(output_path),
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path) as archive:
        direct = {key: archive[key] for key in archive.files}
    assert set(direct).issubset(isolated)
    for key, expected in direct.items():
        np.testing.assert_allclose(isolated[key], expected, rtol=1e-6, atol=1e-6)


def test_importing_main_does_not_import_native_runtimes():
    script = (
        "import sys; import main; "
        "assert 'tensorflow' not in sys.modules; "
        "assert 'bornagain' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
