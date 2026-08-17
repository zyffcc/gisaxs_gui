"""Predictor port 的隔离进程 TensorFlow adapter。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...app.jobs import JobRequest, JobRunner, JobResult
from ...features.prediction.application.ports import Predictor
from ...features.prediction.domain import ModelRuntimeInfo, PredictionRequest, PredictionResult
from ..jobs import LocalProcessJobRunner, decode_numpy_tree, encode_numpy_tree
from .errors import (
    TensorFlowIntegrationError,
    TensorFlowModelError,
    TensorFlowNotInstalledError,
    TensorFlowWorkerError,
)
from .manifest import (
    TensorFlowArtifact,
    discover_tensorflow_artifacts,
    validate_model_manifest,
)


def _runtime_info(value: dict[str, Any]) -> ModelRuntimeInfo:
    input_shape = tuple(value.get("input_shape") or ()) or None
    output_shape = tuple(value.get("output_shape") or ()) or None
    return ModelRuntimeInfo(
        artifact_path=Path(value["artifact_path"]),
        runtime_name=str(value.get("runtime_name", "tensorflow")),
        runtime_version=str(value.get("runtime_version", "")),
        input_names=tuple(str(item) for item in value.get("input_names", ())),
        output_names=tuple(str(item) for item in value.get("output_names", ())),
        input_shape=input_shape,
        output_shape=output_shape,
    )


class TensorFlowPredictor(Predictor):
    def __init__(
        self,
        runner: JobRunner | None = None,
        load_timeout_seconds: float = 120.0,
        prediction_timeout_seconds: float = 300.0,
    ):
        self.runner = runner or LocalProcessJobRunner()
        self.load_timeout_seconds = float(load_timeout_seconds)
        self.prediction_timeout_seconds = float(prediction_timeout_seconds)

    def inspect(self, model_path: Path, allow_unsafe_lambda: bool = False) -> ModelRuntimeInfo:
        failures = []
        for artifact in self._artifacts(model_path):
            result = self.runner.run(
                JobRequest(
                    handler="src.gimap.integrations.tensorflow.worker:inspect_tensorflow_model",
                    payload={
                        "artifact_path": str(artifact.path),
                        "allow_unsafe_lambda": bool(allow_unsafe_lambda),
                    },
                    timeout_seconds=self.load_timeout_seconds,
                )
            )
            try:
                value = self._require_success(result)
                runtime = _runtime_info(value)
                self._validate_runtime_contract(artifact, runtime)
                return runtime
            except TensorFlowModelError as exc:
                failures.append(f"{artifact.path}: {exc}")
        raise TensorFlowModelError(
            "No compatible TensorFlow artifact could be loaded:\n- " + "\n- ".join(failures)
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        failures = []
        for artifact in self._artifacts(request.model_path):
            result = self.runner.run(
                JobRequest(
                    handler="src.gimap.integrations.tensorflow.worker:predict_tensorflow_model",
                    payload={
                        "artifact_path": str(artifact.path),
                        "inputs": encode_numpy_tree(request.inputs),
                        "allow_unsafe_lambda": bool(request.allow_unsafe_lambda),
                        "precision_policy": request.precision_policy,
                    },
                    timeout_seconds=request.timeout_seconds or self.prediction_timeout_seconds,
                )
            )
            try:
                value = self._require_success(result)
                runtime = _runtime_info(value["runtime"])
                self._validate_runtime_contract(artifact, runtime)
                return PredictionResult(
                    outputs=decode_numpy_tree(value["outputs"]),
                    runtime=runtime,
                )
            except TensorFlowModelError as exc:
                failures.append(f"{artifact.path}: {exc}")
        raise TensorFlowModelError(
            "No compatible TensorFlow artifact could run prediction:\n- " + "\n- ".join(failures)
        )

    @staticmethod
    def _artifacts(model_path: Path) -> tuple[TensorFlowArtifact, ...]:
        artifacts = discover_tensorflow_artifacts(model_path)
        if not artifacts:
            raise TensorFlowModelError(
                f"No .keras or SavedModel artifact was found under {Path(model_path).expanduser()}."
            )
        for artifact in artifacts:
            validate_model_manifest(artifact)
        return artifacts

    @staticmethod
    def _validate_runtime_contract(
        artifact: TensorFlowArtifact,
        runtime: ModelRuntimeInfo,
    ) -> None:
        required_inputs = {str(item) for item in artifact.manifest.get("required_inputs", ())}
        required_outputs = {str(item) for item in artifact.manifest.get("required_outputs", ())}
        missing_inputs = required_inputs - set(runtime.input_names)
        missing_outputs = required_outputs - set(runtime.output_names)
        if missing_inputs:
            raise TensorFlowModelError(
                "TensorFlow model is missing required inputs: " + ", ".join(sorted(missing_inputs))
            )
        if missing_outputs:
            raise TensorFlowModelError(
                "TensorFlow model is missing required outputs: " + ", ".join(sorted(missing_outputs))
            )

    @staticmethod
    def _require_success(result: JobResult) -> Any:
        if result.succeeded:
            return result.value
        error = result.error
        message = error.message if error is not None else "Unknown TensorFlow worker failure."
        exception_type = error.exception_type if error is not None else ""
        if exception_type == "TensorFlowNotInstalledError":
            raise TensorFlowNotInstalledError(message)
        if exception_type == "TensorFlowModelError":
            raise TensorFlowModelError(message)
        if result.status in {"cancelled", "timed_out"}:
            raise TensorFlowWorkerError(message)
        if error is not None and error.code == "worker_crash":
            raise TensorFlowWorkerError(message)
        raise TensorFlowIntegrationError(message)


class NumpyTensorValue:
    """兼容旧调用方 `.numpy()` 的轻量 NumPy 包装。"""

    def __init__(self, value: Any):
        self._value = np.asarray(value)

    def numpy(self) -> np.ndarray:
        return self._value

    def __array__(self, dtype=None, copy=None):
        value = self._value if dtype is None else self._value.astype(dtype, copy=False)
        if copy:
            value = value.copy()
        return value


def _wrap_tensor_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _wrap_tensor_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wrap_tensor_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return NumpyTensorValue(value)
    return value


class TensorFlowModelProxy:
    """旧调用方可持有的代理；不在当前进程保存 TensorFlow 对象。"""

    def __init__(
        self,
        model_path: Path,
        predictor: TensorFlowPredictor | None = None,
        allow_unsafe_lambda: bool = False,
        precision_policy: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.predictor = predictor or TensorFlowPredictor()
        self.allow_unsafe_lambda = bool(allow_unsafe_lambda)
        self.precision_policy = precision_policy
        self.runtime = self.predictor.inspect(
            self.model_path,
            allow_unsafe_lambda=self.allow_unsafe_lambda,
        )
        self.model_path = self.runtime.artifact_path
        self.input_names = list(self.runtime.input_names)
        self.output_names = list(self.runtime.output_names)
        self.input_shape = self.runtime.input_shape
        self.output_shape = self.runtime.output_shape
        self.artifact_path = self.runtime.artifact_path

    def predict(self, inputs: Any, verbose: int = 0) -> Any:
        del verbose
        result = self.predictor.predict(
            PredictionRequest(
                model_path=self.model_path,
                inputs=inputs,
                allow_unsafe_lambda=self.allow_unsafe_lambda,
                precision_policy=self.precision_policy,
            )
        )
        return result.outputs

    def __call__(self, inputs: Any, training: bool = False) -> Any:
        del training
        return _wrap_tensor_values(self.predict(inputs))


def create_tensorflow_model_proxy(
    model_path: Path,
    *,
    predictor: TensorFlowPredictor | None = None,
    allow_unsafe_lambda: bool = False,
    precision_policy: str | None = None,
) -> TensorFlowModelProxy:
    return TensorFlowModelProxy(
        model_path=model_path,
        predictor=predictor,
        allow_unsafe_lambda=allow_unsafe_lambda,
        precision_policy=precision_policy,
    )
