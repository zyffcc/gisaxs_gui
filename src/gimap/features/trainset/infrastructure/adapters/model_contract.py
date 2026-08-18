"""通过 JobRunner 隔离 TensorFlow 的 Trainset 模型验证 adapter。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .....app.jobs import JobRequest, JobRunner
from ...application.models import ModelContractRequest, ModelContractResult
from ...domain import normalized_layers, static_contract
from .keras_modeling import build_keras_model


def inspect_trainset_model_contract(payload, report, is_cancelled):
    """Worker entry point；TensorFlow 只在隔离进程内导入。"""
    if is_cancelled():
        raise RuntimeError("Model contract validation was cancelled.")
    import tensorflow as tf

    input_shape = tuple(int(value) for value in payload["input_shape"])
    output_size = int(payload["output_size"])
    model = build_keras_model(tf, input_shape, output_size, payload["model_config"])
    result = model(
        np.zeros((1, *input_shape), dtype=np.float32),
        training=False,
    )
    report(1, 1, "TensorFlow forward pass complete")
    return {
        "output_shape": [int(value) for value in result.shape],
        "trainable_weights": int(model.count_params()),
    }


class TensorFlowModelContractAdapter:
    def __init__(self, runner: JobRunner, *, timeout_seconds: float = 60.0):
        self._runner = runner
        self._timeout_seconds = timeout_seconds

    def validate(self, request: ModelContractRequest) -> ModelContractResult:
        summary = static_contract(
            request.input_shape,
            request.output_size,
            normalized_layers(request.model_config),
        )
        result = self._runner.run(
            JobRequest(
                handler=(
                    "src.gimap.features.trainset.infrastructure.adapters.model_contract:"
                    "inspect_trainset_model_contract"
                ),
                payload={
                    "input_shape": list(request.input_shape),
                    "output_size": request.output_size,
                    "model_config": request.model_config,
                },
                timeout_seconds=self._timeout_seconds,
            )
        )
        if not result.succeeded:
            message = result.error.message if result.error is not None else result.status
            return ModelContractResult(static_summary=summary, runtime_error=message)
        value: dict[str, Any] = result.value
        return ModelContractResult(
            static_summary=summary,
            output_shape=tuple(int(item) for item in value["output_shape"]),
            trainable_weights=int(value["trainable_weights"]),
        )
