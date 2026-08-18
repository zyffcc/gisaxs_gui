"""Trainset TensorFlow model-contract adapter regression tests。"""

from __future__ import annotations

import sys

from src.gimap.app.jobs import JobError, JobResult
from src.gimap.features.trainset.application import ModelContractRequest
from src.gimap.features.trainset.infrastructure.adapters import (
    TensorFlowModelContractAdapter,
)


class RecordingRunner:
    def __init__(self, result):
        self.result = result
        self.request = None

    def run(self, request, on_progress=None):
        self.request = request
        return self.result


def _request() -> ModelContractRequest:
    return ModelContractRequest(
        input_shape=(32, 32, 1),
        output_size=2,
        model_config={"layers": [{"type": "flatten"}]},
    )


def test_adapter_sends_serializable_contract_to_worker_without_importing_tensorflow():
    tensorflow_was_loaded = "tensorflow" in sys.modules
    runner = RecordingRunner(
        JobResult(
            job_id="model-contract",
            status="succeeded",
            value={"output_shape": [1, 2], "trainable_weights": 2050},
        )
    )

    result = TensorFlowModelContractAdapter(runner).validate(_request())

    assert result.output_shape == (1, 2)
    assert result.trainable_weights == 2050
    assert runner.request.payload["input_shape"] == [32, 32, 1]
    assert runner.request.handler.endswith(":inspect_trainset_model_contract")
    assert ("tensorflow" in sys.modules) is tensorflow_was_loaded


def test_adapter_reports_worker_failure_as_runtime_unavailable():
    runner = RecordingRunner(
        JobResult(
            job_id="model-contract",
            status="failed",
            error=JobError(
                code="handler_error",
                message="TensorFlow is not installed",
            ),
        )
    )

    result = TensorFlowModelContractAdapter(runner).validate(_request())

    assert result.output_shape is None
    assert result.runtime_error == "TensorFlow is not installed"
    assert "Output (2,) regression parameters" in result.static_summary
