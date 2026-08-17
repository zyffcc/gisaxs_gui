"""Classification ML ports backed by the shared JobRunner。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.gimap.app.jobs import JobRequest

from ...application.models import EmbeddingResult
from .job_serialization import decode_array, deserialize_experiment, encode_array


def _sample_payload(sample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "file_path": sample.file_path,
        "file_name": sample.file_name,
        "label": sample.label,
        "data_type": sample.data_type,
        "raw_shape": sample.raw_shape,
        "included": sample.included,
        "load_status": sample.load_status,
        "qc_status": sample.qc_status,
        "qc_messages": list(sample.qc_messages),
    }


class JobRunnerClassifierTrainer:
    def __init__(self, runner, artifact_root: Path | None = None):
        self.runner = runner
        self.artifact_root = Path(artifact_root or ".gimap_cache/classification_jobs")
        self._active_job_id = None

    def train(self, request, *, on_progress=None):
        if request.feature_matrix.y is None:
            raise ValueError("Classification training requires labels")
        job_id = uuid4().hex
        artifact_dir = self.artifact_root / job_id
        job = JobRequest(
            handler="src.gimap.features.classification.infrastructure.workers:train_classifiers_job",
            payload={
                "X": encode_array(request.feature_matrix.X),
                "y": encode_array(request.feature_matrix.y),
                "samples": [_sample_payload(item) for item in request.feature_matrix.samples],
                "algorithms": [asdict(item) for item in request.algorithms],
                "validation": asdict(request.validation),
                "projection": asdict(request.projection),
                "ranking_metric": request.ranking_metric,
                "artifact_dir": str(artifact_dir),
            },
            timeout_seconds=request.timeout_seconds,
            job_id=job_id,
        )
        self._active_job_id = job_id

        def progress(value):
            if on_progress is not None:
                on_progress(int(value.completed), int(value.total), value.message)

        try:
            result = self.runner.run(job, on_progress=progress)
        finally:
            self._active_job_id = None
        if not result.succeeded:
            raise RuntimeError(result.error.message if result.error else result.status)
        return deserialize_experiment(result.value)

    def cancel(self) -> bool:
        return bool(
            self._active_job_id and self.runner.cancel(self._active_job_id)
        )


class JobRunnerEmbeddingAdapter:
    def __init__(self, runner):
        self.runner = runner
        self._active_job_id = None

    def embed(self, request):
        job = JobRequest(
            handler="src.gimap.features.classification.infrastructure.workers:classification_embedding_job",
            payload={"X": encode_array(request.values), "method": request.method},
            timeout_seconds=request.timeout_seconds,
        )
        self._active_job_id = job.job_id
        try:
            result = self.runner.run(job)
        finally:
            self._active_job_id = None
        if not result.succeeded:
            raise RuntimeError(result.error.message if result.error else result.status)
        return EmbeddingResult(decode_array(result.value["values"]), result.value["method"])

    def cancel(self) -> bool:
        return bool(
            self._active_job_id and self.runner.cancel(self._active_job_id)
        )
