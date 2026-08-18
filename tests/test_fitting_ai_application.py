import json
from pathlib import Path

import numpy as np
import pytest

from src.gimap.app.jobs import JobProgress, JobRequest, JobResult
from src.gimap.features.fitting.application import (
    CandidateGenerationRequest,
    CandidateGenerationResult,
    GenerateCandidates,
    LoadCandidateResults,
    ManageAiFittingArtifacts,
    MapCandidateParameters,
    RefineCandidates,
    ReviewCandidates,
)
from src.gimap.features.fitting.domain import prepare_ai_curve
from src.gimap.features.fitting.infrastructure.adapters import (
    AiPipelinePredictor,
    JsonCandidateRepository,
    LocalAiFittingArtifactRepository,
)
from src.gimap.integrations.jobs import LocalProcessJobRunner
from utils.ai_fitting_pipeline import FittingPipeline
from utils.ai_fitting_profiles import profile_registry


class _FakePredictor:
    def __init__(self):
        self.request = None

    def create_job_request(self, request):
        self.request = request
        return JobRequest("tests.test_job_system:successful_handler", {"kind": "ai"})

    def decode_result(self, request, value):
        return CandidateGenerationResult(
            output_dir=request.output_dir,
            profile_name=request.profile["name"],
            runtime_seconds=float(value["runtime_seconds"]),
            configured_candidates=int(request.profile["candidate_count"]),
            candidates=tuple(value["candidates"]),
            best_log_rmse=float(value["candidates"][0]["best_log_rmse"]),
            exit_code=0,
        )


class _FakeJobRunner:
    def __init__(self):
        self.cancelled = []

    def run(self, request, on_progress=None):
        if on_progress is not None:
            on_progress(JobProgress(request.job_id, 1, 2, "Progress 1/2"))
            on_progress(JobProgress(request.job_id, 2, 2, "Progress 2/2"))
        return JobResult(
            request.job_id,
            "succeeded",
            value={
                "runtime_seconds": 0.25,
                "candidates": [{"rank": 1, "best_log_rmse": 0.12}],
            },
        )

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return True

    def shutdown(self):
        return None


def _request(tmp_path, profile_name="Balanced"):
    profile = profile_registry.get(profile_name)
    return CandidateGenerationRequest(
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        q=np.linspace(0.01, 0.2, 20),
        intensity=np.linspace(100.0, 20.0, 20),
        sigma=np.ones(20),
        profile=profile.to_dict(),
        constraints={"mode": "Fixed K", "exact_nonempty": 2},
        exact_nonempty=2,
    )


def test_candidate_generation_uses_predictor_port_job_runner_and_fixed_seed(tmp_path):
    predictor = _FakePredictor()
    runner = _FakeJobRunner()
    progress = []

    result = GenerateCandidates(predictor, runner).execute(
        _request(tmp_path),
        on_progress=progress.append,
    )

    assert result.profile_name == "Balanced"
    assert result.candidates[0]["best_log_rmse"] == pytest.approx(0.12)
    assert predictor.request.profile["random_seed"] == 123
    assert predictor.request.constraints["exact_nonempty"] == 2
    assert [item.fraction for item in progress] == [0.5, 1.0]


def test_refinement_keeps_profile_and_rejects_fast_zero_refinement(tmp_path):
    generation = GenerateCandidates(_FakePredictor(), _FakeJobRunner())
    refinement = RefineCandidates(generation)

    result = refinement.execute(_request(tmp_path, "Balanced"))

    assert result.configured_candidates == 192
    with pytest.raises(ValueError, match="refinement_count"):
        refinement.execute(_request(tmp_path, "Fast"))


def test_ai_curve_preparation_preserves_negative_axis_and_sigma_fallback():
    q = -np.linspace(0.01, 0.2, 20)
    intensity = np.linspace(20.0, 100.0, 20)

    curve = prepare_ai_curve(q, intensity, axis_filter="negative")

    assert np.all(curve.q > 0)
    assert np.all(np.diff(curve.q) >= 0)
    np.testing.assert_allclose(curve.sigma, np.maximum(0.05 * curve.intensity, 1e-30))


def test_candidate_review_preserves_rank_and_reports_physical_violation():
    rows = [
        {
            "rank": 2,
            "best_log_rmse": 0.2,
            "components": [{"type": "sphere", "params": {"R": 10.0, "D": 15.0}}],
        },
        {
            "rank": 1,
            "best_log_rmse": 0.1,
            "components": [{"type": "sphere", "params": {"R": 10.0, "D": 30.0}}],
        },
    ]

    reviewed = ReviewCandidates().execute(rows)

    assert [row["rank"] for row in reviewed] == [1, 2]
    assert reviewed[0]["constraint_violations"] == []
    assert any("must be" in message for message in reviewed[1]["constraint_violations"])


def test_candidate_parameter_mapping_is_framework_neutral():
    mapping = MapCandidateParameters().execute(
        {
            "components": [
                {
                    "type": "vertical_cylinder",
                    "weight": 0.75,
                    "params": {"R": 12.0, "sigma_R": 0.2, "D": 30.0},
                }
            ],
            "global_params": {"BG": 0.01, "sigma_Res": 0.02, "k": 3.0},
        }
    )

    assert mapping.components[0].shape == "Vertical Cylinder"
    assert mapping.components[0].parameters["radius"] == pytest.approx(12.0)
    assert mapping.global_parameters == {
        "background": 0.01,
        "sigma_res": 0.02,
        "k_value": 3.0,
    }


def test_ai_pipeline_adapter_runs_in_job_process_without_tensorflow(tmp_path, monkeypatch):
    script = tmp_path / "fake_predict.py"
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(add_help=False)\n"
        "p.add_argument('--output_dir', required=True)\n"
        "args,_=p.parse_known_args()\n"
        "out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
        "rows=[{'rank': 1, 'best_log_rmse': 0.05, 'components': []}]\n"
        "(out/'top20_candidates.json').write_text(json.dumps(rows), encoding='utf-8')\n"
        "print('Progress 1/1')\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.keras").write_bytes(b"placeholder")
    import src.gimap.features.fitting.infrastructure.adapters.ai_pipeline as adapter_module

    monkeypatch.setattr(adapter_module, "fitting_pipeline", FittingPipeline(script))
    runner = LocalProcessJobRunner()
    try:
        result = GenerateCandidates(AiPipelinePredictor(), runner).execute(
            CandidateGenerationRequest(
                model_path=model,
                output_dir=tmp_path / "result",
                q=np.linspace(0.01, 0.2, 20),
                intensity=np.linspace(10.0, 1.0, 20),
                sigma=np.ones(20),
                profile=profile_registry.get("Fast").to_dict(),
            )
        )
    finally:
        runner.shutdown()

    assert result.exit_code == 0
    assert result.candidates[0]["best_log_rmse"] == pytest.approx(0.05)


def test_ai_pipeline_adapter_owns_guarded_reusable_output_cleanup(tmp_path):
    output = tmp_path / "current_prediction"
    output.mkdir()
    (output / "stale.txt").write_text("old", encoding="utf-8")
    request = _request(tmp_path)
    request.model_path.mkdir()
    (request.model_path / "model.keras").write_bytes(b"placeholder")
    request = CandidateGenerationRequest(
        model_path=request.model_path,
        output_dir=output,
        q=request.q,
        intensity=request.intensity,
        sigma=request.sigma,
        profile=request.profile,
        clear_output_dir=True,
    )

    AiPipelinePredictor().create_job_request(request)

    assert not (output / "stale.txt").exists()
    assert (output / "input_curve.csv").is_file()

    unsafe = CandidateGenerationRequest(
        model_path=request.model_path,
        output_dir=tmp_path / "arbitrary-output",
        q=request.q,
        intensity=request.intensity,
        sigma=request.sigma,
        profile=request.profile,
        clear_output_dir=True,
    )
    with pytest.raises(ValueError, match="current_prediction"):
        AiPipelinePredictor().create_job_request(unsafe)


def test_load_candidate_results_uses_json_repository_without_gui(tmp_path):
    output = tmp_path / "prediction"
    output.mkdir()
    rows = [{"rank": 1, "best_log_rmse": 0.03}]
    (output / "top20_candidates.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    loaded = LoadCandidateResults(JsonCandidateRepository()).execute(output)

    assert loaded == tuple(rows)


def test_ai_artifact_use_case_preserves_log_and_unique_export_contract(tmp_path):
    source = tmp_path / "current_prediction"
    source.mkdir()
    (source / "top20_candidates.json").write_text("[]", encoding="utf-8")
    artifacts = ManageAiFittingArtifacts(LocalAiFittingArtifactRepository())

    log = artifacts.append_log(source, "Progress 1/2")
    first = artifacts.export_output(source, tmp_path, "20260817_120000")
    second = artifacts.export_output(source, tmp_path, "20260817_120000")

    assert log.read_text(encoding="utf-8") == "Progress 1/2\n"
    assert first.name == "ai_prediction_20260817_120000"
    assert second.name == "ai_prediction_20260817_120000_1"
    assert (first / "top20_candidates.json").is_file()

    with pytest.raises(ValueError, match="outside"):
        artifacts.export_output(source, source / "nested", "stamp")
