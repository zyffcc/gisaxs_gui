"""Replay the previous Cut results after relocation, then exercise the real job runner.

Run with the supported TensorFlow 2.15 CPU environment from the repository root.
This checks integration parity, not new scientific generalization.
"""

from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import (
    WorkflowEngine,
    prepare_sides,
    validate_options,
    read_curve,
    bundled_workflow,
    write_side,
)


def main():
    fixtures = bundled_workflow().parent / "development/fixtures"
    output = ROOT / "validation/workflow_v5"
    output.mkdir(parents=True, exist_ok=True)
    q, y, _ = read_curve(fixtures / "Cut_Data.txt")
    sigma = np.hypot(0.1 * abs(y), 50.0)
    engine = WorkflowEngine()
    evidence = []
    for side, components, sr, nu in (
        ("positive", [2, 2], 0.012953743397082464, 5.000743985235129),
        ("negative", [1, 2, 2], 0.012999919415803547, 5.000483841441652),
    ):
        options = validate_options(
            dict(method="model", side=side, components=components, sigma_res=sr, nu_res=nu, numerical=False)
        )
        item = prepare_sides(q, y, sigma, options)[0]
        start = time.perf_counter()
        result, artifact, raw = engine.fit_side(item, options, lambda *a: None, lambda: False)
        rows = write_side(output / side, item, result, artifact, raw)
        reference = json.loads((fixtures / f"{side}_reference.json").read_text())
        actual = result["solutions"][0]["positive_observation_logrmse"]
        expected = reference["solutions"][0]["positive_observation_logrmse"]
        assert abs(actual - expected) < 1e-5, (side, actual, expected)
        assert result["raw_candidate_count"] == 18
        assert len(rows[0]["display_q"]) == 500
        assert np.array_equal(rows[0]["observed"], item["observed"])
        evidence.append(
            dict(
                side=side,
                actual_logrmse=actual,
                reference_logrmse=expected,
                seconds=time.perf_counter() - start,
                raw_candidates=18,
            )
        )
        print(evidence[-1], flush=True)
    # A partial resolution prior must constrain every returned candidate, including
    # discovery seeds (unconstrained seeds are deliberately not returned).
    options = validate_options(
        dict(method="model", side="positive", components=[2, 2], sigma_res=0.01, numerical=False)
    )
    item = prepare_sides(q, y, sigma, options)[0]
    partial, _, _ = engine.fit_side(item, options, lambda *a: None, lambda: False)
    assert all(
        s["global_parameters"]["sigma_Res"] is None
        or abs(s["global_parameters"]["sigma_Res"] - 0.01) < 1e-8
        for s in partial["solutions"]
    )
    from src.gimap.app.jobs import JobRequest
    from src.gimap.integrations.jobs import LocalProcessJobRunner

    # Real spawned process, batch failure isolation, automatic conditions and both signs.
    result = LocalProcessJobRunner().run(
        JobRequest(
            handler="src.gimap.features.fitting.infrastructure.adapters.workflow_v5:run_workflow_job",
            payload=dict(
                files=[str(fixtures / "missing.txt"), str(fixtures / "Cut_Data.txt")],
                output_dir=str(output / "automatic_batch"),
                options=dict(
                    method="model",
                    numerical=True,
                    search_combinations=3,
                    condition_combinations=1,
                    absolute_noise=50.0,
                ),
            ),
            timeout_seconds=600,
        ),
        on_progress=lambda p: print(p.message, flush=True),
    )
    if not result.succeeded:
        raise RuntimeError(result.error)
    assert [r["status"] for r in result.value["records"]] == ["failed", "complete"]
    assert {r["side"] for r in result.value["candidates"]} == {"positive", "negative"}
    assert any(v < 0 for r in result.value["candidates"] for v in r["observed"])
    runner = LocalProcessJobRunner()
    cancellation_request = JobRequest(
        handler="src.gimap.features.fitting.infrastructure.adapters.workflow_v5:run_workflow_job",
        payload=dict(
            files=[str(fixtures / "Cut_Data.txt")],
            output_dir=str(output / "cancelled_batch"),
            options=dict(method="model", numerical=True),
        ),
        timeout_seconds=120,
    )
    cancelled = runner.run(
        cancellation_request, on_progress=lambda _p: runner.cancel(cancellation_request.job_id)
    )
    assert cancelled.status == "cancelled", cancelled
    payload = dict(
        passed=True,
        parity=evidence,
        batch=result.value["records"],
        partial_resolution_prior_preserved=True,
        process_cancelled=True,
        note="Integration and old Cut replay only; automatic conditions remain experimental.",
    )
    (output / "VERIFIED.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
