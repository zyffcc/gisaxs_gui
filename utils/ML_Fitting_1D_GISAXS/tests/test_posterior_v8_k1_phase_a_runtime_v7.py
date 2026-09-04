from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_capability_v7 import (
    V7PhaseAInputCapability,
    _mint_phase_a_capability,
    _validate_phase_a_capability,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_contract_v7 import (
    DEPENDENCY,
    PHASE_A_LAUNCH_BINDING_SCHEMA,
    PHASE_A_PLAN_SCHEMA,
    PHASE_A_PLAN_VERSION,
    PHASE_A_RECEIPT_SCHEMA,
    PHASE_A_RELEASE_SCHEMA,
    STAGES,
    load_launch_transaction,
    portable_identity,
    self_hashed,
    validate_stage_completion,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_publication_v7 import (
    promote_directory,
    publish_json_exclusive,
    publish_stage_completion,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_runtime_v7 import (
    prepare_worker,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_staging_files_v5 import (
    read_only_identity,
    read_only_json,
)


def _read_only_file(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    path.chmod(0o400)
    return path


def _launch_binding(*, stage: str, job_id: str) -> dict[str, object]:
    return self_hashed(
        {
            "schema": PHASE_A_LAUNCH_BINDING_SCHEMA,
            "status": "VALIDATED",
            "stage": stage,
            "slurm_job_id": job_id,
            "plan_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "release_sha256": "c" * 64,
            "transaction_files": {},
            "upstream": None,
        },
        "binding_sha256",
    )


def _capability(
    root: Path, *, stage: str = "regression", job_id: str = "7001", label: str
):
    staged = _read_only_file(root / f"{label}.json", label.encode("utf-8"))
    launch = _launch_binding(stage=stage, job_id=job_id)
    row = {
        "role": label,
        "path": str(staged),
        "identity": portable_identity(read_only_identity(staged, label)),
    }
    return launch, staged, _mint_phase_a_capability(launch, [row])


def _transaction(root: Path):
    audit = root / "audit"
    audit.mkdir()
    source_archive = _read_only_file(root / "source.tar", b"immutable-source")
    plan_path = audit / "plan.json"
    receipt_path = audit / "receipt.json"
    release_path = audit / "release.json"
    layout = {
        "audit_root": str(audit),
        "plan": str(plan_path),
        "receipt": str(receipt_path),
        "release_completion": str(release_path),
        "failure_audit": str(audit / "failure.json"),
    }
    layout.update(
        {stage + "_completion": str(audit / f"{stage}.complete.json") for stage in STAGES}
    )
    plan = self_hashed(
        {
            "schema_version": PHASE_A_PLAN_SCHEMA,
            "version": PHASE_A_PLAN_VERSION,
            "stage_order": list(STAGES),
            "jobs": {
                stage: {"depends_on": DEPENDENCY[stage]} for stage in STAGES
            },
            "layout": layout,
            "source": {"archive_sha256": sha256(source_archive.read_bytes()).hexdigest()},
            "run_root": str(root),
        },
        "plan_sha256",
    )
    publish_json_exclusive(plan_path, plan)

    job_ids = {stage: str(7001 + index) for index, stage in enumerate(STAGES)}
    receipt = self_hashed(
        {
            "schema": PHASE_A_RECEIPT_SCHEMA,
            "status": "ALL_JOBS_HELD",
            "plan_sha256": plan["plan_sha256"],
            "job_ids": job_ids,
            "dependency_edges": [
                [DEPENDENCY[stage], stage, job_ids[DEPENDENCY[stage]], job_ids[stage]]
                for stage in STAGES
                if DEPENDENCY[stage] is not None
            ],
            "held_scheduler_snapshots": {
                stage: {
                    "job_id": job_ids[stage],
                    "job_state": "PENDING",
                    "reason": "JobHeldUser",
                    "dependency": (
                        None
                        if DEPENDENCY[stage] is None
                        else f"afterok:{job_ids[DEPENDENCY[stage]]}"
                    ),
                }
                for stage in STAGES
            },
        },
        "receipt_sha256",
    )
    publish_json_exclusive(receipt_path, receipt)

    release = self_hashed(
        {
            "schema": PHASE_A_RELEASE_SCHEMA,
            "status": "ALL_JOBS_RELEASED",
            "plan_sha256": plan["plan_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
            "release_order": list(reversed(STAGES)),
            "released_job_ids": [job_ids[stage] for stage in reversed(STAGES)],
            "release_attempts": [
                {
                    "stage": stage,
                    "job_id": job_ids[stage],
                    "returncode": 0,
                    "scheduler_snapshot": {
                        "job_id": job_ids[stage],
                        "job_state": "PENDING",
                        "reason": "Resources",
                        "dependency": (
                            None
                            if DEPENDENCY[stage] is None
                            else f"afterok:{job_ids[DEPENDENCY[stage]]}"
                        ),
                    },
                }
                for stage in reversed(STAGES)
            ],
        },
        "release_sha256",
    )
    publish_json_exclusive(release_path, release)
    return plan, receipt, release, source_archive


def _regression_environment(
    root: Path,
    plan: dict[str, object],
    receipt: dict[str, object],
    source_archive: Path,
) -> dict[str, str]:
    staging = root / "staging"
    staging.mkdir(mode=0o700)
    layout = plan["layout"]
    return {
        "POSTERIOR_V8_V7_PHASE_A_STAGE": "regression",
        "SLURM_JOB_ID": receipt["job_ids"]["regression"],
        "POSTERIOR_V8_V7_PHASE_A_PLAN": layout["plan"],
        "POSTERIOR_V8_V7_PHASE_A_PLAN_SHA256": plan["plan_sha256"],
        "POSTERIOR_V8_V7_PHASE_A_RECEIPT": layout["receipt"],
        "POSTERIOR_V8_V7_PHASE_A_RELEASE": layout["release_completion"],
        "POSTERIOR_V8_V7_PHASE_A_COMPLETION": layout["regression_completion"],
        "POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION": "NONE",
        "POSTERIOR_V8_JOB_STAGING_ROOT": str(staging),
        "POSTERIOR_V8_JOB_SOURCE_ARCHIVE": str(source_archive),
    }


def test_json_publication_is_complete_read_only_exclusive_and_canonical(tmp_path):
    target = tmp_path / "completion.json"
    payload = {"status": "COMPLETE", "value": [2, 1]}

    identity = publish_json_exclusive(target, payload)

    assert identity["mode_octal"] == "0400"
    assert identity["link_count"] == 1
    assert target.read_text(encoding="utf-8") == (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    assert not tuple(tmp_path.glob(".completion.json.*.tmp"))
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_json_exclusive(target, {"status": "FORGED"})
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_model_directory_promotion_keeps_files_frozen_while_moving_entries(tmp_path):
    partial = tmp_path / ".model.partial-7001-test"
    partial.mkdir(mode=0o700)
    expected = {
        "model.keras": b"model-bytes",
        "model.provenance.json": b"provenance-bytes",
        "result.json": b"result-bytes",
    }
    for name, raw in expected.items():
        (partial / name).write_bytes(raw)
    target = tmp_path / "model"

    promote_directory(partial, target)

    assert not partial.exists()
    assert target.stat().st_mode & 0o777 == 0o500
    assert {path.name for path in target.iterdir()} == set(expected)
    for name, raw in expected.items():
        path = target / name
        assert path.read_bytes() == raw
        assert path.stat().st_mode & 0o777 == 0o400
        assert path.stat().st_nlink == 1


def test_failed_model_publication_stays_in_model_parent_for_atomic_promotion(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    target = model_root / "k1-v5-2-phase-a-v5-full-steps1500"
    partial = model_root / f".{target.name}.partial-7006-test"
    partial.mkdir(mode=0o700)
    (partial / "result.json").write_text("{}\n", encoding="utf-8")
    failed_target = target.parent / f"{target.name}-failed-7006"

    promote_directory(partial, failed_target)

    assert not partial.exists()
    assert failed_target.parent == target.parent
    assert failed_target.stat().st_mode & 0o777 == 0o500
    result = failed_target / "result.json"
    assert result.read_text(encoding="utf-8") == "{}\n"
    assert result.stat().st_mode & 0o777 == 0o400
    assert result.stat().st_nlink == 1


def test_capability_is_opaque_single_use_and_detects_staged_input_change(tmp_path):
    with pytest.raises(TypeError, match="opaque"):
        V7PhaseAInputCapability()
    forged = object.__new__(V7PhaseAInputCapability)
    with pytest.raises(RuntimeError, match="not minted"):
        _validate_phase_a_capability(
            forged, stage="regression", phase="pre_use"
        )

    _, _, capability = _capability(tmp_path, label="single-use")
    _validate_phase_a_capability(capability, stage="regression", phase="pre_use")
    _validate_phase_a_capability(capability, stage="regression", phase="post_use")
    with pytest.raises(RuntimeError, match="stale, reused, or out of order"):
        _validate_phase_a_capability(
            capability, stage="regression", phase="pre_use"
        )

    _, staged, changed = _capability(tmp_path, label="changed")
    _validate_phase_a_capability(changed, stage="regression", phase="pre_use")
    staged.chmod(0o600)
    staged.write_bytes(b"CHANGED")
    staged.chmod(0o400)
    with pytest.raises(RuntimeError, match="staged changed changed"):
        _validate_phase_a_capability(changed, stage="regression", phase="post_use")


def test_stage_completion_requires_and_spends_matching_consumed_capability(tmp_path):
    launch, _, capability = _capability(tmp_path, label="completion")
    target = tmp_path / "stage-completion.json"
    with pytest.raises(RuntimeError, match="has not completed post-use"):
        publish_stage_completion(target, launch, (), capability)
    assert not target.exists()

    _validate_phase_a_capability(capability, stage="regression", phase="pre_use")
    _validate_phase_a_capability(capability, stage="regression", phase="post_use")
    published = publish_stage_completion(target, launch, (), capability)
    assert published["job_local_capability"]["pre_post_equal"] is True
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(RuntimeError, match="has not completed post-use"):
        publish_stage_completion(
            tmp_path / "second-completion.json", launch, (), capability
        )
    assert not (tmp_path / "second-completion.json").exists()


def test_prepare_worker_binds_transaction_job_and_completion_end_to_end(tmp_path):
    plan, receipt, release, source_archive = _transaction(tmp_path)
    environment = _regression_environment(tmp_path, plan, receipt, source_archive)

    context = prepare_worker(
        "regression", environment=environment, release_wait_seconds=0.01
    )

    assert set(context.local_paths) == {
        "launch_plan",
        "submission_receipt",
        "release_completion",
    }
    assert context.launch_binding["slurm_job_id"] == receipt["job_ids"]["regression"]
    _validate_phase_a_capability(
        context.capability, stage="regression", phase="pre_use"
    )
    _validate_phase_a_capability(
        context.capability, stage="regression", phase="post_use"
    )
    completion_path = Path(plan["layout"]["regression_completion"])
    publish_stage_completion(
        completion_path,
        context.launch_binding,
        (),
        context.capability,
    )
    completion, _ = read_only_json(completion_path, "regression completion")
    validated = validate_stage_completion(
        completion,
        expected_stage="regression",
        plan=plan,
        receipt=receipt,
        release=release,
    )
    assert validated["status"] == "COMPLETE"


def test_prepare_worker_rejects_job_mismatch_and_extra_input_inventory(tmp_path):
    plan, receipt, _, source_archive = _transaction(tmp_path)
    environment = _regression_environment(tmp_path, plan, receipt, source_archive)
    environment["SLURM_JOB_ID"] = "9999"

    with pytest.raises(ValueError, match="current Slurm job id"):
        prepare_worker(
            "regression", environment=environment, release_wait_seconds=0.01
        )

    unrelated = _read_only_file(tmp_path / "unrelated.json", b"unrelated")
    with pytest.raises(ValueError, match="staged input inventory"):
        prepare_worker(
            "regression",
            extra_inputs={"unexpected": unrelated},
            environment=environment,
            release_wait_seconds=0.01,
        )


def test_launch_transaction_rejects_release_tamper(tmp_path):
    plan, _, release, _ = _transaction(tmp_path)
    release_path = Path(plan["layout"]["release_completion"])
    release_path.chmod(0o600)
    release["status"] = "PARTIAL"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    release_path.chmod(0o400)

    with pytest.raises(ValueError, match="self hash"):
        load_launch_transaction(
            Path(plan["layout"]["plan"]),
            Path(plan["layout"]["receipt"]),
            release_path,
            expected_plan_sha256=plan["plan_sha256"],
        )
