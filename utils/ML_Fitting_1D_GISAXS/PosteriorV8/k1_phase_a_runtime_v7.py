"""Pinned-wrapper runtime for all six K1 Phase-A v7 worker stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
from typing import Mapping, Sequence

from .k1_phase_a_capability_v7 import (
    V7PhaseAInputCapability,
    _consumed_capability_payload,
    _mint_phase_a_capability,
    _validate_phase_a_capability,
)
from .k1_phase_a_contract_v7 import (
    DEPENDENCY,
    STAGES,
    launch_binding,
    load_launch_transaction,
    portable_identity,
    self_hashed,
    validate_stage_completion,
)
from .k1_phase_a_publication_v7 import (
    artifact_rows,
    promote_directory,
    publish_file_from_partial,
    publish_json_exclusive,
    publish_stage_completion,
)
from .k1_staging_files_v5 import (
    copy_regular_exclusive,
    read_only_identity,
    read_only_json,
)


FAILURE_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_worker_failure/v1"


@dataclass(frozen=True)
class PhaseAWorkerContext:
    stage: str
    plan: Mapping[str, object]
    receipt: Mapping[str, object]
    release: Mapping[str, object]
    launch_binding: Mapping[str, object]
    capability: V7PhaseAInputCapability
    local_paths: Mapping[str, Path]


def _required_environment(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise RuntimeError(f"missing Phase-A worker environment {name}")
    return value


def _wait_for_release(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and not path.is_symlink():
        if time.monotonic() >= deadline:
            raise TimeoutError("Phase-A release completion was never published")
        time.sleep(0.2)
    if path.is_symlink():
        raise ValueError("Phase-A release completion must not be a symlink")


def _expected_artifact(
    completion: Mapping[str, object], role: str
) -> Mapping[str, object]:
    matches = [item for item in completion["artifacts"] if item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"upstream completion has no unique {role}")
    return matches[0]


def _copy_bound_input(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    role: str,
) -> dict[str, object]:
    original = read_only_identity(source, f"authoritative Phase-A {role}")
    if original["sha256"] != expected_sha256:
        raise ValueError(f"authoritative Phase-A {role} hash drifted")
    copy_regular_exclusive(
        source,
        destination,
        expected_sha256=expected_sha256,
        name=f"Phase-A {role}",
    )
    local = read_only_identity(destination, f"job-local Phase-A {role}")
    return {"role": role, "path": str(destination.resolve()), "identity": portable_identity(local)}


def prepare_worker(
    stage: str,
    *,
    extra_inputs: Mapping[str, Path] | None = None,
    environment: Mapping[str, str] | None = None,
    release_wait_seconds: float | None = None,
) -> PhaseAWorkerContext:
    if stage not in STAGES:
        raise ValueError("unsupported Phase-A worker stage")
    required_extra_roles = {
        "regression": set(),
        "cross_platform_gate": {"cross_platform_reference"},
        "smoke_dataset": {"cross_platform_pass_marker"},
        "smoke_gate": {
            "k1_dataset",
            "k1_dataset_binding",
            "cross_platform_pass_marker",
        },
        "full_dataset": {"cross_platform_pass_marker"},
        "full_gate": {
            "k1_dataset",
            "k1_dataset_binding",
            "cross_platform_pass_marker",
        },
    }[stage]
    supplied_extra_inputs = dict(extra_inputs or {})
    if set(supplied_extra_inputs) != required_extra_roles:
        raise ValueError(f"Phase-A {stage} staged input inventory drifted")
    env = os.environ if environment is None else environment
    if _required_environment(env, "POSTERIOR_V8_V7_PHASE_A_STAGE") != stage:
        raise RuntimeError("Phase-A stage differs from the Slurm export binding")
    job_id = _required_environment(env, "SLURM_JOB_ID")
    plan_path = Path(_required_environment(env, "POSTERIOR_V8_V7_PHASE_A_PLAN"))
    receipt_path = Path(_required_environment(env, "POSTERIOR_V8_V7_PHASE_A_RECEIPT"))
    release_path = Path(_required_environment(env, "POSTERIOR_V8_V7_PHASE_A_RELEASE"))
    timeout = (
        float(env.get("POSTERIOR_V8_V7_RELEASE_WAIT_SECONDS", "60"))
        if release_wait_seconds is None
        else release_wait_seconds
    )
    _wait_for_release(release_path, timeout)
    plan, receipt, release, transaction_files = load_launch_transaction(
        plan_path,
        receipt_path,
        release_path,
        expected_plan_sha256=_required_environment(
            env, "POSTERIOR_V8_V7_PHASE_A_PLAN_SHA256"
        ),
    )
    completion_path = Path(
        _required_environment(env, "POSTERIOR_V8_V7_PHASE_A_COMPLETION")
    ).resolve()
    if completion_path != Path(plan["layout"][f"{stage}_completion"]).resolve():
        raise ValueError("Phase-A completion output differs from the plan")

    upstream_stage = DEPENDENCY[stage]
    upstream_payload = None
    upstream_file = None
    if upstream_stage is not None:
        supplied = Path(
            _required_environment(env, "POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION")
        )
        expected_path = Path(plan["layout"][f"{upstream_stage}_completion"])
        if supplied.resolve() != expected_path.resolve():
            raise ValueError("Phase-A upstream completion path differs from the plan")
        upstream_payload, upstream_file = read_only_json(
            supplied, f"Phase-A {upstream_stage} completion"
        )
        upstream_payload = validate_stage_completion(
            upstream_payload,
            expected_stage=upstream_stage,
            plan=plan,
            receipt=receipt,
            release=release,
        )
    elif env.get("POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION") != "NONE":
        raise ValueError("regression upstream completion sentinel drifted")

    launch = launch_binding(
        stage=stage,
        slurm_job_id=job_id,
        plan=plan,
        receipt=receipt,
        release=release,
        transaction_files=transaction_files,
        upstream_completion=upstream_payload,
        upstream_completion_file=upstream_file,
    )
    staging = Path(_required_environment(env, "POSTERIOR_V8_JOB_STAGING_ROOT"))
    status = staging.lstat()
    if (
        not staging.is_dir()
        or staging.is_symlink()
        or status.st_uid != os.getuid()
        or (status.st_mode & 0o777) != 0o700
    ):
        raise ValueError("Phase-A job-local staging must be an owner-private 0700 directory")
    input_root = staging / "phase-a-inputs"
    input_root.mkdir(mode=0o700)
    rows: list[dict[str, object]] = []
    local_paths: dict[str, Path] = {}
    authoritative = {
        "launch_plan": (plan_path, transaction_files["plan"]["sha256"]),
        "submission_receipt": (receipt_path, transaction_files["receipt"]["sha256"]),
        "release_completion": (
            release_path,
            transaction_files["release_completion"]["sha256"],
        ),
    }
    if upstream_stage is not None:
        authoritative["upstream_completion"] = (
            Path(str(upstream_file["path"])),
            upstream_file["sha256"],
        )
    for role, (source, expected_sha) in authoritative.items():
        destination = input_root / f"{role}.json"
        rows.append(
            _copy_bound_input(
                source, destination, expected_sha256=str(expected_sha), role=role
            )
        )
        local_paths[role] = destination

    for role, source in supplied_extra_inputs.items():
        if role == "cross_platform_reference":
            expected_sha = plan["cross_platform_reference"]["file_sha256"]
        elif upstream_payload is None:
            raise ValueError("regression has no authoritative extra input contract")
        else:
            expected_sha = _expected_artifact(upstream_payload, role)["identity"]["sha256"]
        destination = input_root / f"{role}{source.suffix}"
        rows.append(
            _copy_bound_input(
                source, destination, expected_sha256=str(expected_sha), role=role
            )
        )
        local_paths[role] = destination

    archive = Path(_required_environment(env, "POSTERIOR_V8_JOB_SOURCE_ARCHIVE"))
    archive_identity = read_only_identity(archive, "job-local Phase-A source archive")
    if archive_identity["sha256"] != plan["source"]["archive_sha256"]:
        raise ValueError("job-local Phase-A source archive differs from the plan")
    rows.append(
        {
            "role": "source_archive",
            "path": str(archive.resolve()),
            "identity": portable_identity(archive_identity),
        }
    )
    capability = _mint_phase_a_capability(launch, rows)
    return PhaseAWorkerContext(stage, plan, receipt, release, launch, capability, local_paths)


def _failure_path(context: PhaseAWorkerContext | None, stage: str, job_id: str) -> Path | None:
    if context is None:
        return None
    root = Path(context.plan["layout"]["audit_root"]) / "failures"
    root.mkdir(mode=0o700, exist_ok=True)
    return root / f"{stage}-{job_id}-failure-v1.json"


def _publish_failure(
    context: PhaseAWorkerContext | None, stage: str, exc: BaseException
) -> None:
    job = os.environ.get("SLURM_JOB_ID", "unknown")
    target = _failure_path(context, stage, job)
    if target is None or target.exists():
        return
    core = {
        "schema": FAILURE_SCHEMA,
        "status": "FAILED_NO_STAGE_COMPLETION",
        "stage": stage,
        "slurm_job_id": job,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc)[:2000],
        "launch_binding": None if context is None else context.launch_binding,
    }
    publish_json_exclusive(target, self_hashed(core, "failure_sha256"))


def _complete(
    context: PhaseAWorkerContext, artifacts: Sequence[tuple[str, Path]]
) -> Mapping[str, object]:
    completion = Path(context.plan["layout"][f"{context.stage}_completion"])
    return publish_stage_completion(
        completion,
        context.launch_binding,
        artifacts,
        context.capability,
    )


def _run_regression(context: PhaseAWorkerContext) -> Mapping[str, object]:
    _validate_phase_a_capability(context.capability, stage=context.stage, phase="pre_use")
    source = Path(os.environ["POSTERIOR_V8_JOB_SOURCE_ROOT"])
    tests = sorted(
        str(path.relative_to(source))
        for path in (source / "utils/ML_Fitting_1D_GISAXS/tests").glob(
            "test_posterior_v8_*.py"
        )
    )
    regression_environment = {
        name: value for name, value in os.environ.items() if name != "SLURM_JOB_ID"
    }
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-c",
            "from pytest import console_main; import sys; root=sys.argv[1]; "
            "sys.path.insert(0,root); sys.argv=['pytest',*sys.argv[2:]]; "
            "raise SystemExit(console_main())",
            str(source),
            "-q",
            *tests,
            "--disable-warnings",
            "-p",
            "no:cacheprovider",
        ],
        cwd=source,
        env={**regression_environment, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Phase-A regression failed with {completed.returncode}")
    _validate_phase_a_capability(context.capability, stage=context.stage, phase="post_use")
    return _complete(context, ())


def _run_cross(context: PhaseAWorkerContext) -> Mapping[str, object]:
    from .run_sobol_cross_platform_gate_v5 import main as cross_main

    _validate_phase_a_capability(context.capability, stage=context.stage, phase="pre_use")
    layout = context.plan["layout"]
    result = cross_main(
        [
            "build-and-compare",
            "--source-archive-sha256", str(context.plan["source"]["archive_sha256"]),
            "--source-manifest-sha256", str(context.plan["source"]["manifest_sha256"]),
            "--source-tree-sha256", str(context.plan["source"]["source_tree_sha256"]),
            "--reference", str(context.local_paths["cross_platform_reference"]),
            "--candidate-output", str(layout["cross_platform_candidate"]),
            "--result-output", str(layout["cross_platform_result"]),
            "--pass-marker", str(layout["cross_platform_pass_marker"]),
        ]
    )
    if result != 0:
        raise RuntimeError("Phase-A cross-platform comparison failed")
    artifacts = (
        ("cross_platform_candidate", Path(layout["cross_platform_candidate"])),
        ("cross_platform_result", Path(layout["cross_platform_result"])),
        ("cross_platform_pass_marker", Path(layout["cross_platform_pass_marker"])),
    )
    for _, path in artifacts:
        path.chmod(0o400)
    _validate_phase_a_capability(context.capability, stage=context.stage, phase="post_use")
    return _complete(context, artifacts)


def _dataset_arguments(context: PhaseAWorkerContext) -> tuple[int, str, int, str, int, Path, Path]:
    env = os.environ
    return (
        int(env["POSTERIOR_V8_V5_K1_RECIPE_COUNT"]),
        env["POSTERIOR_V8_V5_K1_TOPOLOGY"],
        int(env["POSTERIOR_V8_V5_K1_BASE_SEED"]),
        env["POSTERIOR_V8_V5_K1_VIEW_INDICES"],
        int(env["POSTERIOR_V8_V5_K1_PATTERN_ID"]),
        Path(env["POSTERIOR_V8_V5_K1_DATASET_OUTPUT"]),
        Path(env["POSTERIOR_V8_V5_K1_DATASET_BINDING"]),
    )


def _run_dataset(context: PhaseAWorkerContext) -> Mapping[str, object]:
    from .build_k1_memorization_dataset_v5 import build_v5_k1_memorization_dataset
    from .k1_phase_a_dataset_binding_v5 import publish_v5_k1_phase_a_dataset_binding

    count, topology, seed, views, pattern, target, binding = _dataset_arguments(context)
    partial = target.parent / f".{target.name}.partial-{context.launch_binding['slurm_job_id']}-{secrets.token_hex(4)}"
    build_v5_k1_memorization_dataset(
        partial,
        recipe_count=count,
        topology=topology,
        base_seed=seed,
        view_indices=views,
        pattern_id=pattern,
        allowed_root=Path(context.plan["run_root"]),
        hostname=socket.gethostname(),
        environment=os.environ,
    )
    publish_file_from_partial(partial, target)
    marker = context.local_paths["cross_platform_pass_marker"]
    publish_v5_k1_phase_a_dataset_binding(
        target,
        marker,
        binding,
        original_dataset_path=str(target),
        marker_expected={
            "expected_source": {
                "source_archive_sha256": context.plan["source"]["archive_sha256"],
                "source_manifest_sha256": context.plan["source"]["manifest_sha256"],
                "source_tree_sha256": context.plan["source"]["source_tree_sha256"],
            },
            "reference_file_sha256": context.plan["cross_platform_reference"]["file_sha256"],
            "reference_file_byte_count": context.plan["cross_platform_reference"]["file_byte_count"],
            "reference_manifest_sha256": context.plan["cross_platform_reference"]["manifest_sha256"],
            "scientific_content_sha256": context.plan["cross_platform_reference"]["scientific_content_sha256"],
            "comparison_result_sha256": context.plan["cross_platform_reference"]["comparison_result_sha256"],
            "expected_gate_claim_sha256": context.plan["cross_platform_reference"]["gate_claim_sha256"],
        },
        launch_binding=context.launch_binding,
        capability=context.capability,
    )
    return _complete(
        context,
        (
            ("k1_dataset", target),
            ("k1_dataset_binding", binding),
            (
                "cross_platform_pass_marker",
                Path(os.environ["POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"]),
            ),
        ),
    )


def _run_gate(context: PhaseAWorkerContext) -> Mapping[str, object]:
    from .run_k1_memorization_gate_v5 import (
        V5K1DatasetGateConfig,
        V5K1PhaseATrainingEvidence,
        run_v5_k1_dataset_memorization_gate,
    )

    env = os.environ
    target = Path(env["POSTERIOR_V8_V5_K1_GATE_OUTPUT"])
    partial = target.parent / f".{target.name}.partial-{context.launch_binding['slurm_job_id']}-{secrets.token_hex(4)}"
    source = context.plan["source"]
    reference = context.plan["cross_platform_reference"]
    smoke = env.get("POSTERIOR_V8_V5_K1_SMOKE", "0") == "1"
    evidence = V5K1PhaseATrainingEvidence(
        dataset_binding_path=context.local_paths["k1_dataset_binding"],
        cross_platform_pass_marker_path=context.local_paths["cross_platform_pass_marker"],
        original_dataset_path=env["POSTERIOR_V8_V5_K1_DATASET"],
        source_archive_sha256=source["archive_sha256"],
        source_manifest_sha256=source["manifest_sha256"],
        source_tree_sha256=source["source_tree_sha256"],
        reference_file_sha256=reference["file_sha256"],
        reference_file_byte_count=reference["file_byte_count"],
        reference_manifest_sha256=reference["manifest_sha256"],
        scientific_content_sha256=reference["scientific_content_sha256"],
        comparison_result_sha256=reference["comparison_result_sha256"],
        gate_claim_sha256=reference["gate_claim_sha256"],
    )
    result = run_v5_k1_dataset_memorization_gate(
        context.local_paths["k1_dataset"],
        partial,
        source_root=Path(env["POSTERIOR_V8_JOB_SOURCE_ROOT"]),
        evidence=evidence,
        config=V5K1DatasetGateConfig(
            steps=int(env.get("POSTERIOR_V8_V5_K1_STEPS", "1500")),
            batch_size=int(env.get("POSTERIOR_V8_V5_K1_BATCH_SIZE", "32")),
            learning_rate=float(env.get("POSTERIOR_V8_V5_K1_LEARNING_RATE", "0.003")),
            seed=int(env.get("POSTERIOR_V8_V5_K1_SEED", "20260903")),
            max_final_target_median_rms=float(env.get("POSTERIOR_V8_V5_K1_MAX_FINAL_RMS", "0.01")),
            minimum_loss_reduction=float(env.get("POSTERIOR_V8_V5_K1_MINIMUM_LOSS_REDUCTION", "0.5")),
            width=int(env.get("POSTERIOR_V8_V5_K1_WIDTH", "128")),
            encoder_blocks=int(env.get("POSTERIOR_V8_V5_K1_ENCODER_BLOCKS", "6")),
            smoke=smoke,
        ),
        allowed_root=Path(context.plan["run_root"]),
        dataset_allowed_root=Path(os.environ["POSTERIOR_V8_JOB_STAGING_ROOT"]),
        hostname=socket.gethostname(),
        environment=os.environ,
        launch_binding=context.launch_binding,
        job_local_input_capability=context.capability,
        authoritative_output_dir=target,
    )
    if not smoke and not result["stage_a_passed"]:
        failed_target = target.parent / (
            f"{target.name}-failed-{context.launch_binding['slurm_job_id']}"
        )
        promote_directory(
            partial,
            failed_target,
        )
        raise RuntimeError(
            "formal Phase-A memorization gate failed; immutable failed model "
            f"retained at {failed_target}"
        )
    promote_directory(partial, target)
    return _complete(
        context,
        (
            ("k1_dataset", Path(env["POSTERIOR_V8_V5_K1_DATASET"])),
            ("k1_dataset_binding", Path(env["POSTERIOR_V8_V5_K1_DATASET_BINDING"])),
            (
                "cross_platform_pass_marker",
                Path(env["POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"]),
            ),
            ("k1_model", target / "model.keras"),
            ("k1_model_provenance", target / "model.provenance.json"),
            ("k1_gate_result", target / "result.json"),
        ),
    )


def run_stage_from_environment(stage: str) -> Mapping[str, object]:
    context: PhaseAWorkerContext | None = None
    try:
        extra: dict[str, Path] = {}
        if stage == "cross_platform_gate":
            extra["cross_platform_reference"] = Path(
                os.environ["POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE"]
            )
        elif stage in {"smoke_dataset", "full_dataset"}:
            extra["cross_platform_pass_marker"] = Path(
                os.environ["POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"]
            )
        elif stage in {"smoke_gate", "full_gate"}:
            extra = {
                "k1_dataset": Path(os.environ["POSTERIOR_V8_V5_K1_DATASET"]),
                "k1_dataset_binding": Path(os.environ["POSTERIOR_V8_V5_K1_DATASET_BINDING"]),
                "cross_platform_pass_marker": Path(
                    os.environ["POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"]
                ),
            }
        context = prepare_worker(stage, extra_inputs=extra)
        if stage == "regression":
            return _run_regression(context)
        if stage == "cross_platform_gate":
            return _run_cross(context)
        if stage in {"smoke_dataset", "full_dataset"}:
            return _run_dataset(context)
        return _run_gate(context)
    except BaseException as exc:
        _publish_failure(context, stage, exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1 or values[0] not in STAGES:
        raise SystemExit("usage: k1_phase_a_runtime_v7.py STAGE")
    result = run_stage_from_environment(values[0])
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PhaseAWorkerContext", "main", "prepare_worker", "run_stage_from_environment"]
