from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_k1_phase_a_dag_v5 import (
    CURRENT_RUN_ROOT_NAME,
    PHASE_ROOT_NAME,
    PLAN_FILENAME,
    RECEIPT_FILENAME,
    V5CommandResult,
    V5K1PhaseALaunchConfig,
    V5K1PhaseALaunchError,
    build_v5_k1_phase_a_launch_plan,
    launch_v5_k1_phase_a_dag,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_cross_platform_manifest_v5 import (
    build_v5_sobol_cross_platform_manifest,
)


_REQUIRED_SOURCE = (
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/launch_k1_phase_a_dag_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_cross_platform_contract_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_cross_platform_manifest_io_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_cross_platform_manifest_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/run_sobol_cross_platform_gate_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_a_cross_platform_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_a_dataset_binding_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/build_k1_memorization_dataset_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/run_k1_memorization_gate_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/memorization_gate_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model_v5_contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/amplitude_query_sampling_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/amplitude_query_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_query_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_amplitude_join_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_amplitude_recipe_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_geometry_recipe_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_numeric_canonicalization_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_recipe_coordinates_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_recipe_physics_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_recipe_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/sobol_universal_query_design_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/regression_cpu.sbatch",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/v5_sobol_cross_platform_gate_cpu.sbatch",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/v5_k1_memorization_dataset_cpu.sbatch",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/v5_k1_memorization_gpu.sbatch",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_amplitude_query_sampling_v5.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_sobol_recipe_v5.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_sobol_universal_query_design_v5.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_study_protocol.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_sobol_cross_platform_manifest_v5.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_a_artifact_binding_v5.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_dataset_gate_worker_v5.py",
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_a_launcher_v5.py",
)
_PACKAGING_REQUIRED = (
    "AGENTS.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "utils/__init__.py",
    "src/gimap/example.py",
    "docs/architecture/example.md",
    "docs/research/example.md",
)


def _write_working_source(root: Path, *, marker: str = "immutable test source") -> Path:
    for relative in (*_PACKAGING_REQUIRED, *_REQUIRED_SOURCE):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{marker}: {relative}\n", encoding="utf-8")
    return root


@lru_cache(maxsize=None)
def _reference_text(source_identity: tuple[str, str, str]) -> str:
    archive, manifest, tree = source_identity
    return build_v5_sobol_cross_platform_manifest(
        source_archive_sha256=archive,
        source_manifest_sha256=manifest,
        source_tree_sha256=tree,
    ).to_json()


def _reference_with_source(encoded: str, source: dict[str, str]) -> str:
    payload = json.loads(encoded)
    payload.pop("manifest_sha256")
    payload["source"] = source
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["manifest_sha256"] = sha256(canonical.encode()).hexdigest()
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _config(tmp_path: Path) -> tuple[V5K1PhaseALaunchConfig, Path, Path]:
    dust = tmp_path / "data/dust/user/zhaiyufe"
    working = _write_working_source(tmp_path / "working-tree")
    archive = dust / "source-archives/posterior-v5-2.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_identity = build_source_snapshot(working, archive)
    source = dust / "source-snapshots/posterior-v5-2-0123456789ab"
    source.parent.mkdir(parents=True, exist_ok=True)
    extracted_identity = extract_source_snapshot(
        archive,
        source,
        expected_sha256=archive_identity["archive_sha256"],
    )
    source_identity = (
        str(archive_identity["archive_sha256"]),
        str(archive_identity["manifest_sha256"]),
        str(extracted_identity["source_tree_sha256"]),
    )
    reference = dust / "cross-platform-references/darwin-reference-v2.json"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(_reference_text(source_identity), encoding="utf-8")
    reference.chmod(0o444)
    run_root = dust / "MaxwellRuns" / CURRENT_RUN_ROOT_NAME
    (run_root / "logs").mkdir(parents=True)
    return (
        V5K1PhaseALaunchConfig(
            source_root=source,
            source_archive_path=archive,
            run_root=run_root,
            source_archive_sha256=str(archive_identity["archive_sha256"]),
            cross_platform_reference_path=reference,
            cross_platform_reference_sha256=sha256(reference.read_bytes()).hexdigest(),
        ),
        dust,
        source,
    )


def test_dry_run_builds_exact_six_stage_afterok_plan_without_writes(tmp_path):
    config, dust, _ = _config(tmp_path)
    result = launch_v5_k1_phase_a_dag(config, allowed_root=dust)
    plan = result["plan"]

    assert result["status"] == "dry_run"
    assert result["writes_performed"] is False
    assert plan["phase_a_scope"] == {
        "topology": ["sphere"],
        "branch_pattern_id": 0,
        "single_branch_capacity_diagnostic": True,
        "balanced_all_k1_topologies_and_legal_branches": False,
    }
    assert plan["protocol_k1_memorization_stage_complete"] is False
    assert plan["full_k1_all_legal_branches_pending_fail_closed"] is True
    assert plan["pending_balanced_k1_scope"]["total_topology_pattern_branches"] == 12
    assert plan["stage_order"] == [
        "regression",
        "cross_platform_gate",
        "smoke_dataset",
        "smoke_gate",
        "full_dataset",
        "full_gate",
    ]
    assert plan["source_archive"]["path"] == str(config.source_archive_path)
    assert plan["source_archive"]["sha256"] == config.source_archive_sha256
    assert plan["source_archive"]["verification"].startswith("archive_manifest")
    assert len(plan["source_archive"]["manifest_sha256"]) == 64
    assert len(plan["source"]["source_tree_sha256"]) == 64
    assert plan["source"]["exact_manifest_file_set_verified"] is True
    assert plan["source"]["read_only_tree_verified"] is True
    assert plan["source"]["symlink_free_lexical_paths_verified"] is True
    assert set(plan["wrapper_sha256"]) == {
        "regression",
        "cross_platform_gate",
        "dataset",
        "gate",
    }
    reference = plan["cross_platform_reference"]
    assert reference["path"] == str(config.cross_platform_reference_path)
    assert reference["file_sha256"] == config.cross_platform_reference_sha256
    assert reference["manifest_schema"].endswith("/v2")
    assert reference["manifest_version"].endswith("_v2")
    assert plan["layout"]["cross_platform_candidate"].endswith("candidate-v2.json")
    assert reference["source"] == {
        "source_archive_sha256": plan["source"]["archive_sha256"],
        "source_manifest_sha256": plan["source"]["manifest_sha256"],
        "source_tree_sha256": plan["source"]["source_tree_sha256"],
    }
    assert len(reference["gate_claim_sha256"]) == 64
    assert plan["jobs"]["smoke_dataset"]["environment"]["POSTERIOR_V8_V5_K1_RECIPE_COUNT"] == 2
    assert plan["jobs"]["full_dataset"]["environment"]["POSTERIOR_V8_V5_K1_RECIPE_COUNT"] == 512
    assert plan["jobs"]["full_gate"]["environment"]["POSTERIOR_V8_V5_K1_STEPS"] == 1500
    assert plan["jobs"]["smoke_dataset"]["environment"][
        "POSTERIOR_V8_V5_K1_DATASET_BINDING"
    ] == plan["layout"]["smoke_dataset_binding"]
    assert plan["jobs"]["smoke_gate"]["environment"][
        "POSTERIOR_V8_V5_K1_DATASET_BINDING"
    ] == plan["layout"]["smoke_dataset_binding"]
    assert plan["jobs"]["full_dataset"]["environment"][
        "POSTERIOR_V8_V5_K1_DATASET_BINDING"
    ] == plan["layout"]["full_dataset_binding"]
    assert plan["jobs"]["full_gate"]["environment"][
        "POSTERIOR_V8_V5_K1_DATASET_BINDING"
    ] == plan["layout"]["full_dataset_binding"]
    assert "v4" not in json.dumps(plan, sort_keys=True)
    assert "--dependency=afterok:REGRESSION_JOB_ID" in plan["submission_preview"][
        "cross_platform_gate"
    ]
    assert "--dependency=afterok:CROSS_PLATFORM_GATE_JOB_ID" in plan[
        "submission_preview"
    ]["smoke_dataset"]
    assert "--dependency=afterok:SMOKE_GATE_JOB_ID" in plan["submission_preview"]["full_dataset"]
    assert all(
        any(value.startswith(f"--{stream}={config.run_root}/logs/") for value in command)
        for command in plan["submission_preview"].values()
        for stream in ("output", "error")
    )
    for job in plan["jobs"].values():
        environment = job["environment"]
        assert environment["POSTERIOR_V8_V5_SOURCE_ARCHIVE"] == str(
            config.source_archive_path
        )
        assert environment["POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"] == (
            config.source_archive_sha256
        )
        assert environment["POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"] == (
            plan["source"]["manifest_sha256"]
        )
        assert environment["POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"] == (
            plan["source"]["source_tree_sha256"]
        )
        if job["depends_on"] is not None:
            assert environment["POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"] == (
                plan["layout"]["cross_platform_pass_marker"]
            )
            assert environment[
                "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256"
            ] == reference["gate_claim_sha256"]
    assert not (config.run_root / PHASE_ROOT_NAME).exists()


def test_plan_rejects_path_escape_wrong_run_root_writable_or_symlink_source(tmp_path):
    config, dust, source = _config(tmp_path)
    outside = replace(config, source_root=tmp_path / "outside")
    with pytest.raises(ValueError, match="under"):
        build_v5_k1_phase_a_launch_plan(outside, allowed_root=dust)

    wrong_run = dust / "MaxwellRuns/not-the-frozen-run"
    (wrong_run / "logs").mkdir(parents=True)
    with pytest.raises(ValueError, match="exactly"):
        build_v5_k1_phase_a_launch_plan(replace(config, run_root=wrong_run), allowed_root=dust)

    source.chmod(0o755)
    with pytest.raises(ValueError, match="read-only"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)
    source.chmod(0o555)

    nested = source / "utils"
    nested.chmod(0o755)
    with pytest.raises(ValueError, match="source directory must be read-only"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)
    nested.chmod(0o555)

    config.source_archive_path.chmod(0o644)
    with pytest.raises(ValueError, match="source archive must be read-only"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)
    config.source_archive_path.chmod(0o444)

    selected = source / "src/gimap/example.py"
    selected.chmod(0o644)
    with pytest.raises(ValueError, match="source file must be read-only"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)
    selected.chmod(0o444)

    linked = dust / "linked-source"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build_v5_k1_phase_a_launch_plan(replace(config, source_root=linked), allowed_root=dust)

    linked_archive = dust / "linked-archive.tar"
    linked_archive.symlink_to(config.source_archive_path)
    with pytest.raises(ValueError, match="symlink"):
        build_v5_k1_phase_a_launch_plan(
            replace(config, source_archive_path=linked_archive), allowed_root=dust
        )

    config.cross_platform_reference_path.chmod(0o644)
    with pytest.raises(ValueError, match="read-only"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)
    config.cross_platform_reference_path.chmod(0o444)

    linked_reference = dust / "linked-reference.json"
    linked_reference.symlink_to(config.cross_platform_reference_path)
    with pytest.raises(ValueError, match="symlink"):
        build_v5_k1_phase_a_launch_plan(
            replace(config, cross_platform_reference_path=linked_reference),
            allowed_root=dust,
        )

    with pytest.raises(ValueError, match="reference SHA-256"):
        build_v5_k1_phase_a_launch_plan(
            replace(config, cross_platform_reference_sha256="0" * 64),
            allowed_root=dust,
        )

    mismatched_reference = dust / "cross-platform-references/wrong-source-v2.json"
    mismatched_reference.write_text(
        _reference_with_source(
            config.cross_platform_reference_path.read_text(encoding="utf-8"),
            {
                "source_archive_sha256": "a" * 64,
                "source_manifest_sha256": "b" * 64,
                "source_tree_sha256": "c" * 64,
            },
        ),
        encoding="utf-8",
    )
    mismatched_reference.chmod(0o444)
    with pytest.raises(ValueError, match="source identity"):
        build_v5_k1_phase_a_launch_plan(
            replace(
                config,
                cross_platform_reference_path=mismatched_reference,
                cross_platform_reference_sha256=sha256(
                    mismatched_reference.read_bytes()
                ).hexdigest(),
            ),
            allowed_root=dust,
        )

    in_run_reference = config.run_root / "logs/reference.json"
    in_run_reference.write_bytes(config.cross_platform_reference_path.read_bytes())
    in_run_reference.chmod(0o444)
    with pytest.raises(ValueError, match="must be separate"):
        build_v5_k1_phase_a_launch_plan(
            replace(
                config,
                cross_platform_reference_path=in_run_reference,
                cross_platform_reference_sha256=sha256(
                    in_run_reference.read_bytes()
                ).hexdigest(),
            ),
            allowed_root=dust,
        )


def test_plan_rejects_existing_target_receipt_or_job_logs(tmp_path):
    config, dust, _ = _config(tmp_path)
    phase_root = config.run_root / PHASE_ROOT_NAME
    (phase_root / "audit").mkdir(parents=True)
    (phase_root / "audit" / RECEIPT_FILENAME).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="target, receipt, or job manifest"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)

    another, another_dust, _ = _config(tmp_path / "log-case")
    (another.run_root / "logs/k1-phase-a-v5-stale.out").write_text(
        "existing", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="Slurm logs"):
        build_v5_k1_phase_a_launch_plan(another, allowed_root=another_dust)


def test_submit_rejects_non_login_host_before_writes_or_subprocess(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    called = False

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal called
        called = True
        return V5CommandResult(0, "1\n", "")

    with pytest.raises(RuntimeError, match="max-wgs"):
        launch_v5_k1_phase_a_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="local-mac",
        )
    assert called is False
    assert not (config.run_root / PHASE_ROOT_NAME).exists()


def test_plan_requires_actual_archive_and_sha_before_writes_or_subprocess(
    tmp_path, monkeypatch
):
    config, dust, _ = _config(tmp_path)
    config = replace(config, source_archive_sha256=None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    called = False

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal called
        called = True
        return V5CommandResult(0, "1\n", "")

    with pytest.raises(ValueError, match="64 lowercase"):
        launch_v5_k1_phase_a_dag(
            config,
            runner=runner,
            allowed_root=dust,
        )
    assert called is False
    assert not (config.run_root / PHASE_ROOT_NAME).exists()

    missing_archive = replace(
        config,
        source_archive_path=dust / "source-archives/missing.tar",
        source_archive_sha256="0" * 64,
    )
    with pytest.raises(FileNotFoundError):
        launch_v5_k1_phase_a_dag(missing_archive, allowed_root=dust)


def test_submit_records_exact_argv_job_ids_and_dependency_edges(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls: list[tuple[str, ...]] = []
    replies = iter(
        (
            "24391001",
            "24391002;maxwell",
            "24391003",
            "24391004",
            "24391005",
            "24391006",
        )
    )

    def runner(argv: Sequence[str]) -> V5CommandResult:
        calls.append(tuple(argv))
        return V5CommandResult(0, next(replies) + "\n", "")

    receipt = launch_v5_k1_phase_a_dag(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs01.desy.de",
    )

    assert receipt["status"] == "submitted"
    assert receipt["job_ids"] == {
        "regression": "24391001",
        "cross_platform_gate": "24391002",
        "smoke_dataset": "24391003",
        "smoke_gate": "24391004",
        "full_dataset": "24391005",
        "full_gate": "24391006",
    }
    assert len(calls) == 6
    assert "--dependency=afterok:24391001" in calls[1]
    assert "--dependency=afterok:24391002" in calls[2]
    assert "--dependency=afterok:24391003" in calls[3]
    assert "--dependency=afterok:24391004" in calls[4]
    assert "--dependency=afterok:24391005" in calls[5]
    assert all(command[:2] == ("sbatch", "--parsable") for command in calls)
    assert all(any(value.startswith("--output=") for value in command) for command in calls)
    assert all(any(value.startswith("--error=") for value in command) for command in calls)
    assert all("--export=ALL" not in command for command in calls)
    assert receipt["secret_environment_captured"] is False
    assert receipt["source_archive"]["sha256"] == config.source_archive_sha256
    assert receipt["source_archive"]["manifest_sha256"] == (
        receipt["source_manifest_sha256"]
    )
    assert receipt["source_archive"]["verification"].startswith("archive_manifest")
    assert receipt["cross_platform_reference"]["file_sha256"] == (
        config.cross_platform_reference_sha256
    )
    assert receipt["cross_platform_reference"]["gate_claim_sha256"] == receipt[
        "cross_platform_gate_claim_sha256"
    ]
    assert receipt["cancellation_attempted"] is False
    assert receipt["dependency_edges"] == [
        ["regression", "cross_platform_gate"],
        ["cross_platform_gate", "smoke_dataset"],
        ["smoke_dataset", "smoke_gate"],
        ["smoke_gate", "full_dataset"],
        ["full_dataset", "full_gate"],
    ]

    audit = config.run_root / PHASE_ROOT_NAME / "audit"
    stored_plan = json.loads((audit / PLAN_FILENAME).read_text(encoding="utf-8"))
    stored_receipt = json.loads((audit / RECEIPT_FILENAME).read_text(encoding="utf-8"))
    assert (audit / PLAN_FILENAME).stat().st_mode & 0o222 == 0
    assert (audit / RECEIPT_FILENAME).stat().st_mode & 0o222 == 0
    assert stored_plan["plan_sha256"] == receipt["plan_sha256"]
    assert stored_receipt == receipt
    assert [item["argv"] for item in receipt["submission_attempts"]] == [
        list(value) for value in calls
    ]

    with pytest.raises(FileExistsError, match="target, receipt, or job manifest"):
        launch_v5_k1_phase_a_dag(config, allowed_root=dust)


def test_bad_sbatch_parse_preserves_exclusive_failure_receipt(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(V5K1PhaseALaunchError) as raised:
        launch_v5_k1_phase_a_dag(
            config,
            submit=True,
            runner=lambda argv: V5CommandResult(0, "Submitted batch job 2439\n", ""),
            allowed_root=dust,
            hostname="max-wgs02",
        )

    receipt = json.loads(raised.value.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "regression"
    assert "invalid parsable Slurm job id" in receipt["failure"]["message"]
    assert receipt["job_ids"] == {}
    assert len(receipt["submission_attempts"]) == 1
    assert receipt["submission_attempts"][0]["argv"][:2] == ["sbatch", "--parsable"]


def test_archive_sha_and_source_drift_fail_closed(tmp_path, monkeypatch):
    config, dust, source = _config(tmp_path)
    with pytest.raises(ValueError, match="64 lowercase"):
        build_v5_k1_phase_a_launch_plan(
            replace(config, source_archive_sha256="A" * 64), allowed_root=dust
        )
    with pytest.raises(ValueError, match="does not match"):
        build_v5_k1_phase_a_launch_plan(
            replace(config, source_archive_sha256="0" * 64), allowed_root=dust
        )

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = 0

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal calls
        calls += 1
        source.chmod(0o755)
        return V5CommandResult(0, "24392001\n", "")

    with pytest.raises(V5K1PhaseALaunchError) as raised:
        launch_v5_k1_phase_a_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs03",
        )
    receipt = json.loads(raised.value.receipt_path.read_text(encoding="utf-8"))
    assert calls == 1
    assert receipt["job_ids"] == {"regression": "24392001"}
    assert receipt["failure"]["stage"] == "cross_platform_gate"
    assert "read-only" in receipt["failure"]["message"]


def test_missing_source_and_wrong_valid_archive_fail_closed(tmp_path):
    config, dust, source = _config(tmp_path)
    missing = source / "src/gimap/example.py"
    missing.parent.chmod(0o755)
    missing.unlink()
    missing.parent.chmod(0o555)
    with pytest.raises(ValueError, match="file set does not exactly match"):
        build_v5_k1_phase_a_launch_plan(config, allowed_root=dust)

    another, another_dust, _ = _config(tmp_path / "wrong-archive-case")
    wrong_working = _write_working_source(
        tmp_path / "wrong-archive-case/wrong-working-tree",
        marker="different but valid source",
    )
    wrong_archive = another_dust / "source-archives/wrong-valid.tar"
    wrong = build_source_snapshot(wrong_working, wrong_archive)
    with pytest.raises(ValueError, match="manifest does not match|identity mismatch"):
        build_v5_k1_phase_a_launch_plan(
            replace(
                another,
                source_archive_path=wrong_archive,
                source_archive_sha256=str(wrong["archive_sha256"]),
            ),
            allowed_root=another_dust,
        )


def test_archive_replacement_between_submissions_fails_closed(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path)
    wrong_working = _write_working_source(
        tmp_path / "replacement-working-tree",
        marker="replacement source",
    )
    replacement = dust / "source-archives/replacement.tar"
    build_source_snapshot(wrong_working, replacement)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = 0

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal calls
        calls += 1
        config.source_archive_path.unlink()
        replacement.replace(config.source_archive_path)
        return V5CommandResult(0, "24393001\n", "")

    with pytest.raises(V5K1PhaseALaunchError) as raised:
        launch_v5_k1_phase_a_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs04",
        )
    receipt = json.loads(raised.value.receipt_path.read_text(encoding="utf-8"))
    assert calls == 1
    assert receipt["job_ids"] == {"regression": "24393001"}
    assert receipt["failure"]["stage"] == "cross_platform_gate"
    assert "does not match" in receipt["failure"]["message"]


def test_reference_replacement_between_submissions_fails_closed(tmp_path, monkeypatch):
    config, dust, _ = _config(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = 0

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal calls
        calls += 1
        content = config.cross_platform_reference_path.read_bytes()
        config.cross_platform_reference_path.unlink()
        config.cross_platform_reference_path.write_bytes(content)
        config.cross_platform_reference_path.chmod(0o444)
        return V5CommandResult(0, "24394001\n", "")

    with pytest.raises(V5K1PhaseALaunchError) as raised:
        launch_v5_k1_phase_a_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs05",
        )
    receipt = json.loads(raised.value.receipt_path.read_text(encoding="utf-8"))
    assert calls == 1
    assert receipt["job_ids"] == {"regression": "24394001"}
    assert receipt["failure"]["stage"] == "cross_platform_gate"
    assert "reference changed" in receipt["failure"]["message"]


@pytest.mark.parametrize(
    ("relative", "heavy_command"),
    (
        (
            "PosteriorV8/slurm/regression_cpu.sbatch",
            "from pytest import console_main",
        ),
        (
            "PosteriorV8/slurm/v5_sobol_cross_platform_gate_cpu.sbatch",
            "PosteriorV8.run_sobol_cross_platform_gate_v5",
        ),
        (
            "PosteriorV8/slurm/v5_k1_memorization_dataset_cpu.sbatch",
            "PosteriorV8.build_k1_memorization_dataset_v5",
        ),
        (
            "PosteriorV8/slurm/v5_k1_memorization_gpu.sbatch",
            "PosteriorV8.run_k1_memorization_gate_v5",
        ),
    ),
)
def test_phase_a_wrappers_reverify_bound_source_before_work(relative, heavy_command):
    bundle_root = Path(__file__).resolve().parents[1]
    wrapper = (bundle_root / relative).read_text(encoding="utf-8")

    assert "SLURM_JOB_ID" in wrapper
    assert "max-wgs*" in wrapper
    assert "verify-extracted" in wrapper
    assert "POSTERIOR_V8_V5_SOURCE_ARCHIVE" in wrapper
    assert "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" in wrapper
    assert "POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" in wrapper
    assert "POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256" in wrapper
    assert "python -I \"$POSTERIOR_V8_TRUSTED_VERIFIER\"" in wrapper
    assert wrapper.index("cp --") < wrapper.index("sha256sum --")
    assert wrapper.index("sha256sum --") < wrapper.index("tar -xOf")
    extract_call = wrapper.index("    extract \\\n")
    assert wrapper.index("tar -xOf") < extract_call
    assert extract_call < wrapper.index("verify-extracted")
    assert wrapper.index("verify-extracted") < wrapper.index(
        'cd "$POSTERIOR_V8_JOB_SOURCE_ROOT"'
    )
    assert wrapper.index("verify-extracted") < wrapper.index(heavy_command)
    assert 'cd "$POSTERIOR_V8_SOURCE_ROOT"' not in wrapper
    assert "export PYTHONNOUSERSITE=1" in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert "${PYTHONPATH:+" not in wrapper
    assert "python -m " not in wrapper
    assert "python -I -c" in wrapper
    if "regression_cpu" in relative:
        assert "env -u SLURM_JOB_ID conda run" in wrapper
        assert 'sys.path.insert(0, root)' in wrapper
        assert 'sys.argv = ["pytest", *sys.argv[2:]]' in wrapper
    else:
        assert "run_verified_module" in wrapper
        assert "runpy.run_module(module" in wrapper
    if "memorization_gpu" in relative:
        dataset_copy = 'cp -- "$POSTERIOR_V8_V5_K1_DATASET" "$POSTERIOR_V8_JOB_DATASET"'
        assert dataset_copy in wrapper
        assert '--dataset "$POSTERIOR_V8_JOB_DATASET"' in wrapper
        assert wrapper.index("verify-extracted") < wrapper.index(dataset_copy)
        assert wrapper.index(dataset_copy) < wrapper.index(heavy_command)


def test_cross_platform_and_dataset_bindings_are_checked_before_downstream_work():
    bundle_root = Path(__file__).resolve().parents[1]
    slurm = bundle_root / "PosteriorV8/slurm"
    cross = (slurm / "v5_sobol_cross_platform_gate_cpu.sbatch").read_text()
    dataset = (slurm / "v5_k1_memorization_dataset_cpu.sbatch").read_text()
    gpu = (slurm / "v5_k1_memorization_gpu.sbatch").read_text()

    for wrapper in (cross, dataset, gpu):
        assert "GISAXS_JOB_TMP_BASE" in wrapper
        assert "/data/dust/user/zhaiyufe|/data/dust/user/zhaiyufe/*" in wrapper
        assert 'realpath -e -- "$GISAXS_JOB_TMP_BASE"' in wrapper
        assert "mktemp -d" in wrapper
        assert wrapper.index("realpath -e") < wrapper.index("mktemp -d")
        assert 'chmod 0700 "$GISAXS_JOB_CACHE_ROOT"' in wrapper
        assert '! -O "$GISAXS_JOB_CACHE_ROOT"' in wrapper
        assert "stat -c '%a' -- \"$GISAXS_JOB_CACHE_ROOT\"" in wrapper
        assert (
            '--expected-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"'
            in wrapper
        )

    cross_build = cross.index("PosteriorV8.run_sobol_cross_platform_gate_v5")
    cross_marker_check = cross.index("PosteriorV8.k1_phase_a_cross_platform_v5")
    assert cross.index("verify-extracted") < cross_build < cross_marker_check
    assert cross.index('cp -- "$POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE"') < cross_build
    assert cross.index("POSTERIOR_V8_JOB_REFERENCE_SHA256") < cross_build
    assert cross.index('chmod 0400 \\\n') < cross_marker_check

    dataset_marker_check = dataset.index("PosteriorV8.k1_phase_a_cross_platform_v5")
    dataset_build = dataset.index("PosteriorV8.build_k1_memorization_dataset_v5")
    dataset_binding = dataset.index("PosteriorV8.k1_phase_a_dataset_binding_v5")
    assert dataset.index("verify-extracted") < dataset_marker_check < dataset_build
    assert dataset_build < dataset.index('chmod 0400 "$POSTERIOR_V8_V5_K1_DATASET_OUTPUT"')
    assert dataset.index('chmod 0400 "$POSTERIOR_V8_V5_K1_DATASET_OUTPUT"') < dataset_binding
    assert "publish \\" in dataset

    gpu_marker_check = gpu.index("PosteriorV8.k1_phase_a_cross_platform_v5")
    gpu_dataset_copy = gpu.index('cp -- "$POSTERIOR_V8_V5_K1_DATASET"')
    gpu_binding_check = gpu.index("PosteriorV8.k1_phase_a_dataset_binding_v5")
    gpu_train = gpu.index("PosteriorV8.run_k1_memorization_gate_v5")
    assert gpu.index("verify-extracted") < gpu_marker_check < gpu_dataset_copy
    assert gpu_dataset_copy < gpu_binding_check < gpu.index("srun nvidia-smi") < gpu_train
    assert '--dataset "$POSTERIOR_V8_JOB_DATASET"' in gpu
    assert '--binding "$POSTERIOR_V8_JOB_DATASET_BINDING"' in gpu
