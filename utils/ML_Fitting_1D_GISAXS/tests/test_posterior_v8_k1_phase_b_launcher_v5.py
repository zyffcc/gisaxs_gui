from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_phase_a_cross_platform_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import launch_k1_phase_b_dag_v5 as launcher
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_contract_v5 import (
    PHASE_A_SOURCE_PATHS,
    PHASE_B_SOURCE_PATHS,
    source_identity,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_launch_inputs_v5 import (
    PHASE_A_ROOT_NAME,
    V5K1PhaseBLaunchConfig,
    expected_v5_k1_phase_a_paths,
    inspect_v5_k1_phase_b_source,
    resolve_v5_k1_phase_a_input_files,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_b_worker_inputs_v5 import (
    V5K1PhaseBWorkerInputSpec,
    assert_v5_k1_phase_b_worker_inputs_unchanged,
    bind_v5_k1_phase_b_worker_inputs,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_k1_phase_b_dag_v5 import (
    CURRENT_RUN_ROOT_NAME,
    PHASE_ROOT_NAME,
    PLAN_FILENAME,
    RECEIPT_FILENAME,
    V5CommandResult,
    V5K1PhaseBLaunchError,
    build_v5_k1_phase_b_launch_plan,
    launch_v5_k1_phase_b_dag,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)


WRAPPER_RELATIVE = Path(
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/v5_k1_phase_b_gate_cpu.sbatch"
)
LAUNCHER_TEST_RELATIVE = Path(
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_b_launcher_v5.py"
)


def _canonical_sha(value: dict[str, object], self_field: str) -> str:
    core = dict(value)
    supplied = core.pop(self_field)
    reproduced = sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert supplied == reproduced
    return reproduced


def _read_only(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o400)
    return path, sha256(content).hexdigest()


def _stubbed_config(tmp_path: Path, monkeypatch):
    dust = tmp_path / "data/dust/user/zhaiyufe"
    run_root = dust / "MaxwellRuns" / CURRENT_RUN_ROOT_NAME
    (run_root / "logs").mkdir(parents=True)
    source_root = dust / "source-snapshots/new-audited-snapshot"
    source_root.mkdir(parents=True)
    archive, archive_sha = _read_only(
        dust / "source-archives/new-audited-source.tar", b"source archive"
    )
    paths = expected_v5_k1_phase_a_paths(run_root)
    supplied = {}
    for index, (name, path) in enumerate(paths.items(), start=1):
        _, file_sha = _read_only(path, f"{name}-{index}".encode())
        supplied[name] = (path, file_sha)
    config = V5K1PhaseBLaunchConfig(
        source_root=source_root,
        source_archive_path=archive,
        source_archive_sha256=archive_sha,
        run_root=run_root,
        phase_a_dataset_path=supplied["dataset"][0],
        phase_a_dataset_sha256=supplied["dataset"][1],
        phase_a_dataset_binding_path=supplied["dataset_binding"][0],
        phase_a_dataset_binding_sha256=supplied["dataset_binding"][1],
        phase_a_cross_platform_pass_marker_path=supplied["cross_platform_pass_marker"][0],
        phase_a_cross_platform_pass_marker_sha256=supplied["cross_platform_pass_marker"][1],
        phase_a_result_path=supplied["result"][0],
        phase_a_result_sha256=supplied["result"][1],
        phase_a_model_path=supplied["model"][0],
        phase_a_model_sha256=supplied["model"][1],
        phase_a_model_provenance_path=supplied["model_provenance"][0],
        phase_a_model_provenance_sha256=supplied["model_provenance"][1],
    )
    source = {
        "source_root": str(source_root),
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_byte_count": archive.stat().st_size,
        "manifest_sha256": "1" * 64,
        "source_tree_sha256": "2" * 64,
        "required_file_identity": {WRAPPER_RELATIVE.as_posix(): {"sha256": "3" * 64}},
    }
    artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha,
            "byte_count": path.stat().st_size,
            "mode": 0o400,
            "device": 1,
            "inode": index,
            "mtime_ns": index,
        }
        for index, (name, (path, file_sha)) in enumerate(supplied.items(), start=1)
    }
    gate = {
        "gate_claim_sha256": "4" * 64,
        "reference_file_sha256": "5" * 64,
        "reference_file_byte_count": 123,
        "reference_manifest_sha256": "6" * 64,
        "scientific_content_sha256": "7" * 64,
        "comparison_result_sha256": "8" * 64,
    }
    phase_a = {
        "artifacts": artifacts,
        "dataset": {"dataset_file_sha256": artifacts["dataset"]["sha256"]},
        "dataset_validation": {"clean_parent_count": 512},
        "dataset_source_replay": {"all_recorded_sources_reverified": True},
        "dataset_binding_sha256": "9" * 64,
        "cross_platform_gate": gate,
        "cross_platform_gate_claim_sha256": gate["gate_claim_sha256"],
        "phase_a_result_payload_sha256": "a" * 64,
        "phase_a_model_binding_sha256": "b" * 64,
        "phase_a_binding_revalidation": {
            "phase_a_full_pass_reverified": True,
            "source_snapshot_identity": {
                "source_archive_sha256": archive_sha,
                "source_manifest_sha256": "1" * 64,
                "source_tree_sha256": "2" * 64,
            },
        },
        "source_location_semantics": {
            "recorded_phase_a_source_root_is_historical_job_local_execution_path": True
        },
    }
    monkeypatch.setattr(launcher, "inspect_v5_k1_phase_b_source", lambda config, root: source)
    monkeypatch.setattr(
        launcher,
        "inspect_v5_k1_phase_a_inputs",
        lambda config, **kwargs: phase_a,
    )
    return config, dust, source, phase_a


def test_dry_run_is_write_free_and_builds_strict_smoke_formal_afterok_plan(tmp_path, monkeypatch):
    config, dust, source, phase_a = _stubbed_config(tmp_path, monkeypatch)
    result = launch_v5_k1_phase_b_dag(config, allowed_root=dust)
    plan = result["plan"]

    assert result == {
        "status": "dry_run",
        "writes_performed": False,
        "submissions_performed": False,
        "plan": plan,
    }
    assert plan["source"] == source
    assert plan["phase_a_inputs"] == phase_a
    assert plan["stage_order"] == ["engineering_smoke", "formal_gate"]
    assert plan["dependency_edges"] == [["engineering_smoke", "formal_gate"]]
    assert (
        "--dependency=afterok:ENGINEERING_SMOKE_JOB_ID" in plan["submission_preview"]["formal_gate"]
    )
    assert (
        plan["jobs"]["formal_gate"]["environment"]["POSTERIOR_V8_V5_K1_PHASE_B_FORMAL_ACK"] == "YES"
    )
    required_evidence_environment = {
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256",
    }
    for job in plan["jobs"].values():
        assert required_evidence_environment <= set(job["environment"])
        assert (
            job["environment"]["POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT"]
            == source["archive_byte_count"]
        )
        assert job["stdout"].startswith(str(config.run_root / "logs") + "/")
        assert job["stderr"].startswith(str(config.run_root / "logs") + "/")
    assert all("--export=ALL" not in command for command in plan["submission_preview"].values())
    assert "V5_2_R2" in plan["run_root"]
    assert "/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2/" not in json.dumps(plan)
    assert plan["full_k1_all_legal_branches_pending_fail_closed"] is True
    _canonical_sha(plan, "plan_sha256")
    assert not (config.run_root / PHASE_ROOT_NAME).exists()


def test_submit_records_exact_afterok_commands_and_self_hashed_read_only_audits(
    tmp_path, monkeypatch
):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    calls: list[tuple[str, ...]] = []
    replies = iter(("25001001\n", "25001002;maxwell\n"))

    def runner(argv: Sequence[str]) -> V5CommandResult:
        calls.append(tuple(argv))
        return V5CommandResult(0, next(replies), "")

    receipt = launch_v5_k1_phase_b_dag(
        config,
        submit=True,
        runner=runner,
        allowed_root=dust,
        hostname="max-wgs02.desy.de",
        environment={"AWS_SECRET_ACCESS_KEY": "must-not-be-captured"},
    )

    assert receipt["status"] == "submitted"
    assert receipt["job_ids"] == {
        "engineering_smoke": "25001001",
        "formal_gate": "25001002",
    }
    assert len(calls) == 2
    assert calls[0][:2] == ("sbatch", "--parsable")
    assert "--dependency=afterok:25001001" in calls[1]
    assert all(any(item.startswith("--output=") for item in call) for call in calls)
    assert all(any(item.startswith("--error=") for item in call) for call in calls)
    assert "must-not-be-captured" not in repr(calls)
    assert "AWS_SECRET_ACCESS_KEY" not in repr(calls)
    assert receipt["secret_environment_captured"] is False
    _canonical_sha(receipt, "receipt_sha256")

    audit = config.run_root / PHASE_ROOT_NAME / "audit"
    stored_plan = json.loads((audit / PLAN_FILENAME).read_text())
    stored_receipt = json.loads((audit / RECEIPT_FILENAME).read_text())
    assert stored_receipt == receipt
    assert stored_plan["plan_sha256"] == receipt["plan_sha256"]
    assert (audit / PLAN_FILENAME).stat().st_mode & 0o222 == 0
    assert (audit / RECEIPT_FILENAME).stat().st_mode & 0o222 == 0
    assert not Path(stored_plan["layout"]["smoke_output"]).exists()
    assert not Path(stored_plan["layout"]["formal_output"]).exists()
    with pytest.raises(FileExistsError, match="versioned K1 Phase-B root"):
        build_v5_k1_phase_b_launch_plan(config, allowed_root=dust)


def test_non_max_wgs_and_bad_sbatch_are_fail_closed(tmp_path, monkeypatch):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    called = False

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal called
        called = True
        return V5CommandResult(0, "1\n", "")

    with pytest.raises(RuntimeError, match="max-wgs"):
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="local-mac",
        )
    assert called is False
    assert not (config.run_root / PHASE_ROOT_NAME).exists()

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=lambda argv: V5CommandResult(0, "Submitted batch job 12\n", ""),
            allowed_root=dust,
            hostname="max-wgs01",
        )
    receipt = json.loads(raised.value.receipt_path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "engineering_smoke"
    assert receipt["job_ids"] == {}
    _canonical_sha(receipt, "receipt_sha256")


def test_phase_a_drift_between_afterok_submissions_preserves_failure_receipt(tmp_path, monkeypatch):
    config, dust, _, phase_a = _stubbed_config(tmp_path, monkeypatch)
    calls = 0
    inspections = 0

    def inspect(config, **kwargs):
        nonlocal inspections
        inspections += 1
        if inspections < 3:
            return phase_a
        return {**phase_a, "phase_a_result_payload_sha256": "0" * 64}

    monkeypatch.setattr(launcher, "inspect_v5_k1_phase_a_inputs", inspect)

    def runner(argv: Sequence[str]) -> V5CommandResult:
        nonlocal calls
        calls += 1
        return V5CommandResult(0, "25002001\n", "")

    with pytest.raises(V5K1PhaseBLaunchError) as raised:
        launch_v5_k1_phase_b_dag(
            config,
            submit=True,
            runner=runner,
            allowed_root=dust,
            hostname="max-wgs03",
        )
    receipt = json.loads(raised.value.receipt_path.read_text())
    assert calls == 1
    assert receipt["job_ids"] == {"engineering_smoke": "25002001"}
    assert receipt["failure"]["stage"] == "formal_gate"
    assert "Phase-A evidence changed" in receipt["failure"]["message"]


def test_old_diagnostic_run_and_phase_a_roots_are_rejected(tmp_path, monkeypatch):
    config, dust, _, _ = _stubbed_config(tmp_path, monkeypatch)
    old_run = dust / "MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2"
    (old_run / "logs").mkdir(parents=True)
    with pytest.raises(ValueError, match="exactly"):
        build_v5_k1_phase_b_launch_plan(replace(config, run_root=old_run), allowed_root=dust)

    old_dataset, old_sha = _read_only(
        old_run / "k1_phase_a_v5_2_dag_v5/datasets/k1-v5-2-phase-a-v5-full-r512-sphere-v0.gvd5",
        b"old diagnostic dataset",
    )
    with pytest.raises(ValueError, match="exact full-run artifact"):
        resolve_v5_k1_phase_a_input_files(
            replace(
                config,
                phase_a_dataset_path=old_dataset,
                phase_a_dataset_sha256=old_sha,
            ),
            run_root=config.run_root,
            allowed_root=dust,
        )
    assert CURRENT_RUN_ROOT_NAME.endswith("V5_2_R2")
    assert PHASE_A_ROOT_NAME.endswith("dag_v6")


def _synthetic_source_tree(root: Path) -> Path:
    required = {
        Path("AGENTS.md"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("utils/__init__.py"),
        Path("src/gimap/example.py"),
        Path("docs/architecture/example.md"),
        Path("docs/research/example.md"),
        LAUNCHER_TEST_RELATIVE,
        *(Path(value) for value in PHASE_A_SOURCE_PATHS),
        *(Path(value) for value in PHASE_B_SOURCE_PATHS),
    }
    for relative in sorted(required, key=lambda value: value.as_posix()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"immutable test source: {relative.as_posix()}\n")
    return root


def test_archive_snapshot_replay_does_not_depend_on_deleted_phase_a_scratch(tmp_path):
    dust = tmp_path / "data/dust/user/zhaiyufe"
    working = _synthetic_source_tree(tmp_path / "working")
    archive = dust / "source-archives/phase-b-audited.tar"
    archive.parent.mkdir(parents=True)
    archive_identity = build_source_snapshot(working, archive)
    extracted = dust / "source-snapshots/phase-b-audited"
    extracted.parent.mkdir(parents=True)
    extract_source_snapshot(
        archive,
        extracted,
        expected_sha256=archive_identity["archive_sha256"],
    )
    deleted_source = tmp_path / "slurm-24398910/source"
    deleted_dataset = tmp_path / "slurm-24398910/input.gvd5"
    deleted_source.mkdir(parents=True)
    deleted_dataset.write_bytes(b"transient")
    deleted_dataset.unlink()
    deleted_source.rmdir()
    deleted_source.parent.rmdir()
    placeholder = dust / "unused"
    config = V5K1PhaseBLaunchConfig(
        source_root=extracted,
        source_archive_path=archive,
        source_archive_sha256=archive_identity["archive_sha256"],
        run_root=placeholder,
        phase_a_dataset_path=placeholder,
        phase_a_dataset_sha256="0" * 64,
        phase_a_dataset_binding_path=placeholder,
        phase_a_dataset_binding_sha256="0" * 64,
        phase_a_cross_platform_pass_marker_path=placeholder,
        phase_a_cross_platform_pass_marker_sha256="0" * 64,
        phase_a_result_path=placeholder,
        phase_a_result_sha256="0" * 64,
        phase_a_model_path=placeholder,
        phase_a_model_sha256="0" * 64,
        phase_a_model_provenance_path=placeholder,
        phase_a_model_provenance_sha256="0" * 64,
    )

    observed = inspect_v5_k1_phase_b_source(config, dust)
    replayed_phase_a = source_identity(extracted, tuple(sorted(PHASE_A_SOURCE_PATHS)))
    assert observed["archive_sha256"] == archive_identity["archive_sha256"]
    assert observed["manifest_sha256"] == archive_identity["manifest_sha256"]
    assert len(observed["source_tree_sha256"]) == 64
    assert replayed_phase_a["bundle_sha256"]
    assert not deleted_source.exists()
    assert not deleted_dataset.exists()

    selected = extracted / next(iter(sorted(PHASE_A_SOURCE_PATHS)))
    selected.chmod(0o644)
    with pytest.raises(ValueError, match="read-only|identity mismatch"):
        inspect_v5_k1_phase_b_source(config, dust)


def test_slurm_wrapper_revalidates_every_input_before_isolated_worker_execution():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / WRAPPER_RELATIVE.relative_to("utils/ML_Fitting_1D_GISAXS")
    ).read_text()
    required_environment = (
        "POSTERIOR_V8_V5_SOURCE_ARCHIVE",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256",
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER",
        "POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256",
        "POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT",
        "POSTERIOR_V8_V5_K1_PHASE_B_MODE",
    )
    for name in required_environment:
        assert f"${{{name}:?" in wrapper
    assert "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2" in wrapper
    assert "k1_phase_a_v5_2_dag_v6" in wrapper
    assert "k1_phase_a_v5_2_dag_v5" not in wrapper
    assert "PYTHONNOUSERSITE=1" in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert "python -I -c" in wrapper
    assert "python -m utils.ML_Fitting_1D_GISAXS" not in wrapper
    assert 'copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL"' in wrapper
    assert "verify-extracted" in wrapper
    binding = wrapper.index("PosteriorV8.k1_phase_a_dataset_binding_v5")
    worker = wrapper.index("PosteriorV8.run_k1_phase_b_gate_v5")
    assert wrapper.index("verify-extracted") < binding < worker
    assert '--original-dataset-path "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET"' in wrapper
    assert '--original-phase-a-output-dir "$expected_model_root"' in wrapper
    assert '--original-source-root "$POSTERIOR_V8_SOURCE_ROOT"' in wrapper
    assert '--original-source-archive "$POSTERIOR_V8_V5_SOURCE_ARCHIVE"' in wrapper
    assert '--original-phase-a-result "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT"' in wrapper
    assert '--original-phase-a-model "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL"' in wrapper
    assert "verify_checked" in wrapper
    assert "source archive after worker" in wrapper
    assert "dataset binding after worker" in wrapper
    assert "PASS marker after worker" in wrapper
    assert 'exit "$worker_status"' in wrapper


def _worker_input_spec(tmp_path: Path):
    dust = tmp_path / "data/dust/user/zhaiyufe"
    working = _synthetic_source_tree(tmp_path / "authoring")
    archive = dust / "source-archives/audited.tar"
    archive.parent.mkdir(parents=True)
    archive_identity = build_source_snapshot(working, archive)
    source_root = dust / "source-snapshots/audited"
    source_root.parent.mkdir(parents=True)
    source_snapshot = extract_source_snapshot(
        archive, source_root, expected_sha256=archive_identity["archive_sha256"]
    )

    phase_a_output = dust / "MaxwellRuns/audited/models/full"
    dataset, _ = _read_only(dust / "MaxwellRuns/audited/dataset.gvd5", b"dataset")
    result, _ = _read_only(phase_a_output / "result.json", b"result")
    model, _ = _read_only(phase_a_output / "model.keras", b"model")
    provenance, _ = _read_only(phase_a_output / "model.provenance.json", b"provenance")
    authoritative_paths = {
        "source_archive": archive,
        "dataset": dataset,
        "phase_a_result": result,
        "phase_a_model": model,
        "phase_a_model_provenance": provenance,
    }
    expected = {
        name: {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
        for name, path in authoritative_paths.items()
    }

    job_root = tmp_path / "job"
    job_root.mkdir()
    job_archive, _ = _read_only(job_root / "source.tar", archive.read_bytes())
    job_source = job_root / "source"
    extract_source_snapshot(
        job_archive,
        job_source,
        expected_sha256=archive_identity["archive_sha256"],
    )
    local_paths = {}
    for name, original in authoritative_paths.items():
        if name == "source_archive":
            local_paths[name] = job_archive
        else:
            local_paths[name], _ = _read_only(
                job_root / "inputs" / original.name, original.read_bytes()
            )
    spec = V5K1PhaseBWorkerInputSpec(
        allowed_root=dust,
        input_root=job_root,
        original_source_root=source_root,
        job_source_root=job_source,
        phase_a_output_dir=phase_a_output,
        authoritative_paths=authoritative_paths,
        local_paths=local_paths,
        expected=expected,
        source_manifest_sha256=source_snapshot["manifest_sha256"],
        source_tree_sha256=source_snapshot["source_tree_sha256"],
    )
    return spec, authoritative_paths, local_paths


def _replace_read_only(path: Path, content: bytes) -> None:
    path.unlink()
    path.write_bytes(content)
    path.chmod(0o400)


def test_worker_rechecks_authoritative_originals_and_local_copies_after_binding(tmp_path):
    spec, originals, local = _worker_input_spec(tmp_path)
    bound = bind_v5_k1_phase_b_worker_inputs(spec)
    assert bound["authoritative"]["dataset"]["path"] == str(originals["dataset"])
    assert bound["local"]["dataset"]["path"] == str(local["dataset"])

    dataset_bytes = originals["dataset"].read_bytes()
    _replace_read_only(originals["dataset"], dataset_bytes)
    with pytest.raises(RuntimeError, match="changed during Phase-B"):
        assert_v5_k1_phase_b_worker_inputs_unchanged(bound, spec)

    rebound = bind_v5_k1_phase_b_worker_inputs(spec)
    archive_bytes = originals["source_archive"].read_bytes()
    _replace_read_only(originals["source_archive"], archive_bytes)
    with pytest.raises(RuntimeError, match="changed during Phase-B"):
        assert_v5_k1_phase_b_worker_inputs_unchanged(rebound, spec)

    rebound = bind_v5_k1_phase_b_worker_inputs(spec)
    _replace_read_only(originals["phase_a_result"], b"changed-result")
    with pytest.raises(ValueError, match="exported identity"):
        assert_v5_k1_phase_b_worker_inputs_unchanged(rebound, spec)

    _replace_read_only(originals["phase_a_result"], b"result")
    _replace_read_only(local["phase_a_model"], b"changed-local-model")
    with pytest.raises(ValueError, match="job-local phase_a_model"):
        bind_v5_k1_phase_b_worker_inputs(spec)


def test_strict_identity_rejects_final_symlink_and_open_file_replacement(tmp_path, monkeypatch):
    target, _ = _read_only(tmp_path / "target.bin", b"original")
    replacement, _ = _read_only(tmp_path / "replacement.bin", b"replacement")
    real_open = k1_phase_a_cross_platform_v5.os.open

    def replace_with_symlink(path, flags):
        target.unlink()
        target.symlink_to(replacement)
        return real_open(path, flags)

    with monkeypatch.context() as scoped:
        scoped.setattr(k1_phase_a_cross_platform_v5.os, "open", replace_with_symlink)
        with pytest.raises(ValueError, match="symlink before open"):
            k1_phase_a_cross_platform_v5.file_identity(
                target, name="racing input", require_read_only=True
            )

    target.unlink()
    target.write_bytes(b"original")
    target.chmod(0o400)
    real_read = k1_phase_a_cross_platform_v5.os.read
    replacement_done = False

    def replace_while_reading(descriptor, byte_count):
        nonlocal replacement_done
        if not replacement_done:
            replacement_done = True
            _replace_read_only(target, b"replacement")
        return real_read(descriptor, byte_count)

    with monkeypatch.context() as scoped:
        scoped.setattr(k1_phase_a_cross_platform_v5.os, "read", replace_while_reading)
        with pytest.raises(RuntimeError, match="changed while it was hashed"):
            k1_phase_a_cross_platform_v5.file_identity(
                target, name="racing input", require_read_only=True
            )
