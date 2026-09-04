"""Write-free immutable-input inspection for the K1 Phase-B launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Mapping

from .grouped_dataset_v5 import read_v5_grouped_dataset
from .k1_phase_a_cross_platform_v5 import (
    file_identity as strict_file_identity,
    validate_v5_k1_cross_platform_marker_file,
)
from .k1_phase_a_dataset_binding_v5 import (
    validate_v5_k1_phase_a_dataset_binding_file,
)
from .k1_phase_a_contract_v7 import (
    validate_plan as validate_phase_a_plan,
    validate_receipt as validate_phase_a_receipt,
    validate_release as validate_phase_a_release,
    validate_stage_completion as validate_phase_a_stage_completion,
)
from .k1_staging_files_v5 import read_only_json
from .k1_phase_b_contract_v5 import (
    PHASE_A_SOURCE_PATHS,
    PHASE_B_SOURCE_PATHS,
    dataset_identity,
    digest,
    live_gate_contract,
    source_identity,
    strict_json_object,
    validate_v5_k1_phase_a_bindings,
    verify_recorded_dataset_sources,
)
from .launch_k1_phase_a_dag_v5 import (
    CURRENT_RUN_ROOT_NAME,
    PLAN_FILENAME as PHASE_A_PLAN_FILENAME,
    PHASE_ROOT_NAME as PHASE_A_ROOT_NAME,
    RECEIPT_FILENAME as PHASE_A_RECEIPT_FILENAME,
    RELEASE_COMPLETION_FILENAME as PHASE_A_RELEASE_FILENAME,
)
from .package_source_snapshot_v5 import verify_extracted_source_snapshot
from .run_k1_memorization_gate_v5 import (
    MODEL_FILENAME as PHASE_A_MODEL_FILENAME,
    MODEL_PROVENANCE_FILENAME as PHASE_A_MODEL_PROVENANCE_FILENAME,
    RESULT_FILENAME as PHASE_A_RESULT_FILENAME,
    validate_v5_k1_memorization_dataset,
)


MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
_POSTERIOR_ROOT = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
PHASE_B_WRAPPER = _POSTERIOR_ROOT / "slurm/v5_k1_phase_b_gate_cpu.sbatch"
_LAUNCHER = _POSTERIOR_ROOT / "launch_k1_phase_b_dag_v5.py"
_LAUNCHER_TEST = Path(
    "utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_b_launcher_v5.py"
)
_REQUIRED_SOURCE = tuple(
    sorted(
        {
            _LAUNCHER,
            PHASE_B_WRAPPER,
            _LAUNCHER_TEST,
            *(Path(value) for value in PHASE_B_SOURCE_PATHS),
        },
        key=lambda value: value.as_posix(),
    )
)
_PHASE_A_FULL_DATASET = "k1-v5-2-phase-a-v5-full-r512-sphere-v0.gvd5"
_PHASE_A_FULL_DATASET_BINDING = _PHASE_A_FULL_DATASET + ".binding-v1.json"
_PHASE_A_FULL_MODEL_DIRECTORY = "k1-v5-2-phase-a-v5-full-steps1500"


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseBLaunchConfig:
    source_root: Path
    source_archive_path: Path
    source_archive_sha256: str
    run_root: Path
    phase_a_dataset_path: Path
    phase_a_dataset_sha256: str
    phase_a_dataset_binding_path: Path
    phase_a_dataset_binding_sha256: str
    phase_a_cross_platform_pass_marker_path: Path
    phase_a_cross_platform_pass_marker_sha256: str
    phase_a_result_path: Path
    phase_a_result_sha256: str
    phase_a_model_path: Path
    phase_a_model_sha256: str
    phase_a_model_provenance_path: Path
    phase_a_model_provenance_sha256: str


def _lexical_absolute(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("Maxwell paths must be absolute")
    return Path(os.path.abspath(path))


def _under_dust(path: Path, allowed_root: Path, name: str, *, must_exist: bool) -> Path:
    root = allowed_root.resolve(strict=True)
    lexical = _lexical_absolute(path)
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}") from exc
    if not relative.parts:
        raise ValueError(f"{name} must not be the dust root itself")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
        if not current.exists():
            break
    resolved = lexical.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} resolves outside {root}")
    return resolved


def resolve_v5_k1_phase_b_run_root(run_root: Path, allowed_root: Path) -> Path:
    resolved = _under_dust(run_root, allowed_root, "run_root", must_exist=True)
    expected = (allowed_root.resolve(strict=True) / "MaxwellRuns" / CURRENT_RUN_ROOT_NAME).resolve(
        strict=True
    )
    if resolved != expected or not resolved.is_dir():
        raise ValueError(f"run_root must be exactly {expected}")
    logs = resolved / "logs"
    if not logs.is_dir() or logs.is_symlink():
        raise FileNotFoundError("the versioned run root must contain a real logs directory")
    return resolved


def inspect_v5_k1_phase_b_source(
    config: V5K1PhaseBLaunchConfig, allowed_root: Path
) -> dict[str, object]:
    source_root = _under_dust(config.source_root, allowed_root, "source_root", must_exist=True)
    archive = _under_dust(
        config.source_archive_path,
        allowed_root,
        "source_archive_path",
        must_exist=True,
    )
    expected_archive_sha256 = digest(config.source_archive_sha256, "source archive SHA")
    verified = verify_extracted_source_snapshot(
        archive,
        source_root,
        expected_archive_sha256=expected_archive_sha256,
    )
    required_files = {}
    for relative in _REQUIRED_SOURCE:
        target = source_root / relative
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(f"missing immutable Phase-B source: {target}")
        required_files[relative.as_posix()] = strict_file_identity(
            target,
            name=f"Phase-B source {relative.as_posix()}",
            require_read_only=True,
        )
    return {
        **verified,
        "required_file_identity": required_files,
        "snapshot_name": source_root.name,
    }


def _expected_file(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
    allowed_root: Path,
) -> tuple[Path, dict[str, object]]:
    resolved = _under_dust(path, allowed_root, name, must_exist=True)
    identity = strict_file_identity(resolved, name=name, require_read_only=True)
    if identity["sha256"] != digest(expected_sha256, f"{name} expected SHA"):
        raise ValueError(f"{name} SHA-256 does not match the external expectation")
    return resolved, identity


def _portable_file_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {name: value[name] for name in ("sha256", "byte_count")}


def _portable_read_only_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {name: value[name] for name in ("sha256", "byte_count", "mode")}


def _marker_expected(training_evidence: Mapping[str, object]) -> dict[str, object]:
    source = training_evidence["source_snapshot"]
    gate = training_evidence["cross_platform_gate"]
    return {
        "expected_source": dict(source),
        "reference_file_sha256": gate["reference_file_sha256"],
        "reference_file_byte_count": gate["reference_file_byte_count"],
        "reference_manifest_sha256": gate["reference_manifest_sha256"],
        "scientific_content_sha256": gate["scientific_content_sha256"],
        "comparison_result_sha256": gate["comparison_result_sha256"],
        "expected_gate_claim_sha256": gate["gate_claim_sha256"],
    }


def expected_v5_k1_phase_a_paths(run_root: Path) -> dict[str, Path]:
    phase_a_root = run_root / PHASE_A_ROOT_NAME
    dataset = phase_a_root / "datasets" / _PHASE_A_FULL_DATASET
    model_root = phase_a_root / "models" / _PHASE_A_FULL_MODEL_DIRECTORY
    return {
        "dataset": dataset,
        "dataset_binding": phase_a_root / "datasets" / _PHASE_A_FULL_DATASET_BINDING,
        "cross_platform_pass_marker": phase_a_root / "audit" / "sobol-cross-platform-PASS-v1.json",
        "result": model_root / PHASE_A_RESULT_FILENAME,
        "model": model_root / PHASE_A_MODEL_FILENAME,
        "model_provenance": model_root / PHASE_A_MODEL_PROVENANCE_FILENAME,
        "phase_a_completion": phase_a_root / "audit/full-gate-completion-v1.json",
    }


def resolve_v5_k1_phase_a_input_files(
    config: V5K1PhaseBLaunchConfig,
    *,
    run_root: Path,
    allowed_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    supplied = {
        "dataset": (config.phase_a_dataset_path, config.phase_a_dataset_sha256),
        "dataset_binding": (
            config.phase_a_dataset_binding_path,
            config.phase_a_dataset_binding_sha256,
        ),
        "cross_platform_pass_marker": (
            config.phase_a_cross_platform_pass_marker_path,
            config.phase_a_cross_platform_pass_marker_sha256,
        ),
        "result": (config.phase_a_result_path, config.phase_a_result_sha256),
        "model": (config.phase_a_model_path, config.phase_a_model_sha256),
        "model_provenance": (
            config.phase_a_model_provenance_path,
            config.phase_a_model_provenance_sha256,
        ),
    }
    expected_paths = expected_v5_k1_phase_a_paths(run_root)
    paths: dict[str, Path] = {}
    identities: dict[str, dict[str, object]] = {}
    for name, (raw_path, expected_sha) in supplied.items():
        path, identity = _expected_file(
            raw_path,
            expected_sha,
            name=f"Phase-A {name}",
            allowed_root=allowed_root,
        )
        if path != expected_paths[name]:
            raise ValueError(
                f"Phase-A {name} must be the exact full-run artifact {expected_paths[name]}"
            )
        paths[name] = path
        identities[name] = identity
    completion = expected_paths["phase_a_completion"]
    completion_identity = strict_file_identity(
        completion,
        name="Phase-A full-gate completion",
        require_read_only=True,
    )
    paths["phase_a_completion"] = completion
    identities["phase_a_completion"] = completion_identity
    return paths, identities


def inspect_v5_k1_phase_a_inputs(
    config: V5K1PhaseBLaunchConfig,
    *,
    source: Mapping[str, object],
    run_root: Path,
    allowed_root: Path,
) -> dict[str, object]:
    paths, identities = resolve_v5_k1_phase_a_input_files(
        config, run_root=run_root, allowed_root=allowed_root
    )

    phase_a_root = run_root / PHASE_A_ROOT_NAME
    phase_a_plan, _ = read_only_json(
        phase_a_root / "audit" / PHASE_A_PLAN_FILENAME,
        "Phase-A launch plan",
    )
    phase_a_plan = validate_phase_a_plan(phase_a_plan)
    phase_a_receipt, _ = read_only_json(
        phase_a_root / "audit" / PHASE_A_RECEIPT_FILENAME,
        "Phase-A submission receipt",
    )
    phase_a_receipt = validate_phase_a_receipt(phase_a_receipt, phase_a_plan)
    phase_a_release, _ = read_only_json(
        phase_a_root / "audit" / PHASE_A_RELEASE_FILENAME,
        "Phase-A release completion",
    )
    phase_a_release = validate_phase_a_release(
        phase_a_release, phase_a_plan, phase_a_receipt
    )
    phase_a_completion, completion_file_identity = read_only_json(
        paths["phase_a_completion"], "Phase-A full-gate completion"
    )
    phase_a_completion = validate_phase_a_stage_completion(
        phase_a_completion,
        expected_stage="full_gate",
        plan=phase_a_plan,
        receipt=phase_a_receipt,
        release=phase_a_release,
    )

    phase_a_result = strict_json_object(
        paths["result"].read_text(encoding="utf-8"), "Phase-A result"
    )
    model_provenance = strict_json_object(
        paths["model_provenance"].read_text(encoding="utf-8"),
        "Phase-A model provenance",
    )
    dataset, dataset_receipt = read_v5_grouped_dataset(paths["dataset"])
    if _portable_file_identity(identities["dataset"]) != {
        "sha256": dataset_receipt.artifact_sha256,
        "byte_count": dataset_receipt.byte_count,
    }:
        raise ValueError("Phase-A dataset checked-artifact receipt does not bind its file")
    dataset_validation = validate_v5_k1_memorization_dataset(dataset)
    checked_dataset = dataset_identity(paths["dataset"], dataset, dataset_receipt)
    recorded_dataset_sources = verify_recorded_dataset_sources(
        dataset, Path(str(source["source_root"]))
    )
    phase_a_source = source_identity(
        Path(str(source["source_root"])), tuple(sorted(PHASE_A_SOURCE_PATHS))
    )
    protocol = live_gate_contract()
    nested_result = phase_a_result.get("stage_a_memorization_result")
    if not isinstance(nested_result, Mapping):
        raise ValueError("Phase-A nested memorization result is missing")
    snapshot_binding = {
        "source_archive_sha256": source["archive_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_tree_sha256": source["source_tree_sha256"],
    }
    phase_a_binding = validate_v5_k1_phase_a_bindings(
        phase_a_result,
        dataset_identity=checked_dataset,
        model_identity=_portable_file_identity(identities["model"]),
        model_provenance_payload=model_provenance,
        model_provenance_identity=_portable_file_identity(identities["model_provenance"]),
        phase_a_source_identity=phase_a_source,
        source_snapshot_identity=snapshot_binding,
        model_weights_sha256_value=digest(
            nested_result.get("final_weights_sha256"),
            "Phase-A recorded model weights SHA",
        ),
        live_gate_contract_value=protocol,
    )
    if (
        phase_a_result["launch_binding"] != phase_a_completion["launch_binding"]
        or phase_a_result["job_local_capability"]
        != phase_a_completion["job_local_capability"]
    ):
        raise ValueError("Phase-A result does not match its full-gate completion")
    completion_roles = {
        item["role"]: item for item in phase_a_completion["artifacts"]
    }
    expected_completion_artifacts = {
        "k1_dataset": "dataset",
        "k1_dataset_binding": "dataset_binding",
        "cross_platform_pass_marker": "cross_platform_pass_marker",
        "k1_model": "model",
        "k1_model_provenance": "model_provenance",
        "k1_gate_result": "result",
    }
    if set(completion_roles) != set(expected_completion_artifacts):
        raise ValueError("Phase-A full-gate completion artifact inventory drifted")
    for role, artifact_name in expected_completion_artifacts.items():
        item = completion_roles[role]
        if Path(str(item["path"])).resolve() != paths[artifact_name].resolve():
            raise ValueError(f"Phase-A completion {role} path drifted")
        if item["identity"]["sha256"] != identities[artifact_name]["sha256"]:
            raise ValueError(f"Phase-A completion {role} content drifted")

    training_evidence = phase_a_result["training_evidence"]
    marker_expected = _marker_expected(training_evidence)
    marker = validate_v5_k1_cross_platform_marker_file(
        paths["cross_platform_pass_marker"], **marker_expected
    )
    binding = validate_v5_k1_phase_a_dataset_binding_file(
        paths["dataset_binding"],
        dataset_path=paths["dataset"],
        marker_path=paths["cross_platform_pass_marker"],
        expected_original_dataset_path=str(paths["dataset"]),
        marker_expected=marker_expected,
    )
    expected_binding = training_evidence["dataset_completion_binding"]
    if (
        binding["binding_sha256"] != expected_binding["binding_sha256"]
        or _portable_read_only_identity(identities["dataset_binding"]) != expected_binding["file"]
    ):
        raise ValueError("Phase-A result does not bind the supplied dataset binding")
    expected_gate = training_evidence["cross_platform_gate"]
    if (
        marker["marker"]["marker_sha256"] != expected_gate["pass_marker_sha256"]
        or _portable_read_only_identity(identities["cross_platform_pass_marker"])
        != expected_gate["pass_marker_file"]
    ):
        raise ValueError("Phase-A result does not bind the supplied PASS marker")

    for name, path in paths.items():
        after = strict_file_identity(path, name=f"Phase-A {name}", require_read_only=True)
        if after != identities[name]:
            raise RuntimeError(f"Phase-A {name} changed during launcher validation")
    return {
        "artifacts": {
            name: {"path": str(paths[name]), **identities[name]} for name in sorted(paths)
        },
        "dataset": checked_dataset,
        "dataset_validation": dataset_validation,
        "dataset_source_replay": recorded_dataset_sources,
        "dataset_binding_sha256": binding["binding_sha256"],
        "cross_platform_gate": {
            name: training_evidence["cross_platform_gate"][name]
            for name in (
                "gate_claim_sha256",
                "reference_file_sha256",
                "reference_file_byte_count",
                "reference_manifest_sha256",
                "scientific_content_sha256",
                "comparison_result_sha256",
            )
        },
        "cross_platform_gate_claim_sha256": marker["marker"]["gate_claim_sha256"],
        "phase_a_result_payload_sha256": phase_a_result["result_payload_sha256"],
        "phase_a_model_binding_sha256": model_provenance["binding_sha256"],
        "phase_a_completion": {
            "completion_sha256": phase_a_completion["completion_sha256"],
            "file_identity": completion_file_identity,
            "launch_binding_sha256": phase_a_completion["launch_binding_sha256"],
            "slurm_job_id": phase_a_completion["slurm_job_id"],
        },
        "phase_a_binding_revalidation": phase_a_binding,
        "source_location_semantics": {
            "recorded_phase_a_source_root_is_historical_job_local_execution_path": True,
            "recorded_phase_a_source_root": phase_a_result["source"]["source_root"],
            "replayed_immutable_source_root": source["source_root"],
            "identity_anchor": (
                "archive_manifest_tree_triple_plus_exact_phase_a_file_inventory_and_bundle"
            ),
        },
    }


__all__ = [
    "CURRENT_RUN_ROOT_NAME",
    "MAXWELL_DUST_ROOT",
    "PHASE_B_WRAPPER",
    "V5K1PhaseBLaunchConfig",
    "expected_v5_k1_phase_a_paths",
    "inspect_v5_k1_phase_a_inputs",
    "inspect_v5_k1_phase_b_source",
    "resolve_v5_k1_phase_a_input_files",
    "resolve_v5_k1_phase_b_run_root",
]
