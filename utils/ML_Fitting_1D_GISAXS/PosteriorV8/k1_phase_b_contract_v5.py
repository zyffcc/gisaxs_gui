"""Identity, receipt, and fail-closed DTO contract for K1 Phase-B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

import numpy as np

from .candidate_refinement_contract_v5 import V5_EXACT_FORWARD_BUDGET_UNIT
from .grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from .grouped_dataset_v5 import V5GroupedDataset
from .k1_phase_a_capability_v7 import CAPABILITY_SCHEMA as PHASE_A_CAPABILITY_SCHEMA
from .k1_phase_a_contract_v7 import validate_launch_binding_payload
from .memorization_gate_v5 import (
    V5_MEMORIZATION_GATE_SCHEMA,
    V5_MEMORIZATION_GATE_VERSION,
)
from .model_v5_contract import (
    MODEL_V5_NAME,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
    model_v5_contract_payload,
)
from .run_k1_memorization_gate_v5 import (
    MODEL_FILENAME as PHASE_A_MODEL_FILENAME,
    MODEL_PROVENANCE_FILENAME as PHASE_A_MODEL_PROVENANCE_FILENAME,
    V5_K1_MODEL_PROVENANCE_SCHEMA,
    V5_K1_MODEL_PROVENANCE_VERSION,
    V5_K1_TRAINING_EVIDENCE_SCHEMA,
    V5_K1_TRAINING_EVIDENCE_VERSION,
    V5_K1_DATASET_GATE_PUBLISHED_RESULT_FIELDS,
    V5_K1_DATASET_GATE_ROLE,
    V5_K1_DATASET_GATE_SCHEMA,
    V5_K1_DATASET_GATE_VERSION,
)
from .study_protocol import protocol_payload, validate_protocol


V5_K1_PHASE_B_SCHEMA = "gisaxs.posterior_v8.k1_single_branch_phase_b_gate/v12"
V5_K1_PHASE_B_VERSION = (
    "posterior_v8_v5_2_closed_interval_tolerance_single_branch_draw32_gate_v12"
)
V5_K1_PHASE_B_ROLE = (
    "k1_single_branch_capacity_and_objective_diagnostic_not_full_k1_or_model_acceptance"
)
V5_K1_PHASE_B_RESULT_FILENAME = "phase-b-result.json"
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
PROPOSAL_COUNT = 32
SINGLE_DRAW_INDEX = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS = (
    "path",
    "sha256",
    "byte_count",
    "mode",
    "inode",
    "mtime_ns",
    "ctime_ns",
    "nlink",
)
FULL_FILE_IDENTITY_FIELDS = (*CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS, "device")
EXPECTED_GATE_KEYS = (
    "branch_conditioned_local_mdn_single_draw_local_rms_median_lt",
    "branch_conditioned_local_mdn_best_of_32_local_rms_median_lt",
    "branch_conditioned_local_mdn_best_of_32_local_rms_p90_lt",
    "exact_post_refine_raw_log_rmse_p90_lt",
    "exact_post_refine_compatible_rate_gte",
)


def cross_node_stable_file_identity(value: object) -> dict[str, object] | None:
    """Canonicalize a complete local file identity without mount-local ``st_dev``."""

    if not isinstance(value, Mapping) or set(value) != set(FULL_FILE_IDENTITY_FIELDS):
        return None
    return {field: value[field] for field in CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS}


def validate_cross_node_stable_file_identity(
    value: object,
) -> dict[str, object] | None:
    """Accept only the exact canonical cross-node identity field set."""

    if not isinstance(value, Mapping) or set(value) != set(
        CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS
    ):
        return None
    return dict(value)
PHASE_A_MEMORIZATION_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "version",
        "scientific_role",
        "initial_loss",
        "final_loss",
        "initial_target_median_rms",
        "final_target_median_rms",
        "target_count",
        "learnable_target_count",
        "fully_fixed_target_count",
        "varying_coordinate_count",
        "model_input_keys",
        "objective_audit_sha256",
        "trajectory_sha256",
        "final_weights_sha256",
        "passed",
        "config",
        "result_sha256",
    }
)
PHASE_A_SINGLE_BRANCH_CONTRACT_FIELDS = frozenset(
    {
        "criteria_source",
        "phase_b_frozen_criteria",
        "topology_schedule",
        "authoritative_forward_version",
        "stage_a_evaluated_metrics",
        "stage_a_metric_semantics",
        "stage_b_pending_metrics",
        "stage_b_status",
        "single_branch_phase_b_gate_passed",
        "complete_k1_proposal_exact_gate_passed",
        "full_k1_all_legal_branches_gate_status",
    }
)
_POSTERIOR_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
PHASE_A_SOURCE_PATHS = frozenset(
    {
        *(
            (_POSTERIOR_RELATIVE / path.name).as_posix()
            for path in Path(__file__).resolve().parent.glob("*.py")
        ),
        "src/gimap/features/fitting/domain/scattering_model.py",
        "src/gimap/features/fitting/domain/physical_constraints.py",
    }
)


def _validate_phase_a_training_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Phase-A training evidence is missing")
    evidence = dict(value)
    expected_fields = {
        "schema",
        "version",
        "status",
        "source_snapshot",
        "dataset_completion_binding",
        "cross_platform_gate",
        "worker_validation",
        "evidence_sha256",
    }
    if set(evidence) != expected_fields:
        raise ValueError("Phase-A training evidence is incomplete or unsupported")
    core = dict(evidence)
    supplied = digest(core.pop("evidence_sha256"), "Phase-A training evidence SHA")
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != supplied:
        raise ValueError("Phase-A training evidence SHA-256 does not reproduce")
    if (
        evidence["schema"] != V5_K1_TRAINING_EVIDENCE_SCHEMA
        or evidence["version"] != V5_K1_TRAINING_EVIDENCE_VERSION
        or evidence["status"] != "VALIDATED"
        or evidence["worker_validation"]
        != "strict_local_dataset_binding_and_pass_marker_replayed_by_training_process"
    ):
        raise ValueError("Phase-A training evidence identity is incompatible")
    source = evidence["source_snapshot"]
    if not isinstance(source, Mapping) or set(source) != {
        "source_archive_sha256",
        "source_manifest_sha256",
        "source_tree_sha256",
    }:
        raise ValueError("Phase-A source snapshot triple is incomplete")
    for name, raw in source.items():
        digest(raw, f"Phase-A {name}")
    dataset_binding = evidence["dataset_completion_binding"]
    gate = evidence["cross_platform_gate"]
    if not isinstance(dataset_binding, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("Phase-A dataset or cross-platform evidence is missing")
    if set(dataset_binding) != {"binding_sha256", "file", "dataset"}:
        raise ValueError("Phase-A dataset completion evidence is unsupported")
    bound_dataset = dataset_binding.get("dataset")
    if not isinstance(bound_dataset, Mapping) or set(bound_dataset) != {
        "original_path",
        "artifact_sha256",
        "byte_count",
        "dataset_id",
        "dataset_schema",
        "dataset_version",
        "manifest_sha256",
    }:
        raise ValueError("Phase-A bound dataset identity is incomplete")
    for name in ("artifact_sha256", "manifest_sha256"):
        digest(bound_dataset[name], f"Phase-A bound dataset {name}")
    if set(gate) != {
        "gate_claim_sha256",
        "pass_marker_sha256",
        "pass_marker_file",
        "reference_file_sha256",
        "reference_file_byte_count",
        "reference_manifest_sha256",
        "scientific_content_sha256",
        "comparison_result_sha256",
    }:
        raise ValueError("Phase-A cross-platform evidence is unsupported")
    digest(dataset_binding.get("binding_sha256"), "Phase-A dataset binding SHA")
    for name, file_value in (
        ("dataset binding file", dataset_binding.get("file")),
        ("PASS marker file", gate.get("pass_marker_file")),
    ):
        if not isinstance(file_value, Mapping) or set(file_value) != {
            "sha256",
            "byte_count",
            "mode",
        }:
            raise ValueError(f"Phase-A {name} identity is incomplete")
        digest(file_value["sha256"], f"Phase-A {name} SHA")
        if (
            isinstance(file_value["byte_count"], bool)
            or not isinstance(file_value["byte_count"], Integral)
            or int(file_value["byte_count"]) < 1
            or file_value["mode"] != 0o400
        ):
            raise ValueError(f"Phase-A {name} is empty or not owner-read-only")
    for name in (
        "gate_claim_sha256",
        "pass_marker_sha256",
        "reference_file_sha256",
        "reference_manifest_sha256",
        "scientific_content_sha256",
        "comparison_result_sha256",
    ):
        digest(gate.get(name), f"Phase-A {name}")
    reference_bytes = gate.get("reference_file_byte_count")
    if (
        isinstance(reference_bytes, bool)
        or not isinstance(reference_bytes, Integral)
        or int(reference_bytes) < 1
    ):
        raise ValueError("Phase-A reference file byte count must be positive")
    return evidence


def _validate_phase_a_model_provenance(
    value: object,
    *,
    model_identity: Mapping[str, object],
    training_evidence: Mapping[str, object],
    launch_binding: Mapping[str, object],
    job_local_capability: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "version",
        "status",
        "model",
        "training_evidence",
        "launch_binding",
        "job_local_capability",
        "binding_sha256",
    }:
        raise ValueError("Phase-A model provenance is incomplete or unsupported")
    provenance = dict(value)
    core = dict(provenance)
    supplied = digest(core.pop("binding_sha256"), "Phase-A model provenance SHA")
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != supplied:
        raise ValueError("Phase-A model provenance SHA-256 does not reproduce")
    if (
        provenance["schema"] != V5_K1_MODEL_PROVENANCE_SCHEMA
        or provenance["version"] != V5_K1_MODEL_PROVENANCE_VERSION
        or provenance["status"] != "BOUND_TO_INPUTS_PENDING_STAGE_COMPLETION"
    ):
        raise ValueError("Phase-A model provenance identity is incompatible")
    expected_model = {
        "filename": PHASE_A_MODEL_FILENAME,
        **dict(model_identity),
        "reload_graph_contract_passed": True,
    }
    if provenance["model"] != expected_model:
        raise ValueError("Phase-A model provenance does not bind the consumed model")
    if provenance["training_evidence"] != dict(training_evidence):
        raise ValueError("Phase-A model provenance training evidence drift detected")
    if provenance["launch_binding"] != dict(launch_binding):
        raise ValueError("Phase-A model provenance launch binding drift detected")
    if provenance["job_local_capability"] != dict(job_local_capability):
        raise ValueError("Phase-A model provenance capability binding drift detected")
    return provenance


def _validate_phase_a_job_local_capability(
    value: object, *, launch_binding: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "capability",
        "pre_use_rehash_sha256",
        "post_use_rehash_sha256",
        "pre_post_equal",
    }:
        raise ValueError("Phase-A job-local capability is incomplete or unsupported")
    payload = dict(value)
    pre = digest(payload["pre_use_rehash_sha256"], "Phase-A capability pre-use SHA")
    post = digest(payload["post_use_rehash_sha256"], "Phase-A capability post-use SHA")
    if payload["pre_post_equal"] is not True or pre != post:
        raise ValueError("Phase-A job-local capability did not survive pre/post revalidation")
    audit = payload["capability"]
    if not isinstance(audit, Mapping) or set(audit) != {
        "schema",
        "stage",
        "slurm_job_id",
        "launch_binding_sha256",
        "staged_inputs",
        "pre_mint_rehash_sha256",
        "authorization",
        "capability_sha256",
    }:
        raise ValueError("Phase-A capability audit is incomplete or unsupported")
    capability = dict(audit)
    supplied = digest(capability.pop("capability_sha256"), "Phase-A capability SHA")
    if sha256(canonical_json(capability).encode("utf-8")).hexdigest() != supplied:
        raise ValueError("Phase-A capability SHA-256 does not reproduce")
    if (
        audit["schema"] != PHASE_A_CAPABILITY_SCHEMA
        or audit["stage"] != "full_gate"
        or audit["stage"] != launch_binding["stage"]
        or audit["slurm_job_id"] != launch_binding["slurm_job_id"]
        or audit["launch_binding_sha256"] != launch_binding["binding_sha256"]
        or audit["authorization"]
        != {
            "live_registry_required": True,
            "single_use": True,
            "serialized_payload_authorizes_use": False,
            "job_local_read_only_single_link_inputs": True,
        }
    ):
        raise ValueError("Phase-A capability launch or authorization binding drifted")
    rows = audit["staged_inputs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Phase-A capability staged input inventory is missing")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"role", "path", "identity"}:
            raise ValueError("Phase-A capability staged input row is invalid")
        if not isinstance(row["role"], str) or not row["role"]:
            raise ValueError("Phase-A capability staged input role is invalid")
        if not isinstance(row["path"], str) or not Path(row["path"]).is_absolute():
            raise ValueError("Phase-A capability staged input path is invalid")
        identity = row["identity"]
        if not isinstance(identity, Mapping) or set(identity) != {
            "sha256",
            "byte_count",
            "mode_octal",
            "uid",
            "gid",
            "link_count",
        }:
            raise ValueError("Phase-A capability staged input identity is invalid")
        digest(identity["sha256"], "Phase-A staged input SHA")
        if identity["mode_octal"] != "0400" or identity["link_count"] != 1:
            raise ValueError("Phase-A capability input was not read-only and single-link")
    rows_sha = sha256(canonical_json(rows).encode("utf-8")).hexdigest()
    if audit["pre_mint_rehash_sha256"] != rows_sha or pre != rows_sha:
        raise ValueError("Phase-A capability staged-input rehash does not reproduce")
    return payload
PHASE_B_WORKER_RELATIVE_PATH = (
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/run_k1_phase_b_gate_v5.py"
)
PHASE_B_SOURCE_PATHS = (
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/launch_k1_phase_b_dag_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_launch_inputs_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_worker_inputs_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/immutable_submission_file_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_publication_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_capability_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_launch_chain_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_a_cross_platform_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_a_dataset_binding_v5.py",
    PHASE_B_WORKER_RELATIVE_PATH,
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_contract_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_b_evaluation_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/candidate_batch_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/candidate_proposals_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/candidate_refinement_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/candidate_refinement_contract_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/profiled_forward.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/profiled_refinement.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/gui_amplitude_constraints.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/clean_recipe_forward_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/persisted_clean_recipe_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/observation_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/synthetic_recipe_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_artifact_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_dataset_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model_v5_contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/evaluation.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_codec.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_query_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/amplitude_query_v5.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/v5_k1_phase_b_gate_cpu.sbatch",
    "src/gimap/features/fitting/domain/scattering_model.py",
)


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseBGateConfig:
    seed: int = 20260903
    per_candidate_forward_evaluation_limit: int = 128
    per_parent_forward_evaluation_limit: int = 4096
    progress_interval: int = 16
    parent_limit: int | None = None

    def __post_init__(self) -> None:
        seed = positive_integer(self.seed, "seed")
        per_candidate = positive_integer(
            self.per_candidate_forward_evaluation_limit,
            "per_candidate_forward_evaluation_limit",
        )
        per_parent = positive_integer(
            self.per_parent_forward_evaluation_limit,
            "per_parent_forward_evaluation_limit",
        )
        progress = positive_integer(self.progress_interval, "progress_interval")
        parent_limit = (
            None
            if self.parent_limit is None
            else positive_integer(self.parent_limit, "parent_limit")
        )
        if per_parent < PROPOSAL_COUNT * per_candidate:
            raise ValueError(
                "per-parent exact budget must reserve the full configured limit for all 32 seeds"
            )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "per_candidate_forward_evaluation_limit", per_candidate)
        object.__setattr__(self, "per_parent_forward_evaluation_limit", per_parent)
        object.__setattr__(self, "progress_interval", progress)
        object.__setattr__(self, "parent_limit", parent_limit)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseBParentRecord:
    clean_parent_sha256: str
    candidate_query_sha256: str
    geometry_query_sha256: str
    amplitude_query_sha256: str
    topology_id: int
    pattern_id: int
    frozen_seed: int
    varying_dimension_count: int
    proposal_count: int
    single_draw_index: int
    single_draw_local_rms: float | None
    best_of_32_local_rms: float | None
    exact_best_raw_log_rmse: float | None
    exact_compatible: bool
    refinement_status: str
    all_input_seeds_processed: bool
    configured_total_limit: int
    configured_per_candidate_limit: int
    exact_calls_used: int
    exact_calls_remaining: int
    exact_calls_by_phase: tuple[tuple[str, int], ...]
    attempts_recorded: int
    refinement_successes: int
    validation_failures: int
    refinement_failures: int
    per_candidate_budget_exhausted_attempts: int
    total_budget_exhausted_before_seed_attempts: int
    bounds_compliant_attempts: int
    physics_compliant_attempts: int
    amplitude_compliant_attempts: int
    model_output_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["exact_calls_by_phase"] = [list(value) for value in self.exact_calls_by_phase]
        return result


def strict_json_object(encoded: str, name: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {name} field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def digest(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def file_identity(path: Path) -> dict[str, object]:
    result = sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
            byte_count += len(chunk)
    return {"sha256": result.hexdigest(), "byte_count": byte_count}


def under_root(path: Path, root: Path, name: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{name} must be below {resolved_root}")
    return resolved


def source_identity(source_root: Path, paths: Sequence[str]) -> dict[str, object]:
    root = source_root.expanduser().resolve()
    files = {}
    for name in paths:
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise FileNotFoundError(f"source identity path is missing or unsafe: {name}")
        files[name] = file_identity(target)
    return {
        "source_root": str(root),
        "files": files,
        "bundle_sha256": sha256(canonical_json(files).encode("utf-8")).hexdigest(),
    }


def _validated_source_content_identity(
    value: object, *, name: str
) -> dict[str, object]:
    """Validate the portable part of a recorded source identity.

    Phase-A executes from a job-local extracted tree which is intentionally
    disposable.  Its recorded ``source_root`` is therefore execution
    provenance, not the long-lived identity anchor.  The exact per-file
    inventory and its self-consistent bundle hash are portable and are what a
    later phase can replay from the immutable archive/snapshot pair.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "source_root",
        "files",
        "bundle_sha256",
    }:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    execution_root = value["source_root"]
    if not isinstance(execution_root, str) or not Path(execution_root).is_absolute():
        raise ValueError(f"{name} execution source root must be an absolute path")
    raw_files = value["files"]
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise ValueError(f"{name} file inventory is missing")
    files: dict[str, dict[str, object]] = {}
    for raw_path, raw_identity in raw_files.items():
        if not isinstance(raw_path, str):
            raise ValueError(f"{name} source path must be a string")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw_path:
            raise ValueError(f"{name} contains an unsafe source path")
        if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"{name} file identity is incomplete")
        byte_count = raw_identity["byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, Integral)
            or int(byte_count) < 0
        ):
            raise ValueError(f"{name} file byte count must be non-negative")
        files[raw_path] = {
            "sha256": digest(raw_identity["sha256"], f"{name} file SHA"),
            "byte_count": int(byte_count),
        }
    supplied_bundle = digest(value["bundle_sha256"], f"{name} bundle SHA")
    reproduced_bundle = sha256(canonical_json(files).encode("utf-8")).hexdigest()
    if reproduced_bundle != supplied_bundle:
        raise ValueError(f"{name} bundle SHA-256 does not reproduce")
    return {
        "execution_source_root": execution_root,
        "files": files,
        "bundle_sha256": supplied_bundle,
    }


def _validated_source_snapshot_identity(value: object) -> dict[str, str]:
    expected_fields = {
        "source_archive_sha256",
        "source_manifest_sha256",
        "source_tree_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("replayed source snapshot identity is incomplete or unsupported")
    return {
        name: digest(value[name], f"replayed {name}") for name in sorted(expected_fields)
    }


def _validated_relative_source_inventory(
    value: object, *, name: str
) -> dict[str, str]:
    """Return one exact, portable, canonical relative-POSIX source inventory."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} source inventory is missing")
    result: dict[str, str] = {}
    for raw_path, source_sha in value.items():
        if not isinstance(raw_path, str):
            raise ValueError(f"{name} source inventory paths must be exact strings")
        relative = PurePosixPath(raw_path)
        canonical = relative.as_posix()
        if (
            not raw_path
            or raw_path == "."
            or "\\" in raw_path
            or relative.is_absolute()
            or ".." in relative.parts
            or canonical != raw_path
        ):
            raise ValueError(
                f"{name} source inventory path must be canonical relative POSIX: "
                f"{raw_path!r}"
            )
        if canonical in result:
            raise ValueError(f"{name} source inventory contains a duplicate path")
        result[canonical] = digest(source_sha, f"{name} source SHA")
    return result


def _validated_dataset_content_identity(
    value: object, *, name: str
) -> dict[str, object]:
    expected_fields = {
        "path",
        "dataset_id",
        "dataset_schema",
        "dataset_version",
        "dataset_manifest_sha256",
        "dataset_file_sha256",
        "dataset_file_byte_count",
        "dataset_source_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    execution_path = value["path"]
    if not isinstance(execution_path, str) or not Path(execution_path).is_absolute():
        raise ValueError(f"{name} execution dataset path must be absolute")
    byte_count = value["dataset_file_byte_count"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, Integral)
        or int(byte_count) < 1
    ):
        raise ValueError(f"{name} byte count must be positive")
    source_identity = _validated_relative_source_inventory(
        value["dataset_source_sha256"], name=name
    )
    portable = {
        "dataset_id": value["dataset_id"],
        "dataset_schema": value["dataset_schema"],
        "dataset_version": value["dataset_version"],
        "dataset_manifest_sha256": digest(
            value["dataset_manifest_sha256"], f"{name} manifest SHA"
        ),
        "dataset_file_sha256": digest(
            value["dataset_file_sha256"], f"{name} file SHA"
        ),
        "dataset_file_byte_count": int(byte_count),
        "dataset_source_sha256": source_identity,
    }
    if any(
        not isinstance(portable[field], str) or not portable[field]
        for field in ("dataset_id", "dataset_schema", "dataset_version")
    ):
        raise ValueError(f"{name} dataset identity strings must be non-empty")
    return {"execution_dataset_path": execution_path, "portable": portable}


def verify_recorded_dataset_sources(
    dataset: V5GroupedDataset, source_root: Path
) -> dict[str, object]:
    recorded = dataset.manifest.get("source_sha256")
    validated = _validated_relative_source_inventory(
        recorded, name="checked dataset"
    )
    root = source_root.expanduser().resolve()
    replay = {}
    for name, expected in validated.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"dataset source path is missing or unsafe: {name}")
        observed = file_identity(target)["sha256"]
        if observed != expected:
            raise ValueError(f"checked dataset source drift detected: {name}")
        replay[name] = observed
    return {
        "recorded_file_count": len(replay),
        "recorded_source_sha256": dict(sorted(replay.items())),
        "all_recorded_sources_reverified": True,
    }


def dataset_identity(
    path: Path,
    dataset: V5GroupedDataset,
    receipt: V5ArtifactReceipt,
    *,
    authoritative_path: Path | None = None,
) -> dict[str, object]:
    return {
        "path": str(path if authoritative_path is None else authoritative_path),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_schema": dataset.manifest["dataset_schema"],
        "dataset_version": dataset.manifest["dataset_version"],
        "dataset_manifest_sha256": dataset.manifest["manifest_sha256"],
        "dataset_file_sha256": receipt.artifact_sha256,
        "dataset_file_byte_count": receipt.byte_count,
        "dataset_source_sha256": dict(dataset.manifest["source_sha256"]),
    }


def live_gate_contract() -> dict[str, object]:
    protocol = validate_protocol(protocol_payload())
    stage = protocol["stages"]["k1_memorization"]
    gates = stage.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(EXPECTED_GATE_KEYS):
        raise ValueError("K1 protocol gates are missing, extended, or unsupported")
    frozen = {}
    for name in EXPECTED_GATE_KEYS:
        value = gates[name]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"K1 gate {name} must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"K1 gate {name} must be finite")
        frozen[name] = numeric
    recipes = positive_integer(
        stage.get("independent_clean_recipes"), "K1 protocol recipe count"
    )
    views = positive_integer(
        stage.get("training_augmentation_views_per_recipe"),
        "K1 protocol training view count",
    )
    if stage.get("purpose") != "single_branch_capacity_and_objective_diagnostic":
        raise ValueError("K1 protocol stage purpose changed incompatibly")
    if stage.get("topology_schedule") != "single_branch_sphere_pattern0":
        raise ValueError("K1 Phase-B requires the frozen sphere/pattern-0 stage scope")
    return {
        "schema_version": protocol["schema_version"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "json_pointer": "/stages/k1_memorization/gates",
        "recipes": recipes,
        "training_augmentation_views": views,
        "topology_schedule": stage["topology_schedule"],
        "authoritative_forward_version": protocol["authoritative_forward_contract"][
            "forward_version"
        ],
        "gates": frozen,
    }


def model_weights_sha256(model: object) -> str:
    weights = getattr(model, "weights", None)
    if weights is None:
        raise TypeError("loaded model does not expose weights")
    result = sha256()
    for value in weights:
        array = np.ascontiguousarray(value.numpy())
        weight_name = getattr(value, "path", value.name)
        result.update(str(weight_name).encode("utf-8"))
        result.update(b"\0")
        result.update(array.dtype.str.encode("ascii"))
        result.update(canonical_json(list(array.shape)).encode("ascii"))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def validate_v5_k1_phase_a_bindings(
    payload: Mapping[str, object],
    *,
    dataset_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    model_provenance_payload: Mapping[str, object],
    model_provenance_identity: Mapping[str, object],
    phase_a_source_identity: Mapping[str, object],
    source_snapshot_identity: Mapping[str, object],
    model_weights_sha256_value: str,
    live_gate_contract_value: Mapping[str, object],
) -> dict[str, object]:
    """Revalidate the complete Phase-A receipt against live immutable inputs."""

    if (
        not isinstance(payload, Mapping)
        or set(payload) != V5_K1_DATASET_GATE_PUBLISHED_RESULT_FIELDS
    ):
        raise ValueError("Phase-A result fields are incomplete, extended, or unsupported")
    value = dict(payload)
    result_digest = digest(value["result_payload_sha256"], "Phase-A result payload SHA")
    core = dict(value)
    core.pop("result_payload_sha256")
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != result_digest:
        raise ValueError("Phase-A result payload SHA-256 does not reproduce")
    if (
        value["schema_version"],
        value["version"],
        value["scientific_role"],
    ) != (V5_K1_DATASET_GATE_SCHEMA, V5_K1_DATASET_GATE_VERSION, V5_K1_DATASET_GATE_ROLE):
        raise ValueError("Phase-A result schema/version/role is incompatible")
    required_flags = {
        "model_acceptance_evidence": False,
        "writes_performed": True,
        "gate_executed": True,
        "stage_a_pass_enforced": True,
        "stage_a_configured_wiring_gate_passed": True,
        "stage_a_passed": True,
        "single_branch_phase_b_gate_passed": False,
        "complete_k1_proposal_exact_gate_passed": False,
    }
    if any(value.get(name) is not expected for name, expected in required_flags.items()):
        raise ValueError("Phase-A result has incomplete, smoke, or non-passing state")
    if value["status"] != "stage_a_passed":
        raise ValueError("Phase-A result is not a full passing run")
    requested = value.get("requested_config")
    resolved_model = value.get("resolved_model_config")
    if (
        not isinstance(requested, Mapping)
        or requested.get("smoke") is not False
        or not isinstance(resolved_model, Mapping)
        or resolved_model.get("mixture_components") != 1
    ):
        raise ValueError("Phase-A must be a non-smoke one-component MDN run")
    expected_model = {
        "schema_version": MODEL_V5_SCHEMA,
        "model_version": MODEL_V5_VERSION,
        "model_name": MODEL_V5_NAME,
    }
    if value["live_model_identity"] != expected_model:
        raise ValueError("Phase-A live model identity drift detected")
    if value["model_contract"] != model_v5_contract_payload():
        raise ValueError("Phase-A model contract drift detected")

    raw_launch_binding = value.get("launch_binding")
    if not isinstance(raw_launch_binding, Mapping):
        raise ValueError("Phase-A result has no launch binding")
    launch_binding = validate_launch_binding_payload(
        raw_launch_binding, expected_stage="full_gate"
    )
    job_local_capability = _validate_phase_a_job_local_capability(
        value.get("job_local_capability"), launch_binding=launch_binding
    )

    training_evidence = _validate_phase_a_training_evidence(
        value.get("training_evidence")
    )
    replayed_snapshot = _validated_source_snapshot_identity(source_snapshot_identity)
    if training_evidence["source_snapshot"] != replayed_snapshot:
        raise ValueError(
            "Phase-A training evidence does not bind the replayed source archive, "
            "manifest, and tree"
        )

    if value.get("full_k1_all_legal_branches_gate_status") != (
        "pending_fail_closed_requires_balanced_12_branch_cohort"
    ):
        raise ValueError("Phase-A result improperly claims full K1 completion")
    contract = value.get("single_branch_k1_gate_contract")
    if (
        not isinstance(contract, Mapping)
        or set(contract) != PHASE_A_SINGLE_BRANCH_CONTRACT_FIELDS
    ):
        raise ValueError("Phase-A single-branch gate contract is incomplete or unsupported")
    expected_criteria = {
        name: live_gate_contract_value[name]
        for name in ("schema_version", "protocol_version", "protocol_sha256", "json_pointer")
    }
    if contract.get("criteria_source") != expected_criteria:
        raise ValueError("Phase-A study-protocol identity drift detected")
    if contract.get("phase_b_frozen_criteria") != live_gate_contract_value["gates"]:
        raise ValueError("Phase-A K1 gate values drift detected")
    if contract.get("authoritative_forward_version") != live_gate_contract_value[
        "authoritative_forward_version"
    ]:
        raise ValueError("Phase-A authoritative forward identity drift detected")
    if contract.get("stage_a_evaluated_metrics") != [
        "configured_training_objective_reduction",
        "deterministic_mixture_median_target_local_rms_median",
    ] or contract.get("stage_a_metric_semantics") != (
        "sigmoid_mixture_loc_not_a_stochastic_single_draw_and_not_a_phase_b_metric"
    ):
        raise ValueError("Phase-A metric semantics are incomplete or incompatible")
    if (
        contract.get("stage_b_status") != "pending_not_executed_fail_closed"
        or contract.get("single_branch_phase_b_gate_passed") is not False
        or contract.get("complete_k1_proposal_exact_gate_passed") is not False
        or contract.get("topology_schedule")
        != live_gate_contract_value["topology_schedule"]
        or contract.get("full_k1_all_legal_branches_gate_status")
        != "pending_fail_closed_requires_balanced_12_branch_cohort"
    ):
        raise ValueError("Phase-A result improperly claims Phase-B completion")
    expected_pending = {
        name.removesuffix("_lt").removesuffix("_gte") for name in EXPECTED_GATE_KEYS
    }
    if set(contract.get("stage_b_pending_metrics", ())) != expected_pending:
        raise ValueError("Phase-A result did not leave every protocol metric to Phase-B")
    recorded_dataset = _validated_dataset_content_identity(
        value["dataset"], name="Phase-A recorded dataset"
    )
    replayed_dataset = _validated_dataset_content_identity(
        dataset_identity, name="Phase-A replayed dataset"
    )
    if recorded_dataset["portable"] != replayed_dataset["portable"]:
        raise ValueError("Phase-A portable dataset content identity drift detected")
    bound_dataset = training_evidence["dataset_completion_binding"]["dataset"]
    if bound_dataset["original_path"] != replayed_dataset["execution_dataset_path"]:
        raise ValueError(
            "Phase-A dataset completion binding original path does not match the "
            "replayed immutable dataset"
        )
    expected_bound_dataset = {
        "original_path": bound_dataset["original_path"],
        "artifact_sha256": dataset_identity["dataset_file_sha256"],
        "byte_count": dataset_identity["dataset_file_byte_count"],
        "dataset_id": dataset_identity["dataset_id"],
        "dataset_schema": dataset_identity["dataset_schema"],
        "dataset_version": dataset_identity["dataset_version"],
        "manifest_sha256": dataset_identity["dataset_manifest_sha256"],
    }
    if bound_dataset != expected_bound_dataset:
        raise ValueError("Phase-A training evidence does not bind the consumed dataset")
    validation = value.get("dataset_validation")
    recipes = live_gate_contract_value["recipes"]
    views = live_gate_contract_value["training_augmentation_views"]
    if (
        not isinstance(validation, Mapping)
        or validation.get("clean_parent_count") != recipes
        or validation.get("observation_view_count") != recipes * views
        or validation.get("known_truth_target_count") != recipes
        or not isinstance(validation.get("learnable_target_count"), Integral)
        or not isinstance(validation.get("fully_fixed_target_count"), Integral)
        or validation.get("learnable_target_count") < 1
        or validation.get("fully_fixed_target_count") < 0
        or validation.get("learnable_target_count")
        + validation.get("fully_fixed_target_count")
        != recipes
    ):
        raise ValueError(
            "Phase-A dataset cardinality is not the frozen single-branch cohort"
        )

    recorded_source = _validated_source_content_identity(
        value.get("source"), name="Phase-A recorded source"
    )
    replayed_source = _validated_source_content_identity(
        phase_a_source_identity, name="Phase-A replayed source"
    )
    if set(recorded_source["files"]) != PHASE_A_SOURCE_PATHS:
        raise ValueError("Phase-A source inventory is incomplete or unsupported")
    if set(replayed_source["files"]) != PHASE_A_SOURCE_PATHS:
        raise ValueError("replayed Phase-A source inventory is incomplete or unsupported")
    if (
        recorded_source["files"] != replayed_source["files"]
        or recorded_source["bundle_sha256"] != replayed_source["bundle_sha256"]
    ):
        raise ValueError("Phase-A portable source content identity drift detected")
    model_provenance = _validate_phase_a_model_provenance(
        model_provenance_payload,
        model_identity=model_identity,
        training_evidence=training_evidence,
        launch_binding=launch_binding,
        job_local_capability=job_local_capability,
    )
    expected_artifact = {
        "filename": PHASE_A_MODEL_FILENAME,
        **dict(model_identity),
        "reload_graph_contract_passed": True,
        "provenance": {
            "filename": PHASE_A_MODEL_PROVENANCE_FILENAME,
            **dict(model_provenance_identity),
            "binding_sha256": model_provenance["binding_sha256"],
        },
    }
    if value.get("model_artifact") != expected_artifact:
        raise ValueError("Phase-A model artifact drift detected")

    nested = value.get("stage_a_memorization_result")
    if not isinstance(nested, Mapping) or set(nested) != PHASE_A_MEMORIZATION_RESULT_FIELDS:
        raise ValueError("Phase-A memorization result fields are incomplete or unsupported")
    if (
        nested.get("schema_version") != V5_MEMORIZATION_GATE_SCHEMA
        or nested.get("version") != V5_MEMORIZATION_GATE_VERSION
        or nested.get("scientific_role")
        != "wiring_and_memorization_diagnostic_not_model_acceptance"
    ):
        raise ValueError("Phase-A memorization result identity is incompatible")
    nested_core = dict(nested)
    nested_digest = digest(nested_core.pop("result_sha256", None), "Phase-A nested result SHA")
    if sha256(canonical_json(nested_core).encode("utf-8")).hexdigest() != nested_digest:
        raise ValueError("Phase-A nested memorization SHA-256 does not reproduce")
    metric_names = (
        "initial_loss",
        "final_loss",
        "initial_target_median_rms",
        "final_target_median_rms",
    )
    if any(not np.isfinite(float(nested.get(name, np.nan))) for name in metric_names):
        raise ValueError("Phase-A memorization metrics contain NaN or infinity")
    if (
        nested.get("passed") is not True
        or nested.get("target_count") != recipes
        or nested.get("learnable_target_count")
        != validation.get("learnable_target_count")
        or nested.get("fully_fixed_target_count")
        != validation.get("fully_fixed_target_count")
    ):
        raise ValueError("Phase-A configured memorization diagnostic did not fully pass")
    if nested.get("final_weights_sha256") != digest(
        model_weights_sha256_value, "loaded model weights SHA"
    ):
        raise ValueError("loaded model weights disagree with the Phase-A training result")
    nested_config = nested.get("config")
    if not isinstance(nested_config, Mapping):
        raise ValueError("Phase-A memorization configuration is missing")
    final_rms_limit = float(nested_config.get("max_final_target_median_rms", np.nan))
    minimum_reduction = float(nested_config.get("minimum_loss_reduction", np.nan))
    if (
        not np.all(np.isfinite((final_rms_limit, minimum_reduction)))
        or float(nested["final_target_median_rms"]) > final_rms_limit
        or float(nested["initial_loss"]) - float(nested["final_loss"])
        < minimum_reduction
    ):
        raise ValueError("Phase-A configured mixture-median wiring criteria do not reproduce")
    return {
        "phase_a_result_payload_sha256": result_digest,
        "phase_a_recorded_execution_dataset_path": recorded_dataset[
            "execution_dataset_path"
        ],
        "reverified_immutable_dataset_path": replayed_dataset[
            "execution_dataset_path"
        ],
        "dataset_location_equality_required": False,
        "portable_dataset_identity": replayed_dataset["portable"],
        "source_snapshot_identity": replayed_snapshot,
        "phase_a_recorded_execution_source_root": recorded_source[
            "execution_source_root"
        ],
        "reverified_immutable_source_root": replayed_source[
            "execution_source_root"
        ],
        "source_location_equality_required": False,
        "portable_source_bundle_sha256": replayed_source["bundle_sha256"],
        "dataset_identity_reverified": True,
        "source_identity_reverified": True,
        "model_file_identity_reverified": True,
        "model_provenance_identity_reverified": True,
        "training_evidence_reverified": True,
        "model_graph_identity_reverified": True,
        "model_weights_identity_reverified": True,
        "study_protocol_identity_reverified": True,
        "phase_a_launch_binding_reverified": True,
        "phase_a_job_local_capability_reverified": True,
        "phase_a_full_pass_reverified": True,
    }


def execution_guard(hostname: str, environment: Mapping[str, str]) -> str:
    if hostname.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 Phase-B must run on a Slurm worker, not max-wgs")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or int(job_id) < 1:
        raise RuntimeError("K1 Phase-B execution requires a valid SLURM_JOB_ID")
    return job_id


__all__ = [
    "CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS",
    "EXPECTED_GATE_KEYS",
    "FULL_FILE_IDENTITY_FIELDS",
    "MAXWELL_DUST_ROOT",
    "PHASE_A_SOURCE_PATHS",
    "PHASE_B_SOURCE_PATHS",
    "PHASE_B_WORKER_RELATIVE_PATH",
    "PROPOSAL_COUNT",
    "SINGLE_DRAW_INDEX",
    "V5_EXACT_FORWARD_BUDGET_UNIT",
    "V5_K1_PHASE_B_RESULT_FILENAME",
    "V5_K1_PHASE_B_ROLE",
    "V5_K1_PHASE_B_SCHEMA",
    "V5_K1_PHASE_B_VERSION",
    "V5K1PhaseBGateConfig",
    "V5K1PhaseBParentRecord",
    "cross_node_stable_file_identity",
    "dataset_identity",
    "digest",
    "execution_guard",
    "file_identity",
    "live_gate_contract",
    "model_weights_sha256",
    "source_identity",
    "strict_json_object",
    "under_root",
    "validate_cross_node_stable_file_identity",
    "validate_v5_k1_phase_a_bindings",
    "verify_recorded_dataset_sources",
]
