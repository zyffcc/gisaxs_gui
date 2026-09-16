"""Run the checked K=1 V5.2 memorization/wiring gate on a Slurm worker.

This command is deliberately narrower than model training or acceptance.  It
only proves that a live one-component V5.2 graph can consume a checked K=1
known-truth grouped dataset and optimize the existing memorization objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import platform
import socket
import tempfile
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf

from .candidate_supervision_v5 import (
    KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
    KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
    SEARCH_OUTCOME_CODE,
)
from .contract import topology_from_id
from .grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    candidate_input,
    candidate_label,
    clean_array,
    read_v5_grouped_dataset,
)
from .grouped_known_truth_oracle_v5 import V5_ORACLE_PROTOCOL_SHA256
from .memorization_gate_v5 import V5MemorizationGateConfig, run_v5_memorization_gate
from .k1_phase_a_cross_platform_v5 import (
    file_identity as strict_file_identity,
    validate_v5_k1_cross_platform_marker_file,
)
from .k1_phase_a_dataset_binding_v5 import (
    validate_v5_k1_phase_a_dataset_binding_file,
)
from .k1_phase_a_capability_v7 import (
    V7PhaseAInputCapability,
    _consumed_capability_payload,
    _validate_phase_a_capability,
)
from .k1_phase_a_contract_v7 import (
    PHASE_A_LAUNCH_BINDING_SCHEMA,
    validate_launch_binding_payload,
)
from .model_v5 import (
    build_branch_conditioned_proposal_model,
    validate_model_v5_graph_contract,
)
from .model_v5_contract import (
    MODEL_V5_NAME,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
    model_v5_contract_payload,
)
from .study_protocol import protocol_payload
from .training_objective_v5 import DEFAULT_LOCAL_COVERAGE_WEIGHT


V5_K1_DATASET_GATE_SCHEMA = (
    "gisaxs.posterior_v8.k1_single_branch_dataset_memorization_gate/v8"
)
V5_K1_DATASET_GATE_VERSION = (
    "posterior_v8_v5_2_cosine_center_aligned_bounded_memory_fixed_query_aware_wiring_gate_v8"
)
V5_K1_DATASET_GATE_ROLE = (
    "single_branch_sphere_pattern0_memorization_wiring_not_model_acceptance"
)
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
RESULT_FILENAME = "result.json"
MODEL_FILENAME = "model.keras"
MODEL_PROVENANCE_FILENAME = "model.provenance.json"
V5_K1_MODEL_PROVENANCE_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_model_provenance/v2"
V5_K1_MODEL_PROVENANCE_VERSION = (
    "posterior_v8_model_bytes_dataset_source_gate_and_launch_binding_v2"
)
V5_K1_TRAINING_EVIDENCE_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_training_evidence/v1"
V5_K1_TRAINING_EVIDENCE_VERSION = (
    "posterior_v8_worker_revalidated_dataset_source_and_cross_platform_gate_v1"
)
_SMOKE_STEPS = 2
_SMOKE_WIDTH = 32
_SMOKE_ENCODER_BLOCKS = 1
V5_K1_DATASET_GATE_PUBLISHED_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "version",
        "scientific_role",
        "model_acceptance_evidence",
        "requested_config",
        "resolved_gate_config",
        "resolved_model_config",
        "live_model_identity",
        "model_contract",
        "single_branch_k1_gate_contract",
        "dataset",
        "dataset_validation",
        "source",
        "training_evidence",
        "launch_binding",
        "job_local_capability",
        "output_dir",
        "created_at_utc",
        "status",
        "writes_performed",
        "gate_executed",
        "stage_a_pass_enforced",
        "stage_a_configured_wiring_gate_passed",
        "stage_a_passed",
        "stage_a_memorization_result",
        "single_branch_phase_b_gate_passed",
        "complete_k1_proposal_exact_gate_passed",
        "full_k1_all_legal_branches_gate_status",
        "model_artifact",
        "execution",
        "publication",
        "result_payload_sha256",
    }
)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseATrainingEvidence:
    """Every external identity the real Phase-A worker must revalidate itself."""

    dataset_binding_path: Path
    cross_platform_pass_marker_path: Path
    original_dataset_path: str
    source_archive_sha256: str
    source_manifest_sha256: str
    source_tree_sha256: str
    reference_file_sha256: str
    reference_file_byte_count: int
    reference_manifest_sha256: str
    scientific_content_sha256: str
    comparison_result_sha256: str
    gate_claim_sha256: str

    @property
    def source(self) -> dict[str, str]:
        return {
            "source_archive_sha256": self.source_archive_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_tree_sha256": self.source_tree_sha256,
        }

    @property
    def marker_expected(self) -> dict[str, object]:
        return {
            "expected_source": self.source,
            "reference_file_sha256": self.reference_file_sha256,
            "reference_file_byte_count": self.reference_file_byte_count,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "scientific_content_sha256": self.scientific_content_sha256,
            "comparison_result_sha256": self.comparison_result_sha256,
            "expected_gate_claim_sha256": self.gate_claim_sha256,
        }


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, kw_only=True)
class V5K1DatasetGateConfig:
    steps: int = 18000
    batch_size: int = 32
    learning_rate: float = 3.0e-3
    final_learning_rate: float = 3.0e-5
    learning_rate_schedule: str = "cosine_decay"
    seed: int = 20260903
    local_mdn_weight: float = 1.0
    local_coverage_weight: float = DEFAULT_LOCAL_COVERAGE_WEIGHT
    operational_top_l_alignment_weight: float = 1.0
    max_final_target_median_rms: float = 0.01
    minimum_loss_reduction: float = 0.5
    width: int = 128
    encoder_blocks: int = 6
    smoke: bool = False

    def __post_init__(self) -> None:
        gate = V5MemorizationGateConfig(
            steps=self.steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            final_learning_rate=self.final_learning_rate,
            learning_rate_schedule=self.learning_rate_schedule,
            seed=self.seed,
            local_mdn_weight=self.local_mdn_weight,
            local_coverage_weight=self.local_coverage_weight,
            operational_top_l_alignment_weight=(
                self.operational_top_l_alignment_weight
            ),
            max_final_target_median_rms=self.max_final_target_median_rms,
            minimum_loss_reduction=self.minimum_loss_reduction,
        )
        object.__setattr__(self, "steps", gate.steps)
        object.__setattr__(self, "batch_size", gate.batch_size)
        object.__setattr__(self, "learning_rate", gate.learning_rate)
        object.__setattr__(
            self, "final_learning_rate", gate.final_learning_rate
        )
        object.__setattr__(
            self, "learning_rate_schedule", gate.learning_rate_schedule
        )
        object.__setattr__(self, "seed", gate.seed)
        object.__setattr__(self, "local_mdn_weight", gate.local_mdn_weight)
        object.__setattr__(
            self, "local_coverage_weight", gate.local_coverage_weight
        )
        object.__setattr__(
            self,
            "operational_top_l_alignment_weight",
            gate.operational_top_l_alignment_weight,
        )
        object.__setattr__(self, "max_final_target_median_rms", gate.max_final_target_median_rms)
        object.__setattr__(self, "minimum_loss_reduction", gate.minimum_loss_reduction)
        object.__setattr__(self, "width", _positive_integer(self.width, "width"))
        object.__setattr__(
            self,
            "encoder_blocks",
            _positive_integer(self.encoder_blocks, "encoder_blocks"),
        )
        if type(self.smoke) is not bool:
            raise TypeError("smoke must be a bool")

    def resolved(self) -> tuple[V5MemorizationGateConfig, dict[str, int]]:
        steps = min(self.steps, _SMOKE_STEPS) if self.smoke else self.steps
        width = min(self.width, _SMOKE_WIDTH) if self.smoke else self.width
        encoder_blocks = (
            min(self.encoder_blocks, _SMOKE_ENCODER_BLOCKS) if self.smoke else self.encoder_blocks
        )
        return (
            V5MemorizationGateConfig(
                steps=steps,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                final_learning_rate=self.final_learning_rate,
                learning_rate_schedule=self.learning_rate_schedule,
                seed=self.seed,
                local_mdn_weight=self.local_mdn_weight,
                local_coverage_weight=self.local_coverage_weight,
                operational_top_l_alignment_weight=(
                    self.operational_top_l_alignment_weight
                ),
                max_final_target_median_rms=self.max_final_target_median_rms,
                minimum_loss_reduction=self.minimum_loss_reduction,
            ),
            {
                "width": width,
                "encoder_blocks": encoder_blocks,
                "mixture_components": 1,
            },
        )


def _strict_object(encoded: object, name: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {name} field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(str(encoded), object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _one_item(value: object, name: str) -> object:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"K=1 gate requires exactly one {name}")
    return value[0]


def validate_v5_k1_memorization_dataset(dataset: V5GroupedDataset) -> dict[str, object]:
    """Fail closed unless every row is a valid K=1 one-call known-truth target."""

    if not isinstance(dataset, V5GroupedDataset):
        raise TypeError("dataset must be a checked V5GroupedDataset")
    if dataset.manifest["build_policy"]["generating_candidate_only"] is not True:
        raise ValueError("K=1 gate requires a generating-candidate-only warmup dataset")
    if dataset.candidate_count != dataset.recipe_count:
        raise ValueError("K=1 gate requires exactly one known-truth candidate per clean parent")

    arrays = dataset.arrays
    recipe_payloads = tuple(
        _strict_object(value, "clean recipe")
        for value in arrays[clean_array("recipe_canonical_json")]
    )
    canonical_topologies: list[tuple[str, ...]] = []
    for payload in recipe_payloads:
        truth = _one_item(payload.get("truth_components"), "truth component")
        query = payload.get("query")
        amplitude = payload.get("amplitude_composition")
        amplitude_query = payload.get("amplitude_query")
        if not all(
            isinstance(value, Mapping) for value in (truth, query, amplitude, amplitude_query)
        ):
            raise ValueError("K=1 clean recipe has incomplete query or amplitude provenance")
        topology = tuple(str(value) for value in query.get("topology", ()))
        _one_item(list(topology), "query topology component")
        _one_item(query.get("component_bounds"), "query component bound")
        _one_item(amplitude_query.get("component_intensities"), "amplitude Int range")
        if amplitude.get("component_count") != 1:
            raise ValueError("K=1 gate requires amplitude component_count=1")
        if str(truth.get("shape")) != topology[0]:
            raise ValueError("K=1 truth shape disagrees with its clean query topology")
        canonical_topologies.append(topology)
    if any(value != ("sphere",) for value in canonical_topologies):
        raise ValueError("frozen K1 Phase-A scope requires only the sphere topology")
    if not np.all(arrays[clean_array("target_pattern_id")] == 0):
        raise ValueError("frozen K1 Phase-A scope requires only pattern_id=0")

    topology_ids = arrays[candidate_input("branch_topology_id")].reshape(-1)
    candidate_recipe = arrays["candidate_context__recipe_index"]
    for topology_id, recipe_index in zip(topology_ids, candidate_recipe):
        topology = topology_from_id(int(topology_id))
        if len(topology) != 1 or topology != canonical_topologies[int(recipe_index)]:
            raise ValueError("candidate branch is not the K=1 clean-parent topology")

    required_values = {
        "search_outcome_code": SEARCH_OUTCOME_CODE["compatible_found"],
        "search_protocol_id": KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
        "search_protocol_sha256": V5_ORACLE_PROTOCOL_SHA256,
        "search_exact_forward_call_budget": 1,
        "search_exact_forward_calls_used": 1,
        "search_termination_reason": KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
        "search_compatible_representative_count": 1,
        "search_completed": True,
        "has_local_target": True,
        "exact_bounds_passed": True,
        "exact_physics_passed": True,
        "generating_candidate_match": True,
    }
    for name, expected in required_values.items():
        if not np.all(arrays[candidate_label(name)] == expected):
            raise ValueError(f"K=1 gate requires known-truth warmup field {name}={expected!r}")
    if not np.all(arrays[candidate_label("exact_metric_value")] == 0.0):
        raise ValueError("known-truth warmup exact metric must be zero")
    varying = arrays[candidate_label("varying_dimension_mask")].astype(np.bool_, copy=False)
    target = arrays[candidate_label("target_local")]
    varying_per_target = np.sum(varying, axis=-1)
    learnable_target_count = int(np.count_nonzero(varying_per_target > 0))
    fully_fixed_target_count = int(dataset.candidate_count - learnable_target_count)
    if learnable_target_count < 1:
        raise ValueError("K=1 warmup dataset needs at least one learnable target")
    if not np.all(target[~varying] == np.float32(0.5)):
        raise ValueError("fixed/inactive K=1 target coordinates must equal canonical 0.5")

    x = arrays["observation__input__x"]
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("checked observations do not define a padded point dimension")
    return {
        "clean_parent_count": dataset.recipe_count,
        "observation_view_count": dataset.observation_count,
        "known_truth_target_count": dataset.candidate_count,
        "learnable_target_count": learnable_target_count,
        "fully_fixed_target_count": fully_fixed_target_count,
        "varying_coordinate_count": int(np.count_nonzero(varying)),
        "joined_positive_example_count": dataset.joined_count,
        "max_points": int(x.shape[1]),
        "topology_ids": sorted({int(value) for value in topology_ids}),
        "pattern_ids": [0],
        "topology_schedule": "single_branch_sphere_pattern0",
        "split_ids": sorted({str(value) for value in arrays[clean_array("split_id")]}),
        "oracle_protocol_id": KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
        "oracle_protocol_sha256": V5_ORACLE_PROTOCOL_SHA256,
        "scientific_role": V5_K1_DATASET_GATE_ROLE,
    }


def _file_identity(path: Path) -> dict[str, object]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {"sha256": digest.hexdigest(), "byte_count": byte_count}


def _source_identity(source_root: Path) -> dict[str, object]:
    root = source_root.expanduser().resolve()
    posterior_root = root / "utils/ML_Fitting_1D_GISAXS/PosteriorV8"
    expected = posterior_root / "run_k1_memorization_gate_v5.py"
    if expected.resolve() != Path(__file__).resolve():
        raise ValueError("source_root does not own the imported K1 gate worker")
    sources = sorted(posterior_root.glob("*.py"))
    sources.extend(
        (
            root / "src/gimap/features/fitting/domain/scattering_model.py",
            root / "src/gimap/features/fitting/domain/physical_constraints.py",
        )
    )
    files = {
        path.relative_to(root).as_posix(): _file_identity(path) for path in sources
    }
    return {
        "source_root": str(root),
        "files": files,
        "bundle_sha256": sha256(canonical_json(files).encode()).hexdigest(),
    }


def _single_branch_k1_gate_contract() -> dict[str, object]:
    protocol = protocol_payload()
    stage = protocol["stages"]["k1_memorization"]
    if stage.get("topology_schedule") != "single_branch_sphere_pattern0":
        raise ValueError("live K1 Phase-A protocol is not the frozen single branch")
    if stage.get("purpose") != "single_branch_capacity_and_objective_diagnostic":
        raise ValueError("live K1 Phase-A protocol purpose drift detected")
    return {
        "criteria_source": {
            "schema_version": protocol["schema_version"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol["protocol_sha256"],
            "json_pointer": "/stages/k1_memorization/gates",
        },
        "phase_b_frozen_criteria": dict(stage["gates"]),
        "topology_schedule": stage["topology_schedule"],
        "authoritative_forward_version": protocol["authoritative_forward_contract"][
            "forward_version"
        ],
        "stage_a_evaluated_metrics": [
            "configured_training_objective_reduction",
            "deterministic_mixture_median_target_local_rms_median",
        ],
        "stage_a_metric_semantics": (
            "sigmoid_mixture_loc_not_a_stochastic_single_draw_and_not_a_phase_b_metric"
        ),
        "stage_b_pending_metrics": [
            "branch_conditioned_local_mdn_single_draw_local_rms_median",
            "branch_conditioned_local_mdn_best_of_32_local_rms_median",
            "branch_conditioned_local_mdn_best_of_32_local_rms_p90",
            "exact_post_refine_raw_log_rmse_p90",
            "exact_post_refine_compatible_rate",
        ],
        "stage_b_status": "pending_not_executed_fail_closed",
        "single_branch_phase_b_gate_passed": False,
        "complete_k1_proposal_exact_gate_passed": False,
        "full_k1_all_legal_branches_gate_status": (
            "pending_fail_closed_requires_balanced_12_branch_cohort"
        ),
    }


def _under_root(path: Path, root: Path, name: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{name} must be below {resolved_root}")
    return resolved


def _dataset_identity(
    dataset_path: Path,
    dataset: V5GroupedDataset,
    receipt: V5ArtifactReceipt,
) -> dict[str, object]:
    return {
        "path": str(dataset_path),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_schema": dataset.manifest["dataset_schema"],
        "dataset_version": dataset.manifest["dataset_version"],
        "dataset_manifest_sha256": dataset.manifest["manifest_sha256"],
        "dataset_file_sha256": receipt.artifact_sha256,
        "dataset_file_byte_count": receipt.byte_count,
        "dataset_source_sha256": dict(dataset.manifest["source_sha256"]),
    }


def _portable_file_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "sha256": value["sha256"],
        "byte_count": value["byte_count"],
        "mode": value["mode"],
    }


def _validated_training_evidence(
    evidence: V5K1PhaseATrainingEvidence,
    *,
    dataset_file: Path,
    dataset_root: Path,
) -> dict[str, object]:
    if not isinstance(evidence, V5K1PhaseATrainingEvidence):
        raise TypeError("evidence must be V5K1PhaseATrainingEvidence")
    marker_path = _under_root(
        Path(evidence.cross_platform_pass_marker_path),
        dataset_root,
        "cross_platform_pass_marker_path",
    )
    binding_path = _under_root(
        Path(evidence.dataset_binding_path), dataset_root, "dataset_binding_path"
    )
    marker_expected = evidence.marker_expected
    marker_result = validate_v5_k1_cross_platform_marker_file(
        marker_path, **marker_expected
    )
    binding = validate_v5_k1_phase_a_dataset_binding_file(
        binding_path,
        dataset_path=dataset_file,
        marker_path=marker_path,
        expected_original_dataset_path=evidence.original_dataset_path,
        marker_expected=marker_expected,
    )
    binding_file = strict_file_identity(
        binding_path,
        name="K1 dataset completion binding",
        require_read_only=True,
    )
    marker_file = marker_result["file"]
    core = {
        "schema": V5_K1_TRAINING_EVIDENCE_SCHEMA,
        "version": V5_K1_TRAINING_EVIDENCE_VERSION,
        "status": "VALIDATED",
        "source_snapshot": evidence.source,
        "dataset_completion_binding": {
            "binding_sha256": binding["binding_sha256"],
            "file": _portable_file_identity(binding_file),
            "dataset": dict(binding["dataset"]),
        },
        "cross_platform_gate": {
            "gate_claim_sha256": evidence.gate_claim_sha256,
            "pass_marker_sha256": marker_result["marker"]["marker_sha256"],
            "pass_marker_file": _portable_file_identity(marker_file),
            "reference_file_sha256": evidence.reference_file_sha256,
            "reference_file_byte_count": evidence.reference_file_byte_count,
            "reference_manifest_sha256": evidence.reference_manifest_sha256,
            "scientific_content_sha256": evidence.scientific_content_sha256,
            "comparison_result_sha256": evidence.comparison_result_sha256,
        },
        "worker_validation": (
            "strict_local_dataset_binding_and_pass_marker_replayed_by_training_process"
        ),
    }
    return {
        **core,
        "evidence_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def _execution_guard(hostname: str, environment: Mapping[str, str]) -> str:
    if hostname.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 memorization must run on a Slurm worker, not max-wgs")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or int(job_id) < 1:
        raise RuntimeError("K1 memorization execution requires a valid SLURM_JOB_ID")
    return job_id


def _publish_model(model: tf.keras.Model, target: Path) -> dict[str, object]:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.stem}.", suffix=".keras"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        replay = tf.keras.models.load_model(temporary, compile=False)
        validate_model_v5_graph_contract(replay)
        temporary.chmod(0o400)
        identity = _file_identity(temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite model artifact: {target}") from None
        return {"filename": target.name, **identity, "reload_graph_contract_passed": True}
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(target: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite result artifact: {target}") from None
    finally:
        temporary.unlink(missing_ok=True)


def run_v5_k1_dataset_memorization_gate(
    dataset_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str],
    evidence: V5K1PhaseATrainingEvidence,
    config: V5K1DatasetGateConfig = V5K1DatasetGateConfig(),
    dry_run: bool = False,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    dataset_allowed_root: Path | None = None,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
    launch_binding: Mapping[str, object] | None = None,
    job_local_input_capability: V7PhaseAInputCapability | None = None,
    authoritative_output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Validate one checked shard and optionally execute/publish its gate."""

    if not isinstance(config, V5K1DatasetGateConfig):
        raise TypeError("config must be V5K1DatasetGateConfig")
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    dataset_root = allowed_root if dataset_allowed_root is None else dataset_allowed_root
    dataset_file = _under_root(Path(dataset_path), dataset_root, "dataset_path")
    output = _under_root(Path(output_dir), allowed_root, "output_dir")
    recorded_output = (
        output
        if authoritative_output_dir is None
        else _under_root(Path(authoritative_output_dir), allowed_root, "authoritative_output_dir")
    )
    if not dataset_file.is_file():
        raise FileNotFoundError(f"checked grouped dataset does not exist: {dataset_file}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")

    if not dry_run:
        if (
            not isinstance(launch_binding, Mapping)
            or launch_binding.get("schema") != PHASE_A_LAUNCH_BINDING_SCHEMA
            or launch_binding.get("stage") not in {"smoke_gate", "full_gate"}
            or job_local_input_capability is None
        ):
            raise ValueError("real Phase-A gate requires its v7 launch binding and capability")
        launch_binding = validate_launch_binding_payload(launch_binding)
        _validate_phase_a_capability(
            job_local_input_capability,
            stage=str(launch_binding["stage"]),
            phase="pre_use",
        )

    dataset, receipt = read_v5_grouped_dataset(dataset_file)
    validation = validate_v5_k1_memorization_dataset(dataset)
    training_evidence = _validated_training_evidence(
        evidence,
        dataset_file=dataset_file,
        dataset_root=Path(dataset_root),
    )
    gate_config, model_config = config.resolved()
    common = {
        "schema_version": V5_K1_DATASET_GATE_SCHEMA,
        "version": V5_K1_DATASET_GATE_VERSION,
        "scientific_role": V5_K1_DATASET_GATE_ROLE,
        "model_acceptance_evidence": False,
        "requested_config": asdict(config),
        "resolved_gate_config": asdict(gate_config),
        "resolved_model_config": model_config,
        "live_model_identity": {
            "schema_version": MODEL_V5_SCHEMA,
            "model_version": MODEL_V5_VERSION,
            "model_name": MODEL_V5_NAME,
        },
        "model_contract": model_v5_contract_payload(),
        "single_branch_k1_gate_contract": _single_branch_k1_gate_contract(),
        "dataset": _dataset_identity(dataset_file, dataset, receipt),
        "dataset_validation": validation,
        "source": _source_identity(Path(source_root)),
        "training_evidence": training_evidence,
        "output_dir": str(recorded_output),
        "launch_binding": None if launch_binding is None else dict(launch_binding),
        "job_local_capability": None,
    }
    if dry_run:
        return {
            **common,
            "status": "checked_dry_run",
            "writes_performed": False,
            "gate_executed": False,
        }

    host = socket.gethostname() if hostname is None else hostname
    selected_environment = os.environ if environment is None else environment
    job_id = _execution_guard(host, selected_environment)
    if job_id != launch_binding["slurm_job_id"]:
        raise RuntimeError("current Slurm job id differs from the Phase-A launch binding")
    model, gate_result = run_v5_memorization_gate(
        dataset,
        lambda: build_branch_conditioned_proposal_model(
            max_points=validation["max_points"], **model_config
        ),
        config=gate_config,
    )
    validate_model_v5_graph_contract(model)

    if (
        _validated_training_evidence(
            evidence,
            dataset_file=dataset_file,
            dataset_root=Path(dataset_root),
        )
        != training_evidence
    ):
        raise RuntimeError("Phase-A training evidence changed during model optimization")

    _validate_phase_a_capability(
        job_local_input_capability,
        stage=str(launch_binding["stage"]),
        phase="post_use",
    )
    common["job_local_capability"] = _consumed_capability_payload(
        job_local_input_capability
    )

    output.mkdir(mode=0o700, exist_ok=False)
    model_identity = _publish_model(model, output / MODEL_FILENAME)
    model_provenance_core = {
        "schema": V5_K1_MODEL_PROVENANCE_SCHEMA,
        "version": V5_K1_MODEL_PROVENANCE_VERSION,
        "status": "BOUND_TO_INPUTS_PENDING_STAGE_COMPLETION",
        "model": dict(model_identity),
        "training_evidence": training_evidence,
        "launch_binding": dict(launch_binding),
        "job_local_capability": common["job_local_capability"],
    }
    model_provenance = {
        **model_provenance_core,
        "binding_sha256": sha256(
            canonical_json(model_provenance_core).encode("utf-8")
        ).hexdigest(),
    }
    model_provenance_path = output / MODEL_PROVENANCE_FILENAME
    _publish_json(model_provenance_path, model_provenance)
    model_provenance_file = _file_identity(model_provenance_path)
    if (
        _strict_object(
            model_provenance_path.read_text(encoding="utf-8"), "model provenance"
        )
        != model_provenance
        or _file_identity(model_provenance_path) != model_provenance_file
    ):
        raise RuntimeError("published model provenance did not replay exactly")
    if (
        _validated_training_evidence(
            evidence,
            dataset_file=dataset_file,
            dataset_root=Path(dataset_root),
        )
        != training_evidence
    ):
        raise RuntimeError("Phase-A training evidence changed during model publication")
    stage_a_passed = bool(gate_result.passed)
    status = (
        "stage_a_smoke_completed"
        if config.smoke
        else ("stage_a_passed" if stage_a_passed else "stage_a_failed")
    )
    core = {
        **common,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "writes_performed": True,
        "gate_executed": True,
        "stage_a_pass_enforced": not config.smoke,
        "stage_a_configured_wiring_gate_passed": stage_a_passed,
        "stage_a_passed": stage_a_passed,
        "stage_a_memorization_result": gate_result.audit_payload(),
        "single_branch_phase_b_gate_passed": False,
        "complete_k1_proposal_exact_gate_passed": False,
        "full_k1_all_legal_branches_gate_status": (
            "pending_fail_closed_requires_balanced_12_branch_cohort"
        ),
        "model_artifact": {
            **model_identity,
            "provenance": {
                "filename": MODEL_PROVENANCE_FILENAME,
                **model_provenance_file,
                "binding_sha256": model_provenance["binding_sha256"],
            },
        },
        "execution": {
            "hostname": host,
            "slurm_job_id": job_id,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
        },
        "publication": "exclusive_output_directory_and_atomic_exclusive_files",
    }
    payload = {
        **core,
        "result_payload_sha256": sha256(canonical_json(core).encode()).hexdigest(),
    }
    _publish_json(output / RESULT_FILENAME, payload)
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dataset-binding", required=True, type=Path)
    parser.add_argument("--cross-platform-pass-marker", required=True, type=Path)
    parser.add_argument("--original-dataset-path", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--reference-file-sha256", required=True)
    parser.add_argument("--reference-file-byte-count", required=True, type=int)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--scientific-content-sha256", required=True)
    parser.add_argument("--comparison-result-sha256", required=True)
    parser.add_argument("--gate-claim-sha256", required=True)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--final-learning-rate", type=float, default=3.0e-5)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine_decay"),
        default="cosine_decay",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--local-mdn-weight", type=float, default=1.0)
    parser.add_argument(
        "--local-coverage-weight",
        type=float,
        default=DEFAULT_LOCAL_COVERAGE_WEIGHT,
    )
    parser.add_argument(
        "--operational-top-l-alignment-weight", type=float, default=1.0
    )
    parser.add_argument("--max-final-target-median-rms", type=float, default=0.01)
    parser.add_argument("--minimum-loss-reduction", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--encoder-blocks", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_v5_k1_dataset_memorization_gate(
        args.dataset,
        args.output_dir,
        source_root=args.source_root,
        evidence=V5K1PhaseATrainingEvidence(
            dataset_binding_path=args.dataset_binding,
            cross_platform_pass_marker_path=args.cross_platform_pass_marker,
            original_dataset_path=args.original_dataset_path,
            source_archive_sha256=args.source_archive_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            source_tree_sha256=args.source_tree_sha256,
            reference_file_sha256=args.reference_file_sha256,
            reference_file_byte_count=args.reference_file_byte_count,
            reference_manifest_sha256=args.reference_manifest_sha256,
            scientific_content_sha256=args.scientific_content_sha256,
            comparison_result_sha256=args.comparison_result_sha256,
            gate_claim_sha256=args.gate_claim_sha256,
        ),
        config=V5K1DatasetGateConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            final_learning_rate=args.final_learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            seed=args.seed,
            local_mdn_weight=args.local_mdn_weight,
            local_coverage_weight=args.local_coverage_weight,
            operational_top_l_alignment_weight=(
                args.operational_top_l_alignment_weight
            ),
            max_final_target_median_rms=args.max_final_target_median_rms,
            minimum_loss_reduction=args.minimum_loss_reduction,
            width=args.width,
            encoder_blocks=args.encoder_blocks,
            smoke=args.smoke,
        ),
        dry_run=args.dry_run,
        dataset_allowed_root=args.dataset_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.dry_run or args.smoke:
        return 0
    return 0 if result["stage_a_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MAXWELL_DUST_ROOT",
    "MODEL_FILENAME",
    "MODEL_PROVENANCE_FILENAME",
    "RESULT_FILENAME",
    "V5_K1_DATASET_GATE_ROLE",
    "V5_K1_DATASET_GATE_PUBLISHED_RESULT_FIELDS",
    "V5_K1_DATASET_GATE_SCHEMA",
    "V5_K1_DATASET_GATE_VERSION",
    "V5_K1_MODEL_PROVENANCE_SCHEMA",
    "V5_K1_MODEL_PROVENANCE_VERSION",
    "V5_K1_TRAINING_EVIDENCE_SCHEMA",
    "V5_K1_TRAINING_EVIDENCE_VERSION",
    "V5K1DatasetGateConfig",
    "V5K1PhaseATrainingEvidence",
    "main",
    "run_v5_k1_dataset_memorization_gate",
    "validate_v5_k1_memorization_dataset",
]
