"""Fail-closed worker replay for V5 formal-production contract smoke.

No curve is generated until :func:`prepare_v5_formal_production_worker_shard`
has reproduced the global plan, source, scientific artifacts, stage, shard,
selected observation view, and exact output membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from .calibrated_search_threshold_v5 import (
    V5CheckedCompatibilityCalibration,
    read_v5_checked_compatibility_calibration,
)
from .exact_search_executor_v5 import build_v5_frozen_exact_search_protocol
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    read_v5_frozen_local_sobol_schedule,
)
from .formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
    formal_label_observation_policy_payload,
)
from .formal_production_search_contract_v5 import (
    V5FormalProductionSearchShard,
    V5FormalProductionSearchStage,
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
    formal_production_query_contract_payload,
    plan_v5_formal_production_search_shard,
)
from .formal_production_search_plan_v5 import (
    V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA,
    V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION,
    V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED,
)
from .formal_production_search_runtime_v5 import (
    V5FormalProductionExecutableShard,
)
from .frozen_search_launch_contracts_v5 import (
    fingerprint_v5_frozen_search_source,
)
from .frozen_search_pipeline_contract_v5 import (
    V5SelectedTopologySearchSchedule,
)
from .frozen_search_pipeline_v5 import V5FrozenSearchExecution
from .grouped_artifact_v5 import canonical_json
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from .sobol_design_v5 import V5SobolDesign
from .split_design_v5 import V5SplitPlan


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GLOBAL_PLAN_FIELDS = {
    "schema",
    "version",
    "study_id",
    "scientific_scope",
    "source",
    "source_identity_sha256",
    "split_plan",
    "sobol_design",
    "query_contract",
    "query_contract_sha256",
    "observation_selection_contract",
    "calibration_identity",
    "calibration_identity_sha256",
    "stage_order",
    "stages",
    "shards",
    "membership_summary",
    "non_overlap",
    "promotion_boundary",
    "plan_sha256",
}


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _same_json(left: object, right: object) -> bool:
    """Compare payloads after the tuple-to-array JSON boundary."""

    return canonical_json(left) == canonical_json(right)


def validate_v5_formal_production_global_plan_payload(
    payload: object, *, expected_plan_sha256: str
) -> dict[str, object]:
    """Validate an already decoded global plan against all live contracts."""

    expected = _digest(expected_plan_sha256, "expected_plan_sha256")
    if not isinstance(payload, dict) or set(payload) != _GLOBAL_PLAN_FIELDS:
        raise ValueError("formal global plan fields are incomplete or unsupported")
    if (
        payload["schema"] != V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA
        or payload["version"] != V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION
    ):
        raise ValueError("unsupported formal global plan contract")
    recorded = _digest(payload["plan_sha256"], "plan_sha256")
    core = {key: value for key, value in payload.items() if key != "plan_sha256"}
    replayed = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    if recorded != replayed or recorded != expected:
        raise ValueError("formal global plan SHA-256 does not reproduce")
    promotion = _mapping(payload["promotion_boundary"], "promotion_boundary")
    if promotion.get("training_promotion_enabled_by_this_module") is not (
        V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED
    ):
        raise ValueError("formal global plan promotion capability drifted")
    # This worker remains non-training unless a separate, checked per-shard
    # authorization is explicitly supplied to the execution boundary.
    if promotion.get("membership_proof_alone_authorizes_training") is not False:
        raise ValueError("formal global plan weakened the evidence promotion boundary")
    if payload["query_contract"] != formal_production_query_contract_payload() or (
        payload["query_contract_sha256"]
        != V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256
    ):
        raise ValueError("formal query contract changed after planning")
    observation = _mapping(
        payload["observation_selection_contract"],
        "observation_selection_contract",
    )
    if (
        not _same_json(
            observation.get("payload"), formal_label_observation_policy_payload()
        )
        or observation.get("policy_id")
        != V5_FORMAL_LABEL_OBSERVATION_POLICY_ID
        or observation.get("policy_sha256")
        != V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256
        or observation.get("selected_views_per_clean_parent") != 1
        or observation.get("measurement_sigma_required") is not True
    ):
        raise ValueError("formal observation-selection contract drifted")
    if payload["stage_order"] != list(V5_FORMAL_PRODUCTION_STAGE_IDS):
        raise ValueError("formal stage order changed after planning")
    return dict(payload)


def read_v5_formal_production_global_plan(
    path: str | Path, *, expected_plan_sha256: str
) -> dict[str, object]:
    """Read a canonical global plan and reject stale/live-contract drift."""

    selected = Path(path)
    before = selected.read_bytes()
    try:
        payload = json.loads(before)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("formal global plan is not valid JSON") from exc
    after = selected.read_bytes()
    if before != after:
        raise RuntimeError("formal global plan changed while it was being read")
    return validate_v5_formal_production_global_plan_payload(
        payload, expected_plan_sha256=expected_plan_sha256
    )


def _verify_source(
    source_root: Path, global_plan: Mapping[str, object]
) -> V5FormalProductionSourceIdentity:
    observed = V5FormalProductionSourceIdentity.from_fingerprint(
        fingerprint_v5_frozen_search_source(source_root)
    )
    if (
        observed.audit_payload() != global_plan["source"]
        or observed.sha256 != global_plan["source_identity_sha256"]
    ):
        raise RuntimeError("source bundle changed after formal planning")
    return observed


def _stage_from_artifacts(
    *,
    stage_payload: Mapping[str, object],
    schedule_path: Path,
    calibration: V5CheckedCompatibilityCalibration,
) -> V5FormalProductionSearchStage:
    row = dict(stage_payload)
    recorded_stage_sha = _digest(row.pop("stage_sha256"), "stage_sha256")
    schedule, _ = read_v5_frozen_local_sobol_schedule(schedule_path)
    schedule.verify_runtime_replay()
    if (
        not _same_json(schedule.audit_payload(), row.get("local_sobol_schedule"))
        or schedule.sha256 != row.get("local_sobol_schedule_sha256")
    ):
        raise ValueError("formal local-Sobol schedule changed after planning")
    optimizer = V5FrozenExactOptimizerSchedule(
        **_mapping(row.get("optimizer_schedule"), "optimizer_schedule")
    )
    if optimizer.sha256 != row.get("optimizer_schedule_sha256"):
        raise ValueError("formal optimizer schedule does not reproduce")
    topology_payload = _mapping(row.get("topology_schedule"), "topology_schedule")
    topology = V5SelectedTopologySearchSchedule(
        schedule_id=topology_payload.get("schedule_id"),
        selected_topology_ids=tuple(topology_payload.get("selected_topology_ids", ())),
        schema_version=topology_payload.get("schema_version"),
        version=topology_payload.get("version"),
    )
    if (
        not _same_json(topology.audit_payload(), topology_payload)
        or topology.sha256 != row.get("topology_schedule_sha256")
    ):
        raise ValueError("formal topology schedule does not reproduce")
    protocol_payload = _mapping(row.get("protocol"), "protocol")
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id=protocol_payload.get("protocol_id"),
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        calibration_identity=calibration.identity,
        delta_separation_threshold=protocol_payload.get(
            "delta_separation_threshold"
        ),
    )
    if (
        not _same_json(protocol.audit_payload(), protocol_payload)
        or protocol.sha256 != row.get("protocol_sha256")
    ):
        raise ValueError("formal paper/full protocol does not reproduce")
    stage = V5FormalProductionSearchStage(
        stage_id=row.get("stage_id"),
        topology_schedule=topology,
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        protocol=protocol,
        schema=row.get("schema"),
        version=row.get("version"),
    )
    if not _same_json(stage.audit_payload(), row) or stage.sha256 != recorded_stage_sha:
        raise ValueError("formal stage does not reproduce its global member")
    return stage


def replay_v5_formal_production_stage_from_artifacts(
    *,
    stage_payload: Mapping[str, object],
    schedule_path: str | Path,
    calibration: V5CheckedCompatibilityCalibration,
) -> V5FormalProductionSearchStage:
    """Replay one launch-bound formal stage for a worker adapter."""

    return _stage_from_artifacts(
        stage_payload=stage_payload,
        schedule_path=Path(schedule_path),
        calibration=calibration,
    )


def _select_shard_payload(
    global_plan: Mapping[str, object],
    *,
    stage_id: str,
    target_split: str,
    array_task_id: int,
    expected_shard_plan_sha256: str,
) -> dict[str, object]:
    if isinstance(array_task_id, bool) or not isinstance(array_task_id, int):
        raise TypeError("array_task_id must be an integer")
    rows = global_plan.get("shards")
    if not isinstance(rows, list):
        raise ValueError("formal global plan has no shard list")
    matching = [
        _mapping(value, "shard")
        for value in rows
        if isinstance(value, Mapping)
        and value.get("stage_id") == stage_id
        and value.get("target_split") == target_split
    ]
    if array_task_id < 0 or array_task_id >= len(matching):
        raise ValueError("Slurm array task has no unique formal shard member")
    selected = matching[array_task_id]
    if selected.get("shard_plan_sha256") != _digest(
        expected_shard_plan_sha256, "expected_shard_plan_sha256"
    ):
        raise ValueError("Slurm task/shard SHA-256 binding does not match")
    return selected


def _safe_output_path(
    run_root: Path, output_relative_path: object
) -> Path:
    if run_root.is_symlink():
        raise ValueError("formal run root cannot be a symlink")
    root = run_root.resolve(strict=True)
    if not isinstance(output_relative_path, str):
        raise ValueError("shard output path must be text")
    relative = PurePosixPath(output_relative_path)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ValueError("shard output path is unsafe")
    lexical_output = root / Path(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("shard output escaped or traversed a symlink")
    output = lexical_output.resolve(strict=False)
    if output == root or root not in output.parents:
        raise ValueError("shard output escaped the formal run root")
    if lexical_output.exists() or lexical_output.is_symlink():
        raise FileExistsError("formal shard output already exists; reuse is forbidden")
    return output


@dataclass(frozen=True)
class V5PreparedFormalProductionWorkerShard:
    global_plan_sha256: str
    source_root: Path
    output_root: Path
    runtime_shard: V5FormalProductionExecutableShard
    execution: V5FrozenSearchExecution

    def verify_source_bundle(self) -> str:
        identity = _verify_source(
            self.source_root,
            {"source": self._source_payload, "source_identity_sha256": self._source_sha},
        )
        return identity.bundle_sha256

    # Private immutable copies let the publication guards recheck the full
    # source identity, not merely a caller-supplied digest.
    _source_payload: Mapping[str, object]
    _source_sha: str


def prepare_v5_formal_production_worker_shard(
    *,
    global_plan_path: str | Path,
    expected_global_plan_sha256: str,
    expected_shard_plan_sha256: str,
    stage_id: str,
    target_split: str,
    array_task_id: int,
    source_root: str | Path,
    run_root: str | Path,
    split_plan_path: str | Path,
    sobol_design_path: str | Path,
    local_sobol_schedule_path: str | Path,
    calibration_path: str | Path,
) -> V5PreparedFormalProductionWorkerShard:
    """Replay every binding needed before the established curve generator runs."""

    global_plan = read_v5_formal_production_global_plan(
        global_plan_path, expected_plan_sha256=expected_global_plan_sha256
    )
    source = _verify_source(Path(source_root), global_plan)
    split_plan = V5SplitPlan.from_json(Path(split_plan_path).read_text(encoding="utf-8"))
    split_row = _mapping(global_plan["split_plan"], "split_plan")
    if (
        not _same_json(json.loads(split_plan.canonical_json), split_row.get("payload"))
        or split_plan.sha256 != split_row.get("sha256")
    ):
        raise ValueError("formal split plan changed after planning")
    sobol_design = V5SobolDesign.from_json(
        Path(sobol_design_path).read_text(encoding="utf-8")
    )
    design_row = _mapping(global_plan["sobol_design"], "sobol_design")
    if (
        not _same_json(sobol_design.payload(), design_row.get("payload"))
        or sobol_design.sha256 != design_row.get("sha256")
    ):
        raise ValueError("formal Sobol design changed after planning")
    calibration_identity = _mapping(
        global_plan["calibration_identity"], "calibration_identity"
    )
    calibration = read_v5_checked_compatibility_calibration(
        calibration_path,
        expected_artifact_sha256=calibration_identity.get("artifact_sha256"),
        expected_file_sha256=calibration_identity.get("file_sha256"),
    )
    if (
        not _same_json(calibration.identity.audit_payload(), calibration_identity)
        or calibration.identity.sha256 != global_plan["calibration_identity_sha256"]
    ):
        raise ValueError("checked calibration identity changed after planning")
    stages = global_plan.get("stages")
    matching_stages = [
        value
        for value in stages if isinstance(value, Mapping) and value.get("stage_id") == stage_id
    ] if isinstance(stages, list) else []
    if len(matching_stages) != 1:
        raise ValueError("worker stage is not a unique global-plan member")
    stage = _stage_from_artifacts(
        stage_payload=matching_stages[0],
        schedule_path=Path(local_sobol_schedule_path),
        calibration=calibration,
    )
    shard_row = _select_shard_payload(
        global_plan,
        stage_id=stage_id,
        target_split=target_split,
        array_task_id=array_task_id,
        expected_shard_plan_sha256=expected_shard_plan_sha256,
    )
    raw_shard_sha = _digest(shard_row.pop("shard_plan_sha256"), "shard_plan_sha256")
    recipes = shard_row.get("recipes")
    if not isinstance(recipes, list):
        raise ValueError("formal shard has no explicit recipes")
    shard = plan_v5_formal_production_search_shard(
        split_plan=split_plan,
        sobol_design=sobol_design,
        stage=stage,
        target_split=target_split,
        sobol_indices=tuple(value["sobol_index"] for value in recipes),
        output_relative_path=shard_row.get("output_relative_path"),
        candidate_view_indices=tuple(shard_row.get("candidate_view_indices", ())),
    )
    if not _same_json(shard.audit_payload(), shard_row) or shard.sha256 != raw_shard_sha:
        raise ValueError("formal shard or selected singleton view does not replay")
    runtime = V5FormalProductionExecutableShard.replay(
        shard, split_plan=split_plan, sobol_design=sobol_design
    )
    output = _safe_output_path(Path(run_root), shard.output_relative_path)
    execution = V5FrozenSearchExecution(
        seed_schedule=stage.seed_schedule,
        optimizer_schedule=stage.optimizer_schedule,
        protocol=stage.protocol,
        calibration=calibration,
        launch_source_bundle_sha256=source.bundle_sha256,
        launch_plan_sha256=global_plan["plan_sha256"],
    )
    return V5PreparedFormalProductionWorkerShard(
        global_plan_sha256=global_plan["plan_sha256"],
        source_root=Path(source_root).resolve(strict=True),
        output_root=output,
        runtime_shard=runtime,
        execution=execution,
        _source_payload=dict(global_plan["source"]),
        _source_sha=global_plan["source_identity_sha256"],
    )


__all__ = [
    "V5PreparedFormalProductionWorkerShard",
    "prepare_v5_formal_production_worker_shard",
    "read_v5_formal_production_global_plan",
    "replay_v5_formal_production_stage_from_artifacts",
    "validate_v5_formal_production_global_plan_payload",
]
