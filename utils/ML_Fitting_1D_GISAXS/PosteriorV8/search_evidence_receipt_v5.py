"""Task-bound audit receipts for V5.1 frozen-search training evidence.

The search sidecar is a compact training view.  It is not, by itself, proof
that the referenced executor artifacts exist or were replayed against their
original tasks.  This module performs that expensive audit once, records every
evidence file by relative path and digest, and lets the trainer require the
resulting immutable receipt before consuming full-stage labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Callable, Mapping, Sequence
import zipfile

import numpy as np

from .candidate_supervision_v5 import SEARCH_OUTCOME_CODE
from .exact_search_executor_v5 import (
    V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
    V5_EXACT_SEARCH_EXECUTOR_VERSION,
    read_v5_exact_search_executor_artifact,
    v5_exact_search_artifact_path,
)
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import (
    clean_array,
    observation_array,
    read_v5_grouped_dataset,
)
from .frozen_search_pipeline_contract_v5 import (
    V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX,
    V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX,
)
from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
    V5_FORMAL_PRODUCTION_CONSUMER_ROLES,
)
from .search_supervision_contract_v5 import (
    V5FrozenExactSearchProtocol,
    V5FrozenSearchTask,
    V5UniversalSearchSpec,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from .search_supervision_sidecar_v5 import (
    branch_array,
    query_array,
    read_v5_search_supervision_sidecar,
)


V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA = (
    "gisaxs.posterior_v8.task_bound_search_evidence_receipt/v4"
)
V5_SEARCH_EVIDENCE_RECEIPT_VERSION = (
    "posterior_v8_formal_plan_authorized_role_isolated_executor_replay_v4"
)
V5_SEARCH_EVIDENCE_AUDIT_POLICY = (
    "every_completed_sidecar_branch_task_bound_replayed_against_authoritative_forward_v1"
)
V5_SEARCH_LABEL_PURPOSE_PILOT = "engineering_throughput_pilot_not_training_eligible"
V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE = (
    "formal_calibrated_contract_smoke_not_training_eligible"
)
V5_SEARCH_LABEL_PURPOSE_TRAINING = "model_development_full_training"
V5_SEARCH_LABEL_PURPOSES = (
    V5_SEARCH_LABEL_PURPOSE_PILOT,
    V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
)
V5_SEARCH_TRAINING_RECEIPT_PROMOTION_ENABLED = True
V5_SEARCH_EVIDENCE_RECEIPT_SUFFIX = ".evidence-receipt.json"
V5_SEARCH_LABEL_BINDING_SCHEMA = "gisaxs.posterior_v8.frozen_search_label_binding/v4"
V5_SEARCH_PIPELINE_TRAINING_SIDECAR_PREFIX = "formal-production-training-"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_FIELDS = {
    "schema",
    "version",
    "receipt_id",
    "label_purpose",
    "full_training_eligible",
    "audit_policy",
    "launch_source_bundle_sha256",
    "launch_plan_sha256",
    "shard_plan_sha256",
    "label_binding",
    "label_binding_sha256",
    "formal_production_authorization",
    "formal_production_authorization_sha256",
    "executor_root_relative_path",
    "parent",
    "sidecar",
    "executor_source_sha256",
    "executor_source_bundle_sha256",
    "branch_evidence",
    "counts",
    "receipt_sha256",
}
_ENTRY_FIELDS = {
    "branch_row",
    "query_index",
    "global_branch_key",
    "exact_curve_sha256",
    "relative_path",
    "artifact_id",
    "artifact_sha256",
    "artifact_schema",
    "artifact_version",
    "task_audit_sha256",
    "executor_task_payload_sha256",
    "outcome",
    "exact_forward_calls_used",
}


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_receipt_path_for_sidecar(
    sidecar_path: str | os.PathLike[str],
) -> Path:
    path = Path(sidecar_path)
    return path.with_suffix(V5_SEARCH_EVIDENCE_RECEIPT_SUFFIX)


def _task_sha256(task_payload: Mapping[str, object]) -> str:
    return sha256(canonical_json(dict(task_payload)).encode("utf-8")).hexdigest()


def _source_bundle_sha256(source_hashes: Mapping[str, object]) -> str:
    return sha256(canonical_json(dict(source_hashes)).encode("utf-8")).hexdigest()


def _validate_label_purpose_for_protocol(
    label_purpose: str,
    protocol: V5FrozenExactSearchProtocol,
) -> bool:
    formal = (
        protocol.protocol_tier
        == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
    )
    if label_purpose == V5_SEARCH_LABEL_PURPOSE_PILOT and formal:
        raise ValueError("engineering-pilot evidence requires an engineering protocol")
    if label_purpose in {
        V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
        V5_SEARCH_LABEL_PURPOSE_TRAINING,
    } and not formal:
        raise ValueError("formal search evidence requires a calibrated formal protocol")
    return formal


def _validate_formal_authorization_binding(
    authorization: V5FormalProductionSearchAuthorization | None,
    *,
    label_purpose: str,
    protocol: V5FrozenExactSearchProtocol,
    launch_source_bundle_sha256: str,
    launch_plan_sha256: str,
    shard_plan_sha256: str,
) -> V5FormalProductionSearchAuthorization | None:
    if label_purpose != V5_SEARCH_LABEL_PURPOSE_TRAINING:
        if authorization is not None:
            raise ValueError("pilot/smoke evidence cannot carry a production authorization")
        return None
    if not V5_SEARCH_TRAINING_RECEIPT_PROMOTION_ENABLED:  # pragma: no cover
        raise RuntimeError("training receipt promotion is disabled")
    if not isinstance(authorization, V5FormalProductionSearchAuthorization):
        raise ValueError(
            "TRAINING evidence requires a plan-derived formal production authorization"
        )
    calibration_sha = (
        None
        if protocol.calibration_identity is None
        else protocol.calibration_identity.artifact_sha256
    )
    if (
        authorization.launch_source_bundle_sha256 != launch_source_bundle_sha256
        or authorization.launch_plan_sha256 != launch_plan_sha256
        or authorization.shard_plan_sha256 != shard_plan_sha256
        or authorization.protocol_sha256 != protocol.sha256
        or authorization.seed_schedule_sha256 != protocol.seed_schedule_sha256
        or authorization.optimizer_schedule_sha256 != protocol.optimizer_schedule_sha256
        or authorization.calibration_artifact_sha256 != calibration_sha
        or authorization.consumer_role not in V5_FORMAL_PRODUCTION_CONSUMER_ROLES
    ):
        raise ValueError("formal production authorization escaped its launch/protocol binding")
    return authorization


def build_v5_search_label_binding(
    *,
    protocol: V5FrozenExactSearchProtocol,
    seed_schedule_sha256: str,
    optimizer_schedule_sha256: str,
    launch_source_bundle_sha256: str,
    launch_plan_sha256: str,
    shard_plan_sha256: str,
    label_purpose: str,
    formal_production_authorization: (
        V5FormalProductionSearchAuthorization | None
    ) = None,
) -> dict[str, object]:
    """Build the canonical identity used by sidecar, receipt, and completion."""

    if label_purpose not in V5_SEARCH_LABEL_PURPOSES:
        raise ValueError("label_purpose is unsupported")
    _validate_label_purpose_for_protocol(label_purpose, protocol)
    seed_sha = _digest(seed_schedule_sha256, "seed_schedule_sha256")
    optimizer_sha = _digest(
        optimizer_schedule_sha256, "optimizer_schedule_sha256"
    )
    if (
        seed_sha != protocol.seed_schedule_sha256
        or optimizer_sha != protocol.optimizer_schedule_sha256
    ):
        raise ValueError("label binding schedules disagree with the frozen protocol")
    calibration_sha = (
        None
        if protocol.calibration_identity is None
        else protocol.calibration_identity.artifact_sha256
    )
    authorization = _validate_formal_authorization_binding(
        formal_production_authorization,
        label_purpose=label_purpose,
        protocol=protocol,
        launch_source_bundle_sha256=_digest(
            launch_source_bundle_sha256, "launch_source_bundle_sha256"
        ),
        launch_plan_sha256=_digest(launch_plan_sha256, "launch_plan_sha256"),
        shard_plan_sha256=_digest(shard_plan_sha256, "shard_plan_sha256"),
    )
    core = {
        "schema": V5_SEARCH_LABEL_BINDING_SCHEMA,
        "launch_source_bundle_sha256": _digest(
            launch_source_bundle_sha256, "launch_source_bundle_sha256"
        ),
        "launch_plan_sha256": _digest(launch_plan_sha256, "launch_plan_sha256"),
        "shard_plan_sha256": _digest(shard_plan_sha256, "shard_plan_sha256"),
        "protocol_sha256": protocol.sha256,
        "protocol_tier": protocol.protocol_tier,
        "seed_schedule_sha256": seed_sha,
        "optimizer_schedule_sha256": optimizer_sha,
        "calibration_artifact_sha256": calibration_sha,
        "formal_production_authorization_sha256": (
            None if authorization is None else authorization.sha256
        ),
        "authorized_consumer_role": (
            None if authorization is None else authorization.consumer_role
        ),
        "label_purpose": label_purpose,
        "full_training_eligible": label_purpose == V5_SEARCH_LABEL_PURPOSE_TRAINING,
    }
    return {
        **core,
        "label_binding_sha256": sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def _sidecar_prefix_for_label_purpose(label_purpose: str) -> str:
    if label_purpose == V5_SEARCH_LABEL_PURPOSE_PILOT:
        return V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX
    if label_purpose == V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE:
        return V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX
    if label_purpose == V5_SEARCH_LABEL_PURPOSE_TRAINING:
        return V5_SEARCH_PIPELINE_TRAINING_SIDECAR_PREFIX
    raise ValueError("label purpose has no sidecar prefix")


def _executor_manifest_metadata(path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            member = archive.getinfo("manifest.json")
            if member.file_size > 8 * 1024 * 1024:
                raise ValueError("executor manifest is unexpectedly large")
            value = json.loads(archive.read(member))
    except (KeyError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError("executor evidence has no readable checked manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("executor evidence manifest must be an object")
    core = dict(value)
    supplied = _digest(core.pop("manifest_sha256", None), "executor manifest SHA-256")
    if supplied != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("executor evidence manifest SHA-256 does not reproduce")
    return value


def _safe_relative_path(
    path: Path,
    base: Path,
    *,
    expected_kind: str,
) -> str:
    if path.is_symlink():
        raise ValueError(f"{expected_kind} cannot be a symbolic link")
    selected = path.resolve(strict=True)
    root = base.resolve(strict=True)
    if selected == root or root not in selected.parents:
        raise ValueError(f"{expected_kind} must be below the receipt root")
    if expected_kind == "executor evidence" and not selected.is_file():
        raise ValueError("executor evidence must be a regular file")
    if expected_kind == "executor evidence root" and not selected.is_dir():
        raise ValueError("executor evidence root must be a directory")
    relative = selected.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or not relative:
        raise ValueError("executor evidence relative path is unsafe")
    return relative


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _ordered_tasks(
    specs: Sequence[V5UniversalSearchSpec],
    protocol: V5FrozenExactSearchProtocol,
    parent,
) -> tuple[V5FrozenSearchTask, ...]:
    ordered = tuple(sorted(specs, key=lambda value: value.parent_observation_index))
    result = []
    for query_index, spec in enumerate(ordered):
        observation_index = spec.parent_observation_index
        recipe_index = int(
            parent.arrays[observation_array("recipe_index")][observation_index]
        )
        for branch_index in range(spec.context.branch_count):
            result.append(
                V5FrozenSearchTask(
                    query_index=query_index,
                    clean_group_id=str(
                        parent.arrays[clean_array("clean_group_id")][recipe_index]
                    ),
                    recipe_id=str(
                        parent.arrays[clean_array("recipe_id")][recipe_index]
                    ),
                    observation_id=str(
                        parent.arrays[observation_array("observation_id")][
                            observation_index
                        ]
                    ),
                    universal_context=spec.context,
                    exact_observation=spec.exact_observation,
                    query_catalog_artifact_id=spec.query_catalog_artifact_id,
                    query_catalog_artifact_sha256=spec.query_catalog_artifact_sha256,
                    branch_index=branch_index,
                    protocol=protocol,
                    calibrated_threshold=spec.calibrated_threshold,
                )
            )
    return tuple(result)


def _compare_executor_to_sidecar(
    *,
    sidecar,
    row: int,
    task: V5FrozenSearchTask,
    artifact,
    executor_relative_path: str,
    executor_source_bundle_sha256: str,
) -> None:
    arrays = sidecar.arrays
    manifest = artifact.manifest
    representatives = json.loads(
        str(arrays[branch_array("compatible_representatives_json")][row])
    )
    expected = {
        "query_index": int(arrays[branch_array("query_index")][row]),
        "global_branch_key": str(arrays[branch_array("global_branch_key")][row]),
        "exact_curve_sha256": str(
            arrays[query_array("exact_curve_sha256")][task.query_index]
        ),
        "artifact_id": str(arrays[branch_array("executor_artifact_id")][row]),
        "artifact_sha256": str(
            arrays[branch_array("executor_artifact_sha256")][row]
        ),
        "outcome": next(
            name
            for name, code in SEARCH_OUTCOME_CODE.items()
            if code
            == int(
                sidecar.arrays[
                    "branch__label__search_outcome_code"
                ][row]
            )
        ),
        "completed": bool(arrays[branch_array("runner_completed")][row]),
        "termination_reason": str(
            arrays[branch_array("runner_termination_reason")][row]
        ),
        "exact_forward_calls_used": int(
            arrays[branch_array("exact_forward_calls_used")][row]
        ),
        "executor_artifact_relative_path": str(
            arrays[branch_array("executor_artifact_relative_path")][row]
        ),
        "executor_artifact_schema": str(
            arrays[branch_array("executor_artifact_schema")][row]
        ),
        "executor_artifact_version": str(
            arrays[branch_array("executor_artifact_version")][row]
        ),
        "executor_source_bundle_sha256": str(
            arrays[branch_array("executor_source_bundle_sha256")][row]
        ),
        "task_audit_sha256": str(
            arrays[branch_array("task_audit_sha256")][row]
        ),
    }
    if expected["query_index"] != task.query_index:
        raise ValueError("sidecar branch ordering escaped the replay task ordering")
    if manifest.get("artifact_id") != expected["artifact_id"]:
        raise ValueError("executor artifact ID disagrees with the sidecar")
    if artifact.receipt.artifact_sha256 != expected["artifact_sha256"]:
        raise ValueError("executor artifact digest disagrees with the sidecar")
    if (
        manifest.get("outcome") != expected["outcome"]
        or manifest.get("completed") is not expected["completed"]
        or manifest.get("termination_reason") != expected["termination_reason"]
        or manifest.get("ledger", {}).get("exact_forward_calls_used")
        != expected["exact_forward_calls_used"]
        or manifest.get("representatives") != representatives
    ):
        raise ValueError("executor outcome evidence disagrees with the sidecar label")
    if (
        expected["executor_artifact_relative_path"] != executor_relative_path
        or expected["executor_artifact_schema"]
        != V5_EXACT_SEARCH_EXECUTOR_SCHEMA
        or expected["executor_artifact_version"]
        != V5_EXACT_SEARCH_EXECUTOR_VERSION
        or expected["executor_source_bundle_sha256"]
        != executor_source_bundle_sha256
        or expected["task_audit_sha256"] != task.audit_sha256
        or manifest.get("task", {}).get("task_audit_sha256")
        != task.audit_sha256
    ):
        raise ValueError("executor task/evidence index disagrees with the sidecar")


@dataclass(frozen=True)
class V5SearchEvidenceReceipt:
    path: Path
    manifest: Mapping[str, object]
    file_sha256: str

    @property
    def sidecar_artifact_sha256(self) -> str:
        return str(self.manifest["sidecar"]["artifact_sha256"])

    @property
    def full_training_eligible(self) -> bool:
        return bool(self.manifest["full_training_eligible"])


def audit_and_write_v5_search_evidence_receipt(
    *,
    parent_dataset_path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
    specs: Sequence[V5UniversalSearchSpec],
    protocol: V5FrozenExactSearchProtocol,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
    executor_directory: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    launch_source_bundle_sha256: str,
    launch_plan_sha256: str,
    shard_plan_sha256: str,
    label_purpose: str,
    pre_publish_guard: Callable[[], object],
    formal_production_authorization: (
        V5FormalProductionSearchAuthorization | None
    ) = None,
) -> V5SearchEvidenceReceipt:
    """Replay every executor task and publish one immutable training gate."""

    if label_purpose not in V5_SEARCH_LABEL_PURPOSES:
        raise ValueError("label_purpose is unsupported")
    if not callable(pre_publish_guard):
        raise TypeError("pre_publish_guard must be callable")
    launch_sha = _digest(
        launch_source_bundle_sha256, "launch_source_bundle_sha256"
    )
    launch_plan_sha = _digest(launch_plan_sha256, "launch_plan_sha256")
    shard_plan_sha = _digest(shard_plan_sha256, "shard_plan_sha256")
    formal_protocol = _validate_label_purpose_for_protocol(label_purpose, protocol)
    authorization = _validate_formal_authorization_binding(
        formal_production_authorization,
        label_purpose=label_purpose,
        protocol=protocol,
        launch_source_bundle_sha256=launch_sha,
        launch_plan_sha256=launch_plan_sha,
        shard_plan_sha256=shard_plan_sha,
    )
    label_binding = build_v5_search_label_binding(
        protocol=protocol,
        seed_schedule_sha256=seed_schedule.sha256,
        optimizer_schedule_sha256=optimizer_schedule.sha256,
        launch_source_bundle_sha256=launch_sha,
        launch_plan_sha256=launch_plan_sha,
        shard_plan_sha256=shard_plan_sha,
        label_purpose=label_purpose,
        formal_production_authorization=authorization,
    )
    output = Path(output_path).resolve()
    evidence_root = Path(executor_directory).resolve(strict=True)
    parent_path = Path(parent_dataset_path).resolve(strict=True)
    selected_sidecar_path = Path(sidecar_path).resolve(strict=True)
    parent, parent_receipt = read_v5_grouped_dataset(parent_path)
    sidecar, sidecar_receipt = read_v5_search_supervision_sidecar(
        selected_sidecar_path
    )
    if authorization is not None:
        observed_source = pre_publish_guard()
        if observed_source != authorization.launch_source_bundle_sha256:
            raise RuntimeError(
                "formal source guard did not reproduce the authorized source bundle"
            )
        observed_indices = tuple(
            int(value) for value in parent.arrays[clean_array("sobol_index")]
        )
        observed_groups = tuple(
            str(value) for value in parent.arrays[clean_array("clean_group_id")]
        )
        observed_splits = {
            str(value) for value in parent.arrays[clean_array("split_id")]
        }
        if (
            observed_indices != authorization.recipe_sobol_indices
            or observed_groups != authorization.clean_group_ids
            or observed_splits != {authorization.target_split}
            or parent.recipe_count != authorization.expected_query_count
            or sidecar.manifest["split_id"] != authorization.target_split
        ):
            raise ValueError(
                "formal production parent/sidecar escaped authorized shard membership"
            )
    if sidecar.manifest["parent_grouped_artifact"]["artifact_sha256"] != (
        parent_receipt.artifact_sha256
    ):
        raise ValueError("sidecar and receipt audit parent do not match")
    if sidecar.manifest["protocol_sha256"] != protocol.sha256:
        raise ValueError("sidecar and receipt audit protocol do not match")
    expected_sidecar_id = (
        _sidecar_prefix_for_label_purpose(label_purpose)
        + str(label_binding["label_binding_sha256"])
    )
    if sidecar.manifest["sidecar_id"] != expected_sidecar_id:
        raise ValueError("sidecar identity does not bind the canonical launch/task label")
    if sidecar.manifest["executor_evidence_index_complete"] is not True:
        raise ValueError("sidecar has no complete task-bound executor evidence index")
    if sidecar.manifest["counts"]["unverified"] != 0:
        raise ValueError("search evidence receipt requires every branch to complete")
    tasks = _ordered_tasks(specs, protocol, parent)
    if len(tasks) != sidecar.branch_count:
        raise ValueError("receipt task list does not cover every sidecar branch")
    if authorization is not None and len(tasks) != authorization.expected_branch_count:
        raise ValueError("formal production task count disagrees with its authorization")
    entries: list[dict[str, object]] = []
    executor_sources: Mapping[str, object] | None = None
    for row, task in enumerate(tasks):
        path = v5_exact_search_artifact_path(
            evidence_root, task, seed_schedule, optimizer_schedule
        )
        artifact = read_v5_exact_search_executor_artifact(path, task=task)
        sources = artifact.manifest.get("source_sha256")
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError("executor artifact has no concrete source hashes")
        if executor_sources is None:
            executor_sources = dict(sources)
        elif dict(sources) != dict(executor_sources):
            raise ValueError("executor artifacts were produced by different sources")
        source_bundle_sha = _source_bundle_sha256(sources)
        executor_relative_path = _safe_relative_path(
            path,
            evidence_root,
            expected_kind="executor evidence",
        )
        _compare_executor_to_sidecar(
            sidecar=sidecar,
            row=row,
            task=task,
            artifact=artifact,
            executor_relative_path=executor_relative_path,
            executor_source_bundle_sha256=source_bundle_sha,
        )
        task_payload = artifact.manifest["task"]
        entries.append(
            {
                "branch_row": row,
                "query_index": task.query_index,
                "global_branch_key": task.branch.global_key.wire_key,
                "exact_curve_sha256": task.exact_curve_sha256,
                "relative_path": _safe_relative_path(
                    path,
                    output.parent,
                    expected_kind="executor evidence",
                ),
                "artifact_id": artifact.manifest["artifact_id"],
                "artifact_sha256": artifact.receipt.artifact_sha256,
                "artifact_schema": artifact.manifest["artifact_schema"],
                "artifact_version": artifact.manifest["artifact_version"],
                "task_audit_sha256": task.audit_sha256,
                "executor_task_payload_sha256": _task_sha256(task_payload),
                "outcome": artifact.manifest["outcome"],
                "exact_forward_calls_used": artifact.manifest["ledger"][
                    "exact_forward_calls_used"
                ],
            }
        )
    assert executor_sources is not None
    counts = dict(sidecar.manifest["counts"])
    formal_tasks = all(task.full_training_label_permitted for task in tasks)
    if formal_tasks is not formal_protocol:
        raise ValueError("receipt protocol tier disagrees with its replay tasks")
    if sidecar.manifest["full_training_labels_permitted"] is not formal_tasks:
        raise ValueError("sidecar training-label claim disagrees with its replay tasks")
    eligible = (
        label_purpose == V5_SEARCH_LABEL_PURPOSE_TRAINING and formal_tasks
    )
    if label_purpose == V5_SEARCH_LABEL_PURPOSE_TRAINING and not formal_tasks:
        raise ValueError(
            "full-training evidence requires a paper/full calibrated search protocol"
        )
    if eligible and str(sidecar.manifest["sidecar_id"]).startswith(
        V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX
    ):
        raise ValueError("a throughput-pilot sidecar cannot receive a training receipt")
    exact_calls_used = int(
        np.sum(sidecar.arrays[branch_array("exact_forward_calls_used")])
    )
    if authorization is not None and exact_calls_used != (
        authorization.expected_exact_forward_calls
    ):
        raise ValueError("formal production exact-forward budget is incomplete")
    core = {
        "schema": V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA,
        "version": V5_SEARCH_EVIDENCE_RECEIPT_VERSION,
        "receipt_id": f"search-evidence-{sidecar_receipt.artifact_sha256}",
        "label_purpose": label_purpose,
        "full_training_eligible": eligible,
        "audit_policy": V5_SEARCH_EVIDENCE_AUDIT_POLICY,
        "launch_source_bundle_sha256": launch_sha,
        "launch_plan_sha256": launch_plan_sha,
        "shard_plan_sha256": shard_plan_sha,
        "label_binding": label_binding,
        "label_binding_sha256": label_binding["label_binding_sha256"],
        "formal_production_authorization": (
            None if authorization is None else authorization.to_payload()
        ),
        "formal_production_authorization_sha256": (
            None if authorization is None else authorization.sha256
        ),
        "executor_root_relative_path": _safe_relative_path(
            evidence_root,
            output.parent,
            expected_kind="executor evidence root",
        ),
        "parent": {
            "dataset_id": parent.manifest["dataset_id"],
            "artifact_sha256": parent_receipt.artifact_sha256,
            "manifest_sha256": parent_receipt.manifest_sha256,
        },
        "sidecar": {
            "sidecar_id": sidecar.manifest["sidecar_id"],
            "artifact_sha256": sidecar_receipt.artifact_sha256,
            "manifest_sha256": sidecar_receipt.manifest_sha256,
            "protocol_sha256": sidecar.manifest["protocol_sha256"],
            "split_id": sidecar.manifest["split_id"],
        },
        "executor_source_sha256": dict(executor_sources),
        "executor_source_bundle_sha256": _source_bundle_sha256(executor_sources),
        "branch_evidence": entries,
        "counts": {
            "branches": len(entries),
            "queries": sidecar.query_count,
            "exact_forward_calls_used": exact_calls_used,
        },
    }
    manifest = {
        **core,
        "receipt_sha256": sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }
    observed_source = pre_publish_guard()
    if authorization is not None and observed_source != (
        authorization.launch_source_bundle_sha256
    ):
        raise RuntimeError(
            "formal source changed before evidence-receipt publication"
        )
    _write_json_exclusive(output, manifest)
    return read_v5_search_evidence_receipt(
        output,
        parent_dataset_path=parent_path,
        sidecar_path=selected_sidecar_path,
        require_training_eligible=eligible,
    )


def _load_receipt(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("search evidence receipt must be a regular file")
    raw = path.read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("search evidence receipt is unexpectedly large")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("search evidence receipt is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise ValueError("search evidence receipt fields are incomplete")
    core = dict(value)
    supplied = _digest(core.pop("receipt_sha256"), "receipt_sha256")
    if supplied != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("search evidence receipt SHA-256 does not reproduce")
    return value


def read_v5_search_evidence_receipt(
    path: str | os.PathLike[str],
    *,
    parent_dataset_path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
    require_training_eligible: bool = True,
    expected_consumer_role: str | None = None,
) -> V5SearchEvidenceReceipt:
    """Check receipt bindings and every referenced evidence file without loading arrays."""

    selected = Path(path).resolve(strict=True)
    manifest = _load_receipt(selected)
    if (
        manifest["schema"] != V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA
        or manifest["version"] != V5_SEARCH_EVIDENCE_RECEIPT_VERSION
        or manifest["audit_policy"] != V5_SEARCH_EVIDENCE_AUDIT_POLICY
        or manifest["label_purpose"] not in V5_SEARCH_LABEL_PURPOSES
        or manifest["full_training_eligible"]
        is not (manifest["label_purpose"] == V5_SEARCH_LABEL_PURPOSE_TRAINING)
    ):
        raise ValueError("search evidence receipt identity or purpose is incompatible")
    if require_training_eligible and not manifest["full_training_eligible"]:
        raise ValueError("non-training search evidence cannot enter full training")
    raw_authorization = manifest["formal_production_authorization"]
    authorization = (
        None
        if raw_authorization is None
        else V5FormalProductionSearchAuthorization.from_payload(raw_authorization)
    )
    if manifest["label_purpose"] == V5_SEARCH_LABEL_PURPOSE_TRAINING:
        if authorization is None or manifest[
            "formal_production_authorization_sha256"
        ] != authorization.sha256:
            raise ValueError("TRAINING receipt has no valid formal production authorization")
    elif authorization is not None or manifest[
        "formal_production_authorization_sha256"
    ] is not None:
        raise ValueError("pilot/smoke receipt cannot carry a formal authorization")
    if expected_consumer_role is not None:
        if expected_consumer_role not in V5_FORMAL_PRODUCTION_CONSUMER_ROLES:
            raise ValueError("expected_consumer_role is unsupported")
        if authorization is None or authorization.consumer_role != expected_consumer_role:
            raise ValueError("search evidence is not authorized for this consumer role")
    _digest(manifest["launch_source_bundle_sha256"], "launch source bundle")
    _digest(manifest["launch_plan_sha256"], "launch plan SHA-256")
    _digest(manifest["shard_plan_sha256"], "shard plan SHA-256")
    source_hashes = manifest["executor_source_sha256"]
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("search evidence receipt source hashes are missing")
    for name, digest in source_hashes.items():
        _text(name, "executor source filename")
        _digest(digest, f"executor source SHA-256[{name}]")
    if manifest["executor_source_bundle_sha256"] != _source_bundle_sha256(
        source_hashes
    ):
        raise ValueError("executor source bundle SHA-256 does not reproduce")
    executor_root_relative = PurePosixPath(
        _text(
            manifest["executor_root_relative_path"],
            "executor_root_relative_path",
        )
    )
    if executor_root_relative.is_absolute() or ".." in executor_root_relative.parts:
        raise ValueError("search evidence receipt contains an unsafe executor root")
    executor_root = (
        selected.parent / Path(*executor_root_relative.parts)
    ).resolve(strict=True)
    if selected.parent.resolve() not in executor_root.parents or not executor_root.is_dir():
        raise ValueError("search evidence executor root escaped the receipt directory")
    parent, parent_receipt = read_v5_grouped_dataset(parent_dataset_path)
    sidecar, sidecar_receipt = read_v5_search_supervision_sidecar(sidecar_path)
    parent_binding = sidecar.manifest["parent_grouped_artifact"]
    if (
        parent_binding.get("dataset_id") != parent.manifest["dataset_id"]
        or parent_binding.get("artifact_sha256")
        != parent_receipt.artifact_sha256
        or parent_binding.get("manifest_sha256")
        != parent_receipt.manifest_sha256
    ):
        raise ValueError("receipt sidecar does not bind the supplied grouped parent")
    if manifest["parent"] != {
        "dataset_id": parent.manifest["dataset_id"],
        "artifact_sha256": parent_receipt.artifact_sha256,
        "manifest_sha256": parent_receipt.manifest_sha256,
    }:
        raise ValueError("search evidence receipt binds a different parent")
    if manifest["sidecar"] != {
        "sidecar_id": sidecar.manifest["sidecar_id"],
        "artifact_sha256": sidecar_receipt.artifact_sha256,
        "manifest_sha256": sidecar_receipt.manifest_sha256,
        "protocol_sha256": sidecar.manifest["protocol_sha256"],
        "split_id": sidecar.manifest["split_id"],
    }:
        raise ValueError("search evidence receipt binds a different sidecar")
    if sidecar.manifest["executor_evidence_index_complete"] is not True:
        raise ValueError("receipt sidecar lost its complete executor evidence index")
    protocol = V5FrozenExactSearchProtocol(**dict(sidecar.manifest["protocol"]))
    _validate_label_purpose_for_protocol(manifest["label_purpose"], protocol)
    supplied_binding = manifest["label_binding"]
    if not isinstance(supplied_binding, Mapping):
        raise ValueError("search evidence label binding is missing")
    expected_binding = build_v5_search_label_binding(
        protocol=protocol,
        seed_schedule_sha256=supplied_binding.get("seed_schedule_sha256"),
        optimizer_schedule_sha256=supplied_binding.get(
            "optimizer_schedule_sha256"
        ),
        launch_source_bundle_sha256=manifest["launch_source_bundle_sha256"],
        launch_plan_sha256=manifest["launch_plan_sha256"],
        shard_plan_sha256=manifest["shard_plan_sha256"],
        label_purpose=manifest["label_purpose"],
        formal_production_authorization=authorization,
    )
    if (
        dict(supplied_binding) != expected_binding
        or manifest["label_binding_sha256"]
        != expected_binding["label_binding_sha256"]
        or sidecar.manifest["sidecar_id"]
        != _sidecar_prefix_for_label_purpose(manifest["label_purpose"])
        + str(expected_binding["label_binding_sha256"])
    ):
        raise ValueError("search evidence label binding does not reproduce")
    if manifest["full_training_eligible"] and (
        protocol.protocol_tier != V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
    ):
        raise ValueError(
            "search evidence training eligibility disagrees with the calibrated protocol"
        )
    if manifest["full_training_eligible"] and str(
        sidecar.manifest["sidecar_id"]
    ).startswith(V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX):
        raise ValueError("a throughput-pilot sidecar cannot enter full training")
    if authorization is not None:
        observed_indices = tuple(
            int(value) for value in parent.arrays[clean_array("sobol_index")]
        )
        observed_groups = tuple(
            str(value) for value in parent.arrays[clean_array("clean_group_id")]
        )
        observed_splits = {
            str(value) for value in parent.arrays[clean_array("split_id")]
        }
        if (
            authorization.launch_source_bundle_sha256
            != manifest["launch_source_bundle_sha256"]
            or authorization.launch_plan_sha256 != manifest["launch_plan_sha256"]
            or authorization.shard_plan_sha256 != manifest["shard_plan_sha256"]
            or authorization.protocol_sha256 != sidecar.manifest["protocol_sha256"]
            or authorization.target_split != sidecar.manifest["split_id"]
            or authorization.recipe_sobol_indices != observed_indices
            or authorization.clean_group_ids != observed_groups
            or observed_splits != {authorization.target_split}
            or authorization.expected_query_count != sidecar.query_count
            or authorization.expected_branch_count != sidecar.branch_count
        ):
            raise ValueError("formal authorization no longer matches its checked artifacts")
    entries = manifest["branch_evidence"]
    if not isinstance(entries, list) or len(entries) != sidecar.branch_count:
        raise ValueError("search evidence receipt does not cover every branch")
    for row, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
            raise ValueError("search evidence receipt branch entry is incomplete")
        relative = PurePosixPath(_text(entry["relative_path"], "relative_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("search evidence receipt contains an unsafe relative path")
        evidence = (selected.parent / Path(*relative.parts)).resolve(strict=True)
        if selected.parent.resolve() not in evidence.parents or not evidence.is_file():
            raise ValueError("search evidence escaped the receipt directory")
        sidecar_relative = PurePosixPath(
            _text(
                sidecar.arrays[
                    branch_array("executor_artifact_relative_path")
                ][row],
                "sidecar executor_artifact_relative_path",
            )
        )
        if sidecar_relative.is_absolute() or ".." in sidecar_relative.parts:
            raise ValueError("sidecar executor evidence path is unsafe")
        indexed_evidence = (
            executor_root / Path(*sidecar_relative.parts)
        ).resolve(strict=True)
        if indexed_evidence != evidence:
            raise ValueError("receipt evidence path disagrees with the sidecar index")
        metadata = _executor_manifest_metadata(evidence)
        query_index = int(sidecar.arrays[branch_array("query_index")][row])
        sidecar_outcome = next(
            name
            for name, code in SEARCH_OUTCOME_CODE.items()
            if code
            == int(
                sidecar.arrays["branch__label__search_outcome_code"][row]
            )
        )
        sidecar_completed = bool(
            sidecar.arrays[branch_array("runner_completed")][row]
        )
        sidecar_termination = str(
            sidecar.arrays[branch_array("runner_termination_reason")][row]
        )
        sidecar_representatives = json.loads(
            str(
                sidecar.arrays[
                    branch_array("compatible_representatives_json")
                ][row]
            )
        )
        task_payload = metadata.get("task")
        if not isinstance(task_payload, Mapping):
            raise ValueError("executor evidence task payload is missing")
        expected = {
            "branch_row": row,
            "query_index": query_index,
            "global_branch_key": str(
                sidecar.arrays[branch_array("global_branch_key")][row]
            ),
            "exact_curve_sha256": str(
                sidecar.arrays[query_array("exact_curve_sha256")][
                    query_index
                ]
            ),
            "artifact_id": str(
                sidecar.arrays[branch_array("executor_artifact_id")][row]
            ),
            "artifact_sha256": str(
                sidecar.arrays[branch_array("executor_artifact_sha256")][row]
            ),
            "artifact_schema": V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
            "artifact_version": V5_EXACT_SEARCH_EXECUTOR_VERSION,
            "task_audit_sha256": str(
                sidecar.arrays[branch_array("task_audit_sha256")][row]
            ),
            "executor_task_payload_sha256": _task_sha256(
                task_payload
            ),
            "outcome": sidecar_outcome,
            "exact_forward_calls_used": int(
                sidecar.arrays[branch_array("exact_forward_calls_used")][row]
            ),
        }
        for name, value in expected.items():
            if entry.get(name) != value:
                raise ValueError(f"search evidence receipt {name} disagrees with sidecar")
        _digest(entry["task_audit_sha256"], "executor task audit SHA-256")
        _digest(
            entry["executor_task_payload_sha256"],
            "executor task payload SHA-256",
        )
        if (
            metadata.get("artifact_id") != entry["artifact_id"]
            or metadata.get("artifact_schema") != entry["artifact_schema"]
            or metadata.get("artifact_version") != entry["artifact_version"]
            or metadata.get("ledger", {}).get("exact_forward_calls_used")
            != entry["exact_forward_calls_used"]
            or metadata.get("outcome") != sidecar_outcome
            or metadata.get("completed") is not sidecar_completed
            or metadata.get("termination_reason") != sidecar_termination
            or metadata.get("representatives") != sidecar_representatives
            or metadata.get("source_sha256") != source_hashes
            or metadata.get("protocol_sha256") != sidecar.manifest["protocol_sha256"]
            or task_payload.get("query_index") != query_index
            or task_payload.get("exact_curve_sha256")
            != entry["exact_curve_sha256"]
            or task_payload.get("universal_query_sha256")
            != sidecar.arrays[
                query_array("universal_query_sha256")
            ][query_index]
            or task_payload.get("global_branch_key")
            != entry["global_branch_key"]
            or task_payload.get("context_sha256")
            != sidecar.arrays[branch_array("context_sha256")][row]
            or task_payload.get("task_audit_sha256")
            != entry["task_audit_sha256"]
            or metadata.get("selected_metric_name")
            != sidecar.arrays[query_array("selected_metric_name")][query_index]
            or metadata.get("selected_threshold_name")
            != sidecar.arrays[
                query_array("selected_threshold_name")
            ][query_index]
            or metadata.get("selected_threshold_source_id")
            != sidecar.arrays[
                query_array("selected_threshold_source_id")
            ][query_index]
            or metadata.get("selected_threshold_value")
            != float(
                sidecar.arrays[
                    query_array("selected_threshold_value")
                ][query_index]
            )
            or sidecar.arrays[
                branch_array("executor_artifact_schema")
            ][row]
            != entry["artifact_schema"]
            or sidecar.arrays[
                branch_array("executor_artifact_version")
            ][row]
            != entry["artifact_version"]
            or sidecar.arrays[
                branch_array("executor_source_bundle_sha256")
            ][row]
            != manifest["executor_source_bundle_sha256"]
        ):
            raise ValueError("executor manifest metadata disagrees with its receipt")
        if _file_sha256(evidence) != entry["artifact_sha256"]:
            raise ValueError("executor evidence file changed after task-bound audit")
    counts = manifest["counts"]
    if counts != {
        "branches": sidecar.branch_count,
        "queries": sidecar.query_count,
        "exact_forward_calls_used": int(
            np.sum(sidecar.arrays[branch_array("exact_forward_calls_used")])
        ),
    }:
        raise ValueError("search evidence receipt counts do not reproduce")
    if authorization is not None and counts["exact_forward_calls_used"] != (
        authorization.expected_exact_forward_calls
    ):
        raise ValueError("formal authorization exact-forward budget does not reproduce")
    if sidecar.manifest["counts"]["unverified"] != 0:
        raise ValueError("training receipt no longer has complete branch evidence")
    return V5SearchEvidenceReceipt(
        path=selected,
        manifest=manifest,
        file_sha256=_file_sha256(selected),
    )


__all__ = [
    "V5SearchEvidenceReceipt",
    "V5_SEARCH_EVIDENCE_AUDIT_POLICY",
    "V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA",
    "V5_SEARCH_EVIDENCE_RECEIPT_VERSION",
    "V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE",
    "V5_SEARCH_LABEL_PURPOSE_PILOT",
    "V5_SEARCH_LABEL_PURPOSE_TRAINING",
    "V5_SEARCH_PIPELINE_TRAINING_SIDECAR_PREFIX",
    "V5_SEARCH_TRAINING_RECEIPT_PROMOTION_ENABLED",
    "audit_and_write_v5_search_evidence_receipt",
    "build_v5_search_label_binding",
    "evidence_receipt_path_for_sidecar",
    "read_v5_search_evidence_receipt",
]
