"""Pure membership replay for formal-production V5 search receipts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Mapping

import numpy as np

from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
    V5FormalProductionSearchPlan,
    authorize_v5_formal_production_search_shard,
)
from .grouped_artifact_v5 import canonical_json
from .search_evidence_receipt_v5 import (
    V5_SEARCH_EVIDENCE_AUDIT_POLICY,
    V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA,
    V5_SEARCH_EVIDENCE_RECEIPT_VERSION,
    V5_SEARCH_LABEL_BINDING_SCHEMA,
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
)
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)


V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_receipt_membership_proof/v2"
)

_RECEIPT_FIELDS = frozenset(
    {
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
)
_LABEL_BINDING_FIELDS = frozenset(
    {
        "schema",
        "launch_source_bundle_sha256",
        "launch_plan_sha256",
        "shard_plan_sha256",
        "protocol_sha256",
        "protocol_tier",
        "seed_schedule_sha256",
        "optimizer_schedule_sha256",
        "calibration_artifact_sha256",
        "formal_production_authorization_sha256",
        "authorized_consumer_role",
        "label_purpose",
        "full_training_eligible",
        "label_binding_sha256",
    }
)
_BRANCH_EVIDENCE_FIELDS = frozenset(
    {
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
)
_COMPLETED_OUTCOMES = frozenset(
    {
        "compatible_found",
        "no_compatible_found_within_frozen_search_budget",
    }
)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _source_bundle_sha256(source_hashes: Mapping[str, object]) -> str:
    checked = {}
    for name, digest in source_hashes.items():
        checked[_text(name, "executor source filename")] = _digest(
            digest, f"executor source SHA-256[{name}]"
        )
    if not checked:
        raise ValueError("executor source hashes cannot be empty")
    return sha256(canonical_json(checked).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V5FormalProductionMembershipProof:
    study_id: str
    launch_plan_sha256: str
    shard_plan_sha256: str
    stage_id: str
    split: str
    output_relative_path: str
    receipt_sha256: str
    label_binding_sha256: str
    authorization_sha256: str
    authorized_consumer_role: str
    schema: str = V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "launch_plan_sha256": self.launch_plan_sha256,
            "shard_plan_sha256": self.shard_plan_sha256,
            "stage_id": self.stage_id,
            "split": self.split,
            "output_relative_path": self.output_relative_path,
            "receipt_sha256": self.receipt_sha256,
            "label_binding_sha256": self.label_binding_sha256,
            "authorization_sha256": self.authorization_sha256,
            "authorized_consumer_role": self.authorized_consumer_role,
            "contract_membership_verified": True,
            "evidence_files_verified_here": False,
            "training_authorized_by_this_proof": False,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def _receipt_manifest(receipt: object) -> Mapping[str, object]:
    if isinstance(receipt, Mapping):
        return receipt
    manifest = getattr(receipt, "manifest", None)
    if isinstance(manifest, Mapping):
        return manifest
    raise TypeError("receipt must be a manifest mapping or expose .manifest")


def verify_v5_formal_production_receipt_membership(
    plan: V5FormalProductionSearchPlan,
    *,
    output_relative_path: str,
    receipt: object,
) -> V5FormalProductionMembershipProof:
    """Prove a TRAINING receipt's exact global, shard, schedule, and budget membership.

    The file-backed evidence reader must run first.  This verifier deliberately
    performs no I/O and its proof alone does not authorize training.
    """

    if not isinstance(plan, V5FormalProductionSearchPlan):
        raise TypeError("plan has an invalid type")
    manifest = dict(_receipt_manifest(receipt))
    if frozenset(manifest) != _RECEIPT_FIELDS:
        raise ValueError("receipt fields are incomplete or unsupported")
    core = dict(manifest)
    receipt_sha = _digest(core.pop("receipt_sha256"), "receipt_sha256")
    if receipt_sha != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("receipt SHA-256 does not reproduce")
    if (
        manifest["schema"] != V5_SEARCH_EVIDENCE_RECEIPT_SCHEMA
        or manifest["version"] != V5_SEARCH_EVIDENCE_RECEIPT_VERSION
        or manifest["audit_policy"] != V5_SEARCH_EVIDENCE_AUDIT_POLICY
    ):
        raise ValueError("receipt evidence contract is incompatible")
    if manifest["label_purpose"] != V5_SEARCH_LABEL_PURPOSE_TRAINING or (
        manifest["full_training_eligible"] is not True
    ):
        raise ValueError("formal production membership requires a TRAINING receipt")
    if manifest["launch_source_bundle_sha256"] != plan.source.bundle_sha256:
        raise ValueError("receipt source bundle is not a member of the global plan")
    if manifest["launch_plan_sha256"] != plan.sha256:
        raise ValueError("receipt launch-plan SHA-256 is not the canonical global plan")

    shard_sha = _digest(manifest["shard_plan_sha256"], "shard_plan_sha256")
    matching = tuple(value for value in plan.shards if value.sha256 == shard_sha)
    if len(matching) != 1:
        raise ValueError("receipt shard-plan SHA-256 is not a unique global-plan member")
    shard = matching[0]
    if output_relative_path != shard.output_relative_path:
        raise ValueError("receipt output location is not the planned shard location")
    stage = shard.stage
    authorization_payload = _mapping(
        manifest["formal_production_authorization"],
        "formal_production_authorization",
    )
    authorization = V5FormalProductionSearchAuthorization.from_payload(
        authorization_payload
    )
    expected_authorization = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=shard.sha256
    )
    if (
        authorization.to_payload() != expected_authorization.to_payload()
        or manifest["formal_production_authorization_sha256"]
        != expected_authorization.sha256
    ):
        raise ValueError("receipt authorization is not the plan-derived shard authorization")

    binding = dict(_mapping(manifest["label_binding"], "label_binding"))
    if frozenset(binding) != _LABEL_BINDING_FIELDS:
        raise ValueError("receipt label-binding fields are incomplete or unsupported")
    expected_binding_core = {
        "schema": V5_SEARCH_LABEL_BINDING_SCHEMA,
        "launch_source_bundle_sha256": plan.source.bundle_sha256,
        "launch_plan_sha256": plan.sha256,
        "shard_plan_sha256": shard.sha256,
        "protocol_sha256": stage.protocol.sha256,
        "protocol_tier": V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        "seed_schedule_sha256": stage.seed_schedule.sha256,
        "optimizer_schedule_sha256": stage.optimizer_schedule.sha256,
        "calibration_artifact_sha256": plan.calibration.identity.artifact_sha256,
        "formal_production_authorization_sha256": expected_authorization.sha256,
        "authorized_consumer_role": expected_authorization.consumer_role,
        "label_purpose": V5_SEARCH_LABEL_PURPOSE_TRAINING,
        "full_training_eligible": True,
    }
    expected_binding_sha = sha256(
        canonical_json(expected_binding_core).encode("utf-8")
    ).hexdigest()
    expected_binding = {
        **expected_binding_core,
        "label_binding_sha256": expected_binding_sha,
    }
    if binding != expected_binding or manifest["label_binding_sha256"] != (
        expected_binding_sha
    ):
        raise ValueError("receipt label binding is not a member of the staged shard")

    sidecar = _mapping(manifest["sidecar"], "sidecar")
    if sidecar.get("protocol_sha256") != stage.protocol.sha256 or sidecar.get(
        "split_id"
    ) != shard.target_split:
        raise ValueError("receipt sidecar escaped the staged protocol or split")
    source_hashes = _mapping(
        manifest["executor_source_sha256"], "executor_source_sha256"
    )
    if manifest["executor_source_bundle_sha256"] != _source_bundle_sha256(source_hashes):
        raise ValueError("receipt executor source bundle does not reproduce")

    counts = _mapping(manifest["counts"], "counts")
    expected_counts = {
        "branches": shard.expected_branch_count,
        "queries": shard.expected_query_count,
        "exact_forward_calls_used": shard.expected_exact_forward_calls,
    }
    if dict(counts) != expected_counts:
        raise ValueError("receipt counts disagree with the exact shard budget")
    evidence = manifest["branch_evidence"]
    if not isinstance(evidence, list) or len(evidence) != shard.expected_branch_count:
        raise ValueError("receipt branch evidence does not cover the planned shard")
    expected_query_indices = tuple(
        query_index
        for query_index, recipe in enumerate(shard.recipes)
        for _ in range(recipe.branch_count)
    )
    for row, (entry, query_index) in enumerate(zip(evidence, expected_query_indices)):
        if not isinstance(entry, Mapping) or frozenset(entry) != _BRANCH_EVIDENCE_FIELDS:
            raise ValueError("receipt branch evidence fields are incomplete")
        if (
            _nonnegative_integer(entry["branch_row"], "branch_row") != row
            or _nonnegative_integer(entry["query_index"], "query_index")
            != query_index
            or entry["outcome"] not in _COMPLETED_OUTCOMES
            or _nonnegative_integer(
                entry["exact_forward_calls_used"], "exact_forward_calls_used"
            )
            != stage.exact_forward_call_budget
        ):
            raise ValueError("receipt branch evidence escaped ordering or full budget")

    return V5FormalProductionMembershipProof(
        study_id=plan.study_id,
        launch_plan_sha256=plan.sha256,
        shard_plan_sha256=shard.sha256,
        stage_id=stage.stage_id,
        split=shard.target_split,
        output_relative_path=shard.output_relative_path,
        receipt_sha256=receipt_sha,
        label_binding_sha256=expected_binding_sha,
        authorization_sha256=expected_authorization.sha256,
        authorized_consumer_role=expected_authorization.consumer_role,
    )


__all__ = [
    "V5FormalProductionMembershipProof",
    "V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA",
    "verify_v5_formal_production_receipt_membership",
]
