"""Opaque checked-receipt capability shared by the runner and gate assessor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import re

from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_records_v5 import V5K1PhaseCParentRecord
from .launch_k1_phase_a_dag_v5 import (
    V5_K1_PHASE_A_LAUNCH_SCHEMA,
    V5_K1_PHASE_A_LAUNCH_VERSION,
)
from .model_v5_contract import MODEL_V5_SCHEMA, MODEL_V5_VERSION, model_v5_contract_payload
from .proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY_SCHEMA,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    V5_PROPOSAL_EXECUTION_POLICY_VERSION,
)
from .sobol_cross_platform_contract_v5 import (
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
)
from .universal_inference_contract_v5 import (
    V5_UNIVERSAL_INFERENCE_SCHEMA,
    V5_UNIVERSAL_INFERENCE_VERSION,
)


_REPLAY_RECEIPT_SEAL = object()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
V5_K1_PHASE_C_MODEL_CONTRACT_SHA256 = sha256(
    canonical_json(model_v5_contract_payload()).encode("utf-8")
).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCArtifactBinding:
    """Immutable source, cross-platform gate, launch, and selected-model identity."""

    source_archive_sha256: str
    source_manifest_sha256: str
    source_tree_sha256: str
    source_bundle_sha256: str
    cross_platform_gate_claim_sha256: str
    phase_a_launch_receipt_sha256: str
    model_artifact_sha256: str
    model_weights_sha256: str
    model_training_result_sha256: str
    proposal_execution_policy_sha256: str = V5_PROPOSAL_EXECUTION_POLICY_SHA256
    model_contract_sha256: str = V5_K1_PHASE_C_MODEL_CONTRACT_SHA256
    model_schema: str = MODEL_V5_SCHEMA
    model_version: str = MODEL_V5_VERSION
    cross_platform_manifest_schema: str = V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA
    cross_platform_manifest_version: str = V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION
    phase_a_launch_schema: str = V5_K1_PHASE_A_LAUNCH_SCHEMA
    phase_a_launch_version: str = V5_K1_PHASE_A_LAUNCH_VERSION
    proposal_execution_policy_schema: str = V5_PROPOSAL_EXECUTION_POLICY_SCHEMA
    proposal_execution_policy_version: str = V5_PROPOSAL_EXECUTION_POLICY_VERSION
    universal_inference_schema: str = V5_UNIVERSAL_INFERENCE_SCHEMA
    universal_inference_version: str = V5_UNIVERSAL_INFERENCE_VERSION

    def __post_init__(self) -> None:
        digests = (
            "source_archive_sha256",
            "source_manifest_sha256",
            "source_tree_sha256",
            "source_bundle_sha256",
            "cross_platform_gate_claim_sha256",
            "phase_a_launch_receipt_sha256",
            "model_artifact_sha256",
            "model_weights_sha256",
            "model_training_result_sha256",
            "proposal_execution_policy_sha256",
            "model_contract_sha256",
        )
        for name in digests:
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        actual = (
            self.model_contract_sha256,
            self.model_schema,
            self.model_version,
            self.cross_platform_manifest_schema,
            self.cross_platform_manifest_version,
            self.phase_a_launch_schema,
            self.phase_a_launch_version,
            self.proposal_execution_policy_sha256,
            self.proposal_execution_policy_schema,
            self.proposal_execution_policy_version,
            self.universal_inference_schema,
            self.universal_inference_version,
        )
        expected = (
            V5_K1_PHASE_C_MODEL_CONTRACT_SHA256,
            MODEL_V5_SCHEMA,
            MODEL_V5_VERSION,
            V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
            V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
            V5_K1_PHASE_A_LAUNCH_SCHEMA,
            V5_K1_PHASE_A_LAUNCH_VERSION,
            V5_PROPOSAL_EXECUTION_POLICY_SHA256,
            V5_PROPOSAL_EXECUTION_POLICY_SCHEMA,
            V5_PROPOSAL_EXECUTION_POLICY_VERSION,
            V5_UNIVERSAL_INFERENCE_SCHEMA,
            V5_UNIVERSAL_INFERENCE_VERSION,
        )
        if actual != expected:
            raise ValueError("K1 Phase-C artifact binding uses a stale source/model contract")

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCCheckedReplayReceipt:
    """A receipt that has been byte-checked and replayed in this process."""

    plan_sha256: str
    contract_sha256: str
    evidence_bundle_sha256: str
    receipt_sha256: str
    formal: bool
    records: tuple[V5K1PhaseCParentRecord, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "plan_sha256",
            "contract_sha256",
            "evidence_bundle_sha256",
            "receipt_sha256",
        ):
            _digest(getattr(self, name), name)
        if type(self.formal) is not bool:
            raise TypeError("formal must be an explicit bool")
        if not self.records or not all(
            isinstance(value, V5K1PhaseCParentRecord) for value in self.records
        ):
            raise ValueError("checked receipt requires typed parent records")


def _mint_v5_k1_phase_c_checked_receipt(
    *,
    plan_sha256: str,
    contract_sha256: str,
    evidence_bundle_sha256: str,
    receipt_sha256: str,
    formal: bool,
    records: tuple[V5K1PhaseCParentRecord, ...],
) -> V5K1PhaseCCheckedReplayReceipt:
    return V5K1PhaseCCheckedReplayReceipt(
        plan_sha256=plan_sha256,
        contract_sha256=contract_sha256,
        evidence_bundle_sha256=evidence_bundle_sha256,
        receipt_sha256=receipt_sha256,
        formal=formal,
        records=records,
        _seal=_REPLAY_RECEIPT_SEAL,
    )


def validate_v5_k1_phase_c_checked_receipt(
    value: object,
) -> V5K1PhaseCCheckedReplayReceipt:
    if type(value) is not V5K1PhaseCCheckedReplayReceipt or value._seal is not _REPLAY_RECEIPT_SEAL:
        raise TypeError("formal K1 Phase-C assessment requires a runner-checked replay receipt")
    return value


__all__ = [
    "V5K1PhaseCArtifactBinding",
    "V5K1PhaseCCheckedReplayReceipt",
    "validate_v5_k1_phase_c_checked_receipt",
]
