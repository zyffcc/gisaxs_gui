"""Typed, replayable parameter payloads for paper evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import re

from .evaluation import CandidateInput, ReferenceMode
from .grouped_artifact_v5 import array_sha256, canonical_json


V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA = (
    "gisaxs.posterior_v8.paper_parameter_representative_payload/v1"
)
V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION = (
    "single_typed_parameter_representative_canonical_digest_v1"
)
V5_REFERENCE_REPRESENTATIVE_ROLE = "frozen_reference_representative"
V5_EMITTED_REPRESENTATIVE_ROLE = "actual_user_visible_emitted_representative"
V5_PAPER_REPRESENTATIVE_ROLES = (
    V5_REFERENCE_REPRESENTATIVE_ROLE,
    V5_EMITTED_REPRESENTATIVE_ROLE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_parameter_payload(value: CandidateInput | ReferenceMode) -> dict[str, object]:
    """Serialize exactly one physical representative, never a cluster member set."""

    return {
        "topology_id": value.topology_id,
        "components": [asdict(component) for component in value.components],
        "resolution": None if value.resolution is None else asdict(value.resolution),
        "linear_solution": {
            "background": value.linear_solution.background,
            "particle_amplitudes": list(value.linear_solution.particle_amplitudes),
            "resolution_amplitude": value.linear_solution.resolution_amplitude,
            "k": value.linear_solution.k,
        },
    }


@dataclass(frozen=True, eq=False, kw_only=True)
class V5PaperParameterRepresentativePayload:
    """Auditable envelope for one reference or one actually emitted candidate."""

    representative_id: str
    role: str
    parameter: CandidateInput | ReferenceMode
    global_branch_key: str
    query_context_sha256: str
    source_artifact_sha256: str
    schema: str = V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA
    version: str = V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION

    def __post_init__(self) -> None:
        identifier = _text(self.representative_id, "representative_id")
        role = _text(self.role, "role")
        if role not in V5_PAPER_REPRESENTATIVE_ROLES:
            raise ValueError("representative payload role is unsupported")
        parameter = self.parameter
        if role == V5_REFERENCE_REPRESENTATIVE_ROLE:
            if not isinstance(parameter, ReferenceMode):
                raise TypeError("reference payload requires exactly one ReferenceMode")
            bound_id = parameter.reference_id
        else:
            if not isinstance(parameter, CandidateInput):
                raise TypeError("emitted payload requires exactly one CandidateInput")
            bound_id = parameter.candidate_id
        if bound_id != identifier:
            raise ValueError("representative payload ID disagrees with its typed parameter")
        object.__setattr__(self, "representative_id", identifier)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "global_branch_key", _text(self.global_branch_key, "global_branch_key")
        )
        for name in ("query_context_sha256", "source_artifact_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.schema != V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA or self.version != (
            V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION
        ):
            raise ValueError("representative payload schema/version is unsupported")

    @property
    def canonical_parameter_sha256(self) -> str:
        return sha256(
            canonical_json(_canonical_parameter_payload(self.parameter)).encode("utf-8")
        ).hexdigest()

    def audit_payload(self) -> dict[str, object]:
        exact_intensity_sha256 = None
        bounds_pass = None
        physics_pass = None
        if isinstance(self.parameter, CandidateInput):
            exact_intensity_sha256 = array_sha256(
                "candidate_exact_intensity", self.parameter.exact_intensity
            )
            bounds_pass = self.parameter.bounds_pass
            physics_pass = self.parameter.physics_pass
        return {
            "schema": self.schema,
            "version": self.version,
            "representative_id": self.representative_id,
            "role": self.role,
            "global_branch_key": self.global_branch_key,
            "query_context_sha256": self.query_context_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "canonical_parameter": _canonical_parameter_payload(self.parameter),
            "canonical_parameter_sha256": self.canonical_parameter_sha256,
            "candidate_exact_intensity_sha256": exact_intensity_sha256,
            "candidate_bounds_pass": bounds_pass,
            "candidate_physics_pass": physics_pass,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


__all__ = [
    "V5_EMITTED_REPRESENTATIVE_ROLE",
    "V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA",
    "V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION",
    "V5_REFERENCE_REPRESENTATIVE_ROLE",
    "V5PaperParameterRepresentativePayload",
]
