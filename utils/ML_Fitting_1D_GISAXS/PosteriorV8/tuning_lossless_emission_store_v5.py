"""Persist typed tuning emissions before their exact arrays leave memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from .lossless_representative_artifact_v5 import (
    decode_v5_lossless_representative_payload,
    encode_v5_lossless_representative_payload,
)
from .paper_budget_evaluator_v5 import V5CandidateEmission
from .read_only_json_publication_v5 import publish_read_only_canonical_json


V5_TUNING_LOSSLESS_EMISSION_BINDING_SCHEMA = (
    "gisaxs.posterior_v8.tuning_lossless_emission_binding/v1"
)
V5_TUNING_LOSSLESS_EMISSION_BINDING_VERSION = "typed_pre_audit_exact_f8_atomic_read_only_v1"


@dataclass(frozen=True, kw_only=True)
class V5TuningLosslessEmissionArtifact:
    candidate_id: str
    output_rank: int
    relative_path: str
    file_sha256: str
    upstream_payload_sha256: str
    upstream_source_artifact_sha256: str
    restored_payload_sha256: str

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_TUNING_LOSSLESS_EMISSION_BINDING_SCHEMA,
            "version": V5_TUNING_LOSSLESS_EMISSION_BINDING_VERSION,
            "candidate_id": self.candidate_id,
            "output_rank": self.output_rank,
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "upstream_payload_sha256": self.upstream_payload_sha256,
            "upstream_source_artifact_sha256": (self.upstream_source_artifact_sha256),
            "restored_payload_sha256": self.restored_payload_sha256,
        }


def publish_v5_tuning_lossless_emissions(
    *,
    output_root: Path,
    full_epoch: int,
    query_id: str,
    emissions: tuple[V5CandidateEmission, ...],
) -> tuple[V5TuningLosslessEmissionArtifact, ...]:
    """Write every actual emitted payload and verify its exact typed round trip."""

    if not emissions:
        return ()
    if not all(type(value) is V5CandidateEmission for value in emissions):
        raise TypeError("emissions must contain exact V5CandidateEmission values")
    query_digest = sha256(query_id.encode("utf-8")).hexdigest()
    directory = output_root / "lossless-emissions" / f"epoch-{full_epoch:06d}" / query_digest
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for emission in sorted(emissions, key=lambda value: value.output_rank):
        candidate_digest = sha256(emission.candidate_id.encode("utf-8")).hexdigest()
        relative = (
            Path("lossless-emissions")
            / f"epoch-{full_epoch:06d}"
            / query_digest
            / f"rank-{emission.output_rank:04d}-{candidate_digest}.json"
        )
        raw = encode_v5_lossless_representative_payload(emission.payload)
        path = output_root / relative
        file_sha = publish_read_only_canonical_json(path, raw)
        restored = decode_v5_lossless_representative_payload(raw, source_artifact_sha256=file_sha)
        expected = replace(emission.payload, source_artifact_sha256=file_sha)
        if (
            restored.audit_payload() != expected.audit_payload()
            or restored.sha256 != expected.sha256
            or not np.array_equal(
                restored.parameter.exact_intensity,
                emission.payload.parameter.exact_intensity,
            )
        ):
            raise RuntimeError("lossless tuning emission did not exactly round-trip")
        artifacts.append(
            V5TuningLosslessEmissionArtifact(
                candidate_id=emission.candidate_id,
                output_rank=emission.output_rank,
                relative_path=relative.as_posix(),
                file_sha256=file_sha,
                upstream_payload_sha256=emission.payload.sha256,
                upstream_source_artifact_sha256=(emission.payload.source_artifact_sha256),
                restored_payload_sha256=restored.sha256,
            )
        )
    return tuple(artifacts)


__all__ = [
    "V5TuningLosslessEmissionArtifact",
    "V5_TUNING_LOSSLESS_EMISSION_BINDING_SCHEMA",
    "V5_TUNING_LOSSLESS_EMISSION_BINDING_VERSION",
    "publish_v5_tuning_lossless_emissions",
]
