"""Lossless JSON codec for one typed paper representative.

The ordinary paper audit intentionally records only the exact-intensity
digest.  Durable replay artifacts use this codec while the typed parameter is
still live, so no later consumer has to reconstruct scientific arrays from an
audit summary.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import numpy as np

from .contract import LatentComponentParameters
from .evaluation import CandidateInput, LinearSolutionSnapshot, ReferenceMode
from .grouped_artifact_v5 import array_sha256
from .k1_phase_c_contract_v5 import digest
from .paper_representative_payload_v5 import (
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
    V5PaperParameterRepresentativePayload,
)
from .profiled_forward import ResolutionShape


V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_representative_payload_raw/v1"
)
V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_VERSION = "lossless_typed_phase_c_raw_json_v1"


def _object(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    return dict(value)


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _linear_payload(value: LinearSolutionSnapshot) -> dict[str, object]:
    return {
        "background": value.background,
        "particle_amplitudes": list(value.particle_amplitudes),
        "resolution_amplitude": value.resolution_amplitude,
        "k": value.k,
    }


def _common_parameter_payload(value: CandidateInput | ReferenceMode) -> dict[str, object]:
    return {
        "topology_id": value.topology_id,
        "components": [asdict(component) for component in value.components],
        "resolution": None if value.resolution is None else asdict(value.resolution),
        "linear_solution": _linear_payload(value.linear_solution),
    }


def encode_v5_lossless_representative_payload(
    payload: V5PaperParameterRepresentativePayload,
) -> dict[str, object]:
    """Encode one exact typed payload, including an emitted intensity vector."""

    if type(payload) is not V5PaperParameterRepresentativePayload:
        raise TypeError("payload must be an exact V5PaperParameterRepresentativePayload")
    parameter = payload.parameter
    common = _common_parameter_payload(parameter)
    if isinstance(parameter, CandidateInput):
        intensity = np.ascontiguousarray(parameter.exact_intensity)
        if intensity.dtype.str != "<f8":
            raise ValueError("emitted exact intensity must already use little-endian float64")
        if intensity.ndim != 1 or intensity.size < 1:
            raise ValueError("emitted exact intensity must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0.0):
            raise ValueError("emitted exact intensity must be finite and strictly positive")
        encoded_parameter = {
            **common,
            "candidate_id": parameter.candidate_id,
            "proposal_rank": parameter.proposal_rank,
            "exact_intensity": intensity.tolist(),
            "exact_intensity_dtype": "<f8",
            "exact_intensity_shape": list(intensity.shape),
            "exact_intensity_order": "C",
            "exact_intensity_sha256": array_sha256("candidate_exact_intensity", intensity),
            "bounds_pass": parameter.bounds_pass,
            "physics_pass": parameter.physics_pass,
            "proposal_score_raw": parameter.proposal_score_raw,
        }
    elif isinstance(parameter, ReferenceMode):
        encoded_parameter = {**common, "reference_id": parameter.reference_id}
    else:  # pragma: no cover - guarded by the typed envelope
        raise TypeError("representative parameter has an unsupported concrete type")
    return {
        "schema": V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_SCHEMA,
        "version": V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_VERSION,
        "representative_id": payload.representative_id,
        "role": payload.role,
        "global_branch_key": payload.global_branch_key,
        "query_context_sha256": payload.query_context_sha256,
        "parameter": encoded_parameter,
    }


def _linear(value: object) -> LinearSolutionSnapshot:
    row = _object(
        value,
        {"background", "particle_amplitudes", "resolution_amplitude", "k"},
        "linear_solution",
    )
    return LinearSolutionSnapshot(
        background=row["background"],
        particle_amplitudes=tuple(_sequence(row["particle_amplitudes"], "particle_amplitudes")),
        resolution_amplitude=row["resolution_amplitude"],
        k=row["k"],
    )


def _components(value: object) -> tuple[LatentComponentParameters, ...]:
    fields = {
        name
        for name, definition in LatentComponentParameters.__dataclass_fields__.items()
        if definition.init
    }
    return tuple(
        LatentComponentParameters(**_object(item, fields, f"components[{index}]"))
        for index, item in enumerate(_sequence(value, "components"))
    )


def _resolution(value: object) -> ResolutionShape | None:
    if value is None:
        return None
    return ResolutionShape(**_object(value, {"sigma_res", "nu_res"}, "resolution"))


def _candidate_parameter(value: object) -> CandidateInput:
    common = {"topology_id", "components", "resolution", "linear_solution"}
    fields = {
        *common,
        "candidate_id",
        "proposal_rank",
        "exact_intensity",
        "exact_intensity_dtype",
        "exact_intensity_shape",
        "exact_intensity_order",
        "exact_intensity_sha256",
        "bounds_pass",
        "physics_pass",
        "proposal_score_raw",
    }
    row = _object(value, fields, "candidate parameter")
    shape = row["exact_intensity_shape"]
    if (
        row["exact_intensity_dtype"] != "<f8"
        or row["exact_intensity_order"] != "C"
        or not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 1
    ):
        raise ValueError("candidate exact intensity must be non-empty 1-D <f8 C-order")
    exact_values = _sequence(row["exact_intensity"], "exact_intensity")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in exact_values):
        raise ValueError("candidate exact intensity requires JSON numeric scalars")
    intensity = np.ascontiguousarray(np.asarray(exact_values, dtype=np.dtype("<f8")))
    if list(intensity.shape) != shape:
        raise ValueError("candidate exact-intensity shape does not reproduce")
    if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0.0):
        raise ValueError("candidate exact intensity must be finite and strictly positive")
    if array_sha256("candidate_exact_intensity", intensity) != digest(
        row["exact_intensity_sha256"], "exact_intensity_sha256"
    ):
        raise ValueError("candidate exact-intensity SHA-256 does not reproduce")
    return CandidateInput(
        candidate_id=row["candidate_id"],
        proposal_rank=row["proposal_rank"],
        topology_id=row["topology_id"],
        components=_components(row["components"]),
        resolution=_resolution(row["resolution"]),
        linear_solution=_linear(row["linear_solution"]),
        exact_intensity=intensity,
        bounds_pass=row["bounds_pass"],
        physics_pass=row["physics_pass"],
        proposal_score_raw=row["proposal_score_raw"],
    )


def decode_v5_lossless_representative_payload(
    value: object,
    *,
    source_artifact_sha256: str,
) -> V5PaperParameterRepresentativePayload:
    """Decode a hash-checked raw object into its exact typed representative."""

    row = _object(
        value,
        {
            "schema",
            "version",
            "representative_id",
            "role",
            "global_branch_key",
            "query_context_sha256",
            "parameter",
        },
        "lossless representative artifact",
    )
    if (row["schema"], row["version"]) != (
        V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_SCHEMA,
        V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_VERSION,
    ):
        raise ValueError("unsupported lossless representative schema/version")
    common = {"topology_id", "components", "resolution", "linear_solution"}
    if row["role"] == V5_EMITTED_REPRESENTATIVE_ROLE:
        parameter: CandidateInput | ReferenceMode = _candidate_parameter(row["parameter"])
    elif row["role"] == V5_REFERENCE_REPRESENTATIVE_ROLE:
        parameter_row = _object(row["parameter"], {"reference_id", *common}, "reference parameter")
        parameter = ReferenceMode(
            reference_id=parameter_row["reference_id"],
            topology_id=parameter_row["topology_id"],
            components=_components(parameter_row["components"]),
            resolution=_resolution(parameter_row["resolution"]),
            linear_solution=_linear(parameter_row["linear_solution"]),
        )
    else:
        raise ValueError("representative role is unsupported")
    return V5PaperParameterRepresentativePayload(
        representative_id=row["representative_id"],
        role=row["role"],
        parameter=parameter,
        global_branch_key=row["global_branch_key"],
        query_context_sha256=row["query_context_sha256"],
        source_artifact_sha256=digest(source_artifact_sha256, "source_artifact_sha256"),
    )


__all__ = [
    "V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_SCHEMA",
    "V5_LOSSLESS_REPRESENTATIVE_ARTIFACT_VERSION",
    "decode_v5_lossless_representative_payload",
    "encode_v5_lossless_representative_payload",
]
