"""Strict parsing and comparison for V5.2 cross-platform Sobol manifests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping

from .branch_catalog import BRANCH_PATTERN_COUNT
from .contract import NUM_TOPOLOGIES
from .sobol_cross_platform_contract_v5 import (
    DESIGN_FIELDS,
    GENERATING_ENTRY_FIELDS,
    LAYER_FIELDS,
    PROBE_ENTRY_FIELDS,
    SCOPE_FIELDS,
    SOURCE_FIELDS,
    TOPOLOGY_ENTRY_FIELDS,
    TOP_LEVEL_FIELDS,
    VALIDATION_CONTRACT,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA,
    V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION,
    V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT,
    V5_SOBOL_CROSS_PLATFORM_PREFIX_START,
    V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP,
    V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES,
    V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS,
    canonical_json,
    digest,
    exact_integer,
    exact_keys,
    layer_hashes,
    neutral_clean_group_id,
    pretty_json,
    scope_payload,
    sha256_json,
    v5_sobol_runtime_free_semantic_design,
)


def _strict_json_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate cross-platform manifest field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cross-platform artifact is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("cross-platform artifact must contain one JSON object")
    return payload


def _validate_generating_entries(entries: object) -> list[Mapping[str, object]]:
    if not isinstance(entries, list) or len(entries) != V5_SOBOL_CROSS_PLATFORM_PREFIX_COUNT:
        raise ValueError("manifest must contain exactly 1024 generating entries")
    validated = []
    expected = range(
        V5_SOBOL_CROSS_PLATFORM_PREFIX_START,
        V5_SOBOL_CROSS_PLATFORM_PREFIX_STOP,
    )
    hash_fields = GENERATING_ENTRY_FIELDS - {
        "sobol_index",
        "generating_topology_id",
        "branch_pattern_id",
    }
    for expected_index, raw in zip(expected, entries):
        entry = exact_keys(raw, GENERATING_ENTRY_FIELDS, "generating entry")
        if exact_integer(entry["sobol_index"], "sobol_index") != expected_index:
            raise ValueError("generating entries must be ordered Sobol indices 0..1023")
        for name in hash_fields:
            digest(entry[name], f"generating_entries[{expected_index}].{name}")
        topology_id = exact_integer(entry["generating_topology_id"], "topology_id")
        if topology_id not in V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS:
            raise ValueError("generating topology_id is outside the frozen catalog")
        pattern_id = exact_integer(entry["branch_pattern_id"], "branch_pattern_id")
        if not 0 <= pattern_id < BRANCH_PATTERN_COUNT:
            raise ValueError("branch_pattern_id is outside the frozen wire catalog")
        validated.append(entry)
    return validated


def _validate_probe_entries(entries: object) -> list[Mapping[str, object]]:
    if not isinstance(entries, list) or len(entries) != len(
        V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES
    ):
        raise ValueError("manifest must contain exactly ten topology probe entries")
    validated = []
    for expected_index, raw in zip(V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES, entries):
        entry = exact_keys(raw, PROBE_ENTRY_FIELDS, "topology probe entry")
        if exact_integer(entry["sobol_index"], "probe sobol_index") != expected_index:
            raise ValueError("topology probe entries are not in the frozen index order")
        digest(entry["unit_coordinates_sha256"], "probe unit_coordinates_sha256")
        digest(
            entry["normalized_query_design_sha256"],
            "probe normalized_query_design_sha256",
        )
        digest(entry["topology_queries_sha256"], "probe topology_queries_sha256")
        generating = exact_integer(
            entry["generating_topology_id"], "probe generating_topology_id"
        )
        if generating not in V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS:
            raise ValueError("probe generating topology is outside the frozen catalog")
        queries = entry["topology_queries"]
        if not isinstance(queries, list) or len(queries) != NUM_TOPOLOGIES:
            raise ValueError("every topology probe must contain all 34 topology queries")
        for expected_topology, raw_query in zip(
            V5_SOBOL_CROSS_PLATFORM_TOPOLOGY_IDS, queries
        ):
            query = exact_keys(raw_query, TOPOLOGY_ENTRY_FIELDS, "topology query")
            if exact_integer(query["topology_id"], "topology_id") != expected_topology:
                raise ValueError("topology queries must be ordered by all 34 topology IDs")
            for name in TOPOLOGY_ENTRY_FIELDS - {"topology_id"}:
                digest(query[name], f"topology_queries[{expected_topology}].{name}")
        if sha256_json(queries) != entry["topology_queries_sha256"]:
            raise ValueError("per-probe topology query SHA-256 does not reproduce")
        validated.append(entry)
    return validated


def _validate_manifest_core(core: Mapping[str, object]) -> None:
    exact_keys(core, TOP_LEVEL_FIELDS, "cross-platform manifest")
    if core["schema"] != V5_SOBOL_CROSS_PLATFORM_MANIFEST_SCHEMA:
        raise ValueError("unsupported cross-platform manifest schema")
    if core["version"] != V5_SOBOL_CROSS_PLATFORM_MANIFEST_VERSION:
        raise ValueError("unsupported cross-platform manifest version")
    source = exact_keys(core["source"], SOURCE_FIELDS, "manifest source")
    for name in sorted(SOURCE_FIELDS):
        digest(source[name], name)
    design = exact_keys(core["design"], DESIGN_FIELDS, "manifest design")
    if dict(design) != v5_sobol_runtime_free_semantic_design():
        raise ValueError("runtime-free Sobol semantic design is incompatible")
    scope = exact_keys(core["scope"], SCOPE_FIELDS, "manifest scope")
    if dict(scope) != scope_payload():
        raise ValueError("cross-platform manifest scope is incompatible")
    if core["validation_contract"] != VALIDATION_CONTRACT:
        raise ValueError("cross-platform validation contract is incompatible")

    generating = _validate_generating_entries(core["generating_entries"])
    semantic_digest = design["semantic_sha256"]
    for index, entry in enumerate(generating):
        if entry["clean_group_id"] != neutral_clean_group_id(semantic_digest, index):
            raise ValueError("neutral clean-group identity does not reproduce")
    probes = _validate_probe_entries(core["topology_probe_entries"])
    for probe in probes:
        generating_entry = generating[int(probe["sobol_index"])]
        if (
            probe["unit_coordinates_sha256"]
            != generating_entry["unit_coordinates_sha256"]
            or probe["generating_topology_id"]
            != generating_entry["generating_topology_id"]
        ):
            raise ValueError("probe identity disagrees with its generating prefix entry")
        generating_query = probe["topology_queries"][
            int(probe["generating_topology_id"])
        ]
        if (
            generating_query["geometry_query_sha256"]
            != generating_entry["geometry_query_sha256"]
            or generating_query["amplitude_query_sha256"]
            != generating_entry["amplitude_query_sha256"]
        ):
            raise ValueError("probe generating query disagrees with its clean recipe")
    layers = exact_keys(core["layer_sha256"], LAYER_FIELDS, "manifest layers")
    for name in sorted(LAYER_FIELDS):
        digest(layers[name], name)
    expected_layers = layer_hashes(
        design=design,
        scope=scope,
        generating_entries=generating,
        probe_entries=probes,
    )
    if dict(layers) != expected_layers:
        raise ValueError("cross-platform manifest layer SHA-256 values do not reproduce")


@dataclass(frozen=True)
class V5SobolCrossPlatformManifest:
    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        payload = _strict_json_object(self.canonical_json)
        _validate_manifest_core(payload)
        canonical = canonical_json(payload)
        if self.canonical_json != canonical:
            raise ValueError("manifest core JSON is not canonical")
        if digest(self.sha256, "manifest_sha256") != sha256(
            canonical.encode("utf-8")
        ).hexdigest():
            raise ValueError("cross-platform manifest SHA-256 does not reproduce")

    @property
    def payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)

    @property
    def layer_sha256(self) -> dict[str, str]:
        return dict(self.payload["layer_sha256"])

    def to_json(self) -> str:
        payload = self.payload
        payload["manifest_sha256"] = self.sha256
        return pretty_json(payload)

    @classmethod
    def from_json(cls, encoded: str) -> "V5SobolCrossPlatformManifest":
        payload = _strict_json_object(encoded)
        if set(payload) != TOP_LEVEL_FIELDS | {"manifest_sha256"}:
            raise ValueError("cross-platform manifest fields are incomplete or unsupported")
        supplied = digest(payload.pop("manifest_sha256"), "manifest_sha256")
        _validate_manifest_core(payload)
        result = cls(canonical_json(payload), supplied)
        if encoded != result.to_json():
            raise ValueError("cross-platform manifest JSON is not canonical")
        return result


@dataclass(frozen=True)
class V5SobolCrossPlatformComparison:
    passed: bool
    first_mismatch_path: str | None
    reference_manifest_sha256: str
    candidate_manifest_sha256: str
    reference_scientific_content_sha256: str
    candidate_scientific_content_sha256: str
    reference_source_identity: tuple[str, str, str]
    candidate_source_identity: tuple[str, str, str]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        if (self.first_mismatch_path is None) != self.passed:
            raise ValueError("first_mismatch_path must be absent exactly when comparison passes")
        for name in (
            "reference_manifest_sha256",
            "candidate_manifest_sha256",
            "reference_scientific_content_sha256",
            "candidate_scientific_content_sha256",
        ):
            digest(getattr(self, name), name)
        for prefix, identity in (
            ("reference", self.reference_source_identity),
            ("candidate", self.candidate_source_identity),
        ):
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise TypeError(f"{prefix}_source_identity must be a three-item tuple")
            for field, value in zip(sorted(SOURCE_FIELDS), identity):
                digest(value, f"{prefix}_source_identity.{field}")

    @staticmethod
    def _source_payload(identity: tuple[str, str, str]) -> dict[str, str]:
        return dict(zip(sorted(SOURCE_FIELDS), identity))

    @property
    def reference_source_payload(self) -> dict[str, str]:
        return self._source_payload(self.reference_source_identity)

    @property
    def candidate_source_payload(self) -> dict[str, str]:
        return self._source_payload(self.candidate_source_identity)

    def audit_payload(self) -> dict[str, object]:
        core = {
            "schema": "gisaxs.posterior_v8.sobol_cross_platform_gate_result/v1",
            "version": "posterior_v8_byte_exact_manifest_compare_fail_closed_v1",
            "passed": self.passed,
            "first_mismatch_path": self.first_mismatch_path,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "reference_scientific_content_sha256": (
                self.reference_scientific_content_sha256
            ),
            "candidate_scientific_content_sha256": (
                self.candidate_scientific_content_sha256
            ),
            "reference_source": self.reference_source_payload,
            "candidate_source": self.candidate_source_payload,
            "comparison_semantics": "exact_normalized_manifest_content",
        }
        return {**core, "result_sha256": sha256_json(core)}

    def to_json(self) -> str:
        return pretty_json(self.audit_payload())

    @classmethod
    def from_json(cls, encoded: str) -> "V5SobolCrossPlatformComparison":
        payload = _strict_json_object(encoded)
        expected_fields = {
            "schema",
            "version",
            "passed",
            "first_mismatch_path",
            "reference_manifest_sha256",
            "candidate_manifest_sha256",
            "reference_scientific_content_sha256",
            "candidate_scientific_content_sha256",
            "reference_source",
            "candidate_source",
            "comparison_semantics",
            "result_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("cross-platform gate-result fields are incomplete or unsupported")
        core = dict(payload)
        supplied = digest(core.pop("result_sha256"), "result_sha256")
        if supplied != sha256_json(core):
            raise ValueError("cross-platform gate-result SHA-256 does not reproduce")
        for name in ("reference_source", "candidate_source"):
            exact_keys(payload[name], SOURCE_FIELDS, name)
        mismatch = payload["first_mismatch_path"]
        if mismatch is not None and (not isinstance(mismatch, str) or not mismatch):
            raise ValueError("first_mismatch_path must be null or non-empty")
        result = cls(
            passed=payload["passed"],
            first_mismatch_path=mismatch,
            reference_manifest_sha256=payload["reference_manifest_sha256"],
            candidate_manifest_sha256=payload["candidate_manifest_sha256"],
            reference_scientific_content_sha256=payload[
                "reference_scientific_content_sha256"
            ],
            candidate_scientific_content_sha256=payload[
                "candidate_scientific_content_sha256"
            ],
            reference_source_identity=tuple(
                payload["reference_source"][name] for name in sorted(SOURCE_FIELDS)
            ),
            candidate_source_identity=tuple(
                payload["candidate_source"][name] for name in sorted(SOURCE_FIELDS)
            ),
        )
        if payload != result.audit_payload() or encoded != result.to_json():
            raise ValueError("cross-platform gate-result JSON is not canonical")
        return result


def _first_mismatch(left: object, right: object, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        if set(left) != set(right):
            return path
        for key in sorted(left):
            mismatch = _first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            mismatch = _first_mismatch(left_value, right_value, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if left == right else path


def compare_v5_sobol_cross_platform_manifests(
    reference: V5SobolCrossPlatformManifest,
    candidate: V5SobolCrossPlatformManifest,
) -> V5SobolCrossPlatformComparison:
    if not isinstance(reference, V5SobolCrossPlatformManifest) or not isinstance(
        candidate, V5SobolCrossPlatformManifest
    ):
        raise TypeError("reference and candidate must be strict cross-platform manifests")
    reference_payload = reference.payload
    candidate_payload = candidate.payload
    mismatch = _first_mismatch(reference_payload, candidate_payload)
    return V5SobolCrossPlatformComparison(
        passed=mismatch is None,
        first_mismatch_path=mismatch,
        reference_manifest_sha256=reference.sha256,
        candidate_manifest_sha256=candidate.sha256,
        reference_scientific_content_sha256=reference.layer_sha256[
            "scientific_content_sha256"
        ],
        candidate_scientific_content_sha256=candidate.layer_sha256[
            "scientific_content_sha256"
        ],
        reference_source_identity=tuple(
            reference_payload["source"][name] for name in sorted(SOURCE_FIELDS)
        ),
        candidate_source_identity=tuple(
            candidate_payload["source"][name] for name in sorted(SOURCE_FIELDS)
        ),
    )


__all__ = [
    "V5SobolCrossPlatformComparison",
    "V5SobolCrossPlatformManifest",
    "compare_v5_sobol_cross_platform_manifests",
]
