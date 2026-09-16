"""Audited discrete-coordinate forcing for balanced K=1 Sobol blocks.

The continuous Sobol coordinates remain untouched.  A branch-owned block only
replaces the four categorical coordinates that select the K=1 shape, D policy,
Resolution policy, and the (now singleton) feasible branch catalog.  The
original point and the transformed point both remain hash-bound evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

from .contextual_branch_catalog import PRESENCE_POLICIES
from .contract import NUM_TOPOLOGIES
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCH_BY_ID,
    V5K1PhaseCBranch,
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    validate_v5_sobol_recipe_coordinates,
)


V5_K1_BRANCH_FORCING_SCHEMA = "gisaxs.posterior_v8.k1_branch_forced_coordinates/v1"
V5_K1_BRANCH_FORCING_VERSION = (
    "posterior_v8_v5_2_k1_shape_d_resolution_block_forcing_v1"
)
V5_K1_BRANCH_FORCED_COORDINATE_NAMES = (
    "discrete.topology",
    "geometry.slot_1.D_policy",
    "geometry.resolution_policy",
    "discrete.branch_within_feasible_catalog",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _coordinates_sha256(values: Sequence[float]) -> str:
    return sha256(_canonical_json(list(values)).encode("utf-8")).hexdigest()


def _bucket_center(index: int, count: int) -> float:
    return (float(index) + 0.5) / float(count)


def _presence_coordinate(present: bool) -> float:
    policy = "required" if present else "absent"
    return _bucket_center(PRESENCE_POLICIES.index(policy), len(PRESENCE_POLICIES))


def v5_k1_branch_forcing_contract() -> dict[str, object]:
    """Return the stable scientific contract for K1 block-owned categories."""

    core = {
        "schema": V5_K1_BRANCH_FORCING_SCHEMA,
        "version": V5_K1_BRANCH_FORCING_VERSION,
        "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "forced_coordinate_names": list(V5_K1_BRANCH_FORCED_COORDINATE_NAMES),
        "continuous_coordinate_policy": "preserve_every_non_forced_coordinate_bit_exactly",
        "categorical_coordinate_policy": "replace_with_open_bucket_center",
        "d_policy": {"absent": "absent", "present": "required"},
        "resolution_policy": {"absent": "absent", "present": "required"},
        "feasible_branch_catalog_after_forcing": "singleton_requested_wire_pattern",
        "original_and_forced_coordinate_hashes_required": True,
    }
    return {
        **core,
        "contract_sha256": sha256(_canonical_json(core).encode("utf-8")).hexdigest(),
    }


@dataclass(frozen=True)
class V5K1ForcedSobolCoordinates:
    """One replayable transform from a raw Sobol point to a labelled K1 branch."""

    branch: V5K1PhaseCBranch
    original_coordinates: tuple[float, ...]
    forced_coordinates: tuple[float, ...]
    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        canonical = K1_PHASE_C_BRANCH_BY_ID.get(self.branch.branch_id)
        if canonical is None or canonical != self.branch:
            raise ValueError("branch must be one canonical K1 branch")
        original = validate_v5_sobol_recipe_coordinates(self.original_coordinates)
        replay = _force_coordinates(original, canonical)
        if self.forced_coordinates != replay:
            raise ValueError("forced coordinates do not replay from branch and original point")
        payload = _forcing_payload(canonical, original, replay)
        encoded = _canonical_json(payload)
        digest = sha256(encoded.encode("utf-8")).hexdigest()
        if self.canonical_json != encoded or self.sha256 != digest:
            raise ValueError("K1 branch-forcing audit identity does not reproduce")

    def audit_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def _forced_values(branch: V5K1PhaseCBranch) -> dict[str, float]:
    return {
        "discrete.topology": _bucket_center(branch.topology_id, NUM_TOPOLOGIES),
        "geometry.slot_1.D_policy": _presence_coordinate(branch.d_present),
        "geometry.resolution_policy": _presence_coordinate(branch.resolution_present),
        "discrete.branch_within_feasible_catalog": 0.5,
    }


def _force_coordinates(
    original: tuple[float, ...], branch: V5K1PhaseCBranch
) -> tuple[float, ...]:
    values = list(original)
    for name, value in _forced_values(branch).items():
        values[V5_SOBOL_RECIPE_COORDINATE_INDEX[name]] = value
    return validate_v5_sobol_recipe_coordinates(values)


def _forcing_payload(
    branch: V5K1PhaseCBranch,
    original: tuple[float, ...],
    forced: tuple[float, ...],
) -> dict[str, object]:
    replacements = []
    for name in V5_K1_BRANCH_FORCED_COORDINATE_NAMES:
        index = V5_SOBOL_RECIPE_COORDINATE_INDEX[name]
        replacements.append(
            {
                "coordinate_name": name,
                "coordinate_index": index,
                "original_value": original[index],
                "forced_value": forced[index],
            }
        )
    return {
        "schema": V5_K1_BRANCH_FORCING_SCHEMA,
        "version": V5_K1_BRANCH_FORCING_VERSION,
        "contract_sha256": v5_k1_branch_forcing_contract()["contract_sha256"],
        "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "branch": branch.audit_payload(),
        "original_coordinates_sha256": _coordinates_sha256(original),
        "forced_coordinates_sha256": _coordinates_sha256(forced),
        "replacements": replacements,
    }


def force_v5_k1_branch_coordinates(
    unit_coordinates: Sequence[float],
    *,
    branch_id: str,
) -> V5K1ForcedSobolCoordinates:
    """Force only the categorical selectors for one canonical K1 branch."""

    if not isinstance(branch_id, str) or not branch_id:
        raise ValueError("branch_id must be a non-empty string")
    try:
        branch = K1_PHASE_C_BRANCH_BY_ID[branch_id]
    except KeyError as exc:
        raise ValueError("branch_id is not in the canonical K1 catalog") from exc
    original = validate_v5_sobol_recipe_coordinates(unit_coordinates)
    forced = _force_coordinates(original, branch)
    payload = _forcing_payload(branch, original, forced)
    encoded = _canonical_json(payload)
    return V5K1ForcedSobolCoordinates(
        branch=branch,
        original_coordinates=original,
        forced_coordinates=forced,
        canonical_json=encoded,
        sha256=sha256(encoded.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "V5_K1_BRANCH_FORCED_COORDINATE_NAMES",
    "V5_K1_BRANCH_FORCING_SCHEMA",
    "V5_K1_BRANCH_FORCING_VERSION",
    "V5K1ForcedSobolCoordinates",
    "force_v5_k1_branch_coordinates",
    "v5_k1_branch_forcing_contract",
]
