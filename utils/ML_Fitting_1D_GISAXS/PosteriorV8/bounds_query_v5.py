"""Query-first heterogeneous GUI bounds for the V5 proposal system.

The query is sampled before a branch or inverse target.  D and Resolution
may therefore be optional, and repeated same-shape slots retain their own
physical ranges.  A branch-specific codec is built only after the contextual
wire catalog has been constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .bounds_first_contract import BOUNDS_EMBEDDING_VERSION, bounds_embedding
from .branch_catalog import decode_branch_pattern
from .branch_codec import (
    BRANCH_CODEC_VERSION,
    COMPONENT_STRIDE,
    RESOLUTION_OFFSET,
    ProfiledBranchCodec,
    ResolutionBounds,
    UNIT_CUBE_DIMENSIONS,
)
from .contextual_branch_catalog import (
    CONTEXTUAL_BRANCH_CATALOG_VERSION,
    PRESENCE_POLICIES,
    PresencePolicy,
    build_contextual_branch_catalog,
)
from .contract import (
    CYLINDER,
    MAX_COMPONENTS,
    TOPOLOGIES,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    latent_component_to_gui,
    topology_id_for,
)
from .profiled_forward import ResolutionShape
from .sobol_numeric_canonicalization_v5 import (
    V5_FAST_NUMERIC_POLICY_VERSION,
    v5_numeric_policy_sha256,
    validate_v5_numeric_policy,
)


V5_BOUNDS_QUERY_SCHEMA = "gisaxs.posterior_v8.bounds_query/v3"
V5_BOUNDS_QUERY_VERSION = (
    "posterior_v8_codec_and_numeric_contract_bound_query_first_gui_bounds_v3"
)
V5_BRANCH_CONDITION_VERSION = "posterior_v8_numeric_policy_bound_local_branch_condition_v2"
V5_LOCAL_TARGET_VERSION = (
    "posterior_v8_contextual_local_open_uniform_exact_policy_replay_contract_v4"
)
V5_LOCAL_TARGET_OPEN_EPSILON = 1.0e-5

AXIS_RANGE_REGIMES = ("full", "wide", "narrow", "fixed")
AXIS_RANGE_PLACEMENTS = (
    "interior",
    "asymmetric_low",
    "asymmetric_high",
    "edge_low",
    "edge_high",
)


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _presence_policy(value: object, name: str) -> PresencePolicy:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in PRESENCE_POLICIES:
        raise ValueError(f"{name} must be absent, optional, or required")
    return value


@dataclass(frozen=True)
class AxisRangeDesign:
    """Truth-independent provenance for one physical GUI interval."""

    axis_key: str
    regime: str
    placement: str

    def __post_init__(self) -> None:
        if not isinstance(self.axis_key, str) or not self.axis_key.strip():
            raise ValueError("axis_key must be a non-empty string")
        if self.regime not in AXIS_RANGE_REGIMES:
            raise ValueError(f"regime must be one of {AXIS_RANGE_REGIMES}")
        if self.placement not in AXIS_RANGE_PLACEMENTS:
            raise ValueError(f"placement must be one of {AXIS_RANGE_PLACEMENTS}")


def _available_dimension_mask(
    component_bounds: Sequence[GuiComponentBounds],
    resolution_bounds: ResolutionBounds | None,
) -> tuple[bool, ...]:
    mask = [False] * UNIT_CUBE_DIMENSIONS
    for slot, bounds in enumerate(component_bounds):
        offset = slot * COMPONENT_STRIDE
        mask[offset : offset + 2] = (True, True)
        if bounds.shape == CYLINDER:
            mask[offset + 2 : offset + 4] = (True, True)
        if bounds.D is not None:
            mask[offset + 4 : offset + 6] = (True, True)
    if resolution_bounds is not None:
        mask[RESOLUTION_OFFSET:] = (True, True)
    return tuple(mask)


def _expected_axis_keys(
    component_bounds: Sequence[GuiComponentBounds],
    resolution_bounds: ResolutionBounds | None,
) -> tuple[str, ...]:
    keys: list[str] = []
    for slot, bounds in enumerate(component_bounds):
        axes = ["R", "sigma_R"]
        if bounds.shape == CYLINDER:
            axes.extend(("h", "sigma_h"))
        if bounds.D is not None:
            axes.extend(("D", "sigma_D"))
        keys.extend(f"component[{slot}].{axis}" for axis in axes)
    if resolution_bounds is not None:
        keys.extend(("resolution.sigma_res", "resolution.nu_res"))
    return tuple(keys)


def _query_payload(
    *,
    query_seed: int,
    generation_attempt: int,
    component_bounds: Sequence[GuiComponentBounds],
    resolution_presence_policy: PresencePolicy,
    resolution_bounds: ResolutionBounds | None,
    axis_designs: Sequence[AxisRangeDesign],
    numeric_policy_version: str,
) -> dict[str, object]:
    return {
        "schema": V5_BOUNDS_QUERY_SCHEMA,
        "version": V5_BOUNDS_QUERY_VERSION,
        "bounds_embedding_version": BOUNDS_EMBEDDING_VERSION,
        "branch_codec_version": BRANCH_CODEC_VERSION,
        "contextual_branch_catalog_version": CONTEXTUAL_BRANCH_CATALOG_VERSION,
        "numeric_policy_version": numeric_policy_version,
        "numeric_policy_sha256": v5_numeric_policy_sha256(numeric_policy_version),
        "query_seed": int(query_seed),
        "generation_attempt": int(generation_attempt),
        "topology": [value.shape for value in component_bounds],
        "component_bounds": [asdict(value) for value in component_bounds],
        "resolution_presence_policy": resolution_presence_policy,
        "resolution_bounds": (None if resolution_bounds is None else asdict(resolution_bounds)),
        "axis_designs": [asdict(value) for value in axis_designs],
    }


@dataclass(frozen=True)
class V5BoundsQuery:
    """One immutable user-range context, created before selecting a truth."""

    query_seed: int
    generation_attempt: int
    component_bounds: tuple[GuiComponentBounds, ...]
    resolution_presence_policy: PresencePolicy
    resolution_bounds: ResolutionBounds | None
    numeric_policy_version: str
    axis_designs: tuple[AxisRangeDesign, ...]
    topology_id: int
    policy_wire_pattern_ids: tuple[int, ...]
    feasible_wire_pattern_ids: tuple[int, ...]
    bounds_embedding: tuple[float, ...]
    available_dimension_mask: tuple[bool, ...]
    canonical_json: str
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        query_seed: int,
        generation_attempt: int,
        component_bounds: Sequence[GuiComponentBounds],
        resolution_presence_policy: PresencePolicy,
        resolution_bounds: ResolutionBounds | None,
        axis_designs: Sequence[AxisRangeDesign],
        numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
    ) -> "V5BoundsQuery":
        seed = _non_negative_integer(query_seed, "query_seed")
        attempt = _non_negative_integer(generation_attempt, "generation_attempt")
        bounds = tuple(component_bounds)
        if not 1 <= len(bounds) <= MAX_COMPONENTS or not all(
            isinstance(value, GuiComponentBounds) for value in bounds
        ):
            raise ValueError("component_bounds must contain one to four GUI bounds")
        topology = tuple(value.shape for value in bounds)
        if topology not in TOPOLOGIES:
            raise ValueError("component bounds must use canonical topology order")
        policy = _presence_policy(resolution_presence_policy, "resolution_presence_policy")
        numeric_policy = validate_v5_numeric_policy(numeric_policy_version)
        if policy == "absent" and resolution_bounds is not None:
            raise ValueError("Resolution-absent query must not define bounds")
        if policy != "absent" and not isinstance(resolution_bounds, ResolutionBounds):
            raise TypeError("optional/required Resolution query needs bounds")
        designs = tuple(axis_designs)
        if not all(isinstance(value, AxisRangeDesign) for value in designs):
            raise TypeError("axis_designs must contain AxisRangeDesign values")
        expected_keys = _expected_axis_keys(bounds, resolution_bounds)
        if tuple(value.axis_key for value in designs) != expected_keys:
            raise ValueError("axis_designs must cover every present GUI axis in order")

        catalog = build_contextual_branch_catalog(bounds, resolution_presence_policy=policy)
        feasible = []
        for pattern_id in catalog.wire_pattern_ids:
            d_flags, resolution_present = decode_branch_pattern(pattern_id)
            try:
                ProfiledBranchCodec.build(
                    topology,
                    bounds,
                    d_flags[: len(bounds)],
                    resolution_bounds=(resolution_bounds if resolution_present else None),
                    numeric_policy_version=numeric_policy,
                )
            except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError):
                continue
            feasible.append(pattern_id)
        if not feasible:
            raise ValueError("user bounds contain no continuously feasible hard branch")
        payload = _query_payload(
            query_seed=seed,
            generation_attempt=attempt,
            component_bounds=bounds,
            resolution_presence_policy=policy,
            resolution_bounds=resolution_bounds,
            axis_designs=designs,
            numeric_policy_version=numeric_policy,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(
            seed,
            attempt,
            bounds,
            policy,
            resolution_bounds,
            numeric_policy,
            designs,
            topology_id_for(topology),
            catalog.wire_pattern_ids,
            tuple(feasible),
            bounds_embedding(
                bounds,
                resolution_bounds,
                numeric_policy_version=numeric_policy,
            ),
            _available_dimension_mask(bounds, resolution_bounds),
            canonical,
            sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        seed = _non_negative_integer(self.query_seed, "query_seed")
        attempt = _non_negative_integer(self.generation_attempt, "generation_attempt")
        bounds = tuple(self.component_bounds)
        topology = tuple(value.shape for value in bounds)
        if topology not in TOPOLOGIES:
            raise ValueError("component bounds must use canonical topology order")
        policy = _presence_policy(self.resolution_presence_policy, "resolution_presence_policy")
        numeric_policy = validate_v5_numeric_policy(self.numeric_policy_version)
        if policy == "absent" and self.resolution_bounds is not None:
            raise ValueError("Resolution-absent query must not define bounds")
        if policy != "absent" and not isinstance(self.resolution_bounds, ResolutionBounds):
            raise TypeError("optional/required Resolution query needs bounds")
        expected_keys = _expected_axis_keys(bounds, self.resolution_bounds)
        if tuple(value.axis_key for value in self.axis_designs) != expected_keys:
            raise ValueError("axis_designs must cover every present GUI axis in order")
        catalog = build_contextual_branch_catalog(bounds, resolution_presence_policy=policy)
        feasible = []
        for pattern_id in catalog.wire_pattern_ids:
            d_flags, resolution_present = decode_branch_pattern(pattern_id)
            try:
                ProfiledBranchCodec.build(
                    topology,
                    bounds,
                    d_flags[: len(bounds)],
                    resolution_bounds=(self.resolution_bounds if resolution_present else None),
                    numeric_policy_version=numeric_policy,
                )
            except (
                FloatingPointError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                continue
            feasible.append(pattern_id)
        payload = _query_payload(
            query_seed=seed,
            generation_attempt=attempt,
            component_bounds=bounds,
            resolution_presence_policy=policy,
            resolution_bounds=self.resolution_bounds,
            axis_designs=self.axis_designs,
            numeric_policy_version=numeric_policy,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = (
            topology_id_for(topology),
            numeric_policy,
            catalog.wire_pattern_ids,
            tuple(feasible),
            bounds_embedding(
                bounds,
                self.resolution_bounds,
                numeric_policy_version=numeric_policy,
            ),
            _available_dimension_mask(bounds, self.resolution_bounds),
            canonical,
            sha256(canonical.encode("utf-8")).hexdigest(),
        )
        actual = (
            self.topology_id,
            self.numeric_policy_version,
            self.policy_wire_pattern_ids,
            self.feasible_wire_pattern_ids,
            self.bounds_embedding,
            self.available_dimension_mask,
            self.canonical_json,
            self.sha256,
        )
        if not feasible or actual != expected:
            raise ValueError("V5 bounds query does not reproduce its derived contract")

    @property
    def topology(self) -> tuple[str, ...]:
        return tuple(value.shape for value in self.component_bounds)

    def codec_for(self, pattern_id: int) -> ProfiledBranchCodec:
        if pattern_id not in self.feasible_wire_pattern_ids:
            raise ValueError("pattern_id is not feasible in this V5 user query")
        d_flags, resolution_present = decode_branch_pattern(pattern_id)
        return ProfiledBranchCodec.build(
            self.topology,
            self.component_bounds,
            d_flags[: len(self.component_bounds)],
            resolution_bounds=(self.resolution_bounds if resolution_present else None),
            numeric_policy_version=self.numeric_policy_version,
        )


@dataclass(frozen=True)
class V5BranchCondition:
    """Model-ready context for one feasible branch in a V5 query."""

    query_sha256: str
    numeric_policy_version: str
    topology_id: int
    pattern_id: int
    d_present: tuple[bool, ...]
    resolution_present: bool
    bounds_embedding: tuple[float, ...]
    available_dimension_mask: tuple[bool, ...]
    active_dimension_mask: tuple[bool, ...]
    varying_dimension_mask: tuple[bool, ...]
    version: str = V5_BRANCH_CONDITION_VERSION


def branch_condition(query: V5BoundsQuery, pattern_id: int) -> V5BranchCondition:
    if not isinstance(query, V5BoundsQuery):
        raise TypeError("query must be a V5BoundsQuery")
    codec = query.codec_for(pattern_id)
    d_flags, resolution_present = decode_branch_pattern(pattern_id)
    active = codec.active_mask
    available = query.available_dimension_mask
    varying = codec.varying_mask
    if any(a and not p for a, p in zip(active, available)):
        raise RuntimeError("branch active mask escaped available GUI axes")
    if any(v and not a for v, a in zip(varying, active)):
        raise RuntimeError("branch varying mask escaped active axes")
    return V5BranchCondition(
        query_sha256=query.sha256,
        numeric_policy_version=query.numeric_policy_version,
        topology_id=query.topology_id,
        pattern_id=pattern_id,
        d_present=d_flags,
        resolution_present=resolution_present,
        bounds_embedding=query.bounds_embedding,
        available_dimension_mask=available,
        active_dimension_mask=active,
        varying_dimension_mask=varying,
    )


@dataclass(frozen=True)
class V5SolutionTarget:
    """One branch-conditional local target sampled after its V5 query."""

    query: V5BoundsQuery
    pattern_id: int
    target_seed: int
    local_target_unit: tuple[float, ...]
    truth_components: tuple[GuiComponentParameters, ...]
    truth_resolution: ResolutionShape | None
    physical_numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION
    version: str = V5_LOCAL_TARGET_VERSION

    def __post_init__(self) -> None:
        if self.version != V5_LOCAL_TARGET_VERSION:
            raise ValueError("unsupported V5 local solution-target version")
        if not isinstance(self.query, V5BoundsQuery):
            raise TypeError("query must be a V5BoundsQuery")
        numeric_policy = validate_v5_numeric_policy(self.physical_numeric_policy_version)
        if numeric_policy != self.query.numeric_policy_version:
            raise ValueError("V5 target numeric policy does not match its query")
        codec = self.query.codec_for(self.pattern_id)
        latent = tuple(codec.decode(self.local_target_unit)[0])
        gui = codec.latent_components_to_gui(latent)
        _, decoded_resolution = codec.decode(self.local_target_unit)
        if gui != self.truth_components or decoded_resolution != self.truth_resolution:
            raise ValueError("V5 solution target does not decode to its physical truth")
        _non_negative_integer(self.target_seed, "target_seed")


def full_range_axis_designs(
    component_bounds: Sequence[GuiComponentBounds],
    resolution_bounds: ResolutionBounds | None,
) -> tuple[AxisRangeDesign, ...]:
    """Return explicit provenance for externally supplied full-domain bounds."""

    return tuple(
        AxisRangeDesign(axis_key, "full", "interior")
        for axis_key in _expected_axis_keys(component_bounds, resolution_bounds)
    )


def bounds_query_from_json(encoded: str, expected_sha256: str) -> V5BoundsQuery:
    """Strictly reconstruct one persisted geometry query and all derived fields."""

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate bounds-query field {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("bounds query is not strict JSON") from exc
    expected_fields = {
        "schema",
        "version",
        "bounds_embedding_version",
        "branch_codec_version",
        "contextual_branch_catalog_version",
        "numeric_policy_version",
        "numeric_policy_sha256",
        "query_seed",
        "generation_attempt",
        "topology",
        "component_bounds",
        "resolution_presence_policy",
        "resolution_bounds",
        "axis_designs",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("bounds-query fields are incomplete or unsupported")

    interval_fields = {"low", "high"}

    def interval(value: object, name: str) -> ClosedInterval:
        if not isinstance(value, Mapping) or set(value) != interval_fields:
            raise ValueError(f"{name} must be one closed interval")
        return ClosedInterval(low=value["low"], high=value["high"])

    component_fields = {
        "shape",
        "R",
        "sigma_R",
        "h",
        "sigma_h",
        "D",
        "sigma_D",
        "allow_D_absent",
    }
    axis_fields = {"axis_key", "regime", "placement"}
    try:
        components = []
        if not isinstance(payload["component_bounds"], list):
            raise ValueError("component_bounds must be a list")
        for index, value in enumerate(payload["component_bounds"]):
            if not isinstance(value, Mapping) or set(value) != component_fields:
                raise ValueError("component-bound fields are incomplete or unsupported")
            optional = {
                name: (
                    None
                    if value[name] is None
                    else interval(value[name], f"component_bounds[{index}].{name}")
                )
                for name in ("h", "sigma_h", "D", "sigma_D")
            }
            components.append(
                GuiComponentBounds(
                    shape=value["shape"],
                    R=interval(value["R"], f"component_bounds[{index}].R"),
                    sigma_R=interval(value["sigma_R"], f"component_bounds[{index}].sigma_R"),
                    allow_D_absent=value["allow_D_absent"],
                    **optional,
                )
            )
        resolution_payload = payload["resolution_bounds"]
        if resolution_payload is None:
            resolution = None
        else:
            if not isinstance(resolution_payload, Mapping) or set(resolution_payload) != {
                "sigma_res",
                "nu_res",
            }:
                raise ValueError("resolution-bound fields are incomplete or unsupported")
            resolution = ResolutionBounds(
                sigma_res=interval(resolution_payload["sigma_res"], "resolution.sigma_res"),
                nu_res=interval(resolution_payload["nu_res"], "resolution.nu_res"),
            )
        if not isinstance(payload["axis_designs"], list) or any(
            not isinstance(value, Mapping) or set(value) != axis_fields
            for value in payload["axis_designs"]
        ):
            raise ValueError("axis-design fields are incomplete or unsupported")
        result = V5BoundsQuery.create(
            query_seed=payload["query_seed"],
            generation_attempt=payload["generation_attempt"],
            component_bounds=tuple(components),
            resolution_presence_policy=payload["resolution_presence_policy"],
            resolution_bounds=resolution,
            axis_designs=tuple(AxisRangeDesign(**value) for value in payload["axis_designs"]),
            numeric_policy_version=payload["numeric_policy_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("persisted bounds query is invalid") from exc
    if result.canonical_json != encoded or result.sha256 != expected_sha256:
        raise ValueError("bounds query JSON/SHA-256 does not reproduce")
    return result


__all__ = [
    "AXIS_RANGE_PLACEMENTS",
    "AXIS_RANGE_REGIMES",
    "V5_BOUNDS_QUERY_SCHEMA",
    "V5_BOUNDS_QUERY_VERSION",
    "V5_BRANCH_CONDITION_VERSION",
    "V5_LOCAL_TARGET_OPEN_EPSILON",
    "V5_LOCAL_TARGET_VERSION",
    "AxisRangeDesign",
    "V5BoundsQuery",
    "V5BranchCondition",
    "V5SolutionTarget",
    "branch_condition",
    "bounds_query_from_json",
    "full_range_axis_designs",
]
