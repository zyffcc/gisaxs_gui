"""Truth-independent samplers for the V5 GUI-bounds query contract."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np

from .bounds_query_v5 import (
    AXIS_RANGE_PLACEMENTS,
    AXIS_RANGE_REGIMES,
    V5_LOCAL_TARGET_OPEN_EPSILON,
    AxisRangeDesign,
    V5BoundsQuery,
    V5SolutionTarget,
    full_range_axis_designs,
)
from .branch_codec import ResolutionBounds
from .canonical_component_slots import canonicalize_component_slots
from .contextual_branch_catalog import (
    PRESENCE_POLICIES,
    PresencePolicy,
    build_contextual_branch_catalog,
)
from .contract import (
    CYLINDER,
    D_DOMAIN,
    D_WIDTH_FRACTION_DOMAIN,
    H_DOMAIN,
    R_DOMAIN,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SIZE_WIDTH_FRACTION_DOMAIN,
    TOPOLOGIES,
    VERTICAL_CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    full_component_bounds,
    latent_component_to_gui,
)


_SIGMA_R_ABSOLUTE_DOMAIN = ClosedInterval(
    R_DOMAIN.low * SIZE_WIDTH_FRACTION_DOMAIN.low,
    R_DOMAIN.high * SIZE_WIDTH_FRACTION_DOMAIN.high,
)
_SIGMA_H_ABSOLUTE_DOMAIN = ClosedInterval(
    H_DOMAIN.low * SIZE_WIDTH_FRACTION_DOMAIN.low,
    H_DOMAIN.high * SIZE_WIDTH_FRACTION_DOMAIN.high,
)
_SIGMA_D_ABSOLUTE_DOMAIN = ClosedInterval(
    D_DOMAIN.low * D_WIDTH_FRACTION_DOMAIN.low,
    D_DOMAIN.high * D_WIDTH_FRACTION_DOMAIN.high,
)

V5_SOLUTION_TARGET_SAMPLER_VERSION = (
    "posterior_v8_complete_slot_contract_open_uniform_local_target_sampler_v2"
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


def _axis_domain(shape: str, axis: str) -> tuple[ClosedInterval, bool]:
    if axis == "R":
        return R_DOMAIN, True
    if axis == "sigma_R":
        return (
            SIZE_WIDTH_FRACTION_DOMAIN if shape == VERTICAL_CYLINDER else _SIGMA_R_ABSOLUTE_DOMAIN
        ), True
    if axis == "h":
        return H_DOMAIN, True
    if axis == "sigma_h":
        return _SIGMA_H_ABSOLUTE_DOMAIN, True
    if axis == "D":
        return D_DOMAIN, True
    if axis == "sigma_D":
        return _SIGMA_D_ABSOLUTE_DOMAIN, True
    if axis == "sigma_res":
        return RESOLUTION_SIGMA_DOMAIN, True
    if axis == "nu_res":
        return RESOLUTION_NU_DOMAIN, False
    raise ValueError(f"unknown V5 GUI axis {axis!r}")


def _sample_axis_interval(
    rng: np.random.Generator,
    *,
    shape: str,
    axis: str,
    axis_key: str,
    ordinal: int,
) -> tuple[ClosedInterval, AxisRangeDesign]:
    domain, log_space = _axis_domain(shape, axis)
    regime_offset = int(rng.integers(0, len(AXIS_RANGE_REGIMES)))
    placement_offset = int(rng.integers(0, len(AXIS_RANGE_PLACEMENTS)))
    regime = AXIS_RANGE_REGIMES[(regime_offset + ordinal) % len(AXIS_RANGE_REGIMES)]
    placement = AXIS_RANGE_PLACEMENTS[(placement_offset + 2 * ordinal) % len(AXIS_RANGE_PLACEMENTS)]
    design = AxisRangeDesign(axis_key, regime, placement)
    low = float(np.log(domain.low)) if log_space else domain.low
    high = float(np.log(domain.high)) if log_space else domain.high
    span = high - low
    if regime == "full":
        return domain, design
    if regime == "fixed":
        if placement == "edge_low":
            position = 0.0
        elif placement == "edge_high":
            position = 1.0
        elif placement == "asymmetric_low":
            position = float(rng.uniform(0.02, 0.20))
        elif placement == "asymmetric_high":
            position = float(rng.uniform(0.80, 0.98))
        else:
            position = float(rng.uniform(0.15, 0.85))
        start = stop = low + position * span
    else:
        width = (
            float(rng.uniform(0.35, 0.80)) if regime == "wide" else float(rng.uniform(0.03, 0.20))
        )
        available = 1.0 - width
        if placement == "edge_low":
            start_fraction = 0.0
        elif placement == "edge_high":
            start_fraction = available
        elif placement == "asymmetric_low":
            start_fraction = float(rng.uniform(0.0, 0.20 * available))
        elif placement == "asymmetric_high":
            start_fraction = float(rng.uniform(0.80 * available, available))
        else:
            start_fraction = float(rng.uniform(0.0, available))
        start = low + start_fraction * span
        stop = start + width * span
    if log_space:
        start, stop = float(np.exp(start)), float(np.exp(stop))
    start = min(max(start, domain.low), domain.high)
    stop = min(max(stop, domain.low), domain.high)
    return ClosedInterval(start, stop), design


def _component_bounds(
    rng: np.random.Generator,
    *,
    shape: str,
    slot: int,
    d_policy: PresencePolicy,
    ordinal: int,
) -> tuple[GuiComponentBounds, tuple[AxisRangeDesign, ...], int]:
    values: dict[str, object] = {"shape": shape}
    designs: list[AxisRangeDesign] = []
    axes = ["R", "sigma_R"]
    if shape == CYLINDER:
        axes.extend(("h", "sigma_h"))
    if d_policy != "absent":
        axes.extend(("D", "sigma_D"))
    for axis in axes:
        interval, design = _sample_axis_interval(
            rng,
            shape=shape,
            axis=axis,
            axis_key=f"component[{slot}].{axis}",
            ordinal=ordinal,
        )
        values[axis] = interval
        designs.append(design)
        ordinal += 1
    values["allow_D_absent"] = d_policy == "optional"
    return GuiComponentBounds(**values), tuple(designs), ordinal


def sample_v5_bounds_query(
    topology: Sequence[str],
    *,
    query_seed: int,
    d_policies: Sequence[PresencePolicy] | None = None,
    resolution_presence_policy: PresencePolicy = "optional",
    max_attempts: int = 1024,
) -> V5BoundsQuery:
    """Sample heterogeneous per-axis ranges without reading a truth value."""

    shapes = tuple(topology)
    if shapes not in TOPOLOGIES:
        raise ValueError("topology must be one canonical Posterior V8 topology")
    seed = _non_negative_integer(query_seed, "query_seed")
    attempts = _non_negative_integer(max_attempts, "max_attempts")
    if attempts < 1:
        raise ValueError("max_attempts must be positive")
    if d_policies is None:
        policies = ("optional",) * len(shapes)
    else:
        policies = tuple(
            _presence_policy(value, f"d_policies[{index}]")
            for index, value in enumerate(d_policies)
        )
        if len(policies) != len(shapes):
            raise ValueError("d_policies length must match topology")
    resolution_policy = _presence_policy(resolution_presence_policy, "resolution_presence_policy")
    last_error: Exception | None = None
    for attempt in range(attempts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt, 0x5635424F]))
        try:
            bounds = []
            designs: list[AxisRangeDesign] = []
            ordinal = 0
            for slot, (shape, d_policy) in enumerate(zip(shapes, policies)):
                item, item_designs, ordinal = _component_bounds(
                    rng,
                    shape=shape,
                    slot=slot,
                    d_policy=d_policy,
                    ordinal=ordinal,
                )
                bounds.append(item)
                designs.extend(item_designs)
            resolution_bounds = None
            if resolution_policy != "absent":
                sigma_res, sigma_design = _sample_axis_interval(
                    rng,
                    shape=shapes[0],
                    axis="sigma_res",
                    axis_key="resolution.sigma_res",
                    ordinal=ordinal,
                )
                ordinal += 1
                nu_res, nu_design = _sample_axis_interval(
                    rng,
                    shape=shapes[0],
                    axis="nu_res",
                    axis_key="resolution.nu_res",
                    ordinal=ordinal,
                )
                designs.extend((sigma_design, nu_design))
                resolution_bounds = ResolutionBounds(sigma_res, nu_res)
            return V5BoundsQuery.create(
                query_seed=seed,
                generation_attempt=attempt,
                component_bounds=tuple(bounds),
                resolution_presence_policy=resolution_policy,
                resolution_bounds=resolution_bounds,
                axis_designs=tuple(designs),
            )
        except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(
        f"could not sample a feasible V5 bounds query after {attempts} attempts"
    ) from last_error


def sample_v5_solution_target(
    query: V5BoundsQuery,
    *,
    target_seed: int,
    pattern_id: int | None = None,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
) -> V5SolutionTarget:
    """Select a feasible branch and draw truth after the complete query freezes.

    When component intensity ranges are present, both branch and continuous
    target canonicalization use the complete slot contract.  Omitting them is
    conservative and preserves every labelled geometry slot.
    """

    if not isinstance(query, V5BoundsQuery):
        raise TypeError("query must be a V5BoundsQuery")
    seed = _non_negative_integer(target_seed, "target_seed")
    catalog = build_contextual_branch_catalog(
        query.component_bounds,
        component_intensity_bounds=component_intensity_bounds,
        resolution_presence_policy=query.resolution_presence_policy,
    )
    feasible_patterns = tuple(
        value
        for value in catalog.wire_pattern_ids
        if value in query.feasible_wire_pattern_ids
    )
    if not feasible_patterns:  # pragma: no cover - query/catalog validation guards this
        raise RuntimeError("complete V5 slot contract contains no feasible branch")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x56355447]))
    sampled_pattern = feasible_patterns[int(rng.integers(0, len(feasible_patterns)))]
    if pattern_id is None:
        selected = sampled_pattern
    else:
        selected = int(pattern_id)
        if selected not in feasible_patterns:
            raise ValueError("pattern_id is not feasible and canonical in the complete slot query")
    codec = query.codec_for(selected)
    active_values = rng.uniform(
        V5_LOCAL_TARGET_OPEN_EPSILON,
        1.0 - V5_LOCAL_TARGET_OPEN_EPSILON,
        len(codec.active_indices),
    )
    latent, _, resolution = codec.decode_active(active_values)
    canonical = canonicalize_component_slots(
        codec,
        latent,
        resolution,
        component_intensity_bounds=component_intensity_bounds,
    )
    components = codec.latent_components_to_gui(canonical.components)
    return V5SolutionTarget(
        query=query,
        pattern_id=selected,
        target_seed=seed,
        local_target_unit=canonical.coordinates.unit_cube,
        truth_components=components,
        truth_resolution=canonical.resolution,
        physical_numeric_policy_version=query.numeric_policy_version,
    )


def full_range_v5_query(
    topology: Sequence[str],
    *,
    query_seed: int = 0,
) -> V5BoundsQuery:
    """Convenience query with optional D/Resolution over the complete domain."""

    shapes = tuple(topology)
    if shapes not in TOPOLOGIES:
        raise ValueError("topology must be one canonical Posterior V8 topology")
    bounds = tuple(full_component_bounds(shape, d_policy="optional") for shape in shapes)
    resolution = ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)
    return V5BoundsQuery.create(
        query_seed=query_seed,
        generation_attempt=0,
        component_bounds=bounds,
        resolution_presence_policy="optional",
        resolution_bounds=resolution,
        axis_designs=full_range_axis_designs(bounds, resolution),
    )


__all__ = [
    "V5_SOLUTION_TARGET_SAMPLER_VERSION",
    "full_range_v5_query",
    "sample_v5_bounds_query",
    "sample_v5_solution_target",
]
