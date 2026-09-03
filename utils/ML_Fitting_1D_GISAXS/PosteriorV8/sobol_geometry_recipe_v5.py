"""Direct geometry-query, branch, and local-target Sobol transforms for V5.1."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bounds_query_v5 import (
    AXIS_RANGE_PLACEMENTS,
    AXIS_RANGE_REGIMES,
    V5_LOCAL_TARGET_OPEN_EPSILON,
    AxisRangeDesign,
    V5BoundsQuery,
    V5SolutionTarget,
)
from .branch_codec import (
    COMPONENT_STRIDE,
    COMPONENT_UNIT_AXES,
    HARD_CORE_SPACING_MARGIN,
    INACTIVE_UNIT_VALUE,
    RESOLUTION_OFFSET,
    RESOLUTION_UNIT_AXES,
    UNIT_CUBE_DIMENSIONS,
    ResolutionBounds,
)
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
    latent_component_to_gui,
    topology_from_id,
)
from .sobol_recipe_coordinates_v5 import V5SobolCoordinateReader
from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    v5_numeric_ops,
)
from .profiled_forward import ResolutionShape


_D_ENABLED_R_DOMAIN = ClosedInterval(R_DOMAIN.low, 80.0)
_D_ENABLED_H_DOMAIN = ClosedInterval(H_DOMAIN.low, 350.0)
V5_DIRECT_GEOMETRY_TARGET_VERSION = (
    "posterior_v8_complete_slot_numeric_contract_direct_sobol_geometry_target_v5"
)
_DIRECT_NUMERIC = v5_numeric_ops(V5_DETERMINISTIC_NUMERIC_POLICY_VERSION)


def _choose(unit: float, values: tuple[str, ...]) -> str:
    return values[min(int(unit * len(values)), len(values) - 1)]


def _interpolate(low: float, high: float, unit: float) -> float:
    if low == high:
        return float(low)
    return float(low + unit * (high - low))


def _axis_interval(
    reader: V5SobolCoordinateReader,
    prefix: str,
    domain: ClosedInterval,
    *,
    log_space: bool,
) -> tuple[ClosedInterval, str, str]:
    regime = _choose(reader.take(f"{prefix}.regime"), AXIS_RANGE_REGIMES)
    placement = _choose(reader.take(f"{prefix}.placement"), AXIS_RANGE_PLACEMENTS)
    if regime == "full":
        return domain, regime, placement
    low = _DIRECT_NUMERIC.log(domain.low) if log_space else domain.low
    high = _DIRECT_NUMERIC.log(domain.high) if log_space else domain.high
    if regime == "fixed":
        if placement == "edge_low":
            position = 0.0
        elif placement == "edge_high":
            position = 1.0
        else:
            unit = reader.take(f"{prefix}.position")
            if placement == "asymmetric_low":
                position = 0.02 + 0.18 * unit
            elif placement == "asymmetric_high":
                position = 0.80 + 0.18 * unit
            else:
                position = 0.15 + 0.70 * unit
        start = stop = _interpolate(low, high, position)
    else:
        width_unit = reader.take(f"{prefix}.width")
        width = 0.35 + 0.45 * width_unit if regime == "wide" else 0.03 + 0.17 * width_unit
        available = 1.0 - width
        if placement == "edge_low":
            start_fraction = 0.0
        elif placement == "edge_high":
            start_fraction = available
        else:
            unit = reader.take(f"{prefix}.position")
            if placement == "asymmetric_low":
                start_fraction = 0.20 * available * unit
            elif placement == "asymmetric_high":
                start_fraction = available * (0.80 + 0.20 * unit)
            else:
                start_fraction = available * unit
        start = _interpolate(low, high, start_fraction)
        stop = start + width * (high - low)
    if log_space:
        start, stop = _DIRECT_NUMERIC.exp(start), _DIRECT_NUMERIC.exp(stop)
        if regime == "fixed":
            # Fixed physical endpoints pass through log/exp and later through
            # products such as sigma_D=D*fraction.  Keep a tiny, declared
            # generator-side interior margin so those exact-domain boundaries
            # cannot cross by one ulp during authoritative GUI conversion.
            if placement == "edge_low":
                start = stop = domain.low + max(
                    abs(domain.low) * 1.0e-12,
                    (domain.high - domain.low) * 1.0e-14,
                )
            elif placement == "edge_high":
                start = stop = domain.high - max(
                    abs(domain.high) * 1.0e-12,
                    (domain.high - domain.low) * 1.0e-14,
                )
    start = min(domain.high, max(domain.low, start))
    stop = min(domain.high, max(domain.low, stop))
    if stop < start and np.isclose(stop, start, rtol=1.0e-14, atol=0.0):
        stop = start
    return ClosedInterval(start, stop), regime, placement


def _width_interval(mean: ClosedInterval, fraction: ClosedInterval) -> ClosedInterval:
    # Outward rounding is essential for fixed coupled ranges.  Without it,
    # ``(mean * fraction) / mean`` can round one ulp above the identical
    # fraction endpoint and turn a mathematically non-empty closed interval
    # into an empty one inside the authoritative codec.
    raw_low = float(mean.low * fraction.low)
    raw_high = float(mean.high * fraction.high)
    low = float(np.nextafter(raw_low, 0.0))
    high = float(np.nextafter(raw_high, np.inf))
    return ClosedInterval(low, high)


def _design(axis_key: str, regime: str, placement: str) -> AxisRangeDesign:
    return AxisRangeDesign(axis_key, regime, placement)


def _required_spacing(shape: str, r: float, h: float | None) -> float:
    if shape == CYLINDER:
        if h is None:
            raise RuntimeError("cylinder exclusion size requires h")
        size = _DIRECT_NUMERIC.hypot(2.0 * r, h)
    else:
        size = 2.0 * r
    if not np.isfinite(size) or size <= 0.0:
        raise RuntimeError("could not evaluate the authoritative exclusion size")
    return HARD_CORE_SPACING_MARGIN * size


def _direct_v5_geometry_query_for_topology(
    reader: V5SobolCoordinateReader,
    *,
    topology: tuple[str, ...],
    sobol_index: int,
) -> V5BoundsQuery:
    policies = tuple(
        _choose(reader.take(f"geometry.slot_{slot + 1}.D_policy"), PRESENCE_POLICIES)
        for slot in range(len(topology))
    )
    resolution_policy: PresencePolicy = _choose(
        reader.take("geometry.resolution_policy"), PRESENCE_POLICIES
    )
    bounds: list[GuiComponentBounds] = []
    designs: list[AxisRangeDesign] = []
    for slot, (shape, d_policy) in enumerate(zip(topology, policies)):
        prefix = f"geometry.slot_{slot + 1}"
        d_enabled = d_policy != "absent"
        r, regime, placement = _axis_interval(
            reader,
            f"{prefix}.R",
            _D_ENABLED_R_DOMAIN if d_enabled else R_DOMAIN,
            log_space=True,
        )
        sigma_r_fraction, fraction_regime, fraction_placement = _axis_interval(
            reader,
            f"{prefix}.sigma_R_fraction",
            SIZE_WIDTH_FRACTION_DOMAIN,
            log_space=True,
        )
        values: dict[str, object] = {
            "shape": shape,
            "R": r,
            "sigma_R": (
                sigma_r_fraction
                if shape == VERTICAL_CYLINDER
                else _width_interval(r, sigma_r_fraction)
            ),
        }
        designs.extend(
            (
                _design(f"component[{slot}].R", regime, placement),
                _design(f"component[{slot}].sigma_R", fraction_regime, fraction_placement),
            )
        )
        h: ClosedInterval | None = None
        if shape == CYLINDER:
            h, regime, placement = _axis_interval(
                reader,
                f"{prefix}.h",
                _D_ENABLED_H_DOMAIN if d_enabled else H_DOMAIN,
                log_space=True,
            )
            sigma_h_fraction, fraction_regime, fraction_placement = _axis_interval(
                reader,
                f"{prefix}.sigma_h_fraction",
                SIZE_WIDTH_FRACTION_DOMAIN,
                log_space=True,
            )
            values.update(h=h, sigma_h=_width_interval(h, sigma_h_fraction))
            designs.extend(
                (
                    _design(f"component[{slot}].h", regime, placement),
                    _design(f"component[{slot}].sigma_h", fraction_regime, fraction_placement),
                )
            )
        if d_enabled:
            required = _required_spacing(shape, r.low, None if h is None else h.low)
            safe_low = max(D_DOMAIN.low, required * (1.0 + 1.0e-8))
            if safe_low >= D_DOMAIN.high:  # pragma: no cover - guarded domains prevent this
                raise RuntimeError("D-enabled geometry domains cannot admit hard-core spacing")
            d, regime, placement = _axis_interval(
                reader,
                f"{prefix}.D",
                ClosedInterval(safe_low, D_DOMAIN.high),
                log_space=True,
            )
            sigma_d_fraction, fraction_regime, fraction_placement = _axis_interval(
                reader,
                f"{prefix}.sigma_D_fraction",
                D_WIDTH_FRACTION_DOMAIN,
                log_space=True,
            )
            values.update(
                D=d,
                sigma_D=_width_interval(d, sigma_d_fraction),
                allow_D_absent=d_policy == "optional",
            )
            designs.extend(
                (
                    _design(f"component[{slot}].D", regime, placement),
                    _design(f"component[{slot}].sigma_D", fraction_regime, fraction_placement),
                )
            )
        bounds.append(GuiComponentBounds(**values))

    resolution_bounds = None
    if resolution_policy != "absent":
        sigma_res, sigma_regime, sigma_placement = _axis_interval(
            reader,
            "geometry.resolution.sigma_res",
            RESOLUTION_SIGMA_DOMAIN,
            log_space=True,
        )
        nu_res, nu_regime, nu_placement = _axis_interval(
            reader,
            "geometry.resolution.nu_res",
            RESOLUTION_NU_DOMAIN,
            log_space=False,
        )
        resolution_bounds = ResolutionBounds(sigma_res, nu_res)
        designs.extend(
            (
                _design("resolution.sigma_res", sigma_regime, sigma_placement),
                _design("resolution.nu_res", nu_regime, nu_placement),
            )
        )
    return V5BoundsQuery.create(
        query_seed=sobol_index,
        generation_attempt=0,
        component_bounds=tuple(bounds),
        resolution_presence_policy=resolution_policy,
        resolution_bounds=resolution_bounds,
        axis_designs=tuple(designs),
        numeric_policy_version=V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    )


def direct_v5_geometry_query(
    reader: V5SobolCoordinateReader,
    *,
    sobol_index: int,
) -> V5BoundsQuery:
    """Create the generating-topology query directly from named axes."""

    topology = TOPOLOGIES[
        min(int(reader.take("discrete.topology") * len(TOPOLOGIES)), len(TOPOLOGIES) - 1)
    ]
    return _direct_v5_geometry_query_for_topology(
        reader,
        topology=topology,
        sobol_index=sobol_index,
    )


def direct_v5_geometry_query_for_topology(
    reader: V5SobolCoordinateReader,
    *,
    topology_id: int,
    sobol_index: int,
) -> V5BoundsQuery:
    """Map the same slot coordinates into one explicitly selected topology.

    ``discrete.topology`` is intentionally not consumed here: it identifies the
    generating topology only.  The caller-supplied topology is an explicit
    search-query selection and must be recorded by the owning audit artifact.
    """

    return _direct_v5_geometry_query_for_topology(
        reader,
        topology=topology_from_id(topology_id),
        sobol_index=sobol_index,
    )


def _target_coordinate_name(index: int) -> str:
    if index < RESOLUTION_OFFSET:
        slot, offset = divmod(index, COMPONENT_STRIDE)
        return f"target.slot_{slot + 1}.{COMPONENT_UNIT_AXES[offset]}"
    return f"target.resolution.{RESOLUTION_UNIT_AXES[index - RESOLUTION_OFFSET]}"


def direct_v5_solution_target(
    reader: V5SobolCoordinateReader,
    query: V5BoundsQuery,
    *,
    pattern_id: int,
    sobol_index: int,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
) -> V5SolutionTarget:
    """Map active coordinates through the complete slot-contract branch codec."""

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
    if pattern_id not in feasible_patterns:
        raise ValueError("pattern_id is not canonical and feasible in the complete slot query")
    codec = query.codec_for(pattern_id)
    values = [INACTIVE_UNIT_VALUE] * UNIT_CUBE_DIMENSIONS
    epsilon = V5_LOCAL_TARGET_OPEN_EPSILON
    for index in codec.active_indices:
        unit = reader.take(_target_coordinate_name(index))
        values[index] = epsilon + (1.0 - 2.0 * epsilon) * unit
    latent, resolution = codec.decode(values)
    canonical = canonicalize_component_slots(
        codec,
        latent,
        resolution,
        component_intensity_bounds=component_intensity_bounds,
    )
    # Re-encoding a coupled physical boundary during slot canonicalization can
    # round an interior value to exactly 0 or 1.  Clamp the final canonical
    # local representative back to the same declared open target support and
    # derive truth from that final coordinate, preserving exact replay.
    local_target = list(canonical.coordinates.unit_cube)
    for index in codec.active_indices:
        local_target[index] = min(
            1.0 - epsilon,
            max(epsilon, local_target[index]),
        )
    roundtrip_components, roundtrip_resolution = codec.decode(local_target)
    return V5SolutionTarget(
        query=query,
        pattern_id=pattern_id,
        target_seed=sobol_index,
        local_target_unit=tuple(local_target),
        truth_components=codec.latent_components_to_gui(roundtrip_components),
        truth_resolution=roundtrip_resolution,
        physical_numeric_policy_version=(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
    )


__all__ = [
    "V5_DIRECT_GEOMETRY_TARGET_VERSION",
    "direct_v5_geometry_query",
    "direct_v5_geometry_query_for_topology",
    "direct_v5_solution_target",
]
