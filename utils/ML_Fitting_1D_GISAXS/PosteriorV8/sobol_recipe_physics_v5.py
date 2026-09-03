"""Direct query-first physical orchestration for a named V5.2 Sobol point."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .amplitude_query_sampling_v5 import V5AmplitudeRangeRegimes
from .amplitude_query_v5 import V5AmplitudeQuery
from .bounds_query_v5 import V5BoundsQuery, V5SolutionTarget
from .branch_catalog import decode_branch_pattern
from .sobol_amplitude_recipe_v5 import (
    V5DirectAmplitudeComposition,
    direct_v5_amplitude_composition,
    direct_v5_amplitude_query,
)
from .sobol_geometry_recipe_v5 import (
    V5_DIRECT_GEOMETRY_TARGET_VERSION,
    direct_v5_geometry_query,
    direct_v5_solution_target,
)
from .sobol_recipe_coordinates_v5 import V5SobolCoordinateReader
from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
)
from .universal_query_contract_v5 import V5TopologyQuery


V5_DIRECT_PHYSICS_VERSION = (
    "posterior_v8_v5_2_complete_slot_numeric_contract_sobol_physical_map_v7"
)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class V5DirectPhysics:
    query: V5BoundsQuery
    amplitude_query: V5AmplitudeQuery
    amplitude_range_regimes: V5AmplitudeRangeRegimes
    target: V5SolutionTarget
    amplitude: V5DirectAmplitudeComposition
    feasible_amplitude_regimes: tuple[str, ...]
    inactive_coordinate_names: tuple[str, ...]
    geometry_target_version: str = V5_DIRECT_GEOMETRY_TARGET_VERSION
    version: str = V5_DIRECT_PHYSICS_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.query, V5BoundsQuery):
            raise TypeError("query must be a V5BoundsQuery")
        if not isinstance(self.amplitude_query, V5AmplitudeQuery):
            raise TypeError("amplitude_query must be a V5AmplitudeQuery")
        if not isinstance(self.target, V5SolutionTarget):
            raise TypeError("target must be a V5SolutionTarget")
        if not isinstance(self.amplitude, V5DirectAmplitudeComposition):
            raise TypeError("amplitude must be a V5DirectAmplitudeComposition")
        if not isinstance(self.amplitude_range_regimes, V5AmplitudeRangeRegimes):
            raise TypeError("amplitude_range_regimes must be a V5AmplitudeRangeRegimes")
        if self.amplitude_range_regimes.component_count != len(self.query.topology):
            raise ValueError("amplitude range assignments do not match query topology")
        if (self.amplitude_range_regimes.int_res is None) != (
            self.query.resolution_presence_policy == "absent"
        ):
            raise ValueError("amplitude range assignments do not match Resolution policy")
        if self.target.query != self.query:
            raise ValueError("direct Sobol target does not belong to its physics query")
        if self.query.numeric_policy_version != V5_DETERMINISTIC_NUMERIC_POLICY_VERSION:
            raise ValueError("direct Sobol geometry query must use deterministic numeric policy")
        if self.amplitude_query.numeric_policy_version != self.query.numeric_policy_version:
            raise ValueError("direct Sobol amplitude and geometry numeric policies differ")
        if self.target.physical_numeric_policy_version != self.query.numeric_policy_version:
            raise ValueError("direct Sobol target and query numeric policies differ")
        if self.target.pattern_id not in self.query.feasible_wire_pattern_ids:
            raise ValueError("direct Sobol target branch is not feasible in its query")
        _, resolution_present = decode_branch_pattern(self.target.pattern_id)
        if resolution_present != self.amplitude.resolution_present:
            raise ValueError("direct Sobol target and amplitude Resolution states differ")
        if self.amplitude.component_count != len(self.query.topology):
            raise ValueError("direct Sobol amplitude does not match query topology")
        constraint = self.amplitude_query.constraint_for_branch(
            resolution_present=resolution_present
        )
        if not constraint.contains(
            self.amplitude.coefficient_vector,
            k=self.amplitude.k,
            atol=2.0e-9,
        ):
            raise ValueError("direct Sobol amplitude escaped its query")
        if self.geometry_target_version != V5_DIRECT_GEOMETRY_TARGET_VERSION:
            raise ValueError("unsupported direct Sobol geometry-target version")
        if self.version != V5_DIRECT_PHYSICS_VERSION:
            raise ValueError("unsupported direct Sobol physics version")

    @property
    def amplitude_range_regime(self) -> str:
        """Compatibility summary; scientific provenance uses the per-axis assignment."""

        return self.amplitude_range_regimes.summary


def direct_v5_physics_from_sobol(
    unit_coordinates,
    *,
    sobol_index: int,
) -> V5DirectPhysics:
    """Map query -> branch/target -> composition directly, without a PRNG."""

    index = _nonnegative_integer(sobol_index, "sobol_index")
    reader = V5SobolCoordinateReader.create(unit_coordinates)
    query = direct_v5_geometry_query(reader, sobol_index=index)
    amplitude_query, range_regimes = direct_v5_amplitude_query(reader, query)
    complete_query = V5TopologyQuery(query, amplitude_query)
    branch_offset = min(
        int(
            reader.take("discrete.branch_within_feasible_catalog")
            * len(complete_query.feasible_wire_pattern_ids)
        ),
        len(complete_query.feasible_wire_pattern_ids) - 1,
    )
    pattern_id = complete_query.feasible_wire_pattern_ids[branch_offset]
    target = direct_v5_solution_target(
        reader,
        query,
        pattern_id=pattern_id,
        sobol_index=index,
        component_intensity_bounds=amplitude_query.component_intensities,
    )
    _, resolution_present = decode_branch_pattern(pattern_id)
    constraint = amplitude_query.constraint_for_branch(resolution_present=resolution_present)
    amplitude, feasible_regimes = direct_v5_amplitude_composition(reader, constraint)
    return V5DirectPhysics(
        query=query,
        amplitude_query=amplitude_query,
        amplitude_range_regimes=range_regimes,
        target=target,
        amplitude=amplitude,
        feasible_amplitude_regimes=feasible_regimes,
        inactive_coordinate_names=reader.inactive,
    )


__all__ = [
    "V5_DIRECT_PHYSICS_VERSION",
    "V5DirectAmplitudeComposition",
    "V5DirectPhysics",
    "direct_v5_physics_from_sobol",
]
