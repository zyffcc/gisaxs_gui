"""Query-local, gauge-invariant distance for V5.2 solution families.

The GUI exposes ``k`` and the component ``Int_i`` values independently, but
the forward model observes only their products.  Consequently, changing
``k`` while compensating every ``Int`` is a parameterization gauge and must
not manufacture an extra solution family.  Conversely, either normalizing all
linear coefficients to a composition or dividing by the full linear query
span destroys decade resolution in the default wide amplitude query.  This
contract compares every effective coefficient in a zero-safe logarithmic
coordinate defined only by the user's query, thereby quotienting only the
exact GUI gauge.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import permutations
from typing import Protocol

import numpy as np

from .evaluation import CandidateInput, ReferenceMode
from .grouped_artifact_v5 import canonical_json


V5_QUERY_PARAMETER_DISTANCE_SCHEMA = "gisaxs.posterior_v8.query_parameter_distance/v2"
V5_QUERY_PARAMETER_DISTANCE_VERSION = (
    "posterior_v8_query_local_geometry_composition_zero_safe_log_scale_group_max_v2"
)
V5_QUERY_PARAMETER_DISTANCE_SCOPE = (
    "one contextual topology/wire branch; user-query-normalized nonlinear "
    "geometry and effective-coefficient composition; query-local zero-safe logarithmic "
    "BG/particle/resolution scale; quotients only the exact shared-k GUI gauge"
)
V5_QUERY_PARAMETER_DISTANCE_ZERO_SAFE_DECADES = 34.0
V5_QUERY_PARAMETER_DISTANCE_LOG_DECADES_PER_UNIT = 10.0
V5_QUERY_PARAMETER_DISTANCE_PAYLOAD = {
    "schema": V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    "version": V5_QUERY_PARAMETER_DISTANCE_VERSION,
    "scope": V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    "geometry_normalization": (
        "branch_codec_user_query_unit_coordinates_selected_by_branch_varying_dimension_mask"
    ),
    "effective_coefficient_composition": "BG_a_i_optional_a_res_divided_by_total",
    "effective_coefficient_axes": "BG_a_i_optional_a_res",
    "effective_coefficient_bounds": (
        "exact_GUI_auxiliary_kappa_projected_polytope_marginal_axis_bounds"
    ),
    "effective_coefficient_coordinate": (
        "[log10(1+x/s)-log10(1+low/s)]/10"
    ),
    "zero_safe_scale": (
        "s=max(low,high*10^-34,float64_tiny); fixed axes contribute zero"
    ),
    "zero_safe_dynamic_decades": V5_QUERY_PARAMETER_DISTANCE_ZERO_SAFE_DECADES,
    "log_decades_per_unit": V5_QUERY_PARAMETER_DISTANCE_LOG_DECADES_PER_UNIT,
    "distance_norm": (
        "max(RMS(varying_geometry_plus_varying_effective_composition),"
        "RMS(varying_effective_log_scale_axes))"
    ),
    "dimension_rule": (
        "fixed geometry and fixed effective-coefficient marginals are omitted; "
        "if every query axis is fixed, two legal candidates have distance zero"
    ),
    "gui_gauge": "same_effective_coefficients_are_equal_independent_of_k_witness",
    "non_gauge_scale_quotients": "none",
    "slot_exchangeability": ("only_equal_complete_geometry_D_and_component_Int_query_contracts"),
    "cross_contextual_branch_distance": "infinity",
    "fixed_interval_rule": "equal_is_zero_otherwise_infinity",
}
V5_QUERY_PARAMETER_DISTANCE_SHA256 = sha256(
    canonical_json(V5_QUERY_PARAMETER_DISTANCE_PAYLOAD).encode("utf-8")
).hexdigest()


class _TaskLike(Protocol):
    @property
    def branch(self): ...

    @property
    def codec(self): ...

    @property
    def amplitude_constraint(self): ...

    @property
    def universal_context(self): ...


def _slot_equivalence_key(task: _TaskLike, slot: int) -> tuple[int, bool]:
    branch = task.branch
    topology_query = task.universal_context.topology_queries[branch.topology_batch_index]
    equivalence_class = next(
        class_index
        for class_index, members in enumerate(
            topology_query.contextual_branch_catalog.d_equivalence_classes
        )
        if slot in members
    )
    return equivalence_class, bool(branch.condition.d_present[slot])


def legal_query_slot_permutations(task: _TaskLike) -> tuple[tuple[int, ...], ...]:
    """Return only permutations authorized by the complete GUI query contract."""

    count = len(task.codec.topology)
    signatures = tuple(_slot_equivalence_key(task, slot) for slot in range(count))
    result = tuple(
        order
        for order in permutations(range(count))
        if all(
            signatures[destination] == signatures[source]
            for destination, source in enumerate(order)
        )
    )
    if not result or tuple(range(count)) not in result:
        raise RuntimeError("complete query contract lost the identity slot permutation")
    return result


def _query_log_coordinate(
    value: float,
    low: float,
    high: float,
) -> float:
    """Map one non-negative coefficient into a query-local log-scale coordinate.

    A zero lower bound needs a positive transition scale for a logarithmic
    coordinate.  Tying that scale to the query upper bound makes the mapping
    invariant to a change of intensity units.  The frozen 34-decade window
    covers the current default effective-amplitude support (about
    ``1e-18..1e16``) while ``log1p`` keeps zero finite and the map injective.
    """

    values = np.asarray((value, low, high), dtype=np.float64)
    if not np.all(np.isfinite(values)) or low > high:
        raise ValueError("distance interval and values must be finite and ordered")
    if low < 0.0:
        raise ValueError("effective coefficient intervals must be non-negative")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        float(np.finfo(np.float64).tiny), abs(value), abs(low), abs(high)
    )
    span = float(high - low)
    if span == 0.0:
        if abs(value - low) > tolerance:
            raise ValueError("value escaped a fixed effective-coefficient interval")
        return 0.0
    if value < low - tolerance or value > high + tolerance:
        raise ValueError("value escaped its effective-coefficient interval")
    clipped = min(max(float(value), float(low)), float(high))
    relative_floor = high * 10.0 ** (-V5_QUERY_PARAMETER_DISTANCE_ZERO_SAFE_DECADES)
    scale = max(float(low), float(relative_floor), float(np.finfo(np.float64).tiny))
    lower_coordinate = float(np.log1p(low / scale))
    upper_coordinate = float(np.log1p(high / scale))
    if not np.isfinite(upper_coordinate) or upper_coordinate <= lower_coordinate:  # pragma: no cover
        raise RuntimeError("effective-coefficient log coordinate collapsed")
    coordinate = (
        float(np.log1p(clipped / scale)) - lower_coordinate
    ) / (np.log(10.0) * V5_QUERY_PARAMETER_DISTANCE_LOG_DECADES_PER_UNIT)
    if not np.isfinite(coordinate):  # pragma: no cover
        raise RuntimeError("effective-coefficient log coordinate is non-finite")
    return float(max(coordinate, 0.0))


def _coefficient_coordinates(
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if coefficients.shape != lower.shape or upper.shape != lower.shape:
        raise RuntimeError("effective-coefficient coordinate shapes disagree")
    return np.asarray(
        [
            _query_log_coordinate(value, low, high)
            for value, low, high in zip(coefficients, lower, upper)
        ],
        dtype=np.float64,
    )


def _effective_coefficients(
    candidate: CandidateInput | ReferenceMode,
    order: tuple[int, ...],
) -> np.ndarray:
    linear = candidate.linear_solution
    values = [
        linear.background,
        *(linear.particle_amplitudes[index] for index in order),
    ]
    if candidate.resolution is not None:
        values.append(linear.resolution_amplitude)
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("effective forward coefficients must be finite and non-negative")
    if float(np.sum(result[1 : 1 + len(order)])) <= 0.0:
        raise ValueError("effective particle total must be strictly positive")
    return result


def _candidate_matches_branch(
    task: _TaskLike,
    candidate: CandidateInput | ReferenceMode,
) -> bool:
    branch = task.branch
    if candidate.topology_id != branch.global_key.topology_id:
        return False
    if (candidate.resolution is not None) != bool(branch.condition.resolution_present):
        return False
    if tuple(value.log_D is not None for value in candidate.components) != tuple(
        bool(value) for value in branch.condition.d_present[: len(candidate.components)]
    ):
        return False
    return True


def _validated_coefficients(
    task: _TaskLike,
    candidate: CandidateInput | ReferenceMode,
    order: tuple[int, ...],
) -> np.ndarray:
    coefficients = _effective_coefficients(candidate, order)
    if not task.amplitude_constraint.contains(
        coefficients,
        k=candidate.linear_solution.k,
    ):
        raise ValueError("candidate effective coefficients and explicit k escaped the GUI query")
    return coefficients


def query_local_parameter_distance(
    task: _TaskLike,
    left: CandidateInput | ReferenceMode,
    right: CandidateInput | ReferenceMode,
) -> float:
    """Return an RMS distance for physical solution families within one query.

    Different topology/D/Resolution branches are incomparable and therefore
    have infinite distance.  The shared GUI ``k`` witness is validated against
    the query but is not itself a distance axis: candidates with identical
    effective forward coefficients are gauge-equivalent.
    """

    if not isinstance(left, (CandidateInput, ReferenceMode)) or not isinstance(
        right, (CandidateInput, ReferenceMode)
    ):
        raise TypeError("left and right must be CandidateInput or ReferenceMode values")
    if not _candidate_matches_branch(task, left) or not _candidate_matches_branch(task, right):
        return float("inf")

    identity = tuple(range(len(left.components)))
    left_coefficients = _validated_coefficients(task, left, identity)
    left_coordinates = task.codec.encode(left.components, left.resolution)
    active = np.asarray(left_coordinates.active_mask, dtype=bool)
    varying_geometry = np.asarray(
        task.branch.condition.varying_dimension_mask,
        dtype=bool,
    )
    if varying_geometry.shape != active.shape or np.any(varying_geometry & ~active):
        raise RuntimeError("branch varying-dimension mask is incompatible with its codec")
    left_geometry = np.asarray(left_coordinates.unit_cube, dtype=np.float64)[
        varying_geometry
    ]

    constraint = task.amplitude_constraint
    polytope = constraint.coefficient_polytope()
    coefficient_lower = np.asarray(polytope.axis_lower, dtype=np.float64)
    coefficient_upper = np.asarray(polytope.axis_upper, dtype=np.float64)
    varying_coefficients = coefficient_upper > coefficient_lower
    left_amplitude = _coefficient_coordinates(
        left_coefficients,
        coefficient_lower,
        coefficient_upper,
    )

    best = float("inf")
    for order in legal_query_slot_permutations(task):
        components = tuple(right.components[index] for index in order)
        try:
            right_coefficients = _validated_coefficients(task, right, order)
            right_coordinates = task.codec.encode(components, right.resolution)
        except ValueError:
            continue
        right_geometry = np.asarray(right_coordinates.unit_cube, dtype=np.float64)[
            varying_geometry
        ]
        try:
            right_amplitude = _coefficient_coordinates(
                right_coefficients,
                coefficient_lower,
                coefficient_upper,
            )
        except ValueError:
            continue
        left_composition = (
            left_coefficients / float(np.sum(left_coefficients))
        )[varying_coefficients]
        right_composition = (
            right_coefficients / float(np.sum(right_coefficients))
        )[varying_coefficients]
        structural_delta = np.concatenate(
            (
                left_geometry - right_geometry,
                left_composition - right_composition,
            )
        )
        structural_distance = (
            0.0
            if structural_delta.size == 0
            else float(np.sqrt(np.mean(np.square(structural_delta))))
        )
        scale_delta = (left_amplitude - right_amplitude)[varying_coefficients]
        scale_distance = (
            0.0
            if scale_delta.size == 0
            else float(np.sqrt(np.mean(np.square(scale_delta))))
        )
        best = min(best, max(structural_distance, scale_distance))
    return best


__all__ = [
    "V5_QUERY_PARAMETER_DISTANCE_PAYLOAD",
    "V5_QUERY_PARAMETER_DISTANCE_SCHEMA",
    "V5_QUERY_PARAMETER_DISTANCE_SCOPE",
    "V5_QUERY_PARAMETER_DISTANCE_SHA256",
    "V5_QUERY_PARAMETER_DISTANCE_VERSION",
    "V5_QUERY_PARAMETER_DISTANCE_LOG_DECADES_PER_UNIT",
    "V5_QUERY_PARAMETER_DISTANCE_ZERO_SAFE_DECADES",
    "legal_query_slot_permutations",
    "query_local_parameter_distance",
]
