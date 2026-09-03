from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import ProfiledBranchCodec
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_branch_catalog import (
    build_contextual_branch_catalog,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    gui_component_to_latent,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gui_amplitude_constraints import (
    GuiAmplitudeConstraint,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_PAYLOAD,
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
    legal_query_slot_permutations,
    query_local_parameter_distance,
)


_FAMILY_MERGE_DELTA = 0.08


def _component_bounds() -> GuiComponentBounds:
    return GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(5.0, 30.0),
        sigma_R=ClosedInterval(0.5, 3.0),
    )


def _task(
    *,
    count: int = 1,
    background: ClosedInterval = ClosedInterval(0.0, 1.0),
    k: ClosedInterval = ClosedInterval(0.1, 10.0),
    intensities: tuple[ClosedInterval, ...] | None = None,
):
    bounds = tuple(_component_bounds() for _ in range(count))
    intensity_ranges = intensities or tuple(ClosedInterval(0.01, 1.0) for _ in range(count))
    catalog = build_contextual_branch_catalog(
        bounds,
        component_intensity_bounds=intensity_ranges,
        resolution_presence_policy="absent",
    )
    constraint = GuiAmplitudeConstraint(
        background=background,
        component_intensities=intensity_ranges,
        k=k,
        resolution_present=False,
    )
    codec = ProfiledBranchCodec.build(
        (SPHERE,) * count,
        bounds,
        (False,) * count,
    )
    topology_id = topology_id_for((SPHERE,) * count)
    branch = SimpleNamespace(
        topology_batch_index=0,
        global_key=SimpleNamespace(topology_id=topology_id),
        condition=SimpleNamespace(
            d_present=(False, False, False, False),
            resolution_present=False,
            varying_dimension_mask=codec.active_mask,
        ),
    )
    return SimpleNamespace(
        branch=branch,
        codec=codec,
        amplitude_constraint=constraint,
        universal_context=SimpleNamespace(
            topology_queries=(SimpleNamespace(contextual_branch_catalog=catalog),)
        ),
    )


def _candidate(
    task,
    candidate_id: str,
    amplitudes: tuple[float, ...],
    *,
    k: float,
    background: float = 0.0,
    radii: tuple[float, ...] | None = None,
) -> CandidateInput:
    radii = radii or tuple(10.0 + 5.0 * index for index in range(len(amplitudes)))
    components = tuple(
        gui_component_to_latent(GuiComponentParameters(SPHERE, R=radius, sigma_R=0.1 * radius))
        for radius in radii
    )
    return CandidateInput(
        candidate_id=candidate_id,
        proposal_rank=1,
        topology_id=task.branch.global_key.topology_id,
        components=components,
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=background,
            particle_amplitudes=amplitudes,
            resolution_amplitude=0.0,
            k=k,
        ),
        exact_intensity=np.ones(4, dtype=np.float64),
        bounds_pass=True,
        physics_pass=True,
    )


def test_effective_particle_scale_is_not_erased_by_composition_normalization() -> None:
    task = _task()
    weak = _candidate(task, "weak", (0.1,), k=0.1)
    strong = _candidate(task, "strong", (10.0,), k=10.0)

    assert query_local_parameter_distance(task, weak, strong) > 0.08


@pytest.mark.parametrize(
    ("left_amplitude", "right_amplitude"),
    ((0.1, 10.0), (10.0, 1.0e6)),
)
def test_default_full_range_retains_cross_decade_particle_scale(
    left_amplitude: float,
    right_amplitude: float,
) -> None:
    task = _task(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        intensities=(ClosedInterval(0.0, 1.0e8),),
    )
    left = _candidate(
        task,
        "left",
        (left_amplitude,),
        k=left_amplitude,
    )
    right = _candidate(
        task,
        "right",
        (right_amplitude,),
        k=right_amplitude,
    )

    forward = query_local_parameter_distance(task, left, right)
    reverse = query_local_parameter_distance(task, right, left)
    assert forward > _FAMILY_MERGE_DELTA
    assert reverse == pytest.approx(forward, rel=1.0e-14, abs=1.0e-14)


def test_default_full_range_retains_cross_decade_background_scale() -> None:
    task = _task(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0, 1.0),
        intensities=(ClosedInterval(1.0, 1.0),),
    )
    weak = _candidate(task, "weak-bg", (1.0,), k=1.0, background=0.1)
    strong = _candidate(task, "strong-bg", (1.0,), k=1.0, background=10.0)

    assert query_local_parameter_distance(task, weak, strong) > _FAMILY_MERGE_DELTA


def test_shared_k_reparameterization_is_a_zero_distance_gui_gauge() -> None:
    task = _task(
        background=ClosedInterval(0.5, 0.5),
        k=ClosedInterval(1.0, 2.0),
        intensities=(ClosedInterval(1.0, 2.0),),
    )
    first = _candidate(task, "k-one", (2.0,), k=1.0, background=0.5)
    second = _candidate(task, "k-two", (2.0,), k=2.0, background=0.5)

    assert query_local_parameter_distance(task, first, second) == pytest.approx(0.0)


def test_absolute_background_scale_remains_a_distance_axis() -> None:
    task = _task(
        background=ClosedInterval(0.1, 10.0),
        k=ClosedInterval(1.0, 1.0),
        intensities=(ClosedInterval(1.0, 1.0),),
    )
    weak = _candidate(task, "weak-bg", (1.0,), k=1.0, background=0.1)
    strong = _candidate(task, "strong-bg", (1.0,), k=1.0, background=10.0)

    assert query_local_parameter_distance(task, weak, strong) > 0.08


def test_zero_boundary_is_finite_injective_and_symmetric() -> None:
    task = _task(
        count=2,
        background=ClosedInterval(0.0, 0.0),
        k=ClosedInterval(1.0, 1.0),
        intensities=(ClosedInterval(0.0, 1.0), ClosedInterval(1.0, 1.0)),
    )
    at_zero = _candidate(task, "at-zero", (0.0, 1.0), k=1.0)
    at_upper = _candidate(task, "at-upper", (1.0, 1.0), k=1.0)

    forward = query_local_parameter_distance(task, at_zero, at_upper)
    reverse = query_local_parameter_distance(task, at_upper, at_zero)
    assert np.isfinite(forward)
    assert forward > _FAMILY_MERGE_DELTA
    assert reverse == pytest.approx(forward, rel=1.0e-14, abs=1.0e-14)


def test_fixed_coefficient_intervals_are_stable_and_fail_closed() -> None:
    task = _task(
        background=ClosedInterval(0.5, 0.5),
        k=ClosedInterval(2.0, 2.0),
        intensities=(ClosedInterval(3.0, 3.0),),
    )
    first = _candidate(task, "first", (6.0,), k=2.0, background=0.5)
    shifted_geometry = _candidate(
        task,
        "shifted-geometry",
        (6.0,),
        k=2.0,
        background=0.5,
        radii=(20.0,),
    )
    invalid_fixed_value = _candidate(
        task,
        "invalid-fixed-value",
        (6.1,),
        k=2.0,
        background=0.5,
    )

    assert query_local_parameter_distance(task, first, first) == pytest.approx(0.0)
    assert np.isfinite(query_local_parameter_distance(task, first, shifted_geometry))
    with pytest.raises(ValueError, match="escaped the GUI query"):
        query_local_parameter_distance(task, invalid_fixed_value, first)


def test_geometry_uses_only_the_branch_varying_dimension_mask() -> None:
    task = _task(
        background=ClosedInterval(0.0, 0.0),
        k=ClosedInterval(1.0, 1.0),
        intensities=(ClosedInterval(1.0, 1.0),),
    )
    varying = [False] * len(task.codec.active_mask)
    varying[0] = True
    task.branch.condition.varying_dimension_mask = tuple(varying)
    left = _candidate(task, "left", (1.0,), k=1.0, radii=(10.0,))
    right = _candidate(task, "right", (1.0,), k=1.0, radii=(20.0,))
    left_encoded = task.codec.encode(left.components, left.resolution).unit_cube[0]
    right_encoded = task.codec.encode(right.components, right.resolution).unit_cube[0]

    assert query_local_parameter_distance(task, left, right) == pytest.approx(
        abs(left_encoded - right_encoded)
    )


def test_k4_single_varying_amplitude_axis_is_not_diluted_by_fixed_marginals() -> None:
    task = _task(
        count=4,
        background=ClosedInterval(0.0, 0.0),
        k=ClosedInterval(1.0, 1.0),
        intensities=(
            ClosedInterval(0.1, 10.0),
            ClosedInterval(1.0, 1.0),
            ClosedInterval(1.0, 1.0),
            ClosedInterval(1.0, 1.0),
        ),
    )
    left = _candidate(task, "left", (0.1, 1.0, 1.0, 1.0), k=1.0)
    right = _candidate(task, "right", (10.0, 1.0, 1.0, 1.0), k=1.0)

    distance = query_local_parameter_distance(task, left, right)
    assert np.isfinite(distance)
    assert distance > 0.0
    assert query_local_parameter_distance(task, right, left) == pytest.approx(distance)


def test_all_fixed_query_axes_give_zero_for_legal_candidates() -> None:
    task = _task(
        background=ClosedInterval(0.5, 0.5),
        k=ClosedInterval(2.0, 2.0),
        intensities=(ClosedInterval(3.0, 3.0),),
    )
    task.branch.condition.varying_dimension_mask = (False,) * len(task.codec.active_mask)
    first = _candidate(task, "first", (6.0,), k=2.0, background=0.5)
    second = _candidate(task, "second", (6.0,), k=2.0, background=0.5)

    assert query_local_parameter_distance(task, first, second) == pytest.approx(0.0)


def test_distance_contract_declares_zero_safe_effective_coefficient_coordinates() -> None:
    assert V5_QUERY_PARAMETER_DISTANCE_SCHEMA.endswith("/v2")
    assert "zero_safe_log_scale_group_max" in V5_QUERY_PARAMETER_DISTANCE_VERSION
    assert V5_QUERY_PARAMETER_DISTANCE_PAYLOAD["non_gauge_scale_quotients"] == "none"
    assert V5_QUERY_PARAMETER_DISTANCE_PAYLOAD["zero_safe_dynamic_decades"] == 34.0
    assert V5_QUERY_PARAMETER_DISTANCE_PAYLOAD["log_decades_per_unit"] == 10.0
    assert "fixed geometry" in V5_QUERY_PARAMETER_DISTANCE_PAYLOAD["dimension_rule"]


def test_only_complete_query_equivalent_slots_may_be_permuted() -> None:
    equal = _task(count=2)
    assert legal_query_slot_permutations(equal) == ((0, 1), (1, 0))
    first = _candidate(equal, "first", (0.2, 0.8), k=1.0, radii=(10.0, 20.0))
    swapped = _candidate(
        equal,
        "swapped",
        (0.8, 0.2),
        k=1.0,
        radii=(20.0, 10.0),
    )
    assert query_local_parameter_distance(equal, first, swapped) == pytest.approx(0.0)

    unequal = _task(
        count=2,
        intensities=(ClosedInterval(0.01, 0.4), ClosedInterval(0.5, 1.0)),
    )
    assert legal_query_slot_permutations(unequal) == ((0, 1),)
