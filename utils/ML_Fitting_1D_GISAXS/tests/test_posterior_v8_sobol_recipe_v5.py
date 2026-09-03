from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import qmc

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.clean_recipe_forward_v5 import (
    authoritative_v5_gui_parameters,
    evaluate_v5_clean_recipe_forward,
    validate_v5_clean_recipe_like,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    V5_LOCAL_TARGET_OPEN_EPSILON,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_REGIMES,
    V5_BACKGROUND_DOMAIN,
    V5_COMPONENT_INTENSITY_DOMAIN,
    V5_INT_RES_DOMAIN,
    V5_K_DOMAIN,
    V5AmplitudeRangeRegimes,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    ClosedInterval,
    TOPOLOGIES,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_branch_catalog import (
    PRESENCE_POLICIES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gui_amplitude_constraints import (
    GuiAmplitudeConstraint,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_amplitude_recipe_v5 import (
    direct_v5_amplitude_composition,
    direct_v5_amplitude_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    V5DesignPoint,
    V5SobolDesign,
    materialize_v5_design_points,
    materialize_v5_design_points_for_indices,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_DIM,
    V5SobolCoordinateReader,
    v5_sobol_recipe_coordinate_contract,
    v5_sobol_recipe_design,
    validate_v5_sobol_recipe_coordinates,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_physics_v5 import (
    direct_v5_physics_from_sobol,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_v5 import (
    V5SobolCleanRecipe,
    authoritative_v5_sobol_gui_parameters,
    evaluate_v5_sobol_clean_recipe,
    materialize_v5_sobol_clean_recipe,
    v5_sobol_ood_materialization_contract,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    OOD_LABELS,
    V5SplitCounts,
    V5SplitPlan,
)


def _coordinates(value: float = 0.371) -> list[float]:
    return [value] * V5_SOBOL_RECIPE_DIM


def _point(coordinates, *, index: int = 7) -> V5DesignPoint:
    return V5DesignPoint(
        sobol_index=index,
        assigned_split="train",
        ood_label=None,
        clean_group_id=sha256(f"group:{index}".encode()).hexdigest(),
        unit_coordinates=tuple(coordinates),
    )


def _small_split_plan() -> V5SplitPlan:
    return V5SplitPlan.create(
        V5SplitCounts(
            train=3,
            tuning_validation=2,
            calibration=2,
            test=2,
            reference=2,
            ood_topology=1,
            ood_range_width=1,
            ood_weak_component=1,
            ood_acquisition_policy=1,
        ),
        start_index=0,
        guard_band=3,
    )


def _regime_coordinate(regime: str) -> float:
    return (V5_AMPLITUDE_RANGE_REGIMES.index(regime) + 0.5) / len(
        V5_AMPLITUDE_RANGE_REGIMES
    )


def _amplitude_axis_intervals(query):
    result = {
        "BG": (query.background, V5_BACKGROUND_DOMAIN),
        "k": (query.k, V5_K_DOMAIN),
    }
    result.update(
        {
            f"Int_{slot + 1}": (interval, V5_COMPONENT_INTENSITY_DOMAIN)
            for slot, interval in enumerate(query.component_intensities)
        }
    )
    if query.int_res is not None:
        result["int_Res"] = (query.int_res, V5_INT_RES_DOMAIN)
    return result


def _assert_amplitude_interval_matches_regime(interval, domain, regime):
    assert domain.low <= interval.low <= interval.high <= domain.high
    if regime == "full":
        assert interval == domain
    elif regime == "fixed":
        assert interval.low == interval.high
    elif regime == "edge_low":
        assert interval.low == domain.low
    elif regime == "edge_high":
        assert interval.high == domain.high


def test_coordinate_dictionary_is_complete_versioned_and_snapshot_stable():
    contract = v5_sobol_recipe_coordinate_contract()

    assert V5_SOBOL_RECIPE_DIM == 168
    assert len(V5_SOBOL_RECIPE_COORDINATE_NAMES) == len(set(V5_SOBOL_RECIPE_COORDINATE_NAMES))
    assert contract["coordinate_names"] == list(V5_SOBOL_RECIPE_COORDINATE_NAMES)
    assert contract["consumption"].endswith("no_coordinate_derived_PRNG_seed")
    assert contract["fixed_discrete_mappings"]["discrete.topology"] == [
        "+".join(value) for value in TOPOLOGIES
    ]
    assert (
        V5_SOBOL_RECIPE_COORDINATE_SHA256
        == "acaf19814ed43a0c0578d0798371da21a7d10d45e87fd2858793ec760105fdec"
    )
    assert "paired complete geometry_plus_amplitude query" in contract[
        "contextual_discrete_mappings"
    ]["discrete.branch_within_feasible_catalog"]


def test_direct_sobol_composition_supports_fixed_k10_int2_and_persists_the_gauge():
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.5, 0.5),
        component_intensities=(ClosedInterval(2.0, 2.0),),
        k=ClosedInterval(10.0, 10.0),
        resolution_present=False,
    )
    reader = V5SobolCoordinateReader.create(np.zeros(V5_SOBOL_RECIPE_DIM))
    result, _ = direct_v5_amplitude_composition(reader, constraint)
    assert result.k == 10.0
    assert result.particle_weights == (2.0,)
    assert result.particle_amplitudes == (20.0,)
    assert constraint.contains(result.coefficient_vector, k=result.k)


def test_direct_materialization_replays_hash_and_uses_single_shared_exact_forward():
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    point = _point(_coordinates())

    first = materialize_v5_sobol_clean_recipe(point, design)
    replay = materialize_v5_sobol_clean_recipe(point, design)

    assert first == replay
    assert validate_v5_clean_recipe_like(first) is first
    assert sha256(first.canonical_json.encode()).hexdigest() == first.sha256
    assert first.recipe_seed == point.sobol_index
    assert "direct_no_coordinate_derived_seed_or_PRNG" in first.canonical_json
    constraint = first.amplitude_query.constraint_for_branch(
        resolution_present=first.amplitude.resolution_present
    )
    assert constraint.contains(first.amplitude.coefficient_vector, atol=2.0e-9)
    assert authoritative_v5_sobol_gui_parameters is authoritative_v5_gui_parameters
    assert evaluate_v5_sobol_clean_recipe is evaluate_v5_clean_recipe_forward
    q = np.geomspace(1.0e-3, 3.0, 73)
    curve = evaluate_v5_sobol_clean_recipe(first, q)
    assert curve.shape == q.shape
    assert np.all(np.isfinite(curve))
    assert np.all(curve > 0.0)
    assert not curve.flags.writeable


def test_clean_physics_path_does_not_invoke_numpy_random(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct Sobol clean physics invoked a PRNG")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)

    physics = direct_v5_physics_from_sobol(_coordinates(0.217), sobol_index=11)

    assert physics.query.query_seed == 11
    assert physics.target.target_seed == 11
    assert physics.amplitude.audit_payload()["coordinate_consumption"] == "direct_no_PRNG"


@pytest.mark.parametrize("value", [0.0, np.nextafter(1.0, 0.0)])
def test_half_open_coordinate_boundaries_map_without_rejection_or_retry(value):
    physics = direct_v5_physics_from_sobol(_coordinates(value), sobol_index=3)

    constraint = physics.amplitude_query.constraint_for_branch(
        resolution_present=physics.amplitude.resolution_present
    )
    assert constraint.contains(physics.amplitude.coefficient_vector, atol=2.0e-9)
    assert all(
        0.0 < coordinate < 1.0
        for coordinate, active in zip(
            physics.target.local_target_unit,
            physics.query.codec_for(physics.target.pattern_id).active_mask,
        )
        if active
    )


def test_named_coordinates_materialize_the_required_mixed_amplitude_query():
    coordinates = _coordinates(0.371)
    topology_id = topology_id_for(("Sphere", "Sphere"))
    coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]] = (
        topology_id + 0.5
    ) / len(TOPOLOGIES)
    coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["geometry.resolution_policy"]] = 0.9
    expected = {
        "BG": "fixed",
        "k": "full",
        "Int_1": "narrow",
        "Int_2": "edge_high",
        "int_Res": "edge_low",
    }
    for axis, regime in expected.items():
        coordinates[
            V5_SOBOL_RECIPE_COORDINATE_INDEX[f"amplitude.query.{axis}.regime"]
        ] = _regime_coordinate(regime)

    physics = direct_v5_physics_from_sobol(coordinates, sobol_index=31)

    assert physics.query.resolution_presence_policy == "required"
    assignment = physics.amplitude_range_regimes
    assert assignment.audit_payload()["axis_regimes"] == {
        "BG": "fixed",
        "k": "full",
        "Int_1": "narrow",
        "Int_2": "edge_high",
        "Int_3": None,
        "Int_4": None,
        "int_Res": "edge_low",
    }
    assert assignment.summary == "mixed"
    intervals = _amplitude_axis_intervals(physics.amplitude_query)
    for axis, regime in expected.items():
        _assert_amplitude_interval_matches_regime(*intervals[axis], regime)
    assert len(physics.amplitude_query.model_embedding(1.0e4)) == 21
    assert physics.amplitude.audit_payload()["coefficient_fraction_is_observability"] is False
    inactive = set(physics.inactive_coordinate_names)
    assert {
        f"amplitude.query.Int_{slot}.{field}"
        for slot in (3, 4)
        for field in ("regime", "width", "position")
    }.issubset(inactive)


def test_topology_coordinate_has_an_explicit_one_to_one_bin_map():
    seen = []
    for topology_id, expected in enumerate(TOPOLOGIES):
        coordinates = _coordinates(0.0)
        coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]] = (
            topology_id + 0.5
        ) / len(TOPOLOGIES)
        physics = direct_v5_physics_from_sobol(coordinates, sobol_index=topology_id)
        seen.append(physics.query.topology)
        assert physics.query.topology == expected
    assert tuple(seen) == TOPOLOGIES


def test_wrong_dimension_range_and_coordinate_dictionary_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        validate_v5_sobol_recipe_coordinates(_coordinates()[:-1])
    for bad in (-0.01, 1.0, np.nan, np.inf):
        values = _coordinates()
        values[4] = bad
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            validate_v5_sobol_recipe_coordinates(values)

    wrong_design = V5SobolDesign(coordinate_names=("not_the_recipe",), scramble_seed=3)
    with pytest.raises(ValueError, match="frozen direct-recipe"):
        materialize_v5_sobol_clean_recipe(_point(_coordinates()), wrong_design)


def test_recipe_and_nested_physics_tampering_are_rejected():
    design = v5_sobol_recipe_design(scramble_seed=17)
    recipe = materialize_v5_sobol_clean_recipe(_point(_coordinates(0.613)), design)

    with pytest.raises(ValueError, match="audit hash"):
        replace(recipe, sha256="0" * 64)
    with pytest.raises(ValueError, match="unsupported direct Sobol clean-recipe generator"):
        replace(
            recipe,
            generator_version=(
                "posterior_v8_v5_2_complete_slot_contract_gui_k_int_sobol_clean_recipe_v3"
            ),
        )
    with pytest.raises(ValueError, match="unsupported direct Sobol clean-recipe schema"):
        replace(recipe, schema_version="gisaxs.posterior_v8.direct_sobol_clean_recipe/v3")
    with pytest.raises(ValueError, match="unsupported direct Sobol physics version"):
        replace(
            recipe.physics,
            version=(
                "posterior_v8_v5_2_complete_slot_contract_gui_k_int_sobol_physical_map_v3"
            ),
        )
    with pytest.raises(ValueError, match="unsupported direct amplitude generator"):
        replace(
            recipe.amplitude,
            generator_version=(
                "posterior_v8_query_contained_independent_gui_k_int_sobol_composition_v2"
            ),
        )
    with pytest.raises(ValueError, match="unsupported direct Sobol geometry-target version"):
        replace(recipe.physics, geometry_target_version="stale-v1")
    original = recipe.physics.amplitude_range_regimes
    other_regime = "wide" if original.background != "wide" else "narrow"
    tampered_assignment = V5AmplitudeRangeRegimes.create(
        original.component_count,
        resolution_presence_policy=recipe.query.resolution_presence_policy,
        background=other_regime,
        k=original.k,
        component_intensities=original.active_component_intensities,
        int_res=original.int_res,
    )
    tampered_physics = replace(
        recipe.physics,
        amplitude_range_regimes=tampered_assignment,
    )
    with pytest.raises(ValueError, match="does not replay"):
        replace(recipe, physics=tampered_physics)


def test_recipe_json_binds_design_point_split_and_inactive_coordinates():
    design = v5_sobol_recipe_design(scramble_seed=91)
    point = V5DesignPoint(
        sobol_index=101,
        assigned_split="test",
        ood_label=None,
        clean_group_id=sha256(b"test-101").hexdigest(),
        unit_coordinates=tuple(_coordinates(0.42)),
    )
    recipe = V5SobolCleanRecipe.create(point=point, design=design)
    payload = json.loads(recipe.canonical_json)

    assert payload["source"]["sobol_index"] == 101
    assert payload["source"]["assigned_split"] == "test"
    assert payload["source"]["ood_label"] is None
    assert payload["source"]["sobol_design_sha256"] == design.sha256
    assert payload["source"]["coordinate_contract_sha256"] == (V5_SOBOL_RECIPE_COORDINATE_SHA256)
    assert payload["source"]["inactive_coordinate_names"]
    assert payload["physics"]["amplitude_range_regimes_sha256"] == (
        recipe.physics.amplitude_range_regimes.sha256
    )


def test_exact_scrambled_sobol_index_589_per_axis_range_regression():
    coordinates = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(10)[589]

    physics = direct_v5_physics_from_sobol(coordinates, sobol_index=589)

    assert physics.query.topology == (
        "sphere",
        "cylinder",
        "cylinder",
        "cylinder",
    )
    assert physics.target.pattern_id == 19
    assert physics.amplitude.regime == "balanced_particles"
    assert physics.amplitude_range_regimes.active_component_intensities == (
        "full",
        "fixed",
        "fixed",
        "edge_low",
    )


def test_exact_scrambled_sobol_index_753_fraction_boundary_regression():
    coordinates = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(10)[753]

    physics = direct_v5_physics_from_sobol(coordinates, sobol_index=753)

    assert physics.query.topology == (
        "sphere",
        "cylinder",
        "vertical_cylinder",
        "vertical_cylinder",
    )
    assert physics.target.pattern_id == 30
    assert physics.amplitude.regime == "balanced_particles"
    assert physics.amplitude_range_regimes.active_component_intensities == (
        "fixed",
        "wide",
        "edge_low",
        "wide",
    )


def test_exact_scrambled_sobol_index_118_fixed_intensity_stays_contained():
    coordinates = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(10)[118]

    physics = direct_v5_physics_from_sobol(coordinates, sobol_index=118)
    constraint = physics.amplitude_query.constraint_for_branch(
        resolution_present=physics.amplitude.resolution_present
    )

    assert constraint.component_intensities[0].low == (
        constraint.component_intensities[0].high
    )
    assert constraint.contains(
        physics.amplitude.coefficient_vector,
        k=physics.amplitude.k,
        atol=2.0e-9,
    )


def test_exact_scrambled_sobol_index_179_strict_hard_core_finishes_and_replays():
    coordinates = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(10)[179]

    physics = direct_v5_physics_from_sobol(coordinates, sobol_index=179)
    codec = physics.query.codec_for(physics.target.pattern_id)
    components, resolution = codec.decode(physics.target.local_target_unit)

    assert codec.latent_components_to_gui(components) == physics.target.truth_components
    assert resolution == physics.target.truth_resolution


def test_design_json_replay_binds_hash_names_and_coordinate_contract():
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    encoded = design.to_json()
    payload = json.loads(encoded)

    assert V5SobolDesign.from_json(encoded) == design
    assert payload["design_sha256"] == design.sha256
    assert payload["coordinate_names"] == list(V5_SOBOL_RECIPE_COORDINATE_NAMES)
    assert payload["coordinate_contract_sha256"] == V5_SOBOL_RECIPE_COORDINATE_SHA256

    payload["coordinate_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not reproduce"):
        V5SobolDesign.from_json(json.dumps(payload))

    duplicate = encoded.rstrip()[:-1] + ',"design_sha256":"' + design.sha256 + '"}'
    with pytest.raises(ValueError, match="duplicate Sobol-design JSON field"):
        V5SobolDesign.from_json(duplicate)


def test_selected_design_indices_are_byte_exact_without_prefix_allocation():
    plan = _small_split_plan()
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    requested = (plan.blocks[3].start, plan.blocks[0].start, plan.blocks[4].stop - 1)

    selected = materialize_v5_design_points_for_indices(plan, design, requested)
    prefix = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(6)

    assert tuple(point.sobol_index for point in selected) == requested
    for point in selected:
        assert np.array_equal(
            np.asarray(point.unit_coordinates),
            prefix[point.sobol_index],
        )
    complete = {point.sobol_index: point for point in materialize_v5_design_points(plan, design)}
    assert selected == tuple(complete[index] for index in requested)


def test_selected_design_indices_reject_guard_duplicate_and_invalid_inputs():
    plan = _small_split_plan()
    design = v5_sobol_recipe_design(scramble_seed=20260903)

    with pytest.raises(ValueError, match="guard band"):
        materialize_v5_design_points_for_indices(
            plan,
            design,
            (plan.guard_blocks[0].start,),
        )
    with pytest.raises(ValueError, match="unique"):
        materialize_v5_design_points_for_indices(plan, design, (0, 0))
    with pytest.raises(ValueError, match="cannot be empty"):
        materialize_v5_design_points_for_indices(plan, design, ())
    with pytest.raises(TypeError, match="sequence"):
        materialize_v5_design_points_for_indices(plan, design, "5")


@pytest.mark.skipif(
    os.environ.get("GISAXS_RUN_SLOW_SOBOL_STRESS") != "1",
    reason="set GISAXS_RUN_SLOW_SOBOL_STRESS=1 for the 1024-point amplitude-range audit",
)
def test_amplitude_query_map_covers_frozen_1024_point_scrambled_prefix():
    points = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(10)

    seen = {
        axis: set()
        for axis in ("BG", "k", "Int_1", "Int_2", "Int_3", "Int_4", "int_Res")
    }
    for coordinates in points:
        reader = V5SobolCoordinateReader.create(coordinates)
        topology_unit = reader.take("discrete.topology")
        topology = TOPOLOGIES[
            min(int(topology_unit * len(TOPOLOGIES)), len(TOPOLOGIES) - 1)
        ]
        resolution_unit = reader.take("geometry.resolution_policy")
        resolution_policy = PRESENCE_POLICIES[
            min(
                int(resolution_unit * len(PRESENCE_POLICIES)),
                len(PRESENCE_POLICIES) - 1,
            )
        ]
        amplitude_context = SimpleNamespace(
            topology=topology,
            resolution_presence_policy=resolution_policy,
        )
        amplitude_query, range_regimes = direct_v5_amplitude_query(
            reader,
            amplitude_context,
        )
        for resolution_present in amplitude_query.allowed_resolution_states:
            amplitude_query.constraint_for_branch(resolution_present=resolution_present)
        assignments = range_regimes.audit_payload()["axis_regimes"]
        intervals = _amplitude_axis_intervals(amplitude_query)
        inactive = set(reader.inactive)
        for axis, regime in assignments.items():
            if regime is None:
                assert {
                    f"amplitude.query.{axis}.{field}"
                    for field in ("regime", "width", "position")
                }.issubset(inactive)
                continue
            seen[axis].add(regime)
            _assert_amplitude_interval_matches_regime(*intervals[axis], regime)
    assert all(regimes == set(V5_AMPLITUDE_RANGE_REGIMES) for regimes in seen.values())


@pytest.mark.skipif(
    os.environ.get("GISAXS_RUN_SLOW_SOBOL_STRESS") != "1",
    reason="set GISAXS_RUN_SLOW_SOBOL_STRESS=1 for the 1024-point full-physics audit",
)
def test_full_physics_materializes_and_stays_in_every_frozen_1024_point_query():
    points = qmc.Sobol(
        d=V5_SOBOL_RECIPE_DIM,
        scramble=True,
        bits=52,
        seed=20260903,
    ).random_base2(10)
    seen_topologies = set()
    for index, coordinates in enumerate(points):
        physics = direct_v5_physics_from_sobol(coordinates, sobol_index=index)
        seen_topologies.add(physics.query.topology_id)
        amplitude_embedding = physics.amplitude_query.model_embedding(1.0)
        assert len(amplitude_embedding) == 21
        assert np.all(np.isfinite(amplitude_embedding))
        codec = physics.query.codec_for(physics.target.pattern_id)
        active = codec.active_mask
        assert all(
            V5_LOCAL_TARGET_OPEN_EPSILON <= value <= 1.0 - V5_LOCAL_TARGET_OPEN_EPSILON
            for value, present in zip(physics.target.local_target_unit, active)
            if present
        )
        for bounds, truth in zip(
            physics.query.component_bounds,
            physics.target.truth_components,
        ):
            for axis in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_D"):
                interval = getattr(bounds, axis)
                value = getattr(truth, axis)
                if value is not None:
                    assert interval is not None
                    assert interval.low <= value <= interval.high
        if physics.target.truth_resolution is not None:
            assert physics.query.resolution_bounds is not None
            assert physics.query.resolution_bounds.contains(
                physics.target.truth_resolution
            )
        constraint = physics.amplitude_query.constraint_for_branch(
            resolution_present=physics.amplitude.resolution_present
        )
        assert constraint.contains(
            physics.amplitude.coefficient_vector,
            k=physics.amplitude.k,
            atol=2.0e-9,
        )
        assert all(
            interval.low <= value <= interval.high
            for interval, value in zip(
                constraint.component_intensities,
                physics.amplitude.component_intensities,
            )
        )
    assert seen_topologies == set(range(len(TOPOLOGIES)))


@pytest.mark.parametrize("ood_label", OOD_LABELS)
def test_unimplemented_ood_transforms_fail_closed_instead_of_relabelling_iid(ood_label):
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    point = V5DesignPoint(
        sobol_index=701,
        assigned_split="ood",
        ood_label=ood_label,
        clean_group_id=sha256(ood_label.encode()).hexdigest(),
        unit_coordinates=tuple(_coordinates(0.37)),
    )

    with pytest.raises(ValueError, match="fail-closed"):
        materialize_v5_sobol_clean_recipe(point, design)

    contract = v5_sobol_ood_materialization_contract()
    assert contract["ordinary_iid_point_may_be_relabelled_ood"] is False
    assert contract["acquisition_policy_holdout"]["owner"] == "observation_v5"
