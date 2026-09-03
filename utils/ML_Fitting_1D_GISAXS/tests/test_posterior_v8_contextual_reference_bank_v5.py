from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    AxisRangeDesign,
    V5BoundsQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    bind_v5_calibrated_observation_threshold,
    compatibility_stratum_from_v5_observation,
    inspect_v5_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    fit_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_reference_bank_v5 import (
    V5_CONTEXTUAL_REFERENCE_CLAIM,
    bind_v5_contextual_reference_candidate,
    build_v5_contextual_reference_bank,
    build_v5_contextual_reference_scope,
    contextual_candidate_evidence_from_task,
    contextual_reference_parameter_distance,
    legal_contextual_slot_permutations,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    gui_component_to_latent,
    latent_component_to_gui,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_executor_v5 import (
    build_v5_frozen_exact_search_protocol,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_view,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    evaluate_gui_forward_snapshot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5ExactSearchObservation,
    V5FrozenSearchTask,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_contract_v5 import (
    V5TopologyQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    build_v5_universal_candidate_context,
)


_SOURCE = {
    "exact_search_executor_v5.py": sha256(b"executor-source/test").hexdigest(),
    "profiled_forward.py": sha256(b"forward-source/test").hexdigest(),
}


def _geometry_bounds(low: float = 5.0, high: float = 30.0) -> GuiComponentBounds:
    return GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(low, high),
        sigma_R=ClosedInterval(0.5, 3.0),
    )


def _geometry_query(bounds) -> V5BoundsQuery:
    return V5BoundsQuery.create(
        query_seed=4321,
        generation_attempt=0,
        component_bounds=bounds,
        resolution_presence_policy="absent",
        resolution_bounds=None,
        axis_designs=tuple(
            AxisRangeDesign(f"component[{slot}].{axis}", "wide", "interior")
            for slot in range(2)
            for axis in ("R", "sigma_R")
        ),
    )


def _amplitude_query(intervals) -> V5AmplitudeQuery:
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 10.0),
        k=ClosedInterval(0.1, 10.0),
        component_intensities=intervals,
        resolution_presence_policy="absent",
        int_res=None,
    )


@pytest.fixture(scope="module")
def formal_base(tmp_path_factory):
    recipe = sample_v5_clean_recipe((SPHERE, SPHERE), recipe_seed=88031, pattern_id=0)
    view = next(
        value
        for index in range(8)
        if (
            value := build_v5_observation_data_view(recipe, index, split_id="validation")
        ).acceptance_sigma_log
        is not None
    )
    stratum = compatibility_stratum_from_v5_observation(view)
    calibration = fit_compatibility_calibration(
        tuple(
            CompatibilityCalibrationSample(
                sample_id=f"reference-calibration-{index}",
                independent_group_id=f"reference-group-{index}",
                stratum=stratum,
                score=1.0e12 + index,
                effective_valid_point_count=view.effective_valid_point_count,
                acquisition_policy_id=view.acquisition_policy_id,
                measurement_sigma_available=True,
            )
            for index in range(9)
        ),
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.8,
        minimum_samples_per_stratum=5,
    )
    path = tmp_path_factory.mktemp("contextual-reference") / "calibration.json"
    write_compatibility_calibration_atomic(path, calibration)
    checked = inspect_v5_compatibility_calibration(path)
    threshold = bind_v5_calibrated_observation_threshold(checked, view)
    seeds = V5FrozenLocalSobolSchedule.generate(
        schedule_id="contextual-reference-test", point_count=2, base_seed=19
    )
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id="contextual-reference-test",
        direct_scout_seed_count=1,
        per_seed_forward_evaluation_limit=1,
    )
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id="contextual-reference-paper-test",
        seed_schedule=seeds,
        optimizer_schedule=optimizer,
        protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        calibration_identity=checked.identity,
        delta_separation_threshold=0.08,
    )
    exact = V5ExactSearchObservation.from_observation_view(
        view, curve_id="contextual-reference-observation"
    )
    return view, exact, threshold, protocol


def _task(formal_base, *, bounds=None, intensities=None) -> V5FrozenSearchTask:
    view, exact, threshold, protocol = formal_base
    geometry = _geometry_query(
        (_geometry_bounds(), _geometry_bounds()) if bounds is None else bounds
    )
    amplitude = _amplitude_query(
        (ClosedInterval(0.0, 1.0),) * 2 if intensities is None else intensities
    )
    context = build_v5_universal_candidate_context(
        view.preprocessed,
        view.uncertainty,
        (V5TopologyQuery(geometry, amplitude),),
    )
    return V5FrozenSearchTask(
        query_index=0,
        clean_group_id="contextual-reference-clean-group",
        recipe_id="contextual-reference-recipe",
        observation_id=exact.observed_curve.curve_id,
        universal_context=context,
        exact_observation=exact,
        query_catalog_artifact_id="query-catalog/contextual-reference-test",
        query_catalog_artifact_sha256=sha256(b"query-catalog/test").hexdigest(),
        branch_index=context.global_branch_keys.index("topology-03:wire-00"),
        protocol=protocol,
        calibrated_threshold=threshold,
    )


def _candidate(task, candidate_id, radii, amplitudes, *, rank=1, gui_k=None) -> CandidateInput:
    components = tuple(
        gui_component_to_latent(GuiComponentParameters(SPHERE, R=radius, sigma_R=0.1 * radius))
        for radius in radii
    )
    linear = LinearSolutionSnapshot(
        background=0.01,
        particle_amplitudes=tuple(amplitudes),
        resolution_amplitude=0.0,
        k=gui_k,
    )
    exact = evaluate_gui_forward_snapshot(
        task.observed_curve.q,
        tuple(latent_component_to_gui(value) for value in components),
        resolution=None,
        background=linear.background,
        particle_amplitudes=linear.particle_amplitudes,
        resolution_amplitude=0.0,
        gui_k=linear.k,
    )
    return CandidateInput(
        candidate_id=candidate_id,
        proposal_rank=rank,
        topology_id=task.branch.global_key.topology_id,
        components=components,
        resolution=None,
        linear_solution=linear,
        exact_intensity=exact,
        bounds_pass=True,
        physics_pass=True,
    )


def _bind(scope, task, candidate):
    evidence = contextual_candidate_evidence_from_task(
        task,
        source_bundle_sha256=scope.source_bundle_sha256,
        executor_artifact_id=f"executor/{candidate.candidate_id}",
        executor_artifact_sha256=sha256(candidate.candidate_id.encode()).hexdigest(),
    )
    return bind_v5_contextual_reference_candidate(scope, task, candidate, evidence)


def test_scope_contains_full_gui_bounds_polytope_and_scientific_provenance(formal_base):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    payload = json.loads(scope.audit_json)
    query = payload["full_gui_query_contracts"][0]

    assert payload["claim"] == V5_CONTEXTUAL_REFERENCE_CLAIM
    assert payload["exact_curve_sha256"] == task.exact_curve_sha256
    assert payload["protocol_sha256"] == task.protocol.sha256
    assert payload["calibrated_threshold_sha256"] == task.calibrated_threshold.sha256
    assert payload["source_bundle_sha256"] == scope.source_bundle_sha256
    assert payload["distance_contract"]["schema"] == V5_QUERY_PARAMETER_DISTANCE_SCHEMA
    assert payload["distance_contract"]["version"] == V5_QUERY_PARAMETER_DISTANCE_VERSION
    assert payload["distance_contract"]["sha256"] == V5_QUERY_PARAMETER_DISTANCE_SHA256
    assert query["geometry_query"]["component_bounds"][0]["R"] == {
        "high": 30.0,
        "low": 5.0,
    }
    polytope = query["branches"][0]["amplitude_constraint"]["coefficient_polytope"]
    assert polytope["canonical_gauge"].startswith("exists kappa")
    assert polytope["auxiliary_k"]["name"] == "kappa"
    assert polytope["auxiliary_k"]["ratio_names"] == ["Int_1", "Int_2"]


def test_binding_rejects_an_explicit_k_that_has_only_existential_coefficient_support(
    formal_base,
):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    candidate = _candidate(
        task,
        "wrong-explicit-k",
        (10.0, 20.0),
        (0.3, 0.7),
        gui_k=100.0,
    )
    assert task.amplitude_constraint.contains(
        (
            candidate.linear_solution.background,
            *candidate.linear_solution.particle_amplitudes,
        )
    )
    with pytest.raises(ValueError, match="concrete complete-query GUI ranges"):
        _bind(scope, task, candidate)


def test_equal_complete_slot_context_canonicalizes_permutations_and_deduplicates(formal_base):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    first = _bind(scope, task, _candidate(task, "first", (10.0, 20.0), (0.3, 0.7)))
    second = _bind(
        scope,
        task,
        _candidate(task, "second", (20.0, 10.0), (0.7, 0.3), rank=2),
    )

    assert legal_contextual_slot_permutations(task) == ((0, 1), (1, 0))
    assert first.canonical_parameter_sha256 == second.canonical_parameter_sha256
    assert contextual_reference_parameter_distance(scope, first, second) == pytest.approx(0.0)
    forward = build_v5_contextual_reference_bank(scope, (first, second))
    reverse = build_v5_contextual_reference_bank(scope, (second, first))
    assert forward.audit_sha256 == reverse.audit_sha256
    assert len(forward.representatives) == 1
    assert set(forward.representatives[0].member_candidate_ids) == {"first", "second"}
    assert forward.audit_payload()["candidates"][0]["canonical_parameter"]["components"]


def test_shared_k_gauge_witnesses_collapse_to_one_physical_family(formal_base):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    first = _bind(
        scope,
        task,
        _candidate(task, "gauge-k-one", (10.0, 20.0), (0.3, 0.7), gui_k=1.0),
    )
    second = _bind(
        scope,
        task,
        _candidate(
            task,
            "gauge-k-two",
            (10.0, 20.0),
            (0.3, 0.7),
            rank=2,
            gui_k=2.0,
        ),
    )

    assert first.canonical_parameter_sha256 != second.canonical_parameter_sha256
    assert contextual_reference_parameter_distance(scope, first, second) == pytest.approx(0.0)
    bank = build_v5_contextual_reference_bank(scope, (first, second))
    assert len(bank.representatives) == 1
    assert set(bank.representatives[0].member_candidate_ids) == {
        "gauge-k-one",
        "gauge-k-two",
    }


def test_observable_particle_scale_is_not_collapsed_by_composition(formal_base):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    weak = _bind(
        scope,
        task,
        _candidate(task, "weak-scale", (10.0, 20.0), (0.05, 0.05), gui_k=0.1),
    )
    strong = _bind(
        scope,
        task,
        _candidate(
            task,
            "strong-scale",
            (10.0, 20.0),
            (5.0, 5.0),
            rank=2,
            gui_k=10.0,
        ),
    )

    assert contextual_reference_parameter_distance(scope, weak, strong) > scope.delta
    assert len(build_v5_contextual_reference_bank(scope, (weak, strong)).representatives) == 2


def test_complete_linkage_does_not_chain_a_bridge_cluster(formal_base):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    # Identical component geometries make these exact curves identical while
    # the normalized compositions form A--B--C with only the endpoints farther
    # apart than delta.  B is the deterministic first leader, so a leader-only
    # assignment would incorrectly put all three in one cluster.
    bridge = tuple(
        _bind(
            scope,
            task,
            _candidate(
                task,
                candidate_id,
                (10.0, 10.0),
                (first_weight, 1.0 - first_weight),
                rank=rank,
            ),
        )
        for rank, (candidate_id, first_weight) in enumerate(
            (("bridge-a", 0.1), ("bridge-b", 0.2), ("bridge-c", 0.3)), 1
        )
    )
    left, middle, right = bridge
    assert contextual_reference_parameter_distance(scope, left, middle) <= scope.delta
    assert contextual_reference_parameter_distance(scope, middle, right) <= scope.delta
    assert contextual_reference_parameter_distance(scope, left, right) > scope.delta

    forward = build_v5_contextual_reference_bank(scope, bridge)
    reverse = build_v5_contextual_reference_bank(scope, tuple(reversed(bridge)))
    assert forward.audit_sha256 == reverse.audit_sha256
    assert len(forward.representatives) == 2
    by_id = {value.candidate.candidate_id: value for value in bridge}
    for representative in forward.representatives:
        members = tuple(by_id[value] for value in representative.member_candidate_ids)
        assert all(
            contextual_reference_parameter_distance(scope, first, second) <= scope.delta
            for index, first in enumerate(members)
            for second in members[index + 1 :]
        )


@pytest.mark.parametrize(
    ("bounds", "intensities"),
    (
        (
            (_geometry_bounds(5.0, 25.0), _geometry_bounds(8.0, 30.0)),
            (ClosedInterval(0.0, 1.0),) * 2,
        ),
        (
            (_geometry_bounds(), _geometry_bounds()),
            (ClosedInterval(0.2, 0.8), ClosedInterval(0.1, 0.9)),
        ),
    ),
)
def test_heterogeneous_geometry_or_amplitude_slots_are_not_exchangeable(
    formal_base, bounds, intensities
):
    task = _task(formal_base, bounds=bounds, intensities=intensities)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    first = _bind(scope, task, _candidate(task, "slot-a", (10.0, 20.0), (0.3, 0.7)))
    second = _bind(
        scope,
        task,
        _candidate(task, "slot-b", (20.0, 10.0), (0.7, 0.3), rank=2),
    )

    assert legal_contextual_slot_permutations(task) == ((0, 1),)
    assert first.canonical_parameter_sha256 != second.canonical_parameter_sha256
    assert contextual_reference_parameter_distance(scope, first, second) > scope.delta
    assert len(build_v5_contextual_reference_bank(scope, (first, second)).representatives) == 2


@pytest.mark.parametrize(
    "field",
    (
        "geometry_query_sha256",
        "amplitude_constraint_sha256",
        "source_bundle_sha256",
        "protocol_sha256",
        "calibration_identity_sha256",
        "calibrated_threshold_sha256",
    ),
)
def test_declared_query_bounds_source_protocol_and_calibration_mismatch_fail_closed(
    formal_base, field
):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    candidate = _candidate(task, "mismatch", (10.0, 20.0), (0.3, 0.7))
    evidence = contextual_candidate_evidence_from_task(
        task,
        source_bundle_sha256=scope.source_bundle_sha256,
        executor_artifact_id="executor/mismatch",
        executor_artifact_sha256="3" * 64,
    )
    changed_identity = tuple(
        (name, "f" * 64 if name == field else value) for name, value in evidence.identity
    )

    with pytest.raises(ValueError, match=field):
        bind_v5_contextual_reference_candidate(
            scope,
            task,
            candidate,
            replace(evidence, identity=changed_identity),
        )


def test_different_query_task_and_nonreplaying_exact_curve_fail_closed(formal_base):
    task = _task(formal_base)
    scope = build_v5_contextual_reference_scope(task, source_sha256=_SOURCE, reference_delta=0.08)
    changed_task = _task(
        formal_base,
        bounds=(_geometry_bounds(5.0, 29.0), _geometry_bounds(5.0, 29.0)),
    )
    changed_candidate = _candidate(changed_task, "changed-query", (10.0, 20.0), (0.3, 0.7))
    changed_evidence = contextual_candidate_evidence_from_task(
        changed_task,
        source_bundle_sha256=scope.source_bundle_sha256,
        executor_artifact_id="executor/changed-query",
        executor_artifact_sha256="4" * 64,
    )
    with pytest.raises(ValueError, match="escaped the query/observation"):
        bind_v5_contextual_reference_candidate(
            scope, changed_task, changed_candidate, changed_evidence
        )

    candidate = _candidate(task, "bad-forward", (10.0, 20.0), (0.3, 0.7))
    evidence = contextual_candidate_evidence_from_task(
        task,
        source_bundle_sha256=scope.source_bundle_sha256,
        executor_artifact_id="executor/bad-forward",
        executor_artifact_sha256="5" * 64,
    )
    with pytest.raises(ValueError, match="does not replay"):
        bind_v5_contextual_reference_candidate(
            scope,
            task,
            replace(candidate, exact_intensity=candidate.exact_intensity * 1.001),
            evidence,
        )
