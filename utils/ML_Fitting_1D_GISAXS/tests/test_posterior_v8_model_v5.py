from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    bounds_embedding,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    branch_condition,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import ResolutionBounds
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_branch_catalog import (
    build_contextual_branch_catalog,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    ClosedInterval,
    GuiComponentBounds,
    SPHERE,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5 import (
    V5ContractStamp,
    build_branch_conditioned_proposal_model,
    validate_model_v5_graph_contract,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import (
    MODEL_V5_INPUT_KEYS,
    MODEL_V5_NAME,
    MODEL_V5_OUTPUT_KEYS,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective import (
    active_dimension_mask_for,
)


def _sphere_bounds(*, shifted: bool = False) -> GuiComponentBounds:
    if shifted:
        radius = ClosedInterval(20.0, 40.0)
        sigma_radius = ClosedInterval(2.0, 6.0)
    else:
        radius = ClosedInterval(10.0, 30.0)
        sigma_radius = ClosedInterval(1.0, 4.0)
    return GuiComponentBounds(
        shape=SPHERE,
        R=radius,
        sigma_R=sigma_radius,
        D=ClosedInterval(50.0, 100.0),
        sigma_D=ClosedInterval(5.0, 20.0),
        allow_D_absent=True,
    )


def _resolution_bounds() -> ResolutionBounds:
    return ResolutionBounds(
        sigma_res=ClosedInterval(0.01, 0.08),
        nu_res=ClosedInterval(2.0, 8.0),
    )


def _inputs(
    component_bounds,
    pattern_id: int,
    *,
    resolution_bounds=None,
    amplitude_query=None,
    max_points: int = 16,
):
    bounds = tuple(component_bounds)
    topology_id = topology_id_for(tuple(value.shape for value in bounds))
    embedding = np.asarray(bounds_embedding(bounds, resolution_bounds), dtype=np.float32)
    available = embedding[2::3].copy()
    active = np.asarray(active_dimension_mask_for(topology_id, pattern_id), dtype=np.float32)
    q = np.linspace(-1.0, 1.0, max_points, dtype=np.float32)
    x = np.stack((q, np.sin(q), np.cos(q)), axis=-1)[np.newaxis]
    if amplitude_query is None:
        amplitude_query = V5AmplitudeQuery.create(
            background=ClosedInterval(0.0, 1.0e5),
            k=ClosedInterval(1.0e-2, 1.0e6),
            component_intensities=tuple(ClosedInterval(0.0, 1.0) for _ in bounds),
            resolution_presence_policy=("optional" if resolution_bounds is not None else "absent"),
            int_res=(
                ClosedInterval(0.0, 1.0e4) if resolution_bounds is not None else None
            ),
        )
    return {
        "x": x,
        "point_mask": np.ones((1, max_points), dtype=np.bool_),
        "global_features": np.asarray([[0.1, -0.2, 0.3, 0.4, -0.5]], np.float32),
        "uncertainty_provenance": np.asarray([[1.0, 0.0, 0.0]], np.float32),
        "branch_topology_id": np.asarray([[topology_id]], np.int32),
        "branch_pattern_id": np.asarray([[pattern_id]], np.int32),
        "geometry_bounds_embedding": embedding[np.newaxis],
        "amplitude_bounds_embedding": np.asarray(
            [amplitude_query.model_embedding(1.0e4)], dtype=np.float32
        ),
        "available_dimension_mask": available[np.newaxis],
        "active_dimension_mask": active[np.newaxis],
        "varying_dimension_mask": active[np.newaxis].copy(),
    }


def _small_model():
    return build_branch_conditioned_proposal_model(
        max_points=16,
        width=8,
        encoder_blocks=1,
        mixture_components=3,
    )


def test_v5_outputs_one_candidate_score_and_branch_conditioned_local_mdn():
    model = _small_model()
    inputs = _inputs(
        (_sphere_bounds(), _sphere_bounds(shifted=True)),
        1,
        resolution_bounds=_resolution_bounds(),
    )
    outputs = model(inputs, training=False)
    assert set(outputs) == set(MODEL_V5_OUTPUT_KEYS)
    assert "topology_logits" not in outputs
    assert "branch_pattern_logits" not in outputs
    assert tuple(outputs["proposal_search_yield_logit"].shape) == (1, 1)
    assert tuple(outputs["mixture_logits"].shape) == (1, 3)
    assert tuple(outputs["mixture_loc"].shape) == (1, 3, 26)
    assert tuple(outputs["mixture_logscale"].shape) == (1, 3, 26)
    assert all(np.all(np.isfinite(value.numpy())) for value in outputs.values())
    assert model.name == MODEL_V5_NAME
    assert {value.name.split(":", 1)[0] for value in model.inputs} == set(MODEL_V5_INPUT_KEYS)


def test_optional_d_and_resolution_are_available_while_selected_branch_is_absent():
    inputs = _inputs(
        (_sphere_bounds(),),
        0,
        resolution_bounds=_resolution_bounds(),
    )
    available = inputs["available_dimension_mask"][0]
    active = inputs["active_dimension_mask"][0]
    varying = inputs["varying_dimension_mask"][0]
    assert np.all(available[[4, 5, 24, 25]] == 1.0)
    assert np.all(active[[4, 5, 24, 25]] == 0.0)
    assert np.all(varying[[4, 5, 24, 25]] == 0.0)
    _small_model()(inputs, training=False)


def test_heterogeneous_same_shape_nonstatic_canonical_assignment_is_allowed():
    bounds = (_sphere_bounds(), _sphere_bounds(shifted=True))
    catalog = build_contextual_branch_catalog(bounds, resolution_presence_policy="optional")
    assert 1 in catalog.wire_pattern_ids
    inputs = _inputs(bounds, 1, resolution_bounds=_resolution_bounds())
    outputs = _small_model()(inputs, training=False)
    assert np.isfinite(outputs["proposal_search_yield_logit"].numpy()).all()


def test_every_externally_enumerated_feasible_query_branch_can_be_scored():
    query = full_range_v5_query((SPHERE,), query_seed=73)
    model = _small_model()
    observed = []
    for pattern_id in query.feasible_wire_pattern_ids:
        condition = branch_condition(query, pattern_id)
        inputs = _inputs(
            query.component_bounds,
            pattern_id,
            resolution_bounds=query.resolution_bounds,
        )
        inputs.update(
            {
                "branch_topology_id": np.asarray([[condition.topology_id]], dtype=np.int32),
                "branch_pattern_id": np.asarray([[condition.pattern_id]], dtype=np.int32),
                "geometry_bounds_embedding": np.asarray(
                    [condition.bounds_embedding], dtype=np.float32
                ),
                "available_dimension_mask": np.asarray(
                    [condition.available_dimension_mask], dtype=np.float32
                ),
                "active_dimension_mask": np.asarray(
                    [condition.active_dimension_mask], dtype=np.float32
                ),
                "varying_dimension_mask": np.asarray(
                    [condition.varying_dimension_mask], dtype=np.float32
                ),
            }
        )
        output = model(inputs, training=False)
        observed.append(pattern_id)
        assert np.isfinite(output["proposal_search_yield_logit"].numpy()).all()
    assert tuple(observed) == query.feasible_wire_pattern_ids


def test_compatibility_score_has_graph_dependencies_on_bounds_and_branch():
    tf.keras.utils.set_random_seed(20260903)
    model = _small_model()
    bounds = (_sphere_bounds(), _sphere_bounds(shifted=True))
    first = _inputs(bounds, 1, resolution_bounds=_resolution_bounds())

    changed_bounds = (_sphere_bounds(shifted=True), _sphere_bounds())
    second = _inputs(changed_bounds, 1, resolution_bounds=_resolution_bounds())
    second["x"] = first["x"].copy()
    second["point_mask"] = first["point_mask"].copy()
    second["global_features"] = first["global_features"].copy()

    changed_branch = _inputs(bounds, 2, resolution_bounds=_resolution_bounds())
    changed_branch["x"] = first["x"].copy()
    changed_branch["point_mask"] = first["point_mask"].copy()
    changed_branch["global_features"] = first["global_features"].copy()

    changed_uncertainty = {name: value.copy() for name, value in first.items()}
    changed_uncertainty["uncertainty_provenance"][:] = (0.0, 0.0, 1.0)

    changed_amplitude = {name: value.copy() for name, value in first.items()}
    amplitude_query = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 10.0),
        k=ClosedInterval(1.0e3, 1.0e4),
        component_intensities=(ClosedInterval(0.1, 0.9), ClosedInterval(0.1, 0.9)),
        resolution_presence_policy="optional",
        int_res=ClosedInterval(0.0, 0.01),
    )
    changed_amplitude["amplitude_bounds_embedding"] = np.asarray(
        [amplitude_query.model_embedding(1.0e4)], dtype=np.float32
    )

    score = model(first, training=False)["proposal_search_yield_logit"].numpy()
    bounds_score = model(second, training=False)["proposal_search_yield_logit"].numpy()
    branch_score = model(changed_branch, training=False)["proposal_search_yield_logit"].numpy()
    uncertainty_score = model(changed_uncertainty, training=False)[
        "proposal_search_yield_logit"
    ].numpy()
    amplitude_score = model(changed_amplitude, training=False)[
        "proposal_search_yield_logit"
    ].numpy()
    assert not np.array_equal(score, bounds_score)
    assert not np.array_equal(score, branch_score)
    assert not np.array_equal(score, uncertainty_score)
    assert not np.array_equal(score, amplitude_score)


def test_amplitude_context_accepts_independent_int_ranges_below_and_above_unit_sum():
    tf.keras.utils.set_random_seed(20260904)
    model = _small_model()
    bounds = (_sphere_bounds(), _sphere_bounds(shifted=True))
    for intervals in (
        (ClosedInterval(0.01, 0.10), ClosedInterval(0.02, 0.20)),
        (ClosedInterval(2.0, 3.0), ClosedInterval(4.0, 5.0)),
    ):
        query = V5AmplitudeQuery.create(
            background=ClosedInterval(0.0, 10.0),
            k=ClosedInterval(1.0, 10.0),
            component_intensities=intervals,
            resolution_presence_policy="absent",
            int_res=None,
        )
        inputs = _inputs(bounds, 0, amplitude_query=query)
        output = model(inputs, training=False)
        assert np.isfinite(output["proposal_search_yield_logit"].numpy()).all()


def test_v5_safe_serialization_preserves_predictions_and_contract_stamp(tmp_path):
    tf.keras.utils.set_random_seed(37)
    model = validate_model_v5_graph_contract(_small_model())
    inputs = _inputs((_sphere_bounds(),), 0, resolution_bounds=_resolution_bounds())
    expected = model(inputs, training=False)
    path = tmp_path / "branch-conditioned-v5.keras"
    model.save(path)

    loaded = tf.keras.models.load_model(path, safe_mode=True)
    validate_model_v5_graph_contract(loaded)
    actual = loaded(inputs, training=False)
    for name in expected:
        np.testing.assert_allclose(expected[name], actual[name], rtol=0.0, atol=0.0)
    stamp = loaded.get_layer("v5_contract_stamp")
    assert isinstance(stamp, V5ContractStamp)
    assert stamp.schema_version == MODEL_V5_SCHEMA
    assert stamp.model_version == MODEL_V5_VERSION


def test_invalid_masks_and_wire_branch_fail_closed():
    model = _small_model()
    bounds = (_sphere_bounds(),)

    unavailable_active = _inputs(bounds, 1)
    unavailable_active["geometry_bounds_embedding"][0, 3 * 4 : 3 * 4 + 3] = (
        0.5,
        0.5,
        0.0,
    )
    unavailable_active["geometry_bounds_embedding"][0, 3 * 5 : 3 * 5 + 3] = (
        0.5,
        0.5,
        0.0,
    )
    unavailable_active["available_dimension_mask"][0, [4, 5]] = 0.0
    with pytest.raises(tf.errors.InvalidArgumentError, match="must be available"):
        model(unavailable_active, training=False)

    varying_inactive = _inputs(bounds, 0)
    varying_inactive["varying_dimension_mask"][0, 4] = 1.0
    with pytest.raises(tf.errors.InvalidArgumentError, match="must be active"):
        model(varying_inactive, training=False)

    invalid_wire = _inputs(bounds, 0)
    invalid_wire["branch_pattern_id"][0, 0] = 2
    with pytest.raises(tf.errors.InvalidArgumentError, match="invalid"):
        model(invalid_wire, training=False)

    invalid_topology = _inputs(bounds, 0)
    invalid_topology["branch_topology_id"][0, 0] = 34
    with pytest.raises(tf.errors.InvalidArgumentError, match=r"\[0, 33\]"):
        model(invalid_topology, training=False)

    inconsistent_presence = _inputs(bounds, 0)
    inconsistent_presence["available_dimension_mask"][0, 4] = 0.0
    with pytest.raises(tf.errors.InvalidArgumentError, match="presence"):
        model(inconsistent_presence, training=False)

    invalid_uncertainty = _inputs(bounds, 0)
    invalid_uncertainty["uncertainty_provenance"][0] = (1.0, 1.0, 0.0)
    with pytest.raises(tf.errors.InvalidArgumentError, match="one-hot"):
        model(invalid_uncertainty, training=False)

    invalid_amplitude_presence = _inputs(bounds, 0)
    invalid_amplitude_presence["amplitude_bounds_embedding"][0, 3 * 3 : 3 * 3 + 3] = (
        0.0,
        1.0,
        1.0,
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="presence conflicts"):
        model(invalid_amplitude_presence, training=False)


def test_legacy_graph_is_rejected_by_v5_graph_validator():
    value = tf.keras.Input((1,), name="legacy_input")
    legacy = tf.keras.Model(value, tf.keras.layers.Dense(1)(value), name="legacy_v3")
    with pytest.raises(ValueError, match="V5 name"):
        validate_model_v5_graph_contract(legacy)
