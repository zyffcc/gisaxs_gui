from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_DOWN, getcontext
from hashlib import sha256
import json
import math

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    branch_codec,
    sobol_geometry_recipe_v5,
    sobol_numeric_canonicalization_v5,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    NUM_TOPOLOGIES,
    TOPOLOGIES,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_amplitude_recipe_v5 import (
    direct_v5_amplitude_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    V5DesignPoint,
    V5SobolDesign,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_geometry_recipe_v5 import (
    direct_v5_geometry_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_DECIMAL_PRECISION,
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    V5_FAST_NUMERIC_POLICY_VERSION,
    v5_numeric_ops,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_physics_v5 import (
    direct_v5_physics_from_sobol,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_DIM,
    V5SobolCoordinateReader,
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_universal_query_design_v5 import (
    V5SobolUniversalTopologyQueryDesign,
    materialize_v5_sobol_universal_topology_query_design,
)


def _point(
    *,
    value: float = 0.371,
    topology_id: int = 12,
    index: int = 7,
    split: str = "train",
    ood_label: str | None = None,
) -> V5DesignPoint:
    coordinates = [value] * V5_SOBOL_RECIPE_DIM
    coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]] = (
        topology_id + 0.5
    ) / NUM_TOPOLOGIES
    return V5DesignPoint(
        sobol_index=index,
        assigned_split=split,
        ood_label=ood_label,
        clean_group_id=sha256(f"group:{index}".encode()).hexdigest(),
        unit_coordinates=tuple(coordinates),
    )


def _design():
    return v5_sobol_recipe_design(scramble_seed=20260903)


def _by_topology(artifact):
    return {value.topology_id: value for value in artifact.topology_queries}


def test_all_34_queries_replay_and_generating_pair_is_the_original_direct_pair():
    point = _point()
    artifact = materialize_v5_sobol_universal_topology_query_design(
        point,
        _design(),
        selected_topology_ids=tuple(reversed(range(NUM_TOPOLOGIES))),
    )
    replay = V5SobolUniversalTopologyQueryDesign.from_json(artifact.to_json())

    assert replay == artifact
    assert artifact.selected_topology_ids == tuple(range(NUM_TOPOLOGIES))
    assert tuple(value.topology for value in artifact.topology_queries) == TOPOLOGIES
    assert artifact.generating_topology_id == 12
    assert sha256(artifact.canonical_json.encode()).hexdigest() == artifact.sha256

    reader = V5SobolCoordinateReader.create(point.unit_coordinates)
    original_geometry = direct_v5_geometry_query(reader, sobol_index=point.sobol_index)
    original_amplitude, _ = direct_v5_amplitude_query(reader, original_geometry)
    generating = _by_topology(artifact)[artifact.generating_topology_id]
    assert generating.geometry == original_geometry
    assert generating.amplitude == original_amplitude
    # Snapshot the conservative geometry-only identity.  It includes the
    # versioned boundary-first, interior-fallback branch codec contract.
    assert original_geometry.sha256 == (
        "a4df8533438f144fe9689bfe310ace9c615dfd5e163e3657f45fc35f31b77fce"
    )
    assert original_amplitude.sha256 == (
        "88623ec2e103962e323dda20c9cad50f2e24c5f857289f1d8ec3d911784ecae4"
    )

    payload = json.loads(artifact.canonical_json)
    assert payload["source"]["sobol_design_sha256"] == _design().sha256
    assert len(payload["source"]["design_point_sha256"]) == 64
    assert len(payload["source"]["unit_coordinates_sha256"]) == 64
    assert payload["selected_topology_ids"] == list(range(NUM_TOPOLOGIES))
    assert payload["selected_topology_count"] == NUM_TOPOLOGIES
    assert payload["range_policy"][
        "same_exact_168d_named_point_reused_for_every_selected_topology"
    ]
    assert "each_BG_k_Int_i_and_int_Res_axis" in payload["range_policy"][
        "amplitude_axis_rule"
    ]
    assert not payload["range_policy"]["topology_queries_are_statistically_independent"]
    assert not payload["range_policy"]["generating_parameters_condition_alternative_ranges"]
    assert not payload["range_policy"]["curve_or_search_result_conditions_range_generation"]
    generating_payload = payload["topology_queries"][artifact.generating_topology_id]
    assert len(generating_payload["amplitude_range_regimes_sha256"]) == 64
    assert generating_payload["amplitude_range_regimes"]["normalization"] == "none"


def test_direct_sobol_decimal_policy_precedes_platform_libm_and_ignores_ambient_context():
    numeric = v5_numeric_ops(V5_DETERMINISTIC_NUMERIC_POLICY_VERSION)
    original_precision = getcontext().prec
    original_rounding = getcontext().rounding
    try:
        getcontext().prec = 7
        getcontext().rounding = ROUND_DOWN
        first = numeric.exp(-6.035015354914252)
        getcontext().prec = 53
        second = numeric.exp(-6.035015354914252)
    finally:
        getcontext().prec = original_precision
        getcontext().rounding = original_rounding

    assert V5_DETERMINISTIC_DECIMAL_PRECISION == 80
    assert first == second == 0.0023934597756974873
    assert first.hex() == "0x1.39b72eebfcaf3p-9"


@pytest.mark.parametrize(
    "value",
    (
        -3.0,
        -2.0,
        -1.1,
        -0.5,
        0.0,
        0.5,
        float(np.nextafter(0.5, np.inf)),
        1.000001,
        1.01,
        1.1,
        2.0,
        float(np.nextafter(2.0, np.inf)),
        3.0,
    ),
)
def test_deterministic_atan_range_reduction_covers_every_branch(value):
    result = v5_numeric_ops(V5_DETERMINISTIC_NUMERIC_POLICY_VERSION).atan(value)

    assert math.isfinite(result)
    assert result == pytest.approx(math.atan(value), rel=0.0, abs=2.0e-15)


def test_entire_direct_physics_never_calls_fast_transcendental_backend(monkeypatch):
    point = _point()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct deterministic path called platform libm")

    monkeypatch.setattr(sobol_numeric_canonicalization_v5.math, "log", forbidden)
    monkeypatch.setattr(sobol_numeric_canonicalization_v5.math, "exp", forbidden)
    monkeypatch.setattr(sobol_numeric_canonicalization_v5.math, "sqrt", forbidden)
    monkeypatch.setattr(sobol_numeric_canonicalization_v5.math, "hypot", forbidden)
    monkeypatch.setattr(sobol_numeric_canonicalization_v5.math, "atan", forbidden)
    monkeypatch.setattr(branch_codec.np, "log", forbidden)
    monkeypatch.setattr(branch_codec.np, "exp", forbidden)
    monkeypatch.setattr(branch_codec.np, "sqrt", forbidden)
    monkeypatch.setattr(sobol_geometry_recipe_v5.np, "log", forbidden)
    monkeypatch.setattr(sobol_geometry_recipe_v5.np, "exp", forbidden)

    physics = direct_v5_physics_from_sobol(
        point.unit_coordinates,
        sobol_index=point.sobol_index,
    )
    assert physics.query.numeric_policy_version == V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
    assert physics.amplitude_query.numeric_policy_version == (
        V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
    )
    embedding = physics.amplitude_query.model_embedding(1.0e4)
    assert len(embedding) == 21


def test_direct_target_rejects_numeric_policy_and_noncanonical_truth_tampering():
    point = _point(value=0.613)
    physics = direct_v5_physics_from_sobol(
        point.unit_coordinates,
        sobol_index=point.sobol_index,
    )
    with pytest.raises(ValueError, match="numeric policy"):
        replace(
            physics.target,
            physical_numeric_policy_version=V5_FAST_NUMERIC_POLICY_VERSION,
        )
    original = physics.target.truth_components[0]
    tampered = replace(original, R=float(np.nextafter(original.R, np.inf)))
    with pytest.raises(ValueError, match="physical truth"):
        replace(
            physics.target,
            truth_components=(tampered, *physics.target.truth_components[1:]),
        )


def test_shared_slot_coordinates_have_explicit_correlated_range_semantics():
    generating_id = topology_id_for(("Sphere", "Cylinder"))
    same_k_id = topology_id_for(("Cylinder", "Vertical Cylinder"))
    different_k_id = topology_id_for(("Sphere",))
    artifact = materialize_v5_sobol_universal_topology_query_design(
        _point(topology_id=generating_id, value=0.613),
        _design(),
        selected_topology_ids=(different_k_id, generating_id, same_k_id),
    )
    entries = _by_topology(artifact)

    generating = entries[generating_id]
    same_k = entries[same_k_id]
    assert generating.amplitude == same_k.amplitude
    assert generating.geometry.resolution_presence_policy == (
        same_k.geometry.resolution_presence_policy
    )
    assert generating.geometry.axis_designs[0:2] == same_k.geometry.axis_designs[0:2]

    payloads = {
        value["topology_id"]: value
        for value in json.loads(artifact.canonical_json)["topology_queries"]
    }
    generating_usage = payloads[generating_id]
    alternative_usage = payloads[same_k_id]
    assert "discrete.topology" in generating_usage["used_coordinate_names"]
    assert "discrete.topology" in alternative_usage["inactive_coordinate_names"]
    assert set(alternative_usage["used_coordinate_names"]).isdisjoint(
        alternative_usage["inactive_coordinate_names"]
    )
    assert set(alternative_usage["used_coordinate_names"]) | set(
        alternative_usage["inactive_coordinate_names"]
    ) == set(V5_SOBOL_RECIPE_COORDINATE_NAMES)


def test_explicit_subsets_are_order_independent_and_preserve_common_queries():
    point = _point(topology_id=12, value=0.217)
    first = materialize_v5_sobol_universal_topology_query_design(
        point,
        _design(),
        selected_topology_ids=(18, 12, 3),
    )
    reordered = materialize_v5_sobol_universal_topology_query_design(
        point,
        _design(),
        selected_topology_ids=(3, 18, 12),
    )
    second = materialize_v5_sobol_universal_topology_query_design(
        point,
        _design(),
        selected_topology_ids=(25, 12, 18),
    )

    assert first == reordered
    assert first.sha256 != second.sha256
    assert _by_topology(first)[12] == _by_topology(second)[12]
    assert _by_topology(first)[18] == _by_topology(second)[18]

    with pytest.raises(ValueError, match="include the generating topology"):
        materialize_v5_sobol_universal_topology_query_design(
            point,
            _design(),
            selected_topology_ids=(3, 18),
        )
    with pytest.raises(ValueError, match="unique"):
        materialize_v5_sobol_universal_topology_query_design(
            point,
            _design(),
            selected_topology_ids=(12, 12),
        )
    with pytest.raises(TypeError, match="integer"):
        materialize_v5_sobol_universal_topology_query_design(
            point,
            _design(),
            selected_topology_ids=(12, True),
        )


def test_universal_query_design_does_not_invoke_a_seed_bridge_or_PRNG(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("universal topology-query design invoked a PRNG")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)

    artifact = materialize_v5_sobol_universal_topology_query_design(
        _point(value=0.427),
        _design(),
        selected_topology_ids=tuple(range(NUM_TOPOLOGIES)),
    )

    assert len(artifact.topology_queries) == NUM_TOPOLOGIES
    assert "direct_named_transforms_no_seed_bridge" in artifact.canonical_json


def test_object_and_serialized_tampering_fail_closed():
    artifact = materialize_v5_sobol_universal_topology_query_design(
        _point(),
        _design(),
        selected_topology_ids=(3, 12, 18),
    )

    with pytest.raises(ValueError, match="does not replay"):
        replace(artifact, sha256="0" * 64)
    with pytest.raises(ValueError, match="does not replay"):
        replace(artifact, topology_queries=artifact.topology_queries[:-1])
    changed_coordinates = list(artifact.unit_coordinates)
    changed_coordinates[4] = 0.911
    with pytest.raises(ValueError, match="does not replay"):
        replace(artifact, unit_coordinates=tuple(changed_coordinates))

    encoded = json.loads(artifact.to_json())
    encoded["topology_queries"][0]["geometry_query_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not reproduce"):
        V5SobolUniversalTopologyQueryDesign.from_json(json.dumps(encoded))

    duplicate = artifact.to_json().replace(
        '"schema":',
        '"schema":"duplicate","schema":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        V5SobolUniversalTopologyQueryDesign.from_json(duplicate)

    with pytest.raises(ValueError, match="unsupported universal-query design version"):
        replace(
            artifact,
            version=(
                "posterior_v8_same_named_point_independent_gui_int_cross_topology_queries_v2"
            ),
        )

    legacy = json.loads(artifact.to_json())
    legacy["schema"] = "gisaxs.posterior_v8.direct_sobol_universal_topology_query_design/v2"
    with pytest.raises(ValueError, match="does not reproduce"):
        V5SobolUniversalTopologyQueryDesign.from_json(json.dumps(legacy))


def test_wrong_design_and_unregistered_ood_point_fail_closed():
    wrong_design = V5SobolDesign(coordinate_names=("wrong",), scramble_seed=17)
    with pytest.raises(ValueError, match="frozen direct-recipe"):
        materialize_v5_sobol_universal_topology_query_design(
            _point(),
            wrong_design,
            selected_topology_ids=(12,),
        )

    legacy_coordinate_design = V5SobolDesign(
        coordinate_names=V5_SOBOL_RECIPE_COORDINATE_NAMES,
        scramble_seed=20260903,
        coordinate_contract_sha256=(
            "90392c3f2754d22ea427c9d25ddd4f80fe35029411602aa5d2be966c52dfb408"
        ),
    )
    with pytest.raises(ValueError, match="coordinate contract hash"):
        materialize_v5_sobol_universal_topology_query_design(
            _point(),
            legacy_coordinate_design,
            selected_topology_ids=(12,),
        )

    with pytest.raises(ValueError, match="fail-closed"):
        materialize_v5_sobol_universal_topology_query_design(
            _point(split="ood", ood_label="topology_holdout"),
            _design(),
            selected_topology_ids=(12,),
        )
