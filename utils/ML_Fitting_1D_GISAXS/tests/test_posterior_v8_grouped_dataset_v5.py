from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    V5_ORACLE_PROTOCOL_SHA256,
    V5GroupedRecipeSpec,
    build_tiny_v5_grouped_dataset,
    build_v5_grouped_solution_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    SEARCH_OUTCOME_CODE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    NUM_TOPOLOGIES,
    SPHERE,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_amplitude_join_v5 import (
    amplitude_query_from_json,
    joined_amplitude_embeddings,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    V5_GROUPED_DATASET_SCHEMA,
    V5GroupedDataset,
    candidate_context_array,
    candidate_input,
    candidate_label,
    clean_array,
    observation_array,
    observation_input,
    read_v5_grouped_dataset,
    write_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import MODEL_V5_INPUT_KEYS
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    V5DesignPoint,
    materialize_v5_design_points,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_DIM,
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_v5 import (
    V5_SOBOL_CLEAN_RECIPE_SCHEMA,
    V5_SOBOL_CLEAN_RECIPE_VERSION,
    materialize_v5_sobol_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import sample_v5_clean_recipe
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_contract_v5 import (
    V5TopologyQuery,
)


def _full_dataset():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903, pattern_id=0)
    spec = V5GroupedRecipeSpec(
        recipe=recipe,
        split_id="train",
        view_indices=(0, 1),
    )
    return recipe, build_v5_grouped_solution_dataset(
        (spec,),
        dataset_id="grouped-test",
        generating_only=False,
    )


def _exchangeable_two_sphere_direct_recipe():
    coordinates = np.zeros(V5_SOBOL_RECIPE_DIM, dtype=np.float64)
    topology_id = topology_id_for((SPHERE, SPHERE))
    coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]] = (
        topology_id + 0.5
    ) / NUM_TOPOLOGIES
    for slot in (1, 2):
        coordinates[
            V5_SOBOL_RECIPE_COORDINATE_INDEX[f"geometry.slot_{slot}.D_policy"]
        ] = 0.5
        coordinates[
            V5_SOBOL_RECIPE_COORDINATE_INDEX[
                f"amplitude.composition.Int_fraction_{slot}"
            ]
        ] = 0.5
    for name in ("amplitude.composition.k", "amplitude.composition.BG"):
        coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX[name]] = 0.5
    point = V5DesignPoint(
        sobol_index=41,
        assigned_split="train",
        ood_label=None,
        clean_group_id=sha256(b"complete-slot-contract-group").hexdigest(),
        unit_coordinates=tuple(coordinates),
    )
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    recipe = materialize_v5_sobol_clean_recipe(point, design)
    return point, design, recipe


def test_grouped_builder_replays_and_never_duplicates_curve_per_candidate():
    recipe, first = _full_dataset()
    _, replay = _full_dataset()

    assert first.manifest["manifest_sha256"] == replay.manifest["manifest_sha256"]
    assert set(first.arrays) == set(replay.arrays)
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], replay.arrays[name])
    assert first.recipe_count == 1
    assert first.observation_count == 2
    assert first.candidate_count == len(recipe.query.feasible_wire_pattern_ids) > 1
    assert first.joined_count == first.observation_count * first.candidate_count
    assert first.arrays[observation_input("x")].shape[0] == first.observation_count
    assert not any("candidate_context__input__x" == name for name in first.arrays)
    assert not any("input__amplitude_bounds_embedding" in name for name in first.arrays)
    assert clean_array("amplitude_query_canonical_json") in first.arrays
    assert observation_array("intensity_reference") in first.arrays


def test_non_generating_rows_use_paired_slot_equivalence_not_geometry_only_catalog():
    point, design, recipe = _exchangeable_two_sphere_direct_recipe()
    paired = V5TopologyQuery(recipe.query, recipe.amplitude_query)
    assert len(recipe.query.feasible_wire_pattern_ids) == 4
    assert paired.feasible_wire_pattern_ids == (0, 2, 3)

    spec = V5GroupedRecipeSpec.from_design_point(
        recipe,
        point,
        split_plan_sha256=sha256(b"test-split-plan").hexdigest(),
        sobol_design_sha256=design.sha256,
        view_indices=(0,),
    )
    dataset = build_v5_grouped_solution_dataset(
        (spec,),
        dataset_id="paired-slot-equivalence",
        generating_only=False,
    )

    assert dataset.candidate_count == len(paired.feasible_wire_pattern_ids) == 3
    assert tuple(dataset.arrays[candidate_input("branch_pattern_id")].reshape(-1)) == (
        paired.feasible_wire_pattern_ids
    )


def test_join_has_exact_model_and_label_keys_dtypes_shapes_and_derived_amplitude():
    recipe, dataset = _full_dataset()
    inputs, labels = dataset.joined_numpy()

    assert tuple(inputs) == MODEL_V5_INPUT_KEYS
    assert tuple(labels) == CANDIDATE_SUPERVISION_TENSOR_KEYS
    count = dataset.joined_count
    assert inputs["x"].shape == (count, 1000, 3)
    assert inputs["point_mask"].shape == (count, 1000)
    assert inputs["amplitude_bounds_embedding"].shape == (count, 21)
    assert inputs["x"].dtype == np.float32
    assert inputs["point_mask"].dtype == np.bool_
    assert inputs["branch_pattern_id"].dtype == np.int32
    assert labels["target_local"].dtype == np.float32
    assert labels["search_outcome_code"].dtype == np.int32
    assert labels["search_artifact_id"].dtype.kind == "U"

    observed, candidates = dataset.join_indices()
    expected = np.asarray(
        [
            recipe.amplitude_query.model_embedding(
                dataset.arrays[observation_array("intensity_reference")][index]
            )
            for index in observed
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(inputs["amplitude_bounds_embedding"], expected)
    assert np.array_equal(
        dataset.arrays[observation_array("recipe_index")][observed],
        dataset.arrays[candidate_context_array("recipe_index")][candidates],
    )


def test_generating_truth_is_one_call_positive_and_all_unsearched_branches_are_masked():
    recipe, dataset = _full_dataset()
    outcomes = dataset.arrays[candidate_label("search_outcome_code")]
    positive = outcomes == SEARCH_OUTCOME_CODE["compatible_found"]
    unverified = outcomes == SEARCH_OUTCOME_CODE["unverified"]

    assert np.count_nonzero(positive) == 1
    assert np.count_nonzero(unverified) == dataset.candidate_count - 1
    assert np.all(dataset.arrays[candidate_label("search_completed")][positive])
    assert np.all(dataset.arrays[candidate_label("search_exact_forward_calls_used")][positive] == 1)
    assert np.all(
        dataset.arrays[candidate_label("search_protocol_sha256")][positive]
        == V5_ORACLE_PROTOCOL_SHA256
    )
    assert np.all(dataset.arrays[candidate_label("search_completed")][unverified] == 0)
    assert np.all(dataset.arrays[candidate_label("search_artifact_id")][unverified] == "")
    assert np.all(dataset.arrays[candidate_label("has_local_target")][unverified] == 0)
    assert np.all(dataset.arrays[candidate_label("generating_candidate_match")][unverified] == 0)
    np.testing.assert_array_equal(
        dataset.arrays[candidate_label("target_local")][positive][0],
        np.asarray(recipe.target.local_target_unit, dtype=np.float32),
    )
    candidate_index = int(np.flatnonzero(positive)[0])
    constraint_payload = json.loads(
        dataset.arrays[candidate_context_array("amplitude_constraint_json")][candidate_index]
    )
    expected_constraint = recipe.amplitude_query.constraint_for_branch(
        resolution_present=recipe.amplitude.resolution_present
    ).to_audit_dict()
    assert constraint_payload == expected_constraint


def test_amplitude_query_is_observation_independent_and_cross_parent_join_fails():
    first = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903, pattern_id=0)
    second = sample_v5_clean_recipe(("sphere",), recipe_seed=20260904, pattern_id=0)
    query = amplitude_query_from_json(
        first.amplitude_query.canonical_json,
        first.amplitude_query.sha256,
    )
    assert query == first.amplitude_query
    reference = np.asarray((10.0, 370.0), dtype=np.float64)
    original = query.model_embedding(reference[0])
    scaled = query.rescaled_intensity(37.0).model_embedding(reference[1])
    np.testing.assert_allclose(original, scaled, rtol=0.0, atol=2.0e-16)

    with pytest.raises(ValueError, match="crossed clean recipe"):
        joined_amplitude_embeddings(
            query_json=(
                first.amplitude_query.canonical_json,
                second.amplitude_query.canonical_json,
            ),
            query_sha256=(first.amplitude_query.sha256, second.amplitude_query.sha256),
            observation_intensity_reference=(10.0, 20.0),
            observation_recipe_index=(0, 1),
            candidate_recipe_index=(0, 1),
            observation_indices=np.asarray((0,), dtype=np.int32),
            candidate_indices=np.asarray((1,), dtype=np.int32),
        )


def test_artifact_round_trip_tamper_old_schema_and_no_overwrite(tmp_path):
    _, dataset = _full_dataset()
    path = tmp_path / "grouped-v5.gvd5"
    receipt = write_v5_grouped_dataset(dataset, path)
    loaded, loaded_receipt = read_v5_grouped_dataset(path)
    assert loaded.manifest == dataset.manifest
    assert loaded_receipt.artifact_sha256 == receipt.artifact_sha256
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_grouped_dataset(dataset, path)

    changed_arrays = dict(dataset.arrays)
    changed = np.array(changed_arrays[observation_input("x")], copy=True)
    changed[0, 0, 0] += 0.1
    changed_arrays[observation_input("x")] = changed
    with pytest.raises(ValueError, match="metadata or SHA-256"):
        V5GroupedDataset(dataset.manifest, changed_arrays)

    old = deepcopy(dict(dataset.manifest))
    old["dataset_schema"] = "gisaxs.posterior_v8.grouped_candidate_dataset/v0"
    core = dict(old)
    core.pop("manifest_sha256")
    old["manifest_sha256"] = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="unsupported V5 grouped-dataset schema"):
        V5GroupedDataset(old, dataset.arrays)


def test_self_consistent_manifest_hash_cannot_change_frozen_scientific_semantics():
    _, dataset = _full_dataset()
    mutations = (
        lambda value: value.__setitem__("stage", "different-stage"),
        lambda value: value["build_policy"].__setitem__("unsearched_branch_is_negative", True),
        lambda value: value["table_fields"]["clean"].remove("recipe_sha256"),
        lambda value: value["oracle_protocol"]["payload"].__setitem__(
            "negative_labels_produced", True
        ),
        lambda value: value["split_semantics"].__setitem__(
            "inheritance", "views_may_cross_parents"
        ),
    )
    for mutate in mutations:
        changed = deepcopy(dict(dataset.manifest))
        mutate(changed)
        core = dict(changed)
        core.pop("manifest_sha256")
        changed["manifest_sha256"] = sha256(canonical_json(core).encode("utf-8")).hexdigest()
        with pytest.raises(ValueError):
            V5GroupedDataset(changed, dataset.arrays)


def test_generating_only_tiny_fixture_is_direct_k1_input():
    dataset = build_tiny_v5_grouped_dataset()
    inputs, labels = dataset.joined_numpy(include_unverified=False)
    assert dataset.recipe_count == dataset.observation_count == dataset.candidate_count == 1
    assert dataset.joined_count == 1
    assert inputs["branch_pattern_id"].tolist() == [[0]]
    assert labels["has_local_target"].tolist() == [True]
    assert dataset.manifest["build_policy"]["generating_candidate_only"] is True
    source_hashes = dataset.manifest["source_sha256"]
    assert "utils/ML_Fitting_1D_GISAXS/PosteriorV8/amplitude_query_sampling_v5.py" in source_hashes
    assert "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_query_sampling_v5.py" in source_hashes
    assert "src/gimap/features/fitting/domain/scattering_model.py" in source_hashes
    assert "src/gimap/features/fitting/domain/physical_constraints.py" in source_hashes


def test_tf_data_index_join_matches_numpy_without_candidate_curve_storage():
    tf = pytest.importorskip("tensorflow")
    _, dataset = _full_dataset()
    expected_inputs, expected_labels = dataset.joined_numpy()
    tf_inputs, tf_labels = next(
        iter(dataset.as_tensorflow_dataset(batch_size=dataset.joined_count))
    )
    assert tuple(tf_inputs) == MODEL_V5_INPUT_KEYS
    assert tuple(tf_labels) == CANDIDATE_SUPERVISION_TENSOR_KEYS
    for name, expected in expected_inputs.items():
        np.testing.assert_array_equal(tf_inputs[name].numpy(), expected)
    for name, expected in expected_labels.items():
        actual = tf_labels[name].numpy()
        if expected.dtype.kind == "U":
            actual = np.asarray([value.decode("utf-8") for value in actual], dtype=np.str_)
        np.testing.assert_array_equal(actual, expected)


def test_sobol_parent_fields_are_all_or_none():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=1, pattern_id=0)
    with pytest.raises(ValueError, match="must be supplied together"):
        V5GroupedRecipeSpec(
            recipe=recipe,
            split_id="train",
            sobol_index=3,
            split_plan_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="audit hash"):
        V5GroupedRecipeSpec(recipe=replace(recipe, sha256="0" * 64), split_id="train")


def test_direct_sobol_recipe_builds_grouped_shard_with_frozen_parent_identity():
    counts = V5SplitCounts(
        train=1,
        tuning_validation=1,
        calibration=1,
        test=1,
        reference=1,
        ood_topology=1,
        ood_range_width=1,
        ood_weak_component=1,
        ood_acquisition_policy=1,
    )
    plan = V5SplitPlan.create(counts, guard_band=1)
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    point = materialize_v5_design_points(plan, design)[0]
    recipe = materialize_v5_sobol_clean_recipe(point, design)
    with pytest.raises(ValueError, match="complete Sobol parent provenance"):
        V5GroupedRecipeSpec(recipe=recipe, split_id=point.assigned_split)
    spec = V5GroupedRecipeSpec.from_design_point(
        recipe,
        point,
        split_plan_sha256=plan.sha256,
        sobol_design_sha256=design.sha256,
    )
    dataset = build_v5_grouped_solution_dataset(
        (spec,),
        dataset_id="direct-sobol-grouped-test",
        generating_only=True,
    )

    assert dataset.manifest["clean_recipe_identity"] == {
        "protocol_version": dataset.manifest["oracle_protocol"]["payload"][
            "clean_recipe_protocol_version"
        ],
        "schema_version": V5_SOBOL_CLEAN_RECIPE_SCHEMA,
        "generator_version": V5_SOBOL_CLEAN_RECIPE_VERSION,
    }
    assert dataset.arrays[clean_array("recipe_schema_version")].tolist() == [
        V5_SOBOL_CLEAN_RECIPE_SCHEMA
    ]
    assert dataset.arrays[clean_array("recipe_generator_version")].tolist() == [
        V5_SOBOL_CLEAN_RECIPE_VERSION
    ]
    assert dataset.arrays[clean_array("clean_group_id")].tolist() == [point.clean_group_id]
    assert dataset.arrays[clean_array("sobol_index")].tolist() == [point.sobol_index]
    assert dataset.arrays[clean_array("split_plan_sha256")].tolist() == [plan.sha256]
    assert dataset.arrays[clean_array("sobol_design_sha256")].tolist() == [design.sha256]
    inputs, labels = dataset.joined_numpy(include_unverified=False)
    assert tuple(inputs) == MODEL_V5_INPUT_KEYS
    assert labels["has_local_target"].tolist() == [True]

    seeded = sample_v5_clean_recipe(("sphere",), recipe_seed=11, pattern_id=0)
    with pytest.raises(ValueError, match="cannot mix clean recipe schemas"):
        build_v5_grouped_solution_dataset(
            (V5GroupedRecipeSpec(recipe=seeded, split_id="train"), spec),
            dataset_id="mixed-generator-shard",
            generating_only=True,
        )
