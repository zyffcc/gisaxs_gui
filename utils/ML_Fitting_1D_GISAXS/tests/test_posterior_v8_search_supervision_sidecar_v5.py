from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_v5_grouped_solution_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    SEARCH_OUTCOME_CODE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    ObservedCurve,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import (
    array_manifest,
    array_sha256,
    canonical_json,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    observation_array,
    read_v5_grouped_dataset,
    write_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import MODEL_V5_INPUT_KEYS
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    build_v5_observation_data_views,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5CompatibleRepresentativeReference,
    V5ExactSearchObservation,
    V5FrozenBranchSearchResult,
    V5FrozenExactSearchProtocol,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_overlay_v5 import (
    read_v5_search_supervision_overlay,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_sidecar_v5 import (
    BRANCH_MODEL_INPUT_KEYS,
    OBSERVATION_MODEL_INPUT_KEYS,
    V5SearchSupervisionSidecar,
    V5UniversalSearchSpec,
    branch_array,
    branch_input,
    branch_label,
    collect_v5_search_supervision_sidecar,
    query_array,
    read_v5_search_supervision_sidecar,
    write_v5_search_supervision_sidecar,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    search_supervision_sidecar_validation_v5 as sidecar_validation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    V5TopologyQuery,
    build_v5_universal_candidate_context,
)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _amplitude_query(query):
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=tuple(ClosedInterval(0.0, 1.0) for _ in query.topology),
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 1.0e8),
    )


def _protocol():
    return V5FrozenExactSearchProtocol(
        protocol_id="v5-sidecar-test-search/v1",
        evaluator_version="authoritative-forward-evaluator/v1",
        authoritative_forward_id="gui-empirical-forward/v1",
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        threshold_name="raw_curve_equivalence_logrmse",
        threshold_value=0.02,
        threshold_source_id="paper-protocol/test",
        missing_acceptance_sigma_metric_name=RAW_LOG_RMSE_METRIC,
        missing_acceptance_sigma_threshold_name="raw_curve_equivalence_logrmse",
        missing_acceptance_sigma_threshold_value=0.02,
        exact_forward_call_budget=8,
        seed_schedule_id="fixed-sobol-seeds/test",
        seed_schedule_sha256=_hash("fixed-sobol-seeds/test"),
        optimizer_schedule_id="bounded-refinement/test",
        optimizer_schedule_sha256=_hash("bounded-refinement/test"),
        termination_policy_id="positive-early-stop-or-exhaust/test",
        representative_distance_id="normalized-parameter-distance/test",
        representative_distance_sha256=_hash("normalized-parameter-distance/test"),
        delta_separation_threshold=0.08,
    )


def _parent_and_specs(
    tmp_path,
    *,
    views=(0, 1),
    split="train",
    seed=32001,
    competitor_topology=("cylinder",),
):
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=seed, pattern_id=0)
    dataset = build_v5_grouped_solution_dataset(
        (
            V5GroupedRecipeSpec(
                recipe=recipe,
                split_id=split,
                view_indices=tuple(views),
            ),
        ),
        dataset_id=f"search-sidecar-parent-{seed}-{split}",
        generating_only=False,
    )
    parent_path = tmp_path / f"parent-{seed}-{split}.gvd5"
    parent_receipt = write_v5_grouped_dataset(dataset, parent_path)
    competitor = full_range_v5_query(competitor_topology, query_seed=seed + 1000)
    entries = (
        V5TopologyQuery(recipe.query, recipe.amplitude_query),
        V5TopologyQuery(competitor, _amplitude_query(competitor)),
    )
    observations = build_v5_observation_data_views(
        recipe,
        tuple(views),
        split_id=split,
    )
    specs = tuple(
        V5UniversalSearchSpec(
            parent_observation_index=index,
            context=build_v5_universal_candidate_context(
                observation.preprocessed,
                observation.uncertainty,
                entries,
            ),
            exact_observation=V5ExactSearchObservation.from_observation_view(
                observation,
                curve_id=str(
                    dataset.arrays[observation_array("observation_id")][index]
                ),
            ),
            query_catalog_artifact_id=f"universal-query-catalog/{seed}",
            query_catalog_artifact_sha256=_hash(
                f"universal-query-catalog/{seed}"
            ),
        )
        for index, observation in enumerate(observations)
    )
    return parent_path, parent_receipt, specs


def _completed_runner(task):
    branch = task.branch
    identity = f"{task.observation_id}/{branch.global_key.wire_key}"
    common = {
        "universal_query_sha256": task.universal_context.audit_sha256,
        "exact_curve_sha256": task.exact_curve_sha256,
        "global_branch_key": branch.global_key,
        "context_sha256": branch.context_sha256,
        "completed": True,
        "executor_artifact_id": f"executor/{_hash(identity)[:16]}",
        "executor_artifact_sha256": _hash(f"executor/{identity}"),
    }
    if branch.global_index % 2:
        return V5FrozenBranchSearchResult(
            **common,
            outcome="no_compatible_found_within_frozen_search_budget",
            exact_forward_calls_used=8,
            termination_reason="exact_forward_budget_exhausted_without_compatible",
        )
    representative = V5CompatibleRepresentativeReference(
        artifact_id=f"exact/{_hash(identity)[:16]}",
        artifact_sha256=_hash(f"exact/{identity}"),
        representative_set_id=f"set/{_hash(identity)[:16]}",
        representative_set_sha256=_hash(f"set/{identity}"),
        cluster_id="cluster-0000",
        metric_value=0.01,
        bounds_passed=True,
        physics_passed=True,
        target_local=(0.5,) * 26,
    )
    return V5FrozenBranchSearchResult(
        **common,
        outcome="compatible_found",
        exact_forward_calls_used=8,
        termination_reason="frozen_full_budget_completed_with_compatible_representatives",
        representatives=(representative,),
    )


def _build_complete(tmp_path, *, views=(0, 1), seed=32001):
    parent_path, parent_receipt, specs = _parent_and_specs(
        tmp_path, views=views, seed=seed
    )
    calls = []

    def runner(task):
        calls.append(
            (
                task.universal_context.audit_sha256,
                task.branch.global_key.wire_key,
                task.branch.context_sha256,
            )
        )
        return _completed_runner(task)

    sidecar = collect_v5_search_supervision_sidecar(
        parent_path,
        specs,
        sidecar_id=f"sidecar-{seed}",
        protocol=_protocol(),
        runner=runner,
    )
    sidecar_path = tmp_path / f"sidecar-{seed}.gvd5"
    sidecar_receipt = write_v5_search_supervision_sidecar(sidecar, sidecar_path)
    return (
        parent_path,
        parent_receipt,
        sidecar_path,
        sidecar_receipt,
        sidecar,
        specs,
        calls,
    )


def _self_consistent_sidecar(sidecar, arrays, *, mutate_manifest=None):
    manifest = deepcopy(dict(sidecar.manifest))
    manifest["arrays"] = array_manifest(arrays)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    core = dict(manifest)
    core.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    return V5SearchSupervisionSidecar(manifest, arrays)


def test_full_catalog_sidecar_is_immutable_and_materializes_exact_training_join(tmp_path):
    (
        parent_path,
        parent_receipt,
        sidecar_path,
        sidecar_receipt,
        sidecar,
        specs,
        calls,
    ) = _build_complete(tmp_path)

    assert sidecar.query_count == len(specs) == 2
    assert sidecar.branch_count == sum(value.context.branch_count for value in specs)
    assert len(calls) == sidecar.branch_count == len(set(calls))
    assert sidecar.manifest["parent_grouped_artifact"]["artifact_sha256"] == (
        parent_receipt.artifact_sha256
    )
    assert sidecar.manifest["counts"]["unverified"] == 0
    assert sidecar.manifest["counts"]["compatible_found"] > 0
    assert sidecar.manifest["counts"]["completed_negative"] > 0
    source_paths = set(sidecar.manifest["source_sha256"])
    assert {
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/candidate_batch_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_artifact_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_dataset_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/grouped_amplitude_join_v5.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/exact_search_executor_v5.py",
        "src/gimap/features/fitting/domain/scattering_model.py",
        "src/gimap/features/fitting/domain/physical_constraints.py",
    } <= source_paths
    assert sidecar.manifest["table_fields"]["branch_inputs"] == list(
        BRANCH_MODEL_INPUT_KEYS
    )
    assert BRANCH_MODEL_INPUT_KEYS == tuple(
        name for name in MODEL_V5_INPUT_KEYS if name not in OBSERVATION_MODEL_INPUT_KEYS
    )
    assert all(
        branch_input(name) not in sidecar.arrays
        for name in OBSERVATION_MODEL_INPUT_KEYS
    )
    assert sidecar.manifest["build_policy"][
        "alternative_topology_queries_are_explicit_caller_supplied"
    ]
    assert sidecar.manifest["build_policy"][
        "exact_float64_intensity_reference_is_persisted_per_query"
    ]
    parent, _ = read_v5_grouped_dataset(parent_path)
    references = sidecar.arrays[query_array("intensity_reference")]
    parent_references = parent.arrays[observation_array("intensity_reference")]
    assert references.dtype == np.dtype(np.float64)
    assert references.tobytes(order="C") == parent_references.tobytes(order="C")
    for reference, digest in zip(
        references,
        sidecar.arrays[query_array("intensity_reference_sha256")],
    ):
        assert str(digest) == array_sha256(
            query_array("intensity_reference"),
            np.asarray(reference, dtype=np.float64),
        )
    for index, spec in enumerate(specs):
        curve = spec.exact_observation.observed_curve
        mask = sidecar.arrays[query_array("exact_curve_point_mask")][index]
        assert np.array_equal(sidecar.arrays[query_array("exact_curve_q")][index][mask], curve.q)
        assert np.array_equal(
            sidecar.arrays[query_array("exact_curve_intensity")][index][mask],
            curve.intensity,
        )
        catalog = json.loads(
            str(sidecar.arrays[query_array("topology_query_catalog_json")][index])
        )
        assert [entry["topology_id"] for entry in catalog] == list(
            spec.context.topology_ids
        )
        assert all(entry["geometry_query_canonical_json"] for entry in catalog)
        assert all(entry["amplitude_query_canonical_json"] for entry in catalog)

    loaded, loaded_receipt = read_v5_search_supervision_sidecar(sidecar_path)
    assert loaded_receipt.artifact_sha256 == sidecar_receipt.artifact_sha256
    assert loaded.manifest == sidecar.manifest
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_search_supervision_sidecar(sidecar, sidecar_path)

    overlay = read_v5_search_supervision_overlay(parent_path, sidecar_path)
    inputs, labels = overlay.joined_numpy()
    assert tuple(inputs) == MODEL_V5_INPUT_KEYS
    assert tuple(labels) == CANDIDATE_SUPERVISION_TENSOR_KEYS
    assert inputs["x"].shape == (sidecar.branch_count, 1000, 3)
    assert labels["search_outcome_code"].shape == (sidecar.branch_count,)
    assert overlay.protocol_sha256 == _protocol().sha256
    assert overlay.outcome_counts()["unverified"] == 0

    pytest.importorskip("tensorflow")
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective_v5 import (
        compute_v5_candidate_training_objective,
    )

    count = sidecar.branch_count
    outputs = {
        "proposal_search_yield_logit": np.zeros((count, 1), dtype=np.float32),
        "mixture_logits": np.zeros((count, 2), dtype=np.float32),
        "mixture_loc": np.zeros((count, 2, 26), dtype=np.float32),
        "mixture_logscale": np.zeros((count, 2, 26), dtype=np.float32),
    }
    metrics = compute_v5_candidate_training_objective(outputs, labels)
    assert np.isfinite(metrics["loss"].numpy())
    assert metrics["verified_count"].numpy() == sidecar.branch_count


def test_exchangeable_complete_slot_query_branch_rows_and_sidecar_replay_agree(tmp_path):
    parent_path, _, specs = _parent_and_specs(
        tmp_path,
        views=(0,),
        seed=32051,
        competitor_topology=("sphere", "sphere"),
    )
    context = specs[0].context
    repeated = next(value for value in context.topology_queries if len(value.topology) == 2)
    assert len(repeated.geometry.feasible_wire_pattern_ids) == 8
    assert repeated.feasible_wire_pattern_ids == (0, 2, 3, 16, 18, 19)
    expected_count = sum(
        len(value.feasible_wire_pattern_ids) for value in context.topology_queries
    )
    assert context.branch_count == expected_count

    sidecar = collect_v5_search_supervision_sidecar(
        parent_path,
        specs,
        sidecar_id="paired-slot-equivalence-sidecar",
        protocol=_protocol(),
        runner=_completed_runner,
    )
    assert sidecar.branch_count == expected_count
    assert int(sidecar.arrays[query_array("branch_count")][0]) == expected_count
    repeated_rows = np.flatnonzero(
        sidecar.arrays[branch_array("topology_id")] == repeated.topology_id
    )
    assert tuple(sidecar.arrays[branch_array("pattern_id")][repeated_rows]) == (
        repeated.feasible_wire_pattern_ids
    )

    path = tmp_path / "paired-slot-equivalence-sidecar.gvd5"
    receipt = write_v5_search_supervision_sidecar(sidecar, path)
    replay, replay_receipt = read_v5_search_supervision_sidecar(path)
    assert replay.branch_count == expected_count
    assert replay_receipt.artifact_sha256 == receipt.artifact_sha256


def test_unverified_partial_run_is_audited_but_default_training_fails_closed(tmp_path):
    parent_path, _, specs = _parent_and_specs(tmp_path, views=(0,), seed=32002)

    def runner(task):
        if task.branch.global_index:
            return _completed_runner(task)
        return V5FrozenBranchSearchResult(
            universal_query_sha256=task.universal_context.audit_sha256,
            exact_curve_sha256=task.exact_curve_sha256,
            global_branch_key=task.branch.global_key,
            context_sha256=task.branch.context_sha256,
            outcome="unverified",
            completed=False,
            exact_forward_calls_used=2,
            termination_reason="executor_failed_before_protocol_completion",
            executor_artifact_id="failed-run/audit",
            executor_artifact_sha256=_hash("failed-run/audit"),
        )

    sidecar = collect_v5_search_supervision_sidecar(
        parent_path,
        specs,
        sidecar_id="partial-sidecar",
        protocol=_protocol(),
        runner=runner,
    )
    path = tmp_path / "partial-sidecar.gvd5"
    write_v5_search_supervision_sidecar(sidecar, path)
    overlay = read_v5_search_supervision_overlay(parent_path, path)
    assert sidecar.manifest["counts"]["unverified"] == 1
    unverified = (
        sidecar.arrays[branch_label("search_outcome_code")]
        == SEARCH_OUTCOME_CODE["unverified"]
    )
    assert sidecar.arrays[branch_array("exact_forward_calls_used")][unverified].tolist() == [2]
    assert sidecar.arrays[branch_label("search_artifact_id")][unverified].tolist() == [""]
    with pytest.raises(ValueError, match="completed branch catalog"):
        overlay.joined_numpy()
    _, labels = overlay.joined_numpy(
        require_completed_catalog=False,
        require_positive_and_negative=False,
    )
    assert np.count_nonzero(
        labels["search_outcome_code"] == SEARCH_OUTCOME_CODE["unverified"]
    ) == 1


def test_multiple_delta_separated_local_targets_expand_without_overweighting_branch(tmp_path):
    parent_path, _, specs = _parent_and_specs(tmp_path, views=(0,), seed=32010)

    def runner(task):
        result = _completed_runner(task)
        if result.outcome != "compatible_found":
            return result
        first = result.representatives[0]
        target = np.full(26, 0.5, dtype=np.float64)
        varying = np.asarray(task.branch.condition.varying_dimension_mask, dtype=np.bool_)
        assert np.any(varying)
        target[varying] = 0.6
        second = V5CompatibleRepresentativeReference(
            artifact_id=f"{first.artifact_id}/second",
            artifact_sha256=_hash(f"{first.artifact_sha256}/second"),
            representative_set_id=first.representative_set_id,
            representative_set_sha256=first.representative_set_sha256,
            cluster_id="cluster-0001",
            metric_value=0.015,
            bounds_passed=True,
            physics_passed=True,
            target_local=tuple(target),
        )
        return replace(result, representatives=(first, second))

    sidecar = collect_v5_search_supervision_sidecar(
        parent_path,
        specs,
        sidecar_id="multi-target-sidecar",
        protocol=_protocol(),
        runner=runner,
    )
    sidecar_path = tmp_path / "multi-target-sidecar.gvd5"
    write_v5_search_supervision_sidecar(sidecar, sidecar_path)
    overlay = read_v5_search_supervision_overlay(parent_path, sidecar_path)
    _, labels = overlay.joined_numpy()
    positive_branch_count = sidecar.manifest["counts"]["compatible_found"]
    assert labels["search_outcome_code"].size == sidecar.branch_count + positive_branch_count
    assert np.count_nonzero(labels["has_local_target"]) == 2 * positive_branch_count
    for artifact_id in np.unique(labels["search_artifact_id"]):
        selected = labels["search_artifact_id"] == artifact_id
        assert np.sum(labels["candidate_weight"][selected]) == pytest.approx(1.0)


def test_negative_requires_complete_frozen_budget_and_branch_identity(tmp_path):
    parent_path, _, specs = _parent_and_specs(tmp_path, views=(0,), seed=32003)

    def short_negative(task):
        branch = task.branch
        return V5FrozenBranchSearchResult(
            universal_query_sha256=task.universal_context.audit_sha256,
            exact_curve_sha256=task.exact_curve_sha256,
            global_branch_key=branch.global_key,
            context_sha256=branch.context_sha256,
            outcome="no_compatible_found_within_frozen_search_budget",
            completed=True,
            exact_forward_calls_used=7,
            termination_reason="exact_forward_budget_exhausted_without_compatible",
            executor_artifact_id="short-negative",
            executor_artifact_sha256=_hash("short-negative"),
        )

    with pytest.raises(ValueError, match="did not consume"):
        collect_v5_search_supervision_sidecar(
            parent_path,
            specs,
            sidecar_id="invalid-short-negative",
            protocol=_protocol(),
            runner=short_negative,
        )

    def swapped_branch(task):
        result = _completed_runner(task)
        other = task.universal_context.branches[-1].global_key
        if other == task.branch.global_key:
            other = task.universal_context.branches[0].global_key
        return replace(result, global_branch_key=other)

    with pytest.raises(ValueError, match="different query/branch/context"):
        collect_v5_search_supervision_sidecar(
            parent_path,
            specs,
            sidecar_id="invalid-swapped-branch",
            protocol=_protocol(),
            runner=swapped_branch,
        )


def test_positive_also_requires_complete_equal_branch_budget(tmp_path):
    parent_path, _, specs = _parent_and_specs(tmp_path, views=(0,), seed=32004)

    def short_positive(task):
        result = _completed_runner(task)
        if result.outcome != "compatible_found":
            return result
        return replace(
            result,
            exact_forward_calls_used=7,
            termination_reason="compatible_representative_found_early",
        )

    with pytest.raises(ValueError, match="equal per-branch budget"):
        collect_v5_search_supervision_sidecar(
            parent_path,
            specs,
            sidecar_id="invalid-short-positive",
            protocol=_protocol(),
            runner=short_positive,
        )


@pytest.mark.parametrize(
    "input_name",
    ("geometry_bounds_embedding", "amplitude_bounds_embedding"),
)
def test_alternative_topology_embedding_must_reproduce_canonical_query(
    tmp_path, input_name
):
    *_, sidecar, specs, _ = _build_complete(tmp_path, views=(0,), seed=32014)
    generating_topology_id = specs[0].context.topology_queries[0].topology_id
    alternative_rows = np.flatnonzero(
        sidecar.arrays[branch_array("topology_id")] != generating_topology_id
    )
    assert alternative_rows.size
    changed_arrays = dict(sidecar.arrays)
    changed = np.array(changed_arrays[branch_input(input_name)], copy=True)
    changed[alternative_rows[0]] = np.float32(1.0e30)
    changed_arrays[branch_input(input_name)] = changed
    with pytest.raises(ValueError, match="amplitude bounds embedding|branch model input"):
        _self_consistent_sidecar(sidecar, changed_arrays)


def test_standalone_replay_uses_persisted_binary64_reference_without_inverse_libm(
    tmp_path, monkeypatch
):
    *_, sidecar, _, _ = _build_complete(tmp_path, views=(0,), seed=32015)

    class ForbiddenInverseLibm:
        def __getattr__(self, name):
            raise AssertionError(f"validator attempted inverse libm call: {name}")

    monkeypatch.setattr(
        sidecar_validation,
        "math",
        ForbiddenInverseLibm(),
        raising=False,
    )
    replay = V5SearchSupervisionSidecar(
        dict(sidecar.manifest),
        dict(sidecar.arrays),
    )
    assert replay.manifest == sidecar.manifest


def test_intensity_reference_digest_and_semantic_tamper_fail_closed(tmp_path):
    *_, sidecar, _, _ = _build_complete(tmp_path, views=(0,), seed=32016)
    reference_name = query_array("intensity_reference")
    digest_name = query_array("intensity_reference_sha256")

    changed_arrays = dict(sidecar.arrays)
    changed_references = np.array(changed_arrays[reference_name], copy=True)
    changed_references[0] *= 2.0
    changed_arrays[reference_name] = changed_references
    with pytest.raises(ValueError, match="intensity reference SHA-256"):
        _self_consistent_sidecar(sidecar, changed_arrays)

    changed_digests = np.array(changed_arrays[digest_name], copy=True)
    changed_digests[0] = array_sha256(
        reference_name,
        np.asarray(changed_references[0], dtype=np.float64),
    )
    changed_arrays[digest_name] = changed_digests
    with pytest.raises(ValueError, match="persisted intensity reference"):
        _self_consistent_sidecar(sidecar, changed_arrays)


def test_float64_neighbor_reference_tamper_fails_exact_parent_join(tmp_path):
    parent_path, *_, sidecar, _, _ = _build_complete(
        tmp_path,
        views=(0,),
        seed=32017,
    )
    reference_name = query_array("intensity_reference")
    digest_name = query_array("intensity_reference_sha256")
    changed_arrays = dict(sidecar.arrays)
    changed_references = np.array(changed_arrays[reference_name], copy=True)
    changed_references[0] = np.nextafter(changed_references[0], np.float64(np.inf))
    assert changed_references[0] != sidecar.arrays[reference_name][0]
    changed_arrays[reference_name] = changed_references
    changed_digests = np.array(changed_arrays[digest_name], copy=True)
    changed_digests[0] = array_sha256(
        reference_name,
        np.asarray(changed_references[0], dtype=np.float64),
    )
    changed_arrays[digest_name] = changed_digests
    changed_sidecar = _self_consistent_sidecar(sidecar, changed_arrays)
    changed_path = tmp_path / "float64-neighbor-reference-tamper.gvd5"
    write_v5_search_supervision_sidecar(changed_sidecar, changed_path)
    with pytest.raises(ValueError, match="byte-for-byte with grouped parent"):
        read_v5_search_supervision_overlay(parent_path, changed_path)


def test_missing_observation_partial_catalog_and_protocol_tamper_fail_closed(tmp_path):
    parent_path, _, specs = _parent_and_specs(tmp_path, views=(0, 1), seed=32004)
    with pytest.raises(ValueError, match="every parent observation exactly once"):
        collect_v5_search_supervision_sidecar(
            parent_path,
            specs[:1],
            sidecar_id="partial-observations",
            protocol=_protocol(),
            runner=_completed_runner,
        )

    *_, sidecar, _, _ = _build_complete(tmp_path, views=(0,), seed=32005)
    changed_arrays = dict(sidecar.arrays)
    branch_count = np.array(changed_arrays[query_array("branch_count")], copy=True)
    branch_count[0] -= 1
    changed_arrays[query_array("branch_count")] = branch_count
    with pytest.raises(ValueError, match="catalog is partial"):
        _self_consistent_sidecar(sidecar, changed_arrays)

    def change_protocol(manifest):
        changed = dict(manifest["protocol"])
        changed["threshold_value"] = 0.03
        manifest["protocol"] = changed
        manifest["protocol_sha256"] = V5FrozenExactSearchProtocol(**changed).sha256

    with pytest.raises(ValueError, match="protocol|search record"):
        _self_consistent_sidecar(sidecar, dict(sidecar.arrays), mutate_manifest=change_protocol)


def test_cross_split_and_wrong_parent_artifact_fail_closed(tmp_path):
    tuning_path, _, tuning_specs = _parent_and_specs(
        tmp_path,
        views=(0,),
        split="tuning_validation",
        seed=32009,
    )
    tuning = collect_v5_search_supervision_sidecar(
        tuning_path,
        tuning_specs,
        sidecar_id="tuning-sidecar",
        protocol=_protocol(),
        runner=_completed_runner,
    )
    assert tuning.manifest["split_id"] == "tuning_validation"
    tuning_sidecar_path = tmp_path / "tuning-sidecar.gvd5"
    write_v5_search_supervision_sidecar(tuning, tuning_sidecar_path)
    tuning_overlay = read_v5_search_supervision_overlay(
        tuning_path, tuning_sidecar_path
    )
    assert tuning_overlay.sidecar.query_count == 1

    wrong_split_path, _, wrong_split_specs = _parent_and_specs(
        tmp_path,
        views=(0,),
        split="calibration",
        seed=32006,
    )
    with pytest.raises(ValueError, match="model-development-split only"):
        collect_v5_search_supervision_sidecar(
            wrong_split_path,
            wrong_split_specs,
            sidecar_id="wrong-split",
            protocol=_protocol(),
            runner=_completed_runner,
        )

    parent_path, _, sidecar_path, *_ = _build_complete(
        tmp_path, views=(0,), seed=32007
    )
    different_parent, _, _ = _parent_and_specs(tmp_path, views=(0,), seed=32008)
    assert parent_path != different_parent
    with pytest.raises(ValueError, match="different grouped parent"):
        read_v5_search_supervision_overlay(different_parent, sidecar_path)


def test_exact_curve_is_explicit_acceptance_evidence_and_tamper_fails_closed(tmp_path):
    parent_path, _, specs = _parent_and_specs(
        tmp_path, views=(0, 1), seed=32011
    )
    with_sigma = next(
        value
        for value in specs
        if value.exact_observation.observed_curve.sigma_log is not None
    )
    without_sigma = next(
        value
        for value in specs
        if value.exact_observation.observed_curve.sigma_log is None
    )

    missing = without_sigma.exact_observation
    q, intensity, encoder_sigma = missing.observation_view.preprocessed.valid_arrays()
    with pytest.raises(ValueError, match="encoder-only uncertainty"):
        V5ExactSearchObservation(
            observed_curve=ObservedCurve(
                curve_id=missing.observed_curve.curve_id,
                source_kind="synthetic",
                q=q,
                intensity=intensity,
                sigma_log=encoder_sigma / intensity,
            ),
            observation_view=missing.observation_view,
            acceptance_sigma_source_id=(
                f"{missing.observation_view.acquisition_policy_id}|acceptance_sigma_log"
            ),
        )

    wrong_view = V5ExactSearchObservation.from_observation_view(
        missing.observation_view,
        curve_id=with_sigma.exact_observation.observed_curve.curve_id,
    )
    with pytest.raises(ValueError, match="curve/provenance disagrees"):
        collect_v5_search_supervision_sidecar(
            parent_path,
            (replace(with_sigma, exact_observation=wrong_view), without_sigma),
            sidecar_id="wrong-exact-view",
            protocol=_protocol(),
            runner=_completed_runner,
        )

    *_, sidecar, _, _ = _build_complete(tmp_path, views=(0,), seed=32012)
    changed_arrays = dict(sidecar.arrays)
    intensity_values = np.array(
        changed_arrays[query_array("exact_curve_intensity")], copy=True
    )
    mask = changed_arrays[query_array("exact_curve_point_mask")][0]
    intensity_values[0, np.flatnonzero(mask)[0]] *= 1.01
    changed_arrays[query_array("exact_curve_intensity")] = intensity_values
    with pytest.raises(ValueError, match="physical curve SHA-256"):
        _self_consistent_sidecar(sidecar, changed_arrays)


def test_clean_group_views_require_one_explicit_topology_range_catalog(tmp_path):
    parent_path, _, specs = _parent_and_specs(
        tmp_path, views=(0, 1), seed=32001
    )
    changed = replace(
        specs[1],
        query_catalog_artifact_sha256=_hash("different-query-catalog"),
    )
    with pytest.raises(ValueError, match="one frozen topology query catalog"):
        collect_v5_search_supervision_sidecar(
            parent_path,
            (specs[0], changed),
            sidecar_id="mixed-query-catalog",
            protocol=_protocol(),
            runner=_completed_runner,
        )
