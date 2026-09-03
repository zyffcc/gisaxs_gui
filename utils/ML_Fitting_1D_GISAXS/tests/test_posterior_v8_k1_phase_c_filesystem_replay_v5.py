from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    V5BoundsQuery,
    full_range_axis_designs,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import (
    array_sha256,
    canonical_json,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_PRODUCT_METHOD_ID,
    v5_k1_phase_c_contract_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_filesystem_replay_v5 import (
    V5K1PhaseCFilesystemReplayAdapter,
    _authorize_v5_k1_phase_c_formal_filesystem_replay,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_raw_artifacts_v5 import (
    RAW_ARTIFACT_BINDING_SCHEMA,
    RAW_BRANCH_SIDECAR_SCHEMA,
    RAW_DISTANCE_CONTEXT_SCHEMA,
    RAW_EVALUATOR_CONFIG_SCHEMA,
    RAW_METHOD_TRACE_SCHEMA,
    RAW_PARENT_PROVENANCE_SCHEMA,
    RAW_REFERENCE_BANK_SCHEMA,
    RAW_REFERENCE_TRACE_SCHEMA,
    RAW_REPRESENTATIVE_PAYLOAD_SCHEMA,
    RAW_SPLIT_RECEIPT_SCHEMA,
    V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
    V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS,
    V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
    V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_replay_runner_v5 import (
    V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER,
    run_v5_k1_phase_c_replay,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_payload_v5 import (
    V5PaperParameterRepresentativePayload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_sidecar_v5 import (
    V5_SEARCH_SIDECAR_SCHEMA,
    V5_SEARCH_SIDECAR_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_contract_v5 import V5TopologyQuery
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_phase_c_replay_v5 import (
    _artifact_binding,
    _replay_evidence,
)


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _raw(schema: str, **values: object) -> dict[str, object]:
    return {
        "schema": schema,
        "version": V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
        **values,
    }


def _parameter_payload(payload: V5PaperParameterRepresentativePayload) -> dict[str, object]:
    parameter = payload.parameter
    common = {
        "topology_id": parameter.topology_id,
        "components": [asdict(value) for value in parameter.components],
        "resolution": None if parameter.resolution is None else asdict(parameter.resolution),
        "linear_solution": {
            "background": parameter.linear_solution.background,
            "particle_amplitudes": list(parameter.linear_solution.particle_amplitudes),
            "resolution_amplitude": parameter.linear_solution.resolution_amplitude,
            "k": parameter.linear_solution.k,
        },
    }
    if hasattr(parameter, "candidate_id"):
        return {
            **common,
            "candidate_id": parameter.candidate_id,
            "proposal_rank": parameter.proposal_rank,
            "exact_intensity": parameter.exact_intensity.tolist(),
            "exact_intensity_dtype": "<f8",
            "exact_intensity_shape": list(parameter.exact_intensity.shape),
            "exact_intensity_order": "C",
            "exact_intensity_sha256": array_sha256(
                "candidate_exact_intensity", parameter.exact_intensity
            ),
            "bounds_pass": parameter.bounds_pass,
            "physics_pass": parameter.physics_pass,
            "proposal_score_raw": parameter.proposal_score_raw,
        }
    return {**common, "reference_id": parameter.reference_id}


def _write_snapshot(
    root: Path,
    *,
    plan,
    legacy_branch_schema: bool = False,
) -> tuple[Path, str, Path]:
    exact_calls = tuple(
        __import__(
            "utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_budget_evaluator_v5",
            fromlist=["V5ExactForwardCall"],
        ).V5ExactForwardCall(exact_call_index=index, elapsed_seconds=float(index))
        for index in range(1, 4097)
    )
    _, _, config, original = _replay_evidence(exact_calls)
    binding = _artifact_binding()
    provenance = replace(
        original.provenance,
        source_bundle_sha256=binding.source_bundle_sha256,
        model_artifact_sha256=binding.model_artifact_sha256,
    )
    methods = tuple(
        replace(
            value,
            source_bundle_sha256_used=binding.source_bundle_sha256,
            model_artifact_sha256_used=(
                binding.model_artifact_sha256
                if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
                else None
            ),
            proposal_execution_policy_sha256_used=(
                V5_PROPOSAL_EXECUTION_POLICY_SHA256
                if value.trace.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
                else None
            ),
        )
        for value in original.methods
    )
    bank_source = replace(
        original.reference_bank, source_bundle_sha256=binding.source_bundle_sha256
    )
    root.mkdir()
    file_rows: list[dict[str, object]] = []
    paths: dict[str, Path] = {}

    def add(file_id: str, role: str, payload: dict[str, object]) -> tuple[str, str]:
        path = root / f"{file_id}.json"
        encoded = canonical_json(payload).encode("utf-8")
        path.write_bytes(encoded)
        path.chmod(0o444)
        digest_value = sha256(encoded).hexdigest()
        file_rows.append(
            {
                "file_id": file_id,
                "role": role,
                "relative_path": path.name,
                "sha256": digest_value,
            }
        )
        paths[file_id] = path
        return file_id, digest_value

    source_fields = (
        "source_archive_sha256",
        "source_manifest_sha256",
        "source_tree_sha256",
        "source_bundle_sha256",
    )
    model_fields = (
        "cross_platform_gate_claim_sha256",
        "phase_a_launch_receipt_sha256",
        "model_artifact_sha256",
        "model_weights_sha256",
        "model_training_result_sha256",
        "proposal_execution_policy_sha256",
    )
    binding_id, _ = add(
        "artifact-binding",
        "artifact_binding",
        _raw(
            RAW_ARTIFACT_BINDING_SCHEMA,
            artifact_binding=binding.audit_payload(),
            source_provenance={name: getattr(binding, name) for name in source_fields},
            model_provenance={name: getattr(binding, name) for name in model_fields},
        ),
    )
    policy_id, policy_file_sha = add(
        "proposal-policy",
        "proposal_execution_policy",
        V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
    )
    assert policy_file_sha == V5_PROPOSAL_EXECUTION_POLICY_SHA256
    provenance_id, _ = add(
        "parent-provenance",
        "parent_provenance",
        _raw(RAW_PARENT_PROVENANCE_SCHEMA, provenance=asdict(provenance)),
    )

    bounds = GuiComponentBounds(
        shape=SPHERE,
        R=ClosedInterval(5.0, 30.0),
        sigma_R=ClosedInterval(0.5, 3.0),
    )
    geometry = V5BoundsQuery.create(
        query_seed=19,
        generation_attempt=0,
        component_bounds=(bounds,),
        resolution_presence_policy="absent",
        resolution_bounds=None,
        axis_designs=full_range_axis_designs((bounds,), None),
    )
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0),
        k=ClosedInterval(0.1, 10.0),
        component_intensities=(ClosedInterval(0.01, 10.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    topology_query = V5TopologyQuery(geometry=geometry, amplitude=amplitude)
    distance_id, _ = add(
        "distance-context",
        "distance_context",
        _raw(
            RAW_DISTANCE_CONTEXT_SCHEMA,
            query_context_sha256=provenance.evaluation_query_context_sha256,
            universal_query_sha256=provenance.universal_query_sha256,
            topology_queries=[
                {
                    "topology_id": topology_query.topology_id,
                    "topology_query_sha256": topology_query.sha256,
                    "geometry_query_canonical_json": geometry.canonical_json,
                    "geometry_query_sha256": geometry.sha256,
                    "amplitude_query_canonical_json": amplitude.canonical_json,
                    "amplitude_query_sha256": amplitude.sha256,
                }
            ],
        ),
    )

    representative_file_ids: dict[str, str] = {}
    representative_payloads: dict[str, V5PaperParameterRepresentativePayload] = {}
    source_payloads = [value.payload for value in original.reference_set.representatives]
    source_payloads.extend(
        emission.payload for method in methods for emission in method.trace.candidate_emissions
    )
    for source_payload in source_payloads:
        if source_payload.representative_id in representative_file_ids:
            continue
        file_id = f"payload-{source_payload.representative_id}"
        _, file_sha = add(
            file_id,
            "representative_payload",
            _raw(
                RAW_REPRESENTATIVE_PAYLOAD_SCHEMA,
                representative_id=source_payload.representative_id,
                role=source_payload.role,
                global_branch_key=source_payload.global_branch_key,
                query_context_sha256=source_payload.query_context_sha256,
                parameter=_parameter_payload(source_payload),
            ),
        )
        representative_file_ids[source_payload.representative_id] = file_id
        representative_payloads[source_payload.representative_id] = replace(
            source_payload, source_artifact_sha256=file_sha
        )

    branch_ids = []
    first_branch_path = None
    for index, branch in enumerate(original.branch_searches):
        file_id = f"branch-{index:02d}"
        branch_ids.append(file_id)
        add(
            file_id,
            "branch_search_sidecar",
            _raw(
                RAW_BRANCH_SIDECAR_SCHEMA,
                search_evidence_schema=(
                    "legacy/search-sidecar/v4"
                    if legacy_branch_schema and index == 0
                    else V5_SEARCH_SIDECAR_SCHEMA
                ),
                search_evidence_version=V5_SEARCH_SIDECAR_VERSION,
                branch_id=branch.branch_id,
                frozen_search_yield_rank=branch.frozen_search_yield_rank,
                completed=branch.completed,
                candidates=[asdict(value) for value in branch.candidates],
            ),
        )
        if index == 0:
            first_branch_path = paths[file_id]

    reference_bank_id, reference_bank_sha = add(
        "reference-bank",
        "reference_bank",
        _raw(
            RAW_REFERENCE_BANK_SCHEMA,
            reference_bank_schema=bank_source.schema,
            reference_bank_version=bank_source.version,
            calibration_identity_sha256=bank_source.calibration_identity_sha256,
            calibrated_threshold_sha256=bank_source.calibrated_threshold_sha256,
            exact_judge_sha256=bank_source.exact_judge_sha256,
            source_bundle_sha256=bank_source.source_bundle_sha256,
            query_id=bank_source.query_id,
            pairing_unit_id=bank_source.pairing_unit_id,
            query_context_sha256=bank_source.query_context_sha256,
            reference_set_id=original.reference_set.reference_set_id,
            comparison_protocol_id=original.reference_set.comparison_protocol_id,
            comparison_protocol_sha256=original.reference_set.comparison_protocol_sha256,
            candidate_ids=list(bank_source.candidate_ids),
            exact_compatible_candidate_ids=list(bank_source.exact_compatible_candidate_ids),
            representative_clusters=[asdict(value) for value in bank_source.representative_clusters],
            representative_payloads=[
                {
                    "representative_id": value.representative_id,
                    "payload_file_id": representative_file_ids[value.representative_id],
                    "payload_sha256": representative_payloads[value.representative_id].sha256,
                }
                for value in original.reference_set.representatives
            ],
            distance_schema=bank_source.distance_schema,
            distance_version=bank_source.distance_version,
            distance_sha256=bank_source.distance_sha256,
        ),
    )
    reference_trace_id, _ = add(
        "reference-trace",
        "reference_search_trace",
        _raw(
            RAW_REFERENCE_TRACE_SCHEMA,
            query_id=bank_source.query_id,
            pairing_unit_id=bank_source.pairing_unit_id,
            calibration_identity_sha256=bank_source.calibration_identity_sha256,
            calibrated_threshold_sha256=bank_source.calibrated_threshold_sha256,
            exact_judge_sha256=bank_source.exact_judge_sha256,
            source_bundle_sha256=bank_source.source_bundle_sha256,
            configured_exact_call_budget=bank_source.configured_exact_call_budget,
            consumed_exact_calls=bank_source.consumed_exact_calls,
            exact_forward_calls=[asdict(value) for value in bank_source.exact_forward_calls],
            candidate_judgements=[asdict(value) for value in bank_source.candidate_judgements],
            enumeration_complete=bank_source.enumeration_complete,
            network_free=bank_source.network_free,
        ),
    )

    method_ids = []
    for index, method in enumerate(methods):
        trace = method.trace
        file_id = f"method-{index}"
        method_ids.append(file_id)
        add(
            file_id,
            "method_exact_call_trace",
            _raw(
                RAW_METHOD_TRACE_SCHEMA,
                status=method.status,
                query_id=trace.query_id,
                pairing_unit_id=trace.pairing_unit_id,
                method_id=trace.method_id,
                method_protocol_id=trace.method_protocol_id,
                method_protocol_sha256=trace.method_protocol_sha256,
                trace_id=trace.trace_id,
                reference_set_id=trace.reference_set_id,
                reference_set_sha256=reference_bank_sha,
                comparison_protocol_id=trace.comparison_protocol_id,
                comparison_protocol_sha256=trace.comparison_protocol_sha256,
                exact_forward_call_budget=trace.exact_forward_call_budget,
                exact_forward_calls=[asdict(value) for value in trace.exact_forward_calls],
                candidate_emissions=[
                    {
                        "available_after_call": value.available_after_call,
                        "output_rank": value.output_rank,
                        "candidate_id": value.candidate_id,
                        "compatibility_status": value.compatibility_status,
                        "elapsed_seconds": value.elapsed_seconds,
                        "payload_file_id": representative_file_ids[value.candidate_id],
                        "payload_sha256": representative_payloads[value.candidate_id].sha256,
                    }
                    for value in trace.candidate_emissions
                ],
                source_bundle_sha256_used=method.source_bundle_sha256_used,
                model_artifact_sha256_used=method.model_artifact_sha256_used,
                proposal_execution_policy_sha256_used=(
                    method.proposal_execution_policy_sha256_used
                ),
            ),
        )

    split_id, _ = add(
        "split",
        "split_receipt",
        _raw(
            RAW_SPLIT_RECEIPT_SCHEMA,
            split_id=provenance.clean_parent_sha256
            and __import__(
                "utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5",
                fromlist=["K1_PHASE_C_SPLIT_ID"],
            ).K1_PHASE_C_SPLIT_ID,
            plan_sha256=plan.sha256,
            included_clean_parent_sha256s=[provenance.clean_parent_sha256],
            excluded_population_sha256s=[_sha("excluded-population")],
            disjointness_verified=True,
        ),
    )
    evaluator_id, _ = add(
        "evaluator",
        "evaluator_config",
        _raw(RAW_EVALUATOR_CONFIG_SCHEMA, config=asdict(config)),
    )
    core = {
        "schema": V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
        "version": V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
        "status": "complete",
        "formal": plan.formal,
        "artifact_sha256_semantics": V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS,
        "plan_sha256": plan.sha256,
        "contract_sha256": v5_k1_phase_c_contract_payload()["contract_sha256"],
        "artifact_binding_file_id": binding_id,
        "proposal_execution_policy_file_id": policy_id,
        "split_receipt_file_id": split_id,
        "evaluator_config_file_id": evaluator_id,
        "files": file_rows,
        "parents": [
            {
                "clean_parent_sha256": provenance.clean_parent_sha256,
                "provenance_file_id": provenance_id,
                "distance_context_file_id": distance_id,
                "branch_sidecar_file_ids": branch_ids,
                "reference_bank_file_id": reference_bank_id,
                "reference_trace_file_id": reference_trace_id,
                "method_trace_file_ids": method_ids,
            }
        ],
    }
    manifest = {
        **core,
        "manifest_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    encoded = canonical_json(manifest).encode("utf-8")
    manifest_path.write_bytes(encoded)
    manifest_path.chmod(0o444)
    assert first_branch_path is not None
    return manifest_path, sha256(encoded).hexdigest(), first_branch_path


def test_filesystem_adapter_reopens_lossless_raw_files_and_rebuilds_bundle(tmp_path) -> None:
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
        build_v5_k1_phase_c_plan,
    )

    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    manifest, manifest_sha, _ = _write_snapshot(tmp_path / "snapshot", plan=plan)
    adapter = V5K1PhaseCFilesystemReplayAdapter(
        manifest, expected_manifest_file_sha256=manifest_sha
    )
    bundle = adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())

    assert len(bundle.parents) == 1
    assert len(bundle.parents[0].branch_searches) == 12
    assert len(bundle.parents[0].methods) == 3
    emitted = bundle.parents[0].methods[0].trace.candidate_emissions[0]
    assert np.array_equal(emitted.payload.parameter.exact_intensity, np.asarray([1.0]))
    adapter.revalidate_bundle(
        bundle=bundle, plan=plan, contract=v5_k1_phase_c_contract_payload()
    )
    adapter.revalidate_bundle(
        bundle=bundle, plan=plan, contract=v5_k1_phase_c_contract_payload()
    )


def test_revalidation_rejects_same_bytes_replaced_at_same_path(tmp_path) -> None:
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
        build_v5_k1_phase_c_plan,
    )

    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    manifest, manifest_sha, branch_path = _write_snapshot(tmp_path / "snapshot", plan=plan)
    adapter = V5K1PhaseCFilesystemReplayAdapter(
        manifest, expected_manifest_file_sha256=manifest_sha
    )
    bundle = adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
    replacement = branch_path.with_suffix(".replacement")
    replacement.write_bytes(branch_path.read_bytes())
    replacement.chmod(0o444)
    os.replace(replacement, branch_path)

    with pytest.raises(RuntimeError, match="files or identities changed"):
        adapter.revalidate_bundle(
            bundle=bundle, plan=plan, contract=v5_k1_phase_c_contract_payload()
        )


def test_legacy_sidecar_fails_closed_even_when_file_hash_matches(tmp_path) -> None:
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
        build_v5_k1_phase_c_plan,
    )

    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    manifest, manifest_sha, _ = _write_snapshot(
        tmp_path / "snapshot", plan=plan, legacy_branch_schema=True
    )
    adapter = V5K1PhaseCFilesystemReplayAdapter(
        manifest, expected_manifest_file_sha256=manifest_sha
    )
    with pytest.raises(ValueError, match="current search contract"):
        adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())


def test_formal_runner_rejects_adapter_impersonation_before_loading(tmp_path) -> None:
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
        build_v5_k1_phase_c_plan,
    )

    plan = build_v5_k1_phase_c_plan(formal=True)

    class Impostor:
        adapter_id = "v5_k1_phase_c_production_filesystem_replay"
        adapter_version = "canonical_read_only_nofollow_per_file_sha_inode_double_revalidation_v1"

        def load_bundle(self, **_kwargs):
            raise AssertionError("formal adapter identity check must precede loading")

    with pytest.raises(RuntimeError, match=V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER):
        run_v5_k1_phase_c_replay(
            plan=plan, port=Impostor(), receipt_path=tmp_path / "receipt.json"
        )


def test_formal_capability_hard_blocks_test_raw_without_audited_writer(tmp_path) -> None:
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
        build_v5_k1_phase_c_plan,
    )

    fixture = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    plan = replace(
        fixture,
        formal=True,
        total_parent_count=1,
        sha256=_sha("test-only-formal-rejection-plan"),
    )
    contract = v5_k1_phase_c_contract_payload()
    manifest, manifest_sha, _ = _write_snapshot(tmp_path / "snapshot", plan=plan)
    adapter = V5K1PhaseCFilesystemReplayAdapter(
        manifest, expected_manifest_file_sha256=manifest_sha
    )
    bundle = adapter.load_bundle(plan=plan, contract=contract)
    adapter.revalidate_bundle(bundle=bundle, plan=plan, contract=contract)
    adapter.revalidate_bundle(bundle=bundle, plan=plan, contract=contract)

    with pytest.raises(RuntimeError, match=V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER):
        _authorize_v5_k1_phase_c_formal_filesystem_replay(
            adapter, bundle=bundle, plan=plan, contract=contract
        )


def test_formal_runner_with_real_adapter_fails_closed_without_audited_writer(
    tmp_path,
) -> None:
    """Test-only serialized evidence must never impersonate a production writer."""

    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
        build_v5_k1_phase_c_plan,
    )

    fixture = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    manifest, manifest_sha, _ = _write_snapshot(
        tmp_path / "snapshot", plan=fixture
    )
    adapter = V5K1PhaseCFilesystemReplayAdapter(
        manifest, expected_manifest_file_sha256=manifest_sha
    )
    formal = build_v5_k1_phase_c_plan(formal=True)
    receipt = tmp_path / "must-not-exist.json"

    with pytest.raises(RuntimeError, match=V5_K1_PHASE_C_PRODUCTION_ADAPTER_BLOCKER):
        run_v5_k1_phase_c_replay(plan=formal, port=adapter, receipt_path=receipt)
    assert not receipt.exists()
    with pytest.raises(RuntimeError, match="loaded"):
        _ = adapter.equivalence_distance_matcher
