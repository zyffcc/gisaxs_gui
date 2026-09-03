"""Run the checked V5.2 sphere/pattern-0 Phase-B capacity diagnostic.

The formal mode evaluates exactly all 512 clean parents.  ``--parent-limit``
is an explicit engineering throughput smoke and can never pass a gate.  Both
modes require a Slurm worker and publish an exclusive, self-hashed receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np

from .candidate_proposals_v5 import V5BatchedProposalOutput, V5_PROPOSAL_SAMPLER_VERSION
from .candidate_refinement_contract_v5 import (
    V5_EXACT_REFINEMENT_SCHEMA,
    V5_EXACT_REFINEMENT_VERSION,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import read_v5_grouped_dataset
from .k1_phase_b_contract_v5 import (
    EXPECTED_GATE_KEYS,
    MAXWELL_DUST_ROOT,
    PHASE_A_SOURCE_PATHS,
    PHASE_B_SOURCE_PATHS,
    PROPOSAL_COUNT,
    V5_K1_PHASE_B_RESULT_FILENAME,
    V5_K1_PHASE_B_ROLE,
    V5_K1_PHASE_B_SCHEMA,
    V5_K1_PHASE_B_VERSION,
    V5K1PhaseBGateConfig,
    V5K1PhaseBParentRecord,
    dataset_identity,
    execution_guard,
    file_identity,
    live_gate_contract,
    model_weights_sha256,
    source_identity,
    strict_json_object,
    under_root,
    validate_v5_k1_phase_a_bindings,
    verify_recorded_dataset_sources,
)
from .k1_phase_b_evaluation_v5 import (
    as_numpy_outputs,
    assess_v5_k1_phase_b_records,
    build_parent_record,
    parent_seed,
    replay_parent_contexts,
)
from .k1_phase_b_worker_inputs_v5 import (
    V5K1PhaseBWorkerInputSpec,
    add_v5_k1_phase_b_worker_input_arguments,
    assert_v5_k1_phase_b_worker_inputs_unchanged,
    bind_v5_k1_phase_b_worker_inputs,
    v5_k1_phase_b_worker_input_kwargs,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS, MODEL_V5_OUTPUT_KEYS
from .run_k1_memorization_gate_v5 import (
    MODEL_FILENAME as PHASE_A_MODEL_FILENAME,
    MODEL_PROVENANCE_FILENAME as PHASE_A_MODEL_PROVENANCE_FILENAME,
    RESULT_FILENAME as PHASE_A_RESULT_FILENAME,
    validate_v5_k1_memorization_dataset,
)


def _publish_json(target: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite result artifact: {target}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _validated_paths(
    dataset_path: str | os.PathLike[str],
    result_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    allowed_root: Path,
    input_allowed_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    dataset = under_root(Path(dataset_path), input_allowed_root, "dataset_path")
    result = under_root(Path(result_path), input_allowed_root, "phase_a_result_path")
    model = under_root(Path(model_path), input_allowed_root, "phase_a_model_path")
    model_provenance = model.parent / PHASE_A_MODEL_PROVENANCE_FILENAME
    output = under_root(Path(output_dir), allowed_root, "output_dir")
    for path, name in (
        (dataset, "checked dataset"),
        (result, "Phase-A result"),
        (model, "Phase-A model"),
        (model_provenance, "Phase-A model provenance"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")
    return dataset, result, model, model_provenance, output


def _branch_scope(contexts) -> tuple[tuple[str, ...], int, int]:
    values = {
        (
            tuple(context[0].query.topology),
            int(context[1].branch_conditions[0].topology_id),
            int(context[0].target.pattern_id),
        )
        for context in contexts
    }
    if len(values) != 1:
        raise ValueError("single-branch Phase-B worker received a mixed branch cohort")
    scope = values.pop()
    if scope[0] != ("sphere",) or scope[2] != 0:
        raise ValueError("frozen K1 Phase-B cohort must be exactly sphere/pattern-0")
    return scope


def run_v5_k1_phase_b_gate(
    dataset_path: str | os.PathLike[str],
    phase_a_result_path: str | os.PathLike[str],
    phase_a_model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str],
    source_archive_path: str | os.PathLike[str],
    source_archive_sha256: str,
    source_archive_byte_count: int,
    source_manifest_sha256: str,
    source_tree_sha256: str,
    original_source_root: str | os.PathLike[str],
    original_source_archive_path: str | os.PathLike[str],
    original_dataset_path: str | os.PathLike[str],
    dataset_sha256: str,
    dataset_byte_count: int,
    original_phase_a_output_dir: str | os.PathLike[str],
    original_phase_a_result_path: str | os.PathLike[str],
    phase_a_result_sha256: str,
    phase_a_result_byte_count: int,
    original_phase_a_model_path: str | os.PathLike[str],
    phase_a_model_sha256: str,
    phase_a_model_byte_count: int,
    original_phase_a_model_provenance_path: str | os.PathLike[str],
    phase_a_model_provenance_sha256: str,
    phase_a_model_provenance_byte_count: int,
    config: V5K1PhaseBGateConfig = V5K1PhaseBGateConfig(),
    allowed_root: Path = MAXWELL_DUST_ROOT,
    input_allowed_root: Path | None = None,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Reverify Phase A, execute Phase B, and exclusively publish one receipt."""

    if not isinstance(config, V5K1PhaseBGateConfig):
        raise TypeError("config must be a V5K1PhaseBGateConfig")
    source_root_path = Path(source_root).expanduser().resolve()
    if (source_root_path / PHASE_B_SOURCE_PATHS[0]).resolve() != Path(__file__).resolve():
        raise ValueError("source_root does not own the imported K1 Phase-B worker")
    selected_input_root = allowed_root if input_allowed_root is None else input_allowed_root
    dataset_file, result_file, model_file, model_provenance_file, output = _validated_paths(
        dataset_path,
        phase_a_result_path,
        phase_a_model_path,
        output_dir,
        allowed_root=allowed_root,
        input_allowed_root=selected_input_root,
    )
    authoritative_dataset = under_root(
        Path(original_dataset_path), allowed_root, "original_dataset_path"
    )
    authoritative_phase_a_output = under_root(
        Path(original_phase_a_output_dir), allowed_root, "original_phase_a_output_dir"
    )
    if not authoritative_dataset.is_file():
        raise FileNotFoundError("the authoritative Phase-A dataset is missing")
    if not authoritative_phase_a_output.is_dir():
        raise FileNotFoundError("the authoritative Phase-A output directory is missing")
    worker_input_spec = V5K1PhaseBWorkerInputSpec(
        allowed_root=allowed_root,
        input_root=selected_input_root,
        original_source_root=Path(original_source_root),
        job_source_root=Path(source_root),
        phase_a_output_dir=authoritative_phase_a_output,
        authoritative_paths={
            "source_archive": Path(original_source_archive_path),
            "dataset": Path(original_dataset_path),
            "phase_a_result": Path(original_phase_a_result_path),
            "phase_a_model": Path(original_phase_a_model_path),
            "phase_a_model_provenance": Path(original_phase_a_model_provenance_path),
        },
        local_paths={
            "source_archive": Path(source_archive_path),
            "dataset": Path(dataset_path),
            "phase_a_result": Path(phase_a_result_path),
            "phase_a_model": Path(phase_a_model_path),
            "phase_a_model_provenance": (
                Path(phase_a_model_path).parent / PHASE_A_MODEL_PROVENANCE_FILENAME
            ),
        },
        expected={
            "source_archive": {
                "sha256": source_archive_sha256,
                "byte_count": source_archive_byte_count,
            },
            "dataset": {
                "sha256": dataset_sha256,
                "byte_count": dataset_byte_count,
            },
            "phase_a_result": {
                "sha256": phase_a_result_sha256,
                "byte_count": phase_a_result_byte_count,
            },
            "phase_a_model": {
                "sha256": phase_a_model_sha256,
                "byte_count": phase_a_model_byte_count,
            },
            "phase_a_model_provenance": {
                "sha256": phase_a_model_provenance_sha256,
                "byte_count": phase_a_model_provenance_byte_count,
            },
        },
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
    )
    bound_worker_inputs = bind_v5_k1_phase_b_worker_inputs(worker_input_spec)
    source_snapshot = bound_worker_inputs["source_snapshot"]
    snapshot_binding = {
        "source_archive_sha256": source_snapshot["archive_sha256"],
        "source_manifest_sha256": source_snapshot["manifest_sha256"],
        "source_tree_sha256": source_snapshot["source_tree_sha256"],
    }
    host = socket.gethostname() if hostname is None else hostname
    selected_environment = os.environ if environment is None else environment
    job_id = execution_guard(host, selected_environment)

    dataset, receipt = read_v5_grouped_dataset(dataset_file)
    dataset_validation = validate_v5_k1_memorization_dataset(dataset)
    protocol = live_gate_contract()
    expected_parents = int(protocol["recipes"])
    if dataset.recipe_count != expected_parents:
        raise ValueError("checked dataset is not the frozen 512-parent single-branch stage")
    expected_views = expected_parents * int(protocol["training_augmentation_views"])
    if dataset.observation_count != expected_views:
        raise ValueError("checked dataset does not have the frozen one-view-per-parent design")
    if config.parent_limit is not None and config.parent_limit >= expected_parents:
        raise ValueError("parent_limit is smoke-only; omit it for the formal 512-parent gate")
    engineering_subset = config.parent_limit is not None
    selected_parent_count = config.parent_limit or expected_parents

    dataset_sources = verify_recorded_dataset_sources(dataset, source_root_path)
    bound_dataset_identity = dataset_identity(
        dataset_file,
        dataset,
        receipt,
        authoritative_path=authoritative_dataset,
    )
    phase_a_result = strict_json_object(
        result_file.read_text(encoding="utf-8"), "Phase-A result"
    )
    if result_file.name != PHASE_A_RESULT_FILENAME:
        raise ValueError("Phase-A result must use its versioned result filename")
    if result_file.parent != model_file.parent or model_file.name != PHASE_A_MODEL_FILENAME:
        raise ValueError("Phase-A result and model paths do not form one saved artifact")
    if Path(str(phase_a_result.get("output_dir", ""))).resolve() != (
        authoritative_phase_a_output
    ):
        raise ValueError("Phase-A result output directory drift detected")
    phase_a_source = source_identity(source_root_path, tuple(sorted(PHASE_A_SOURCE_PATHS)))
    phase_b_source_before = source_identity(source_root_path, PHASE_B_SOURCE_PATHS)
    model_identity = file_identity(model_file)
    model_provenance_identity = file_identity(model_provenance_file)
    model_provenance_payload = strict_json_object(
        model_provenance_file.read_text(encoding="utf-8"),
        "Phase-A model provenance",
    )
    phase_a_result_identity = file_identity(result_file)

    import tensorflow as tf

    from .model_v5 import validate_model_v5_graph_contract

    model = tf.keras.models.load_model(model_file, compile=False)
    validate_model_v5_graph_contract(model)
    weights_digest = model_weights_sha256(model)
    phase_a_binding = validate_v5_k1_phase_a_bindings(
        phase_a_result,
        dataset_identity=bound_dataset_identity,
        model_identity=model_identity,
        model_provenance_payload=model_provenance_payload,
        model_provenance_identity=model_provenance_identity,
        phase_a_source_identity=phase_a_source,
        source_snapshot_identity=snapshot_binding,
        model_weights_sha256_value=weights_digest,
        live_gate_contract_value=protocol,
    )

    total_started = time.perf_counter()
    replay_started = time.perf_counter()
    contexts = replay_parent_contexts(dataset, parent_limit=selected_parent_count)
    replay_seconds = time.perf_counter() - replay_started
    topology, topology_id, pattern_id = _branch_scope(contexts)
    combined_inputs = {
        name: np.concatenate([context[1].model_inputs[name] for context in contexts], axis=0)
        for name in MODEL_V5_INPUT_KEYS
    }
    inference_started = time.perf_counter()
    raw_outputs = as_numpy_outputs(model(combined_inputs, training=False))
    inference_seconds = time.perf_counter() - inference_started
    if set(raw_outputs) != set(MODEL_V5_OUTPUT_KEYS):
        raise ValueError("saved Phase-A model returned an incompatible output inventory")
    parsed = V5BatchedProposalOutput.from_mapping(raw_outputs, branch_count=len(contexts))
    if parsed.mixture_count != 1:
        raise ValueError("saved Phase-A model is not a one-component MDN")

    records = []
    exact_started = time.perf_counter()
    raw_threshold = float(protocol["gates"][EXPECTED_GATE_KEYS[3]])
    for index, (recipe, batch, target, varying, curve) in enumerate(contexts):
        sliced = {name: value[index : index + 1] for name, value in raw_outputs.items()}
        records.append(
            build_parent_record(
                recipe=recipe,
                batch=batch,
                target=target,
                varying=varying,
                curve=curve,
                model_outputs=sliced,
                frozen_parent_seed=parent_seed(config.seed, recipe.sha256),
                raw_compatibility_threshold=raw_threshold,
                config=config,
            )
        )
        if (index + 1) % config.progress_interval == 0 or index + 1 == len(contexts):
            print(
                f"K1 Phase-B completed {index + 1}/{len(contexts)} clean parents",
                file=sys.stderr,
                flush=True,
            )
    exact_seconds = time.perf_counter() - exact_started
    assessment = assess_v5_k1_phase_b_records(
        records,
        gate_contract=protocol,
        engineering_subset=engineering_subset,
    )
    total_seconds = time.perf_counter() - total_started

    phase_b_source = source_identity(source_root_path, PHASE_B_SOURCE_PATHS)
    assert_v5_k1_phase_b_worker_inputs_unchanged(
        bound_worker_inputs, worker_input_spec
    )
    if phase_b_source != phase_b_source_before:
        raise RuntimeError("K1 Phase-B source changed during execution")
    if source_identity(
        source_root_path, tuple(sorted(PHASE_A_SOURCE_PATHS))
    ) != phase_a_source:
        raise RuntimeError("bound Phase-A source changed during K1 Phase-B execution")
    if verify_recorded_dataset_sources(dataset, source_root_path) != dataset_sources:
        raise RuntimeError("recorded dataset source changed during K1 Phase-B execution")
    if file_identity(dataset_file) != {
        "sha256": receipt.artifact_sha256,
        "byte_count": receipt.byte_count,
    }:
        raise RuntimeError("checked dataset changed during K1 Phase-B execution")
    if file_identity(result_file) != phase_a_result_identity:
        raise RuntimeError("Phase-A result changed during K1 Phase-B execution")
    if file_identity(model_file) != model_identity:
        raise RuntimeError("Phase-A model changed during K1 Phase-B execution")
    if file_identity(model_provenance_file) != model_provenance_identity:
        raise RuntimeError("Phase-A model provenance changed during K1 Phase-B execution")
    if model_weights_sha256(model) != weights_digest:
        raise RuntimeError("loaded Phase-A model weights changed during inference")

    calls_used = int(assessment["exact_forward_ledger"]["calls_used"])
    calls_per_second = calls_used / exact_seconds if exact_seconds > 0.0 else None
    theoretical_max_calls = expected_parents * PROPOSAL_COUNT * (
        config.per_candidate_forward_evaluation_limit
    )
    measured_parent_seconds = exact_seconds / selected_parent_count
    peak_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_unit = "bytes" if sys.platform == "darwin" else "kibibytes"
    compute = {
        "mode": "engineering_throughput_smoke" if engineering_subset else "formal_gate",
        "selected_parent_count": selected_parent_count,
        "formal_parent_count": expected_parents,
        "proposal_count_per_parent": PROPOSAL_COUNT,
        "theoretical_formal_exact_call_upper_bound": theoretical_max_calls,
        "formula": "512_parents_x_32_draws_x_per_candidate_forward_limit",
        "replay_seconds": replay_seconds,
        "model_inference_seconds": inference_seconds,
        "exact_refinement_seconds": exact_seconds,
        "total_seconds": total_seconds,
        "measured_exact_calls_per_second": calls_per_second,
        "measured_exact_seconds_per_parent": measured_parent_seconds,
        "projected_formal_seconds_at_measured_parent_throughput": (
            measured_parent_seconds * expected_parents
        ),
        "projected_formal_theoretical_max_seconds_at_measured_call_throughput": (
            None if not calls_per_second else theoretical_max_calls / calls_per_second
        ),
        "process_peak_rss_raw": peak_rss_raw,
        "process_peak_rss_unit": peak_rss_unit,
        "resource_characterization": (
            "exact_SciPy_NumPy_refinement_is_CPU_dominant;TensorFlow_is_one_batched_inference"
        ),
        "formal_resource_plan_status": (
            "pending_measured_smoke_before_CPU_parallel_array_or_hybrid_choice"
            if engineering_subset
            else "formal_serial_plan_explicitly_selected_after_external_smoke_review"
        ),
    }
    single_branch_passed = bool(assessment["single_branch_phase_b_gate_passed"])
    if engineering_subset:
        status = "engineering_throughput_smoke_completed_fail_closed"
    else:
        status = (
            "single_branch_phase_b_passed"
            if single_branch_passed
            else "single_branch_phase_b_failed"
        )
    core = {
        "schema_version": V5_K1_PHASE_B_SCHEMA,
        "version": V5_K1_PHASE_B_VERSION,
        "scientific_role": V5_K1_PHASE_B_ROLE,
        "model_acceptance_evidence": False,
        "paper_model_acceptance_evidence": False,
        "performance_claim_allowed": False,
        "status": status,
        "single_branch_phase_b_gate_passed": single_branch_passed,
        "complete_k1_proposal_exact_gate_passed": False,
        "full_k1_all_legal_branches_gate_status": (
            "pending_fail_closed_requires_balanced_12_branch_cohort"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "protocol": protocol,
        "definitions": {
            "cohort_scope": {
                "topology": list(topology),
                "topology_id": topology_id,
                "pattern_id": pattern_id,
                "legal_k1_branch_coverage": "one_of_twelve",
                "full_k1_claim_allowed": False,
            },
            "branch_selection": "checked_generating_branch_only_no_branch_selection_test",
            "single_draw": (
                "stochastic_draw_index_1_from_each_parent_specific_frozen_rng_stream"
            ),
            "best_of_32": (
                "minimum_local_rms_among_stochastic_draw_indices_1_through_32_"
                "from_the_same_frozen_rng_stream_no_mixture_median"
            ),
            "local_rms": "rms_over_varying_dimension_mask_only",
            "aggregation": "one_value_per_independent_clean_parent_macro_average",
            "neural_input": "checked_phase_a_single_observation_view",
            "exact_target": (
                "noise_free_clean_parent_authoritative_gui_forward_on_frozen_clean_grid"
            ),
            "exact_compatibility": (
                "best_refined_raw_natural_log_rmse_strictly_below_"
                "dynamic_exact_post_refine_raw_log_rmse_p90_gate_threshold"
            ),
            "proposal_sampler_version": V5_PROPOSAL_SAMPLER_VERSION,
            "exact_refinement_schema": V5_EXACT_REFINEMENT_SCHEMA,
            "exact_refinement_version": V5_EXACT_REFINEMENT_VERSION,
        },
        "inputs": {
            "authoritative_original_inputs": bound_worker_inputs["authoritative"],
            "dataset": bound_dataset_identity,
            "dataset_source_replay": dataset_sources,
            "phase_a_result": {
                "path": str(authoritative_phase_a_output / result_file.name),
                **phase_a_result_identity,
                "payload_sha256": phase_a_result["result_payload_sha256"],
            },
            "phase_a_model": {
                "path": str(authoritative_phase_a_output / model_file.name),
                **model_identity,
                "weights_sha256": weights_digest,
                "provenance": {
                    "path": str(
                        authoritative_phase_a_output / model_provenance_file.name
                    ),
                    **model_provenance_identity,
                    "binding_sha256": model_provenance_payload["binding_sha256"],
                },
            },
            "phase_a_binding": phase_a_binding,
            "dataset_validation": dataset_validation,
        },
        "source": phase_b_source,
        "assessment": assessment,
        "compute": compute,
        "parent_records": [value.to_audit_dict() for value in records],
        "execution": {
            "hostname": host,
            "slurm_job_id": job_id,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
        },
        "publication": "exclusive_output_directory_and_atomic_exclusive_json",
    }
    payload = {
        **core,
        "result_payload_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    output.mkdir(mode=0o700, exist_ok=False)
    _publish_json(output / V5_K1_PHASE_B_RESULT_FILENAME, payload)
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    assert_v5_k1_phase_b_worker_inputs_unchanged(
        bound_worker_inputs, worker_input_spec
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--phase-a-result", required=True, type=Path)
    parser.add_argument("--phase-a-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    add_v5_k1_phase_b_worker_input_arguments(parser)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--per-candidate-forward-limit", type=int, default=128)
    parser.add_argument("--per-parent-forward-limit", type=int, default=4096)
    parser.add_argument("--progress-interval", type=int, default=16)
    parser.add_argument("--parent-limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_v5_k1_phase_b_gate(
        args.dataset,
        args.phase_a_result,
        args.phase_a_model,
        args.output_dir,
        **v5_k1_phase_b_worker_input_kwargs(args),
        config=V5K1PhaseBGateConfig(
            seed=args.seed,
            per_candidate_forward_evaluation_limit=args.per_candidate_forward_limit,
            per_parent_forward_evaluation_limit=args.per_parent_forward_limit,
            progress_interval=args.progress_interval,
            parent_limit=args.parent_limit,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] == "engineering_throughput_smoke_completed_fail_closed":
        return 0
    return 0 if result["single_branch_phase_b_gate_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MAXWELL_DUST_ROOT",
    "PROPOSAL_COUNT",
    "V5_K1_PHASE_B_RESULT_FILENAME",
    "V5_K1_PHASE_B_ROLE",
    "V5_K1_PHASE_B_SCHEMA",
    "V5_K1_PHASE_B_VERSION",
    "V5K1PhaseBGateConfig",
    "V5K1PhaseBParentRecord",
    "assess_v5_k1_phase_b_records",
    "main",
    "run_v5_k1_phase_b_gate",
    "validate_v5_k1_phase_a_bindings",
]
