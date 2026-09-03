"""Leakage-safe holdout and acceptance audit for bounds-first V3.

This entry point never evaluates training/tuning rows and never loads a V1/V2
artifact.  Expensive exact verification is a deterministic, optional subset;
set ``--exact-maximum-examples 0`` for a proposal-only smoke audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .bounds_holdout_exact import exact_record, exact_summary
from .bounds_holdout_metrics import (
    SUPPORTED_HOLDOUT_SPLITS,
    SUPPORTED_RANGE_SELECTIONS,
    BoundsHoldoutAuditConfig,
    evaluate_supervised,
    materialize_holdout_rows,
    model_outputs,
    select_holdout_rows,
    supervised_summary,
)
from .bounds_local_inference import CallableBoundsLocalProposalModel
from .bounds_training_audit import (
    HISTORY_FILE,
    load_bounds_trained_proposal_model,
    validate_bounds_training_run,
)
from .bounds_training_data import inspect_bounds_first_shards
from .build_bounds_first_shards import file_sha256, load_shard
from .canonical_branch_catalog import CANONICAL_BRANCH_CATALOG_VERSION
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .evaluation import EVALUATION_AUDIT_SCHEMA, EvaluationThresholds
from .one_click_inference import InferenceBudget
from .proposal_training_audit import MANIFEST_FILE, MODEL_FILE
from .study_protocol import protocol_payload


BOUNDS_HOLDOUT_AUDIT_SCHEMA = "gisaxs.posterior_v8.bounds_holdout_acceptance/v1"
BOUNDS_HOLDOUT_AUDIT_VERSION = "posterior_v8_bounds_local_holdout_acceptance_v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = sorted(root.rglob("*.py")) + [
        root / "slurm" / "bounds_holdout_v3_gpu.sbatch"
    ]
    return {
        path.relative_to(root).as_posix(): file_sha256(path) for path in paths
    }


def _validate_dataset_binding(manifest, dataset) -> None:
    trained = manifest.get("dataset_audit")
    if not isinstance(trained, dict) and not isinstance(trained, Mapping):
        raise ValueError("V3 manifest has no dataset audit")
    if (
        trained.get("consumed_splits") != ["train", "tuning_validation"]
        or trained.get("excluded_splits") != ["calibration", "test"]
        or trained.get("calibration_test_rows_consumed") != 0
    ):
        raise ValueError("V3 training manifest does not prove holdout isolation")
    if trained.get("fingerprint_sha256") != dataset.payload.get("fingerprint_sha256"):
        raise ValueError("V3 model and immutable V4 holdout shards have different fingerprints")


def _tensor_hashes(inputs, labels):
    result = {}
    for prefix, values in (("input", inputs), ("label", labels)):
        for name, value in values.items():
            array = np.ascontiguousarray(value)
            header = _canonical_json(
                {"dtype": array.dtype.str, "shape": list(array.shape)}
            )
            result[f"{prefix}:{name}"] = sha256(header + array.tobytes()).hexdigest()
    return result


def _write_json_exclusive(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite V3 holdout audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite V3 holdout audit: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_bounds_holdout_audit(
    shard_paths: Sequence[str | os.PathLike[str]],
    training_run: str | os.PathLike[str],
    output: str | os.PathLike[str],
    config: BoundsHoldoutAuditConfig = BoundsHoldoutAuditConfig(),
) -> dict[str, object]:
    """Run one deterministic split-labelled V3 proposal/acceptance audit."""

    if not isinstance(config, BoundsHoldoutAuditConfig):
        raise TypeError("config must be BoundsHoldoutAuditConfig")
    source_hashes_before = _source_hashes()
    supplied_shards = tuple(Path(value) for value in shard_paths)
    if not supplied_shards or any(
        path.is_symlink()
        or not path.is_file()
        or path.with_suffix(".json").is_symlink()
        or not path.with_suffix(".json").is_file()
        for path in supplied_shards
    ):
        raise ValueError("holdout inputs must be regular immutable V4 NPZ/JSON pairs")
    dataset = inspect_bounds_first_shards(supplied_shards)
    supplied_run = Path(training_run)
    if supplied_run.is_symlink():
        raise ValueError("V3 training run must not be a symlink")
    run_dir = supplied_run.resolve()
    manifest = validate_bounds_training_run(run_dir)
    _validate_dataset_binding(manifest, dataset)
    model = load_bounds_trained_proposal_model(run_dir)
    history = json.loads((run_dir / HISTORY_FILE).read_text(encoding="utf-8"))
    artifact_hashes = {
        "model_sha256": file_sha256(run_dir / MODEL_FILE),
        "manifest_sha256": file_sha256(run_dir / MANIFEST_FILE),
        "history_sha256": file_sha256(run_dir / HISTORY_FILE),
    }
    references, eligible_row_count, eligible_recipe_count = select_holdout_rows(
        dataset, config
    )
    inputs, labels, provenance, contexts = materialize_holdout_rows(references)
    outputs = model_outputs(model, inputs, config.batch_size)
    records = evaluate_supervised(inputs, labels, outputs, provenance, contexts, config)
    wrapped = CallableBoundsLocalProposalModel.from_loaded_model(model)
    exact_records = []
    exact_shards = {}
    for reference in references[: config.exact_maximum_examples]:
        shard = exact_shards.get(reference.path)
        if shard is None:
            shard = load_shard(reference.path)
            exact_shards[reference.path] = shard
        exact_records.append(exact_record(reference, wrapped, config, shard=shard))
    tensor_hashes = _tensor_hashes(inputs, labels)
    sources = _source_hashes()
    protocol = protocol_payload()
    if sources != source_hashes_before:
        raise ValueError("PosteriorV8 source changed during holdout evaluation")
    if any(
        file_sha256(run_dir / name) != artifact_hashes[key]
        for name, key in (
            (MODEL_FILE, "model_sha256"),
            (MANIFEST_FILE, "manifest_sha256"),
            (HISTORY_FILE, "history_sha256"),
        )
    ):
        raise ValueError("V3 artifact changed during holdout evaluation")
    if any(
        file_sha256(Path(item["path"])) != item["npz_sha256"]
        or file_sha256(Path(item["path"]).with_suffix(".json"))
        != item["metadata_sha256"]
        for item in dataset.payload["shards"]
    ):
        raise ValueError("immutable V4 shards changed during holdout evaluation")
    payload = {
        "schema": BOUNDS_HOLDOUT_AUDIT_SCHEMA,
        "version": BOUNDS_HOLDOUT_AUDIT_VERSION,
        "study_protocol_version": protocol["protocol_version"],
        "study_protocol_sha256": protocol["protocol_sha256"],
        "status": "complete",
        "interpretation": {
            "scientific_scope": "bounds_first_solution_only_interpolation_pilot",
            "split_role": (
                "threshold_selection_only_not_final_test"
                if config.split == "calibration"
                else (
                    "interpolation_test_diagnostic_not_formal_compatibility_"
                    "acceptance_without_calibration_artifact"
                )
            ),
            "continuous_coordinates": "native_local_unit_of_actual_user_bounds_codec",
            "exact_forward": "authoritative_gui_consistent_profiled_forward_and_refinement",
            "compatibility_threshold_source": "unfrozen_engineering_thresholds",
            "compatibility_calibration_artifact_sha256": None,
            "formal_paper_compatibility_claim_allowed": False,
            "finite_search_failure_is_no_solution": False,
            "primary_exact_success_semantics": (
                "exact_compatible_candidate_and_pre_observability_parameter_modes"
            ),
            "observability_metrics_role": "secondary_diagnostic_not_success_gate",
            "observability": (
                "verified_exact_pipeline_assessment_when_exact_subset_enabled;"
                " provisional_or_unknown_statuses_are_retained"
            ),
        },
        "config": asdict(config),
        "artifact": {
            "training_run": str(run_dir),
            **artifact_hashes,
            "resume_contract_sha256": manifest["resume_contract_sha256"],
            "model_version": manifest["model_version"],
            "branch_catalog_version": CANONICAL_BRANCH_CATALOG_VERSION,
            "component_slots_version": CANONICAL_COMPONENT_SLOTS_VERSION,
            "training_source_sha256": manifest["source_sha256"],
            "training_source_sha256_aggregate": manifest["source_sha256_aggregate"],
        },
        "selection": {
            "split": config.split,
            "range_selection": config.range_selection,
            "statistical_unit": "independent_clean_physical_recipe",
            "observation_view_policy": "one_deterministic_view_per_recipe",
            "eligible_row_count": eligible_row_count,
            "eligible_independent_recipe_count": eligible_recipe_count,
            "evaluated_row_count": len(records),
            "evaluated_independent_recipe_count": len(records),
            "exact_evaluated_row_count": len(exact_records),
            "train_rows_evaluated": 0,
            "tuning_validation_rows_evaluated": 0,
            "calibration_rows_evaluated": (
                len(records) if config.split == "calibration" else 0
            ),
            "test_rows_evaluated": len(records) if config.split == "test" else 0,
            "selection_sha256": sha256(
                _canonical_json(
                    [
                        [item.recipe_index, item.view_index, item.priority]
                        for item in references
                    ]
                )
            ).hexdigest(),
            "exact_selection_sha256": sha256(
                _canonical_json(
                    [
                        [item.recipe_index, item.view_index, item.priority]
                        for item in references[: config.exact_maximum_examples]
                    ]
                )
            ).hexdigest(),
        },
        "immutable_inputs": {
            "dataset_fingerprint_sha256": dataset.payload["fingerprint_sha256"],
            "dataset_split_counts": dataset.payload["split_counts"],
            "shards": dataset.payload["shards"],
            "selected_tensor_sha256": tensor_hashes,
            "selected_tensor_sha256_aggregate": sha256(
                _canonical_json(tensor_hashes)
            ).hexdigest(),
        },
        "supervised_proposal": {
            "summary": supervised_summary(records, config),
            "examples": records,
        },
        "exact_acceptance": {
            "summary": exact_summary(exact_records, config),
            "examples": exact_records,
            "evaluation_schema": EVALUATION_AUDIT_SCHEMA,
        },
        "audit_source_sha256": sources,
        "audit_source_scope": (
            "all PosteriorV8 Python sources plus the bounds-holdout Slurm wrapper"
        ),
        "audit_source_sha256_aggregate": sha256(_canonical_json(sources)).hexdigest(),
    }
    _write_json_exclusive(Path(output).resolve(), payload)
    return payload


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=SUPPORTED_HOLDOUT_SPLITS, default="test")
    parser.add_argument(
        "--range-selection", choices=SUPPORTED_RANGE_SELECTIONS, default="all"
    )
    parser.add_argument("--maximum-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topology-ks", type=_csv_ints, default=(1, 3, 8))
    parser.add_argument("--canonical-joint-ks", type=_csv_ints, default=(1, 8, 32))
    parser.add_argument("--oracle-mixture-limit", type=int, default=12)
    parser.add_argument("--oracle-samples-per-mixture", type=int, default=2)
    parser.add_argument("--oracle-best-of-n", type=_csv_ints, default=(1, 4, 12, 24))
    parser.add_argument("--exact-maximum-examples", type=int, default=8)
    parser.add_argument("--target-parameter-mode-count", type=int, default=1)
    parser.add_argument("--forward-evaluation-limit", type=int, default=1024)
    parser.add_argument("--per-candidate-forward-limit", type=int, default=128)
    parser.add_argument("--raw-exact-log-rmse-max", type=float, default=0.10)
    parser.add_argument("--standardized-exact-log-rmse-max", type=float, default=3.0)
    parser.add_argument("--parameter-mode-distance-max", type=float, default=0.08)
    parser.add_argument("--curve-equivalence-log-rmse-max", type=float, default=0.02)
    parser.add_argument("--reference-mode-distance-max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budget = InferenceBudget(
        topology_beam_size=8,
        branch_beam_size=8,
        mixture_components_per_branch=4,
        samples_per_mixture=2,
        per_candidate_forward_evaluation_limit=args.per_candidate_forward_limit,
        forward_evaluation_limit=args.forward_evaluation_limit,
    )
    thresholds = EvaluationThresholds(
        raw_exact_log_rmse_max=args.raw_exact_log_rmse_max,
        standardized_exact_log_rmse_max=args.standardized_exact_log_rmse_max,
        parameter_mode_distance_max=args.parameter_mode_distance_max,
        raw_curve_equivalence_log_rmse_max=args.curve_equivalence_log_rmse_max,
        reference_mode_distance_max=args.reference_mode_distance_max,
    )
    config = BoundsHoldoutAuditConfig(
        split=args.split,
        range_selection=args.range_selection,
        maximum_examples=args.maximum_examples,
        batch_size=args.batch_size,
        topology_ks=args.topology_ks,
        canonical_joint_ks=args.canonical_joint_ks,
        oracle_mixture_limit=args.oracle_mixture_limit,
        oracle_samples_per_mixture=args.oracle_samples_per_mixture,
        oracle_best_of_n=args.oracle_best_of_n,
        exact_maximum_examples=args.exact_maximum_examples,
        target_parameter_mode_count=args.target_parameter_mode_count,
        seed=args.seed,
        inference_budget=budget,
        thresholds=thresholds,
    )
    payload = run_bounds_holdout_audit(
        args.shards, args.training_run, args.output, config
    )
    print(
        json.dumps(
            {
                "split": config.split,
                "supervised": payload["supervised_proposal"]["summary"],
                "exact": payload["exact_acceptance"]["summary"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BOUNDS_HOLDOUT_AUDIT_SCHEMA",
    "BOUNDS_HOLDOUT_AUDIT_VERSION",
    "BoundsHoldoutAuditConfig",
    "SUPPORTED_HOLDOUT_SPLITS",
    "build_parser",
    "main",
    "run_bounds_holdout_audit",
]
