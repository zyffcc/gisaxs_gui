"""Worker runtime for one K1 training seed and its tuning handoff."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
from typing import Mapping, Sequence

from .job_local_input_capability_v5 import _mint_job_local_input_capability
from .k1_input_closure_v5 import rehash_staged_artifacts
from .k1_job_staging_v5 import (
    V5_K1_JOB_STAGING_SCHEMA,
    V5_K1_JOB_STAGING_VERSION,
    build_job_staging_proof,
    stage_training_inputs,
    verify_training_artifact_bytes,
)
from .k1_staging_files_v5 import (
    checked_json,
    file_sha256,
    freeze_staging_tree,
    lexical_no_symlinks,
)
from .k1_staging_receipt_v5 import finalize_staging_proof, validate_staging_proof
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import (
    K1_TRAINING_HANDOFF_FILENAME,
    MAXWELL_DUST_ROOT,
    replay_v5_k1_training_chain_fingerprints,
    validate_v5_k1_training_chain_plan,
)


V5_K1_SEED_BINDING_SCHEMA = "gisaxs.posterior_v8.k1_training_seed_binding/v2"
V5_K1_SEED_BINDING_VERSION = (
    "posterior_v8_v5_2_job_private_immutable_training_result_input_binding_v2"
)
V5_K1_TUNING_HANDOFF_SCHEMA = "gisaxs.posterior_v8.k1_tuning_handoff/v2"
V5_K1_TUNING_HANDOFF_VERSION = (
    "posterior_v8_v5_2_job_private_collector_complete_seed_inventory_pending_tuning_v2"
)
K1_SEED_BINDING_FILENAME = "k1-seed-binding-v1.json"
_SEED_BINDING_FIELDS = {
    "schema",
    "version",
    "status",
    "plan_sha256",
    "model_seed",
    "source_archive_sha256",
    "source_bundle_sha256",
    "input_inventory_sha256",
    "dataset_manifest_bundle_sha256",
    "training_result_manifest",
    "training_result_manifest_file_sha256",
    "training_result_sha256",
    "best_model",
    "best_model_sha256",
    "full_checkpoints",
    "checkpoint_selection_status",
    "paper_model_eligible",
    "k1_phase_c_passed",
    "job_local_staging",
    "binding_sha256",
}


def _load_plan(
    path: Path,
    *,
    expected_sha256: str | None = None,
    allow_job_local_copy: bool = False,
) -> dict[str, object]:
    value = validate_v5_k1_training_chain_plan(checked_json(path))
    if expected_sha256 is not None and value["plan_sha256"] != digest(
        expected_sha256, "expected_plan_sha256"
    ):
        raise ValueError("runtime plan SHA-256 differs from the Slurm export binding")
    if not allow_job_local_copy and str(path.resolve()) != value["layout"]["plan"]:
        raise ValueError("runtime plan path is not its frozen launch location")
    return value


def _worker_guard(*, dry_run: bool, hostname: str, environment: Mapping[str, str]) -> None:
    if dry_run:
        return
    if hostname.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 training runtime is forbidden on the Maxwell login node")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or int(job_id) < 1:
        raise RuntimeError("K1 training runtime requires a Slurm worker allocation")


def verify_v5_k1_training_artifact_bytes(
    plan: Mapping[str, object], *, allowed_root: Path = MAXWELL_DUST_ROOT
) -> dict[str, object]:
    return verify_training_artifact_bytes(plan, allowed_root=allowed_root)


def _seed_run(plan: Mapping[str, object], array_index: int) -> Mapping[str, object]:
    if isinstance(array_index, bool) or not isinstance(array_index, int):
        raise TypeError("array_index must be an integer")
    runs = plan["seed_runs"]
    if not 0 <= array_index < len(runs):
        raise ValueError("array_index is outside the frozen model-seed schedule")
    selected = runs[array_index]
    if selected["array_index"] != array_index:
        raise RuntimeError("model-seed array mapping does not replay")
    return selected


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _validated_result_manifest(path: Path) -> dict[str, object]:
    result = checked_json(path)
    core = dict(result)
    supplied = digest(core.pop("result_sha256", None), "training result SHA-256")
    if supplied != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("training result manifest SHA-256 does not reproduce")
    if result.get("status") != "complete":
        raise ValueError("training result manifest is not complete")
    return result


def run_v5_k1_training_seed(
    plan_path: str | os.PathLike[str],
    array_index: int,
    *,
    expected_plan_sha256: str | None = None,
    dry_run: bool = False,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
    job_staging_root: str | os.PathLike[str] | None = None,
    job_source_root: str | os.PathLike[str] | None = None,
    job_source_archive: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Validate one seed binding and optionally invoke the existing grouped trainer."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    _worker_guard(dry_run=dry_run, hostname=host, environment=env)
    staging_values = (job_staging_root, job_source_root, job_source_archive)
    if not dry_run and any(value is None for value in staging_values):
        raise RuntimeError("non-dry-run K1 training requires complete job-private staging")
    path = lexical_no_symlinks(Path(plan_path), "runtime K1 plan").resolve(strict=True)
    plan = _load_plan(
        path,
        expected_sha256=expected_plan_sha256,
        allow_job_local_copy=not dry_run,
    )
    selected = _seed_run(plan, array_index)
    output = Path(selected["output"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite an existing K1 seed output")
    if dry_run:
        verification = verify_v5_k1_training_artifact_bytes(plan, allowed_root=allowed_root)
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "model_seed": selected["model_seed"],
            "input_verification": verification,
            "paper_model_eligible": False,
        }
    if not plan["execution_gate"]["submission_allowed"]:
        raise RuntimeError("K1 training plan is blocked and cannot execute")

    common_staging = build_job_staging_proof(
        plan,
        plan_path=path,
        staging_root=Path(job_staging_root),
        source_root=Path(job_source_root),
        source_archive=Path(job_source_archive),
        worker_kind="training_seed",
        array_index=array_index,
        environment=env,
        allowed_root=allowed_root,
    )
    staged = stage_training_inputs(
        plan,
        staging_root=Path(job_staging_root),
        allowed_root=allowed_root,
    )
    freeze_staging_tree(Path(job_staging_root))

    from .grouped_training_data_v5 import V5GroupedTrainingConfig
    from .grouped_training_v5 import train_v5_grouped_model

    config = plan["configuration"]
    training = V5GroupedTrainingConfig(
        warmup_epochs=config["warmup_epochs"],
        full_epochs=config["full_epochs"],
        recipes_per_replica=config["recipes_per_replica"],
        validation_recipes_per_batch=config["validation_recipes_per_batch"],
        learning_rate=config["learning_rate"],
        seed=selected["model_seed"],
        width=config["width"],
        encoder_blocks=config["encoder_blocks"],
        mixture_components=config["mixture_components"],
        mixed_precision=config["mixed_precision"],
        train_split=config["train_split"],
        validation_split=config["validation_split"],
    )
    sidecars = staged["local_inputs"]
    ordered_arguments = (
        *sidecars["train_datasets"],
        *sidecars["validation_datasets"],
        *sidecars["train_sidecars"],
        *sidecars["validation_sidecars"],
    )
    capability = _mint_job_local_input_capability(
        plan,
        common_staging,
        staged,
        environment=env,
        allowed_root=allowed_root,
    )
    if tuple(capability.audit_payload()["trainer_argument_local_paths"]) != tuple(
        ordered_arguments
    ):
        raise RuntimeError("wrapper mint returned a different trainer argument set")
    staged["trainer_staging_capability"] = capability.audit_payload()
    result = train_v5_grouped_model(
        sidecars["train_datasets"],
        sidecars["validation_datasets"],
        output,
        training,
        train_sidecar_paths=sidecars["train_sidecars"] or None,
        validation_sidecar_paths=sidecars["validation_sidecars"] or None,
        job_local_input_capability=capability,
    )
    if result.output_dir.resolve(strict=True) != output.resolve(strict=True):
        raise RuntimeError("grouped trainer returned an unexpected output directory")
    manifest = _validated_result_manifest(result.manifest_path)
    post_training_rehash_sha256 = rehash_staged_artifacts(
        staged,
        plan=plan,
        allowed_root=allowed_root,
    )
    staging_proof = finalize_staging_proof(
        common_staging,
        plan=plan,
        staged_inputs=staged,
        trainer_manifest=manifest,
        trainer_capability=capability,
        post_training_rehash_sha256=post_training_rehash_sha256,
        allowed_root=allowed_root,
    )
    staging_proof = validate_staging_proof(
        staging_proof,
        plan,
        worker_kind="training_seed",
        expected_array_index=array_index,
        require_live_staging=True,
        environment=env,
        allowed_root=allowed_root,
    )
    result_file_sha = file_sha256(result.manifest_path)
    best_model_sha = file_sha256(result.best_model_path)
    if best_model_sha != digest(manifest.get("best_model_sha256"), "best model SHA-256"):
        raise RuntimeError("grouped trainer best-model bytes disagree with its manifest")
    core = {
        "schema": V5_K1_SEED_BINDING_SCHEMA,
        "version": V5_K1_SEED_BINDING_VERSION,
        "status": "training_complete_tuning_pending",
        "plan_sha256": plan["plan_sha256"],
        "model_seed": selected["model_seed"],
        "source_archive_sha256": plan["source"]["archive_sha256"],
        "source_bundle_sha256": plan["source"]["bundle_sha256"],
        "input_inventory_sha256": plan["input_inventory"]["inventory_sha256"],
        "dataset_manifest_bundle_sha256": plan["input_inventory"]["dataset_manifest_bundle_sha256"],
        "training_result_manifest": str(result.manifest_path),
        "training_result_manifest_file_sha256": result_file_sha,
        "training_result_sha256": digest(manifest.get("result_sha256"), "training result SHA-256"),
        "best_model": str(result.best_model_path),
        "best_model_sha256": best_model_sha,
        "full_checkpoints": manifest.get("full_checkpoints", []),
        "checkpoint_selection_status": result.checkpoint_selection_status,
        "paper_model_eligible": False,
        "k1_phase_c_passed": False,
        "job_local_staging": staging_proof,
    }
    payload = {
        **core,
        "binding_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    _write_json_exclusive(output / K1_SEED_BINDING_FILENAME, payload)
    return payload


def _validate_seed_binding(
    value: Mapping[str, object],
    plan: Mapping[str, object],
    expected_seed: int,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SEED_BINDING_FIELDS:
        raise ValueError("seed binding fields are incomplete or unsupported")
    payload = dict(value)
    supplied = digest(payload.pop("binding_sha256", None), "binding_sha256")
    if supplied != sha256(canonical_json(payload).encode("utf-8")).hexdigest():
        raise ValueError("seed binding SHA-256 does not reproduce")
    if (
        payload.get("schema") != V5_K1_SEED_BINDING_SCHEMA
        or payload.get("version") != V5_K1_SEED_BINDING_VERSION
        or payload.get("status") != "training_complete_tuning_pending"
        or payload.get("plan_sha256") != plan["plan_sha256"]
        or payload.get("model_seed") != expected_seed
        or payload.get("source_archive_sha256") != plan["source"]["archive_sha256"]
        or payload.get("source_bundle_sha256") != plan["source"]["bundle_sha256"]
        or payload.get("input_inventory_sha256")
        != plan["input_inventory"]["inventory_sha256"]
        or payload.get("dataset_manifest_bundle_sha256")
        != plan["input_inventory"]["dataset_manifest_bundle_sha256"]
        or payload.get("paper_model_eligible") is not False
        or payload.get("k1_phase_c_passed") is not False
    ):
        raise ValueError("seed binding identity or claim boundary drifted")
    matching_runs = [item for item in plan["seed_runs"] if item["model_seed"] == expected_seed]
    if len(matching_runs) != 1:
        raise ValueError("seed binding does not map to one frozen output")
    staging = validate_staging_proof(
        payload["job_local_staging"],
        plan,
        worker_kind="training_seed",
        expected_array_index=matching_runs[0]["array_index"],
        allowed_root=allowed_root,
    )
    payload["job_local_staging"] = staging
    output = Path(matching_runs[0]["output"]).resolve(strict=True)
    result_path = Path(payload["training_result_manifest"]).resolve(strict=True)
    if result_path.parent != output:
        raise ValueError("training result manifest escaped the frozen seed output")
    if file_sha256(result_path) != payload["training_result_manifest_file_sha256"]:
        raise ValueError("training result manifest changed after seed binding")
    result = _validated_result_manifest(result_path)
    if result.get("result_sha256") != payload["training_result_sha256"]:
        raise ValueError("training result SHA disagrees with its seed binding")
    best_model = Path(payload["best_model"]).resolve(strict=True)
    if best_model.parent != output or file_sha256(best_model) != payload["best_model_sha256"]:
        raise ValueError("best model escaped or changed after seed binding")
    if result.get("best_model_sha256") != payload["best_model_sha256"]:
        raise ValueError("best model SHA disagrees with the training result")
    if result.get("full_checkpoints") != payload["full_checkpoints"]:
        raise ValueError("full checkpoint inventory disagrees with the training result")
    for checkpoint in payload["full_checkpoints"]:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("full checkpoint record must be an object")
        relative = Path(str(checkpoint.get("relative_path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("full checkpoint has an unsafe relative path")
        checkpoint_path = (output / relative).resolve(strict=True)
        if not checkpoint_path.is_relative_to(output) or file_sha256(
            checkpoint_path
        ) != digest(checkpoint.get("file_sha256"), "full checkpoint SHA-256"):
            raise ValueError("full checkpoint escaped or changed after seed binding")
    selection = result.get("checkpoint_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("status") != payload["checkpoint_selection_status"]
    ):
        raise ValueError("checkpoint selection status disagrees with the training result")
    return {**payload, "binding_sha256": supplied}


def collect_v5_k1_training_handoff(
    plan_path: str | os.PathLike[str],
    *,
    expected_plan_sha256: str | None = None,
    dry_run: bool = False,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
    job_staging_root: str | os.PathLike[str] | None = None,
    job_source_root: str | os.PathLike[str] | None = None,
    job_source_archive: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Collect all seed outputs; never substitute validation loss for paper tuning."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    _worker_guard(dry_run=dry_run, hostname=host, environment=env)
    staging_values = (job_staging_root, job_source_root, job_source_archive)
    if not dry_run and any(value is None for value in staging_values):
        raise RuntimeError("non-dry-run K1 collection requires complete job-private staging")
    path = lexical_no_symlinks(Path(plan_path), "runtime K1 plan").resolve(strict=True)
    plan = _load_plan(
        path,
        expected_sha256=expected_plan_sha256,
        allow_job_local_copy=not dry_run,
    )
    replay_v5_k1_training_chain_fingerprints(plan, allowed_root=allowed_root)
    expected = [Path(value["binding_receipt"]) for value in plan["seed_runs"]]
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "expected_seed_binding_receipts": [str(value) for value in expected],
            "paper_checkpoint_selection_complete": False,
        }
    if not plan["execution_gate"]["submission_allowed"]:
        raise RuntimeError("K1 training plan is blocked and cannot execute")
    common_staging = build_job_staging_proof(
        plan,
        plan_path=path,
        staging_root=Path(job_staging_root),
        source_root=Path(job_source_root),
        source_archive=Path(job_source_archive),
        worker_kind="collector",
        array_index=None,
        environment=env,
        allowed_root=allowed_root,
    )
    freeze_staging_tree(Path(job_staging_root))
    collector_staging = finalize_staging_proof(
        common_staging,
        plan=plan,
        staged_inputs=None,
        trainer_manifest=None,
        allowed_root=allowed_root,
    )
    collector_staging = validate_staging_proof(
        collector_staging,
        plan,
        worker_kind="collector",
        expected_array_index=None,
        require_live_staging=True,
        environment=env,
        allowed_root=allowed_root,
    )
    bindings = [
        _validate_seed_binding(
            checked_json(path),
            plan,
            run["model_seed"],
            allowed_root=allowed_root,
        )
        for path, run in zip(expected, plan["seed_runs"])
    ]
    core = {
        "schema": V5_K1_TUNING_HANDOFF_SCHEMA,
        "version": V5_K1_TUNING_HANDOFF_VERSION,
        "status": "complete_seed_inventory_pending_exact_budget_tuning",
        "plan_sha256": plan["plan_sha256"],
        "mode": plan["mode"],
        "seed_bindings": bindings,
        "selection_split": "tuning_validation",
        "engineering_best_model_role": ("validation_objective_convenience_candidate_only"),
        "paper_checkpoint_selection_complete": False,
        "paper_model_eligible": False,
        "k1_phase_c_passed": False,
        "job_local_staging": collector_staging,
        "required_next_runtime": (
            "produce one V5TuningCheckpointEvaluation per retained full epoch under "
            "equal exact-forward budgets, then invoke paper_checkpoint_selector_v5"
        ),
    }
    payload = {
        **core,
        "handoff_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    target = Path(plan["layout"]["tuning_handoff"])
    if target.name != K1_TRAINING_HANDOFF_FILENAME:
        raise RuntimeError("tuning handoff filename escaped the contract")
    _write_json_exclusive(target, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("train-seed")
    seed.add_argument("--plan", required=True, type=Path)
    seed.add_argument("--expected-plan-sha256", required=True)
    seed.add_argument("--array-index", required=True, type=int)
    seed.add_argument("--job-staging-root", type=Path)
    seed.add_argument("--job-source-root", type=Path)
    seed.add_argument("--job-source-archive", type=Path)
    seed.add_argument("--dry-run", action="store_true")
    collect = subparsers.add_parser("collect")
    collect.add_argument("--plan", required=True, type=Path)
    collect.add_argument("--expected-plan-sha256", required=True)
    collect.add_argument("--job-staging-root", type=Path)
    collect.add_argument("--job-source-root", type=Path)
    collect.add_argument("--job-source-archive", type=Path)
    collect.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train-seed":
        result = run_v5_k1_training_seed(
            args.plan,
            args.array_index,
            expected_plan_sha256=args.expected_plan_sha256,
            dry_run=args.dry_run,
            job_staging_root=args.job_staging_root,
            job_source_root=args.job_source_root,
            job_source_archive=args.job_source_archive,
        )
    else:
        result = collect_v5_k1_training_handoff(
            args.plan,
            expected_plan_sha256=args.expected_plan_sha256,
            dry_run=args.dry_run,
            job_staging_root=args.job_staging_root,
            job_source_root=args.job_source_root,
            job_source_archive=args.job_source_archive,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "K1_SEED_BINDING_FILENAME",
    "V5_K1_SEED_BINDING_SCHEMA",
    "V5_K1_SEED_BINDING_VERSION",
    "V5_K1_JOB_STAGING_SCHEMA",
    "V5_K1_JOB_STAGING_VERSION",
    "V5_K1_TUNING_HANDOFF_SCHEMA",
    "V5_K1_TUNING_HANDOFF_VERSION",
    "collect_v5_k1_training_handoff",
    "main",
    "run_v5_k1_training_seed",
    "verify_v5_k1_training_artifact_bytes",
]
