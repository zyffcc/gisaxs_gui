"""Write-free plan for a balanced twelve-branch K1 training seed array."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from .k1_training_chain_contract_v5 import (
    V5_K1_FORMAL_MINIMUM_MODEL_SEEDS,
    V5_K1_TRAINING_CHAIN_SCHEMA,
    V5_K1_TRAINING_CHAIN_VERSION,
    V5_K1_TRAINING_MODES,
    canonical_json,
    digest,
    positive_integer,
    v5_k1_training_runtime_capabilities,
    validate_v5_k1_training_inventory,
)
from .k1_staging_files_v5 import file_sha256, read_regular_bytes
from .package_source_snapshot_v5 import (
    SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA,
    SOURCE_SNAPSHOT_EXTRACTED_TREE_VERSION,
    SOURCE_TREE_HASH_SEMANTICS,
    verify_extracted_source_snapshot,
)


MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
K1_TRAINING_PLAN_FILENAME = "k1-training-plan-v1.json"
K1_TRAINING_LAUNCH_RECEIPT_FILENAME = "k1-training-launch-receipt-v1.json"
K1_TRAINING_HANDOFF_FILENAME = "k1-tuning-handoff-v1.json"
POSTERIOR_ROOT_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
GPU_WRAPPER_RELATIVE = POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_training_seed_gpu4.sbatch"
COLLECT_WRAPPER_RELATIVE = POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_training_collect_cpu.sbatch"
K1_TRAINING_REQUIRED_SOURCE_FILES = (
    POSTERIOR_ROOT_RELATIVE / "package_source_snapshot_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_training_chain_contract_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_training_chain_dataset_audit_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_staging_files_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_job_staging_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_input_closure_v5.py",
    POSTERIOR_ROOT_RELATIVE / "job_local_input_capability_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_staging_receipt_inputs_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_staging_receipt_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_training_chain_plan_v5.py",
    POSTERIOR_ROOT_RELATIVE / "k1_training_chain_runtime_v5.py",
    POSTERIOR_ROOT_RELATIVE / "launch_k1_training_chain_v5.py",
    POSTERIOR_ROOT_RELATIVE / "grouped_training_v5.py",
    POSTERIOR_ROOT_RELATIVE / "grouped_training_shards_v5.py",
    POSTERIOR_ROOT_RELATIVE / "grouped_training_data_v5.py",
    POSTERIOR_ROOT_RELATIVE / "train_grouped_v5.py",
    POSTERIOR_ROOT_RELATIVE / "model_v5.py",
    POSTERIOR_ROOT_RELATIVE / "model_v5_contract.py",
    POSTERIOR_ROOT_RELATIVE / "k1_phase_c_contract_v5.py",
    POSTERIOR_ROOT_RELATIVE / "sobol_recipe_coordinates_v5.py",
    POSTERIOR_ROOT_RELATIVE / "amplitude_query_sampling_v5.py",
    GPU_WRAPPER_RELATIVE,
    COLLECT_WRAPPER_RELATIVE,
)
_SOURCE_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True, kw_only=True)
class V5K1TrainingChainConfig:
    source_root: Path
    source_archive: Path
    expected_source_archive_sha256: str
    input_inventory: Path
    expected_inventory_file_sha256: str
    run_root: Path
    mode: str
    model_seeds: tuple[int, ...]
    warmup_epochs: int = 10
    full_epochs: int = 0
    recipes_per_replica: int = 4
    validation_recipes_per_batch: int = 16
    width: int = 128
    encoder_blocks: int = 6
    mixture_components: int = 12
    learning_rate: float = 1.0e-4
    mixed_precision: bool = True


def _file_sha256(path: Path) -> str:
    return file_sha256(path, "K1 frozen input")


def _under_root(path: Path, root: Path, name: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    allowed = root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {allowed}") from exc
    if not relative.parts:
        raise ValueError(f"{name} must not be the allowed root")
    current = allowed
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink")
        if not current.exists():
            break
    resolved = lexical.resolve(strict=must_exist)
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"{name} resolves outside {allowed}")
    return resolved


def fingerprint_v5_k1_training_source(source_root: Path) -> dict[str, object]:
    root = source_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source_root must be a real directory")
    missing = [
        value.as_posix()
        for value in K1_TRAINING_REQUIRED_SOURCE_FILES
        if not (root / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"immutable source snapshot is incomplete: {missing}")
    identities = {}
    bundle = sha256()
    write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    writable_files = []
    for relative in K1_TRAINING_REQUIRED_SOURCE_FILES:
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"source snapshot contains a symlink: {relative}")
        if stat.S_IMODE(path.stat().st_mode) & write_mask:
            writable_files.append(relative.as_posix())
        file_sha = _file_sha256(path)
        identities[relative.as_posix()] = file_sha
        encoded = relative.as_posix().encode("utf-8")
        bundle.update(len(encoded).to_bytes(4, "big"))
        bundle.update(encoded)
        bundle.update(bytes.fromhex(file_sha))
    return {
        "root": str(root),
        "bundle_sha256": bundle.hexdigest(),
        "required_file_sha256": identities,
        "source_snapshot_write_bits_set": bool(
            stat.S_IMODE(root.stat().st_mode) & write_mask
        ),
        "source_files_with_write_bits": writable_files,
    }


def _inventory(path: Path) -> tuple[dict[str, object], str]:
    encoded = read_regular_bytes(
        path,
        "K1 input inventory",
        maximum_bytes=32 * 1024 * 1024,
    )
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input inventory is not valid JSON") from exc
    return validate_v5_k1_training_inventory(value), sha256(encoded).hexdigest()


def _seeds(values: Sequence[int], mode: str) -> tuple[int, ...]:
    supplied = tuple(values)
    result = []
    for index, value in enumerate(supplied):
        seed = positive_integer(value, f"model_seeds[{index}]")
        if seed >= 2**31:
            raise ValueError("model seeds must fit in signed int32")
        result.append(seed)
    selected = tuple(result)
    if len(selected) != len(set(selected)):
        raise ValueError("model seeds must be unique")
    if mode == "engineering_e1" and len(selected) != 1:
        raise ValueError("engineering_e1 requires exactly one model seed")
    if mode == "formal_multiseed" and len(selected) < V5_K1_FORMAL_MINIMUM_MODEL_SEEDS:
        raise ValueError("formal_multiseed requires at least five model seeds")
    return selected


def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _artifact_paths(inventory: Mapping[str, object]) -> dict[str, list[str]]:
    result = {"train": [], "tuning_validation": []}
    for role in result:
        result[role] = [value["path"] for value in inventory["artifacts"][role]]
    return result


def _sidecar_paths(inventory: Mapping[str, object]) -> dict[str, list[str]]:
    result = {"train": [], "tuning_validation": []}
    for role in result:
        result[role] = [
            value["sidecar_path"]
            for value in inventory["artifacts"][role]
            if value["sidecar_path"] is not None
        ]
    return result


def _validate_input_paths(inventory: Mapping[str, object], allowed_root: Path) -> None:
    for role in ("train", "tuning_validation"):
        for artifact in inventory["artifacts"][role]:
            for name in ("path", "sidecar_path", "evidence_receipt_path"):
                raw = artifact[name]
                if raw is None:
                    continue
                path = _under_root(Path(raw), allowed_root, f"{role}.{name}", must_exist=True)
                if not path.is_file():
                    raise ValueError(f"{role}.{name} must be a regular file")


def _layout(run_root: Path) -> dict[str, str]:
    return {
        "run_root": str(run_root),
        "logs": str(run_root / "logs"),
        "audit": str(run_root / "audit"),
        "models": str(run_root / "models"),
        "plan": str(run_root / "audit" / K1_TRAINING_PLAN_FILENAME),
        "launch_receipt": str(run_root / "audit" / K1_TRAINING_LAUNCH_RECEIPT_FILENAME),
        "tuning_handoff": str(run_root / "audit" / K1_TRAINING_HANDOFF_FILENAME),
    }


def _seed_runs(run_root: Path, seeds: Sequence[int]) -> list[dict[str, object]]:
    return [
        {
            "array_index": index,
            "model_seed": seed,
            "output": str(run_root / "models" / f"seed-{seed:010d}"),
            "binding_receipt": str(
                run_root / "models" / f"seed-{seed:010d}" / "k1-seed-binding-v1.json"
            ),
        }
        for index, seed in enumerate(seeds)
    ]


def _formal_blockers(inventory: Mapping[str, object], mode: str) -> list[str]:
    blockers: list[str] = []
    if mode != "formal_multiseed":
        return blockers
    supervision = inventory["full_search_supervision"]
    if not supervision["all_artifacts_declare_training_eligible"]:
        blockers.append("input inventory has no fully promoted search supervision")
    blockers.extend(supervision["current_runtime_capabilities"]["blocked_interfaces"])
    return blockers


def _job_local_input_security_contract() -> dict[str, object]:
    return {
        "status": "closed_by_live_wrapper_minted_capability_v2",
        "capability_is_process_live_and_single_use": True,
        "serialized_capability_or_receipt_authorizes_training": False,
        "wrapper_created_owner_private_job_tmp_root_required": True,
        "wrapper_mint_is_exclusive_one_shot_and_live_consumed": True,
        "capability_has_no_public_factory": True,
        "mint_token_seal_and_digest_are_never_audited": True,
        "shared_scratch_base_alone_authorizes_training": False,
        "original_and_job_local_bytes_rehashed_pre_and_post_training": True,
        "source_archive_manifest_and_tree_reverified_pre_and_post_training": True,
        "trainer_arguments_must_equal_verified_local_copy_set": True,
        "output_remains_restricted_to_user_dust": True,
    }


def _scientific_role(mode: str) -> str:
    return (
        "engineering_all12_k1_warmup_and_tuning_validation_diagnostic"
        if mode == "engineering_e1"
        else "formal_multiseed_k1_training_and_exact_budget_tuning_candidate_chain"
    )


def _slurm_contract(source_root: Path, run_count: int) -> dict[str, object]:
    return {
        "training_array_spec": f"0-{run_count - 1}",
        "training_wrapper": str(source_root / GPU_WRAPPER_RELATIVE),
        "collection_wrapper": str(source_root / COLLECT_WRAPPER_RELATIVE),
        "training_resources": {
            "partition": "allgpu",
            "account": "hasylab",
            "gpus_per_task": 4,
            "cpus_per_task": 16,
            "memory_gib": 128,
            "walltime_hours": 12,
        },
        "collection_resources": {
            "partition": "allcpu",
            "cpus_per_task": 1,
            "memory_gib": 4,
            "walltime_minutes": 30,
        },
        "collection_dependency": "afterok:training_seed_array",
    }


def _resource_estimate(run_count: int) -> dict[str, object]:
    return {
        "training_job_count": run_count,
        "maximum_allocated_gpu_hours": 4 * 12 * run_count,
        "maximum_allocated_cpu_core_hours_for_training": 16 * 12 * run_count,
        "exact_budget_tuning_runtime_included": False,
        "estimate_scope": "Slurm_allocation_ceiling_not_queue_or_measured_runtime",
    }


def _claim_limits() -> dict[str, bool]:
    return {
        "phase_a_is_full_k1_evidence": False,
        "engineering_warmup_trains_search_yield_ranking": False,
        "validation_objective_selects_paper_checkpoint": False,
        "tuning_handoff_is_checkpoint_selection": False,
        "chain_completion_is_k1_phase_c_pass": False,
        "chain_completion_is_paper_model_acceptance": False,
    }


def build_v5_k1_training_chain_plan(
    config: V5K1TrainingChainConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Validate immutable inputs and return a deterministic submission plan."""

    if not isinstance(config, V5K1TrainingChainConfig):
        raise TypeError("config must be V5K1TrainingChainConfig")
    if config.mode not in V5_K1_TRAINING_MODES:
        raise ValueError(f"mode must be one of {V5_K1_TRAINING_MODES}")
    seeds = _seeds(config.model_seeds, config.mode)
    warmup_epochs = positive_integer(config.warmup_epochs, "warmup_epochs")
    if isinstance(config.full_epochs, bool) or not isinstance(config.full_epochs, int):
        raise TypeError("full_epochs must be an integer")
    if config.full_epochs < 0:
        raise ValueError("full_epochs must be non-negative")
    if config.mode == "engineering_e1" and config.full_epochs != 0:
        raise ValueError("engineering_e1 is restricted to warmup-only training")
    if config.mode == "formal_multiseed" and config.full_epochs < 1:
        raise ValueError("formal_multiseed requires at least one full epoch")
    integers = {
        "recipes_per_replica": config.recipes_per_replica,
        "validation_recipes_per_batch": config.validation_recipes_per_batch,
        "width": config.width,
        "encoder_blocks": config.encoder_blocks,
        "mixture_components": config.mixture_components,
    }
    integers = {name: positive_integer(value, name) for name, value in integers.items()}
    learning_rate = _positive_float(config.learning_rate, "learning_rate")
    if type(config.mixed_precision) is not bool:
        raise TypeError("mixed_precision must be a bool")

    source_root = _under_root(config.source_root, allowed_root, "source_root", must_exist=True)
    source = fingerprint_v5_k1_training_source(source_root)
    if source["source_snapshot_write_bits_set"] or source["source_files_with_write_bits"]:
        raise ValueError("source_root and required files must be an immutable read-only snapshot")
    source_archive = _under_root(
        config.source_archive, allowed_root, "source_archive", must_exist=True
    )
    if not source_archive.is_file() or not any(
        source_archive.name.endswith(value) for value in _SOURCE_ARCHIVE_SUFFIXES
    ):
        raise ValueError("source_archive must be a regular .tar, .tar.gz, or .tgz file")
    expected_archive = digest(
        config.expected_source_archive_sha256, "expected_source_archive_sha256"
    )
    actual_archive = _file_sha256(source_archive)
    if actual_archive != expected_archive:
        raise ValueError("source archive SHA-256 does not match the explicit expectation")
    archive_binding = verify_extracted_source_snapshot(
        source_archive,
        source_root,
        expected_archive_sha256=expected_archive,
    )
    if (
        archive_binding["schema_version"] != SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA
        or archive_binding["version"] != SOURCE_SNAPSHOT_EXTRACTED_TREE_VERSION
        or archive_binding["archive_sha256"] != actual_archive
        or archive_binding["source_root"] != str(source_root)
        or archive_binding["exact_manifest_file_set_verified"] is not True
        or archive_binding["exact_manifest_directory_set_verified"] is not True
        or archive_binding["symlink_free_lexical_paths_verified"] is not True
        or archive_binding["read_only_tree_verified"] is not True
        or archive_binding["verified"] is not True
    ):
        raise RuntimeError("authoritative source archive/tree binding is incomplete")
    inventory_path = _under_root(
        config.input_inventory, allowed_root, "input_inventory", must_exist=True
    )
    if not inventory_path.is_file():
        raise ValueError("input_inventory must be a regular file")
    inventory, inventory_file_sha = _inventory(inventory_path)
    if inventory_file_sha != digest(
        config.expected_inventory_file_sha256, "expected_inventory_file_sha256"
    ):
        raise ValueError("input inventory file SHA-256 does not match the explicit expectation")
    if inventory["source_archive_sha256"] != actual_archive:
        raise ValueError("input inventory is bound to another source archive")
    if inventory["source_bundle_sha256"] != source["bundle_sha256"]:
        raise ValueError("input inventory is bound to another source bundle")
    _validate_input_paths(inventory, allowed_root)

    run_root = _under_root(config.run_root, allowed_root, "run_root", must_exist=False)
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("refusing to reuse a K1 training chain run root")
    if run_root == source_root or run_root.is_relative_to(source_root):
        raise ValueError("run_root cannot be inside the immutable source snapshot")
    layout = _layout(run_root)
    seed_runs = _seed_runs(run_root, seeds)
    blockers = _formal_blockers(inventory, config.mode)
    submission_allowed = not blockers
    artifact_paths = _artifact_paths(inventory)
    sidecar_paths = _sidecar_paths(inventory)
    core = {
        "schema": V5_K1_TRAINING_CHAIN_SCHEMA,
        "version": V5_K1_TRAINING_CHAIN_VERSION,
        "mode": config.mode,
        "scientific_role": _scientific_role(config.mode),
        "source": {
            **source,
            "archive_path": str(source_archive),
            "archive_sha256": actual_archive,
            "archive_tree_binding": archive_binding,
        },
        "input_inventory": {
            "path": str(inventory_path),
            "file_sha256": inventory_file_sha,
            "inventory_sha256": inventory["inventory_sha256"],
            "dataset_manifest_bundle_sha256": inventory["dataset_contract"][
                "dataset_manifest_bundle_sha256"
            ],
            "coordinate_contract_sha256": inventory["coordinate_contract"]["sha256"],
            "amplitude_range_assignment_contract": inventory["amplitude_range_assignment_contract"],
            "k1_phase_c_disjointness_receipt_sha256": inventory["splits"][
                "k1_phase_c_disjointness_receipt_sha256"
            ],
        },
        "layout": layout,
        "configuration": {
            "model_seeds": list(seeds),
            "minimum_formal_model_seeds": V5_K1_FORMAL_MINIMUM_MODEL_SEEDS,
            "warmup_epochs": warmup_epochs,
            "full_epochs": config.full_epochs,
            **integers,
            "learning_rate": learning_rate,
            "mixed_precision": config.mixed_precision,
            "train_split": "train",
            "validation_split": "tuning_validation",
        },
        "inputs": {
            "train_datasets": artifact_paths["train"],
            "validation_datasets": artifact_paths["tuning_validation"],
            "train_sidecars": sidecar_paths["train"],
            "validation_sidecars": sidecar_paths["tuning_validation"],
        },
        "seed_runs": seed_runs,
        "slurm": _slurm_contract(source_root, len(seed_runs)),
        "execution_gate": {
            "submission_allowed": submission_allowed,
            "blockers": blockers,
            "formal_chain_complete": bool(config.mode == "formal_multiseed" and submission_allowed),
            "job_local_input_security": _job_local_input_security_contract(),
            "login_node_work": "hash_contract_path_checks_plan_publication_and_sbatch_only",
            "training_compute": "Slurm_GPU_worker_only",
            "tuning_handoff_compute": "Slurm_CPU_worker_only",
        },
        "resource_estimate": _resource_estimate(len(seed_runs)),
        "claim_limits": _claim_limits(),
    }
    return {**core, "plan_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest()}


def _exact_mapping(
    value: object, fields: set[str], name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    return dict(value)


def validate_v5_k1_training_chain_plan(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("K1 training chain plan must be a mapping")
    expected_top_level = {
        "schema",
        "version",
        "mode",
        "scientific_role",
        "source",
        "input_inventory",
        "layout",
        "configuration",
        "inputs",
        "seed_runs",
        "slurm",
        "execution_gate",
        "resource_estimate",
        "claim_limits",
        "plan_sha256",
    }
    if set(payload) != expected_top_level:
        raise ValueError("K1 training chain plan fields are incomplete or unsupported")
    value = dict(payload)
    supplied = digest(value.pop("plan_sha256", None), "plan_sha256")
    if supplied != sha256(canonical_json(value).encode("utf-8")).hexdigest():
        raise ValueError("K1 training chain plan SHA-256 does not reproduce")
    if (value.get("schema"), value.get("version")) != (
        V5_K1_TRAINING_CHAIN_SCHEMA,
        V5_K1_TRAINING_CHAIN_VERSION,
    ):
        raise ValueError("K1 training chain schema/version drifted")
    mode = value.get("mode")
    if mode not in V5_K1_TRAINING_MODES:
        raise ValueError("K1 training chain mode is unsupported")
    if value.get("scientific_role") != _scientific_role(mode):
        raise ValueError("K1 training chain scientific role drifted")

    source = _exact_mapping(
        value["source"],
        {
            "root",
            "bundle_sha256",
            "required_file_sha256",
            "source_snapshot_write_bits_set",
            "source_files_with_write_bits",
            "archive_path",
            "archive_sha256",
            "archive_tree_binding",
        },
        "source",
    )
    if not all(
        isinstance(source[name], str) and Path(source[name]).is_absolute()
        for name in ("root", "archive_path")
    ):
        raise ValueError("source root and archive path must be absolute strings")
    digest(source["bundle_sha256"], "source bundle SHA-256")
    digest(source["archive_sha256"], "source archive SHA-256")
    identities = _exact_mapping(
        source["required_file_sha256"],
        {item.as_posix() for item in K1_TRAINING_REQUIRED_SOURCE_FILES},
        "required source identities",
    )
    for relative, identity in identities.items():
        digest(identity, f"required source identity {relative}")
    if source["source_snapshot_write_bits_set"] is not False:
        raise ValueError("source snapshot root is not frozen read-only")
    if source["source_files_with_write_bits"] != []:
        raise ValueError("required source files are not frozen read-only")
    binding = source["archive_tree_binding"]
    binding = _exact_mapping(
        binding,
        {
            "schema_version",
            "version",
            "archive_path",
            "archive_sha256",
            "archive_byte_count",
            "manifest_sha256",
            "selected_file_count",
            "source_root",
            "source_tree_sha256",
            "source_tree_file_count",
            "source_tree_hash_semantics",
            "exact_manifest_file_set_verified",
            "exact_manifest_directory_set_verified",
            "symlink_free_lexical_paths_verified",
            "read_only_tree_verified",
            "verified",
        },
        "authoritative source archive/tree binding",
    )
    if (
        binding["schema_version"] != SOURCE_SNAPSHOT_EXTRACTED_TREE_SCHEMA
        or binding["version"] != SOURCE_SNAPSHOT_EXTRACTED_TREE_VERSION
        or binding["archive_path"] != source["archive_path"]
        or binding["archive_sha256"] != source["archive_sha256"]
        or binding["source_root"] != source["root"]
        or binding["source_tree_hash_semantics"] != SOURCE_TREE_HASH_SEMANTICS
        or any(
            binding[name] is not True
            for name in (
                "exact_manifest_file_set_verified",
                "exact_manifest_directory_set_verified",
                "symlink_free_lexical_paths_verified",
                "read_only_tree_verified",
                "verified",
            )
        )
    ):
        raise ValueError("authoritative source archive/tree binding drifted")
    digest(binding["manifest_sha256"], "source manifest SHA-256")
    digest(binding["source_tree_sha256"], "source tree SHA-256")
    for name in ("archive_byte_count", "selected_file_count", "source_tree_file_count"):
        positive_integer(binding[name], name)
    if binding["selected_file_count"] != binding["source_tree_file_count"]:
        raise ValueError("source archive and extracted-tree file counts disagree")

    inventory_identity = _exact_mapping(
        value["input_inventory"],
        {
            "path",
            "file_sha256",
            "inventory_sha256",
            "dataset_manifest_bundle_sha256",
            "coordinate_contract_sha256",
            "amplitude_range_assignment_contract",
            "k1_phase_c_disjointness_receipt_sha256",
        },
        "input inventory identity",
    )
    if not isinstance(inventory_identity["path"], str) or not Path(
        inventory_identity["path"]
    ).is_absolute():
        raise ValueError("input inventory path must be an absolute string")
    for name in (
        "file_sha256",
        "inventory_sha256",
        "dataset_manifest_bundle_sha256",
        "coordinate_contract_sha256",
        "k1_phase_c_disjointness_receipt_sha256",
    ):
        digest(inventory_identity[name], name)
    _exact_mapping(
        inventory_identity["amplitude_range_assignment_contract"],
        {"schema", "version"},
        "amplitude range assignment contract",
    )

    configuration = _exact_mapping(
        value["configuration"],
        {
            "model_seeds",
            "minimum_formal_model_seeds",
            "warmup_epochs",
            "full_epochs",
            "recipes_per_replica",
            "validation_recipes_per_batch",
            "width",
            "encoder_blocks",
            "mixture_components",
            "learning_rate",
            "mixed_precision",
            "train_split",
            "validation_split",
        },
        "configuration",
    )
    seeds = _seeds(configuration["model_seeds"], mode)
    if configuration["minimum_formal_model_seeds"] != V5_K1_FORMAL_MINIMUM_MODEL_SEEDS:
        raise ValueError("minimum formal model seed count drifted")
    positive_integer(configuration["warmup_epochs"], "warmup_epochs")
    full_epochs = configuration["full_epochs"]
    if isinstance(full_epochs, bool) or not isinstance(full_epochs, int) or full_epochs < 0:
        raise ValueError("full_epochs must be a non-negative integer")
    if (mode == "engineering_e1" and full_epochs != 0) or (
        mode == "formal_multiseed" and full_epochs < 1
    ):
        raise ValueError("full_epochs are incompatible with the training mode")
    for name in (
        "recipes_per_replica",
        "validation_recipes_per_batch",
        "width",
        "encoder_blocks",
        "mixture_components",
    ):
        positive_integer(configuration[name], name)
    if isinstance(configuration["learning_rate"], bool) or not isinstance(
        configuration["learning_rate"], (int, float)
    ):
        raise TypeError("learning_rate must be numeric")
    _positive_float(configuration["learning_rate"], "learning_rate")
    if type(configuration["mixed_precision"]) is not bool:
        raise TypeError("mixed_precision must be a bool")
    if (configuration["train_split"], configuration["validation_split"]) != (
        "train",
        "tuning_validation",
    ):
        raise ValueError("training and tuning-validation split identities drifted")

    layout = _exact_mapping(
        value["layout"],
        {"run_root", "logs", "audit", "models", "plan", "launch_receipt", "tuning_handoff"},
        "layout",
    )
    if not isinstance(layout["run_root"], str) or not Path(layout["run_root"]).is_absolute():
        raise ValueError("run_root must be an absolute string")
    if layout != _layout(Path(layout["run_root"])):
        raise ValueError("versioned immutable output layout drifted")
    runs = value.get("seed_runs")
    if not isinstance(runs, list) or runs != _seed_runs(Path(layout["run_root"]), seeds):
        raise ValueError("seed run inventory does not match the configured seeds and layout")

    inputs = _exact_mapping(
        value["inputs"],
        {"train_datasets", "validation_datasets", "train_sidecars", "validation_sidecars"},
        "training inputs",
    )
    for name, items in inputs.items():
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not Path(item).is_absolute() for item in items
        ):
            raise ValueError(f"{name} must be a list of absolute paths")
    if not inputs["train_datasets"] or not inputs["validation_datasets"]:
        raise ValueError("train and tuning-validation dataset lists must be non-empty")
    if len(set((*inputs["train_datasets"], *inputs["validation_datasets"]))) != len(
        inputs["train_datasets"]
    ) + len(inputs["validation_datasets"]):
        raise ValueError("train and tuning-validation dataset paths must be disjoint")

    if value["slurm"] != _slurm_contract(Path(source["root"]), len(seeds)):
        raise ValueError("Slurm resource or wrapper contract drifted")
    gate = _exact_mapping(
        value["execution_gate"],
        {
            "submission_allowed",
            "blockers",
            "formal_chain_complete",
            "job_local_input_security",
            "login_node_work",
            "training_compute",
            "tuning_handoff_compute",
        },
        "execution gate",
    )
    if type(gate["submission_allowed"]) is not bool or type(
        gate["formal_chain_complete"]
    ) is not bool:
        raise TypeError("execution gate booleans are invalid")
    if not isinstance(gate["blockers"], list) or any(
        not isinstance(item, str) or not item for item in gate["blockers"]
    ):
        raise ValueError("execution blockers must be a list of non-empty strings")
    if (
        gate["login_node_work"]
        != "hash_contract_path_checks_plan_publication_and_sbatch_only"
        or gate["training_compute"] != "Slurm_GPU_worker_only"
        or gate["tuning_handoff_compute"] != "Slurm_CPU_worker_only"
        or gate["formal_chain_complete"]
        != (mode == "formal_multiseed" and gate["submission_allowed"])
        or gate["job_local_input_security"] != _job_local_input_security_contract()
    ):
        raise ValueError("execution gate semantics drifted")
    if gate["submission_allowed"] is not (not gate["blockers"]):
        raise ValueError("execution allowance disagrees with its blockers")
    if mode == "engineering_e1":
        if gate["blockers"] != [] or gate["submission_allowed"] is not True:
            raise ValueError("engineering worker security gate is not closed and enabled")
    else:
        runtime_blockers = v5_k1_training_runtime_capabilities()["blocked_interfaces"]
        allowed_blockers = {
            "input inventory has no fully promoted search supervision",
            *runtime_blockers,
        }
        if (
            len(gate["blockers"]) != len(set(gate["blockers"]))
            or not set(runtime_blockers).issubset(gate["blockers"])
            or not set(gate["blockers"]).issubset(allowed_blockers)
        ):
            raise ValueError("formal execution blockers drifted from live capabilities")
    if value["resource_estimate"] != _resource_estimate(len(seeds)):
        raise ValueError("resource estimate drifted from the frozen Slurm allocation")
    if value["claim_limits"] != _claim_limits():
        raise ValueError("K1 training plan claim limits drifted")
    return {**value, "plan_sha256": supplied}


def replay_v5_k1_training_chain_fingerprints(
    plan: Mapping[str, object], *, allowed_root: Path = MAXWELL_DUST_ROOT
) -> dict[str, object]:
    value = validate_v5_k1_training_chain_plan(plan)
    source_root = _under_root(
        Path(value["source"]["root"]), allowed_root, "source_root", must_exist=True
    )
    source = fingerprint_v5_k1_training_source(source_root)
    if source != {
        name: value["source"][name]
        for name in (
            "root",
            "bundle_sha256",
            "required_file_sha256",
            "source_snapshot_write_bits_set",
            "source_files_with_write_bits",
        )
    }:
        raise RuntimeError("source snapshot changed after K1 training planning")
    archive = _under_root(
        Path(value["source"]["archive_path"]),
        allowed_root,
        "source_archive",
        must_exist=True,
    )
    if _file_sha256(archive) != value["source"]["archive_sha256"]:
        raise RuntimeError("source archive changed after K1 training planning")
    expected_binding = value["source"]["archive_tree_binding"]
    archive_binding = verify_extracted_source_snapshot(
        archive,
        source_root,
        expected_archive_sha256=value["source"]["archive_sha256"],
        expected_manifest_sha256=expected_binding["manifest_sha256"],
        expected_source_tree_sha256=expected_binding["source_tree_sha256"],
    )
    if archive_binding != expected_binding:
        raise RuntimeError("authoritative source archive/tree binding changed after planning")
    inventory_path = _under_root(
        Path(value["input_inventory"]["path"]),
        allowed_root,
        "input_inventory",
        must_exist=True,
    )
    inventory, file_sha = _inventory(inventory_path)
    expected = value["input_inventory"]
    replayed_identity = {
        "path": str(inventory_path),
        "file_sha256": file_sha,
        "inventory_sha256": inventory["inventory_sha256"],
        "dataset_manifest_bundle_sha256": inventory["dataset_contract"][
            "dataset_manifest_bundle_sha256"
        ],
        "coordinate_contract_sha256": inventory["coordinate_contract"]["sha256"],
        "amplitude_range_assignment_contract": inventory[
            "amplitude_range_assignment_contract"
        ],
        "k1_phase_c_disjointness_receipt_sha256": inventory["splits"][
            "k1_phase_c_disjointness_receipt_sha256"
        ],
    }
    if replayed_identity != expected:
        raise RuntimeError("K1 training inventory changed after planning")
    if (
        inventory["source_archive_sha256"] != value["source"]["archive_sha256"]
        or inventory["source_bundle_sha256"] != value["source"]["bundle_sha256"]
    ):
        raise RuntimeError("K1 inventory source binding changed after planning")
    _validate_input_paths(inventory, allowed_root)
    expected_inputs = {
        "train_datasets": _artifact_paths(inventory)["train"],
        "validation_datasets": _artifact_paths(inventory)["tuning_validation"],
        "train_sidecars": _sidecar_paths(inventory)["train"],
        "validation_sidecars": _sidecar_paths(inventory)["tuning_validation"],
    }
    if value["inputs"] != expected_inputs:
        raise RuntimeError("training input paths drifted from the frozen inventory")
    blockers = _formal_blockers(inventory, value["mode"])
    expected_allowed = not blockers
    if (
        value["execution_gate"]["blockers"] != blockers
        or value["execution_gate"]["submission_allowed"] is not expected_allowed
    ):
        raise RuntimeError("execution gate drifted from live trainer capabilities")
    return {
        "source": source,
        "source_archive_tree_binding": archive_binding,
        "inventory": inventory,
    }


__all__ = [
    "COLLECT_WRAPPER_RELATIVE",
    "GPU_WRAPPER_RELATIVE",
    "K1_TRAINING_HANDOFF_FILENAME",
    "K1_TRAINING_LAUNCH_RECEIPT_FILENAME",
    "K1_TRAINING_PLAN_FILENAME",
    "K1_TRAINING_REQUIRED_SOURCE_FILES",
    "MAXWELL_DUST_ROOT",
    "V5K1TrainingChainConfig",
    "build_v5_k1_training_chain_plan",
    "fingerprint_v5_k1_training_source",
    "replay_v5_k1_training_chain_fingerprints",
    "validate_v5_k1_training_chain_plan",
]
