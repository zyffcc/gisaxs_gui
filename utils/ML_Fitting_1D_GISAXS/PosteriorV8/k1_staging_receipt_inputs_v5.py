"""Validation of persisted K1 training-input receipt claims."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .job_local_input_capability_v5 import (
    V5_JOB_LOCAL_INPUT_CAPABILITY_SCHEMA,
    V5_JOB_LOCAL_INPUT_CAPABILITY_VERSION,
)
from .k1_input_closure_v5 import expected_copy_contract
from .k1_staging_files_v5 import (
    WRITE_BITS,
    checked_zip_manifest,
    existing_under_root,
    read_only_identity,
    regular_identity,
)
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import replay_v5_k1_training_chain_fingerprints


_CAPABILITY_FIELDS = {
    "schema",
    "version",
    "worker_kind",
    "slurm_job_id",
    "slurm_array_task_id",
    "scratch_base",
    "scratch_base_source",
    "scratch_base_device",
    "scratch_base_inode",
    "job_tmp_root",
    "job_tmp_root_device",
    "job_tmp_root_inode",
    "job_tmp_root_uid",
    "job_tmp_root_mode_octal",
    "job_tmp_root_owner_private",
    "wrapper_mint_binding",
    "runtime_cache_root",
    "staging_root",
    "staging_root_device",
    "staging_root_inode",
    "staging_root_owner_private",
    "scientific_plan_sha256",
    "plan_binding",
    "source_binding",
    "input_inventory_sha256",
    "original_to_local_inputs",
    "trainer_argument_local_paths",
    "pre_training_rehash_sha256",
    "authorization",
    "capability_sha256",
}
_ROW_FIELDS = {
    "kind",
    "role",
    "trainer_role",
    "original_path",
    "job_local_path",
    "sha256",
    "byte_count",
    "manifest_sha256",
    "original_device",
    "original_inode",
    "original_mtime_ns",
    "original_ctime_ns",
    "job_local_device",
    "job_local_inode",
    "job_local_mtime_ns",
    "job_local_ctime_ns",
    "job_local_link_count",
    "job_local_mode_octal",
}


def validate_training_input_receipt(
    raw_binding: object,
    raw_result: object,
    plan: Mapping[str, object],
    *,
    root: Path,
    allowed_root: Path,
    require_live: bool,
    common_plan: Mapping[str, object],
    common_source: Mapping[str, object],
    common_staging: Mapping[str, object],
    job_id: str,
    array_index: int,
) -> None:
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
        "inventory",
        "original_to_local_inputs",
        "trainer_argument_local_paths",
        "pre_training_rehash_sha256",
        "post_training_rehash_sha256",
        "pre_post_byte_identity_equal",
        "capability_attestation",
        "serialized_receipt_authorizes_training",
    }:
        raise ValueError("training input receipt fields are incomplete or unsupported")
    if not isinstance(raw_result, Mapping) or set(raw_result) != {
        "status",
        "result_sha256",
        "trainer_plan_sha256",
        "input_artifacts",
    }:
        raise ValueError("trainer result receipt fields are incomplete or unsupported")
    binding = dict(raw_binding)
    if (
        binding["pre_post_byte_identity_equal"] is not True
        or binding["serialized_receipt_authorizes_training"] is not False
        or binding["pre_training_rehash_sha256"]
        != binding["post_training_rehash_sha256"]
    ):
        raise ValueError("training input receipt weakens the TOCTOU boundary")
    pre_rehash = digest(
        binding["pre_training_rehash_sha256"], "pre-training rehash SHA-256"
    )
    digest(binding["post_training_rehash_sha256"], "post-training rehash SHA-256")
    attestation = binding["capability_attestation"]
    if not isinstance(attestation, Mapping) or set(attestation) != {
        "capability",
        "pre_training_rehash_sha256",
        "post_training_rehash_sha256",
        "pre_post_byte_identity_equal",
    }:
        raise ValueError("capability post-training attestation is incomplete")
    if (
        attestation["pre_training_rehash_sha256"] != pre_rehash
        or attestation["post_training_rehash_sha256"] != pre_rehash
        or attestation["pre_post_byte_identity_equal"] is not True
    ):
        raise ValueError("capability attestation disagrees with the TOCTOU receipt")
    capability = attestation["capability"]
    if not isinstance(capability, Mapping) or set(capability) != _CAPABILITY_FIELDS:
        raise ValueError("capability audit fields are incomplete or unsupported")
    capability_core = dict(capability)
    capability_sha = digest(
        capability_core.pop("capability_sha256"), "capability audit SHA-256"
    )
    if capability_sha != sha256(
        canonical_json(capability_core).encode("utf-8")
    ).hexdigest():
        raise ValueError("capability audit SHA-256 does not reproduce")
    if (
        capability["schema"] != V5_JOB_LOCAL_INPUT_CAPABILITY_SCHEMA
        or capability["version"] != V5_JOB_LOCAL_INPUT_CAPABILITY_VERSION
        or capability["worker_kind"] != "training_seed"
        or capability["slurm_job_id"] != job_id
        or capability["slurm_array_task_id"] != array_index
        or capability["scratch_base"] != str(root.parents[1])
        or capability["scratch_base_source"]
        != "POSTERIOR_V8_SCRATCH_BASE"
        or capability["job_tmp_root"] != str(root.parent)
        or capability["runtime_cache_root"] != str(root.parent / "runtime-cache")
        or capability["wrapper_mint_binding"]
        != {
            **common_staging["wrapper_mint"],
            "consumed_into_live_registry": True,
            "token_or_token_digest_disclosed": False,
        }
        or any(
            capability[name] != common_staging[name]
            for name in (
                "scratch_base_device",
                "scratch_base_inode",
                "job_tmp_root_device",
                "job_tmp_root_inode",
                "job_tmp_root_uid",
                "job_tmp_root_mode_octal",
                "job_tmp_root_owner_private",
                "staging_root_device",
                "staging_root_inode",
                "staging_root_owner_private",
            )
        )
        or capability["staging_root"] != str(root)
        or capability["scientific_plan_sha256"] != plan["plan_sha256"]
        or capability["plan_binding"] != common_plan
        or capability["source_binding"] != common_source
        or capability["input_inventory_sha256"]
        != plan["input_inventory"]["inventory_sha256"]
        or capability["pre_training_rehash_sha256"] != pre_rehash
        or capability["authorization"]
        != {
            "live_registry_required": True,
            "single_use": True,
            "serialized_payload_authorizes_use": False,
            "shared_scratch_base_alone_authorizes_use": False,
            "minted_after_complete_live_staging_reverification": True,
        }
    ):
        raise ValueError("capability audit disagrees with the staging proof")

    replay = replay_v5_k1_training_chain_fingerprints(plan, allowed_root=allowed_root)
    live_plan = {**plan, "_live_staging_root": str(root)}
    expected, expected_local_inputs = expected_copy_contract(
        live_plan, replay["inventory"]
    )
    rows = binding["original_to_local_inputs"]
    if (
        not isinstance(rows, list)
        or rows != capability["original_to_local_inputs"]
        or len(rows) != len(expected)
    ):
        raise ValueError("capability input mappings are incomplete")
    for index, (row, frozen) in enumerate(zip(rows, expected)):
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise ValueError("capability input mapping fields drifted")
        digest(row["sha256"], f"capability input {index} SHA-256")
        if row["manifest_sha256"] is not None:
            digest(row["manifest_sha256"], f"capability input {index} manifest")
        for name in (
            "byte_count",
            "original_device",
            "original_inode",
            "original_mtime_ns",
            "original_ctime_ns",
            "job_local_device",
            "job_local_inode",
            "job_local_mtime_ns",
            "job_local_ctime_ns",
            "job_local_link_count",
        ):
            if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
                raise ValueError(f"capability input {index} {name} is invalid")
        if row["job_local_link_count"] != 1:
            raise ValueError("job-local input has an external hard-link claim")
        try:
            local_mode = int(row["job_local_mode_octal"], 8)
        except (TypeError, ValueError) as exc:
            raise ValueError("job-local input mode claim is invalid") from exc
        if local_mode & WRITE_BITS:
            raise ValueError("job-local input receipt claims writable bytes")
        for name in (
            "kind",
            "role",
            "trainer_role",
            "original_path",
            "job_local_path",
            "sha256",
            "manifest_sha256",
        ):
            expected_name = "frozen_sha256" if name == "sha256" else name
            if row[name] != frozen[expected_name]:
                raise ValueError(
                    "capability input mapping disagrees with the frozen plan"
                )
        original = Path(row["original_path"])
        local = Path(row["job_local_path"])
        if not original.is_absolute() or not local.is_absolute():
            raise ValueError("capability input paths must be absolute")
        if not local.is_relative_to(root / "inputs"):
            raise ValueError("capability local input escaped the private input tree")
        existing_under_root(original, allowed_root, f"receipt original input {index}")
        if require_live:
            original_identity = regular_identity(
                original, f"receipt original input {index}"
            )
            local_identity = read_only_identity(
                local, f"receipt job-local input {index}"
            )
            if (
                original_identity["sha256"] != row["sha256"]
                or local_identity["sha256"] != row["sha256"]
                or original_identity["byte_count"] != row["byte_count"]
                or local_identity["byte_count"] != row["byte_count"]
                or original_identity["device"] != row["original_device"]
                or original_identity["inode"] != row["original_inode"]
                or original_identity["mtime_ns"] != row["original_mtime_ns"]
                or original_identity["ctime_ns"] != row["original_ctime_ns"]
                or local_identity["device"] != row["job_local_device"]
                or local_identity["inode"] != row["job_local_inode"]
                or local_identity["mtime_ns"] != row["job_local_mtime_ns"]
                or local_identity["ctime_ns"] != row["job_local_ctime_ns"]
                or local_identity["mode_octal"] != row["job_local_mode_octal"]
            ):
                raise RuntimeError("live capability input identity changed")
            if row["manifest_sha256"] is not None and checked_zip_manifest(local)[
                "manifest_sha256"
            ] != row["manifest_sha256"]:
                raise RuntimeError("live capability manifest identity changed")
    if sha256(canonical_json(rows).encode("utf-8")).hexdigest() != pre_rehash:
        raise ValueError("capability input mapping rehash does not reproduce")
    trainer_paths = tuple(
        value
        for name in (
            "train_datasets",
            "validation_datasets",
            "train_sidecars",
            "validation_sidecars",
        )
        for value in expected_local_inputs[name]
    )
    if (
        binding["trainer_argument_local_paths"] != list(trainer_paths)
        or capability["trainer_argument_local_paths"] != list(trainer_paths)
    ):
        raise ValueError("capability trainer argument set drifted")
    inventory = binding["inventory"]
    if not isinstance(inventory, Mapping) or (
        inventory.get("job_local_path") != rows[0]["job_local_path"]
        or inventory.get("frozen_sha256") != rows[0]["sha256"]
    ):
        raise ValueError("capability inventory receipt drifted")
    result = dict(raw_result)
    if result["status"] != "complete":
        raise ValueError("trainer result receipt is not complete")
    digest(result["result_sha256"], "trainer result SHA-256")
    digest(result["trainer_plan_sha256"], "trainer plan SHA-256")
    trainer_rows = [
        row
        for row in rows
        if row["kind"] in {"grouped_parent", "frozen_search_sidecar"}
    ]
    expected_artifacts = {
        (row["job_local_path"], row["sha256"]) for row in trainer_rows
    }
    result_artifacts = result["input_artifacts"]
    if not isinstance(result_artifacts, list) or {
        (item.get("path"), item.get("artifact_sha256"))
        for item in result_artifacts
        if isinstance(item, Mapping)
    } != expected_artifacts or len(result_artifacts) != len(expected_artifacts):
        raise ValueError("trainer result input artifacts drifted from the capability")


__all__ = ["validate_training_input_receipt"]
