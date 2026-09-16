"""Non-authorizing persisted receipts for K1 job-local staging."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Mapping

from .job_local_input_capability_v5 import (
    _job_local_capability_post_validation_payload,
)
from .k1_input_closure_v5 import rehash_staged_artifacts
from .k1_job_staging_v5 import (
    V5_K1_JOB_STAGING_SCHEMA,
    V5_K1_JOB_STAGING_VERSION,
    rebuild_job_staging_proof,
)
from .k1_staging_receipt_inputs_v5 import validate_training_input_receipt
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import (
    MAXWELL_DUST_ROOT,
    validate_v5_k1_training_chain_plan,
)


_COMMON_FIELDS = {
    "schema",
    "version",
    "worker_kind",
    "slurm_job_id",
    "slurm_array_task_id",
    "staging_root",
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
    "wrapper_mint",
    "runtime_cache_root",
    "staging_root_device",
    "staging_root_inode",
    "staging_root_owner_private",
    "plan",
    "source",
}
_PROOF_FIELDS = _COMMON_FIELDS | {
    "status",
    "input_binding",
    "trainer_result",
    "proof_sha256",
}


def finalize_staging_proof(
    common: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    staged_inputs: Mapping[str, object] | None,
    trainer_manifest: Mapping[str, object] | None,
    trainer_capability: object | None = None,
    post_training_rehash_sha256: str | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Finalize an auditable receipt; the receipt itself grants no authority."""

    value = validate_v5_k1_training_chain_plan(plan)
    if not isinstance(common, Mapping) or set(common) != _COMMON_FIELDS:
        raise TypeError("common staging proof fields are incomplete or unsupported")
    worker_kind = common.get("worker_kind")
    training_values = (
        staged_inputs,
        trainer_manifest,
        trainer_capability,
        post_training_rehash_sha256,
    )
    if worker_kind == "training_seed" and any(item is None for item in training_values):
        raise ValueError("training staging receipt requires its complete live closure")
    if worker_kind == "collector" and any(item is not None for item in training_values):
        raise ValueError("collector staging receipt cannot claim trainer input use")
    if worker_kind not in {"training_seed", "collector"}:
        raise ValueError("staging receipt worker kind is unsupported")

    input_binding = None
    trainer_result = None
    if worker_kind == "training_seed":
        attestation = _job_local_capability_post_validation_payload(trainer_capability)
        post_rehash = digest(
            post_training_rehash_sha256, "post-training input rehash SHA-256"
        )
        live_rehash = rehash_staged_artifacts(
            staged_inputs,
            plan=value,
            allowed_root=allowed_root,
        )
        if (
            post_rehash != live_rehash
            or post_rehash != attestation["pre_training_rehash_sha256"]
            or post_rehash != attestation["post_training_rehash_sha256"]
            or attestation["pre_post_byte_identity_equal"] is not True
        ):
            raise RuntimeError("trainer input closure changed across training")
        if not isinstance(trainer_manifest, Mapping):
            raise TypeError("trainer manifest must be a mapping")
        manifest_core = dict(trainer_manifest)
        result_sha = digest(
            manifest_core.pop("result_sha256", None), "training result SHA-256"
        )
        if result_sha != sha256(
            canonical_json(manifest_core).encode("utf-8")
        ).hexdigest() or trainer_manifest.get("status") != "complete":
            raise ValueError("training result manifest is incomplete or does not reproduce")
        if trainer_manifest.get("job_local_input_capability") != attestation:
            raise RuntimeError("trainer result omitted its live capability attestation")
        rows = attestation["capability"]["original_to_local_inputs"]
        trainer_rows = [
            row
            for row in rows
            if row["kind"] in {"grouped_parent", "frozen_search_sidecar"}
        ]
        expected_artifacts = {
            (row["job_local_path"], row["sha256"]) for row in trainer_rows
        }
        supplied_artifacts = trainer_manifest.get("input_artifacts")
        if not isinstance(supplied_artifacts, list) or {
            (item.get("path"), item.get("artifact_sha256"))
            for item in supplied_artifacts
            if isinstance(item, Mapping)
        } != expected_artifacts or len(supplied_artifacts) != len(expected_artifacts):
            raise RuntimeError(
                "trainer manifest inputs differ from the capability-authorized paths"
            )
        input_binding = {
            "inventory": deepcopy(staged_inputs["inventory"]),
            "original_to_local_inputs": deepcopy(rows),
            "trainer_argument_local_paths": list(
                attestation["capability"]["trainer_argument_local_paths"]
            ),
            "pre_training_rehash_sha256": attestation[
                "pre_training_rehash_sha256"
            ],
            "post_training_rehash_sha256": post_rehash,
            "pre_post_byte_identity_equal": True,
            "capability_attestation": attestation,
            "serialized_receipt_authorizes_training": False,
        }
        trainer_result = {
            "status": "complete",
            "result_sha256": result_sha,
            "trainer_plan_sha256": trainer_manifest.get("plan_sha256"),
            "input_artifacts": deepcopy(supplied_artifacts),
        }
    core = {
        **deepcopy(dict(common)),
        "status": "complete_post_use_reverified",
        "input_binding": input_binding,
        "trainer_result": trainer_result,
    }
    proof = {
        **core,
        "proof_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    return validate_staging_proof(
        proof,
        value,
        worker_kind=worker_kind,
        expected_array_index=common.get("slurm_array_task_id"),
        require_live_staging=False,
        allowed_root=allowed_root,
    )


def validate_staging_proof(
    raw: object,
    plan: Mapping[str, object],
    *,
    worker_kind: str,
    expected_array_index: int | None,
    require_live_staging: bool = False,
    environment: Mapping[str, str] | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Validate receipt claims, optionally replaying every live node file."""

    if type(require_live_staging) is not bool:
        raise TypeError("require_live_staging must be a bool")
    if not isinstance(raw, Mapping) or set(raw) != _PROOF_FIELDS:
        raise TypeError("job-local staging proof fields are incomplete or unsupported")
    payload = deepcopy(dict(raw))
    supplied = digest(payload.pop("proof_sha256"), "staging proof SHA-256")
    if supplied != sha256(canonical_json(payload).encode("utf-8")).hexdigest():
        raise ValueError("job-local staging proof SHA-256 does not reproduce")
    value = validate_v5_k1_training_chain_plan(plan)
    if (
        payload["schema"] != V5_K1_JOB_STAGING_SCHEMA
        or payload["version"] != V5_K1_JOB_STAGING_VERSION
        or payload["status"] != "complete_post_use_reverified"
        or payload["worker_kind"] != worker_kind
        or payload["slurm_array_task_id"] != expected_array_index
        or payload["scratch_base_source"]
        != "POSTERIOR_V8_SCRATCH_BASE"
        or payload["job_tmp_root_owner_private"] is not True
        or payload["staging_root_owner_private"] is not True
    ):
        raise ValueError("job-local staging proof identity drifted")
    job_id = payload["slurm_job_id"]
    if not isinstance(job_id, str) or not job_id.isdigit() or int(job_id) < 1:
        raise ValueError("job-local staging proof has an invalid Slurm job ID")
    scratch = Path(str(payload["scratch_base"]))
    job_tmp = Path(str(payload["job_tmp_root"]))
    root = Path(str(payload["staging_root"]))
    if (
        not scratch.is_absolute()
        or not job_tmp.is_absolute()
        or not root.is_absolute()
        or scratch != Path(os.path.abspath(scratch))
        or job_tmp != Path(os.path.abspath(job_tmp))
        or root != Path(os.path.abspath(root))
        or job_tmp.parent != scratch
        or root != job_tmp / "staging"
        or Path(str(payload["runtime_cache_root"])) != job_tmp / "runtime-cache"
    ):
        raise ValueError("staging root escaped its wrapper-created temporary root")
    allowed = allowed_root.resolve(strict=True)
    if not scratch.is_relative_to(allowed):
        raise ValueError("Slurm temporary root must remain under the user dust root")
    if worker_kind == "training_seed" and (
        isinstance(expected_array_index, bool)
        or not isinstance(expected_array_index, int)
        or expected_array_index < 0
    ):
        raise ValueError("training staging proof has an invalid Slurm array index")
    if worker_kind == "collector" and expected_array_index is not None:
        raise ValueError("collector staging proof cannot claim a Slurm array index")
    prefix = (
        f"gisaxs-v5-k1-seed-{job_id}-{expected_array_index}-"
        if worker_kind == "training_seed"
        else f"gisaxs-v5-k1-collect-{job_id}-"
    )
    if re.fullmatch(re.escape(prefix) + r"[A-Za-z0-9]{8}", job_tmp.name) is None:
        raise ValueError("wrapper-created temporary root escaped its Slurm job identity")
    for name in (
        "scratch_base_device",
        "scratch_base_inode",
        "job_tmp_root_device",
        "job_tmp_root_inode",
        "job_tmp_root_uid",
        "staging_root_device",
        "staging_root_inode",
    ):
        if isinstance(payload[name], bool) or not isinstance(payload[name], int) or payload[name] < 0:
            raise ValueError(f"{name} is invalid")
    try:
        job_tmp_mode = int(payload["job_tmp_root_mode_octal"], 8)
    except (TypeError, ValueError) as exc:
        raise TypeError("job-private temporary root mode is invalid") from exc
    if job_tmp_mode != 0o700 or payload["job_tmp_root_uid"] != os.getuid():
        raise ValueError("staging proof does not bind an owner-private job temp root")
    wrapper_mint = payload["wrapper_mint"]
    if (
        not isinstance(wrapper_mint, Mapping)
        or set(wrapper_mint)
        != {
            "path",
            "byte_count",
            "mode_octal",
            "device",
            "inode",
            "uid",
            "link_count",
            "mtime_ns",
            "ctime_ns",
        }
        or wrapper_mint["path"] != str(job_tmp / ".posterior-v8-wrapper-mint")
        or wrapper_mint["byte_count"] != 32
        or wrapper_mint["mode_octal"] != "0400"
        or wrapper_mint["device"] != payload["job_tmp_root_device"]
        or wrapper_mint["uid"] != payload["job_tmp_root_uid"]
        or wrapper_mint["link_count"] != 1
        or isinstance(wrapper_mint["mtime_ns"], bool)
        or not isinstance(wrapper_mint["mtime_ns"], int)
        or isinstance(wrapper_mint["ctime_ns"], bool)
        or not isinstance(wrapper_mint["ctime_ns"], int)
        or isinstance(wrapper_mint["inode"], bool)
        or not isinstance(wrapper_mint["inode"], int)
        or wrapper_mint["inode"] < 1
    ):
        raise ValueError("staging proof has an invalid wrapper one-shot mint binding")
    common_plan = payload["plan"]
    common_source = payload["source"]
    if not isinstance(common_plan, Mapping) or not isinstance(common_source, Mapping):
        raise TypeError("staging plan/source binding must be mappings")
    expected_plan_fields = {
        "scientific_plan_sha256",
        "original_path",
        "job_local_path",
        "original_file_sha256",
        "job_local_file_sha256",
        "original_byte_count",
        "original_device",
        "original_inode",
        "original_mtime_ns",
        "original_ctime_ns",
        "job_local_byte_count",
        "job_local_device",
        "job_local_inode",
        "job_local_uid",
        "job_local_link_count",
        "job_local_mtime_ns",
        "job_local_ctime_ns",
        "byte_for_byte_copy_verified",
        "mode_octal",
    }
    expected_source_fields = {
        "original_archive_path",
        "job_local_archive_path",
        "archive_sha256",
        "archive_mode_octal",
        "original_archive_byte_count",
        "original_archive_device",
        "original_archive_inode",
        "original_archive_mtime_ns",
        "original_archive_ctime_ns",
        "job_local_archive_byte_count",
        "job_local_archive_device",
        "job_local_archive_inode",
        "job_local_archive_uid",
        "job_local_archive_link_count",
        "job_local_archive_mtime_ns",
        "job_local_archive_ctime_ns",
        "byte_for_byte_archive_copy_verified",
        "original_source_root",
        "job_local_source_root",
        "source_bundle_sha256",
        "manifest_sha256",
        "source_tree_sha256",
        "job_local_source_file_identities",
        "job_local_source_live_identity_sha256",
        "exact_tree_verified",
        "runtime_imported_from_job_local_tree",
    }
    if set(common_plan) != expected_plan_fields or set(common_source) != expected_source_fields:
        raise ValueError("staging plan/source binding fields drifted")
    if (
        common_plan["scientific_plan_sha256"] != value["plan_sha256"]
        or common_plan["original_path"] != value["layout"]["plan"]
        or common_plan["original_file_sha256"] != common_plan["job_local_file_sha256"]
        or common_plan["original_byte_count"] != common_plan["job_local_byte_count"]
        or common_plan["job_local_link_count"] != 1
        or common_plan["byte_for_byte_copy_verified"] is not True
        or common_plan["job_local_path"] != str(root / "k1-training-plan.json")
        or common_source["original_archive_path"] != value["source"]["archive_path"]
        or common_source["job_local_archive_path"] != str(root / "source-snapshot.tar")
        or common_source["job_local_source_root"] != str(root / "source")
        or common_source["original_source_root"] != value["source"]["root"]
        or common_source["archive_sha256"] != value["source"]["archive_sha256"]
        or common_source["original_archive_byte_count"]
        != common_source["job_local_archive_byte_count"]
        or common_source["job_local_archive_link_count"] != 1
        or common_source["source_bundle_sha256"] != value["source"]["bundle_sha256"]
        or common_source["manifest_sha256"]
        != value["source"]["archive_tree_binding"]["manifest_sha256"]
        or common_source["source_tree_sha256"]
        != value["source"]["archive_tree_binding"]["source_tree_sha256"]
        or common_source["byte_for_byte_archive_copy_verified"] is not True
        or common_source["exact_tree_verified"] is not True
        or common_source["runtime_imported_from_job_local_tree"] is not True
    ):
        raise ValueError("staging plan/source binding disagrees with the K1 plan")
    for name in ("original_file_sha256", "job_local_file_sha256"):
        digest(common_plan[name], name)
    for name in (
        "archive_sha256",
        "source_bundle_sha256",
        "manifest_sha256",
        "source_tree_sha256",
        "job_local_source_live_identity_sha256",
    ):
        digest(common_source[name], name)
    source_identities = common_source["job_local_source_file_identities"]
    expected_source_paths = {
        "SOURCE-MANIFEST.json": value["source"]["archive_tree_binding"][
            "manifest_sha256"
        ],
        **value["source"]["required_file_sha256"],
    }
    if (
        not isinstance(source_identities, list)
        or len(source_identities) != len(expected_source_paths)
        or sha256(canonical_json(source_identities).encode("utf-8")).hexdigest()
        != common_source["job_local_source_live_identity_sha256"]
    ):
        raise ValueError("job-local source live identity receipt drifted")
    source_identity_fields = {
        "relative_path",
        "path",
        "sha256",
        "byte_count",
        "mode_octal",
        "device",
        "inode",
        "uid",
        "gid",
        "link_count",
        "mtime_ns",
        "ctime_ns",
        "regular_file",
        "read_only",
    }
    seen_source_paths: set[str] = set()
    for identity in source_identities:
        if not isinstance(identity, Mapping) or set(identity) != source_identity_fields:
            raise ValueError("job-local source file identity receipt drifted")
        relative = identity["relative_path"]
        relative_path = Path(relative) if isinstance(relative, str) else Path("..")
        if (
            not isinstance(relative, str)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen_source_paths
            or identity["sha256"] != expected_source_paths.get(relative)
            or identity["path"]
            != str(Path(common_source["job_local_source_root"]) / relative_path)
            or identity.get("regular_file") is not True
            or identity.get("read_only") is not True
            or identity.get("link_count") != 1
            or isinstance(identity.get("gid"), bool)
            or not isinstance(identity.get("gid"), int)
            or identity["gid"] < 0
        ):
            raise ValueError("job-local source file identity receipt drifted")
        seen_source_paths.add(relative)
    if seen_source_paths != set(expected_source_paths):
        raise ValueError("job-local source file identity set drifted")

    if worker_kind == "collector":
        if payload["input_binding"] is not None or payload["trainer_result"] is not None:
            raise ValueError("collector staging proof cannot claim trainer input use")
    else:
        validate_training_input_receipt(
            payload["input_binding"],
            payload["trainer_result"],
            value,
            root=root,
            allowed_root=allowed_root,
            require_live=require_live_staging,
            common_plan=common_plan,
            common_source=common_source,
            common_staging=payload,
            job_id=job_id,
            array_index=expected_array_index,
        )
    if require_live_staging:
        live_environment = os.environ if environment is None else environment
        common = {name: payload[name] for name in _COMMON_FIELDS}
        rebuilt = rebuild_job_staging_proof(
            value,
            common,
            environment=live_environment,
            allowed_root=allowed,
            wrapper_mint_consumed=worker_kind == "training_seed",
        )
        if rebuilt != common:
            raise RuntimeError("live job-local source/plan staging identity drifted")
    return {**payload, "proof_sha256": supplied}


__all__ = ["finalize_staging_proof", "validate_staging_proof"]
