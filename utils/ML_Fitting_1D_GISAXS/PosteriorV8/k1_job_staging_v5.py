"""Job-private immutable staging for the Maxwell K1 training chain."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from typing import Mapping

from .grouped_dataset_v5 import V5_GROUPED_DATASET_SCHEMA, V5_GROUPED_DATASET_VERSION
from .k1_training_chain_contract_v5 import (
    canonical_json,
    digest,
    validate_v5_k1_training_inventory,
)
from .k1_training_chain_dataset_audit_v5 import audit_v5_k1_training_datasets
from .k1_training_chain_plan_v5 import (
    MAXWELL_DUST_ROOT,
    fingerprint_v5_k1_training_source,
    replay_v5_k1_training_chain_fingerprints,
    validate_v5_k1_training_chain_plan,
)
from .k1_staging_files_v5 import (
    checked_json,
    checked_zip_manifest,
    copy_regular_exclusive,
    file_sha256,
    freeze_staging_tree,
    lexical_no_symlinks,
    read_only_identity,
    regular_identity,
)
from .package_source_snapshot_v5 import MANIFEST_NAME, verify_extracted_source_snapshot


V5_K1_JOB_STAGING_SCHEMA = "gisaxs.posterior_v8.k1_job_private_staging/v3"
V5_K1_JOB_STAGING_VERSION = (
    "posterior_v8_v5_2_dust_private_live_capability_pre_post_source_rehash_v3"
)


def build_job_staging_proof(
    plan: Mapping[str, object],
    *,
    plan_path: Path,
    staging_root: Path,
    source_root: Path,
    source_archive: Path,
    worker_kind: str,
    array_index: int | None,
    environment: Mapping[str, str],
    allowed_root: Path = MAXWELL_DUST_ROOT,
    consumed_wrapper_mint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if worker_kind not in {"training_seed", "collector"}:
        raise ValueError("worker_kind is unsupported")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or int(job_id) < 1:
        raise RuntimeError("job-private staging requires a Slurm job ID")
    if worker_kind == "training_seed":
        task_id = environment.get("SLURM_ARRAY_TASK_ID", "")
        if not task_id.isdigit() or int(task_id) != array_index:
            raise RuntimeError(
                "job-private training staging requires its exact Slurm array task ID"
            )
    expected_name = (
        f"gisaxs-v5-k1-seed-{job_id}-{array_index}-"
        if worker_kind == "training_seed"
        else f"gisaxs-v5-k1-collect-{job_id}-"
    )
    root = lexical_no_symlinks(staging_root, "job staging root").resolve(strict=True)
    job_tmp_value = environment.get("POSTERIOR_V8_JOB_TMP_ROOT")
    if (
        not isinstance(job_tmp_value, str)
        or not job_tmp_value
        or not Path(job_tmp_value).is_absolute()
    ):
        raise RuntimeError("wrapper did not export its job-private temporary root")
    # Host-local temporary defaults must never redirect dataset copies or caches.
    scratch_variable = "POSTERIOR_V8_SCRATCH_BASE"
    scratch_value = environment.get(scratch_variable)
    if not isinstance(scratch_value, str) or not Path(scratch_value).is_absolute():
        raise RuntimeError("wrapper scratch base must be an absolute path")
    scratch = lexical_no_symlinks(Path(scratch_value), "wrapper scratch base").resolve(
        strict=True
    )
    scratch_status = scratch.stat()
    if not scratch.is_dir() or scratch.is_symlink():
        raise ValueError("wrapper scratch base must be a real directory")
    job_tmp = lexical_no_symlinks(
        Path(job_tmp_value), "wrapper job-private temporary root"
    ).resolve(strict=True)
    job_tmp_status = job_tmp.stat()
    job_tmp_mode = stat.S_IMODE(job_tmp_status.st_mode)
    if (
        not job_tmp.is_dir()
        or job_tmp.is_symlink()
        or job_tmp.parent != scratch
        or re.fullmatch(re.escape(expected_name) + r"[A-Za-z0-9]{8}", job_tmp.name)
        is None
        or job_tmp_status.st_uid != os.getuid()
        or job_tmp_mode != 0o700
    ):
        raise ValueError(
            "wrapper job-private temporary root escaped its Slurm job binding"
        )
    expected_entries = {"staging", "runtime-cache"}
    mint_path = job_tmp / ".posterior-v8-wrapper-mint"
    if consumed_wrapper_mint is None:
        expected_entries.add(mint_path.name)
    if {path.name for path in job_tmp.iterdir()} != expected_entries:
        raise ValueError("wrapper job-private temporary root has unexpected prior content")
    if consumed_wrapper_mint is None:
        mint_identity = read_only_identity(mint_path, "wrapper one-shot mint")
        if (
            mint_identity["uid"] != os.getuid()
            or mint_identity["mode_octal"] != "0400"
            or mint_identity["byte_count"] != 32
            or mint_identity["device"] != job_tmp_status.st_dev
        ):
            raise ValueError("wrapper one-shot mint is not a private fresh token")
        wrapper_mint = {
            name: mint_identity[name]
            for name in (
                "path",
                "byte_count",
                "mode_octal",
                "device",
                "inode",
                "uid",
                "link_count",
                "mtime_ns",
                "ctime_ns",
            )
        }
    else:
        expected_mint_fields = {
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
        if (
            not isinstance(consumed_wrapper_mint, Mapping)
            or set(consumed_wrapper_mint) != expected_mint_fields
            or consumed_wrapper_mint["path"] != str(mint_path)
            or consumed_wrapper_mint["byte_count"] != 32
            or consumed_wrapper_mint["mode_octal"] != "0400"
            or consumed_wrapper_mint["device"] != job_tmp_status.st_dev
            or consumed_wrapper_mint["uid"] != os.getuid()
            or consumed_wrapper_mint["link_count"] != 1
            or isinstance(consumed_wrapper_mint["mtime_ns"], bool)
            or not isinstance(consumed_wrapper_mint["mtime_ns"], int)
            or isinstance(consumed_wrapper_mint["ctime_ns"], bool)
            or not isinstance(consumed_wrapper_mint["ctime_ns"], int)
            or mint_path.exists()
            or mint_path.is_symlink()
        ):
            raise ValueError("consumed wrapper one-shot mint identity drifted")
        wrapper_mint = dict(consumed_wrapper_mint)
    runtime_cache = lexical_no_symlinks(
        job_tmp / "runtime-cache", "wrapper runtime cache root"
    ).resolve(strict=True)
    runtime_cache_status = runtime_cache.stat()
    runtime_cache_mode = stat.S_IMODE(runtime_cache_status.st_mode)
    if (
        not runtime_cache.is_dir()
        or runtime_cache.is_symlink()
        or runtime_cache_status.st_uid != os.getuid()
        or runtime_cache_mode != 0o700
    ):
        raise ValueError("wrapper runtime cache root is not owner-private")
    run_root = Path(plan["layout"]["run_root"]).resolve(strict=True)
    allowed = allowed_root.resolve(strict=True)
    if (
        not root.is_dir()
        or root.is_symlink()
        or root.stat().st_uid != os.getuid()
        or root != job_tmp / "staging"
        or not run_root.is_relative_to(allowed)
        or not scratch.is_relative_to(allowed)
    ):
        raise ValueError("job staging root escaped its Slurm scratch/job identity")
    root_mode = stat.S_IMODE(root.stat().st_mode)
    if root_mode & (stat.S_IRWXG | stat.S_IRWXO) or not root_mode & stat.S_IRWXU:
        raise ValueError("job staging root must be owner-private")
    local_plan = read_only_identity(plan_path, "job-local K1 plan")
    original_plan = Path(plan["layout"]["plan"])
    original_plan_identity = regular_identity(original_plan, "original frozen K1 plan")
    if original_plan_identity["sha256"] != local_plan["sha256"]:
        raise RuntimeError("job-local K1 plan is not a byte-for-byte copy of the launch plan")
    if Path(local_plan["path"]) != root / "k1-training-plan.json":
        raise ValueError("job-local K1 plan escaped the private staging contract")
    local_archive = read_only_identity(source_archive, "job-local source archive")
    original_archive_identity = regular_identity(
        Path(plan["source"]["archive_path"]), "original frozen source archive"
    )
    if (
        Path(local_archive["path"]) != root / "source-snapshot.tar"
        or local_archive["sha256"] != plan["source"]["archive_sha256"]
        or original_archive_identity["sha256"] != local_archive["sha256"]
    ):
        raise RuntimeError("job-local source archive differs from the frozen source identity")
    local_source = lexical_no_symlinks(source_root, "job-local source root").resolve(strict=True)
    if local_source != root / "source" or not local_source.is_dir() or local_source.is_symlink():
        raise ValueError("job-local source tree escaped the private staging root")
    expected_binding = plan["source"]["archive_tree_binding"]
    local_binding = verify_extracted_source_snapshot(
        Path(local_archive["path"]),
        local_source,
        expected_archive_sha256=plan["source"]["archive_sha256"],
        expected_manifest_sha256=expected_binding["manifest_sha256"],
        expected_source_tree_sha256=expected_binding["source_tree_sha256"],
    )
    path_fields = {"archive_path", "source_root", "schema_version", "version"}
    for field, expected in expected_binding.items():
        if field not in path_fields and local_binding[field] != expected:
            raise RuntimeError(f"job-local source binding differs at {field}")
    fingerprint = fingerprint_v5_k1_training_source(local_source)
    for field in (
        "bundle_sha256",
        "required_file_sha256",
        "source_snapshot_write_bits_set",
        "source_files_with_write_bits",
    ):
        if fingerprint[field] != plan["source"][field]:
            raise RuntimeError("job-local source bundle differs from the frozen source bundle")
    source_file_identities = []
    for relative in (MANIFEST_NAME, *plan["source"]["required_file_sha256"]):
        identity = read_only_identity(
            local_source / relative,
            f"job-local source file {relative}",
        )
        expected_sha256 = (
            expected_binding["manifest_sha256"]
            if relative == MANIFEST_NAME
            else plan["source"]["required_file_sha256"][relative]
        )
        if identity["sha256"] != expected_sha256:
            raise RuntimeError(f"job-local source file identity drifted: {relative}")
        source_file_identities.append(
            {"relative_path": relative, **identity}
        )
    source_live_identity_sha256 = sha256(
        canonical_json(source_file_identities).encode("utf-8")
    ).hexdigest()
    if not Path(__file__).resolve(strict=True).is_relative_to(local_source):
        raise RuntimeError("K1 staging runtime was not imported from the job-local source tree")
    return {
        "schema": V5_K1_JOB_STAGING_SCHEMA,
        "version": V5_K1_JOB_STAGING_VERSION,
        "worker_kind": worker_kind,
        "slurm_job_id": job_id,
        "slurm_array_task_id": array_index,
        "staging_root": str(root),
        "scratch_base": str(scratch),
        "scratch_base_source": scratch_variable,
        "scratch_base_device": scratch_status.st_dev,
        "scratch_base_inode": scratch_status.st_ino,
        "job_tmp_root": str(job_tmp),
        "job_tmp_root_device": job_tmp_status.st_dev,
        "job_tmp_root_inode": job_tmp_status.st_ino,
        "job_tmp_root_uid": job_tmp_status.st_uid,
        "job_tmp_root_mode_octal": f"{job_tmp_mode:04o}",
        "job_tmp_root_owner_private": True,
        "wrapper_mint": wrapper_mint,
        "runtime_cache_root": str(runtime_cache),
        "staging_root_device": root.stat().st_dev,
        "staging_root_inode": root.stat().st_ino,
        "staging_root_owner_private": True,
        "plan": {
            "scientific_plan_sha256": plan["plan_sha256"],
            "original_path": str(original_plan),
            "job_local_path": local_plan["path"],
            "original_file_sha256": original_plan_identity["sha256"],
            "job_local_file_sha256": local_plan["sha256"],
            "original_byte_count": original_plan_identity["byte_count"],
            "original_device": original_plan_identity["device"],
            "original_inode": original_plan_identity["inode"],
            "original_mtime_ns": original_plan_identity["mtime_ns"],
            "original_ctime_ns": original_plan_identity["ctime_ns"],
            "job_local_byte_count": local_plan["byte_count"],
            "job_local_device": local_plan["device"],
            "job_local_inode": local_plan["inode"],
            "job_local_uid": local_plan["uid"],
            "job_local_link_count": local_plan["link_count"],
            "job_local_mtime_ns": local_plan["mtime_ns"],
            "job_local_ctime_ns": local_plan["ctime_ns"],
            "byte_for_byte_copy_verified": True,
            "mode_octal": local_plan["mode_octal"],
        },
        "source": {
            "original_archive_path": plan["source"]["archive_path"],
            "job_local_archive_path": local_archive["path"],
            "archive_sha256": local_archive["sha256"],
            "archive_mode_octal": local_archive["mode_octal"],
            "original_archive_byte_count": original_archive_identity["byte_count"],
            "original_archive_device": original_archive_identity["device"],
            "original_archive_inode": original_archive_identity["inode"],
            "original_archive_mtime_ns": original_archive_identity["mtime_ns"],
            "original_archive_ctime_ns": original_archive_identity["ctime_ns"],
            "job_local_archive_byte_count": local_archive["byte_count"],
            "job_local_archive_device": local_archive["device"],
            "job_local_archive_inode": local_archive["inode"],
            "job_local_archive_uid": local_archive["uid"],
            "job_local_archive_link_count": local_archive["link_count"],
            "job_local_archive_mtime_ns": local_archive["mtime_ns"],
            "job_local_archive_ctime_ns": local_archive["ctime_ns"],
            "byte_for_byte_archive_copy_verified": True,
            "original_source_root": plan["source"]["root"],
            "job_local_source_root": str(local_source),
            "source_bundle_sha256": fingerprint["bundle_sha256"],
            "manifest_sha256": local_binding["manifest_sha256"],
            "source_tree_sha256": local_binding["source_tree_sha256"],
            "job_local_source_file_identities": source_file_identities,
            "job_local_source_live_identity_sha256": source_live_identity_sha256,
            "exact_tree_verified": True,
            "runtime_imported_from_job_local_tree": True,
        },
    }


def rebuild_job_staging_proof(
    plan: Mapping[str, object],
    common: Mapping[str, object],
    *,
    environment: Mapping[str, str],
    allowed_root: Path,
    wrapper_mint_consumed: bool = False,
) -> dict[str, object]:
    """Replay the common plan/source/job-root proof from its bound live paths."""

    return build_job_staging_proof(
        plan,
        plan_path=Path(common["plan"]["job_local_path"]),
        staging_root=Path(common["staging_root"]),
        source_root=Path(common["source"]["job_local_source_root"]),
        source_archive=Path(common["source"]["job_local_archive_path"]),
        worker_kind=common["worker_kind"],
        array_index=common["slurm_array_task_id"],
        environment=environment,
        allowed_root=allowed_root,
        consumed_wrapper_mint=(
            common["wrapper_mint"] if wrapper_mint_consumed else None
        ),
    )


def verify_training_artifact_bytes(
    plan: Mapping[str, object], *, allowed_root: Path = MAXWELL_DUST_ROOT
) -> dict[str, object]:
    value = validate_v5_k1_training_chain_plan(plan)
    inventory = replay_v5_k1_training_chain_fingerprints(
        value, allowed_root=allowed_root
    )["inventory"]
    verified = []
    for role in ("train", "tuning_validation"):
        for artifact in inventory["artifacts"][role]:
            dataset_path = Path(artifact["path"]).resolve(strict=True)
            if file_sha256(dataset_path) != artifact["artifact_sha256"]:
                raise RuntimeError("grouped dataset bytes changed after inventory freeze")
            manifest = checked_zip_manifest(dataset_path)
            if (
                manifest["manifest_sha256"] != artifact["manifest_sha256"]
                or (manifest.get("dataset_schema"), manifest.get("dataset_version"))
                != (V5_GROUPED_DATASET_SCHEMA, V5_GROUPED_DATASET_VERSION)
            ):
                raise RuntimeError("grouped dataset manifest identity drifted")
            row = {
                "role": role,
                "path": str(dataset_path),
                "artifact_sha256": artifact["artifact_sha256"],
                "manifest_sha256": artifact["manifest_sha256"],
            }
            if artifact["sidecar_path"] is not None:
                sidecar = Path(artifact["sidecar_path"]).resolve(strict=True)
                evidence = Path(artifact["evidence_receipt_path"]).resolve(strict=True)
                if file_sha256(sidecar) != artifact["sidecar_artifact_sha256"]:
                    raise RuntimeError("search sidecar bytes changed after inventory freeze")
                if checked_zip_manifest(sidecar)["manifest_sha256"] != artifact[
                    "sidecar_manifest_sha256"
                ]:
                    raise RuntimeError("search sidecar manifest identity drifted")
                if file_sha256(evidence) != artifact["evidence_receipt_sha256"]:
                    raise RuntimeError("search evidence receipt changed after inventory freeze")
                row.update(
                    sidecar_artifact_sha256=artifact["sidecar_artifact_sha256"],
                    evidence_receipt_sha256=artifact["evidence_receipt_sha256"],
                )
            verified.append(row)
    return {
        "inventory_sha256": inventory["inventory_sha256"],
        "dataset_manifest_bundle_sha256": inventory["dataset_contract"][
            "dataset_manifest_bundle_sha256"
        ],
        "verified_artifacts": verified,
        "dataset_audit": audit_v5_k1_training_datasets(inventory),
    }


def _copy_evidence_closure(
    original_receipt: Path, local_receipt: Path, expected_sha256: str
) -> list[dict[str, object]]:
    receipt_copy = copy_regular_exclusive(
        original_receipt,
        local_receipt,
        expected_sha256=expected_sha256,
        name="search evidence receipt",
    )
    receipt_copy.update(kind="task_bound_search_evidence_receipt", trainer_consumed=True)
    entries = checked_json(local_receipt).get("branch_evidence")
    if not isinstance(entries, list) or not entries:
        raise ValueError("search evidence receipt has no branch evidence closure")
    copied = [receipt_copy]
    destinations: dict[Path, tuple[Path, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError("search evidence receipt branch entry is invalid")
        relative = Path(str(entry.get("relative_path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("search evidence receipt contains an unsafe relative path")
        expected = digest(entry.get("artifact_sha256"), "executor evidence SHA-256")
        source, destination = original_receipt.parent / relative, local_receipt.parent / relative
        prior = destinations.get(destination)
        if prior is not None:
            if prior != (source, expected):
                raise ValueError("search evidence closure aliases different immutable files")
            continue
        evidence = copy_regular_exclusive(
            source, destination, expected_sha256=expected, name=f"executor evidence {index}"
        )
        evidence.update(kind="executor_evidence", trainer_consumed=False)
        copied.append(evidence)
        destinations[destination] = (source, expected)
    return copied


def stage_training_inputs(
    plan: Mapping[str, object],
    *,
    staging_root: Path,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    value = validate_v5_k1_training_chain_plan(plan)
    inputs_root = lexical_no_symlinks(staging_root, "job staging root").resolve(
        strict=True
    ) / "inputs"
    if inputs_root.exists() or inputs_root.is_symlink():
        raise FileExistsError("refusing to reuse a job-private training input tree")
    inputs_root.mkdir(mode=0o700)
    replay = replay_v5_k1_training_chain_fingerprints(value, allowed_root=allowed_root)
    local_inventory = inputs_root / "k1-input-inventory.json"
    inventory_copy = copy_regular_exclusive(
        Path(value["input_inventory"]["path"]),
        local_inventory,
        expected_sha256=value["input_inventory"]["file_sha256"],
        name="K1 input inventory",
    )
    inventory_copy.update(kind="input_inventory", trainer_consumed=False)
    inventory = validate_v5_k1_training_inventory(checked_json(local_inventory))
    if inventory != replay["inventory"]:
        raise RuntimeError("job-local inventory differs from the replayed inventory")
    local_audit_inventory = deepcopy(inventory)
    copies: list[dict[str, object]] = []
    local_inputs = {name: [] for name in (
        "train_datasets", "validation_datasets", "train_sidecars", "validation_sidecars"
    )}
    for role, prefix in (("train", "train"), ("tuning_validation", "validation")):
        for index, artifact in enumerate(inventory["artifacts"][role]):
            artifact_root = inputs_root / role / f"artifact-{index:04d}"
            artifact_root.mkdir(mode=0o700, parents=True)
            original = Path(artifact["path"])
            local = artifact_root / f"grouped-parent{original.suffix}"
            parent = copy_regular_exclusive(
                original,
                local,
                expected_sha256=artifact["artifact_sha256"],
                name=f"{role} grouped dataset {index}",
            )
            if checked_zip_manifest(local)["manifest_sha256"] != artifact["manifest_sha256"]:
                raise RuntimeError("staged grouped dataset manifest identity drifted")
            parent.update(
                role=role,
                trainer_role=prefix,
                kind="grouped_parent",
                manifest_sha256=artifact["manifest_sha256"],
                trainer_consumed=True,
            )
            copies.append(parent)
            local_inputs[f"{prefix}_datasets"].append(str(local))
            local_audit_inventory["artifacts"][role][index]["path"] = str(local)
            if artifact["sidecar_path"] is None:
                continue
            original_sidecar = Path(artifact["sidecar_path"])
            local_sidecar = artifact_root / f"search-sidecar{original_sidecar.suffix}"
            sidecar = copy_regular_exclusive(
                original_sidecar,
                local_sidecar,
                expected_sha256=artifact["sidecar_artifact_sha256"],
                name=f"{role} search sidecar {index}",
            )
            if checked_zip_manifest(local_sidecar)["manifest_sha256"] != artifact[
                "sidecar_manifest_sha256"
            ]:
                raise RuntimeError("staged search sidecar manifest identity drifted")
            sidecar.update(
                role=role,
                trainer_role=prefix,
                kind="frozen_search_sidecar",
                manifest_sha256=artifact["sidecar_manifest_sha256"],
                trainer_consumed=True,
            )
            copies.append(sidecar)
            local_inputs[f"{prefix}_sidecars"].append(str(local_sidecar))
            local_audit_inventory["artifacts"][role][index]["sidecar_path"] = str(
                local_sidecar
            )
            local_receipt = local_sidecar.with_suffix(".evidence-receipt.json")
            evidence = _copy_evidence_closure(
                Path(artifact["evidence_receipt_path"]),
                local_receipt,
                artifact["evidence_receipt_sha256"],
            )
            for item in evidence:
                item.update(role=role, trainer_role=prefix)
            copies.extend(evidence)
            local_audit_inventory["artifacts"][role][index]["evidence_receipt_path"] = str(
                local_receipt
            )
    dataset_audit = audit_v5_k1_training_datasets(local_audit_inventory)
    freeze_staging_tree(inputs_root)
    return {
        "inventory": inventory_copy,
        "artifacts": copies,
        "local_inputs": local_inputs,
        "dataset_audit": dataset_audit,
    }


__all__ = [
    "V5_K1_JOB_STAGING_SCHEMA",
    "V5_K1_JOB_STAGING_VERSION",
    "build_job_staging_proof",
    "checked_json",
    "file_sha256",
    "freeze_staging_tree",
    "lexical_no_symlinks",
    "rebuild_job_staging_proof",
    "stage_training_inputs",
    "verify_training_artifact_bytes",
]
