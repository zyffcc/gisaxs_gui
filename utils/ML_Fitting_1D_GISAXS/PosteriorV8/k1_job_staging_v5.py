"""Job-private immutable staging for the Maxwell K1 training chain."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping
import zipfile

from .grouped_dataset_v5 import V5_GROUPED_DATASET_SCHEMA, V5_GROUPED_DATASET_VERSION
from .k1_training_chain_contract_v5 import (
    canonical_json,
    digest,
    validate_v5_k1_training_inventory,
)
from .k1_training_chain_dataset_audit_v5 import audit_v5_k1_training_datasets
from .k1_training_chain_plan_v5 import (
    MAXWELL_DUST_ROOT,
    V5_K1_WORKER_SECURITY_BLOCKER,
    fingerprint_v5_k1_training_source,
    replay_v5_k1_training_chain_fingerprints,
    validate_v5_k1_training_chain_plan,
)
from .package_source_snapshot_v5 import verify_extracted_source_snapshot


V5_K1_JOB_STAGING_SCHEMA = "gisaxs.posterior_v8.k1_job_private_staging/v1"
V5_K1_JOB_STAGING_VERSION = (
    "posterior_v8_v5_2_original_to_job_private_read_only_copy_rehash_v1"
)
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def lexical_no_symlinks(path: Path, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
    return lexical


def _open_regular(path: Path, name: str):
    lexical = lexical_no_symlinks(path, name)
    descriptor = os.open(lexical, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{name} must be a regular file")
        return lexical, descriptor
    except Exception:
        os.close(descriptor)
        raise


def file_sha256(path: Path, name: str = "file") -> str:
    value = sha256()
    _, descriptor = _open_regular(path, name)
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _strict_object(raw: bytes, name: str) -> dict[str, object]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def checked_json(path: Path, *, maximum_bytes: int = 32 * 1024 * 1024) -> dict[str, object]:
    _, descriptor = _open_regular(path, f"JSON artifact {path}")
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise ValueError(f"JSON artifact is unexpectedly large: {path}")
    return _strict_object(raw, f"JSON artifact {path}")


def _checked_zip_manifest(path: Path) -> dict[str, object]:
    try:
        _, descriptor = _open_regular(path, "checked artifact")
        with os.fdopen(descriptor, "rb") as stream, zipfile.ZipFile(stream, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("checked artifact contains duplicate archive members")
            member = archive.getinfo("manifest.json")
            if member.file_size > 32 * 1024 * 1024:
                raise ValueError("checked artifact manifest is unexpectedly large")
            value = _strict_object(archive.read(member), "checked artifact manifest")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"checked artifact has no readable manifest: {path}") from exc
    core = dict(value)
    supplied = digest(core.pop("manifest_sha256", None), "artifact manifest SHA-256")
    if supplied != sha256(canonical_json(core).encode()).hexdigest():
        raise ValueError("checked artifact manifest SHA-256 does not reproduce")
    return value


def _read_only_identity(path: Path, name: str) -> dict[str, object]:
    lexical, descriptor = _open_regular(path, name)
    status = os.fstat(descriptor)
    os.close(descriptor)
    mode = stat.S_IMODE(status.st_mode)
    if mode & _WRITE_BITS:
        raise ValueError(f"{name} must be read-only")
    return {
        "path": str(lexical.resolve(strict=True)),
        "sha256": file_sha256(lexical, name),
        "byte_count": status.st_size,
        "mode_octal": f"{mode:04o}",
        "regular_file": True,
        "read_only": True,
    }


def _copy_regular_exclusive(
    source: Path, destination: Path, *, expected_sha256: str, name: str
) -> dict[str, object]:
    expected = digest(expected_sha256, f"{name} expected SHA-256")
    source_lexical, source_descriptor = _open_regular(source, f"original {name}")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite staged {name}")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
    except Exception:
        os.close(source_descriptor)
        raise
    copied = sha256()
    byte_count = 0
    try:
        with os.fdopen(source_descriptor, "rb") as source_stream, os.fdopen(
            destination_descriptor, "wb"
        ) as destination_stream:
            while chunk := source_stream.read(1024 * 1024):
                destination_stream.write(chunk)
                copied.update(chunk)
                byte_count += len(chunk)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
            os.fchmod(destination_stream.fileno(), 0o400)
        local = _read_only_identity(destination, f"staged {name}")
        if copied.hexdigest() != expected or local["sha256"] != expected:
            raise RuntimeError(f"staged {name} differs from the frozen identity")
        return {
            "original_path": str(source_lexical),
            "job_local_path": local["path"],
            "frozen_sha256": expected,
            "copy_stream_sha256": copied.hexdigest(),
            "post_copy_sha256": local["sha256"],
            "byte_count": byte_count,
            "mode_octal": local["mode_octal"],
            "regular_file": True,
            "read_only": True,
        }
    except Exception:
        destination.unlink(missing_ok=True)
        raise


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
) -> dict[str, object]:
    if worker_kind not in {"training_seed", "collector"}:
        raise ValueError("worker_kind is unsupported")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or int(job_id) < 1:
        raise RuntimeError("job-private staging requires a Slurm job ID")
    expected_name = (
        f"gisaxs-v5-k1-seed-{job_id}-{array_index}-"
        if worker_kind == "training_seed"
        else f"gisaxs-v5-k1-collect-{job_id}-"
    )
    root = lexical_no_symlinks(staging_root, "job staging root").resolve(strict=True)
    scratch_value = environment.get("SLURM_TMPDIR") or environment.get("TMPDIR") or "/tmp"
    scratch = lexical_no_symlinks(Path(scratch_value), "Slurm scratch root").resolve(
        strict=True
    )
    run_root = Path(plan["layout"]["run_root"]).resolve(strict=True)
    allowed = allowed_root.resolve(strict=True)
    if (
        not root.is_dir()
        or root.is_symlink()
        or root.stat().st_uid != os.getuid()
        or root.parent != scratch
        or not root.name.startswith(expected_name)
        or not run_root.is_relative_to(allowed)
        or scratch.is_relative_to(allowed)
    ):
        raise ValueError("job staging root escaped its Slurm scratch/job identity")
    root_mode = stat.S_IMODE(root.stat().st_mode)
    if root_mode & (stat.S_IRWXG | stat.S_IRWXO) or not root_mode & stat.S_IRWXU:
        raise ValueError("job staging root must be owner-private")
    local_plan = _read_only_identity(plan_path, "job-local K1 plan")
    original_plan = Path(plan["layout"]["plan"])
    original_plan_sha = file_sha256(original_plan, "original frozen K1 plan")
    if original_plan_sha != local_plan["sha256"]:
        raise RuntimeError("job-local K1 plan is not a byte-for-byte copy of the launch plan")
    if Path(local_plan["path"]) != root / "k1-training-plan.json":
        raise ValueError("job-local K1 plan escaped the private staging contract")
    local_archive = _read_only_identity(source_archive, "job-local source archive")
    original_archive_sha = file_sha256(
        Path(plan["source"]["archive_path"]), "original frozen source archive"
    )
    if (
        Path(local_archive["path"]) != root / "source-snapshot.tar"
        or local_archive["sha256"] != plan["source"]["archive_sha256"]
        or original_archive_sha != local_archive["sha256"]
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
    if not Path(__file__).resolve(strict=True).is_relative_to(local_source):
        raise RuntimeError("K1 staging runtime was not imported from the job-local source tree")
    return {
        "schema": V5_K1_JOB_STAGING_SCHEMA,
        "version": V5_K1_JOB_STAGING_VERSION,
        "worker_kind": worker_kind,
        "slurm_job_id": job_id,
        "slurm_array_task_id": array_index,
        "staging_root": str(root),
        "scratch_root": str(scratch),
        "staging_root_owner_private": True,
        "plan": {
            "scientific_plan_sha256": plan["plan_sha256"],
            "original_path": str(original_plan),
            "job_local_path": local_plan["path"],
            "original_file_sha256": original_plan_sha,
            "job_local_file_sha256": local_plan["sha256"],
            "byte_for_byte_copy_verified": True,
            "mode_octal": local_plan["mode_octal"],
        },
        "source": {
            "original_archive_path": plan["source"]["archive_path"],
            "job_local_archive_path": local_archive["path"],
            "archive_sha256": local_archive["sha256"],
            "archive_mode_octal": local_archive["mode_octal"],
            "byte_for_byte_archive_copy_verified": True,
            "original_source_root": plan["source"]["root"],
            "job_local_source_root": str(local_source),
            "source_bundle_sha256": fingerprint["bundle_sha256"],
            "manifest_sha256": local_binding["manifest_sha256"],
            "source_tree_sha256": local_binding["source_tree_sha256"],
            "exact_tree_verified": True,
            "runtime_imported_from_job_local_tree": True,
        },
    }


def freeze_staging_tree(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            path = current_path / name
            if path.is_symlink():
                raise ValueError("job-private staging contains a symlink")
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~_WRITE_BITS)
        current_path.chmod(stat.S_IMODE(current_path.stat().st_mode) & ~_WRITE_BITS)


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
            manifest = _checked_zip_manifest(dataset_path)
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
                if _checked_zip_manifest(sidecar)["manifest_sha256"] != artifact[
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
    receipt_copy = _copy_regular_exclusive(
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
        evidence = _copy_regular_exclusive(
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
    inventory_copy = _copy_regular_exclusive(
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
            parent = _copy_regular_exclusive(
                original,
                local,
                expected_sha256=artifact["artifact_sha256"],
                name=f"{role} grouped dataset {index}",
            )
            if _checked_zip_manifest(local)["manifest_sha256"] != artifact["manifest_sha256"]:
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
            sidecar = _copy_regular_exclusive(
                original_sidecar,
                local_sidecar,
                expected_sha256=artifact["sidecar_artifact_sha256"],
                name=f"{role} search sidecar {index}",
            )
            if _checked_zip_manifest(local_sidecar)["manifest_sha256"] != artifact[
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


def rehash_staged_artifacts(staging: Mapping[str, object]) -> str:
    rows = []
    for artifact in (staging["inventory"], *staging["artifacts"]):
        identity = _read_only_identity(
            Path(artifact["job_local_path"]), "staged training input"
        )
        if identity["sha256"] != artifact["frozen_sha256"] or identity[
            "byte_count"
        ] != artifact["byte_count"]:
            raise RuntimeError("a staged training input changed before binding")
        rows.append(
            {
                "job_local_path": identity["path"],
                "sha256": identity["sha256"],
                "byte_count": identity["byte_count"],
                "mode_octal": identity["mode_octal"],
            }
        )
    return sha256(canonical_json(rows).encode()).hexdigest()


def finalize_staging_proof(
    common: Mapping[str, object],
    *,
    staged_inputs: Mapping[str, object] | None,
    trainer_manifest: Mapping[str, object] | None,
    post_training_rehash_sha256: str | None = None,
) -> dict[str, object]:
    """Fail closed until the persisted K1 staging receipt audit is complete."""

    del common, staged_inputs, trainer_manifest, post_training_rehash_sha256
    raise RuntimeError(V5_K1_WORKER_SECURITY_BLOCKER)


def validate_staging_proof(
    raw: object,
    plan: Mapping[str, object],
    *,
    worker_kind: str,
    expected_array_index: int | None,
    require_live_staging: bool = False,
) -> dict[str, object]:
    """Reject persisted staging claims while the receipt audit remains open."""

    del raw, plan, worker_kind, expected_array_index, require_live_staging
    raise RuntimeError(V5_K1_WORKER_SECURITY_BLOCKER)


__all__ = [
    "V5_K1_JOB_STAGING_SCHEMA",
    "V5_K1_JOB_STAGING_VERSION",
    "build_job_staging_proof",
    "checked_json",
    "file_sha256",
    "finalize_staging_proof",
    "freeze_staging_tree",
    "lexical_no_symlinks",
    "rehash_staged_artifacts",
    "stage_training_inputs",
    "validate_staging_proof",
    "verify_training_artifact_bytes",
]
