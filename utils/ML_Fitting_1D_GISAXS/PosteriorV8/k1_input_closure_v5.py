"""Exact original-to-job-local input closure for K1 training."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .k1_staging_files_v5 import (
    checked_json,
    checked_zip_manifest,
    existing_under_root,
    read_only_identity,
    regular_identity,
)
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_training_chain_fingerprints,
    validate_v5_k1_training_chain_plan,
)


def expected_copy_contract(
    plan: Mapping[str, object], inventory: Mapping[str, object]
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    root = Path(plan["_live_staging_root"])
    inputs_root = root / "inputs"
    expected = [
        {
            "original_path": str(Path(plan["input_inventory"]["path"]).resolve(strict=True)),
            "job_local_path": str(inputs_root / "k1-input-inventory.json"),
            "frozen_sha256": plan["input_inventory"]["file_sha256"],
            "kind": "input_inventory",
            "role": None,
            "trainer_role": None,
            "manifest_sha256": None,
        }
    ]
    local_inputs = {
        name: []
        for name in (
            "train_datasets",
            "validation_datasets",
            "train_sidecars",
            "validation_sidecars",
        )
    }
    for role, prefix in (("train", "train"), ("tuning_validation", "validation")):
        for index, artifact in enumerate(inventory["artifacts"][role]):
            artifact_root = inputs_root / role / f"artifact-{index:04d}"
            original = Path(artifact["path"]).resolve(strict=True)
            local = artifact_root / f"grouped-parent{original.suffix}"
            expected.append(
                {
                    "original_path": str(original),
                    "job_local_path": str(local),
                    "frozen_sha256": artifact["artifact_sha256"],
                    "kind": "grouped_parent",
                    "role": role,
                    "trainer_role": prefix,
                    "manifest_sha256": artifact["manifest_sha256"],
                }
            )
            local_inputs[f"{prefix}_datasets"].append(str(local))
            if artifact["sidecar_path"] is None:
                continue
            original_sidecar = Path(artifact["sidecar_path"]).resolve(strict=True)
            local_sidecar = artifact_root / f"search-sidecar{original_sidecar.suffix}"
            expected.append(
                {
                    "original_path": str(original_sidecar),
                    "job_local_path": str(local_sidecar),
                    "frozen_sha256": artifact["sidecar_artifact_sha256"],
                    "kind": "frozen_search_sidecar",
                    "role": role,
                    "trainer_role": prefix,
                    "manifest_sha256": artifact["sidecar_manifest_sha256"],
                }
            )
            local_inputs[f"{prefix}_sidecars"].append(str(local_sidecar))
            original_receipt = Path(artifact["evidence_receipt_path"]).resolve(
                strict=True
            )
            local_receipt = local_sidecar.with_suffix(".evidence-receipt.json")
            expected.append(
                {
                    "original_path": str(original_receipt),
                    "job_local_path": str(local_receipt),
                    "frozen_sha256": artifact["evidence_receipt_sha256"],
                    "kind": "task_bound_search_evidence_receipt",
                    "role": role,
                    "trainer_role": prefix,
                    "manifest_sha256": None,
                }
            )
            destinations: dict[Path, tuple[Path, str]] = {}
            entries = checked_json(original_receipt).get("branch_evidence")
            if not isinstance(entries, list) or not entries:
                raise ValueError("search evidence receipt has no branch evidence closure")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("search evidence receipt branch entry is invalid")
                relative = Path(str(entry.get("relative_path", "")))
                if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                    raise ValueError(
                        "search evidence receipt contains an unsafe relative path"
                    )
                original_evidence = (original_receipt.parent / relative).resolve(
                    strict=True
                )
                local_evidence = local_receipt.parent / relative
                evidence_sha = digest(
                    entry.get("artifact_sha256"), "executor evidence SHA-256"
                )
                prior = destinations.get(local_evidence)
                if prior is not None:
                    if prior != (original_evidence, evidence_sha):
                        raise ValueError(
                            "search evidence closure aliases different immutable files"
                        )
                    continue
                destinations[local_evidence] = (original_evidence, evidence_sha)
                expected.append(
                    {
                        "original_path": str(original_evidence),
                        "job_local_path": str(local_evidence),
                        "frozen_sha256": evidence_sha,
                        "kind": "executor_evidence",
                        "role": role,
                        "trainer_role": prefix,
                        "manifest_sha256": None,
                    }
                )
    return expected, local_inputs


def staged_binding_rows(
    staging: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    inventory: Mapping[str, object],
    allowed_root: Path,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    if not isinstance(staging, Mapping):
        raise TypeError("staged training inputs must be a mapping")
    artifacts = staging.get("artifacts")
    local_inputs = staging.get("local_inputs")
    if not isinstance(artifacts, list) or not isinstance(local_inputs, Mapping):
        raise ValueError("staged training input closure is incomplete")
    root = Path(plan["_live_staging_root"]).resolve(strict=True)
    expected, expected_local_inputs = expected_copy_contract(plan, inventory)
    if dict(local_inputs) != expected_local_inputs:
        raise RuntimeError("trainer arguments drifted from the frozen K1 inventory")
    supplied = [staging.get("inventory"), *artifacts]
    if len(supplied) != len(expected) or any(
        not isinstance(value, Mapping) for value in supplied
    ):
        raise RuntimeError("staged training copy inventory is incomplete")
    rows = []
    seen_originals: set[Path] = set()
    seen_locals: set[Path] = set()
    for index, (artifact, frozen) in enumerate(zip(supplied, expected)):
        original = existing_under_root(
            Path(str(artifact.get("original_path", ""))),
            allowed_root,
            f"original staged input {index}",
        )
        local = existing_under_root(
            Path(str(artifact.get("job_local_path", ""))),
            root,
            f"job-local staged input {index}",
        )
        if original in seen_originals or local in seen_locals:
            raise ValueError("staged input copy mapping contains duplicate paths")
        seen_originals.add(original)
        seen_locals.add(local)
        if (
            str(original) != frozen["original_path"]
            or str(local) != frozen["job_local_path"]
            or artifact.get("frozen_sha256") != frozen["frozen_sha256"]
            or artifact.get("kind") != frozen["kind"]
            or artifact.get("role") != frozen["role"]
            or artifact.get("trainer_role") != frozen["trainer_role"]
        ):
            raise RuntimeError("staged input mapping drifted from its frozen contract")
        original_identity = regular_identity(original, f"original staged input {index}")
        local_identity = read_only_identity(local, f"job-local staged input {index}")
        if (
            original_identity["sha256"] != frozen["frozen_sha256"]
            or local_identity["sha256"] != frozen["frozen_sha256"]
            or original_identity["byte_count"] != local_identity["byte_count"]
            or artifact.get("byte_count") != local_identity["byte_count"]
        ):
            raise RuntimeError(
                "original and job-local training input byte identity disagrees"
            )
        manifest_sha = frozen["manifest_sha256"]
        if manifest_sha is not None and checked_zip_manifest(local)[
            "manifest_sha256"
        ] != manifest_sha:
            raise RuntimeError("job-local dataset/sidecar manifest identity drifted")
        rows.append(
            {
                "kind": frozen["kind"],
                "role": frozen["role"],
                "trainer_role": frozen["trainer_role"],
                "original_path": str(original),
                "job_local_path": str(local),
                "sha256": frozen["frozen_sha256"],
                "byte_count": local_identity["byte_count"],
                "manifest_sha256": manifest_sha,
                "original_device": original_identity["device"],
                "original_inode": original_identity["inode"],
                "original_mtime_ns": original_identity["mtime_ns"],
                "original_ctime_ns": original_identity["ctime_ns"],
                "job_local_device": local_identity["device"],
                "job_local_inode": local_identity["inode"],
                "job_local_mtime_ns": local_identity["mtime_ns"],
                "job_local_ctime_ns": local_identity["ctime_ns"],
                "job_local_link_count": local_identity["link_count"],
                "job_local_mode_octal": local_identity["mode_octal"],
            }
        )
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
    trainer_rows = [
        row
        for row in rows
        if row["kind"] in {"grouped_parent", "frozen_search_sidecar"}
    ]
    if {row["job_local_path"] for row in trainer_rows} != set(trainer_paths):
        raise RuntimeError("trainer arguments do not equal the verified local copies")
    return rows, trainer_paths


def rehash_staged_artifacts(
    staging: Mapping[str, object],
    *,
    plan: Mapping[str, object] | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> str:
    """Rehash the original/local closure; local-only claims never suffice."""

    if plan is None:
        raise TypeError("rehashing staged artifacts requires the authenticated K1 plan")
    value = validate_v5_k1_training_chain_plan(plan)
    replay = replay_v5_k1_training_chain_fingerprints(value, allowed_root=allowed_root)
    live = {
        **value,
        "_live_staging_root": str(
            Path(staging["inventory"]["job_local_path"]).parents[1]
        ),
    }
    rows, _ = staged_binding_rows(
        staging,
        plan=live,
        inventory=replay["inventory"],
        allowed_root=allowed_root,
    )
    return sha256(canonical_json(rows).encode("utf-8")).hexdigest()


__all__ = [
    "expected_copy_contract",
    "rehash_staged_artifacts",
    "staged_binding_rows",
]
