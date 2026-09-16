"""Derive training input descriptors from actual sealed search publications."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket

import numpy as np

from .k1_balanced_full_search_replay_v5 import replay_v5_k1_balanced_full_search_collection
from .k1_balanced_full_search_collector_v5 import _consumer_role, MAXWELL_DUST_ROOT
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from .k1_staging_files_v5 import lexical_no_symlinks, read_only_identity
from .k1_training_chain_contract_v5 import (
    V5K1TrainingArtifact, build_v5_k1_training_inventory, canonical_json,
    validate_v5_k1_training_inventory,
)
from .k1_training_identity_contract_v5 import V5K1TrainingIdentityAuthorization
from .k1_training_identity_runtime_v5 import issue_v5_k1_training_identity_authorization_from_files
from .k1_search_projection_evidence_v5 import V5K1SearchProjectionEvidence
from .k1_training_chain_dataset_audit_v5 import (
    _artifact_audit, _checked_audit_arrays, k1_parent_set_sha256,
)
from .search_evidence_receipt_v5 import read_v5_search_evidence_receipt


def _checked_row(row):
    bindings = (
        ("projected_parent_path", "projected_parent_artifact_sha256"),
        ("sidecar_path", "sidecar_artifact_sha256"),
        ("evidence_receipt_path", "evidence_receipt_file_sha256"),
    )
    before = {}
    for path_key, hash_key in bindings:
        path = lexical_no_symlinks(Path(row[path_key]), path_key).resolve(strict=True)
        if not path.is_relative_to(MAXWELL_DUST_ROOT):
            raise ValueError("training search inputs escaped the Maxwell dust root")
        identity = read_only_identity(path, path_key)
        if identity["mode_octal"] != "0400" or identity["sha256"] != row[hash_key]:
            raise ValueError("training search input differs from its sealed collection")
        before[path_key] = identity
    receipt = read_v5_search_evidence_receipt(
        row["evidence_receipt_path"], parent_dataset_path=row["projected_parent_path"],
        sidecar_path=row["sidecar_path"], require_training_eligible=True,
        expected_consumer_role=_consumer_role(row["role"]),
    )
    if (receipt.file_sha256 != row["evidence_receipt_file_sha256"]
            or receipt.manifest["receipt_sha256"] != row["evidence_receipt_sha256"]):
        raise ValueError("training search receipt file or logical identity differs")
    for key, expected in (("parent", row["projected_parent_artifact_sha256"]),
                          ("sidecar", row["sidecar_artifact_sha256"])):
        if receipt.manifest[key]["artifact_sha256"] != expected:
            raise ValueError("training search receipt artifact binding differs")
    # The claimed branch-pure counts are checked against each persisted recipe below.
    counts = tuple((b.branch_id, row["recipe_count"] if b.branch_id == row["branch_id"] else 0)
                   for b in K1_PHASE_C_BRANCHES)
    artifact = V5K1TrainingArtifact(
        path=row["projected_parent_path"], role=row["role"], split_id=row["role"],
        artifact_sha256=row["projected_parent_artifact_sha256"],
        manifest_sha256=receipt.manifest["parent"]["manifest_sha256"],
        clean_parent_count=row["recipe_count"], branch_counts=counts,
        sidecar_path=row["sidecar_path"], sidecar_artifact_sha256=row["sidecar_artifact_sha256"],
        sidecar_manifest_sha256=receipt.manifest["sidecar"]["manifest_sha256"],
        evidence_receipt_path=row["evidence_receipt_path"],
        # Training inventory binds file bytes, not the receipt's logical self hash.
        evidence_receipt_sha256=receipt.file_sha256, full_training_eligible=True,
    )
    audit = _artifact_audit({"path": artifact.path, "split_id": artifact.split_id,
                             "clean_parent_count": artifact.clean_parent_count,
                             "branch_counts": dict(artifact.branch_counts)})
    for path_key, _ in bindings:
        if read_only_identity(Path(row[path_key]), path_key) != before[path_key]:
            raise RuntimeError("training search input changed during descriptor replay")
    return artifact, tuple(audit["clean_group_ids"])


def _source_projection_mapping(selected, artifact):
    source = selected["parent"]
    paths = (Path(source["path"]), Path(artifact.path))
    identities = []
    arrays = []
    for path, expected_file, expected_manifest in (
        (paths[0], source["artifact_sha256"], source["manifest_sha256"]),
        (paths[1], artifact.artifact_sha256, artifact.manifest_sha256),
    ):
        path = lexical_no_symlinks(path, "projection population").resolve(strict=True)
        if not path.is_relative_to(MAXWELL_DUST_ROOT):
            raise ValueError("projection population escaped the Maxwell dust root")
        identity = read_only_identity(path, "projection population")
        if identity["mode_octal"] != "0400" or identity["sha256"] != expected_file:
            raise ValueError("projection population file binding differs")
        manifest, values = _checked_audit_arrays(path)
        if manifest["manifest_sha256"] != expected_manifest:
            raise ValueError("projection population manifest binding differs")
        identities.append(identity)
        arrays.append(values)
    if arrays[0].keys() != arrays[1].keys() or any(
        arrays[0][name].dtype != arrays[1][name].dtype
        or not np.array_equal(arrays[0][name], arrays[1][name])
        for name in arrays[0]
    ):
        raise ValueError("projected recipe, branch, split or clean-group identity differs from source")
    for path, before in zip(paths, identities, strict=True):
        if read_only_identity(path, "projection population") != before:
            raise RuntimeError("projection population changed during comparison")
    return {
        "array_task_id": selected["array_task_id"], "role": artifact.role,
        "source_path": str(paths[0]), "source_artifact_sha256": source["artifact_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "projected_path": artifact.path, "projected_artifact_sha256": artifact.artifact_sha256,
        "projected_manifest_sha256": artifact.manifest_sha256,
        "clean_parent_count": artifact.clean_parent_count,
        "ordered_recipe_branch_split_group_arrays_equal": True,
    }


def _validate_original_identity(authorization, checked, mappings, parent_hashes):
    if not isinstance(authorization, V5K1TrainingIdentityAuthorization):
        raise TypeError("original identity authorization must be a checked typed contract")
    if checked["plan"]["identity_authorization_sha256"] != authorization.sha256:
        raise ValueError("original identity authorization differs from the search plan")
    for role in ("train", "tuning_validation"):
        population = authorization.population(role)
        rows = tuple(row for row in mappings if row["role"] == role)
        if (
            tuple(sorted(row["source_artifact_sha256"] for row in rows)) != population.artifact_sha256s
            or tuple(sorted(row["source_manifest_sha256"] for row in rows)) != population.manifest_sha256s
            or sum(row["clean_parent_count"] for row in rows) != population.clean_parent_count
            or parent_hashes[role] != population.clean_group_set_sha256
        ):
            raise ValueError(f"projected {role} population does not reproduce original exclusion identity")


def _replay_original_publications(plan, paths):
    fields = {"balanced_dataset_completion", "train_tuning_receipt",
              "phase_c_completion", "three_way_receipt"}
    if not isinstance(paths, dict) or set(paths) != fields:
        raise ValueError("original identity publication paths must contain the exact four inputs")
    expected = V5K1TrainingIdentityAuthorization.from_payload(plan["identity_authorization"])
    if expected.sha256 != plan["identity_authorization_sha256"]:
        raise ValueError("original identity authorization differs from the search plan")
    before, arguments = {}, {}
    for name in sorted(fields):
        path = lexical_no_symlinks(Path(paths[name]), name).resolve(strict=True)
        if not path.is_relative_to(MAXWELL_DUST_ROOT):
            raise ValueError("original identity publication escaped the Maxwell dust root")
        identity = read_only_identity(path, name)
        if identity["mode_octal"] != "0400" or identity["sha256"] != getattr(expected, name + "_file_sha256"):
            raise ValueError("original identity publication differs from the frozen file binding")
        before[name] = identity
        arguments[name + "_path"] = path
    actual = issue_v5_k1_training_identity_authorization_from_files(
        source_archive_sha256=expected.source_archive_sha256,
        source_bundle_sha256=expected.source_bundle_sha256, **arguments,
    )
    if actual.to_payload() != expected.to_payload():
        raise ValueError("replayed original identity authorization differs from the search plan")
    for name, identity in before.items():
        if read_only_identity(arguments[name + "_path"], name) != identity:
            raise RuntimeError("original identity publication changed during replay")
    return actual, before


def read_v5_k1_search_training_inputs(plan_path, *, identity_authorization=None,
                                    identity_publication_paths=None, **collection_binding):
    """Worker-only input preparation; never substitutes for identity authorization.

    The caller supplies the full frozen collection binding accepted by the existing
    collection replay. No caller-provided task rows or eligibility flags are accepted.
    """
    checked = replay_v5_k1_balanced_full_search_collection(plan_path, **collection_binding)
    original_files = None
    if identity_publication_paths is not None:
        actual, original_files = _replay_original_publications(checked["plan"], identity_publication_paths)
        if identity_authorization is not None and identity_authorization.to_payload() != actual.to_payload():
            raise ValueError("caller identity authorization differs from actual publications")
        identity_authorization = actual
    artifacts = {"train": [], "tuning_validation": []}
    groups = {"train": [], "tuning_validation": []}
    selected_by_task = {row["array_task_id"]: row for row in checked["plan"]["parents"]}
    mappings = []
    for row in checked["inventory"]["tasks"]:
        artifact, ids = _checked_row(row)
        mappings.append(_source_projection_mapping(selected_by_task[row["array_task_id"]], artifact))
        artifacts[artifact.role].append(artifact)
        groups[artifact.role].extend(ids)
    parent_hashes = {role: k1_parent_set_sha256(ids) for role, ids in groups.items()}
    if set(groups["train"]) & set(groups["tuning_validation"]):
        raise ValueError("actual training and tuning search parents overlap")
    if identity_authorization is not None:
        _validate_original_identity(identity_authorization, checked, mappings, parent_hashes)
    # Recheck all referenced publications after the final shard has been inspected.
    after = replay_v5_k1_balanced_full_search_collection(plan_path, **collection_binding)
    if checked != after:
        raise RuntimeError("search collection changed while preparing training inputs")
    if identity_authorization is not None:
        _validate_original_identity(identity_authorization, after, mappings, parent_hashes)
    if original_files is not None:
        actual, final_files = _replay_original_publications(after["plan"], identity_publication_paths)
        if final_files != original_files or actual.to_payload() != identity_authorization.to_payload():
            raise RuntimeError("original identity publications changed across training input preparation")
    projection = V5K1SearchProjectionEvidence(
            plan_sha256=checked["plan"]["plan_sha256"],
            inventory_file_sha256=checked["inventory_file_identity"]["sha256"],
            completion_file_sha256=checked["completion_file_identity"]["sha256"],
            original_identity_authorization_sha256=(
                None if identity_authorization is None else identity_authorization.sha256
            ),
            mappings=tuple(mappings),
        )
    inventory = None
    if original_files is not None:
        # Preserve this collection's frozen source. A different training consumer
        # must explicitly rebind and replay it, never relabel this source silently.
        inventory = build_v5_k1_training_inventory(
            source_archive_sha256=checked["plan"]["source"]["archive_sha256"],
            source_bundle_sha256=checked["plan"]["source"]["bundle_sha256"],
            train_artifacts=tuple(artifacts["train"]),
            tuning_artifacts=tuple(artifacts["tuning_validation"]),
            train_parent_set_sha256=parent_hashes["train"],
            tuning_parent_set_sha256=parent_hashes["tuning_validation"],
            train_tuning_disjointness_receipt_sha256=identity_authorization.train_tuning_receipt_sha256,
            k1_phase_c_disjointness_receipt_sha256=identity_authorization.three_way_receipt_sha256,
            identity_authorization=identity_authorization, projection_evidence=projection,
        )
    return {
        "artifacts": {role: tuple(values) for role, values in artifacts.items()},
        "parent_set_sha256": parent_hashes,
        "projection_evidence": projection.to_payload(),
        "training_inventory": inventory,
        "original_identity_authorization_sha256": (
            None if identity_authorization is None else identity_authorization.sha256
        ),
        "collection_inventory_file_sha256": checked["inventory_file_identity"]["sha256"],
        "original_identity_publication_files": original_files,
        "writes_performed": False, "gradient_training_authorized": False,
        "phase_c_disjointness_authorized": False, "scientific_acceptance_evidence": False,
    }


def _seal_json(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)
    return read_only_identity(path, "training input publication")


def publish_v5_k1_search_training_inputs(plan_path, *, output_root,
                                         identity_publication_paths, **collection_binding):
    """Publish a prepared inventory on a worker, without authorizing training.

    Existing roots, including incomplete attempts, are never reused. A missing
    completion means that the inventory is not a completed publication.
    """
    from .k1_balanced_full_search_collector_v5 import _worker_guard

    _worker_guard(dry_run=False, hostname=socket.gethostname(), environment=os.environ)
    root = lexical_no_symlinks(Path(output_root), "training input publication root")
    if not root.is_relative_to(MAXWELL_DUST_ROOT) or root == MAXWELL_DUST_ROOT:
        raise ValueError("training input publication must remain below the Maxwell dust root")
    if root.exists():
        raise FileExistsError("refusing to reuse a training input publication root")
    root.parent.resolve(strict=True)
    checked = read_v5_k1_search_training_inputs(
        plan_path, identity_publication_paths=identity_publication_paths, **collection_binding,
    )
    if checked["training_inventory"] is None or not checked["original_identity_publication_files"]:
        raise ValueError("publication requires actual original identity publications")
    inventory = validate_v5_k1_training_inventory(checked["training_inventory"])
    V5K1SearchProjectionEvidence.from_payload(checked["projection_evidence"])
    if (inventory["splits"]["projection_evidence"] != checked["projection_evidence"]
            or inventory["splits"]["identity_authorization_sha256"]
            != checked["original_identity_authorization_sha256"]):
        raise ValueError("prepared training inventory lost its projection or original identity")
    root.mkdir(mode=0o700)
    inventory_path = root / "training-input-inventory-v3.json"
    identity = _seal_json(inventory_path, inventory)
    # The reader has replayed all inputs before and after preparation. Keep its
    # exact original publication identities so the next worker can replay again.
    completion = {
        "schema": "gisaxs.posterior_v8.k1_search_training_input_publication/v1",
        "status": "PASS", "inventory": identity,
        "projection_evidence_sha256": checked["projection_evidence"]["evidence_sha256"],
        "original_identity_publication_files": checked["original_identity_publication_files"],
        "collection_binding": {key: str(value) for key, value in collection_binding.items()},
        "plan_path": str(Path(plan_path).resolve(strict=True)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "inputs_replayed_before_and_after_preparation": True,
        "completion_written_last": True, "gradient_training_authorized": False,
        "phase_c_disjointness_authorized": False, "scientific_acceptance_evidence": False,
    }
    if read_only_identity(inventory_path, "prepared inventory") != identity:
        raise RuntimeError("prepared inventory changed before completion")
    for name, original in checked["original_identity_publication_files"].items():
        if read_only_identity(Path(original["path"]), name) != original:
            raise RuntimeError("original identity publication changed before completion")
    completion["completion_sha256"] = sha256(canonical_json(completion).encode()).hexdigest()
    completion_path = root / "training-input-completion-v1.json"
    completion_identity = _seal_json(completion_path, completion)
    return {"inventory": identity, "completion": completion_identity,
            "gradient_training_authorized": False, "scientific_acceptance_evidence": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--balanced-dataset-launch-plan-path")
    binding_names = (
        "expected_plan_sha256", "expected_plan_file_sha256", "expected_inventory_file_sha256",
        "expected_completion_file_sha256", "source_root", "source_archive",
        "local_sobol_schedule_path", "calibration_path",
    )
    publication_names = (
        "balanced_dataset_completion", "train_tuning_receipt", "phase_c_completion", "three_way_receipt",
    )
    for name in (*binding_names, *publication_names):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = vars(parser.parse_args(argv))
    binding = {name: args[name] for name in binding_names}
    if args["balanced_dataset_launch_plan_path"] is not None:
        binding["balanced_dataset_launch_plan_path"] = args["balanced_dataset_launch_plan_path"]
    result = publish_v5_k1_search_training_inputs(
        args["plan"], output_root=args["output_root"],
        identity_publication_paths={name: args[name] for name in publication_names},
        **binding,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
