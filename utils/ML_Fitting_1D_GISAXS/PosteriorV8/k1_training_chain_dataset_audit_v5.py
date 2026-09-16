"""Small-array preflight for declared all-twelve-branch K1 training inputs."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .grouped_artifact_v5 import array_sha256, canonical_json
from .grouped_dataset_v5 import (
    V5_GROUPED_DATASET_SCHEMA,
    V5_GROUPED_DATASET_VERSION,
    clean_array,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES


V5_K1_DATASET_AUDIT_SCHEMA = "gisaxs.posterior_v8.k1_training_dataset_audit/v1"
V5_K1_DATASET_AUDIT_VERSION = (
    "posterior_v8_v5_2_partial_checked_array_all12_branch_parent_split_replay_v1"
)
V5_K1_PARENT_SET_HASH_SEMANTICS = (
    "sha256_canonical_json_lexicographically_sorted_unique_clean_group_ids_v1"
)
_AUDIT_ARRAYS = (
    clean_array("recipe_canonical_json"),
    clean_array("target_pattern_id"),
    clean_array("split_id"),
    clean_array("clean_group_id"),
)
_MAX_AUDIT_ARRAY_BYTES = 128 * 1024 * 1024
_BRANCH_BY_WIRE = {
    (branch.shape, branch.pattern_id): branch for branch in K1_PHASE_C_BRANCHES
}


def _strict_json(encoded: bytes | str, name: str) -> dict[str, object]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=no_duplicates)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def k1_parent_set_sha256(clean_group_ids: Sequence[str]) -> str:
    values = tuple(clean_group_ids)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("clean group IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("clean group IDs must be unique")
    return sha256(canonical_json(sorted(values)).encode("utf-8")).hexdigest()


def _checked_audit_arrays(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValueError("grouped artifact has duplicate members or no manifest")
            manifest = _strict_json(archive.read("manifest.json"), "grouped manifest")
            manifest_core = dict(manifest)
            manifest_sha = manifest_core.pop("manifest_sha256", None)
            if not isinstance(manifest_sha, str) or manifest_sha != sha256(
                canonical_json(manifest_core).encode("utf-8")
            ).hexdigest():
                raise ValueError("grouped manifest SHA-256 does not reproduce")
            expected = manifest.get("arrays")
            if not isinstance(expected, Mapping):
                raise ValueError("grouped manifest has no checked array inventory")
            arrays = {}
            for name in _AUDIT_ARRAYS:
                metadata = expected.get(name)
                if not isinstance(metadata, Mapping) or set(metadata) != {
                    "dtype",
                    "shape",
                    "sha256",
                }:
                    raise ValueError(f"grouped manifest lacks checked audit array {name}")
                member_name = f"arrays/{name}.npy"
                member = archive.getinfo(member_name)
                if member.file_size > _MAX_AUDIT_ARRAY_BYTES:
                    raise ValueError(f"K1 audit array is unexpectedly large: {name}")
                array = np.load(BytesIO(archive.read(member)), allow_pickle=False)
                if array.dtype.hasobject or array.ndim != 1:
                    raise ValueError(f"K1 audit array is unsafe or not one-dimensional: {name}")
                observed = {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": array_sha256(name, array),
                }
                if dict(metadata) != observed:
                    raise ValueError(f"K1 audit array identity does not reproduce: {name}")
                arrays[name] = array
    except (KeyError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"grouped artifact is unreadable for K1 audit: {path}") from exc
    if (manifest.get("dataset_schema"), manifest.get("dataset_version")) != (
        V5_GROUPED_DATASET_SCHEMA,
        V5_GROUPED_DATASET_VERSION,
    ):
        raise ValueError("K1 audit received an unsupported grouped dataset")
    return manifest, arrays


def _artifact_audit(artifact: Mapping[str, object]) -> dict[str, object]:
    path = Path(str(artifact["path"])).resolve(strict=True)
    manifest, arrays = _checked_audit_arrays(path)
    count = int(artifact["clean_parent_count"])
    if any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("K1 audit arrays disagree with the declared clean-parent count")
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, Mapping) or manifest_counts.get("clean_recipes") != count:
        raise ValueError("grouped manifest disagrees with the declared clean-parent count")
    expected_split = str(artifact["split_id"])
    if arrays[clean_array("recipe_canonical_json")].dtype.kind != "U" or arrays[
        clean_array("split_id")
    ].dtype.kind != "U" or arrays[clean_array("clean_group_id")].dtype.kind != "U":
        raise ValueError("K1 audit string arrays must use checked Unicode dtypes")
    if arrays[clean_array("target_pattern_id")].dtype.kind not in "iu":
        raise ValueError("K1 target pattern IDs must use an integer dtype")
    splits = arrays[clean_array("split_id")]
    if set(str(value) for value in splits) != {expected_split}:
        raise ValueError("grouped clean parents escaped their frozen training split")
    group_ids = tuple(str(value) for value in arrays[clean_array("clean_group_id")])
    parent_sha = k1_parent_set_sha256(group_ids)
    counts = {branch.branch_id: 0 for branch in K1_PHASE_C_BRANCHES}
    recipes = arrays[clean_array("recipe_canonical_json")]
    patterns = arrays[clean_array("target_pattern_id")]
    # Local import avoids a contract-cycle through the balanced-plan owner.
    from .k1_forced_sobol_recipe_v5 import (
        V5_K1_FORCED_SOBOL_RECIPE_SCHEMA,
        decode_v5_k1_forced_recipe_identity,
    )

    for index, encoded in enumerate(recipes):
        recipe = _strict_json(str(encoded), "clean recipe")
        if recipe.get("schema") == V5_K1_FORCED_SOBOL_RECIPE_SCHEMA:
            identity = decode_v5_k1_forced_recipe_identity(str(encoded))
            branch = next(
                value
                for value in K1_PHASE_C_BRANCHES
                if value.branch_id == identity.branch_id
            )
            if (
                identity.split_id != expected_split
                or identity.clean_group_id != group_ids[index]
                or int(patterns[index]) != branch.pattern_id
            ):
                raise ValueError("forced K1 recipe escaped its stored grouped row")
        else:
            query = recipe.get("query")
            if not isinstance(query, Mapping):
                raise ValueError("clean recipe has no K=1 query")
            topology = query.get("topology")
            pattern = recipe.get("branch_pattern_id")
            if (
                not isinstance(topology, list)
                or len(topology) != 1
                or not isinstance(topology[0], str)
                or isinstance(pattern, bool)
                or not isinstance(pattern, Integral)
                or int(patterns[index]) != int(pattern)
            ):
                raise ValueError("clean recipe is not a replayable K=1 generating branch")
            branch = _BRANCH_BY_WIRE.get((topology[0], int(pattern)))
        if branch is None:
            raise ValueError("clean recipe is outside the twelve legal K1 branches")
        counts[branch.branch_id] += 1
    if counts != dict(artifact["branch_counts"]):
        raise ValueError("replayed K1 branch counts disagree with the frozen inventory")
    return {
        "path": str(path),
        "split_id": expected_split,
        "clean_parent_count": count,
        "parent_set_sha256": parent_sha,
        "clean_group_ids": list(group_ids),
        "branch_counts": counts,
    }


def audit_v5_k1_training_datasets(inventory: Mapping[str, object]) -> dict[str, object]:
    """Replay K1 branch and parent identities without loading curve tensors."""

    rows = {
        role: [_artifact_audit(artifact) for artifact in inventory["artifacts"][role]]
        for role in ("train", "tuning_validation")
    }
    parent_sets = {}
    aggregate_counts = {}
    for role, values in rows.items():
        group_ids = [group_id for value in values for group_id in value["clean_group_ids"]]
        parent_sets[role] = k1_parent_set_sha256(group_ids)
        aggregate_counts[role] = {
            branch.branch_id: sum(value["branch_counts"][branch.branch_id] for value in values)
            for branch in K1_PHASE_C_BRANCHES
        }
        expected = inventory["splits"][role]
        if (
            parent_sets[role] != inventory["splits"][f"{role.split('_')[0]}_parent_set_sha256"]
            or aggregate_counts[role] != expected["branch_counts"]
            or sum(aggregate_counts[role].values()) != expected["clean_parent_count"]
        ):
            raise ValueError(f"replayed {role} parent or branch inventory drifted")
    train_groups = {
        value for row in rows["train"] for value in row["clean_group_ids"]
    }
    tuning_groups = {
        value for row in rows["tuning_validation"] for value in row["clean_group_ids"]
    }
    if train_groups & tuning_groups:
        raise ValueError("train and tuning-validation clean parents overlap")
    return {
        "schema": V5_K1_DATASET_AUDIT_SCHEMA,
        "version": V5_K1_DATASET_AUDIT_VERSION,
        "parent_set_hash_semantics": V5_K1_PARENT_SET_HASH_SEMANTICS,
        "train_parent_set_sha256": parent_sets["train"],
        "tuning_parent_set_sha256": parent_sets["tuning_validation"],
        "train_tuning_disjoint": True,
        "branch_counts": aggregate_counts,
        "artifacts": rows,
    }


__all__ = [
    "V5_K1_DATASET_AUDIT_SCHEMA",
    "V5_K1_DATASET_AUDIT_VERSION",
    "V5_K1_PARENT_SET_HASH_SEMANTICS",
    "audit_v5_k1_training_datasets",
    "k1_parent_set_sha256",
]
