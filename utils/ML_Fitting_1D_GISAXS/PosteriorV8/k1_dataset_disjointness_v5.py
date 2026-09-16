"""Actual recipe/group-hash disjointness receipt for K1 train, tune, and holdout."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .grouped_artifact_v5 import array_sha256, canonical_json
from .grouped_dataset_v5 import (
    V5_GROUPED_DATASET_SCHEMA,
    V5_GROUPED_DATASET_VERSION,
    clean_array,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_SPLIT_ID
from .k1_training_chain_contract_v5 import (
    V5_K1_TRAIN_SPLIT_ID,
    V5_K1_TUNING_SPLIT_ID,
    digest,
)


V5_K1_DATASET_DISJOINTNESS_SCHEMA = (
    "gisaxs.posterior_v8.k1_train_tune_phase_c_disjointness_receipt/v2"
)
V5_K1_DATASET_DISJOINTNESS_VERSION = (
    "posterior_v8_v5_2_actual_recipe_and_clean_group_set_disjointness_v2"
)
V5_K1_TRAIN_TUNING_DISJOINTNESS_SCHEMA = (
    "gisaxs.posterior_v8.k1_train_tuning_disjointness_receipt/v2"
)
V5_K1_TRAIN_TUNING_DISJOINTNESS_VERSION = (
    "posterior_v8_v5_2_actual_train_tuning_recipe_and_clean_group_sets_v2"
)
V5_K1_RECIPE_SET_HASH_SEMANTICS = (
    "sha256_canonical_json_lexicographically_sorted_unique_recipe_sha256s_v1"
)
V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS = (
    "sha256_canonical_json_lexicographically_sorted_unique_clean_group_ids_v1"
)
V5_K1_DATASET_POPULATION_ROLES = (
    "train",
    "tuning_validation",
    "phase_c_holdout",
)
_EXPECTED_SPLIT_BY_ROLE = {
    "train": V5_K1_TRAIN_SPLIT_ID,
    "tuning_validation": V5_K1_TUNING_SPLIT_ID,
    "phase_c_holdout": K1_PHASE_C_SPLIT_ID,
}
_POPULATION_FIELDS = {
    "role",
    "split_id",
    "plan_sha256",
    "artifact_sha256s",
    "manifest_sha256s",
    "recipe_sha256s",
    "clean_group_ids",
    "clean_parent_count",
    "recipe_set_sha256",
    "clean_group_set_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "version",
    "scientific_role",
    "recipe_set_hash_semantics",
    "clean_group_set_hash_semantics",
    "populations",
    "pairwise_intersection_counts",
    "all_recipe_sets_pairwise_disjoint",
    "all_clean_group_sets_pairwise_disjoint",
    "train_tuning_claim_sha256",
    "phase_c_exclusion_claim_sha256",
    "claim_limits",
}
_TRAIN_TUNING_RECEIPT_FIELDS = {
    "schema",
    "version",
    "scientific_role",
    "recipe_set_hash_semantics",
    "clean_group_set_hash_semantics",
    "populations",
    "intersection_counts",
    "recipe_sets_disjoint",
    "clean_group_sets_disjoint",
    "train_tuning_claim_sha256",
    "claim_limits",
}
_POPULATION_ARRAYS = (
    clean_array("recipe_sha256"),
    clean_array("clean_group_id"),
    clean_array("split_id"),
    clean_array("split_plan_sha256"),
)
_MAX_IDENTITY_ARRAY_BYTES = 256 * 1024 * 1024


def _hash_set(values: Sequence[str]) -> str:
    return sha256(canonical_json(sorted(values)).encode("utf-8")).hexdigest()


def _digests(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(digest(value, f"{name}[{index}]") for index, value in enumerate(values))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(sorted(result))


def _file_sha256(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _strict_json_object(encoded: bytes, name: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _under_root(path: Path, allowed_root: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("grouped artifact path must be absolute")
    root = allowed_root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"grouped artifact must be under {root}") from exc
    if not relative.parts:
        raise ValueError("grouped artifact must not be the allowed root")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("grouped artifact path must not traverse a symlink")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("grouped artifact must resolve to a regular file under the root")
    return resolved


@dataclass(frozen=True, kw_only=True)
class V5K1GroupedArtifactBinding:
    path: Path
    artifact_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "artifact_sha256",
            digest(self.artifact_sha256, "artifact_sha256"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            digest(self.manifest_sha256, "manifest_sha256"),
        )


def _grouped_identity_arrays(
    binding: V5K1GroupedArtifactBinding,
    *,
    allowed_root: Path,
) -> dict[str, np.ndarray]:
    path = _under_root(binding.path, allowed_root)
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError("grouped artifact must be immutable 0400/nlink1")
    if _file_sha256(path) != binding.artifact_sha256:
        raise ValueError("grouped artifact SHA-256 does not match its binding")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValueError("grouped artifact has duplicate members or no manifest")
            manifest = _strict_json_object(archive.read("manifest.json"), "grouped manifest")
            core = dict(manifest)
            manifest_sha = core.pop("manifest_sha256", None)
            if (
                manifest_sha != binding.manifest_sha256
                or manifest_sha != sha256(canonical_json(core).encode("utf-8")).hexdigest()
            ):
                raise ValueError("grouped manifest SHA-256 does not reproduce")
            if (manifest.get("dataset_schema"), manifest.get("dataset_version")) != (
                V5_GROUPED_DATASET_SCHEMA,
                V5_GROUPED_DATASET_VERSION,
            ):
                raise ValueError("unsupported grouped dataset for K1 disjointness")
            expected = manifest.get("arrays")
            if not isinstance(expected, Mapping):
                raise ValueError("grouped manifest has no checked array inventory")
            arrays = {}
            for name in _POPULATION_ARRAYS:
                array_metadata = expected.get(name)
                if not isinstance(array_metadata, Mapping) or set(array_metadata) != {
                    "dtype",
                    "shape",
                    "sha256",
                }:
                    raise ValueError(f"grouped manifest lacks identity array {name}")
                member = archive.getinfo(f"arrays/{name}.npy")
                if member.file_size > _MAX_IDENTITY_ARRAY_BYTES:
                    raise ValueError(f"grouped identity array is unexpectedly large: {name}")
                array = np.load(BytesIO(archive.read(member)), allow_pickle=False)
                if array.dtype.hasobject or array.ndim != 1:
                    raise ValueError(f"grouped identity array is unsafe: {name}")
                observed = {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": array_sha256(name, array),
                }
                if dict(array_metadata) != observed:
                    raise ValueError(f"grouped identity array does not reproduce: {name}")
                arrays[name] = array
    except (KeyError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"grouped artifact is unreadable: {path}") from exc
    return arrays


@dataclass(frozen=True, kw_only=True)
class V5K1RecipePopulation:
    """One actual immutable parent population, represented by stored identities."""

    role: str
    split_id: str
    plan_sha256: str
    artifact_sha256s: tuple[str, ...]
    manifest_sha256s: tuple[str, ...]
    recipe_sha256s: tuple[str, ...]
    clean_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in V5_K1_DATASET_POPULATION_ROLES:
            raise ValueError("unsupported K1 dataset population role")
        if self.split_id != _EXPECTED_SPLIT_BY_ROLE[self.role]:
            raise ValueError("K1 dataset population role/split binding is invalid")
        object.__setattr__(self, "plan_sha256", digest(self.plan_sha256, "plan_sha256"))
        artifacts = _digests(self.artifact_sha256s, "artifact_sha256s")
        manifests = _digests(self.manifest_sha256s, "manifest_sha256s")
        if len(artifacts) != len(manifests):
            raise ValueError("artifact and manifest inventories must have equal lengths")
        recipes = _digests(self.recipe_sha256s, "recipe_sha256s")
        groups = _digests(self.clean_group_ids, "clean_group_ids")
        if len(recipes) != len(groups):
            raise ValueError("recipe and clean-group populations must have equal lengths")
        object.__setattr__(self, "artifact_sha256s", artifacts)
        object.__setattr__(self, "manifest_sha256s", manifests)
        object.__setattr__(self, "recipe_sha256s", recipes)
        object.__setattr__(self, "clean_group_ids", groups)

    @property
    def clean_parent_count(self) -> int:
        return len(self.recipe_sha256s)

    @property
    def recipe_set_sha256(self) -> str:
        return _hash_set(self.recipe_sha256s)

    @property
    def clean_group_set_sha256(self) -> str:
        return _hash_set(self.clean_group_ids)

    def audit_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "split_id": self.split_id,
            "plan_sha256": self.plan_sha256,
            "artifact_sha256s": list(self.artifact_sha256s),
            "manifest_sha256s": list(self.manifest_sha256s),
            "recipe_sha256s": list(self.recipe_sha256s),
            "clean_group_ids": list(self.clean_group_ids),
            "clean_parent_count": self.clean_parent_count,
            "recipe_set_sha256": self.recipe_set_sha256,
            "clean_group_set_sha256": self.clean_group_set_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "V5K1RecipePopulation":
        if not isinstance(payload, Mapping) or set(payload) != _POPULATION_FIELDS:
            raise ValueError("K1 population fields are incomplete or unsupported")
        result = cls(
            role=payload["role"],
            split_id=payload["split_id"],
            plan_sha256=payload["plan_sha256"],
            artifact_sha256s=tuple(payload["artifact_sha256s"]),
            manifest_sha256s=tuple(payload["manifest_sha256s"]),
            recipe_sha256s=tuple(payload["recipe_sha256s"]),
            clean_group_ids=tuple(payload["clean_group_ids"]),
        )
        if result.audit_payload() != dict(payload):
            raise ValueError("K1 population derived identities do not reproduce")
        return result


def population_from_v5_k1_grouped_artifacts(
    *,
    role: str,
    split_id: str,
    plan_sha256: str,
    artifacts: Sequence[V5K1GroupedArtifactBinding],
    allowed_root: Path,
) -> V5K1RecipePopulation:
    """Read only checked identity arrays and derive one actual population."""

    bindings = tuple(artifacts)
    if not bindings or not all(
        isinstance(value, V5K1GroupedArtifactBinding) for value in bindings
    ):
        raise ValueError("artifacts must contain grouped-artifact bindings")
    paths = tuple(value.path for value in bindings)
    if len(paths) != len(set(paths)):
        raise ValueError("grouped artifact paths must be unique")
    checked_plan = digest(plan_sha256, "plan_sha256")
    recipes = []
    groups = []
    for binding in bindings:
        arrays = _grouped_identity_arrays(binding, allowed_root=allowed_root)
        sizes = {value.shape for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("grouped identity arrays disagree on clean-parent count")
        if arrays[clean_array("recipe_sha256")].dtype.kind != "U" or arrays[
            clean_array("clean_group_id")
        ].dtype.kind != "U":
            raise ValueError("grouped recipe/group identities must use Unicode arrays")
        if arrays[clean_array("split_id")].dtype.kind != "U" or arrays[
            clean_array("split_plan_sha256")
        ].dtype.kind != "U":
            raise ValueError("grouped split identities must use Unicode arrays")
        if set(str(value) for value in arrays[clean_array("split_id")]) != {split_id}:
            raise ValueError("grouped artifact escaped its declared split")
        if set(str(value) for value in arrays[clean_array("split_plan_sha256")]) != {
            checked_plan
        }:
            raise ValueError("grouped artifact escaped its declared plan")
        recipes.extend(str(value) for value in arrays[clean_array("recipe_sha256")])
        groups.extend(str(value) for value in arrays[clean_array("clean_group_id")])
    return V5K1RecipePopulation(
        role=role,
        split_id=split_id,
        plan_sha256=checked_plan,
        artifact_sha256s=tuple(value.artifact_sha256 for value in bindings),
        manifest_sha256s=tuple(value.manifest_sha256 for value in bindings),
        recipe_sha256s=tuple(recipes),
        clean_group_ids=tuple(groups),
    )


def _claim_payload(
    populations: Mapping[str, V5K1RecipePopulation],
    roles: Sequence[str],
) -> dict[str, object]:
    return {
        "roles": list(roles),
        "populations": {
            role: {
                "split_id": populations[role].split_id,
                "plan_sha256": populations[role].plan_sha256,
                "artifact_sha256s": list(populations[role].artifact_sha256s),
                "manifest_sha256s": list(populations[role].manifest_sha256s),
                "clean_parent_count": populations[role].clean_parent_count,
                "recipe_set_sha256": populations[role].recipe_set_sha256,
                "clean_group_set_sha256": populations[role].clean_group_set_sha256,
            }
            for role in roles
        },
        "recipe_and_clean_group_sets_are_pairwise_disjoint": True,
    }


def build_v5_k1_train_tuning_disjointness_receipt(
    populations: Sequence[V5K1RecipePopulation],
) -> dict[str, object]:
    """Prove only train/tuning disjointness from actual artifact identities."""

    supplied = tuple(populations)
    if not all(isinstance(value, V5K1RecipePopulation) for value in supplied):
        raise TypeError("populations must contain V5K1RecipePopulation values")
    by_role = {value.role: value for value in supplied}
    roles = ("train", "tuning_validation")
    if len(by_role) != len(supplied) or set(by_role) != set(roles):
        raise ValueError("train/tuning receipt requires exactly two role-pure populations")
    if by_role["train"].plan_sha256 != by_role["tuning_validation"].plan_sha256:
        raise ValueError("train and tuning populations must share one balanced plan")
    recipe_count = len(
        set(by_role["train"].recipe_sha256s)
        & set(by_role["tuning_validation"].recipe_sha256s)
    )
    group_count = len(
        set(by_role["train"].clean_group_ids)
        & set(by_role["tuning_validation"].clean_group_ids)
    )
    if recipe_count or group_count:
        raise ValueError("K1 train and tuning populations overlap")
    ordered = {role: by_role[role] for role in roles}
    claim = _claim_payload(ordered, roles)
    core = {
        "schema": V5_K1_TRAIN_TUNING_DISJOINTNESS_SCHEMA,
        "version": V5_K1_TRAIN_TUNING_DISJOINTNESS_VERSION,
        "scientific_role": "actual_train_tuning_identity_disjointness_not_training_authorization",
        "recipe_set_hash_semantics": V5_K1_RECIPE_SET_HASH_SEMANTICS,
        "clean_group_set_hash_semantics": V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS,
        "populations": {role: value.audit_payload() for role, value in ordered.items()},
        "intersection_counts": {
            "recipe_sha256_count": recipe_count,
            "clean_group_id_count": group_count,
        },
        "recipe_sets_disjoint": True,
        "clean_group_sets_disjoint": True,
        "train_tuning_claim_sha256": sha256(
            canonical_json(claim).encode("utf-8")
        ).hexdigest(),
        "claim_limits": {
            "phase_c_exclusion_proven": False,
            "receipt_is_training_inventory": False,
            "receipt_is_training_completion": False,
            "receipt_is_model_acceptance": False,
        },
    }
    return {
        **core,
        "receipt_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def validate_v5_k1_train_tuning_disjointness_receipt(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("K1 train/tuning disjointness receipt must be an object")
    core = dict(payload)
    supplied_sha = digest(core.pop("receipt_sha256", None), "receipt_sha256")
    if set(core) != _TRAIN_TUNING_RECEIPT_FIELDS:
        raise ValueError("K1 train/tuning receipt fields are unsupported")
    if supplied_sha != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("K1 train/tuning receipt SHA-256 does not reproduce")
    populations = core["populations"]
    roles = ("train", "tuning_validation")
    if not isinstance(populations, Mapping) or set(populations) != set(roles):
        raise ValueError("K1 train/tuning population inventory is incomplete")
    replay = build_v5_k1_train_tuning_disjointness_receipt(
        tuple(V5K1RecipePopulation.from_payload(populations[role]) for role in roles)
    )
    if replay != dict(payload):
        raise ValueError("K1 train/tuning disjointness receipt does not replay")
    return replay


def write_v5_k1_train_tuning_disjointness_receipt(
    path: str | Path,
    payload: Mapping[str, object],
) -> Path:
    """Exclusively publish a train/tuning-only receipt as 0400/nlink1."""

    value = validate_v5_k1_train_tuning_disjointness_receipt(payload)
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("train/tuning disjointness receipt path must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a train/tuning receipt")
    if not target.parent.is_dir():
        raise FileNotFoundError("train/tuning receipt parent directory does not exist")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    metadata = target.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("train/tuning disjointness receipt is not 0400/nlink1")
    return target


def build_v5_k1_dataset_disjointness_receipt(
    populations: Sequence[V5K1RecipePopulation],
) -> dict[str, object]:
    """Build a self-hashed receipt from actual stored recipe and group identities."""

    supplied = tuple(populations)
    if not all(isinstance(value, V5K1RecipePopulation) for value in supplied):
        raise TypeError("populations must contain V5K1RecipePopulation values")
    by_role = {value.role: value for value in supplied}
    if len(by_role) != len(supplied) or tuple(sorted(by_role)) != tuple(
        sorted(V5_K1_DATASET_POPULATION_ROLES)
    ):
        raise ValueError("receipt requires exactly train, tuning, and Phase-C populations")
    if by_role["train"].plan_sha256 != by_role["tuning_validation"].plan_sha256:
        raise ValueError("train and tuning populations must share one balanced plan")
    pairs = (
        ("train", "tuning_validation"),
        ("train", "phase_c_holdout"),
        ("tuning_validation", "phase_c_holdout"),
    )
    intersections = {}
    for left, right in pairs:
        recipe_count = len(
            set(by_role[left].recipe_sha256s) & set(by_role[right].recipe_sha256s)
        )
        group_count = len(
            set(by_role[left].clean_group_ids) & set(by_role[right].clean_group_ids)
        )
        intersections[f"{left}__{right}"] = {
            "recipe_sha256_count": recipe_count,
            "clean_group_id_count": group_count,
        }
    if any(
        count
        for values in intersections.values()
        for count in values.values()
    ):
        raise ValueError("K1 train, tuning, and Phase-C populations overlap")
    ordered = {role: by_role[role] for role in V5_K1_DATASET_POPULATION_ROLES}
    train_tuning_claim = _claim_payload(ordered, ("train", "tuning_validation"))
    phase_c_claim = _claim_payload(ordered, V5_K1_DATASET_POPULATION_ROLES)
    core = {
        "schema": V5_K1_DATASET_DISJOINTNESS_SCHEMA,
        "version": V5_K1_DATASET_DISJOINTNESS_VERSION,
        "scientific_role": "actual_parent_identity_disjointness_not_model_acceptance",
        "recipe_set_hash_semantics": V5_K1_RECIPE_SET_HASH_SEMANTICS,
        "clean_group_set_hash_semantics": V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS,
        "populations": {role: value.audit_payload() for role, value in ordered.items()},
        "pairwise_intersection_counts": intersections,
        "all_recipe_sets_pairwise_disjoint": True,
        "all_clean_group_sets_pairwise_disjoint": True,
        "train_tuning_claim_sha256": sha256(
            canonical_json(train_tuning_claim).encode("utf-8")
        ).hexdigest(),
        "phase_c_exclusion_claim_sha256": sha256(
            canonical_json(phase_c_claim).encode("utf-8")
        ).hexdigest(),
        "claim_limits": {
            "receipt_is_training_inventory": False,
            "receipt_is_training_completion": False,
            "receipt_is_phase_c_evaluation": False,
            "receipt_is_model_acceptance": False,
        },
    }
    return {
        **core,
        "receipt_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def validate_v5_k1_dataset_disjointness_receipt(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("K1 dataset disjointness receipt must be an object")
    core = dict(payload)
    supplied_sha = digest(core.pop("receipt_sha256", None), "receipt_sha256")
    if set(core) != _RECEIPT_FIELDS:
        raise ValueError("K1 dataset disjointness receipt fields are unsupported")
    if supplied_sha != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("K1 dataset disjointness receipt SHA-256 does not reproduce")
    populations = core["populations"]
    if not isinstance(populations, Mapping) or set(populations) != set(
        V5_K1_DATASET_POPULATION_ROLES
    ):
        raise ValueError("K1 receipt population inventory is incomplete")
    replay = build_v5_k1_dataset_disjointness_receipt(
        tuple(
            V5K1RecipePopulation.from_payload(populations[role])
            for role in V5_K1_DATASET_POPULATION_ROLES
        )
    )
    if replay != dict(payload):
        raise ValueError("K1 dataset disjointness receipt does not replay")
    return replay


def write_v5_k1_dataset_disjointness_receipt(
    path: str | Path,
    payload: Mapping[str, object],
) -> Path:
    """Exclusively publish a validated receipt as 0400/nlink1."""

    value = validate_v5_k1_dataset_disjointness_receipt(payload)
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError("disjointness receipt parent directory does not exist")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    target.chmod(0o400)
    metadata = target.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("disjointness receipt is not 0400/nlink1")
    return target


__all__ = [
    "V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS",
    "V5_K1_DATASET_DISJOINTNESS_SCHEMA",
    "V5_K1_DATASET_DISJOINTNESS_VERSION",
    "V5_K1_DATASET_POPULATION_ROLES",
    "V5_K1_RECIPE_SET_HASH_SEMANTICS",
    "V5_K1_TRAIN_TUNING_DISJOINTNESS_SCHEMA",
    "V5_K1_TRAIN_TUNING_DISJOINTNESS_VERSION",
    "V5K1GroupedArtifactBinding",
    "V5K1RecipePopulation",
    "build_v5_k1_dataset_disjointness_receipt",
    "build_v5_k1_train_tuning_disjointness_receipt",
    "population_from_v5_k1_grouped_artifacts",
    "validate_v5_k1_dataset_disjointness_receipt",
    "validate_v5_k1_train_tuning_disjointness_receipt",
    "write_v5_k1_dataset_disjointness_receipt",
    "write_v5_k1_train_tuning_disjointness_receipt",
]
