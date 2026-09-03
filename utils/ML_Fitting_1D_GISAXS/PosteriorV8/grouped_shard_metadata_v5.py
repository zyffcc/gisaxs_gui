"""Split and Sobol-parent metadata for compact V5.1 grouped shards."""

from __future__ import annotations

from hashlib import sha256
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .grouped_artifact_v5 import canonical_json
from .sobol_recipe_coordinates_v5 import V5_SOBOL_RECIPE_COORDINATE_SHA256
from .sobol_recipe_v5 import (
    V5_SOBOL_CLEAN_RECIPE_SCHEMA,
    V5_SOBOL_CLEAN_RECIPE_VERSION,
)


V5_FORMAL_SOBOL_SHARD_SELECTION_SCHEMA = (
    "gisaxs.posterior_v8.formal_sobol_grouped_shard_selection/v1"
)
V5_FORMAL_SOBOL_SHARD_SELECTION_VERSION = "posterior_v8_v5_1_single_split_direct_sobol_window_v1"
_FORMAL_SELECTION_FIELDS = {
    "schema",
    "version",
    "target_split",
    "selection_mode",
    "split_offset",
    "requested_count",
    "actual_count",
    "shard_index",
    "sobol_index_runs",
    "sobol_indices_contiguous",
    "view_indices",
    "generating_candidate_only",
    "split_plan_sha256",
    "sobol_design_sha256",
    "coordinate_contract_sha256",
    "selection_sha256",
}


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _index_runs(indices: Sequence[int]) -> list[dict[str, int]]:
    ordered = sorted({_integer(value, "sobol_index") for value in indices})
    if not ordered:
        return []
    runs = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append({"start": start, "stop": previous + 1})
            start = value
        previous = value
    runs.append({"start": start, "stop": previous + 1})
    return runs


def build_formal_sobol_shard_selection(
    *,
    target_split: str,
    selection_mode: str,
    split_offset: int,
    requested_count: int,
    selected_indices: Sequence[int],
    shard_index: int | None,
    view_indices: Sequence[int],
    generating_only: bool,
    split_plan_sha256: str,
    sobol_design_sha256: str,
    coordinate_contract_sha256: str,
) -> dict[str, object]:
    """Freeze the exact split-relative window used by one formal shard."""

    if selection_mode not in {"start_count", "shard_index"}:
        raise ValueError("unsupported formal Sobol shard selection mode")
    if type(generating_only) is not bool:
        raise TypeError("generating_only must be a bool")
    indices = tuple(_integer(value, "selected Sobol index") for value in selected_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("selected Sobol indices must be non-empty and unique")
    views = tuple(_integer(value, "view_index") for value in view_indices)
    if not views or len(set(views)) != len(views):
        raise ValueError("view indices must be non-empty and unique")
    shard = None if shard_index is None else _integer(shard_index, "shard_index")
    if (selection_mode == "shard_index") != (shard is not None):
        raise ValueError("shard_index must be present exactly in shard-index mode")
    core = {
        "schema": V5_FORMAL_SOBOL_SHARD_SELECTION_SCHEMA,
        "version": V5_FORMAL_SOBOL_SHARD_SELECTION_VERSION,
        "target_split": _nonempty(target_split, "target_split"),
        "selection_mode": selection_mode,
        "split_offset": _integer(split_offset, "split_offset"),
        "requested_count": _integer(requested_count, "requested_count", minimum=1),
        "actual_count": len(indices),
        "shard_index": shard,
        "sobol_index_runs": _index_runs(indices),
        "sobol_indices_contiguous": len(_index_runs(indices)) == 1,
        "view_indices": list(views),
        "generating_candidate_only": generating_only,
        "split_plan_sha256": _digest(split_plan_sha256, "split_plan_sha256"),
        "sobol_design_sha256": _digest(sobol_design_sha256, "sobol_design_sha256"),
        "coordinate_contract_sha256": _digest(
            coordinate_contract_sha256, "coordinate_contract_sha256"
        ),
    }
    return {
        **core,
        "selection_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def _split_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    clean_split = arrays["clean__split_id"]
    observation_recipe = arrays["observation__recipe_index"]
    candidate_recipe = arrays["candidate_context__recipe_index"]
    split_ids = sorted(set(str(value) for value in clean_split))
    by_split = {}
    for split_id in split_ids:
        recipe_indices = np.flatnonzero(clean_split == split_id)
        observation_count = sum(
            np.count_nonzero(observation_recipe == index) for index in recipe_indices
        )
        candidate_count = sum(
            np.count_nonzero(candidate_recipe == index) for index in recipe_indices
        )
        joined_count = sum(
            np.count_nonzero(observation_recipe == index)
            * np.count_nonzero(candidate_recipe == index)
            for index in recipe_indices
        )
        by_split[split_id] = {
            "clean_recipes": int(recipe_indices.size),
            "observation_views": int(observation_count),
            "candidates": int(candidate_count),
            "joined_examples": int(joined_count),
        }
    return {
        "split_ids": split_ids,
        "mode": "single_split" if len(split_ids) == 1 else "mixed_split_engineering_only",
        "by_split": by_split,
    }


def _sobol_parent_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    indices = np.asarray(arrays["clean__sobol_index"], dtype=np.int64)
    split_hashes = np.asarray(arrays["clean__split_plan_sha256"])
    design_hashes = np.asarray(arrays["clean__sobol_design_sha256"])
    absent = (indices == -1) & (split_hashes == "") & (design_hashes == "")
    present = (indices >= 0) & (split_hashes != "") & (design_hashes != "")
    if np.all(absent):
        return {"mode": "none"}
    if not np.all(present):
        raise ValueError("Sobol parent fields must be absent or complete for the whole shard")
    if len(set(split_hashes.tolist())) != 1 or len(set(design_hashes.tolist())) != 1:
        raise ValueError("one grouped shard must use one Sobol plan and design")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Sobol indices must be unique within one grouped shard")
    runs = _index_runs(indices.tolist())
    return {
        "mode": "single_plan_design",
        "split_plan_sha256": str(split_hashes[0]),
        "sobol_design_sha256": str(design_hashes[0]),
        "index_count": len(indices),
        "index_min": int(np.min(indices)),
        "index_max": int(np.max(indices)),
        "sobol_index_runs": runs,
        "sobol_indices_contiguous": len(runs) == 1,
    }


def build_grouped_shard_metadata(
    arrays: Mapping[str, np.ndarray],
    selection: Mapping[str, object] | None,
    *,
    generating_only: bool,
    clean_recipe_identity: Mapping[str, str],
) -> dict[str, object]:
    split = _split_inventory(arrays)
    parent = _sobol_parent_inventory(arrays)
    selected = None if selection is None else dict(selection)
    if selected is not None:
        if set(selected) != _FORMAL_SELECTION_FIELDS:
            raise ValueError("formal Sobol shard selection fields are unsupported")
        if (
            selected["schema"] != V5_FORMAL_SOBOL_SHARD_SELECTION_SCHEMA
            or selected["version"] != V5_FORMAL_SOBOL_SHARD_SELECTION_VERSION
        ):
            raise ValueError("formal Sobol shard selection uses an unsupported schema")
        core = dict(selected)
        supplied_hash = core.pop("selection_sha256", None)
        if sha256(canonical_json(core).encode("utf-8")).hexdigest() != supplied_hash:
            raise ValueError("formal Sobol shard selection SHA-256 does not reproduce")
        if set(split["split_ids"]) != {selected.get("target_split")}:
            raise ValueError("formal shard selection disagrees with stored split")
        if parent.get("mode") != "single_plan_design":
            raise ValueError("formal shard selection requires complete Sobol parents")
        if clean_recipe_identity.get("schema_version") != V5_SOBOL_CLEAN_RECIPE_SCHEMA or (
            clean_recipe_identity.get("generator_version") != V5_SOBOL_CLEAN_RECIPE_VERSION
        ):
            raise ValueError("formal shard selection requires direct Sobol clean recipes")
        if selected["coordinate_contract_sha256"] != V5_SOBOL_RECIPE_COORDINATE_SHA256:
            raise ValueError("formal shard selection uses the wrong coordinate contract")
        if selected["generating_candidate_only"] is not generating_only:
            raise ValueError("formal shard selection disagrees with candidate build policy")
        if selected["selection_mode"] == "shard_index":
            if selected["shard_index"] is None or selected["split_offset"] != (
                selected["shard_index"] * selected["requested_count"]
            ):
                raise ValueError("formal shard-index selection arithmetic does not reproduce")
        elif selected["selection_mode"] != "start_count" or selected["shard_index"] is not None:
            raise ValueError("formal start/count selection is invalid")
        for selection_key, parent_key in (
            ("split_plan_sha256", "split_plan_sha256"),
            ("sobol_design_sha256", "sobol_design_sha256"),
            ("actual_count", "index_count"),
            ("sobol_index_runs", "sobol_index_runs"),
            ("sobol_indices_contiguous", "sobol_indices_contiguous"),
        ):
            if selected.get(selection_key) != parent.get(parent_key):
                raise ValueError("formal shard selection disagrees with stored Sobol parents")
        expected_views = tuple(selected["view_indices"])
        observation_recipe = arrays["observation__recipe_index"]
        observed_views = arrays["observation__view_index"]
        for recipe_index in range(len(arrays["clean__recipe_id"])):
            actual_views = tuple(
                int(value) for value in observed_views[observation_recipe == recipe_index]
            )
            if actual_views != expected_views:
                raise ValueError("formal shard selection disagrees with observation views")
    return {
        "split_inventory": split,
        "sobol_parent_inventory": parent,
        "formal_sobol_selection": selected,
        "formal_single_split_sobol_shard": selected is not None,
    }


__all__ = [
    "V5_FORMAL_SOBOL_SHARD_SELECTION_SCHEMA",
    "V5_FORMAL_SOBOL_SHARD_SELECTION_VERSION",
    "build_formal_sobol_shard_selection",
    "build_grouped_shard_metadata",
]
