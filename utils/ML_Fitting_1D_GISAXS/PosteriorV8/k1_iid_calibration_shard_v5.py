"""Immutable score shards for the formal K1 IID compatibility calibration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from numbers import Integral, Real
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np

from .compatibility_calibration import (
    RESERVED_CALIBRATION_SPLIT_ID,
    CompatibilityCalibrationSample,
    CompatibilityStratum,
)
from .evaluation import natural_log_rmse
from .k1_iid_calibration_plan_v5 import (
    V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
    build_v5_k1_iid_calibration_plan,
    compatibility_stratum_for_design,
    materialize_v5_k1_iid_calibration_recipe,
    validate_v5_k1_iid_calibration_plan,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from .observation_v5 import (
    build_v5_observation_data_view,
    sample_v5_uncertainty_provenance,
    v5_acquisition_policy_id,
)
from .simulation import (
    OBSERVATION_DESIGN_STRATUM_UNIVERSE,
    sample_observation_design_stratum,
    sample_observation_view,
)
from .synthetic_recipe_v5 import V5_CLEAN_RECIPE_SCHEMA, V5_CLEAN_RECIPE_VERSION


V5_K1_IID_CALIBRATION_SHARD_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_compatibility_calibration_score_shard/v2"
)
V5_K1_IID_CALIBRATION_SHARD_VERSION = (
    "posterior_v8_v5_2_k1_iid_single_sigma_view_standardized_truth_score_"
    "overflow_safe_rss_v2"
)
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
_TOP_FIELDS = {
    "manifest",
    "samples",
    "scientific_acceptance_evidence",
    "training_authorization_granted",
    "model_selection_authorization_granted",
}
_RECORD_FIELDS = {
    "stratum_ordinal",
    "sample_ordinal",
    "rejection_attempt",
    "branch_id",
    "branch_ordinal",
    "recipe_seed",
    "recipe_sha256",
    "clean_group_id",
    "view_index",
    "sample_id",
    "design_coordinate",
    "compatibility_stratum",
    "score",
    "effective_valid_point_count",
    "acquisition_policy_id",
    "measurement_sigma_available",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _score(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("score must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("score must be finite and non-negative")
    return result


def _set_sha256(values: list[str]) -> str:
    return sha256(_canonical_json(sorted(values)).encode()).hexdigest()


def _selection_sha256(plan_sha256: str, stratum: int, start: int, count: int) -> str:
    return sha256(
        _canonical_json(
            {
                "calibration_plan_sha256": plan_sha256,
                "stratum_ordinal": stratum,
                "sample_ordinal_start": start,
                "sample_count": count,
            }
        ).encode()
    ).hexdigest()


def _sample_record(stratum_ordinal: int, sample_ordinal: int) -> dict[str, object]:
    planned = materialize_v5_k1_iid_calibration_recipe(
        stratum_ordinal=stratum_ordinal,
        sample_ordinal=sample_ordinal,
    )
    observed = build_v5_observation_data_view(
        planned.recipe,
        planned.view_index,
        split_id=RESERVED_CALIBRATION_SPLIT_ID,
    )
    if observed.measurement_sigma is None or observed.acceptance_sigma_log is None:
        raise RuntimeError("calibration accepted a view without measurement sigma")
    source_indices = observed.preprocessed.source_indices[observed.preprocessed.point_mask]
    score = natural_log_rmse(
        observed.clean_intensity[source_indices],
        observed.intensity[source_indices],
        sigma_log=observed.acceptance_sigma_log,
    )
    coordinate = OBSERVATION_DESIGN_STRATUM_UNIVERSE[stratum_ordinal]
    stratum = compatibility_stratum_for_design(coordinate)
    sample = CompatibilityCalibrationSample(
        sample_id=planned.sample_id,
        independent_group_id=planned.clean_group_id,
        stratum=stratum,
        score=score,
        effective_valid_point_count=observed.effective_valid_point_count,
        acquisition_policy_id=observed.acquisition_policy_id,
        measurement_sigma_available=True,
    )
    return {
        "stratum_ordinal": stratum_ordinal,
        "sample_ordinal": sample_ordinal,
        "rejection_attempt": planned.rejection_attempt,
        "branch_id": planned.branch.branch_id,
        "branch_ordinal": K1_PHASE_C_BRANCHES.index(planned.branch),
        "recipe_seed": planned.recipe.recipe_seed,
        "recipe_sha256": planned.recipe.sha256,
        "clean_group_id": planned.clean_group_id,
        "view_index": planned.view_index,
        "sample_id": sample.sample_id,
        "design_coordinate": asdict(coordinate),
        "compatibility_stratum": asdict(stratum),
        "score": sample.score,
        "effective_valid_point_count": sample.effective_valid_point_count,
        "acquisition_policy_id": sample.acquisition_policy_id,
        "measurement_sigma_available": True,
    }


def build_v5_k1_iid_calibration_shard(
    *,
    calibration_plan: Mapping[str, object],
    stratum_ordinal: int,
    sample_ordinal_start: int = 0,
    sample_count: int = V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
) -> dict[str, object]:
    plan = validate_v5_k1_iid_calibration_plan(calibration_plan)
    stratum = _integer(stratum_ordinal, "stratum_ordinal")
    start = _integer(sample_ordinal_start, "sample_ordinal_start")
    count = _integer(sample_count, "sample_count")
    if not 0 <= stratum < len(OBSERVATION_DESIGN_STRATUM_UNIVERSE):
        raise ValueError("stratum_ordinal is outside the frozen universe")
    if start < 0 or count < 1 or start + count > plan["samples_per_stratum"]:
        raise ValueError("requested calibration shard window is invalid")
    records = [_sample_record(stratum, index) for index in range(start, start + count)]
    if (
        len({value["sample_id"] for value in records}) != count
        or len({value["recipe_sha256"] for value in records}) != count
        or len({value["clean_group_id"] for value in records}) != count
    ):
        raise RuntimeError("calibration shard contains duplicate parent identities")
    branches = Counter(value["branch_id"] for value in records)
    recipe_hashes = [value["recipe_sha256"] for value in records]
    group_hashes = [value["clean_group_id"] for value in records]
    sample_hashes = [value["sample_id"] for value in records]
    scores = np.asarray([value["score"] for value in records], dtype=np.float64)
    manifest_core = {
        "schema": V5_K1_IID_CALIBRATION_SHARD_SCHEMA,
        "version": V5_K1_IID_CALIBRATION_SHARD_VERSION,
        "calibration_plan_sha256": plan["plan_sha256"],
        "stratum_ordinal": stratum,
        "design_coordinate": plan["strata"][stratum]["design_coordinate"],
        "compatibility_stratum": plan["strata"][stratum]["compatibility_stratum"],
        "sample_ordinal_start": start,
        "sample_count": count,
        "selection_sha256": _selection_sha256(plan["plan_sha256"], stratum, start, count),
        "clean_recipe_schema": V5_CLEAN_RECIPE_SCHEMA,
        "clean_recipe_version": V5_CLEAN_RECIPE_VERSION,
        "branch_counts": dict(sorted(branches.items())),
        "recipe_set_sha256": _set_sha256(recipe_hashes),
        "clean_group_set_sha256": _set_sha256(group_hashes),
        "sample_set_sha256": _set_sha256(sample_hashes),
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": sha256(_canonical_json(manifest_core).encode()).hexdigest(),
    }
    core = {
        "manifest": manifest,
        "samples": records,
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "model_selection_authorization_granted": False,
    }
    return {**core, "artifact_self_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


def _validate_record(
    record: Mapping[str, object],
    *,
    stratum_ordinal: int,
    expected_sample_ordinal: int,
) -> CompatibilityCalibrationSample:
    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise ValueError("calibration sample fields are incomplete or unsupported")
    value = dict(record)
    if (
        _integer(value["stratum_ordinal"], "stratum_ordinal") != stratum_ordinal
        or _integer(value["sample_ordinal"], "sample_ordinal") != expected_sample_ordinal
    ):
        raise ValueError("calibration sample ordering drifted")
    planned = materialize_v5_k1_iid_calibration_recipe(
        stratum_ordinal=stratum_ordinal,
        sample_ordinal=expected_sample_ordinal,
    )
    if (
        value["rejection_attempt"] != planned.rejection_attempt
        or value["branch_id"] != planned.branch.branch_id
        or value["branch_ordinal"] != K1_PHASE_C_BRANCHES.index(planned.branch)
        or value["recipe_seed"] != planned.recipe.recipe_seed
        or value["recipe_sha256"] != planned.recipe.sha256
        or value["clean_group_id"] != planned.clean_group_id
        or value["view_index"] != planned.view_index
        or value["sample_id"] != planned.sample_id
    ):
        raise ValueError("calibration sample identity does not reproduce")
    coordinate = OBSERVATION_DESIGN_STRATUM_UNIVERSE[stratum_ordinal]
    stratum = compatibility_stratum_for_design(coordinate)
    if value["design_coordinate"] != asdict(coordinate) or value[
        "compatibility_stratum"
    ] != asdict(stratum):
        raise ValueError("calibration sample stratum identity drifted")
    view = sample_observation_view(planned.recipe.recipe_seed, planned.view_index)
    provenance = sample_v5_uncertainty_provenance(
        planned.recipe.recipe_seed, planned.view_index
    )
    if (
        sample_observation_design_stratum(planned.recipe.recipe_seed, planned.view_index)
        != coordinate
        or not provenance.measurement_sigma_available
        or value["acquisition_policy_id"] != v5_acquisition_policy_id(view, provenance)
        or value["measurement_sigma_available"] is not True
    ):
        raise ValueError("calibration acquisition policy does not reproduce")
    return CompatibilityCalibrationSample(
        sample_id=_digest(value["sample_id"], "sample_id"),
        independent_group_id=_digest(value["clean_group_id"], "clean_group_id"),
        stratum=CompatibilityStratum(**value["compatibility_stratum"]),
        score=_score(value["score"]),
        effective_valid_point_count=_integer(
            value["effective_valid_point_count"], "effective_valid_point_count"
        ),
        acquisition_policy_id=value["acquisition_policy_id"],
        measurement_sigma_available=True,
    )


def validate_v5_k1_iid_calibration_shard(
    payload: Mapping[str, object],
    *,
    calibration_plan: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("calibration shard must be an object")
    value = dict(payload)
    supplied = value.pop("artifact_self_sha256", None)
    if supplied != sha256(_canonical_json(value).encode()).hexdigest():
        raise ValueError("calibration shard self-hash does not reproduce")
    if set(value) != _TOP_FIELDS or any(
        value[name] is not False
        for name in (
            "scientific_acceptance_evidence",
            "training_authorization_granted",
            "model_selection_authorization_granted",
        )
    ):
        raise ValueError("calibration shard fields or claim limits are invalid")
    plan = validate_v5_k1_iid_calibration_plan(
        build_v5_k1_iid_calibration_plan()
        if calibration_plan is None
        else calibration_plan
    )
    manifest = value["manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("calibration shard manifest is missing")
    manifest_core = dict(manifest)
    manifest_sha = manifest_core.pop("manifest_sha256", None)
    if manifest_sha != sha256(_canonical_json(manifest_core).encode()).hexdigest():
        raise ValueError("calibration shard manifest hash does not reproduce")
    stratum = _integer(manifest["stratum_ordinal"], "stratum_ordinal")
    start = _integer(manifest["sample_ordinal_start"], "sample_ordinal_start")
    count = _integer(manifest["sample_count"], "sample_count")
    if (
        manifest.get("schema") != V5_K1_IID_CALIBRATION_SHARD_SCHEMA
        or manifest.get("version") != V5_K1_IID_CALIBRATION_SHARD_VERSION
        or manifest.get("calibration_plan_sha256") != plan["plan_sha256"]
        or manifest.get("selection_sha256")
        != _selection_sha256(plan["plan_sha256"], stratum, start, count)
        or manifest.get("design_coordinate") != plan["strata"][stratum]["design_coordinate"]
        or manifest.get("compatibility_stratum")
        != plan["strata"][stratum]["compatibility_stratum"]
        or manifest.get("clean_recipe_schema") != V5_CLEAN_RECIPE_SCHEMA
        or manifest.get("clean_recipe_version") != V5_CLEAN_RECIPE_VERSION
    ):
        raise ValueError("calibration shard manifest contract drifted")
    records = value["samples"]
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("calibration shard sample count drifted")
    checked = [
        _validate_record(
            record,
            stratum_ordinal=stratum,
            expected_sample_ordinal=start + offset,
        )
        for offset, record in enumerate(records)
    ]
    recipe_hashes = [_digest(item["recipe_sha256"], "recipe_sha256") for item in records]
    group_hashes = [item.independent_group_id for item in checked]
    sample_hashes = [item.sample_id for item in checked]
    if min(len(set(values)) for values in (recipe_hashes, group_hashes, sample_hashes)) != count:
        raise ValueError("calibration shard contains duplicate identities")
    scores = np.asarray([item.score for item in checked], dtype=np.float64)
    derived = {
        "branch_counts": dict(sorted(Counter(item["branch_id"] for item in records).items())),
        "recipe_set_sha256": _set_sha256(recipe_hashes),
        "clean_group_set_sha256": _set_sha256(group_hashes),
        "sample_set_sha256": _set_sha256(sample_hashes),
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
    }
    if any(manifest.get(name) != observed for name, observed in derived.items()):
        raise ValueError("calibration shard derived summary does not reproduce")
    return dict(payload)


def write_v5_k1_iid_calibration_shard(
    path: str | os.PathLike[str],
    payload: Mapping[str, object],
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> Path:
    checked = validate_v5_k1_iid_calibration_shard(payload)
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("calibration shard output must be absolute")
    root = allowed_root.resolve(strict=True)
    lexical = Path(os.path.abspath(destination))
    if not lexical.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("calibration shard output must remain under the allowed root")
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError("refusing to overwrite a calibration shard")
    with lexical.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(checked, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    lexical.chmod(0o400)
    metadata = lexical.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("calibration shard was not sealed 0400/nlink1")
    return lexical


__all__ = [
    "V5_K1_IID_CALIBRATION_SHARD_SCHEMA",
    "V5_K1_IID_CALIBRATION_SHARD_VERSION",
    "build_v5_k1_iid_calibration_shard",
    "validate_v5_k1_iid_calibration_shard",
    "write_v5_k1_iid_calibration_shard",
]
