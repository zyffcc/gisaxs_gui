"""Collect formal K1 IID score shards and publish calibrated thresholds."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .compatibility_calibration import (
    CompatibilityCalibrationSample,
    CompatibilityStratum,
    fit_compatibility_calibration,
    load_compatibility_calibration,
    write_compatibility_calibration_atomic,
)
from .grouped_artifact_v5 import canonical_json
from .k1_dataset_disjointness_v5 import (
    V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS,
    V5_K1_RECIPE_SET_HASH_SEMANTICS,
    validate_v5_k1_dataset_disjointness_receipt,
)
from .k1_iid_calibration_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_iid_calibration_launch_inputs,
    validate_v5_k1_iid_calibration_launch_plan,
)
from .k1_iid_calibration_plan_v5 import (
    V5_K1_IID_CALIBRATION_MINIMUM_PER_STRATUM,
    V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
    V5_K1_IID_CALIBRATION_TARGET_COVERAGE,
    V5_K1_IID_CALIBRATION_TOTAL_SAMPLES,
)
from .k1_iid_calibration_shard_v5 import validate_v5_k1_iid_calibration_shard
from .k1_iid_calibration_worker_v5 import (
    V5_K1_IID_CALIBRATION_TASK_COMPLETION_SCHEMA,
    V5_K1_IID_CALIBRATION_TASK_COMPLETION_VERSION,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest


V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_SCHEMA = (
    "gisaxs.posterior_v8.k1_train_tune_phase_c_calibration_disjointness/v2"
)
V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_VERSION = (
    "posterior_v8_v5_2_actual_recipe_group_four_way_disjointness_"
    "overflow_safe_sigma_v2"
)
V5_K1_IID_CALIBRATION_COLLECTION_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_compatibility_calibration_collection/v2"
)
V5_K1_IID_CALIBRATION_COLLECTION_VERSION = (
    "posterior_v8_v5_2_iid_160020_threshold_and_four_way_overflow_safe_sigma_"
    "completion_last_v2"
)
_FOUR_WAY_FIELDS = {
    "schema",
    "version",
    "scientific_role",
    "recipe_set_hash_semantics",
    "clean_group_set_hash_semantics",
    "three_way_receipt",
    "calibration_population",
    "intersection_counts",
    "all_four_populations_recipe_disjoint",
    "all_four_populations_clean_group_disjoint",
    "calibration_exclusion_claim_sha256",
    "claim_limits",
}


def _load_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _immutable(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a real regular file")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _set_sha256(values: Sequence[str]) -> str:
    return sha256(canonical_json(sorted(values)).encode()).hexdigest()


def _sealed_json(
    path: Path,
    core: Mapping[str, object],
    hash_field: str,
    name: str,
) -> dict[str, object]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}")
    payload = {
        **core,
        hash_field: sha256(canonical_json(core).encode()).hexdigest(),
    }
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    _immutable(path, name)
    return payload


def _task_completion(
    task: Mapping[str, object],
    *,
    plan_sha256: str,
    input_identity: Mapping[str, object],
    calibration_plan: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    completion_path = Path(task["completion"])
    _immutable(completion_path, "IID calibration task completion")
    completion = _load_json(
        completion_path,
        "IID calibration task completion",
        4 * 1024 * 1024,
    )
    core = dict(completion)
    supplied = digest(core.pop("completion_sha256", None), "completion_sha256")
    if supplied != sha256(canonical_json(core).encode()).hexdigest():
        raise ValueError("IID calibration task completion self-hash does not reproduce")
    if (
        completion.get("schema") != V5_K1_IID_CALIBRATION_TASK_COMPLETION_SCHEMA
        or completion.get("version") != V5_K1_IID_CALIBRATION_TASK_COMPLETION_VERSION
        or completion.get("status") != "PASS"
        or completion.get("plan_sha256") != plan_sha256
        or completion.get("array_task_id") != task["array_task_id"]
        or completion.get("task") != dict(task)
        or completion.get("immutable_input_identity_pre") != input_identity
        or completion.get("immutable_input_identity_post") != input_identity
        or completion.get("completion_written_after_artifact_seal") is not True
        or completion.get("scientific_acceptance_evidence") is not False
        or completion.get("training_authorization_granted") is not False
        or completion.get("model_selection_authorization_granted") is not False
        or completion.get("compatibility_threshold_authorization_granted") is not False
    ):
        raise ValueError("IID calibration task completion contract drifted")
    artifact_identity = completion.get("artifact")
    artifact_path = Path(task["output"])
    if not isinstance(artifact_identity, Mapping) or artifact_identity.get(
        "path"
    ) != str(artifact_path):
        raise ValueError("IID calibration task artifact identity is incomplete")
    _immutable(artifact_path, "IID calibration score shard")
    metadata = artifact_path.stat()
    if (
        file_sha256(artifact_path, "IID calibration score shard")
        != digest(artifact_identity.get("file_sha256"), "artifact file_sha256")
        or artifact_identity.get("byte_count") != metadata.st_size
        or artifact_identity.get("mode_octal") != "0400"
        or artifact_identity.get("nlink") != 1
    ):
        raise ValueError("IID calibration score shard changed after completion")
    artifact = validate_v5_k1_iid_calibration_shard(
        _load_json(
            artifact_path,
            "IID calibration score shard",
            128 * 1024 * 1024,
        ),
        calibration_plan=calibration_plan,
    )
    manifest = artifact["manifest"]
    if (
        artifact["artifact_self_sha256"]
        != artifact_identity.get("artifact_self_sha256")
        or manifest["manifest_sha256"] != artifact_identity.get("manifest_sha256")
        or manifest["stratum_ordinal"] != task["stratum_ordinal"]
        or manifest["sample_ordinal_start"] != task["sample_ordinal_start"]
        or manifest["sample_count"] != task["sample_count"]
        or manifest["design_coordinate"] != task["design_coordinate"]
        or manifest["compatibility_stratum"] != task["compatibility_stratum"]
    ):
        raise ValueError("IID calibration task artifact window drifted")
    if completion_path.stat().st_mtime_ns < artifact_path.stat().st_mtime_ns:
        raise ValueError("IID calibration task completion was not written last")
    return completion, artifact


def _sample(record: Mapping[str, object]) -> CompatibilityCalibrationSample:
    return CompatibilityCalibrationSample(
        sample_id=record["sample_id"],
        independent_group_id=record["clean_group_id"],
        stratum=CompatibilityStratum(**record["compatibility_stratum"]),
        score=record["score"],
        effective_valid_point_count=record["effective_valid_point_count"],
        acquisition_policy_id=record["acquisition_policy_id"],
        measurement_sigma_available=record["measurement_sigma_available"],
    )


def _build_four_way_receipt(
    *,
    plan: Mapping[str, object],
    three_way: Mapping[str, object],
    recipe_sha256s: Sequence[str],
    clean_group_ids: Sequence[str],
    sample_ids: Sequence[str],
    branch_counts: Mapping[str, int],
    stratum_counts: Mapping[int, int],
    artifact_identities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    calibration_recipes = set(recipe_sha256s)
    calibration_groups = set(clean_group_ids)
    intersections: dict[str, dict[str, int]] = {}
    for role in ("train", "tuning_validation", "phase_c_holdout"):
        population = three_way["populations"][role]
        intersections[f"{role}_vs_calibration"] = {
            "recipe_sha256": len(
                set(population["recipe_sha256s"]) & calibration_recipes
            ),
            "clean_group_id": len(
                set(population["clean_group_ids"]) & calibration_groups
            ),
        }
    if any(count for pair in intersections.values() for count in pair.values()):
        raise RuntimeError("calibration identities overlap train, tune, or Phase-C")
    calibration = {
        "role": "compatibility_calibration",
        "calibration_plan_sha256": plan["calibration_population_plan"]["plan_sha256"],
        "sample_count": len(sample_ids),
        "recipe_count": len(recipe_sha256s),
        "clean_group_count": len(clean_group_ids),
        "recipe_set_sha256": _set_sha256(recipe_sha256s),
        "clean_group_set_sha256": _set_sha256(clean_group_ids),
        "sample_set_sha256": _set_sha256(sample_ids),
        "branch_counts": dict(sorted(branch_counts.items())),
        "stratum_counts": {
            str(key): value for key, value in sorted(stratum_counts.items())
        },
        "score_shards": list(artifact_identities),
    }
    claim_core = {
        "three_way_receipt_sha256": three_way["receipt_sha256"],
        "phase_c_exclusion_claim_sha256": three_way[
            "phase_c_exclusion_claim_sha256"
        ],
        "calibration_plan_sha256": calibration["calibration_plan_sha256"],
        "calibration_recipe_set_sha256": calibration["recipe_set_sha256"],
        "calibration_clean_group_set_sha256": calibration[
            "clean_group_set_sha256"
        ],
        "calibration_sample_set_sha256": calibration["sample_set_sha256"],
        "intersection_counts": intersections,
    }
    receipt_core = {
        "schema": V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_SCHEMA,
        "version": V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_VERSION,
        "scientific_role": "actual_train_tune_phase_c_calibration_identity_exclusion",
        "recipe_set_hash_semantics": V5_K1_RECIPE_SET_HASH_SEMANTICS,
        "clean_group_set_hash_semantics": V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS,
        "three_way_receipt": {
            "path": plan["three_way_disjointness_receipt"]["path"],
            "file_sha256": plan["three_way_disjointness_receipt"]["file_sha256"],
            "receipt_sha256": three_way["receipt_sha256"],
            "phase_c_exclusion_claim_sha256": three_way[
                "phase_c_exclusion_claim_sha256"
            ],
        },
        "calibration_population": calibration,
        "intersection_counts": intersections,
        "all_four_populations_recipe_disjoint": True,
        "all_four_populations_clean_group_disjoint": True,
        "calibration_exclusion_claim_sha256": sha256(
            canonical_json(claim_core).encode()
        ).hexdigest(),
        "claim_limits": {
            "compatibility_calibration_only": True,
            "training_authorization_granted": False,
            "model_selection_authorization_granted": False,
            "phase_c_acceptance_granted": False,
        },
    }
    return {
        **receipt_core,
        "receipt_sha256": sha256(canonical_json(receipt_core).encode()).hexdigest(),
    }


def validate_v5_k1_iid_calibration_four_way_receipt(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate the compact four-population identity-exclusion receipt."""

    if not isinstance(payload, Mapping):
        raise TypeError("four-way disjointness receipt must be an object")
    value = dict(payload)
    supplied = digest(value.pop("receipt_sha256", None), "receipt_sha256")
    if supplied != sha256(canonical_json(value).encode()).hexdigest():
        raise ValueError("four-way disjointness receipt self-hash does not reproduce")
    if set(value) != _FOUR_WAY_FIELDS:
        raise ValueError("four-way disjointness receipt fields are invalid")
    if (
        value["schema"] != V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_SCHEMA
        or value["version"] != V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_VERSION
        or value["recipe_set_hash_semantics"]
        != V5_K1_RECIPE_SET_HASH_SEMANTICS
        or value["clean_group_set_hash_semantics"]
        != V5_K1_CLEAN_GROUP_SET_HASH_SEMANTICS
        or value["all_four_populations_recipe_disjoint"] is not True
        or value["all_four_populations_clean_group_disjoint"] is not True
        or value["claim_limits"]
        != {
            "compatibility_calibration_only": True,
            "training_authorization_granted": False,
            "model_selection_authorization_granted": False,
            "phase_c_acceptance_granted": False,
        }
    ):
        raise ValueError("four-way disjointness receipt contract drifted")
    three_way = value["three_way_receipt"]
    expected_three_way_fields = {
        "path",
        "file_sha256",
        "receipt_sha256",
        "phase_c_exclusion_claim_sha256",
    }
    if not isinstance(three_way, Mapping) or set(three_way) != expected_three_way_fields:
        raise ValueError("four-way receipt lacks its three-way parent identity")
    for name in expected_three_way_fields - {"path"}:
        digest(three_way[name], f"three_way_receipt.{name}")
    calibration = value["calibration_population"]
    expected_calibration_fields = {
        "role",
        "calibration_plan_sha256",
        "sample_count",
        "recipe_count",
        "clean_group_count",
        "recipe_set_sha256",
        "clean_group_set_sha256",
        "sample_set_sha256",
        "branch_counts",
        "stratum_counts",
        "score_shards",
    }
    if (
        not isinstance(calibration, Mapping)
        or set(calibration) != expected_calibration_fields
        or calibration["role"] != "compatibility_calibration"
        or calibration["sample_count"] != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or calibration["recipe_count"] != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or calibration["clean_group_count"] != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or set(calibration["branch_counts"])
        != {branch.branch_id for branch in K1_PHASE_C_BRANCHES}
        or calibration["stratum_counts"]
        != {
            str(index): V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM
            for index in range(60)
        }
        or len(calibration["score_shards"]) != 60
    ):
        raise ValueError("four-way calibration population drifted")
    for name in (
        "calibration_plan_sha256",
        "recipe_set_sha256",
        "clean_group_set_sha256",
        "sample_set_sha256",
    ):
        digest(calibration[name], f"calibration_population.{name}")
    intersections = value["intersection_counts"]
    expected_pairs = {
        f"{role}_vs_calibration"
        for role in ("train", "tuning_validation", "phase_c_holdout")
    }
    if (
        not isinstance(intersections, Mapping)
        or set(intersections) != expected_pairs
        or any(
            pair != {"recipe_sha256": 0, "clean_group_id": 0}
            for pair in intersections.values()
        )
    ):
        raise ValueError("four-way intersection counts do not prove exclusion")
    claim_core = {
        "three_way_receipt_sha256": three_way["receipt_sha256"],
        "phase_c_exclusion_claim_sha256": three_way[
            "phase_c_exclusion_claim_sha256"
        ],
        "calibration_plan_sha256": calibration["calibration_plan_sha256"],
        "calibration_recipe_set_sha256": calibration["recipe_set_sha256"],
        "calibration_clean_group_set_sha256": calibration[
            "clean_group_set_sha256"
        ],
        "calibration_sample_set_sha256": calibration["sample_set_sha256"],
        "intersection_counts": intersections,
    }
    if value["calibration_exclusion_claim_sha256"] != sha256(
        canonical_json(claim_core).encode()
    ).hexdigest():
        raise ValueError("calibration exclusion claim hash does not reproduce")
    return dict(payload)


def collect_v5_k1_iid_calibration(
    plan_path: str | os.PathLike[str],
    *,
    expected_plan_sha256: str,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate all 60 shards, fit thresholds, and publish completion last."""

    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    if host.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("IID calibration collection is forbidden on Maxwell login nodes")
    if not env.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("IID calibration collection requires a Slurm worker")
    path = lexical_no_symlinks(Path(plan_path), "IID calibration launch plan").resolve(
        strict=True
    )
    _immutable(path, "IID calibration launch plan")
    plan = validate_v5_k1_iid_calibration_launch_plan(
        _load_json(path, "IID calibration launch plan", 16 * 1024 * 1024)
    )
    if plan["plan_sha256"] != digest(expected_plan_sha256, "expected_plan_sha256"):
        raise ValueError("IID calibration plan differs from the Slurm export")
    input_before = replay_v5_k1_iid_calibration_launch_inputs(
        plan, allowed_root=allowed_root
    )
    artifact_path = Path(plan["layout"]["calibration_artifact"])
    receipt_path = Path(plan["layout"]["four_way_disjointness_receipt"])
    completion_path = Path(plan["layout"]["collection_completion"])
    if any(
        value.exists() or value.is_symlink()
        for value in (artifact_path, receipt_path, completion_path)
    ):
        raise FileExistsError("refusing to reuse IID calibration collection outputs")
    if not all(value.parent.is_dir() for value in (artifact_path, receipt_path, completion_path)):
        raise FileNotFoundError("IID calibration collection output directories do not exist")

    samples: list[CompatibilityCalibrationSample] = []
    recipe_sha256s: list[str] = []
    clean_group_ids: list[str] = []
    sample_ids: list[str] = []
    task_completion_sha256s: list[str] = []
    artifact_identities: list[dict[str, object]] = []
    branch_counts: Counter[str] = Counter()
    stratum_counts: Counter[int] = Counter()
    for task in plan["tasks"]:
        completion, shard = _task_completion(
            task,
            plan_sha256=plan["plan_sha256"],
            input_identity=input_before,
            calibration_plan=plan["calibration_population_plan"],
        )
        manifest = shard["manifest"]
        records = shard["samples"]
        samples.extend(_sample(record) for record in records)
        recipe_sha256s.extend(record["recipe_sha256"] for record in records)
        clean_group_ids.extend(record["clean_group_id"] for record in records)
        sample_ids.extend(record["sample_id"] for record in records)
        branch_counts.update(record["branch_id"] for record in records)
        stratum_counts[manifest["stratum_ordinal"]] += manifest["sample_count"]
        task_completion_sha256s.append(completion["completion_sha256"])
        artifact_identities.append(
            {
                "array_task_id": task["array_task_id"],
                "stratum_ordinal": task["stratum_ordinal"],
                "path": completion["artifact"]["path"],
                "file_sha256": completion["artifact"]["file_sha256"],
                "artifact_self_sha256": completion["artifact"][
                    "artifact_self_sha256"
                ],
                "manifest_sha256": completion["artifact"]["manifest_sha256"],
                "sample_count": manifest["sample_count"],
            }
        )
    expected_strata = {
        index: V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM for index in range(60)
    }
    if (
        len(samples) != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or dict(stratum_counts) != expected_strata
        or set(branch_counts) != {branch.branch_id for branch in K1_PHASE_C_BRANCHES}
        or min(
            len(set(recipe_sha256s)),
            len(set(clean_group_ids)),
            len(set(sample_ids)),
        )
        != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
    ):
        raise RuntimeError("IID calibration population counts or identities drifted")

    dataset_manifest_sha256 = sha256(
        canonical_json(artifact_identities).encode()
    ).hexdigest()
    split_identities = sorted(
        zip(sample_ids, clean_group_ids, recipe_sha256s, strict=True)
    )
    calibration_split_sha256 = sha256(
        canonical_json(split_identities).encode()
    ).hexdigest()
    artifact = fit_compatibility_calibration(
        samples,
        dataset_manifest_sha256=dataset_manifest_sha256,
        calibration_split_sha256=calibration_split_sha256,
        target_coverage=V5_K1_IID_CALIBRATION_TARGET_COVERAGE,
        minimum_samples_per_stratum=V5_K1_IID_CALIBRATION_MINIMUM_PER_STRATUM,
    )
    summary = artifact.input_summary
    if (
        summary.observation_count != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or summary.independent_group_count != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or summary.recipe_stratum_count != V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        or summary.stratum_count != 60
        or len(artifact.strata) != 60
        or any(
            value.observation_count != V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM
            for value in artifact.strata
        )
    ):
        raise RuntimeError("fitted compatibility calibration counts drifted")
    write_compatibility_calibration_atomic(artifact_path, artifact)
    artifact_path.chmod(0o400)
    _immutable(artifact_path, "compatibility calibration artifact")
    reloaded = load_compatibility_calibration(artifact_path)
    if reloaded.sha256 != artifact.sha256:
        raise RuntimeError("persisted compatibility calibration does not replay")

    three_way_path = Path(plan["three_way_disjointness_receipt"]["path"])
    three_way = validate_v5_k1_dataset_disjointness_receipt(
        _load_json(three_way_path, "three-way disjointness receipt", 64 * 1024 * 1024)
    )
    receipt = _build_four_way_receipt(
        plan=plan,
        three_way=three_way,
        recipe_sha256s=recipe_sha256s,
        clean_group_ids=clean_group_ids,
        sample_ids=sample_ids,
        branch_counts=branch_counts,
        stratum_counts=stratum_counts,
        artifact_identities=artifact_identities,
    )
    validate_v5_k1_iid_calibration_four_way_receipt(receipt)
    _sealed_json(
        receipt_path,
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        "receipt_sha256",
        "four-way disjointness receipt",
    )
    input_after = replay_v5_k1_iid_calibration_launch_inputs(
        plan, allowed_root=allowed_root
    )
    if input_after != input_before:
        raise RuntimeError("IID calibration immutable inputs changed during collection")
    completion_core = {
        "schema": V5_K1_IID_CALIBRATION_COLLECTION_SCHEMA,
        "version": V5_K1_IID_CALIBRATION_COLLECTION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "compatibility_threshold_authorization_granted": True,
        "training_authorization_granted": False,
        "model_selection_authorization_granted": False,
        "phase_c_acceptance_granted": False,
        "plan_sha256": plan["plan_sha256"],
        "slurm_job_id": env["SLURM_JOB_ID"],
        "hostname": host,
        "task_count": len(plan["tasks"]),
        "task_completion_sha256s": task_completion_sha256s,
        "population": {
            "sample_count": len(samples),
            "independent_group_count": summary.independent_group_count,
            "recipe_stratum_count": summary.recipe_stratum_count,
            "stratum_count": summary.stratum_count,
            "samples_per_stratum": V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
            "branch_counts": dict(sorted(branch_counts.items())),
            "recipe_set_sha256": _set_sha256(recipe_sha256s),
            "clean_group_set_sha256": _set_sha256(clean_group_ids),
            "sample_set_sha256": _set_sha256(sample_ids),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "calibration_split_sha256": calibration_split_sha256,
        },
        "calibration_artifact": {
            "path": str(artifact_path),
            "file_sha256": file_sha256(artifact_path, "compatibility calibration"),
            "artifact_sha256": artifact.sha256,
            "input_sha256": artifact.input_sha256,
            "target_coverage": artifact.target_coverage,
            "minimum_samples_per_stratum": artifact.minimum_samples_per_stratum,
            "mode_octal": "0400",
            "nlink": 1,
        },
        "four_way_disjointness_receipt": {
            "path": str(receipt_path),
            "file_sha256": file_sha256(receipt_path, "four-way disjointness receipt"),
            "receipt_sha256": receipt["receipt_sha256"],
            "calibration_exclusion_claim_sha256": receipt[
                "calibration_exclusion_claim_sha256"
            ],
            "mode_octal": "0400",
            "nlink": 1,
        },
        "immutable_input_identity_pre": input_before,
        "immutable_input_identity_post": input_after,
        "completion_written_after_artifact_and_receipt_seal": True,
    }
    result = _sealed_json(
        completion_path,
        completion_core,
        "completion_sha256",
        "IID calibration collection completion",
    )
    if completion_path.stat().st_mtime_ns < max(
        artifact_path.stat().st_mtime_ns,
        receipt_path.stat().st_mtime_ns,
    ):
        raise RuntimeError("IID calibration collection completion was not written last")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect_v5_k1_iid_calibration(
        args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_IID_CALIBRATION_COLLECTION_SCHEMA",
    "V5_K1_IID_CALIBRATION_COLLECTION_VERSION",
    "V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_SCHEMA",
    "V5_K1_IID_CALIBRATION_FOUR_WAY_RECEIPT_VERSION",
    "collect_v5_k1_iid_calibration",
    "main",
    "validate_v5_k1_iid_calibration_four_way_receipt",
]
