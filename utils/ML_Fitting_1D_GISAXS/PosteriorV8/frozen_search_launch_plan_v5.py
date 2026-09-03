"""Write-free Maxwell launch planning for V5.1 frozen label mining."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

from .contract import NUM_TOPOLOGIES
from .frozen_search_pipeline_contract_v5 import (
    V5_SEARCH_PIPELINE_FORMAL_SCOPE,
    V5_SEARCH_PIPELINE_SCOPE,
)
from .frozen_search_launch_contracts_v5 import (
    FROZEN_SEARCH_WRAPPER_RELATIVE,
    LAUNCH_MANIFEST_FILENAME,
    MAXWELL_DUST_ROOT,
    V5FrozenSearchLaunchConfig,
    fingerprint_v5_frozen_search_source,
    load_v5_frozen_search_launch_contracts,
    replay_v5_frozen_search_launch_fingerprints,
    under_v5_launch_root,
    v5_frozen_search_contract_payload,
)
from .frozen_search_workload_v5 import (
    V5FrozenSearchPilotThroughput,
    V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT,
    V5_FROZEN_SEARCH_MAX_PREDICTED_SECONDS,
    V5_FROZEN_SEARCH_PILOT_TOPOLOGY_IDS,
    V5_FROZEN_SEARCH_WORKER_WALLTIME_SECONDS,
    point_workloads_v5,
    shard_windows_v5,
    validate_v5_frozen_search_pilot_scope,
    validate_v5_frozen_search_runtime_gate,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_warmup_launch_plan_v5 import safe_export_value
from .sobol_design_v5 import V5SobolDesign, materialize_v5_design_points_for_indices
from .sobol_recipe_coordinates_v5 import V5_SOBOL_RECIPE_COORDINATE_INDEX
from .search_evidence_receipt_v5 import (
    V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
    V5_SEARCH_LABEL_PURPOSE_PILOT,
)
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from .split_design_v5 import V5SplitPlan


V5_FROZEN_SEARCH_LAUNCH_SCHEMA = (
    "gisaxs.posterior_v8.maxwell_frozen_search_launch/v2"
)
V5_FROZEN_SEARCH_LAUNCH_VERSION = (
    "posterior_v8_atomic_held_arrays_runtime_gate_contract_smoke_v2"
)


def _positive(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _nonnegative(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _split_window_points(
    plan: V5SplitPlan,
    design: V5SobolDesign,
    split: str,
    start: int,
    count: int,
):
    block = next(value for value in plan.blocks if value.name == split)
    if start + count > block.count:
        raise ValueError(f"{split} start/count exceeds the frozen split")
    indices = tuple(range(block.start + start, block.start + start + count))
    return materialize_v5_design_points_for_indices(plan, design, indices)


def _generating_topology_id(point) -> int:
    coordinate = point.unit_coordinates[
        V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]
    ]
    return min(int(coordinate * NUM_TOPOLOGIES), NUM_TOPOLOGIES - 1)


def build_v5_frozen_search_launch_plan(
    config: V5FrozenSearchLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Validate contracts and return a write-free two-array submission plan."""

    if not isinstance(config, V5FrozenSearchLaunchConfig):
        raise TypeError("config must be V5FrozenSearchLaunchConfig")
    train_start = _nonnegative(config.train_start, "train_start")
    validation_start = _nonnegative(config.validation_start, "validation_start")
    train_count = _positive(config.train_recipes, "train_recipes")
    validation_count = _positive(config.validation_recipes, "validation_recipes")
    shard_size = _positive(config.recipes_per_shard, "recipes_per_shard")
    views = tuple(_nonnegative(value, "view_index") for value in config.view_indices)
    if not views or len(set(views)) != len(views):
        raise ValueError("view_indices must be non-empty and unique")
    throughput = V5FrozenSearchPilotThroughput(
        source_id=config.pilot_throughput_source_id,
        effective_seconds_per_exact_forward_call=(
            config.pilot_effective_seconds_per_exact_forward_call
        ),
        runtime_safety_factor=config.runtime_safety_factor,
    )

    source = fingerprint_v5_frozen_search_source(config.source_root)
    run_root = under_v5_launch_root(
        config.run_root, allowed_root, "run_root", must_exist=False
    )
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse an existing run root: {run_root}")
    for name in ("split_plan", "sobol_design", "local_sobol_schedule"):
        selected = under_v5_launch_root(
            getattr(config, name), allowed_root, name, must_exist=True
        )
        if not selected.is_file():
            raise ValueError(f"{name} must be a regular file")
    if config.compatibility_calibration is not None:
        calibration_path = under_v5_launch_root(
            config.compatibility_calibration,
            allowed_root,
            "compatibility_calibration",
            must_exist=True,
        )
        if not calibration_path.is_file():
            raise ValueError("compatibility_calibration must be a regular file")
    if under_v5_launch_root(
        config.source_root, allowed_root, "source_root", must_exist=True
    ) != Path(str(source["source_root"])):
        raise ValueError("source root resolution changed during planning")

    (
        plan,
        design,
        seed_schedule,
        seed_receipt,
        optimizer,
        protocol,
        topology,
        calibration,
        plan_text,
        design_text,
    ) = load_v5_frozen_search_launch_contracts(config)
    validate_v5_frozen_search_pilot_scope(
        train_recipes=train_count,
        validation_recipes=validation_count,
        recipes_per_shard=shard_size,
        selected_topology_ids=topology.selected_topology_ids,
    )
    contracts = v5_frozen_search_contract_payload(
        config,
        plan=plan,
        design=design,
        seed_schedule=seed_schedule,
        seed_receipt=seed_receipt,
        optimizer=optimizer,
        protocol=protocol,
        topology=topology,
        calibration=calibration,
        plan_text=plan_text,
        design_text=design_text,
    )
    formal_contract_smoke = (
        protocol.protocol_tier
        == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
    )
    scientific_scope = (
        V5_SEARCH_PIPELINE_FORMAL_SCOPE
        if formal_contract_smoke
        else V5_SEARCH_PIPELINE_SCOPE
    )
    label_purpose = (
        V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE
        if formal_contract_smoke
        else V5_SEARCH_LABEL_PURPOSE_PILOT
    )
    train_points = _split_window_points(
        plan, design, "train", train_start, train_count
    )
    validation_points = _split_window_points(
        plan,
        design,
        "tuning_validation",
        validation_start,
        validation_count,
    )
    allowed = set(topology.selected_topology_ids)
    missing = [
        (value.sobol_index, _generating_topology_id(value))
        for value in (*train_points, *validation_points)
        if _generating_topology_id(value) not in allowed
    ]
    if missing:
        raise ValueError(
            "selected topology schedule excludes generating parents; first mismatches: "
            f"{missing[:8]}"
        )

    train_workloads = point_workloads_v5(
        train_points,
        design=design,
        selected_topology_ids=topology.selected_topology_ids,
        view_indices=views,
        exact_forward_calls_per_branch=seed_schedule.point_count,
    )
    validation_workloads = point_workloads_v5(
        validation_points,
        design=design,
        selected_topology_ids=topology.selected_topology_ids,
        view_indices=views,
        exact_forward_calls_per_branch=seed_schedule.point_count,
    )

    layout = {
        "logs": str(run_root / "logs"),
        "train_shards": str(run_root / "labels/train"),
        "validation_shards": str(run_root / "labels/tuning_validation"),
        "manifest": str(run_root / LAUNCH_MANIFEST_FILENAME),
    }
    train_windows = shard_windows_v5(
        split="train",
        points=train_points,
        point_workloads=train_workloads,
        split_start=train_start,
        shard_size=shard_size,
        output_directory=Path(layout["train_shards"]),
        throughput=throughput,
    )
    validation_windows = shard_windows_v5(
        split="tuning_validation",
        points=validation_points,
        point_workloads=validation_workloads,
        split_start=validation_start,
        shard_size=shard_size,
        output_directory=Path(layout["validation_shards"]),
        throughput=throughput,
    )
    maximum_predicted_seconds = validate_v5_frozen_search_runtime_gate(
        (*train_windows, *validation_windows)
    )
    for path in (
        *(Path(value["output_root"]) for value in train_windows),
        *(Path(value["output_root"]) for value in validation_windows),
        Path(layout["manifest"]),
    ):
        if path.exists():
            raise FileExistsError(f"planned output already exists: {path}")
    for name, value in {
        **layout,
        "source_root": source["source_root"],
        "split_plan": config.split_plan,
        "sobol_design": config.sobol_design,
        "local_sobol_schedule": config.local_sobol_schedule,
        **(
            {}
            if config.compatibility_calibration is None
            else {"compatibility_calibration": config.compatibility_calibration}
        ),
    }.items():
        safe_export_value(str(value), name, forbid_colon=True)

    arrays = {
        "train": {
            "array_spec": f"0-{len(train_windows) - 1}",
            "task_count": len(train_windows),
            "windows": train_windows,
        },
        "tuning_validation": {
            "array_spec": f"0-{len(validation_windows) - 1}",
            "task_count": len(validation_windows),
            "windows": validation_windows,
        },
    }
    sidecar_paths = {
        split: [f"{value['output_root']}/search-supervision.gvd5" for value in rows["windows"]]
        for split, rows in arrays.items()
    }
    parent_paths = {
        split: [f"{value['output_root']}/grouped-parent.gvd5" for value in rows["windows"]]
        for split, rows in arrays.items()
    }
    core = {
        "schema": V5_FROZEN_SEARCH_LAUNCH_SCHEMA,
        "version": V5_FROZEN_SEARCH_LAUNCH_VERSION,
        "scientific_scope": scientific_scope,
        "run_root": str(run_root),
        "layout": layout,
        "source": source,
        "contracts": contracts,
        "configuration": {
            "train_start": train_start,
            "train_recipes": train_count,
            "validation_start": validation_start,
            "validation_recipes": validation_count,
            "recipes_per_shard": shard_size,
            "view_indices": list(views),
            "selected_topology_ids": list(topology.selected_topology_ids),
        },
        "label_contract": {
            "protocol_tier": protocol.protocol_tier,
            "label_purpose": label_purpose,
            "formal_calibration_contract_smoke_only": formal_contract_smoke,
            "full_training_eligible": False,
            "paper_scale_training_labels_claimed": False,
            "shard_plan_remains_pipeline_pilot_contract": True,
        },
        "workload_gate": {
            **throughput.audit_payload(
                exact_forward_calls_per_branch=seed_schedule.point_count
            ),
            "maximum_predicted_shard_seconds": maximum_predicted_seconds,
            "all_shards_below_timeout_ceiling": True,
            "input_is_not_a_paper_scale_runtime_claim": True,
        },
        "arrays": arrays,
        "afterok_handoff": {
            "automatic_downstream_submission": False,
            "full_trainer_eligible": False,
            "blocked_reason": (
                "K1 pipeline-pilot and formal contract-smoke sidecars require a "
                "separately versioned paper-scale promotion gate"
            ),
            "dependency_template": "afterok:{train_job_id}:{validation_job_id}",
            "train_parent_paths": parent_paths["train"],
            "train_sidecar_paths": sidecar_paths["train"],
            "validation_parent_paths": parent_paths["tuning_validation"],
            "validation_sidecar_paths": sidecar_paths["tuning_validation"],
            "downstream_output_must_be_new": True,
        },
        "login_node_work": "contract/hash/path checks and sbatch only",
        "submission_atomicity": (
            "both arrays submit held; release together after both succeed; otherwise cancel all"
        ),
        "heavy_compute": "Slurm CPU workers only",
        "overwrite_policy": "exclusive versioned run root and child artifacts",
        "failure_recovery": {
            "task_bound_branch_resume_supported": False,
            "failed_shard_directory_is_audit_only": True,
            "retry_requires_a_new_versioned_run_root": True,
            "scale_out_permitted": False,
            "promotion_requirement": (
                "review real Maxwell K1 completion, unverified rate, and measured "
                "seconds per exact-forward call before creating a new launcher version"
            ),
        },
    }
    return {**core, "plan_sha256": sha256(canonical_json(core).encode()).hexdigest()}


__all__ = [
    "FROZEN_SEARCH_WRAPPER_RELATIVE",
    "LAUNCH_MANIFEST_FILENAME",
    "MAXWELL_DUST_ROOT",
    "V5FrozenSearchLaunchConfig",
    "V5_FROZEN_SEARCH_LAUNCH_SCHEMA",
    "V5_FROZEN_SEARCH_LAUNCH_VERSION",
    "V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT",
    "V5_FROZEN_SEARCH_MAX_PREDICTED_SECONDS",
    "V5_FROZEN_SEARCH_PILOT_TOPOLOGY_IDS",
    "V5_FROZEN_SEARCH_WORKER_WALLTIME_SECONDS",
    "build_v5_frozen_search_launch_plan",
    "fingerprint_v5_frozen_search_source",
    "replay_v5_frozen_search_launch_fingerprints",
]
