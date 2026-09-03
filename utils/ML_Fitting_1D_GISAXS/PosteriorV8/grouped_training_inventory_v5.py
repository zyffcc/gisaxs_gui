"""Checked shard inventory, audit payloads, and resident-memory preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
import warnings
import zipfile

import numpy as np

from .candidate_supervision_v5 import SEARCH_OUTCOME_CODE
from .grouped_dataset_v5 import V5GroupedDataset
from .grouped_training_data_v5 import V5_GROUPED_TRAINING_SEMANTICS
from .search_evidence_receipt_v5 import V5SearchEvidenceReceipt
from .search_supervision_overlay_v5 import (
    V5SearchSupervisionOverlay,
    _expand_local_targets,
)
from .search_supervision_sidecar_v5 import branch_label, query_array


V5_MAX_RESIDENT_ARRAY_BYTES = 64 * 1024**3
V5_SHARD_LOADING_POLICY = (
    "checked_zip_shards_kept_separate_but_resident_no_cross_shard_array_merge_v1"
)
V5_SIDECAR_EXPANSION_AUDIT_SCHEMA = (
    "gisaxs.posterior_v8.sidecar_training_expansion_safety/v1"
)
V5_SIDECAR_EXPANSION_AUDIT_VERSION = (
    "actual_overlay_expansion_recipe_and_per_replica_batch_gate_v1"
)
V5_PAPER_GREEN_EXPANDED_ROWS = 4_096
V5_PAPER_GREEN_POSITIVE_NEGATIVE_PAIRS = 1_000_000
V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA = 8_192
V5_VALIDATION_MAX_EXPANDED_ROWS_PER_BATCH = 12_000
V5_MAX_POSITIVE_NEGATIVE_PAIRS_PER_BATCH = 4_000_000


def _positive_batch_size(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result != value or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _integer_distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("sidecar expansion audit requires at least one recipe")
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "min": ordered[0],
        "median": float(np.median(np.asarray(ordered, dtype=np.int64))),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def _role_expansion_records(
    shards: Sequence[V5GroupedShard], role: str
) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    known_codes = set(SEARCH_OUTCOME_CODE.values())
    for shard in shards:
        if shard.role != role or shard.full_overlay is None:
            continue
        arrays = shard.full_overlay.sidecar.arrays
        recipe_values = arrays[branch_label("clean_recipe_index")]
        outcome_values = arrays[branch_label("search_outcome_code")]
        if np.any(recipe_values < 0) or np.any(recipe_values >= shard.dataset.recipe_count):
            raise ValueError("search sidecar recipe index is outside its parent shard")
        if recipe_values.size > 1 and np.any(recipe_values[1:] < recipe_values[:-1]):
            order = np.argsort(recipe_values, kind="stable")
            sorted_recipes = recipe_values[order]
        else:
            order = np.arange(recipe_values.size, dtype=np.int64)
            sorted_recipes = recipe_values
        recipe_ids = np.arange(shard.dataset.recipe_count, dtype=np.int64)
        starts = np.searchsorted(sorted_recipes, recipe_ids, side="left")
        stops = np.searchsorted(sorted_recipes, recipe_ids, side="right")
        for local_recipe in range(shard.dataset.recipe_count):
            selected = order[starts[local_recipe] : stops[local_recipe]].astype(
                np.int32, copy=False
            )
            if selected.size == 0:
                raise ValueError("search sidecar omits a parent recipe")
            raw_outcomes = outcome_values[selected]
            observed_codes = {int(value) for value in raw_outcomes.tolist()}
            if not observed_codes <= known_codes:
                raise ValueError("search sidecar contains an unknown outcome code")
            positive_selected = selected[
                raw_outcomes == SEARCH_OUTCOME_CODE["compatible_found"]
            ]
            expanded_positives, _, _ = _expand_local_targets(
                arrays, positive_selected
            )
            positives = int(expanded_positives.size)
            negatives = int(
                np.count_nonzero(
                    raw_outcomes
                    == SEARCH_OUTCOME_CODE[
                        "no_compatible_found_within_frozen_search_budget"
                    ]
                )
            )
            unverified = int(
                np.count_nonzero(
                    raw_outcomes == SEARCH_OUTCOME_CODE["unverified"]
                )
            )
            records.append(
                {
                    "recipe_index": shard.recipe_offset + local_recipe,
                    "raw_rows": int(selected.size),
                    "expanded_rows": positives + negatives + unverified,
                    "positive_rows": positives,
                    "negative_rows": negatives,
                    "unverified_rows": unverified,
                    "positive_negative_pairs": positives * negatives,
                }
            )
    return records


def _role_expansion_audit(
    records: Sequence[Mapping[str, int]],
    *,
    role: str,
    recipes_per_batch: int,
) -> dict[str, object]:
    if not records:
        raise ValueError(f"{role} sidecar expansion audit has no recipes")
    keys = (
        "raw_rows",
        "expanded_rows",
        "positive_rows",
        "negative_rows",
        "unverified_rows",
        "positive_negative_pairs",
    )
    distributions = {
        key: _integer_distribution([int(record[key]) for record in records])
        for key in keys
    }
    count = min(int(recipes_per_batch), len(records))
    expanded_bound = sum(
        sorted((int(record["expanded_rows"]) for record in records), reverse=True)[
            :count
        ]
    )
    pair_bound = sum(
        sorted(
            (int(record["positive_negative_pairs"]) for record in records),
            reverse=True,
        )[:count]
    )
    row_limit = (
        V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA
        if role == "train"
        else V5_VALIDATION_MAX_EXPANDED_ROWS_PER_BATCH
    )
    if expanded_bound > row_limit or pair_bound > V5_MAX_POSITIVE_NEGATIVE_PAIRS_PER_BATCH:
        status = "red"
        required_action = "reduce_recipe_batch_or_optimize_dense_pairwise_objective"
    elif (
        expanded_bound <= V5_PAPER_GREEN_EXPANDED_ROWS
        and pair_bound <= V5_PAPER_GREEN_POSITIVE_NEGATIVE_PAIRS
    ):
        status = "green"
        required_action = "none"
    else:
        status = "yellow"
        required_action = "maxwell_one_step_peak_memory_smoke_receipt_required"
    encoded = json.dumps(
        [dict(record) for record in records], sort_keys=True, separators=(",", ":")
    )
    return {
        "role": role,
        "recipe_count": len(records),
        "distribution_summary_method": (
            "integer_min_median_nearest_rank_p95_max_and_sum_v1"
        ),
        "per_recipe_distributions": distributions,
        "sum_positive_negative_pairs": distributions[
            "positive_negative_pairs"
        ]["sum"],
        "per_recipe_counts_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
        "batch_projection": {
            "method": "sum_largest_per_recipe_counts_conservative_bound_v1",
            "batch_scope": (
                "per_replica_recipe_batch"
                if role == "train"
                else "validation_recipe_batch"
            ),
            "pair_formula": "sum_r_positive_rows_times_negative_rows",
            "configured_recipes_per_batch": int(recipes_per_batch),
            "recipes_in_projected_batch": count,
            "expanded_rows_B": expanded_bound,
            "positive_negative_pairs": pair_bound,
            "dense_pair_mask_elements_B_squared": expanded_bound * expanded_bound,
            "expanded_rows_limit": row_limit,
            "positive_negative_pairs_limit": (
                V5_MAX_POSITIVE_NEGATIVE_PAIRS_PER_BATCH
            ),
            "green_expanded_rows_limit": V5_PAPER_GREEN_EXPANDED_ROWS,
            "green_positive_negative_pairs_limit": (
                V5_PAPER_GREEN_POSITIVE_NEGATIVE_PAIRS
            ),
            "status": status,
            "hard_gate_passed": status != "red",
            "paper_green": status == "green",
            "required_action": required_action,
        },
    }


def audit_sidecar_training_expansion(
    shards: Sequence[V5GroupedShard],
    *,
    train_recipes_per_replica: int,
    validation_recipes_per_batch: int,
    allow_unsafe_for_engineering: bool,
) -> dict[str, object] | None:
    """Audit actual full-sidecar expansion and reject unsafe formal batches."""

    if type(allow_unsafe_for_engineering) is not bool:
        raise TypeError("allow_unsafe_for_engineering must be boolean")
    train_batch_size = _positive_batch_size(
        train_recipes_per_replica, "train_recipes_per_replica"
    )
    validation_batch_size = _positive_batch_size(
        validation_recipes_per_batch, "validation_recipes_per_batch"
    )
    if not any(shard.full_overlay is not None for shard in shards):
        return None
    roles = {
        "train": _role_expansion_audit(
            _role_expansion_records(shards, "train"),
            role="train",
            recipes_per_batch=train_batch_size,
        ),
        "validation": _role_expansion_audit(
            _role_expansion_records(shards, "validation"),
            role="validation",
            recipes_per_batch=validation_batch_size,
        ),
    }
    red_roles = [
        role
        for role, value in roles.items()
        if value["batch_projection"]["status"] == "red"
    ]
    yellow_roles = [
        role
        for role, value in roles.items()
        if value["batch_projection"]["status"] == "yellow"
    ]
    if red_roles and not allow_unsafe_for_engineering:
        details = "; ".join(
            f"{role}: B={roles[role]['batch_projection']['expanded_rows_B']}/"
            f"{roles[role]['batch_projection']['expanded_rows_limit']}, pairs="
            f"{roles[role]['batch_projection']['positive_negative_pairs']}/"
            f"{roles[role]['batch_projection']['positive_negative_pairs_limit']}"
            for role in red_roles
        )
        raise MemoryError(
            "full-sidecar expansion exceeds the default training safety gate ("
            f"{details}); reduce the recipe batch or use "
            "allow_unsafe_sidecar_expansion_for_engineering=True for an explicitly "
            "non-paper engineering run"
        )
    if allow_unsafe_for_engineering:
        overall_status = "unsafe_engineering_override" if red_roles else "engineering_override"
        readiness = "engineering_only_not_paper_claim_eligible"
        warning = (
            "Explicit unsafe-expansion engineering override is active; this run is "
            "not eligible as a paper training receipt."
        )
        paper_claim_allowed = False
    elif yellow_roles:
        overall_status = "yellow"
        readiness = "maxwell_one_step_peak_memory_smoke_required"
        warning = (
            "Allowed by the hard gate, but a Maxwell one-step peak-memory smoke receipt "
            "is required before a long paper training run."
        )
        paper_claim_allowed = False
    else:
        overall_status = "green"
        readiness = "paper_long_run_preflight_ready"
        warning = "none"
        paper_claim_allowed = True
    result = {
        "schema_version": V5_SIDECAR_EXPANSION_AUDIT_SCHEMA,
        "audit_version": V5_SIDECAR_EXPANSION_AUDIT_VERSION,
        "expansion_source_of_truth": (
            "search_supervision_overlay_v5._expand_local_targets"
        ),
        "roles": roles,
        "red_roles": red_roles,
        "yellow_roles": yellow_roles,
        "overall_status": overall_status,
        "hard_gate_passed": not red_roles,
        "formal_gate_bypassed": bool(
            red_roles and allow_unsafe_for_engineering
        ),
        "paper_claim_allowed": paper_claim_allowed,
        "paper_receipt_eligible_after_required_action": (
            not allow_unsafe_for_engineering
        ),
        "paper_long_run_readiness": readiness,
        "requires_maxwell_one_step_smoke": (
            bool(yellow_roles) and not allow_unsafe_for_engineering
        ),
        "warning": warning,
        "engineering_override_requested": allow_unsafe_for_engineering,
    }
    if overall_status != "green":
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return result


def _declared_array_bytes(path: Path) -> int:
    """Read only ZIP metadata for a preflight bound before materializing arrays."""

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            payload = json.loads(archive.read("manifest.json").decode("utf-8"))
            member_bytes = sum(
                member.file_size
                for member in archive.infolist()
                if member.filename.startswith("arrays/")
            )
        arrays = payload["arrays"]
        if not isinstance(arrays, Mapping):
            raise TypeError
        declared_bytes = 0
        for value in arrays.values():
            if not isinstance(value, Mapping):
                raise TypeError
            shape = value["shape"]
            if not isinstance(shape, list) or any(
                isinstance(item, bool) or int(item) != item or item < 0
                for item in shape
            ):
                raise TypeError
            count = 1
            for item in shape:
                count *= int(item)
            declared_bytes += count * np.dtype(value["dtype"]).itemsize
        return max(declared_bytes, member_bytes)
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot preflight checked artifact array size: {path}") from exc


def preflight_resident_array_bytes(paths: Sequence[Path]) -> int:
    total = sum(_declared_array_bytes(path) for path in paths)
    if total > V5_MAX_RESIDENT_ARRAY_BYTES:
        raise MemoryError(
            "checked ZIP inputs declare more than the 64-GiB resident-array safety "
            "limit; convert the training reader to a checked streaming/mmap format "
            "before paper-scale training"
        )
    return total


def resolved_artifact_paths(
    values: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None,
) -> tuple[Path, ...]:
    if values is None:
        return ()
    raw = (values,) if isinstance(values, (str, os.PathLike)) else tuple(values)
    return tuple(Path(value).resolve() for value in raw)


def _digest_unique_strings(values: np.ndarray) -> dict[str, object]:
    unique = sorted({str(value) for value in values.tolist()})
    encoded = json.dumps(unique, sort_keys=True, separators=(",", ":"))
    return {
        "unique_count": len(unique),
        "set_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
    }


@dataclass(frozen=True)
class V5GroupedShard:
    path: Path
    dataset: V5GroupedDataset
    artifact_sha256: str
    manifest_sha256: str
    role: str
    recipe_offset: int
    full_overlay: V5SearchSupervisionOverlay | None
    full_evidence_receipt: V5SearchEvidenceReceipt | None = None

    def audit_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": str(self.path),
            "dataset_id": self.dataset.manifest["dataset_id"],
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "role": self.role,
            "recipe_offset": self.recipe_offset,
            "recipe_count": self.dataset.recipe_count,
        }
        if self.full_overlay is None:
            payload["search_sidecar"] = None
            payload["search_evidence_receipt"] = None
        else:
            sidecar = self.full_overlay.sidecar
            arrays = sidecar.arrays
            payload["search_sidecar"] = {
                "path": str(self.full_overlay.sidecar_path),
                "sidecar_id": sidecar.manifest["sidecar_id"],
                "artifact_sha256": self.full_overlay.sidecar_artifact_sha256,
                "manifest_sha256": sidecar.manifest["manifest_sha256"],
                "protocol_sha256": sidecar.manifest["protocol_sha256"],
                "counts": dict(sidecar.manifest["counts"]),
                "query_catalog_artifacts": _digest_unique_strings(
                    arrays[query_array("query_catalog_artifact_sha256")]
                ),
                "topology_query_catalogs": _digest_unique_strings(
                    arrays[query_array("topology_query_catalog_sha256")]
                ),
                "complete_branch_catalogs": _digest_unique_strings(
                    arrays[query_array("catalog_sha256")]
                ),
            }
            receipt = self.full_evidence_receipt
            payload["search_evidence_receipt"] = (
                None
                if receipt is None
                else {
                    "path": str(receipt.path),
                    "file_sha256": receipt.file_sha256,
                    "receipt_sha256": receipt.manifest["receipt_sha256"],
                    "label_purpose": receipt.manifest["label_purpose"],
                    "full_training_eligible": receipt.full_training_eligible,
                    "launch_source_bundle_sha256": receipt.manifest[
                        "launch_source_bundle_sha256"
                    ],
                    "executor_source_bundle_sha256": receipt.manifest[
                        "executor_source_bundle_sha256"
                    ],
                    "counts": dict(receipt.manifest["counts"]),
                }
            )
        return payload

    def sidecar_audit_payload(self) -> Mapping[str, object] | None:
        value = self.audit_payload()["search_sidecar"]
        return value if isinstance(value, Mapping) else None


@dataclass(frozen=True)
class V5GroupedTrainingAudit:
    train_artifacts: tuple[Mapping[str, object], ...]
    validation_artifacts: tuple[Mapping[str, object], ...]
    train_sidecar_artifacts: tuple[Mapping[str, object], ...]
    validation_sidecar_artifacts: tuple[Mapping[str, object], ...]
    train_recipe_count: int
    validation_recipe_count: int
    train_full_recipe_count: int
    validation_full_recipe_count: int
    train_outcome_counts: Mapping[str, int]
    validation_outcome_counts: Mapping[str, int]
    train_full_outcome_counts: Mapping[str, int] | None
    validation_full_outcome_counts: Mapping[str, int] | None
    train_protocol_sha256: str
    validation_protocol_sha256: str
    full_search_protocol_sha256: str | None
    phase_data_sources: Mapping[str, Mapping[str, str]]
    split_plan_sha256: str
    sobol_design_sha256: str
    recipe_generator_identity: tuple[str, str, str]
    max_points: int
    replicas: int
    global_recipes_per_step: int
    available_steps_per_epoch: int
    selected_steps_per_epoch: int
    full_available_steps_per_epoch: int
    full_selected_steps_per_epoch: int
    resident_array_bytes: int
    resident_array_limit_bytes: int
    shard_loading_policy: str
    paper_scale_streaming_gate: str
    sidecar_expansion_safety: Mapping[str, object] | None
    full_stage_permitted: bool

    def audit_payload(self) -> dict[str, object]:
        value = asdict(self)
        value["train_artifacts"] = [dict(item) for item in self.train_artifacts]
        value["validation_artifacts"] = [dict(item) for item in self.validation_artifacts]
        value["train_sidecar_artifacts"] = [
            dict(item) for item in self.train_sidecar_artifacts
        ]
        value["validation_sidecar_artifacts"] = [
            dict(item) for item in self.validation_sidecar_artifacts
        ]
        value["train_outcome_counts"] = dict(self.train_outcome_counts)
        value["validation_outcome_counts"] = dict(self.validation_outcome_counts)
        if self.train_full_outcome_counts is not None:
            value["train_full_outcome_counts"] = dict(self.train_full_outcome_counts)
        if self.validation_full_outcome_counts is not None:
            value["validation_full_outcome_counts"] = dict(
                self.validation_full_outcome_counts
            )
        value["phase_data_sources"] = {
            phase: dict(roles) for phase, roles in self.phase_data_sources.items()
        }
        value["recipe_generator_identity"] = list(self.recipe_generator_identity)
        value["recipe_batching_semantics"] = V5_GROUPED_TRAINING_SEMANTICS
        value["unverified_rows_are_negative"] = False
        return value


def training_input_manifest_audit(
    shards: Sequence[V5GroupedShard],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for shard in shards:
        result.append(
            {
                "path": str(shard.path),
                "role": shard.role,
                "kind": "grouped_parent",
                "artifact_sha256": shard.artifact_sha256,
                "manifest": dict(shard.dataset.manifest),
            }
        )
        if shard.full_overlay is not None:
            result.append(
                {
                    "path": str(shard.full_overlay.sidecar_path),
                    "role": shard.role,
                    "kind": "frozen_search_sidecar",
                    "artifact_sha256": shard.full_overlay.sidecar_artifact_sha256,
                    "manifest": dict(shard.full_overlay.sidecar.manifest),
                }
            )
            if shard.full_evidence_receipt is not None:
                result.append(
                    {
                        "path": str(shard.full_evidence_receipt.path),
                        "role": shard.role,
                        "kind": "task_bound_search_evidence_receipt",
                        "artifact_sha256": shard.full_evidence_receipt.file_sha256,
                        "receipt_sha256": shard.full_evidence_receipt.manifest[
                            "receipt_sha256"
                        ],
                        "label_purpose": shard.full_evidence_receipt.manifest[
                            "label_purpose"
                        ],
                        "launch_source_bundle_sha256": (
                            shard.full_evidence_receipt.manifest[
                                "launch_source_bundle_sha256"
                            ]
                        ),
                        "executor_source_bundle_sha256": (
                            shard.full_evidence_receipt.manifest[
                                "executor_source_bundle_sha256"
                            ]
                        ),
                        "counts": dict(
                            shard.full_evidence_receipt.manifest["counts"]
                        ),
                    }
                )
    return result


__all__ = [
    "V5GroupedShard",
    "V5GroupedTrainingAudit",
    "V5_MAX_RESIDENT_ARRAY_BYTES",
    "V5_MAX_POSITIVE_NEGATIVE_PAIRS_PER_BATCH",
    "V5_PAPER_GREEN_EXPANDED_ROWS",
    "V5_PAPER_GREEN_POSITIVE_NEGATIVE_PAIRS",
    "V5_SHARD_LOADING_POLICY",
    "V5_SIDECAR_EXPANSION_AUDIT_SCHEMA",
    "V5_SIDECAR_EXPANSION_AUDIT_VERSION",
    "V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA",
    "V5_VALIDATION_MAX_EXPANDED_ROWS_PER_BATCH",
    "audit_sidecar_training_expansion",
    "preflight_resident_array_bytes",
    "resolved_artifact_paths",
    "training_input_manifest_audit",
]
