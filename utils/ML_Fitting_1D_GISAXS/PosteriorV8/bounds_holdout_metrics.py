"""Supervised holdout metrics in the native V3 bounds-local coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit

from .bounds_first_dataset import RANGE_CODE
from .bounds_first_shards import SPLIT_CODE, BoundsFirstShardConfig, reconstruct_clean_recipe
from .bounds_local_inference import (
    BoundsLocalBranchCondition,
    rank_canonical_joint_branches,
    sample_local_logistic_normal_mixture,
)
from .bounds_local_production import physical_seed_from_local
from .bounds_model_contract import MODEL_INPUT_KEYS, MODEL_OUTPUT_KEYS
from .build_bounds_first_shards import load_shard
from .canonical_branch_catalog import canonical_branch_pattern_is_valid
from .contract import NUM_TOPOLOGIES
from .evaluation import EvaluationThresholds
from .inference_proposals import ContinuousProposalOutput, DiscreteProposalOutput
from .one_click_inference import InferenceBudget
from .production_bridge import (
    ProductionBranchFactory,
    ResolutionSearchPolicy,
    UserSearchSpace,
)
from .reference_bank import CompetingBranch
from .rescue_inference import RescuePolicy


SUPPORTED_HOLDOUT_SPLITS = ("calibration", "test")
SUPPORTED_RANGE_SELECTIONS = ("all", "full", "wide", "narrow")


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _cutoffs(values: Sequence[int], name: str, *, maximum: int | None = None):
    result = tuple(_integer(value, name, minimum=1) for value in values)
    if not result or len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be unique, non-empty, and increasing")
    if maximum is not None and result[-1] > maximum:
        raise ValueError(f"{name} values must not exceed {maximum}")
    return result


@dataclass(frozen=True, kw_only=True)
class BoundsHoldoutAuditConfig:
    split: str = "test"
    range_selection: str = "all"
    maximum_examples: int = 512
    batch_size: int = 64
    topology_ks: tuple[int, ...] = (1, 3, 8)
    canonical_joint_ks: tuple[int, ...] = (1, 8, 32)
    oracle_mixture_limit: int = 12
    oracle_samples_per_mixture: int = 2
    oracle_best_of_n: tuple[int, ...] = (1, 4, 12, 24)
    exact_maximum_examples: int = 8
    target_parameter_mode_count: int = 1
    seed: int = 20260903
    inference_budget: InferenceBudget = field(
        default_factory=lambda: InferenceBudget(
            topology_beam_size=8,
            branch_beam_size=8,
            mixture_components_per_branch=4,
            samples_per_mixture=2,
            per_candidate_forward_evaluation_limit=128,
            forward_evaluation_limit=1024,
        )
    )
    rescue_policy: RescuePolicy = field(
        default_factory=lambda: RescuePolicy(
            target_candidate_count=8,
            fallback_attempt_limit=32,
            sobol_seeds_per_branch=4,
        )
    )
    thresholds: EvaluationThresholds = field(
        default_factory=lambda: EvaluationThresholds(
            raw_exact_log_rmse_max=0.10,
            standardized_exact_log_rmse_max=3.0,
            parameter_mode_distance_max=0.08,
            raw_curve_equivalence_log_rmse_max=0.02,
            reference_mode_distance_max=0.10,
        )
    )

    def __post_init__(self) -> None:
        split = str(self.split).strip().lower()
        if split not in SUPPORTED_HOLDOUT_SPLITS:
            raise ValueError("V3 holdout audit accepts only calibration or test")
        selection = str(self.range_selection).strip().lower()
        if selection not in SUPPORTED_RANGE_SELECTIONS:
            raise ValueError(
                f"range_selection must be one of {SUPPORTED_RANGE_SELECTIONS}"
            )
        for name in (
            "maximum_examples",
            "batch_size",
            "oracle_mixture_limit",
            "oracle_samples_per_mixture",
            "target_parameter_mode_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        object.__setattr__(
            self,
            "exact_maximum_examples",
            _integer(self.exact_maximum_examples, "exact_maximum_examples"),
        )
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(
            self, "topology_ks", _cutoffs(self.topology_ks, "topology_ks", maximum=34)
        )
        object.__setattr__(
            self,
            "canonical_joint_ks",
            _cutoffs(self.canonical_joint_ks, "canonical_joint_ks", maximum=418),
        )
        maximum_oracle = self.oracle_mixture_limit * self.oracle_samples_per_mixture
        object.__setattr__(
            self,
            "oracle_best_of_n",
            _cutoffs(self.oracle_best_of_n, "oracle_best_of_n", maximum=maximum_oracle),
        )
        if not isinstance(self.inference_budget, InferenceBudget):
            raise TypeError("inference_budget must be an InferenceBudget")
        if not isinstance(self.rescue_policy, RescuePolicy):
            raise TypeError("rescue_policy must be a RescuePolicy")
        if not isinstance(self.thresholds, EvaluationThresholds):
            raise TypeError("thresholds must be EvaluationThresholds")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "range_selection", selection)


@dataclass(frozen=True)
class HoldoutRowReference:
    priority: int
    recipe_index: int
    view_index: int
    path: Path
    row: int


def selection_priority(seed: int, recipe: int, view: int) -> int:
    encoded = f"{seed}:{recipe}:{view}:bounds-v3-holdout".encode("ascii")
    return int.from_bytes(sha256(encoded).digest()[:8], "big")


def select_holdout_rows(dataset_audit, config: BoundsHoldoutAuditConfig):
    references = []
    split_code = SPLIT_CODE[config.split]
    range_code = None if config.range_selection == "all" else RANGE_CODE[config.range_selection]
    for path in dataset_audit.shard_paths:
        shard = load_shard(path)
        selected = shard.arrays["assigned_split"] == split_code
        if range_code is not None:
            selected &= shard.arrays["range_regime"] == range_code
        for row in np.flatnonzero(selected):
            recipe = int(shard.arrays["global_recipe_index"][row])
            view = int(shard.arrays["view_index"][row])
            references.append(
                HoldoutRowReference(
                    selection_priority(config.seed, recipe, view),
                    recipe,
                    view,
                    path,
                    int(row),
                )
            )
    references.sort(key=lambda item: (item.priority, item.recipe_index, item.view_index))
    if not references:
        raise ValueError("no V4 holdout rows match the requested split/range")
    # Paper statistics use independent clean recipes, not correlated views.  Pick
    # one deterministic view per recipe while retaining the eligible row count.
    by_recipe = {}
    for reference in references:
        by_recipe.setdefault(reference.recipe_index, reference)
    independent = sorted(
        by_recipe.values(),
        key=lambda item: (item.priority, item.recipe_index, item.view_index),
    )
    return (
        tuple(independent[: config.maximum_examples]),
        len(references),
        len(independent),
    )


def materialize_holdout_rows(references: Sequence[HoldoutRowReference]):
    inputs: dict[str, list[np.ndarray]] = {}
    labels: dict[str, list[np.ndarray]] = {}
    provenance = []
    contexts = []
    current_path = None
    shard = None
    for reference in sorted(references, key=lambda item: (str(item.path), item.row)):
        if reference.path != current_path:
            shard = load_shard(reference.path)
            current_path = reference.path
        assert shard is not None
        row = reference.row
        topology_id = int(shard.arrays["topology_id"][row])
        pattern_id = int(shard.arrays["branch_pattern_id"][row])
        if not canonical_branch_pattern_is_valid(topology_id, pattern_id):
            raise ValueError("holdout row has a noncanonical physical branch")
        clean = reconstruct_clean_recipe(
            BoundsFirstShardConfig(**shard.metadata["config"]), reference.recipe_index
        )
        if (
            clean.branch_pattern_id != pattern_id
            or clean.label.bounds.sha256
            != shard.arrays["bounds_sha256"][row].decode("ascii")
        ):
            raise ValueError("holdout row does not reproduce its physical bounds/branch")
        policy = (
            ResolutionSearchPolicy(
                presence="required", bounds=clean.label.bounds.resolution_bounds
            )
            if clean.label.bounds.resolution_bounds is not None
            else ResolutionSearchPolicy(presence="absent")
        )
        factory = ProductionBranchFactory(
            UserSearchSpace.for_components(clean.label.bounds.component_bounds, resolution=policy)
        )
        context = factory.context_for(
            CompetingBranch(topology_id=topology_id, pattern_id=pattern_id)
        )
        if context is None:
            raise ValueError("truth branch is infeasible in its own V4 user bounds")
        contexts.append(context)
        topology = np.zeros(NUM_TOPOLOGIES, dtype=np.float32)
        topology[topology_id] = 1.0
        d_present = np.zeros(4, dtype=np.float32)
        d_present[: len(clean.label.bounds.d_present)] = clean.label.bounds.d_present
        row_inputs = {
            "x": shard.arrays["x"][row],
            "point_mask": shard.arrays["point_mask"][row],
            "global_features": shard.arrays["global_features"][row],
            "branch_topology": topology,
            "branch_d_present": d_present,
            "branch_resolution_present": np.asarray(
                [float(clean.label.bounds.resolution_bounds is not None)], np.float32
            ),
            "bounds_embedding": shard.arrays["bounds_embedding"][row],
            "active_dimension_mask": shard.arrays["active_dimension_mask"][row].astype(
                np.float32
            ),
            "varying_dimension_mask": shard.arrays["local_varying_mask"][row].astype(
                np.float32
            ),
        }
        if tuple(row_inputs) != MODEL_INPUT_KEYS:
            raise RuntimeError("holdout adapter drifted from the V3 model input contract")
        for name, value in row_inputs.items():
            inputs.setdefault(name, []).append(np.asarray(value))
        for name in ("target_local_unit", "local_varying_mask"):
            labels.setdefault(name, []).append(np.asarray(shard.arrays[name][row]))
        labels.setdefault("topology_id", []).append(np.asarray(topology_id, np.int32))
        labels.setdefault("branch_pattern_id", []).append(np.asarray(pattern_id, np.int32))
        provenance.append(
            {
                "selection_priority": reference.priority,
                "shard_path": str(reference.path),
                "shard_npz_sha256": shard.metadata["npz_sha256"],
                "row": row,
                "recipe_index": reference.recipe_index,
                "view_index": reference.view_index,
                "recipe_group_id": shard.arrays["recipe_group_id"][row].decode("ascii"),
                "bounds_sha256": clean.label.bounds.sha256,
                "component_count": int(shard.arrays["component_count"][row]),
                "range_regime": int(shard.arrays["range_regime"][row]),
                "noise_id": int(shard.arrays["noise_id"][row]),
                "q_window_id": int(shard.arrays["q_window_id"][row]),
            }
        )
    return (
        {name: np.stack(values) for name, values in inputs.items()},
        {name: np.stack(values) for name, values in labels.items()},
        provenance,
        contexts,
    )


def model_outputs(model, inputs, batch_size: int):
    collected: dict[str, list[np.ndarray]] = {}
    count = int(inputs["x"].shape[0])
    for start in range(0, count, batch_size):
        batch = {name: value[start : start + batch_size] for name, value in inputs.items()}
        output = model(batch, training=False)
        if not isinstance(output, Mapping):
            raise TypeError("V3 model must return named outputs")
        if set(output) != set(MODEL_OUTPUT_KEYS):
            raise ValueError("V3 model outputs do not match the frozen contract")
        for name, value in output.items():
            array = np.asarray(value)
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"V3 output {name!r} contains NaN/Inf")
            collected.setdefault(name, []).append(array)
    return {name: np.concatenate(values) for name, values in collected.items()}


def local_rms(local, target, varying) -> float:
    mask = np.asarray(varying, dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(
        np.sqrt(
            np.mean(
                np.square(np.asarray(local, dtype=np.float64)[mask] - target[mask])
            )
        )
    )


def oracle_record(output, condition, target, varying, config, seed):
    continuous = ContinuousProposalOutput(
        mixture_logits=output["mixture_logits"],
        mixture_loc=output["mixture_loc"],
        mixture_logscale=output["mixture_logscale"],
    )
    limit = min(config.oracle_mixture_limit, continuous.mixture_logits.size)
    if config.oracle_best_of_n[-1] > limit * config.oracle_samples_per_mixture:
        raise ValueError("oracle_best_of_n exceeds mixtures available in the model")
    samples = sample_local_logistic_normal_mixture(
        continuous,
        condition,
        mixture_limit=limit,
        samples_per_mixture=config.oracle_samples_per_mixture,
        seed=seed,
    )
    distances = [local_rms(item.local_unit, target, varying) for item in samples]
    order = sorted(
        range(continuous.mixture_logits.size),
        key=lambda index: (-continuous.mixture_logits[index], index),
    )[:limit]
    centers = [
        local_rms(expit(continuous.mixture_loc[index]), target, varying)
        for index in order
    ]
    compliance = {"local_unit_bounds": 0, "user_bounds": 0, "roundtrip": 0, "physics": 0}
    errors = []
    for item in samples:
        local = np.asarray(item.local_unit)
        if np.all((local >= 0.0) & (local <= 1.0)):
            compliance["local_unit_bounds"] += 1
        try:
            components, resolution = condition.context.user_bounds_codec.decode(local)
            compliance["user_bounds"] += 1
            encoded = condition.context.user_bounds_codec.encode(components, resolution)
            if np.allclose(encoded.unit_cube, local, rtol=0.0, atol=5.0e-12):
                compliance["roundtrip"] += 1
            physical_seed_from_local(item)
            compliance["physics"] += 1
        except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "sample_count": len(samples),
        "varying_dimension_count": int(np.count_nonzero(varying)),
        "sample_best_of_n_local_rms": {
            str(n): float(min(distances[:n])) for n in config.oracle_best_of_n
        },
        "best_mixture_center_local_rms": float(min(centers)),
        "compliance_counts": compliance,
        "compliance_errors": errors,
    }


def evaluate_supervised(inputs, labels, outputs, provenance, contexts, config):
    records = []
    for row, metadata in enumerate(provenance):
        discrete = DiscreteProposalOutput(
            topology_logits=outputs["topology_logits"][row],
            branch_pattern_logits=outputs["branch_pattern_logits"][row],
        )
        ranked = rank_canonical_joint_branches(discrete, topology_limit=34)
        topology_id = int(labels["topology_id"][row])
        pattern_id = int(labels["branch_pattern_id"][row])
        truth = next(
            item
            for item in ranked
            if item.branch.topology_id == topology_id
            and item.branch.pattern_id == pattern_id
        )
        condition = BoundsLocalBranchCondition.build(truth, contexts[row])
        oracle_seed = int(
            np.random.SeedSequence(
                [config.seed, metadata["recipe_index"], metadata["view_index"], 0x4F524143]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        oracle = oracle_record(
            {name: outputs[name][row] for name in ("mixture_logits", "mixture_loc", "mixture_logscale")},
            condition,
            np.asarray(labels["target_local_unit"][row], np.float64),
            np.asarray(labels["local_varying_mask"][row], bool),
            config,
            oracle_seed,
        )
        records.append(
            {
                **metadata,
                "split": config.split,
                "topology_id": topology_id,
                "branch_pattern_id": pattern_id,
                "topology_rank": truth.topology_rank,
                "canonical_branch_rank_given_truth_topology": truth.pattern_rank_within_topology,
                "canonical_joint_branch_rank": truth.joint_rank,
                "oracle_branch": oracle,
            }
        )
    return records


def quantiles(values: Sequence[float]):
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def supervised_summary(records, config):
    total_samples = sum(item["oracle_branch"]["sample_count"] for item in records)
    compliance = {
        name: sum(item["oracle_branch"]["compliance_counts"][name] for item in records)
        / total_samples
        for name in ("local_unit_bounds", "user_bounds", "roundtrip", "physics")
    }
    sample_metrics = {
        str(n): quantiles(
            [item["oracle_branch"]["sample_best_of_n_local_rms"][str(n)] for item in records]
        )
        for n in config.oracle_best_of_n
    }
    return {
        "example_count": len(records),
        "topology_recall_at_k": {
            str(k): float(np.mean([item["topology_rank"] <= k for item in records]))
            for k in config.topology_ks
        },
        "canonical_joint_topology_branch_recall_at_k": {
            str(k): float(np.mean([item["canonical_joint_branch_rank"] <= k for item in records]))
            for k in config.canonical_joint_ks
        },
        "canonical_branch_top1_given_truth_topology": float(
            np.mean([item["canonical_branch_rank_given_truth_topology"] == 1 for item in records])
        ),
        "oracle_branch_sample_best_of_n_local_unit_rms": sample_metrics,
        "oracle_branch_best_mixture_center_local_unit_rms": quantiles(
            [item["oracle_branch"]["best_mixture_center_local_rms"] for item in records]
        ),
        "proposal_compliance_rate": compliance,
        "proposal_compliance_error_count": sum(
            len(item["oracle_branch"]["compliance_errors"]) for item in records
        ),
    }


__all__ = [
    "BoundsHoldoutAuditConfig",
    "HoldoutRowReference",
    "SUPPORTED_HOLDOUT_SPLITS",
    "SUPPORTED_RANGE_SELECTIONS",
    "evaluate_supervised",
    "local_rms",
    "materialize_holdout_rows",
    "model_outputs",
    "oracle_record",
    "quantiles",
    "select_holdout_rows",
    "selection_priority",
    "supervised_summary",
]
