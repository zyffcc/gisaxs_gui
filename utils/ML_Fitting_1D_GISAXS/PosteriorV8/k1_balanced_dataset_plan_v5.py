"""Write-free balanced K1 train/tuning Sobol block plan.

This module owns no files and performs no simulation.  It freezes independent
scrambled-Sobol designs for every split/branch pair and binds the categorical
forcing transform.  Actual grouped artifacts must still publish a separate
recipe-hash disjointness receipt before they can enter the K1 inventory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

import numpy as np

from .k1_branch_forcing_v5 import v5_k1_branch_forcing_contract
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from .k1_phase_c_plan_v5 import build_v5_k1_phase_c_plan
from .k1_training_chain_contract_v5 import (
    V5_K1_TRAIN_SPLIT_ID,
    V5_K1_TUNING_SPLIT_ID,
)
from .sobol_recipe_coordinates_v5 import v5_sobol_recipe_design


V5_K1_BALANCED_DATASET_PLAN_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_train_tuning_sobol_plan/v2"
)
V5_K1_BALANCED_DATASET_PLAN_VERSION = (
    "posterior_v8_v5_2_independent_all12_train_tuning_blocks_v2"
)
V5_K1_BALANCED_DATASET_ROLES = ("train", "tuning_validation")
V5_K1_BALANCED_FORMAL_CONFIGURATION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_train_tuning_formal_configuration/v2"
)
V5_K1_BALANCED_FORMAL_CONFIGURATION_VERSION = (
    "posterior_v8_v5_2_engineering_e1_all12_capacity_bounded_freeze_v2"
)
K1_BALANCED_CANONICAL_RUNTIME_PACKAGE = (
    "utils.ML_Fitting_1D_GISAXS.PosteriorV8"
)
K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH = 1152
K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH = 288
K1_BALANCED_FORMAL_RECIPES_PER_SHARD = 288
K1_BALANCED_FORMAL_VIEW_INDICES = (0, 1, 2)
K1_BALANCED_FORMAL_TRAIN_SEED_LABEL = "gisaxs.posterior_v8.k1.all12.train.v1"
K1_BALANCED_FORMAL_TUNING_SEED_LABEL = (
    "gisaxs.posterior_v8.k1.all12.tuning_validation.v1"
)
K1_BALANCED_FORMAL_TRAIN_MASTER_SCRAMBLE_SEED = int.from_bytes(
    sha256(K1_BALANCED_FORMAL_TRAIN_SEED_LABEL.encode("ascii")).digest()[:4],
    "big",
)
K1_BALANCED_FORMAL_TUNING_MASTER_SCRAMBLE_SEED = int.from_bytes(
    sha256(K1_BALANCED_FORMAL_TUNING_SEED_LABEL.encode("ascii")).digest()[:4],
    "big",
)
_UINT32_MAX = (1 << 32) - 1


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def v5_k1_balanced_formal_configuration_payload() -> dict[str, object]:
    """Return the single frozen E1 train/tuning dataset configuration."""

    core = {
        "schema": V5_K1_BALANCED_FORMAL_CONFIGURATION_SCHEMA,
        "version": V5_K1_BALANCED_FORMAL_CONFIGURATION_VERSION,
        "scientific_stage": "engineering_e1_k1_all12_model_development",
        "train_master_scramble_seed": (
            K1_BALANCED_FORMAL_TRAIN_MASTER_SCRAMBLE_SEED
        ),
        "tuning_master_scramble_seed": (
            K1_BALANCED_FORMAL_TUNING_MASTER_SCRAMBLE_SEED
        ),
        "seed_derivation": {
            "algorithm": "uint32_big_endian_first_four_bytes_of_sha256_ascii_label",
            "train_label": K1_BALANCED_FORMAL_TRAIN_SEED_LABEL,
            "tuning_validation_label": K1_BALANCED_FORMAL_TUNING_SEED_LABEL,
        },
        "parents_per_branch": {
            "train": K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH,
            "tuning_validation": K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH,
        },
        "clean_parent_counts": {
            "train": K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH
            * len(K1_PHASE_C_BRANCHES),
            "tuning_validation": K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH
            * len(K1_PHASE_C_BRANCHES),
        },
        "view_indices": list(K1_BALANCED_FORMAL_VIEW_INDICES),
        "recipes_per_shard": K1_BALANCED_FORMAL_RECIPES_PER_SHARD,
        "selection_basis": (
            "capacity_bounded_balanced_all12_E1_model_development_with_"
            "independent_25_percent_tuning_population"
        ),
        "phase_c_population_was_not_used_as_a_seed_or_identity_source": True,
        "phase_c_holdout_remains_separately_materialized_and_disjoint": True,
        "paper_final_model_claim_granted": False,
        "canonical_runtime_package": K1_BALANCED_CANONICAL_RUNTIME_PACKAGE,
    }
    return {
        **core,
        "configuration_sha256": sha256(
            _canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _uint32(value: object, name: str) -> int:
    result = _positive_integer(value, name)
    if result > _UINT32_MAX:
        raise ValueError(f"{name} must fit in uint32")
    return result


def _branch_scramble_seed(*, role: str, master_seed: int, branch_id: str) -> int:
    raw = sha256(
        b"\0".join(
            (
                V5_K1_BALANCED_DATASET_PLAN_VERSION.encode("ascii"),
                role.encode("ascii"),
                str(master_seed).encode("ascii"),
                branch_id.encode("ascii"),
            )
        )
    ).digest()
    return int.from_bytes(raw[:4], "big")


@dataclass(frozen=True)
class V5K1BalancedSobolBlock:
    role: str
    split_id: str
    branch_id: str
    branch_ordinal: int
    master_scramble_seed: int
    scramble_seed: int
    sobol_index_start: int
    parent_count: int
    sobol_design_sha256: str
    branch_forcing_contract_sha256: str
    block_sha256: str

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5K1BalancedDatasetPlan:
    train_master_scramble_seed: int
    tuning_master_scramble_seed: int
    train_parents_per_branch: int
    tuning_parents_per_branch: int
    phase_c_plan_sha256: str
    blocks: tuple[V5K1BalancedSobolBlock, ...]
    canonical_json: str
    sha256: str
    schema_version: str = V5_K1_BALANCED_DATASET_PLAN_SCHEMA
    version: str = V5_K1_BALANCED_DATASET_PLAN_VERSION

    def audit_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def _derive_block(
    *,
    role: str,
    split_id: str,
    branch_ordinal: int,
    master_seed: int,
    parent_count: int,
) -> V5K1BalancedSobolBlock:
    branch = K1_PHASE_C_BRANCHES[branch_ordinal]
    scramble_seed = _branch_scramble_seed(
        role=role,
        master_seed=master_seed,
        branch_id=branch.branch_id,
    )
    design = v5_sobol_recipe_design(scramble_seed=scramble_seed)
    forcing_sha = str(v5_k1_branch_forcing_contract()["contract_sha256"])
    core = {
        "role": role,
        "split_id": split_id,
        "branch": branch.audit_payload(),
        "branch_ordinal": branch_ordinal,
        "master_scramble_seed": master_seed,
        "scramble_seed": scramble_seed,
        "sobol_index_start": 0,
        "parent_count": parent_count,
        "sobol_design_sha256": design.sha256,
        "branch_forcing_contract_sha256": forcing_sha,
        "continuous_coordinates_are_unmodified": True,
    }
    return V5K1BalancedSobolBlock(
        role=role,
        split_id=split_id,
        branch_id=branch.branch_id,
        branch_ordinal=branch_ordinal,
        master_scramble_seed=master_seed,
        scramble_seed=scramble_seed,
        sobol_index_start=0,
        parent_count=parent_count,
        sobol_design_sha256=design.sha256,
        branch_forcing_contract_sha256=forcing_sha,
        block_sha256=sha256(_canonical_json(core).encode("utf-8")).hexdigest(),
    )


def _plan_core(
    *,
    train_seed: int,
    tuning_seed: int,
    train_count: int,
    tuning_count: int,
    phase_c_sha256: str,
    blocks: tuple[V5K1BalancedSobolBlock, ...],
) -> dict[str, object]:
    return {
        "schema": V5_K1_BALANCED_DATASET_PLAN_SCHEMA,
        "version": V5_K1_BALANCED_DATASET_PLAN_VERSION,
        "branch_forcing_contract_sha256": v5_k1_branch_forcing_contract()[
            "contract_sha256"
        ],
        "phase_c_formal_plan_sha256": phase_c_sha256,
        "phase_c_holdout_is_not_materialized_by_this_plan": True,
        "train_master_scramble_seed": train_seed,
        "tuning_master_scramble_seed": tuning_seed,
        "parents_per_branch": {
            "train": train_count,
            "tuning_validation": tuning_count,
        },
        "clean_parent_counts": {
            "train": train_count * len(K1_PHASE_C_BRANCHES),
            "tuning_validation": tuning_count * len(K1_PHASE_C_BRANCHES),
        },
        "branch_balance_is_exact": True,
        "all_block_scramble_seeds_are_unique": True,
        "actual_recipe_hash_disjointness_receipt_required_before_inventory": True,
        "blocks": [value.audit_payload() for value in blocks],
    }


def build_v5_k1_balanced_dataset_plan(
    *,
    train_master_scramble_seed: int,
    tuning_master_scramble_seed: int,
    train_parents_per_branch: int,
    tuning_parents_per_branch: int,
) -> V5K1BalancedDatasetPlan:
    """Build a replayable plan without selecting unreviewed dataset sizes or seeds."""

    train_seed = _uint32(train_master_scramble_seed, "train_master_scramble_seed")
    tuning_seed = _uint32(tuning_master_scramble_seed, "tuning_master_scramble_seed")
    if train_seed == tuning_seed:
        raise ValueError("train and tuning master scramble seeds must differ")
    train_count = _positive_integer(train_parents_per_branch, "train_parents_per_branch")
    tuning_count = _positive_integer(
        tuning_parents_per_branch, "tuning_parents_per_branch"
    )
    role_values = (
        ("train", V5_K1_TRAIN_SPLIT_ID, train_seed, train_count),
        ("tuning_validation", V5_K1_TUNING_SPLIT_ID, tuning_seed, tuning_count),
    )
    blocks = tuple(
        _derive_block(
            role=role,
            split_id=split_id,
            branch_ordinal=branch_ordinal,
            master_seed=master_seed,
            parent_count=parent_count,
        )
        for role, split_id, master_seed, parent_count in role_values
        for branch_ordinal in range(len(K1_PHASE_C_BRANCHES))
    )
    phase_c = build_v5_k1_phase_c_plan(formal=True)
    seeds = tuple(value.scramble_seed for value in blocks)
    holdout_seeds = tuple(value.scramble_seed for value in phase_c.sobol_blocks)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("train/tuning branch-specific Sobol seeds collided")
    if set(seeds) & set(holdout_seeds):
        raise RuntimeError("train/tuning Sobol seed collided with the Phase-C holdout")
    core = _plan_core(
        train_seed=train_seed,
        tuning_seed=tuning_seed,
        train_count=train_count,
        tuning_count=tuning_count,
        phase_c_sha256=phase_c.sha256,
        blocks=blocks,
    )
    encoded = _canonical_json(core)
    return V5K1BalancedDatasetPlan(
        train_master_scramble_seed=train_seed,
        tuning_master_scramble_seed=tuning_seed,
        train_parents_per_branch=train_count,
        tuning_parents_per_branch=tuning_count,
        phase_c_plan_sha256=phase_c.sha256,
        blocks=blocks,
        canonical_json=encoded,
        sha256=sha256(encoded.encode("utf-8")).hexdigest(),
    )


def build_frozen_v5_k1_balanced_dataset_plan() -> V5K1BalancedDatasetPlan:
    """Build the authorized balanced all-K1 E1 train/tuning plan."""

    return build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=(
            K1_BALANCED_FORMAL_TRAIN_MASTER_SCRAMBLE_SEED
        ),
        tuning_master_scramble_seed=(
            K1_BALANCED_FORMAL_TUNING_MASTER_SCRAMBLE_SEED
        ),
        train_parents_per_branch=K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH,
        tuning_parents_per_branch=K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH,
    )


def is_frozen_v5_k1_balanced_dataset_plan(
    plan: V5K1BalancedDatasetPlan,
) -> bool:
    """Return whether a checked plan is exactly the authorized E1 plan."""

    checked = validate_v5_k1_balanced_dataset_plan(plan)
    return checked == build_frozen_v5_k1_balanced_dataset_plan()


def validate_v5_k1_balanced_dataset_plan(
    plan: V5K1BalancedDatasetPlan,
) -> V5K1BalancedDatasetPlan:
    if not isinstance(plan, V5K1BalancedDatasetPlan):
        raise TypeError("plan must be a V5K1BalancedDatasetPlan")
    if (
        plan.schema_version != V5_K1_BALANCED_DATASET_PLAN_SCHEMA
        or plan.version != V5_K1_BALANCED_DATASET_PLAN_VERSION
    ):
        raise ValueError("unsupported K1 balanced dataset plan schema/version")
    replay = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=plan.train_master_scramble_seed,
        tuning_master_scramble_seed=plan.tuning_master_scramble_seed,
        train_parents_per_branch=plan.train_parents_per_branch,
        tuning_parents_per_branch=plan.tuning_parents_per_branch,
    )
    if plan != replay:
        raise ValueError("K1 balanced dataset plan identity does not reproduce")
    return plan


def v5_k1_balanced_dataset_plan_from_payload(
    payload: Mapping[str, object],
) -> V5K1BalancedDatasetPlan:
    """Strictly rebuild a plan from its minimal authoring inputs and identity."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be an object")
    expected = {
        "train_master_scramble_seed",
        "tuning_master_scramble_seed",
        "train_parents_per_branch",
        "tuning_parents_per_branch",
        "plan_sha256",
    }
    if set(payload) != expected:
        raise ValueError("balanced dataset plan authoring fields are incomplete or unsupported")
    replay = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=payload["train_master_scramble_seed"],
        tuning_master_scramble_seed=payload["tuning_master_scramble_seed"],
        train_parents_per_branch=payload["train_parents_per_branch"],
        tuning_parents_per_branch=payload["tuning_parents_per_branch"],
    )
    if payload["plan_sha256"] != replay.sha256:
        raise ValueError("balanced dataset plan SHA-256 does not reproduce")
    return validate_v5_k1_balanced_dataset_plan(replay)


def v5_k1_balanced_dataset_authoring_payload(
    plan: V5K1BalancedDatasetPlan,
) -> dict[str, object]:
    """Return the minimal replay inputs that may be persisted for workers."""

    checked = validate_v5_k1_balanced_dataset_plan(plan)
    return {
        "train_master_scramble_seed": checked.train_master_scramble_seed,
        "tuning_master_scramble_seed": checked.tuning_master_scramble_seed,
        "train_parents_per_branch": checked.train_parents_per_branch,
        "tuning_parents_per_branch": checked.tuning_parents_per_branch,
        "plan_sha256": checked.sha256,
    }


def write_v5_k1_balanced_dataset_authoring_plan(
    path: str | Path,
    plan: V5K1BalancedDatasetPlan,
) -> Path:
    """Exclusively publish a minimal, immutable plan for dataset workers."""

    target = Path(path)
    if not target.is_absolute():
        raise ValueError("balanced dataset authoring plan path must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a balanced dataset authoring plan")
    if not target.parent.is_dir():
        raise FileNotFoundError("balanced dataset authoring plan parent does not exist")
    payload = v5_k1_balanced_dataset_authoring_payload(plan)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    metadata = target.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("balanced dataset authoring plan is not 0400/nlink1")
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="write the single frozen engineering-E1 train/tuning plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.formal:
        raise ValueError("only the explicitly frozen --formal plan may be published")
    plan = build_frozen_v5_k1_balanced_dataset_plan()
    target = write_v5_k1_balanced_dataset_authoring_plan(args.output, plan)
    print(
        json.dumps(
            {
                "path": str(target),
                "plan_sha256": plan.sha256,
                "formal_configuration": v5_k1_balanced_formal_configuration_payload(),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "K1_BALANCED_FORMAL_RECIPES_PER_SHARD",
    "K1_BALANCED_FORMAL_TRAIN_MASTER_SCRAMBLE_SEED",
    "K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH",
    "K1_BALANCED_FORMAL_TUNING_MASTER_SCRAMBLE_SEED",
    "K1_BALANCED_CANONICAL_RUNTIME_PACKAGE",
    "K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH",
    "K1_BALANCED_FORMAL_VIEW_INDICES",
    "V5_K1_BALANCED_DATASET_PLAN_SCHEMA",
    "V5_K1_BALANCED_DATASET_PLAN_VERSION",
    "V5_K1_BALANCED_DATASET_ROLES",
    "V5_K1_BALANCED_FORMAL_CONFIGURATION_SCHEMA",
    "V5_K1_BALANCED_FORMAL_CONFIGURATION_VERSION",
    "V5K1BalancedDatasetPlan",
    "V5K1BalancedSobolBlock",
    "build_frozen_v5_k1_balanced_dataset_plan",
    "build_v5_k1_balanced_dataset_plan",
    "is_frozen_v5_k1_balanced_dataset_plan",
    "main",
    "v5_k1_balanced_formal_configuration_payload",
    "v5_k1_balanced_dataset_authoring_payload",
    "v5_k1_balanced_dataset_plan_from_payload",
    "validate_v5_k1_balanced_dataset_plan",
    "write_v5_k1_balanced_dataset_authoring_plan",
]
