from __future__ import annotations

# ruff: noqa: E402 -- skip cleanly when the optional TensorFlow runtime is absent.

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    grouped_training_inventory_v5 as inventory,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    SEARCH_OUTCOME_CODE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_data_v5 import (
    V5GroupedTrainingConfig,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_inventory_v5 import (
    V5GroupedShard,
    audit_sidecar_training_expansion,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_contract_v5 import (
    V5CompatibleRepresentativeReference,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.search_supervision_sidecar_v5 import (
    branch_array,
    branch_label,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_grouped_v5 import _config, build_parser


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _representatives(count: int, identity: str) -> str:
    return json.dumps(
        [
            V5CompatibleRepresentativeReference(
                artifact_id=f"artifact/{identity}/{index}",
                artifact_sha256=_digest(f"artifact/{identity}/{index}"),
                representative_set_id=f"set/{identity}",
                representative_set_sha256=_digest(f"set/{identity}"),
                cluster_id=f"cluster-{index:04d}",
                metric_value=0.01,
                bounds_passed=True,
                physics_passed=True,
                target_local=(0.5,) * 26,
            ).audit_payload()
            for index in range(count)
        ],
        sort_keys=True,
    )


def _shard(
    role: str,
    *,
    recipe_outcomes: tuple[tuple[tuple[str, int], ...], ...],
    offset: int,
) -> V5GroupedShard:
    recipes: list[int] = []
    outcomes: list[int] = []
    references: list[str] = []
    for recipe_index, rows in enumerate(recipe_outcomes):
        for row_index, (outcome, representative_count) in enumerate(rows):
            recipes.append(recipe_index)
            outcomes.append(SEARCH_OUTCOME_CODE[outcome])
            references.append(
                _representatives(
                    representative_count,
                    f"{role}-{recipe_index}-{row_index}",
                )
            )
    arrays = {
        branch_label("clean_recipe_index"): np.asarray(recipes, dtype=np.int32),
        branch_label("search_outcome_code"): np.asarray(outcomes, dtype=np.int8),
        branch_array("compatible_representatives_json"): np.asarray(references),
    }
    overlay = SimpleNamespace(sidecar=SimpleNamespace(arrays=arrays))
    return V5GroupedShard(
        path=Path(f"/{role}.gvd5"),
        dataset=SimpleNamespace(recipe_count=len(recipe_outcomes)),
        artifact_sha256="a" * 64,
        manifest_sha256="b" * 64,
        role=role,
        recipe_offset=offset,
        full_overlay=overlay,
    )


def _shards() -> tuple[V5GroupedShard, ...]:
    positive = "compatible_found"
    negative = "no_compatible_found_within_frozen_search_budget"
    return (
        _shard(
            "train",
            recipe_outcomes=(
                ((positive, 2), (negative, 0)),
                ((positive, 1), (negative, 0), (negative, 0)),
            ),
            offset=0,
        ),
        _shard(
            "validation",
            recipe_outcomes=(((positive, 1), (negative, 0)),),
            offset=2,
        ),
    )


def test_expansion_audit_counts_actual_targets_and_projects_recipe_batches():
    assert inventory.V5_PAPER_GREEN_EXPANDED_ROWS == 4_096
    assert inventory.V5_PAPER_GREEN_POSITIVE_NEGATIVE_PAIRS == 1_000_000
    assert inventory.V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA == 8_192
    assert inventory.V5_VALIDATION_MAX_EXPANDED_ROWS_PER_BATCH == 12_000
    assert inventory.V5_MAX_POSITIVE_NEGATIVE_PAIRS_PER_BATCH == 4_000_000
    audit = audit_sidecar_training_expansion(
        _shards(),
        train_recipes_per_replica=2,
        validation_recipes_per_batch=1,
        allow_unsafe_for_engineering=False,
    )

    assert audit is not None
    assert audit["overall_status"] == "green"
    assert audit["paper_claim_allowed"] is True
    assert audit["requires_maxwell_one_step_smoke"] is False
    train = audit["roles"]["train"]
    assert train["per_recipe_distributions"]["expanded_rows"] == {
        "min": 3,
        "median": 3.0,
        "p95": 3,
        "max": 3,
        "sum": 6,
    }
    assert train["per_recipe_distributions"]["positive_rows"]["sum"] == 3
    assert train["per_recipe_distributions"]["negative_rows"]["sum"] == 3
    assert train["sum_positive_negative_pairs"] == 4
    assert train["batch_projection"]["expanded_rows_B"] == 6
    assert train["batch_projection"]["positive_negative_pairs"] == 4
    assert train["batch_projection"]["dense_pair_mask_elements_B_squared"] == 36
    assert len(train["per_recipe_counts_sha256"]) == 64


def test_yellow_gate_requires_one_step_maxwell_smoke(monkeypatch):
    monkeypatch.setattr(inventory, "V5_PAPER_GREEN_EXPANDED_ROWS", 2)
    monkeypatch.setattr(inventory, "V5_PAPER_GREEN_POSITIVE_NEGATIVE_PAIRS", 1)

    with pytest.warns(RuntimeWarning, match="Maxwell one-step"):
        audit = audit_sidecar_training_expansion(
            _shards(),
            train_recipes_per_replica=1,
            validation_recipes_per_batch=1,
            allow_unsafe_for_engineering=False,
        )

    assert audit is not None
    assert audit["overall_status"] == "yellow"
    assert audit["paper_claim_allowed"] is False
    assert audit["paper_receipt_eligible_after_required_action"] is True
    assert audit["requires_maxwell_one_step_smoke"] is True
    assert audit["paper_long_run_readiness"] == (
        "maxwell_one_step_peak_memory_smoke_required"
    )
    assert "Maxwell one-step" in audit["warning"]


@pytest.mark.parametrize(
    ("limit_name", "expected_role"),
    (
        ("V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA", "train"),
        ("V5_VALIDATION_MAX_EXPANDED_ROWS_PER_BATCH", "validation"),
        ("V5_MAX_POSITIVE_NEGATIVE_PAIRS_PER_BATCH", "train"),
    ),
)
def test_red_gate_fails_closed_for_train_validation_and_pairs(
    monkeypatch, limit_name, expected_role
):
    monkeypatch.setattr(inventory, limit_name, 0)

    with pytest.raises(MemoryError, match=expected_role):
        audit_sidecar_training_expansion(
            _shards(),
            train_recipes_per_replica=1,
            validation_recipes_per_batch=1,
            allow_unsafe_for_engineering=False,
        )


def test_explicit_unsafe_override_is_never_a_paper_receipt(monkeypatch):
    monkeypatch.setattr(inventory, "V5_TRAIN_MAX_EXPANDED_ROWS_PER_REPLICA", 0)

    with pytest.warns(RuntimeWarning, match="engineering override"):
        audit = audit_sidecar_training_expansion(
            _shards(),
            train_recipes_per_replica=1,
            validation_recipes_per_batch=1,
            allow_unsafe_for_engineering=True,
        )

    assert audit is not None
    assert audit["overall_status"] == "unsafe_engineering_override"
    assert audit["hard_gate_passed"] is False
    assert audit["formal_gate_bypassed"] is True
    assert audit["paper_claim_allowed"] is False
    assert audit["engineering_override_requested"] is True
    assert audit["paper_long_run_readiness"] == (
        "engineering_only_not_paper_claim_eligible"
    )


def test_engineering_override_is_explicit_in_config_and_cli():
    default = V5GroupedTrainingConfig(warmup_epochs=1)
    assert default.allow_unsafe_sidecar_expansion_for_engineering is False
    with pytest.raises(TypeError, match="must be boolean"):
        V5GroupedTrainingConfig(
            warmup_epochs=1,
            allow_unsafe_sidecar_expansion_for_engineering=1,
        )

    parsed = build_parser().parse_args(
        [
            "--train-dataset",
            "train.gvd5",
            "--validation-dataset",
            "validation.gvd5",
            "--output-dir",
            "output",
            "--allow-unsafe-sidecar-expansion-for-engineering",
        ]
    )
    assert parsed.allow_unsafe_sidecar_expansion_for_engineering is True
    assert _config(parsed).allow_unsafe_sidecar_expansion_for_engineering is True
