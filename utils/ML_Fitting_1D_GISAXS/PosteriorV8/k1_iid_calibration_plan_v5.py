"""Frozen IID/exchangeable K1 compatibility-calibration population.

Calibration is deliberately separate from every Sobol/RQMC population.  A
task owns one of the 60 reachable acquisition strata and draws independent K1
clean parents from a frozen branch-balanced mixture.  Rejection depends only
on the requested legal branch and acquisition metadata; it never inspects a
simulated curve or a model result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping

import numpy as np

from .compatibility_calibration import (
    DESIGN_STRATUM_UNIVERSE_SHA256,
    RESERVED_CALIBRATION_SPLIT_ID,
    CompatibilityStratum,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES, V5K1PhaseCBranch
from .observation_v5 import sample_v5_uncertainty_provenance
from .preprocessing import DEFAULT_CONTRACT
from .simulation import (
    MIN_GRID_POINTS,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE,
    OBSERVATION_POLICY_PHASE_PERIOD,
    OBSERVATION_STRATUM_VERSION,
    ObservationDesignStratumCoordinate,
    sample_observation_design_stratum,
    sample_observation_view,
)
from .synthetic_recipe_v5 import V5CleanRecipe, sample_v5_clean_recipe


V5_K1_IID_CALIBRATION_PLAN_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_compatibility_calibration_plan/v2"
)
V5_K1_IID_CALIBRATION_PLAN_VERSION = (
    "posterior_v8_v5_2_k1_all12_iid_exchangeable_60_strata_target160k_"
    "overflow_safe_sigma_v2"
)
V5_K1_IID_CALIBRATION_SAMPLING_DESIGN = (
    "independent_pseudorandom_clean_parent_draws_conditional_on_frozen_"
    "acquisition_stratum_not_sobol_not_qmc"
)
V5_K1_IID_CALIBRATION_MASTER_SEED = 14483825437891294498
V5_K1_IID_CALIBRATION_MASTER_SEED_DERIVATION_SHA256 = (
    "c900e27f4646f52228feac313658ab0480340c2fbfb94593be1e00576d556d68"
)
V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM = 2667
V5_K1_IID_CALIBRATION_TOTAL_SAMPLES = 160020
V5_K1_IID_CALIBRATION_MINIMUM_PER_STRATUM = 200
V5_K1_IID_CALIBRATION_TARGET_COVERAGE = 0.95
V5_K1_IID_CALIBRATION_MAX_REJECTION_ATTEMPTS = 256

_BRANCH_NAMESPACE = 0x4252414E
_RECIPE_NAMESPACE = 0x52454349
_UINT64_MAX = (1 << 64) - 1
_PLAN_FIELDS = {
    "schema",
    "version",
    "scientific_role",
    "sampling_design",
    "randomized_sobol_or_qmc_points_used",
    "calibration_split_id",
    "master_seed",
    "master_seed_derivation_sha256",
    "target_coverage",
    "minimum_samples_per_stratum",
    "samples_per_stratum",
    "total_samples",
    "branch_catalog",
    "branch_sampling",
    "observation_view_selection",
    "design_stratum_universe_sha256",
    "strata",
    "claim_limits",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _uint64_entropy(*values: int) -> int:
    words: list[int] = []
    for value in values:
        selected = _nonnegative_integer(value, "seed coordinate")
        if selected > _UINT64_MAX:
            raise ValueError("seed coordinates must fit in uint64")
        words.extend((selected & 0xFFFFFFFF, selected >> 32))
    state = np.random.SeedSequence(words).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def _point_options() -> tuple[int, ...]:
    maximum = int(DEFAULT_CONTRACT.max_points)
    values = tuple(
        sorted(
            {
                MIN_GRID_POINTS,
                max(MIN_GRID_POINTS, maximum // 4),
                max(MIN_GRID_POINTS, maximum // 2),
                maximum,
            }
        )
    )
    if len(values) != 4:
        raise RuntimeError("the calibration plan requires four acquisition point-count slots")
    return values


def compatibility_stratum_for_design(
    coordinate: ObservationDesignStratumCoordinate,
) -> CompatibilityStratum:
    if not isinstance(coordinate, ObservationDesignStratumCoordinate):
        raise TypeError("coordinate must be an ObservationDesignStratumCoordinate")
    return CompatibilityStratum(
        point_count=_point_options()[coordinate.design_point_count_cycle_slot],
        noise_id=f"{OBSERVATION_STRATUM_VERSION}:noise-{coordinate.noise_id}",
        q_window_id=f"{OBSERVATION_STRATUM_VERSION}:q-window-{coordinate.q_window_id}",
    )


@dataclass(frozen=True)
class V5K1IIDCalibrationRecipe:
    stratum_ordinal: int
    sample_ordinal: int
    rejection_attempt: int
    branch: V5K1PhaseCBranch
    recipe: V5CleanRecipe
    view_index: int
    clean_group_id: str
    sample_id: str


def _view_index_for_stratum(
    recipe_seed: int,
    coordinate: ObservationDesignStratumCoordinate,
) -> int:
    matches = tuple(
        index
        for index in range(OBSERVATION_POLICY_PHASE_PERIOD)
        if sample_observation_design_stratum(recipe_seed, index) == coordinate
    )
    if len(matches) != 1:
        raise RuntimeError("one policy period must map a recipe seed to every stratum exactly once")
    return matches[0]


def materialize_v5_k1_iid_calibration_recipe(
    *,
    stratum_ordinal: int,
    sample_ordinal: int,
    master_seed: int = V5_K1_IID_CALIBRATION_MASTER_SEED,
) -> V5K1IIDCalibrationRecipe:
    """Materialize one conditionally IID clean parent and sigma-present view."""

    stratum_index = _nonnegative_integer(stratum_ordinal, "stratum_ordinal")
    sample_index = _nonnegative_integer(sample_ordinal, "sample_ordinal")
    seed = _nonnegative_integer(master_seed, "master_seed")
    if stratum_index >= len(OBSERVATION_DESIGN_STRATUM_UNIVERSE):
        raise ValueError("stratum_ordinal is outside the frozen universe")
    if sample_index >= V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM:
        raise ValueError("sample_ordinal is outside the frozen stratum population")
    coordinate = OBSERVATION_DESIGN_STRATUM_UNIVERSE[stratum_index]
    branch_ordinal = _uint64_entropy(
        seed, stratum_index, sample_index, _BRANCH_NAMESPACE
    ) % len(K1_PHASE_C_BRANCHES)
    branch = K1_PHASE_C_BRANCHES[branch_ordinal]
    for attempt in range(V5_K1_IID_CALIBRATION_MAX_REJECTION_ATTEMPTS):
        recipe_seed = _uint64_entropy(
            seed,
            stratum_index,
            sample_index,
            attempt,
            _RECIPE_NAMESPACE,
        )
        view_index = _view_index_for_stratum(recipe_seed, coordinate)
        if not sample_v5_uncertainty_provenance(
            recipe_seed, view_index
        ).measurement_sigma_available:
            continue
        try:
            recipe = sample_v5_clean_recipe(
                (branch.shape,),
                recipe_seed=recipe_seed,
                amplitude_range_regime="full",
                pattern_id=branch.pattern_id,
            )
        except ValueError:
            continue
        selected = sample_observation_view(
            recipe_seed,
            view_index,
            max_points=DEFAULT_CONTRACT.max_points,
        )
        if (
            sample_observation_design_stratum(recipe_seed, view_index) != coordinate
            or selected.grid.n_points
            != compatibility_stratum_for_design(coordinate).point_count
        ):
            raise RuntimeError("accepted calibration view escaped its assigned stratum")
        identity_core = {
            "plan_version": V5_K1_IID_CALIBRATION_PLAN_VERSION,
            "master_seed": seed,
            "stratum_ordinal": stratum_index,
            "sample_ordinal": sample_index,
            "branch_id": branch.branch_id,
            "recipe_sha256": recipe.sha256,
            "view_index": view_index,
        }
        clean_group_id = sha256(
            _canonical_json({**identity_core, "kind": "clean_group"}).encode()
        ).hexdigest()
        sample_id = sha256(
            _canonical_json({**identity_core, "kind": "calibration_observation"}).encode()
        ).hexdigest()
        return V5K1IIDCalibrationRecipe(
            stratum_ordinal=stratum_index,
            sample_ordinal=sample_index,
            rejection_attempt=attempt,
            branch=branch,
            recipe=recipe,
            view_index=view_index,
            clean_group_id=clean_group_id,
            sample_id=sample_id,
        )
    raise RuntimeError("calibration recipe rejection budget was exhausted")


def build_v5_k1_iid_calibration_plan() -> dict[str, object]:
    if (
        len(OBSERVATION_DESIGN_STRATUM_UNIVERSE) != 60
        or V5_K1_IID_CALIBRATION_TOTAL_SAMPLES
        != 60 * V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM
    ):
        raise RuntimeError("formal calibration population arithmetic drifted")
    strata = [
        {
            "stratum_ordinal": index,
            "design_coordinate": asdict(coordinate),
            "compatibility_stratum": asdict(compatibility_stratum_for_design(coordinate)),
            "sample_count": V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
        }
        for index, coordinate in enumerate(OBSERVATION_DESIGN_STRATUM_UNIVERSE)
    ]
    core = {
        "schema": V5_K1_IID_CALIBRATION_PLAN_SCHEMA,
        "version": V5_K1_IID_CALIBRATION_PLAN_VERSION,
        "scientific_role": "measurement_compatibility_threshold_only_not_training_or_model_selection",
        "sampling_design": V5_K1_IID_CALIBRATION_SAMPLING_DESIGN,
        "randomized_sobol_or_qmc_points_used": False,
        "calibration_split_id": RESERVED_CALIBRATION_SPLIT_ID,
        "master_seed": V5_K1_IID_CALIBRATION_MASTER_SEED,
        "master_seed_derivation_sha256": (
            V5_K1_IID_CALIBRATION_MASTER_SEED_DERIVATION_SHA256
        ),
        "target_coverage": V5_K1_IID_CALIBRATION_TARGET_COVERAGE,
        "minimum_samples_per_stratum": V5_K1_IID_CALIBRATION_MINIMUM_PER_STRATUM,
        "samples_per_stratum": V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
        "total_samples": V5_K1_IID_CALIBRATION_TOTAL_SAMPLES,
        "branch_catalog": [value.audit_payload() for value in K1_PHASE_C_BRANCHES],
        "branch_sampling": "independent_discrete_uniform_over_12_legal_k1_branches",
        "observation_view_selection": (
            "unique_matching_view_in_indices_0_through_59_then_reject_if_measurement_sigma_missing"
        ),
        "design_stratum_universe_sha256": DESIGN_STRATUM_UNIVERSE_SHA256,
        "strata": strata,
        "claim_limits": {
            "training_authorization_granted": False,
            "model_selection_authorization_granted": False,
            "phase_c_acceptance_granted": False,
        },
    }
    return {**core, "plan_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


def validate_v5_k1_iid_calibration_plan(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("calibration plan must be an object")
    value = dict(payload)
    supplied = value.pop("plan_sha256", None)
    if supplied != sha256(_canonical_json(value).encode()).hexdigest():
        raise ValueError("calibration plan self-hash does not reproduce")
    if set(value) != _PLAN_FIELDS:
        raise ValueError("calibration plan fields are incomplete or unsupported")
    expected = build_v5_k1_iid_calibration_plan()
    if payload != expected:
        raise ValueError("calibration plan differs from the frozen paper population")
    return dict(payload)


__all__ = [
    "V5_K1_IID_CALIBRATION_MASTER_SEED",
    "V5_K1_IID_CALIBRATION_MINIMUM_PER_STRATUM",
    "V5_K1_IID_CALIBRATION_PLAN_SCHEMA",
    "V5_K1_IID_CALIBRATION_PLAN_VERSION",
    "V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM",
    "V5_K1_IID_CALIBRATION_TARGET_COVERAGE",
    "V5_K1_IID_CALIBRATION_TOTAL_SAMPLES",
    "V5K1IIDCalibrationRecipe",
    "build_v5_k1_iid_calibration_plan",
    "compatibility_stratum_for_design",
    "materialize_v5_k1_iid_calibration_recipe",
    "validate_v5_k1_iid_calibration_plan",
]
