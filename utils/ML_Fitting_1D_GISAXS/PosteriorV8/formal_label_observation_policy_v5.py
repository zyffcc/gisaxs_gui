"""Curve-blind observation selection for formal V5 exact-search labels.

Formal compatibility labels require measurement uncertainty.  A clean recipe
may have several deterministic observation views, so this module selects one
view *before* any curve is evaluated.  Selection delegates uncertainty-state
assignment to the existing V5 policy and fails closed unless exactly one
candidate exposes measurement sigma.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral

import numpy as np

from .grouped_artifact_v5 import canonical_json
from .observation_v5 import (
    V5_UNCERTAINTY_VIEW_POLICY_VERSION,
    sample_v5_uncertainty_provenance,
)
from .uncertainty_provenance_v5 import (
    V5_UNCERTAINTY_SCHEMA,
    V5_UNCERTAINTY_VERSION,
)


V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA = (
    "gisaxs.posterior_v8.formal_label_observation_selection/v1"
)
V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION = (
    "posterior_v8_curve_blind_exactly_one_measurement_sigma_view_v1"
)
V5_FORMAL_LABEL_OBSERVATION_POLICY_ID = (
    "posterior-v8/formal-label-observation/seeded-measurement-sigma-exactly-one/v1"
)
V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES = (0, 1)
V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT = 64

_UINT64_MAX = (1 << 64) - 1


def _uint64(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT64_MAX:
        raise ValueError(f"{name} must fit in uint64")
    return result


def _candidate_pool(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("candidate_view_indices must be an ordered finite sequence")
    count = len(value)
    if count == 0:
        raise ValueError("candidate_view_indices cannot be empty")
    if count > V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT:
        raise ValueError(
            "candidate_view_indices exceeds the formal policy safety limit of "
            f"{V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT}"
        )
    result = tuple(_uint64(item, "candidate view index") for item in value)
    if len(result) != len(set(result)):
        raise ValueError("candidate_view_indices must be unique")
    return result


def formal_label_observation_policy_payload() -> dict[str, object]:
    """Return the immutable, curve-blind selector definition."""

    return {
        "schema": V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA,
        "version": V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION,
        "policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
        "selection_rule": "uncertainty.measurement_sigma_available_is_true",
        "required_selection_count": 1,
        "uncertainty_assignment_function": (
            "observation_v5.sample_v5_uncertainty_provenance"
        ),
        "uncertainty_view_policy_version": V5_UNCERTAINTY_VIEW_POLICY_VERSION,
        "uncertainty_schema": V5_UNCERTAINTY_SCHEMA,
        "uncertainty_version": V5_UNCERTAINTY_VERSION,
        "decision_inputs": ["recipe_seed", "candidate_view_indices"],
        "physical_curve_is_decision_input": False,
        "candidate_pool_contract": {
            "caller_supplied": True,
            "ordered": True,
            "unique": True,
            "uint64_indices": True,
            "maximum_count": V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT,
            "default_candidate_view_indices": list(
                V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES
            ),
        },
        "failure_policy": "reject_unless_exactly_one_measurement_sigma_candidate",
    }


V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256 = sha256(
    canonical_json(formal_label_observation_policy_payload()).encode("utf-8")
).hexdigest()


def _selection_payload(
    recipe_seed: int,
    candidate_view_indices: tuple[int, ...],
    selected_view_index: int,
) -> dict[str, object]:
    candidate_rows = []
    for view_index in candidate_view_indices:
        uncertainty = sample_v5_uncertainty_provenance(recipe_seed, view_index)
        candidate_rows.append(
            {
                "view_index": view_index,
                "uncertainty_kind": uncertainty.kind,
                "measurement_sigma_available": uncertainty.measurement_sigma_available,
            }
        )
    return {
        "schema": V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA,
        "version": V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION,
        "policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
        "policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
        "recipe_seed": recipe_seed,
        "candidate_view_indices": list(candidate_view_indices),
        "candidate_uncertainty": candidate_rows,
        "selected_view_index": selected_view_index,
        "selection_count": sum(
            bool(row["measurement_sigma_available"]) for row in candidate_rows
        ),
        "decision_was_curve_blind": True,
    }


def _selected_index(recipe_seed: int, candidate_view_indices: tuple[int, ...]) -> int:
    selected = tuple(
        view_index
        for view_index in candidate_view_indices
        if sample_v5_uncertainty_provenance(
            recipe_seed, view_index
        ).measurement_sigma_available
    )
    if len(selected) != 1:
        raise ValueError(
            "formal label observation policy requires exactly one "
            "measurement-sigma-present candidate; "
            f"found {len(selected)} in {candidate_view_indices}"
        )
    return selected[0]


@dataclass(frozen=True, kw_only=True)
class V5FormalLabelObservationSelection:
    """One replayable, curve-blind selection and its canonical audit record."""

    recipe_seed: int
    candidate_view_indices: tuple[int, ...]
    selected_view_index: int
    audit_json: str
    audit_sha256: str
    schema_version: str = V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA
    version: str = V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION
    policy_id: str = V5_FORMAL_LABEL_OBSERVATION_POLICY_ID
    policy_sha256: str = V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256

    def __post_init__(self) -> None:
        if self.schema_version != V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA:
            raise ValueError("unsupported formal label observation-selection schema")
        if self.version != V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION:
            raise ValueError("unsupported formal label observation-selection version")
        if self.policy_id != V5_FORMAL_LABEL_OBSERVATION_POLICY_ID:
            raise ValueError("unsupported formal label observation-selection policy ID")
        if self.policy_sha256 != V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256:
            raise ValueError("formal label observation-selection policy SHA-256 changed")

        seed = _uint64(self.recipe_seed, "recipe_seed")
        pool = _candidate_pool(self.candidate_view_indices)
        selected = _selected_index(seed, pool)
        supplied_selected = _uint64(self.selected_view_index, "selected_view_index")
        if supplied_selected != selected:
            raise ValueError(
                "selected_view_index does not reproduce the formal observation policy"
            )
        payload = _selection_payload(seed, pool, selected)
        encoded = canonical_json(payload)
        digest = sha256(encoded.encode("utf-8")).hexdigest()
        if self.audit_json != encoded or self.audit_sha256 != digest:
            raise ValueError("formal label observation-selection audit does not reproduce")

        object.__setattr__(self, "recipe_seed", seed)
        object.__setattr__(self, "candidate_view_indices", pool)
        object.__setattr__(self, "selected_view_index", selected)

    @classmethod
    def create(
        cls,
        recipe_seed: int,
        candidate_view_indices: Sequence[int] = (
            V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES
        ),
    ) -> "V5FormalLabelObservationSelection":
        seed = _uint64(recipe_seed, "recipe_seed")
        pool = _candidate_pool(candidate_view_indices)
        selected = _selected_index(seed, pool)
        audit_json = canonical_json(_selection_payload(seed, pool, selected))
        return cls(
            recipe_seed=seed,
            candidate_view_indices=pool,
            selected_view_index=selected,
            audit_json=audit_json,
            audit_sha256=sha256(audit_json.encode("utf-8")).hexdigest(),
        )

    def audit_payload(self) -> dict[str, object]:
        """Return a fresh JSON-compatible copy of the authenticated audit."""

        return _selection_payload(
            self.recipe_seed,
            self.candidate_view_indices,
            self.selected_view_index,
        )


def select_v5_formal_label_observation(
    recipe_seed: int,
    candidate_view_indices: Sequence[int] = V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES,
) -> V5FormalLabelObservationSelection:
    """Select exactly one sigma-present view without accepting curve data."""

    return V5FormalLabelObservationSelection.create(recipe_seed, candidate_view_indices)


__all__ = [
    "V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES",
    "V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT",
    "V5_FORMAL_LABEL_OBSERVATION_POLICY_ID",
    "V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA",
    "V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256",
    "V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION",
    "V5FormalLabelObservationSelection",
    "formal_label_observation_policy_payload",
    "select_v5_formal_label_observation",
]
