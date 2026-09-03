"""Frozen, model-independent Sobol and optimizer schedules for V5.1 search.

The direct local-unit points are physical search starts, not neural samples.
Their exact bytes, order, SciPy runtime, and construction parameters are part
of the logical schedule digest.  The paper schedule always consumes the full
per-branch exact-forward budget, even after a compatible point is found.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy
from scipy.stats import qmc

from .branch_codec import BRANCH_CODEC_VERSION, UNIT_CUBE_DIMENSIONS
from .candidate_refinement_contract_v5 import V5_EXACT_REFINEMENT_VERSION
from .grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    V5ArtifactReceipt,
    array_manifest,
    array_sha256,
    canonical_json,
    read_checked_array_artifact,
    write_checked_array_artifact,
)
from .profiled_forward import PROFILED_AMPLITUDE_SOLVER_VERSION


V5_EXACT_SEARCH_SOBOL_SCHEMA = "gisaxs.posterior_v8.frozen_local_sobol_search/v1"
V5_EXACT_SEARCH_SOBOL_VERSION = (
    "posterior_v8_direct_scrambled_sobol_local_unit_fixed_order_v1"
)
V5_EXACT_SEARCH_SCHEDULE_ARTIFACT = (
    "gisaxs.posterior_v8.frozen_local_sobol_search_artifact/v1"
)
V5_EXACT_SEARCH_OPTIMIZER_SCHEMA = (
    "gisaxs.posterior_v8.frozen_exact_search_optimizer_schedule/v1"
)
V5_EXACT_SEARCH_OPTIMIZER_VERSION = (
    "posterior_v8_exact_metric_ranked_multistart_varying_dimension_budget_v2"
)
V5_EXACT_SEARCH_TERMINATION_POLICY_ID = (
    "paper_full_per_branch_budget_no_positive_early_stop_v1"
)
V5_EXACT_SEARCH_TERMINATION_REASONS = {
    "compatible_found": "frozen_full_budget_completed_with_compatible_representatives",
    "no_compatible_found_within_frozen_search_budget": (
        "exact_forward_budget_exhausted_without_compatible"
    ),
}
V5_EXACT_SEARCH_SEED_ORDER = (
    "direct_scout_prefix_then_exact_metric_ranked_scout_refinement_then_"
    "remaining_sobol_order_until_budget_exhaustion_v1"
)
V5_EXACT_SEARCH_RANKING = (
    "ascending_selected_exact_metric_then_ascending_sobol_seed_index_v1"
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _readonly_points(value: object) -> np.ndarray:
    points = np.array(value, dtype=np.float64, copy=True, order="C")
    if points.ndim != 2 or points.shape[1] != UNIT_CUBE_DIMENSIONS:
        raise ValueError(
            f"local Sobol points must have shape (N, {UNIT_CUBE_DIMENSIONS})"
        )
    if points.shape[0] < 1 or points.shape[0] & (points.shape[0] - 1):
        raise ValueError("local Sobol point count must be a positive power of two")
    if not np.all(np.isfinite(points)) or np.any(points < 0.0) or np.any(points >= 1.0):
        raise ValueError("local Sobol points must be finite and lie in [0, 1)")
    points.setflags(write=False)
    return points


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FrozenLocalSobolSchedule:
    """Exact ordered local-unit starts shared fairly by every branch."""

    schedule_id: str
    base_seed: int
    points: np.ndarray
    scipy_version: str
    engine: str = "scipy.stats.qmc.Sobol"
    scramble: bool = True
    ordering_policy: str = V5_EXACT_SEARCH_SEED_ORDER
    schema_version: str = V5_EXACT_SEARCH_SOBOL_SCHEMA
    version: str = V5_EXACT_SEARCH_SOBOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule_id", _text(self.schedule_id, "schedule_id"))
        object.__setattr__(self, "base_seed", _nonnegative_integer(self.base_seed, "base_seed"))
        object.__setattr__(self, "points", _readonly_points(self.points))
        object.__setattr__(
            self, "scipy_version", _text(self.scipy_version, "scipy_version")
        )
        if self.engine != "scipy.stats.qmc.Sobol":
            raise ValueError("unsupported local Sobol engine")
        if type(self.scramble) is not bool or not self.scramble:
            raise ValueError("the frozen search requires explicitly scrambled Sobol points")
        if self.ordering_policy != V5_EXACT_SEARCH_SEED_ORDER:
            raise ValueError("unsupported exact-search seed ordering policy")
        if self.schema_version != V5_EXACT_SEARCH_SOBOL_SCHEMA:
            raise ValueError("unsupported exact-search Sobol schema")
        if self.version != V5_EXACT_SEARCH_SOBOL_VERSION:
            raise ValueError("unsupported exact-search Sobol version")

    @classmethod
    def generate(
        cls,
        *,
        schedule_id: str,
        point_count: int,
        base_seed: int,
    ) -> "V5FrozenLocalSobolSchedule":
        count = _positive_integer(point_count, "point_count")
        if count & (count - 1):
            raise ValueError("point_count must be a power of two for random_base2")
        seed = _nonnegative_integer(base_seed, "base_seed")
        engine = qmc.Sobol(d=UNIT_CUBE_DIMENSIONS, scramble=True, seed=seed)
        points = engine.random_base2(int(np.log2(count)))
        return cls(
            schedule_id=schedule_id,
            base_seed=seed,
            points=points,
            scipy_version=scipy.__version__,
        )

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def points_sha256(self) -> str:
        return array_sha256("sobol_local_unit", self.points)

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "schedule_id": self.schedule_id,
            "engine": self.engine,
            "scipy_version": self.scipy_version,
            "dimension": UNIT_CUBE_DIMENSIONS,
            "point_count": self.point_count,
            "base_seed": self.base_seed,
            "scramble": self.scramble,
            "generation_method": "random_base2",
            "ordering_policy": self.ordering_policy,
            "points_sha256": self.points_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = canonical_json(self.audit_payload())
        return sha256(encoded.encode("utf-8")).hexdigest()

    def verify_runtime_replay(self) -> None:
        """Regenerate the points only under the exact recorded SciPy runtime."""

        if scipy.__version__ != self.scipy_version:
            raise RuntimeError(
                "cannot claim generated Sobol replay under a different SciPy version"
            )
        replay = type(self).generate(
            schedule_id=self.schedule_id,
            point_count=self.point_count,
            base_seed=self.base_seed,
        )
        if replay.sha256 != self.sha256 or not np.array_equal(replay.points, self.points):
            raise RuntimeError("generated Sobol schedule does not reproduce its frozen bytes")


@dataclass(frozen=True, kw_only=True)
class V5FrozenExactOptimizerSchedule:
    """Model-free allocation of one branch's exact-forward call budget."""

    schedule_id: str
    direct_scout_seed_count: int
    per_seed_forward_evaluation_limit: int
    ftol: float = 1.0e-8
    xtol: float = 1.0e-8
    gtol: float = 1.0e-8
    seed_ordering_policy: str = V5_EXACT_SEARCH_SEED_ORDER
    scout_ranking_policy: str = V5_EXACT_SEARCH_RANKING
    termination_policy_id: str = V5_EXACT_SEARCH_TERMINATION_POLICY_ID
    positive_early_stop: bool = False
    branch_codec_version: str = BRANCH_CODEC_VERSION
    exact_refinement_version: str = V5_EXACT_REFINEMENT_VERSION
    profiled_amplitude_solver_version: str = PROFILED_AMPLITUDE_SOLVER_VERSION
    schema_version: str = V5_EXACT_SEARCH_OPTIMIZER_SCHEMA
    version: str = V5_EXACT_SEARCH_OPTIMIZER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule_id", _text(self.schedule_id, "schedule_id"))
        object.__setattr__(
            self,
            "direct_scout_seed_count",
            _positive_integer(self.direct_scout_seed_count, "direct_scout_seed_count"),
        )
        object.__setattr__(
            self,
            "per_seed_forward_evaluation_limit",
            _positive_integer(
                self.per_seed_forward_evaluation_limit,
                "per_seed_forward_evaluation_limit",
            ),
        )
        for name in ("ftol", "xtol", "gtol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= np.finfo(float).eps:
                raise ValueError(f"{name} must be finite and greater than machine epsilon")
            object.__setattr__(self, name, value)
        if self.seed_ordering_policy != V5_EXACT_SEARCH_SEED_ORDER:
            raise ValueError("unsupported seed ordering policy")
        if self.scout_ranking_policy != V5_EXACT_SEARCH_RANKING:
            raise ValueError("unsupported exact scout ranking policy")
        if self.termination_policy_id != V5_EXACT_SEARCH_TERMINATION_POLICY_ID:
            raise ValueError("paper optimizer must use the full-budget termination policy")
        if type(self.positive_early_stop) is not bool or self.positive_early_stop:
            raise ValueError("paper exact search cannot stop after its first positive")
        if self.branch_codec_version != BRANCH_CODEC_VERSION:
            raise ValueError("optimizer schedule has a stale branch codec")
        if self.exact_refinement_version != V5_EXACT_REFINEMENT_VERSION:
            raise ValueError("optimizer schedule has a stale exact refiner")
        if self.profiled_amplitude_solver_version != PROFILED_AMPLITUDE_SOLVER_VERSION:
            raise ValueError("optimizer schedule has a stale amplitude solver")
        if self.schema_version != V5_EXACT_SEARCH_OPTIMIZER_SCHEMA:
            raise ValueError("unsupported exact-search optimizer schema")
        if self.version != V5_EXACT_SEARCH_OPTIMIZER_VERSION:
            raise ValueError("unsupported exact-search optimizer version")

    def audit_payload(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def sha256(self) -> str:
        encoded = canonical_json(self.audit_payload())
        return sha256(encoded.encode("utf-8")).hexdigest()


def write_v5_frozen_local_sobol_schedule(
    schedule: V5FrozenLocalSobolSchedule,
    path: str | Path,
) -> V5ArtifactReceipt:
    """Publish a standalone checked schedule without replacing an old file."""

    if not isinstance(schedule, V5FrozenLocalSobolSchedule):
        raise TypeError("schedule must be a V5FrozenLocalSobolSchedule")
    arrays = {"sobol_local_unit": schedule.points}
    core = {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "artifact_schema": V5_EXACT_SEARCH_SCHEDULE_ARTIFACT,
        "schedule": schedule.audit_payload(),
        "schedule_sha256": schedule.sha256,
        "arrays": array_manifest(arrays),
    }
    manifest = {**core, "manifest_sha256": sha256(canonical_json(core).encode()).hexdigest()}
    return write_checked_array_artifact(path, manifest=manifest, arrays=arrays)


def read_v5_frozen_local_sobol_schedule(
    path: str | Path,
) -> tuple[V5FrozenLocalSobolSchedule, V5ArtifactReceipt]:
    manifest, arrays, receipt = read_checked_array_artifact(path)
    if manifest.get("artifact_schema") != V5_EXACT_SEARCH_SCHEDULE_ARTIFACT:
        raise ValueError("unsupported frozen Sobol schedule artifact")
    payload = manifest.get("schedule")
    if not isinstance(payload, Mapping):
        raise ValueError("frozen Sobol schedule manifest is incomplete")
    expected = {
        "schema_version",
        "version",
        "schedule_id",
        "engine",
        "scipy_version",
        "dimension",
        "point_count",
        "base_seed",
        "scramble",
        "generation_method",
        "ordering_policy",
        "points_sha256",
    }
    if set(payload) != expected:
        raise ValueError("frozen Sobol schedule metadata is incomplete or unsupported")
    if payload["dimension"] != UNIT_CUBE_DIMENSIONS:
        raise ValueError("frozen Sobol schedule dimension is incompatible")
    if payload["generation_method"] != "random_base2":
        raise ValueError("unsupported frozen Sobol generation method")
    schedule = V5FrozenLocalSobolSchedule(
        schedule_id=payload["schedule_id"],
        base_seed=payload["base_seed"],
        points=arrays["sobol_local_unit"],
        scipy_version=payload["scipy_version"],
        engine=payload["engine"],
        scramble=payload["scramble"],
        ordering_policy=payload["ordering_policy"],
        schema_version=payload["schema_version"],
        version=payload["version"],
    )
    if schedule.audit_payload() != dict(payload) or schedule.sha256 != manifest.get(
        "schedule_sha256"
    ):
        raise ValueError("frozen Sobol schedule metadata or digest does not reproduce")
    return schedule, receipt


__all__ = [
    "V5_EXACT_SEARCH_OPTIMIZER_SCHEMA",
    "V5_EXACT_SEARCH_OPTIMIZER_VERSION",
    "V5_EXACT_SEARCH_RANKING",
    "V5_EXACT_SEARCH_SCHEDULE_ARTIFACT",
    "V5_EXACT_SEARCH_SEED_ORDER",
    "V5_EXACT_SEARCH_SOBOL_SCHEMA",
    "V5_EXACT_SEARCH_SOBOL_VERSION",
    "V5_EXACT_SEARCH_TERMINATION_POLICY_ID",
    "V5_EXACT_SEARCH_TERMINATION_REASONS",
    "V5FrozenExactOptimizerSchedule",
    "V5FrozenLocalSobolSchedule",
    "read_v5_frozen_local_sobol_schedule",
    "write_v5_frozen_local_sobol_schedule",
]
