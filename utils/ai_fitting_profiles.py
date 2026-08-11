"""Configuration-only fitting profiles shared by GUI and in-situ workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class FittingProfile:
    """Controls search cost without changing the fitting pipeline itself."""

    name: str
    candidate_count: int
    multi_start_count: int
    refinement_count: int
    max_evaluations: int
    tolerance: float
    q_stride: int
    sampling_std: float
    progress_interval: int
    uncertainty_samples: int
    compare_full_candidates: bool
    time_budget_seconds: float | None
    random_seed: int = 123
    top_k: int = 20
    target_log_rmse: float = 0.08
    stall_patience: int = 80
    stall_tolerance: float = 1e-4
    complexity_penalty: float = 1.0
    parameter_mode_radius: float = 0.10
    sampling_scales: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.refinement_count < 0 or self.multi_start_count < 0:
            raise ValueError("refinement and multi-start counts must be >= 0")
        if self.max_evaluations < 1 or self.q_stride < 1:
            raise ValueError("max_evaluations and q_stride must be >= 1")
        if self.tolerance < 0 or self.sampling_std <= 0:
            raise ValueError("tolerance must be >= 0 and sampling_std must be > 0")
        if not self.sampling_scales or any(scale <= 0 for scale in self.sampling_scales):
            raise ValueError("sampling_scales must contain positive values")
        if self.time_budget_seconds is not None and self.time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive or None")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "FittingProfile":
        updates.pop("name", None)
        return replace(self, name="Custom", **updates)


PROFILE_DEFAULTS: Dict[str, FittingProfile] = {
    "Fast": FittingProfile(
        name="Fast",
        candidate_count=48,
        multi_start_count=1,
        refinement_count=0,
        max_evaluations=24,
        tolerance=1e-5,
        q_stride=8,
        sampling_std=0.008,
        progress_interval=16,
        uncertainty_samples=0,
        compare_full_candidates=False,
        time_budget_seconds=30.0,
        stall_patience=24,
    ),
    "Balanced": FittingProfile(
        name="Balanced",
        candidate_count=192,
        multi_start_count=2,
        refinement_count=2,
        max_evaluations=40,
        tolerance=1e-8,
        q_stride=4,
        sampling_std=0.005,
        progress_interval=64,
        uncertainty_samples=32,
        compare_full_candidates=False,
        time_budget_seconds=180.0,
        stall_patience=40,
        sampling_scales=(0.5, 1.0, 2.0),
    ),
    "Exhaustive": FittingProfile(
        name="Exhaustive",
        candidate_count=512,
        multi_start_count=8,
        refinement_count=6,
        max_evaluations=120,
        tolerance=1e-10,
        q_stride=1,
        sampling_std=0.01,
        progress_interval=100,
        uncertainty_samples=256,
        compare_full_candidates=True,
        time_budget_seconds=None,
        stall_patience=240,
        sampling_scales=(0.5, 1.0, 2.0, 4.0),
    ),
}

DEFAULT_PROFILE_NAME = "Balanced"


class FittingProfileRegistry:
    def __init__(self, profiles: Iterable[FittingProfile] | None = None) -> None:
        source = profiles if profiles is not None else PROFILE_DEFAULTS.values()
        self._profiles = {profile.name: profile for profile in source}
        if DEFAULT_PROFILE_NAME not in self._profiles:
            raise ValueError(f"Profile registry must contain {DEFAULT_PROFILE_NAME!r}")

    def names(self) -> tuple[str, ...]:
        return tuple(self._profiles)

    def get(self, name: str | None = None) -> FittingProfile:
        key = name or DEFAULT_PROFILE_NAME
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise KeyError(f"Unknown fitting profile {key!r}; choose from {', '.join(self.names())}") from exc

    def restore(self, name: str) -> FittingProfile:
        return self.get(name)


profile_registry = FittingProfileRegistry()
