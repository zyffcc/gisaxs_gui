"""Strict-budget nonlinear reduced-model searches for observability evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from numbers import Integral

import numpy as np

from .branch_catalog import branch_pattern_id
from .branch_codec import ResolutionBounds
from .component_observability import (
    ReducedModelSearchAttempt,
    ReducedModelSearchEvidence,
)
from .contract import GuiComponentBounds, topology_id_for
from .evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    CandidateInput,
    ObservedCurve,
    natural_log_rmse,
)
from .production_bridge import (
    ProductionBranchFactory,
    ProductionExactRefiner,
    ResolutionSearchPolicy,
    UserSearchSpace,
    physical_seed_from_external,
)
from .proposal_sampling import generate_profiled_branch_seed_at_index
from .reference_bank import CompetingBranch


REDUCED_MODEL_SEARCH_VERSION = "posterior_v8_counted_candidate_seed_plus_sobol_reduced_search_v1"


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _label_index(label: str, prefix: str) -> int:
    suffix = label.removeprefix(prefix)
    if not suffix.isdecimal():
        raise ValueError(f"{prefix.removesuffix('_')} label has an invalid index")
    return _integer(int(suffix), f"{prefix.removesuffix('_')} index", minimum=0)


def _primary_score(curve: ObservedCurve, candidate: CandidateInput, metric_name: str) -> float:
    if metric_name == RAW_LOG_RMSE_METRIC:
        return natural_log_rmse(candidate.exact_intensity, curve.intensity)
    if metric_name == STANDARDIZED_LOG_RMSE_METRIC:
        if curve.sigma_log is None:
            raise ValueError("standardized reduced-model scoring requires sigma_log")
        return natural_log_rmse(
            candidate.exact_intensity,
            curve.intensity,
            sigma_log=curve.sigma_log,
        )
    raise ValueError("unsupported reduced-model primary metric")


@dataclass(frozen=True, kw_only=True)
class BoundedReducedModelSearcher:
    """Search one deleted/toggled branch with candidate and Sobol starts.

    The search can prove only ``unneeded`` by finding a compatible reduced fit.
    Exhausting its finite starts is preserved as provisional evidence, never as
    a global non-existence certificate.
    """

    component_bounds: tuple[GuiComponentBounds, ...]
    resolution_bounds: ResolutionBounds | None
    seed: int = 0
    version: str = REDUCED_MODEL_SEARCH_VERSION

    def __post_init__(self) -> None:
        bounds = tuple(self.component_bounds)
        if not bounds or not all(isinstance(item, GuiComponentBounds) for item in bounds):
            raise ValueError("component_bounds must contain GuiComponentBounds")
        if self.resolution_bounds is not None and not isinstance(
            self.resolution_bounds, ResolutionBounds
        ):
            raise TypeError("resolution_bounds must be ResolutionBounds or None")
        seed = _integer(self.seed, "seed", minimum=0)
        if self.version != REDUCED_MODEL_SEARCH_VERSION:
            raise ValueError("unsupported reduced-model search version")
        object.__setattr__(self, "component_bounds", bounds)
        object.__setattr__(self, "seed", seed)

    def _reduced_problem(self, candidate: CandidateInput, label: str):
        if len(candidate.components) != len(self.component_bounds):
            raise ValueError("candidate and reduced-search bounds disagree")
        components = list(candidate.components)
        bounds = list(self.component_bounds)
        resolution = candidate.resolution
        if label.startswith("particle_"):
            index = _label_index(label, "particle_")
            if index >= len(components):
                raise ValueError("particle deletion index is out of range")
            del components[index]
            del bounds[index]
            if not components:
                raise ValueError("K0 is handled by the exhaustive linear null")
        elif label == "resolution":
            if resolution is None:
                raise ValueError("resolution deletion requested for an absent term")
            resolution = None
        elif label.startswith("d_"):
            index = _label_index(label, "d_")
            if index >= len(components) or components[index].log_D is None:
                raise ValueError("D deletion index is absent or out of range")
            components[index] = replace(components[index], log_D=None, sigma_D_fraction=None)
            bounds[index] = replace(bounds[index], allow_D_absent=True)
        else:
            raise ValueError("unsupported reduced-model feature label")

        shapes = tuple(item.shape for item in components)
        if shapes != tuple(item.shape for item in bounds):
            raise ValueError("reduced components and bounds have different topologies")
        topology_id = topology_id_for(shapes)
        d_present = tuple(item.log_D is not None for item in components)
        pattern_id = branch_pattern_id(
            d_present + (False,) * (4 - len(d_present)),
            resolution is not None,
        )
        if resolution is None:
            resolution_policy = ResolutionSearchPolicy(presence="absent")
        else:
            if self.resolution_bounds is None:
                raise ValueError("Resolution-present reduced search requires bounds")
            resolution_policy = ResolutionSearchPolicy(
                presence="required", bounds=self.resolution_bounds
            )
        factory = ProductionBranchFactory(
            UserSearchSpace.for_components(bounds, resolution=resolution_policy)
        )
        context = factory.context_for(
            CompetingBranch(topology_id=topology_id, pattern_id=pattern_id)
        )
        if context is None:
            raise ValueError("reduced branch has no feasible bounded context")
        return context, tuple(components), resolution

    def _stable_seed(self, candidate_id: str, label: str) -> int:
        digest = hashlib.sha256(f"{self.seed}\0{candidate_id}\0{label}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**31)

    def search(
        self,
        curve: ObservedCurve,
        candidate: CandidateInput,
        *,
        label: str,
        primary_metric_name: str,
        primary_compatibility_threshold: float,
        max_forward_evaluations: int,
        required_starts: int,
        per_start_forward_evaluation_limit: int,
    ) -> ReducedModelSearchEvidence:
        total_limit = _integer(max_forward_evaluations, "max_forward_evaluations", minimum=0)
        starts = _integer(required_starts, "required_starts", minimum=1)
        per_start = _integer(
            per_start_forward_evaluation_limit,
            "per_start_forward_evaluation_limit",
            minimum=1,
        )
        threshold = float(primary_compatibility_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("primary_compatibility_threshold is invalid")
        if total_limit == 0:
            return ReducedModelSearchEvidence(
                label=label,
                status="budget_exhausted",
                best_primary_score=None,
                primary_compatibility_threshold=threshold,
                forward_calls=0,
                attempted_starts=0,
                required_starts=starts,
                search_version=self.version,
                attempts=(),
                detail="no exact-forward budget remained for reduced-model search",
            )
        try:
            context, candidate_seed, resolution_seed = self._reduced_problem(candidate, label)
            first = physical_seed_from_external(
                source="retrieval",
                source_id=f"{candidate.candidate_id}:{label}:candidate_seed",
                context=context,
                components=candidate_seed,
                resolution=resolution_seed,
            )
        except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            return ReducedModelSearchEvidence(
                label=label,
                status="search_unavailable",
                best_primary_score=None,
                primary_compatibility_threshold=threshold,
                forward_calls=0,
                attempted_starts=0,
                required_starts=starts,
                search_version=self.version,
                attempts=(),
                detail=f"{type(exc).__name__}: {exc}",
            )

        def seeds():
            yield first
            sobol_seed = self._stable_seed(candidate.candidate_id, label)
            for sequence_index in range(starts - 1):
                generated = generate_profiled_branch_seed_at_index(
                    context.user_bounds_codec,
                    seed=sobol_seed,
                    sequence_index=sequence_index,
                )
                yield physical_seed_from_external(
                    source="sobol",
                    source_id=(f"{candidate.candidate_id}:{label}:sobol_{sequence_index:04d}"),
                    context=context,
                    components=generated.seed_components,
                    resolution=generated.resolution_seed,
                )

        refiner = ProductionExactRefiner(curve=curve)
        used = 0
        best: float | None = None
        attempts: list[ReducedModelSearchAttempt] = []
        incomplete = False
        iterator = iter(seeds())
        for start_index in range(starts):
            remaining = total_limit - used
            if remaining <= 0:
                incomplete = True
                break
            try:
                physical = next(iterator)
            except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
                attempts.append(
                    ReducedModelSearchAttempt(
                        start_index=start_index,
                        source="generation",
                        status="generation_failed",
                        forward_calls=0,
                        primary_score=None,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                incomplete = True
                continue
            allowance = min(remaining, per_start)
            outcome = refiner.refine_physical(
                physical,
                candidate_id=f"{candidate.candidate_id}:{label}:{start_index:03d}",
                proposal_rank=1,
                max_forward_evaluations=allowance,
            )
            if outcome.forward_evaluations > allowance:
                raise RuntimeError("reduced refiner exceeded its exact-forward allowance")
            used += outcome.forward_evaluations
            if outcome.status == "failed":
                attempts.append(
                    ReducedModelSearchAttempt(
                        start_index=start_index,
                        source=physical.source,
                        status="refinement_failed",
                        forward_calls=outcome.forward_evaluations,
                        primary_score=None,
                        detail=outcome.message,
                    )
                )
                incomplete = True
                continue
            if not isinstance(outcome.value, CandidateInput):
                raise TypeError("reduced refiner success did not return CandidateInput")
            score = _primary_score(curve, outcome.value, primary_metric_name)
            best = score if best is None else min(best, score)
            compatible = score <= threshold
            attempts.append(
                ReducedModelSearchAttempt(
                    start_index=start_index,
                    source=physical.source,
                    status=("compatible" if compatible else "incompatible"),
                    forward_calls=outcome.forward_evaluations,
                    primary_score=score,
                    detail=outcome.message,
                )
            )
            if compatible:
                return ReducedModelSearchEvidence(
                    label=label,
                    status="compatible_reduced_model_found",
                    best_primary_score=best,
                    primary_compatibility_threshold=threshold,
                    forward_calls=used,
                    attempted_starts=len(attempts),
                    required_starts=starts,
                    search_version=self.version,
                    attempts=tuple(attempts),
                    detail="compatible reduced fit proves the deleted term is unneeded",
                )

        if len(attempts) < starts:
            status = "budget_exhausted" if used >= total_limit else "search_unavailable"
        elif incomplete:
            status = "search_unavailable"
        else:
            status = "completed_no_compatible_reduced_model"
        return ReducedModelSearchEvidence(
            label=label,
            status=status,
            best_primary_score=best,
            primary_compatibility_threshold=threshold,
            forward_calls=used,
            attempted_starts=len(attempts),
            required_starts=starts,
            search_version=self.version,
            attempts=tuple(attempts),
            detail=(
                "finite search found no compatible reduced fit; this is not a global certificate"
                if status == "completed_no_compatible_reduced_model"
                else "reduced-model search did not complete its declared starts"
            ),
        )


__all__ = [
    "REDUCED_MODEL_SEARCH_VERSION",
    "BoundedReducedModelSearcher",
]
