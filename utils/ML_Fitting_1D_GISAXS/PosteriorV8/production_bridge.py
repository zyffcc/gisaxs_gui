"""Production-safe bounds and exact-refinement bridge for V8 proposals."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Literal, Sequence

import numpy as np

from .branch_catalog import BRANCH_PATTERN_COUNT, branch_pattern_is_valid
from .branch_codec import (
    INACTIVE_UNIT_VALUE,
    BranchCoordinates,
    ProfiledBranchCodec,
    ResolutionBounds,
)
from .contract import (
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    GuiComponentBounds,
    LatentComponentParameters,
    canonical_topology,
    full_component_bounds,
    gui_bounds_to_latent,
    latent_component_to_gui,
    topology_id_for,
)
from .evaluation import CandidateInput, LinearSolutionSnapshot, ObservedCurve
from .inference_proposals import (
    BoundedProposalSample,
    BranchCondition,
    JointBranchScore,
    full_range_condition,
)
from .one_click_inference import RefinementOutcome
from .profiled_forward import (
    ProfiledForwardResult,
    ResolutionShape,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)
from .profiled_refinement import refine_profiled_branch
from .reference_bank import CompetingBranch


PRODUCTION_BRIDGE_VERSION = "posterior_v8_global_decode_user_validate_refine_v1"
SeedSource = Literal["neural", "retrieval", "sobol"]
ResolutionPresence = Literal["absent", "optional", "required"]
_CANDIDATE_ERRORS = (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, kw_only=True)
class ResolutionSearchPolicy:
    presence: ResolutionPresence
    bounds: ResolutionBounds | None = None

    def __post_init__(self) -> None:
        if self.presence not in {"absent", "optional", "required"}:
            raise ValueError("resolution presence must be absent, optional, or required")
        if self.presence == "absent" and self.bounds is not None:
            raise ValueError("resolution-absent policy must not define bounds")
        if self.presence != "absent" and not isinstance(self.bounds, ResolutionBounds):
            raise TypeError("optional/required resolution policy needs ResolutionBounds")

    def permits(self, present: bool) -> bool:
        if type(present) is not bool:
            raise TypeError("present must be boolean")
        return self.presence == "optional" or present == (self.presence == "required")


@dataclass(frozen=True, kw_only=True)
class TopologyUserBounds:
    component_bounds: tuple[GuiComponentBounds, ...]
    topology_id: int = field(init=False)

    def __post_init__(self) -> None:
        bounds = tuple(self.component_bounds)
        if not 1 <= len(bounds) <= 4 or not all(
            isinstance(item, GuiComponentBounds) for item in bounds
        ):
            raise ValueError("component_bounds must contain one to four GuiComponentBounds")
        shapes = tuple(item.shape for item in bounds)
        if shapes != canonical_topology(shapes):
            raise ValueError("component bounds must use canonical topology order")
        object.__setattr__(self, "component_bounds", bounds)
        object.__setattr__(self, "topology_id", topology_id_for(shapes))


@dataclass(frozen=True, kw_only=True)
class UserSearchSpace:
    topologies: tuple[TopologyUserBounds, ...]
    resolution: ResolutionSearchPolicy = ResolutionSearchPolicy(presence="absent")

    def __post_init__(self) -> None:
        topologies = tuple(self.topologies)
        if not topologies or not all(isinstance(item, TopologyUserBounds) for item in topologies):
            raise ValueError("topologies must contain at least one TopologyUserBounds")
        identifiers = tuple(item.topology_id for item in topologies)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("user search-space topology IDs must be unique")
        if not isinstance(self.resolution, ResolutionSearchPolicy):
            raise TypeError("resolution must be a ResolutionSearchPolicy")
        object.__setattr__(
            self, "topologies", tuple(sorted(topologies, key=lambda item: item.topology_id))
        )

    @classmethod
    def for_components(
        cls,
        component_bounds: Sequence[GuiComponentBounds],
        *,
        resolution: ResolutionSearchPolicy | None = None,
    ) -> "UserSearchSpace":
        return cls(
            topologies=(TopologyUserBounds(component_bounds=tuple(component_bounds)),),
            resolution=(
                ResolutionSearchPolicy(presence="absent") if resolution is None else resolution
            ),
        )

    def bounds_for(self, topology_id: int) -> tuple[GuiComponentBounds, ...] | None:
        for topology in self.topologies:
            if topology.topology_id == topology_id:
                return topology.component_bounds
        return None


@dataclass(frozen=True, kw_only=True)
class BranchRuntimeContext:
    branch: CompetingBranch
    full_domain_codec: ProfiledBranchCodec
    user_bounds_codec: ProfiledBranchCodec

    def __post_init__(self) -> None:
        if not isinstance(self.branch, CompetingBranch):
            raise TypeError("branch must be a CompetingBranch")
        for name in ("full_domain_codec", "user_bounds_codec"):
            codec = getattr(self, name)
            if not isinstance(codec, ProfiledBranchCodec):
                raise TypeError(f"{name} must be a ProfiledBranchCodec")
            if (
                codec.topology_id != self.branch.topology_id
                or codec.d_present != self.branch.d_present
                or (codec.resolution_bounds is not None) != self.branch.resolution_present
            ):
                raise ValueError(f"{name} does not match the hard branch")

    @property
    def component_bounds(self) -> tuple[GuiComponentBounds, ...]:
        return self.user_bounds_codec.component_bounds

    @property
    def resolution_bounds(self) -> ResolutionBounds | None:
        return self.user_bounds_codec.resolution_bounds


class ProductionBranchFactory:
    """Filter joint branches and attach full-domain plus user-bound codecs."""

    def __init__(self, search_space: UserSearchSpace):
        if not isinstance(search_space, UserSearchSpace):
            raise TypeError("search_space must be UserSearchSpace")
        self.search_space = search_space

    def context_for(self, branch: CompetingBranch) -> BranchRuntimeContext | None:
        if not isinstance(branch, CompetingBranch):
            raise TypeError("branch must be a CompetingBranch")
        user_bounds = self.search_space.bounds_for(branch.topology_id)
        if user_bounds is None or not self.search_space.resolution.permits(
            branch.resolution_present
        ):
            return None
        policies = tuple(gui_bounds_to_latent(item).d_policy for item in user_bounds)
        if any(
            (policy == "absent" and present) or (policy == "required" and not present)
            for policy, present in zip(policies, branch.d_present)
        ):
            return None
        user_resolution = self.search_space.resolution.bounds if branch.resolution_present else None
        full_resolution = (
            ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)
            if branch.resolution_present
            else None
        )
        try:
            full_codec = ProfiledBranchCodec.build(
                branch.topology,
                tuple(
                    full_component_bounds(shape, d_policy="optional") for shape in branch.topology
                ),
                branch.d_present,
                resolution_bounds=full_resolution,
            )
            user_codec = ProfiledBranchCodec.build(
                branch.topology,
                user_bounds,
                branch.d_present,
                resolution_bounds=user_resolution,
            )
        except ValueError:
            return None
        return BranchRuntimeContext(
            branch=branch,
            full_domain_codec=full_codec,
            user_bounds_codec=user_codec,
        )

    def condition_for(self, scored_branch: JointBranchScore) -> BranchCondition | None:
        if not isinstance(scored_branch, JointBranchScore):
            raise TypeError("scored_branch must be JointBranchScore")
        context = self.context_for(scored_branch.branch)
        if context is None:
            return None
        # V1's model input always remains the global cube, never the user-local cube.
        return replace(full_range_condition(scored_branch), refinement_context=context)

    def feasible_contexts(self) -> tuple[BranchRuntimeContext, ...]:
        contexts = []
        for topology in self.search_space.topologies:
            for pattern_id in range(BRANCH_PATTERN_COUNT):
                if not branch_pattern_is_valid(topology.topology_id, pattern_id):
                    continue
                context = self.context_for(
                    CompetingBranch(topology_id=topology.topology_id, pattern_id=pattern_id)
                )
                if context is not None:
                    contexts.append(context)
        return tuple(contexts)


@dataclass(frozen=True, kw_only=True)
class PhysicalRefinementSeed:
    source: SeedSource
    source_id: str
    context: BranchRuntimeContext
    components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    global_coordinates: BranchCoordinates
    user_local_coordinates: BranchCoordinates
    proposal_score_raw: float | None = None

    def __post_init__(self) -> None:
        if self.source not in {"neural", "retrieval", "sobol"}:
            raise ValueError("seed source must be neural, retrieval, or sobol")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not isinstance(self.context, BranchRuntimeContext):
            raise TypeError("context must be BranchRuntimeContext")
        components = tuple(self.components)
        if not all(isinstance(item, LatentComponentParameters) for item in components):
            raise TypeError("components must contain LatentComponentParameters")
        if tuple(item.shape for item in components) != self.context.branch.topology:
            raise ValueError("seed components do not match the hard topology")
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be ResolutionShape or None")
        if (self.resolution is not None) != self.context.branch.resolution_present:
            raise ValueError("seed resolution presence does not match its branch")
        if not isinstance(self.global_coordinates, BranchCoordinates) or not isinstance(
            self.user_local_coordinates, BranchCoordinates
        ):
            raise TypeError("seed coordinates must be BranchCoordinates")
        if self.global_coordinates.active_mask != self.context.full_domain_codec.active_mask:
            raise ValueError("global coordinate mask does not match the full-domain codec")
        if self.user_local_coordinates.active_mask != self.context.user_bounds_codec.active_mask:
            raise ValueError("local coordinate mask does not match the user codec")
        score = self.proposal_score_raw
        if score is not None and not np.isfinite(float(score)):
            raise ValueError("proposal_score_raw must be finite or None")
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "proposal_score_raw", None if score is None else float(score))

    @property
    def branch(self) -> CompetingBranch:
        return self.context.branch


def _full_range_assertion(proposal: BoundedProposalSample) -> None:
    active = np.asarray(proposal.condition.active_dimension_mask, dtype=bool)
    low = np.asarray(proposal.condition.branch_low)
    high = np.asarray(proposal.condition.branch_high)
    if np.any(low[active] != 0.0) or np.any(high[active] != 1.0):
        raise ValueError("production V1 neural proposals require full-range conditioning")
    if np.any(low[~active] != INACTIVE_UNIT_VALUE) or np.any(high[~active] != INACTIVE_UNIT_VALUE):
        raise ValueError("inactive full-range coordinates must remain canonical")


def physical_seed_from_neural(proposal: BoundedProposalSample) -> PhysicalRefinementSeed:
    """Decode only with the full codec, then validate by user-codec encoding."""

    if not isinstance(proposal, BoundedProposalSample):
        raise TypeError("proposal must be BoundedProposalSample")
    _full_range_assertion(proposal)
    context = proposal.condition.refinement_context
    if not isinstance(context, BranchRuntimeContext):
        raise TypeError("neural proposal is missing its BranchRuntimeContext")
    if context.branch != proposal.condition.branch:
        raise ValueError("neural proposal context belongs to a different branch")
    components, resolution = context.full_domain_codec.decode(proposal.global_unit)
    global_coordinates = context.full_domain_codec.encode(components, resolution)
    if not np.allclose(global_coordinates.unit_cube, proposal.global_unit, rtol=0.0, atol=5e-12):
        raise RuntimeError("full-domain proposal codec did not round-trip")
    # This encode is both the strict physical user-range and hard-core gate.
    user_coordinates = context.user_bounds_codec.encode(components, resolution)
    return PhysicalRefinementSeed(
        source="neural",
        source_id=(
            f"{context.branch.key}:mixture_{proposal.mixture_index:02d}:"
            f"sample_{proposal.sample_index:03d}"
        ),
        context=context,
        components=components,
        resolution=resolution,
        global_coordinates=global_coordinates,
        user_local_coordinates=user_coordinates,
        proposal_score_raw=proposal.raw_model_log_score,
    )


def physical_seed_from_external(
    *,
    source: Literal["retrieval", "sobol"],
    source_id: str,
    context: BranchRuntimeContext,
    components: Sequence[LatentComponentParameters],
    resolution: ResolutionShape | None,
) -> PhysicalRefinementSeed:
    """Validate a physical retrieval/Sobol seed against both codecs."""

    if source not in {"retrieval", "sobol"}:
        raise ValueError("external source must be retrieval or sobol")
    if not isinstance(context, BranchRuntimeContext):
        raise TypeError("context must be BranchRuntimeContext")
    values = tuple(components)
    user_coordinates = context.user_bounds_codec.encode(values, resolution)
    global_coordinates = context.full_domain_codec.encode(values, resolution)
    return PhysicalRefinementSeed(
        source=source,
        source_id=source_id,
        context=context,
        components=values,
        resolution=resolution,
        global_coordinates=global_coordinates,
        user_local_coordinates=user_coordinates,
    )


def physically_duplicate(
    seed: PhysicalRefinementSeed,
    previous: Sequence[PhysicalRefinementSeed],
    *,
    tolerance: float,
) -> bool:
    """Compare physical seeds through the unique full-domain codec inverse."""

    if not isinstance(seed, PhysicalRefinementSeed):
        raise TypeError("seed must be PhysicalRefinementSeed")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    def signature(value: PhysicalRefinementSeed):
        components = sorted(
            (
                item.shape,
                item.log_D is not None,
                tuple(
                    float(number)
                    for number in (
                        item.log_R,
                        item.sigma_R_fraction,
                        *(() if item.log_h is None else (item.log_h, item.sigma_h_fraction)),
                        *(() if item.log_D is None else (item.log_D, item.sigma_D_fraction)),
                    )
                ),
            )
            for item in value.components
        )
        resolution = (
            ()
            if value.resolution is None
            else (value.resolution.sigma_res, value.resolution.nu_res)
        )
        return components, tuple(float(item) for item in resolution)

    components, resolution = signature(seed)
    for other in previous:
        if not isinstance(other, PhysicalRefinementSeed):
            raise TypeError("previous must contain PhysicalRefinementSeed values")
        if (
            other.branch.topology_id != seed.branch.topology_id
            or other.branch.resolution_present != seed.branch.resolution_present
        ):
            continue
        other_components, other_resolution = signature(other)
        if tuple((item[0], item[1]) for item in components) != tuple(
            (item[0], item[1]) for item in other_components
        ):
            continue
        distances = [
            abs(left - right)
            for component, other_component in zip(components, other_components)
            for left, right in zip(component[2], other_component[2])
        ]
        distances.extend(abs(left - right) for left, right in zip(resolution, other_resolution))
        if max(distances, default=0.0) <= tolerance:
            return True
    return False


@dataclass(frozen=True, kw_only=True)
class ProductionExactRefiner:
    curve: ObservedCurve
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.curve, ObservedCurve):
            raise TypeError("curve must be ObservedCurve")
        for name in ("ftol", "xtol", "gtol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= np.finfo(float).eps:
                raise ValueError(f"{name} must exceed machine epsilon")
            object.__setattr__(self, name, value)

    def _candidate(
        self,
        seed: PhysicalRefinementSeed,
        profile: ProfiledForwardResult,
        exact: np.ndarray,
        *,
        candidate_id: str,
        proposal_rank: int,
    ) -> CandidateInput:
        seed.context.user_bounds_codec.encode(
            seed.components,
            seed.resolution,
        )
        return CandidateInput(
            candidate_id=candidate_id,
            proposal_rank=proposal_rank,
            topology_id=seed.branch.topology_id,
            components=seed.components,
            resolution=seed.resolution,
            linear_solution=LinearSolutionSnapshot.from_profiled_forward(profile),
            exact_intensity=exact,
            bounds_pass=True,
            physics_pass=True,
            proposal_score_raw=seed.proposal_score_raw,
        )

    def refine_physical(
        self,
        seed: PhysicalRefinementSeed,
        *,
        candidate_id: str,
        proposal_rank: int,
        max_forward_evaluations: int,
    ) -> RefinementOutcome:
        """Return a GUI-consistent ``CandidateInput`` or one explicit failure."""

        if not isinstance(seed, PhysicalRefinementSeed):
            raise TypeError("seed must be PhysicalRefinementSeed")
        allowance = _positive_integer(max_forward_evaluations, "max_forward_evaluations")
        proposal_rank = _positive_integer(proposal_rank, "proposal_rank")
        profile_sigma = self.curve.intensity * (
            1.0 if self.curve.sigma_log is None else self.curve.sigma_log
        )
        active_dimensions = len(seed.context.user_bounds_codec.active_indices)
        # Reserve finite-difference calls plus three evaluations outside nfev.
        solver_nfev = (allowance - 3) // (active_dimensions + 1)
        try:
            if solver_nfev < 1:
                gui = tuple(latent_component_to_gui(item) for item in seed.components)
                profile = profile_linear_amplitudes(
                    self.curve.q,
                    self.curve.intensity,
                    gui,
                    resolution=seed.resolution,
                    sigma=profile_sigma,
                )
                exact = evaluate_profiled_forward(self.curve.q, profile)
                candidate = self._candidate(
                    seed,
                    profile,
                    exact,
                    candidate_id=candidate_id,
                    proposal_rank=proposal_rank,
                )
                return RefinementOutcome(
                    status="success",
                    forward_evaluations=1,
                    value=candidate,
                    message="exact seed profile returned under a small forward budget",
                )
            result = refine_profiled_branch(
                self.curve.q,
                self.curve.intensity,
                seed.context.component_bounds,
                seed.components,
                resolution_bounds=seed.context.resolution_bounds,
                resolution_seed=seed.resolution,
                sigma_log=self.curve.sigma_log,
                max_nfev=solver_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
            )
            consumed = result.residual_calls + 3
            if consumed > allowance:
                raise RuntimeError("refinement residual calls exceeded the reserved budget")
            final_components = tuple(result.final_latent_components)
            final_resolution = result.final_resolution
            final_seed = PhysicalRefinementSeed(
                source=seed.source,
                source_id=seed.source_id,
                context=seed.context,
                components=final_components,
                resolution=final_resolution,
                global_coordinates=seed.context.full_domain_codec.encode(
                    final_components, final_resolution
                ),
                user_local_coordinates=seed.context.user_bounds_codec.encode(
                    final_components, final_resolution
                ),
                proposal_score_raw=seed.proposal_score_raw,
            )
            candidate = self._candidate(
                final_seed,
                result.final_profile,
                result.exact_forward_intensity,
                candidate_id=candidate_id,
                proposal_rank=proposal_rank,
            )
            return RefinementOutcome(
                status="success",
                forward_evaluations=consumed,
                value=candidate,
                message=(
                    "nonlinear optimizer converged"
                    if result.success
                    else "best exact point returned after optimizer limit"
                ),
            )
        except _CANDIDATE_ERRORS as exc:
            return RefinementOutcome(
                status="failed",
                forward_evaluations=allowance,
                message=f"{type(exc).__name__}: {exc}",
            )

    def refine(
        self,
        proposal: BoundedProposalSample,
        *,
        max_forward_evaluations: int,
    ) -> RefinementOutcome:
        """Implement ``ExactRefinerPort`` for direct one-click use."""

        try:
            seed = physical_seed_from_neural(proposal)
        except _CANDIDATE_ERRORS as exc:
            return RefinementOutcome(
                status="failed",
                forward_evaluations=0,
                message=f"global_to_user_validation: {type(exc).__name__}: {exc}",
            )
        return self.refine_physical(
            seed,
            candidate_id=f"candidate_{seed.source_id}",
            proposal_rank=1,
            max_forward_evaluations=max_forward_evaluations,
        )


__all__ = [
    "PRODUCTION_BRIDGE_VERSION",
    "BranchRuntimeContext",
    "PhysicalRefinementSeed",
    "ProductionBranchFactory",
    "ProductionExactRefiner",
    "ResolutionSearchPolicy",
    "TopologyUserBounds",
    "UserSearchSpace",
    "physical_seed_from_external",
    "physical_seed_from_neural",
    "physically_duplicate",
]
