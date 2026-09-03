"""TensorFlow-free sampling of physical V5 proposals from batched model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np
from scipy.special import expit

from .branch_codec import INACTIVE_UNIT_VALUE, UNIT_CUBE_DIMENSIONS
from .candidate_batch_v5 import V5CandidateContextBatch
from .canonical_component_slots import canonicalize_component_slots
from .contract import LatentComponentParameters
from .profiled_forward import ResolutionShape


V5_PROPOSAL_SAMPLER_VERSION = (
    "posterior_v8_complete_slot_contract_branch_local_logistic_normal_sampler_v2"
)
V5_PROPOSAL_RANKING_SEMANTICS = (
    "frozen_search_yield_logit_then_conditional_mixture_log_weight_"
    "not_mathematical_solvability_or_posterior_v1"
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


@dataclass(frozen=True)
class V5BatchedProposalOutput:
    """Validated NumPy view of one model call over candidate branches."""

    search_yield_logit: np.ndarray
    mixture_logits: np.ndarray
    mixture_loc: np.ndarray
    mixture_logscale: np.ndarray

    @classmethod
    def from_mapping(
        cls,
        outputs: Mapping[str, object],
        *,
        branch_count: int,
    ) -> "V5BatchedProposalOutput":
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        required = {
            "proposal_search_yield_logit",
            "mixture_logits",
            "mixture_loc",
            "mixture_logscale",
        }
        missing = required - set(outputs)
        if missing:
            raise ValueError(f"V5 model output is missing {sorted(missing)}")
        search_yield = _finite_array(
            outputs["proposal_search_yield_logit"],
            (branch_count, 1),
            "proposal_search_yield_logit",
        )[:, 0]
        logits = np.asarray(outputs["mixture_logits"], dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != branch_count or logits.shape[1] < 1:
            raise ValueError("mixture_logits must have shape [branch_count, mixture_count]")
        logits = _finite_array(logits, logits.shape, "mixture_logits")
        mixture_count = logits.shape[1]
        shape = (branch_count, mixture_count, UNIT_CUBE_DIMENSIONS)
        loc = _finite_array(outputs["mixture_loc"], shape, "mixture_loc")
        logscale = _finite_array(outputs["mixture_logscale"], shape, "mixture_logscale")
        if not np.all(np.isfinite(np.exp(logscale))):
            raise ValueError("mixture_logscale produces non-finite scales")
        return cls(search_yield, logits, loc, logscale)

    @property
    def branch_count(self) -> int:
        return int(self.mixture_logits.shape[0])

    @property
    def mixture_count(self) -> int:
        return int(self.mixture_logits.shape[1])


@dataclass(frozen=True, kw_only=True)
class V5LocalProposal:
    """One in-bounds branch-local neural seed, ready for amplitude profiling."""

    query_sha256: str
    topology_id: int
    pattern_id: int
    branch_batch_index: int
    branch_rank: int
    mixture_index: int
    mixture_rank: int
    draw_index: int
    source: str
    search_yield_logit: float
    mixture_log_weight: float
    local_unit: tuple[float, ...]
    latent_components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    version: str = V5_PROPOSAL_SAMPLER_VERSION

    def __post_init__(self) -> None:
        if self.version != V5_PROPOSAL_SAMPLER_VERSION:
            raise ValueError("unsupported V5 proposal sampler version")
        if self.source not in {"mixture_median", "stochastic_draw"}:
            raise ValueError("proposal source must be a mixture median or stochastic draw")
        for name in ("branch_rank", "mixture_rank"):
            _positive_integer(getattr(self, name), name)
        for name in ("branch_batch_index", "mixture_index", "draw_index"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.source == "mixture_median" and self.draw_index != 0:
            raise ValueError("mixture median must use draw_index=0")
        if self.source == "stochastic_draw" and self.draw_index < 1:
            raise ValueError("stochastic draw must use a positive draw_index")
        scores = (float(self.search_yield_logit), float(self.mixture_log_weight))
        if not np.all(np.isfinite(scores)):
            raise ValueError("proposal scores must be finite")
        local = np.asarray(self.local_unit, dtype=np.float64)
        if local.shape != (UNIT_CUBE_DIMENSIONS,) or not np.all(np.isfinite(local)):
            raise ValueError("local_unit must be a finite 26-vector")
        if np.any(local < 0.0) or np.any(local > 1.0):
            raise ValueError("local_unit must lie inside the user cube")
        if not self.latent_components or not all(
            isinstance(value, LatentComponentParameters) for value in self.latent_components
        ):
            raise TypeError("latent_components must contain physical components")
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be ResolutionShape or None")

    @property
    def ranking_key(self) -> tuple[float, float, int, int, int]:
        """Deterministic key; lower sorts earlier without conflating score units."""

        return (
            -float(self.search_yield_logit),
            -float(self.mixture_log_weight),
            int(self.pattern_id),
            int(self.mixture_index),
            int(self.draw_index),
        )


def sample_v5_local_proposals(
    batch: V5CandidateContextBatch,
    outputs: Mapping[str, object] | V5BatchedProposalOutput,
    *,
    mixture_limit: int,
    stochastic_draws_per_mixture: int,
    seed: int,
    include_mixture_medians: bool = True,
) -> tuple[V5LocalProposal, ...]:
    """Sample local MDNs without pruning an entire branch unless requested upstream."""

    if not isinstance(batch, V5CandidateContextBatch):
        raise TypeError("batch must be a V5CandidateContextBatch")
    mixture_limit = _positive_integer(mixture_limit, "mixture_limit")
    if (
        isinstance(stochastic_draws_per_mixture, (bool, np.bool_))
        or not isinstance(stochastic_draws_per_mixture, Integral)
        or stochastic_draws_per_mixture < 0
    ):
        raise ValueError("stochastic_draws_per_mixture must be a non-negative integer")
    if not include_mixture_medians and stochastic_draws_per_mixture == 0:
        raise ValueError("proposal sampling must emit at least one seed per selected mixture")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    parsed = (
        outputs
        if isinstance(outputs, V5BatchedProposalOutput)
        else V5BatchedProposalOutput.from_mapping(outputs, branch_count=batch.branch_count)
    )
    if parsed.branch_count != batch.branch_count:
        raise ValueError("model output branch count does not match the candidate batch")

    branch_order = sorted(
        range(batch.branch_count),
        key=lambda index: (
            -float(parsed.search_yield_logit[index]),
            batch.branch_conditions[index].pattern_id,
        ),
    )
    log_weights = _log_softmax(parsed.mixture_logits)
    proposals: list[V5LocalProposal] = []
    for branch_rank, batch_index in enumerate(branch_order, 1):
        condition = batch.branch_conditions[batch_index]
        codec = batch.query.codec_for(condition.pattern_id)
        varying = np.asarray(condition.varying_dimension_mask, dtype=bool)
        mixture_order = sorted(
            range(parsed.mixture_count),
            key=lambda index: (-float(log_weights[batch_index, index]), index),
        )[: min(mixture_limit, parsed.mixture_count)]
        for mixture_rank, mixture_index in enumerate(mixture_order, 1):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(seed),
                        int(condition.topology_id),
                        int(condition.pattern_id),
                        int(mixture_index),
                    ]
                )
            )
            values: list[tuple[str, int, np.ndarray]] = []
            median = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
            median[varying] = expit(parsed.mixture_loc[batch_index, mixture_index, varying])
            if include_mixture_medians:
                values.append(("mixture_median", 0, median))
            if stochastic_draws_per_mixture:
                noise = rng.normal(
                    size=(stochastic_draws_per_mixture, int(np.count_nonzero(varying)))
                )
                scales = np.exp(parsed.mixture_logscale[batch_index, mixture_index, varying])
                latent = parsed.mixture_loc[batch_index, mixture_index, varying] + noise * scales
                local_draws = expit(latent)
                for draw_index, local_values in enumerate(local_draws, 1):
                    draw = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
                    draw[varying] = local_values
                    values.append(("stochastic_draw", draw_index, draw))

            for source, draw_index, local in values:
                components, resolution = codec.decode(local)
                canonical = canonicalize_component_slots(
                    codec,
                    components,
                    resolution,
                    component_intensity_bounds=batch.amplitude_query.component_intensities,
                )
                proposals.append(
                    V5LocalProposal(
                        query_sha256=batch.query_sha256,
                        topology_id=condition.topology_id,
                        pattern_id=condition.pattern_id,
                        branch_batch_index=batch_index,
                        branch_rank=branch_rank,
                        mixture_index=mixture_index,
                        mixture_rank=mixture_rank,
                        draw_index=draw_index,
                        source=source,
                        search_yield_logit=float(parsed.search_yield_logit[batch_index]),
                        mixture_log_weight=float(log_weights[batch_index, mixture_index]),
                        local_unit=canonical.coordinates.unit_cube,
                        latent_components=canonical.components,
                        resolution=canonical.resolution,
                    )
                )
    proposals.sort(key=lambda value: value.ranking_key)
    return tuple(proposals)


__all__ = [
    "V5_PROPOSAL_RANKING_SEMANTICS",
    "V5_PROPOSAL_SAMPLER_VERSION",
    "V5BatchedProposalOutput",
    "V5LocalProposal",
    "sample_v5_local_proposals",
]
