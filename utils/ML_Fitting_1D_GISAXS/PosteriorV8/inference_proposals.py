"""TensorFlow-free neural proposal primitives for Posterior V8 inference.

The trained model targets the global branch-codec cube.  Sampling conditions
that distribution on the user envelope and exposes its local box coordinate.
This equals rejection plus ``low + (high-low) * u``, but exact truncated-normal
draws avoid retries.  It is not a model trained directly on local coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Callable, Mapping, Protocol

import numpy as np
from scipy.special import expit, log_ndtr
from scipy.stats import truncnorm

from .branch_catalog import (
    BRANCH_PATTERN_COUNT,
    VALID_BRANCH_PATTERN_MASK,
)
from .branch_codec import (
    COMPONENT_STRIDE,
    INACTIVE_UNIT_VALUE,
    RESOLUTION_OFFSET,
    UNIT_CUBE_DIMENSIONS,
)
from .contract import CYLINDER, NUM_TOPOLOGIES
from .reference_bank import CompetingBranch


INFERENCE_PROPOSAL_VERSION = "posterior_v8_joint_beam_global_box_v1"
GLOBAL_TARGET_AFFINE_SEMANTICS = "global_target_exact_truncation_then_local_box_affine_v1"
_LOG_SQRT_TWO_PI = 0.5 * np.log(2.0 * np.pi)


def _positive_integer(value: int, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1 or (maximum is not None and result > maximum):
        suffix = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be positive{suffix}")
    return result


def _finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _log_softmax(values: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)) or not np.any(np.isfinite(values)):
        raise FloatingPointError("log-softmax has no finite support")
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted)))


@dataclass(frozen=True, kw_only=True)
class DiscreteProposalOutput:
    topology_logits: np.ndarray
    branch_pattern_logits: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topology_logits",
            _finite_array(self.topology_logits, (NUM_TOPOLOGIES,), "topology_logits"),
        )
        object.__setattr__(
            self,
            "branch_pattern_logits",
            _finite_array(
                self.branch_pattern_logits,
                (NUM_TOPOLOGIES, BRANCH_PATTERN_COUNT),
                "branch_pattern_logits",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ContinuousProposalOutput:
    mixture_logits: np.ndarray
    mixture_loc: np.ndarray
    mixture_logscale: np.ndarray

    def __post_init__(self) -> None:
        logits = np.asarray(self.mixture_logits, dtype=np.float64)
        if logits.ndim != 1 or logits.size == 0:
            raise ValueError("mixture_logits must be a non-empty vector")
        shape = (logits.size, UNIT_CUBE_DIMENSIONS)
        object.__setattr__(
            self, "mixture_logits", _finite_array(logits, logits.shape, "mixture_logits")
        )
        object.__setattr__(
            self, "mixture_loc", _finite_array(self.mixture_loc, shape, "mixture_loc")
        )
        logscale = _finite_array(self.mixture_logscale, shape, "mixture_logscale")
        if not np.all(np.isfinite(np.exp(logscale))):
            raise ValueError("mixture_logscale produces a non-finite scale")
        object.__setattr__(self, "mixture_logscale", logscale)


@dataclass(frozen=True, kw_only=True)
class JointBranchScore:
    branch: CompetingBranch
    topology_rank: int
    pattern_rank_within_topology: int
    joint_rank: int
    topology_log_score: float
    conditional_pattern_log_score: float
    joint_log_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.branch, CompetingBranch):
            raise TypeError("branch must be a CompetingBranch")
        for name in ("topology_rank", "pattern_rank_within_topology", "joint_rank"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        values = (
            float(self.topology_log_score),
            float(self.conditional_pattern_log_score),
            float(self.joint_log_score),
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("joint branch scores must be finite")


def rank_joint_branches(
    output: DiscreteProposalOutput,
    *,
    topology_limit: int,
) -> tuple[JointBranchScore, ...]:
    """Build a deterministic joint beam from P(topology) P(pattern|topology)."""

    if not isinstance(output, DiscreteProposalOutput):
        raise TypeError("output must be DiscreteProposalOutput")
    topology_limit = _positive_integer(topology_limit, "topology_limit", maximum=NUM_TOPOLOGIES)
    topology_log_scores = _log_softmax(output.topology_logits)
    topology_order = sorted(
        range(NUM_TOPOLOGIES), key=lambda index: (-topology_log_scores[index], index)
    )[:topology_limit]
    preliminary = []
    for topology_rank, topology_id in enumerate(topology_order, 1):
        valid = np.asarray(VALID_BRANCH_PATTERN_MASK[topology_id], dtype=bool)
        valid_ids = np.flatnonzero(valid)
        conditional = _log_softmax(output.branch_pattern_logits[topology_id, valid])
        pattern_order = sorted(
            range(valid_ids.size),
            key=lambda index: (-conditional[index], int(valid_ids[index])),
        )
        for pattern_rank, local_index in enumerate(pattern_order, 1):
            pattern_id = int(valid_ids[local_index])
            topology_score = float(topology_log_scores[topology_id])
            pattern_score = float(conditional[local_index])
            preliminary.append(
                JointBranchScore(
                    branch=CompetingBranch(topology_id=topology_id, pattern_id=pattern_id),
                    topology_rank=topology_rank,
                    pattern_rank_within_topology=pattern_rank,
                    joint_rank=1,
                    topology_log_score=topology_score,
                    conditional_pattern_log_score=pattern_score,
                    joint_log_score=topology_score + pattern_score,
                )
            )
    preliminary.sort(
        key=lambda item: (
            -item.joint_log_score,
            item.branch.topology_id,
            item.branch.pattern_id,
        )
    )
    return tuple(replace(item, joint_rank=rank) for rank, item in enumerate(preliminary, 1))


def expected_active_mask(branch: CompetingBranch) -> tuple[bool, ...]:
    """Return the fixed 26-D mask implied by one hard discrete branch."""

    if not isinstance(branch, CompetingBranch):
        raise TypeError("branch must be a CompetingBranch")
    mask = [False] * UNIT_CUBE_DIMENSIONS
    for slot, shape in enumerate(branch.topology):
        offset = slot * COMPONENT_STRIDE
        mask[offset : offset + 2] = (True, True)
        if shape == CYLINDER:
            mask[offset + 2 : offset + 4] = (True, True)
        if branch.d_present[slot]:
            mask[offset + 4 : offset + 6] = (True, True)
    if branch.resolution_present:
        mask[RESOLUTION_OFFSET:] = (True, True)
    return tuple(mask)


@dataclass(frozen=True, kw_only=True)
class BranchCondition:
    """One hard branch plus a global-coordinate user envelope.

    ``refinement_context`` is deliberately opaque.  A production adapter can
    carry full-domain and user-bounds codecs there, decode ``global_unit`` to
    physical parameters, validate them, and only then encode for the local
    user-bounds codec used by exact refinement.
    """

    scored_branch: JointBranchScore
    branch_low: tuple[float, ...]
    branch_high: tuple[float, ...]
    active_dimension_mask: tuple[bool, ...]
    refinement_context: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scored_branch, JointBranchScore):
            raise TypeError("scored_branch must be a JointBranchScore")
        low = _finite_array(self.branch_low, (UNIT_CUBE_DIMENSIONS,), "branch_low")
        high = _finite_array(self.branch_high, (UNIT_CUBE_DIMENSIONS,), "branch_high")
        active = np.asarray(self.active_dimension_mask)
        if active.shape != (UNIT_CUBE_DIMENSIONS,) or active.dtype.kind != "b":
            raise ValueError("active_dimension_mask must be a boolean vector of length 26")
        expected = expected_active_mask(self.scored_branch.branch)
        if tuple(bool(value) for value in active) != expected:
            raise ValueError("active_dimension_mask does not match the hard branch")
        if np.any(low < 0.0) or np.any(high > 1.0) or np.any(low > high):
            raise ValueError("branch bounds must satisfy 0 <= low <= high <= 1")
        inactive = ~active
        if np.any(low[inactive] != INACTIVE_UNIT_VALUE) or np.any(
            high[inactive] != INACTIVE_UNIT_VALUE
        ):
            raise ValueError("inactive branch bounds must equal the canonical 0.5")
        object.__setattr__(self, "branch_low", tuple(float(value) for value in low))
        object.__setattr__(self, "branch_high", tuple(float(value) for value in high))
        object.__setattr__(self, "active_dimension_mask", tuple(bool(value) for value in active))

    @property
    def branch(self) -> CompetingBranch:
        return self.scored_branch.branch


def full_range_condition(scored_branch: JointBranchScore) -> BranchCondition:
    """Construct the non-leaking full-domain condition for one branch."""

    active = np.asarray(expected_active_mask(scored_branch.branch), dtype=bool)
    low = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
    high = low.copy()
    low[active], high[active] = 0.0, 1.0
    return BranchCondition(
        scored_branch=scored_branch,
        branch_low=tuple(low),
        branch_high=tuple(high),
        active_dimension_mask=tuple(active),
    )


def default_probe_condition() -> BranchCondition:
    """Return a harmless legal branch used only to read curve-only heads."""

    return full_range_condition(
        JointBranchScore(
            branch=CompetingBranch(topology_id=0, pattern_id=0),
            topology_rank=1,
            pattern_rank_within_topology=1,
            joint_rank=1,
            topology_log_score=0.0,
            conditional_pattern_log_score=0.0,
            joint_log_score=0.0,
        )
    )


def _single_output(mapping: Mapping[str, object], name: str, dimensions: int) -> np.ndarray:
    if name not in mapping:
        raise ValueError(f"model output is missing {name!r}")
    array = np.asarray(mapping[name])
    if array.ndim == dimensions + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != dimensions:
        raise ValueError(f"model output {name!r} has the wrong rank")
    return array


def discrete_output_from_mapping(mapping: Mapping[str, object]) -> DiscreteProposalOutput:
    return DiscreteProposalOutput(
        topology_logits=_single_output(mapping, "topology_logits", 1),
        branch_pattern_logits=_single_output(mapping, "branch_pattern_logits", 2),
    )


def continuous_output_from_mapping(mapping: Mapping[str, object]) -> ContinuousProposalOutput:
    return ContinuousProposalOutput(
        mixture_logits=_single_output(mapping, "mixture_logits", 1),
        mixture_loc=_single_output(mapping, "mixture_loc", 2),
        mixture_logscale=_single_output(mapping, "mixture_logscale", 2),
    )


def build_model_inputs(
    curve_inputs: Mapping[str, object], condition: BranchCondition
) -> dict[str, np.ndarray]:
    """Build the model's nine batch-one inputs without importing TensorFlow."""

    if not isinstance(curve_inputs, Mapping):
        raise TypeError("curve_inputs must be a mapping")
    if not isinstance(condition, BranchCondition):
        raise TypeError("condition must be a BranchCondition")
    missing = {"x", "point_mask", "global_features"} - set(curve_inputs)
    if missing:
        raise ValueError(f"curve_inputs is missing {sorted(missing)!r}")
    x = np.asarray(curve_inputs["x"], dtype=np.float32)
    point_mask = np.asarray(curve_inputs["point_mask"])
    global_features = np.asarray(curve_inputs["global_features"], dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if point_mask.ndim == 2 and point_mask.shape[0] == 1:
        point_mask = point_mask[0]
    if global_features.ndim == 2 and global_features.shape[0] == 1:
        global_features = global_features[0]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape [points, 3] or [1, points, 3]")
    if point_mask.shape != (x.shape[0],) or point_mask.dtype.kind != "b":
        raise ValueError("point_mask must be a boolean vector matching x")
    if global_features.shape != (5,) or not np.all(np.isfinite(global_features)):
        raise ValueError("global_features must be a finite vector of length five")
    if not np.any(point_mask) or not np.all(np.isfinite(x[point_mask])):
        raise ValueError("x must contain at least one finite valid point")
    branch = condition.branch
    topology = np.zeros(NUM_TOPOLOGIES, dtype=np.float32)
    topology[branch.topology_id] = 1.0
    d_present = np.zeros(4, dtype=np.float32)
    d_present[: len(branch.d_present)] = branch.d_present
    return {
        "x": x[np.newaxis, ...],
        "point_mask": point_mask[np.newaxis, ...],
        "global_features": global_features[np.newaxis, ...],
        "branch_topology": topology[np.newaxis, ...],
        "branch_d_present": d_present[np.newaxis, ...],
        "branch_resolution_present": np.asarray(
            [[float(branch.resolution_present)]], dtype=np.float32
        ),
        "branch_low": np.asarray([condition.branch_low], dtype=np.float32),
        "branch_high": np.asarray([condition.branch_high], dtype=np.float32),
        "active_dimension_mask": np.asarray([condition.active_dimension_mask], dtype=np.float32),
    }


class ProposalModelPort(Protocol):
    def predict_discrete(self, curve_inputs: Mapping[str, object]) -> DiscreteProposalOutput: ...

    def predict_continuous(
        self, curve_inputs: Mapping[str, object], condition: BranchCondition
    ) -> ContinuousProposalOutput: ...


@dataclass(frozen=True)
class CallableProposalModel:
    """Adapt a Keras-like batch-one callable to the TF-free inference port."""

    predictor: Callable[[Mapping[str, np.ndarray]], Mapping[str, object]]
    probe_condition: BranchCondition = field(default_factory=default_probe_condition)

    def __post_init__(self) -> None:
        if not callable(self.predictor):
            raise TypeError("predictor must be callable")
        if not isinstance(self.probe_condition, BranchCondition):
            raise TypeError("probe_condition must be a BranchCondition")

    def predict_discrete(self, curve_inputs: Mapping[str, object]) -> DiscreteProposalOutput:
        return discrete_output_from_mapping(
            self.predictor(build_model_inputs(curve_inputs, self.probe_condition))
        )

    def predict_continuous(
        self, curve_inputs: Mapping[str, object], condition: BranchCondition
    ) -> ContinuousProposalOutput:
        return continuous_output_from_mapping(
            self.predictor(build_model_inputs(curve_inputs, condition))
        )


def _log_normal_interval(a: float, b: float) -> float:
    if not a < b:
        raise ValueError("standardized truncation bounds must be ordered")
    if np.isneginf(a) and np.isposinf(b):
        return 0.0
    if a >= 0.0:
        high, low = float(log_ndtr(-a)), float(log_ndtr(-b))
    else:
        high, low = float(log_ndtr(b)), float(log_ndtr(a))
    if low == -np.inf:
        return high
    difference = low - high
    if difference >= 0.0:
        midpoint = 0.5 * (a + b)
        approximation = -0.5 * midpoint**2 - _LOG_SQRT_TWO_PI + np.log(b - a)
        if not np.isfinite(approximation):
            raise FloatingPointError("normal interval mass underflowed")
        return float(approximation)
    return float(high + np.log(-np.expm1(difference)))


def _logit_bound(value: float) -> float:
    if value == 0.0:
        return -np.inf
    if value == 1.0:
        return np.inf
    return float(np.log(value) - np.log1p(-value))


@dataclass(frozen=True, kw_only=True)
class BoundedProposalSample:
    condition: BranchCondition
    mixture_index: int
    mixture_rank: int
    sample_index: int
    conditioned_mixture_log_weight: float
    local_box_unit: tuple[float, ...]
    global_unit: tuple[float, ...]
    coordinate_semantics: str = GLOBAL_TARGET_AFFINE_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.condition, BranchCondition):
            raise TypeError("condition must be a BranchCondition")
        for name in ("mixture_index", "mixture_rank", "sample_index"):
            value = getattr(self, name)
            if name == "mixture_index":
                if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                    raise ValueError("mixture_index must be a non-negative integer")
                object.__setattr__(self, name, int(value))
            else:
                object.__setattr__(self, name, _positive_integer(value, name))
        score = float(self.conditioned_mixture_log_weight)
        if not np.isfinite(score):
            raise ValueError("conditioned mixture log weight must be finite")
        if self.coordinate_semantics != GLOBAL_TARGET_AFFINE_SEMANTICS:
            raise ValueError("unsupported proposal coordinate semantics")
        local = _finite_array(self.local_box_unit, (UNIT_CUBE_DIMENSIONS,), "local_box_unit")
        global_unit = _finite_array(self.global_unit, (UNIT_CUBE_DIMENSIONS,), "global_unit")
        low = np.asarray(self.condition.branch_low)
        high = np.asarray(self.condition.branch_high)
        active = np.asarray(self.condition.active_dimension_mask)
        if np.any(local < 0.0) or np.any(local > 1.0):
            raise ValueError("local box coordinates must lie in [0, 1]")
        canonical = (~active) | (low == high)
        if np.any(local[canonical] != INACTIVE_UNIT_VALUE):
            raise ValueError("inactive and fixed local coordinates must equal 0.5")
        expected = low + (high - low) * local
        expected[~active] = INACTIVE_UNIT_VALUE
        if not np.array_equal(global_unit, expected):
            raise ValueError("global coordinates must equal low + (high-low) * local")
        object.__setattr__(self, "local_box_unit", tuple(float(value) for value in local))
        object.__setattr__(self, "global_unit", tuple(float(value) for value in global_unit))

    @property
    def raw_model_log_score(self) -> float:
        return float(
            self.condition.scored_branch.joint_log_score + self.conditioned_mixture_log_weight
        )


def sample_bounded_global_mixture(
    output: ContinuousProposalOutput,
    condition: BranchCondition,
    *,
    mixture_limit: int,
    samples_per_mixture: int,
    seed: int,
) -> tuple[BoundedProposalSample, ...]:
    """Condition the global MDN on a box and expose exact affine local coordinates.

    No coordinate is clipped.  Zero-width active dimensions are fixed exactly;
    inactive dimensions remain the codec's canonical 0.5.
    """

    if not isinstance(output, ContinuousProposalOutput):
        raise TypeError("output must be ContinuousProposalOutput")
    if not isinstance(condition, BranchCondition):
        raise TypeError("condition must be a BranchCondition")
    mixture_limit = _positive_integer(mixture_limit, "mixture_limit")
    samples_per_mixture = _positive_integer(samples_per_mixture, "samples_per_mixture")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    low = np.asarray(condition.branch_low, dtype=np.float64)
    high = np.asarray(condition.branch_high, dtype=np.float64)
    active = np.asarray(condition.active_dimension_mask, dtype=bool)
    varying = active & (high > low)
    fixed = active & (high == low)
    scales = np.exp(output.mixture_logscale)
    log_weights = _log_softmax(output.mixture_logits)
    conditioned = log_weights.copy()
    varying_indices = np.flatnonzero(varying)
    for mixture_index in range(output.mixture_logits.size):
        for dimension in varying_indices:
            location = output.mixture_loc[mixture_index, dimension]
            scale = scales[mixture_index, dimension]
            a = (_logit_bound(low[dimension]) - location) / scale
            b = (_logit_bound(high[dimension]) - location) / scale
            conditioned[mixture_index] += _log_normal_interval(a, b)
        for dimension in np.flatnonzero(fixed & (low > 0.0) & (high < 1.0)):
            latent = _logit_bound(low[dimension])
            location = output.mixture_loc[mixture_index, dimension]
            scale = scales[mixture_index, dimension]
            standardized = (latent - location) / scale
            conditioned[mixture_index] += -0.5 * standardized**2 - np.log(scale) - _LOG_SQRT_TWO_PI
    conditioned = _log_softmax(conditioned)
    mixture_order = sorted(range(conditioned.size), key=lambda index: (-conditioned[index], index))[
        : min(mixture_limit, conditioned.size)
    ]

    generated = []
    for mixture_rank, mixture_index in enumerate(mixture_order, 1):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(seed),
                    condition.branch.topology_id,
                    condition.branch.pattern_id,
                    mixture_index,
                ]
            )
        )
        local = np.full(
            (samples_per_mixture, UNIT_CUBE_DIMENSIONS),
            INACTIVE_UNIT_VALUE,
            dtype=np.float64,
        )
        if varying_indices.size:
            locations = output.mixture_loc[mixture_index, varying_indices]
            selected_scales = scales[mixture_index, varying_indices]
            a = np.asarray(
                [
                    (_logit_bound(low[index]) - location) / scale
                    for index, location, scale in zip(varying_indices, locations, selected_scales)
                ]
            )
            b = np.asarray(
                [
                    (_logit_bound(high[index]) - location) / scale
                    for index, location, scale in zip(varying_indices, locations, selected_scales)
                ]
            )
            latent = truncnorm.rvs(
                a,
                b,
                loc=locations,
                scale=selected_scales,
                size=(samples_per_mixture, varying_indices.size),
                random_state=rng,
            )
            global_draw = expit(latent)
            local_draw = (global_draw - low[varying_indices]) / (
                high[varying_indices] - low[varying_indices]
            )
            if np.any(local_draw < 0.0) or np.any(local_draw > 1.0):
                raise FloatingPointError("truncated sample escaped the user envelope")
            local[:, varying_indices] = local_draw
        global_unit = low + (high - low) * local
        global_unit[:, ~active] = INACTIVE_UNIT_VALUE
        if np.any(global_unit < low) or np.any(global_unit > high):
            raise RuntimeError("affine mapping escaped the user envelope")
        for sample_index in range(samples_per_mixture):
            generated.append(
                BoundedProposalSample(
                    condition=condition,
                    mixture_index=mixture_index,
                    mixture_rank=mixture_rank,
                    sample_index=sample_index + 1,
                    conditioned_mixture_log_weight=float(conditioned[mixture_index]),
                    local_box_unit=tuple(local[sample_index]),
                    global_unit=tuple(global_unit[sample_index]),
                )
            )
    return tuple(generated)


__all__ = [
    "GLOBAL_TARGET_AFFINE_SEMANTICS",
    "INFERENCE_PROPOSAL_VERSION",
    "BoundedProposalSample",
    "BranchCondition",
    "CallableProposalModel",
    "ContinuousProposalOutput",
    "DiscreteProposalOutput",
    "JointBranchScore",
    "ProposalModelPort",
    "build_model_inputs",
    "continuous_output_from_mapping",
    "default_probe_condition",
    "discrete_output_from_mapping",
    "expected_active_mask",
    "full_range_condition",
    "rank_joint_branches",
    "sample_bounded_global_mixture",
]
