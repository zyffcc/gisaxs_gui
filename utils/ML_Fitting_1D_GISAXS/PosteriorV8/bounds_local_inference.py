"""TensorFlow-free inference primitives for the bounds-first V3 proposal model.

The continuous model output in this module is *already* expressed in the
26-D local unit cube of ``BranchRuntimeContext.user_bounds_codec``.  It must
never be passed through the global-target V1 affine/truncation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Callable, Mapping, Protocol

import numpy as np
from scipy.special import expit

from .bounds_first_contract import BOUNDS_EMBEDDING_DIM, bounds_embedding
from .bounds_model_contract import (
    BOUNDS_PROPOSAL_MODEL_VERSION,
    MODEL_FIXED_DIMENSIONS,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
    MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS,
    MODEL_INPUT_KEYS,
    MODEL_OUTPUT_COORDINATE_SEMANTICS,
    MODEL_OUTPUT_KEYS,
    MODEL_PHYSICAL_BRANCH_COUNT,
)
from .branch_codec import INACTIVE_UNIT_VALUE, ProfiledBranchCodec, UNIT_CUBE_DIMENSIONS
from .canonical_branch_catalog import (
    CANONICAL_VALID_BRANCH_PATTERN_MASK,
    canonical_branch_pattern_is_valid,
)
from .contract import NUM_TOPOLOGIES, SPHERE, full_component_bounds, topology_id_for
from .inference_proposals import (
    ContinuousProposalOutput,
    DiscreteProposalOutput,
    JointBranchScore,
    continuous_output_from_mapping,
    discrete_output_from_mapping,
)
from .production_bridge import (
    BranchRuntimeContext,
    ProductionBranchFactory,
    UserSearchSpace,
)
from .reference_bank import CompetingBranch


BOUNDS_LOCAL_INFERENCE_VERSION = "posterior_v8_bounds_local_proposal_v1"
LOCAL_SAMPLE_COORDINATE_SEMANTICS = MODEL_OUTPUT_COORDINATE_SEMANTICS


def _positive_integer(value: int, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1 or (maximum is not None and result > maximum):
        suffix = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be positive{suffix}")
    return result


def _finite_vector(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def codec_effective_varying_mask(codec: ProfiledBranchCodec) -> tuple[bool, ...]:
    """Return dimensions that actually vary after the authoritative codec.

    Semantic presence and effective variation differ for fixed GUI ranges.
    Decode/encode probes also respect coupled GUI-width and hard-core rules.
    """

    if not isinstance(codec, ProfiledBranchCodec):
        raise TypeError("codec must be ProfiledBranchCodec")
    midpoint = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
    result = np.zeros(UNIT_CUBE_DIMENSIONS, dtype=bool)
    for index in codec.active_indices:
        canonical = []
        for endpoint in (1.0e-12, 1.0 - 1.0e-12):
            probe = midpoint.copy()
            probe[index] = endpoint
            components, resolution = codec.decode(probe)
            canonical.append(codec.encode(components, resolution).unit_cube[index])
        result[index] = not np.isclose(canonical[0], canonical[1], rtol=0.0, atol=1e-12)
    return tuple(bool(value) for value in result)


@dataclass(frozen=True, kw_only=True)
class BoundsLocalBranchCondition:
    """One feasible hard branch plus its actual physical GUI bounds."""

    scored_branch: JointBranchScore
    context: BranchRuntimeContext
    bounds_embedding: tuple[float, ...]
    active_dimension_mask: tuple[bool, ...]
    local_varying_mask: tuple[bool, ...]
    input_bounds_coordinate_semantics: str = MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS
    output_coordinate_semantics: str = MODEL_OUTPUT_COORDINATE_SEMANTICS
    branch_catalog_version: str = MODEL_BRANCH_CATALOG_VERSION

    @classmethod
    def build(
        cls,
        scored_branch: JointBranchScore,
        context: BranchRuntimeContext,
    ) -> "BoundsLocalBranchCondition":
        if not isinstance(context, BranchRuntimeContext):
            raise TypeError("context must be BranchRuntimeContext")
        codec = context.user_bounds_codec
        return cls(
            scored_branch=scored_branch,
            context=context,
            bounds_embedding=bounds_embedding(
                codec.component_bounds,
                codec.resolution_bounds,
            ),
            active_dimension_mask=codec.active_mask,
            local_varying_mask=codec_effective_varying_mask(codec),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.scored_branch, JointBranchScore):
            raise TypeError("scored_branch must be JointBranchScore")
        if not isinstance(self.context, BranchRuntimeContext):
            raise TypeError("context must be BranchRuntimeContext")
        if self.context.branch != self.scored_branch.branch:
            raise ValueError("runtime context belongs to a different hard branch")
        if self.input_bounds_coordinate_semantics != MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS:
            raise ValueError("unsupported input-bounds coordinate semantics")
        if self.output_coordinate_semantics != MODEL_OUTPUT_COORDINATE_SEMANTICS:
            raise ValueError("unsupported local output coordinate semantics")
        if self.branch_catalog_version != MODEL_BRANCH_CATALOG_VERSION:
            raise ValueError("unsupported bounds-local branch catalog")
        if not canonical_branch_pattern_is_valid(
            self.context.branch.topology_id,
            self.context.branch.pattern_id,
        ):
            raise ValueError("bounds-local conditions require a canonical hard branch")
        embedded = _finite_vector(
            self.bounds_embedding,
            BOUNDS_EMBEDDING_DIM,
            "bounds_embedding",
        )
        expected_embedding = np.asarray(
            bounds_embedding(
                self.context.component_bounds,
                self.context.resolution_bounds,
            )
        )
        if not np.array_equal(embedded, expected_embedding):
            raise ValueError("bounds_embedding does not match the runtime user bounds")
        active = np.asarray(self.active_dimension_mask)
        varying = np.asarray(self.local_varying_mask)
        if active.shape != (UNIT_CUBE_DIMENSIONS,) or active.dtype.kind != "b":
            raise ValueError("active_dimension_mask must be a boolean 26-D vector")
        if varying.shape != (UNIT_CUBE_DIMENSIONS,) or varying.dtype.kind != "b":
            raise ValueError("local_varying_mask must be a boolean 26-D vector")
        if tuple(bool(value) for value in active) != self.context.user_bounds_codec.active_mask:
            raise ValueError("active_dimension_mask does not match the user codec")
        expected_varying = codec_effective_varying_mask(self.context.user_bounds_codec)
        if tuple(bool(value) for value in varying) != expected_varying:
            raise ValueError("local_varying_mask does not match codec-effective variation")
        if np.any(varying & ~active):
            raise ValueError("varying dimensions must be semantically active")
        object.__setattr__(self, "bounds_embedding", tuple(float(value) for value in embedded))
        object.__setattr__(
            self,
            "active_dimension_mask",
            tuple(bool(value) for value in active),
        )
        object.__setattr__(
            self,
            "local_varying_mask",
            tuple(bool(value) for value in varying),
        )

    @property
    def branch(self) -> CompetingBranch:
        return self.scored_branch.branch


def _curve_tensors(curve_inputs: Mapping[str, object]):
    if not isinstance(curve_inputs, Mapping):
        raise TypeError("curve_inputs must be a mapping")
    missing = {"x", "point_mask", "global_features"} - set(curve_inputs)
    if missing:
        raise ValueError(f"curve_inputs is missing {sorted(missing)!r}")
    x = np.asarray(curve_inputs["x"], dtype=np.float32)
    mask = np.asarray(curve_inputs["point_mask"])
    global_features = np.asarray(curve_inputs["global_features"], dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if global_features.ndim == 2 and global_features.shape[0] == 1:
        global_features = global_features[0]
    point_features = MODEL_FIXED_DIMENSIONS["point_feature_dim"]
    global_feature_count = MODEL_FIXED_DIMENSIONS["global_feature_dim"]
    if x.ndim != 2 or x.shape[1] != point_features:
        raise ValueError("x must have shape [points, 3] or [1, points, 3]")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if mask.shape != (x.shape[0],) or mask.dtype.kind != "b":
        raise ValueError("point_mask must be a boolean vector matching x")
    if not np.any(mask):
        raise ValueError("point_mask must contain at least one valid point")
    if global_features.shape != (global_feature_count,) or not np.all(np.isfinite(global_features)):
        raise ValueError("global_features must be a finite vector of length five")
    return x, mask.astype(bool, copy=False), global_features


def build_bounds_local_model_inputs(
    curve_inputs: Mapping[str, object],
    condition: BoundsLocalBranchCondition,
) -> dict[str, np.ndarray]:
    """Build the exact nine batch-one tensors frozen by the V3 contract."""

    if not isinstance(condition, BoundsLocalBranchCondition):
        raise TypeError("condition must be BoundsLocalBranchCondition")
    x, point_mask, global_features = _curve_tensors(curve_inputs)
    topology = np.zeros(NUM_TOPOLOGIES, dtype=np.float32)
    topology[condition.branch.topology_id] = 1.0
    d_present = np.zeros(4, dtype=np.float32)
    d_present[: len(condition.branch.d_present)] = condition.branch.d_present
    result = {
        "x": x[np.newaxis, ...],
        "point_mask": point_mask[np.newaxis, ...],
        "global_features": global_features[np.newaxis, ...],
        "branch_topology": topology[np.newaxis, ...],
        "branch_d_present": d_present[np.newaxis, ...],
        "branch_resolution_present": np.asarray(
            [[float(condition.branch.resolution_present)]], dtype=np.float32
        ),
        "bounds_embedding": np.asarray([condition.bounds_embedding], dtype=np.float32),
        "active_dimension_mask": np.asarray([condition.active_dimension_mask], dtype=np.float32),
        "varying_dimension_mask": np.asarray([condition.local_varying_mask], dtype=np.float32),
    }
    if tuple(result) != MODEL_INPUT_KEYS:
        raise RuntimeError("V3 input builder drifted from the frozen model contract")
    for name, value in result.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"model input {name!r} contains NaN/Inf")
    return result


class BoundsLocalProposalModelPort(Protocol):
    posterior_v8_model_version: str
    posterior_v8_input_bounds_coordinate_semantics: str
    posterior_v8_output_coordinate_semantics: str
    posterior_v8_branch_catalog_version: str
    posterior_v8_component_slots_version: str

    def predict_discrete(self, curve_inputs: Mapping[str, object]) -> DiscreteProposalOutput: ...

    def predict_continuous(
        self,
        curve_inputs: Mapping[str, object],
        condition: BoundsLocalBranchCondition,
    ) -> ContinuousProposalOutput: ...


def validate_bounds_local_model_port(model: object) -> None:
    if not callable(getattr(model, "predict_discrete", None)) or not callable(
        getattr(model, "predict_continuous", None)
    ):
        raise TypeError("model must implement BoundsLocalProposalModelPort")
    expected = {
        "posterior_v8_model_version": BOUNDS_PROPOSAL_MODEL_VERSION,
        "posterior_v8_input_bounds_coordinate_semantics": (MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS),
        "posterior_v8_output_coordinate_semantics": MODEL_OUTPUT_COORDINATE_SEMANTICS,
        "posterior_v8_branch_catalog_version": MODEL_BRANCH_CATALOG_VERSION,
        "posterior_v8_component_slots_version": MODEL_COMPONENT_SLOTS_VERSION,
    }
    for name, value in expected.items():
        if getattr(model, name, None) != value:
            raise ValueError(f"model has incompatible {name}")


def _default_probe_condition() -> BoundsLocalBranchCondition:
    factory = ProductionBranchFactory(
        UserSearchSpace.for_components((full_component_bounds(SPHERE, d_policy="absent"),))
    )
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    if context is None:  # pragma: no cover
        raise RuntimeError("could not construct the V3 probe branch")
    score = JointBranchScore(
        branch=branch,
        topology_rank=1,
        pattern_rank_within_topology=1,
        joint_rank=1,
        topology_log_score=0.0,
        conditional_pattern_log_score=0.0,
        joint_log_score=0.0,
    )
    return BoundsLocalBranchCondition.build(score, context)


@dataclass(frozen=True)
class CallableBoundsLocalProposalModel:
    """Strictly adapt a loaded/built V3 Keras-like callable."""

    predictor: Callable[[Mapping[str, np.ndarray]], Mapping[str, object]]
    posterior_v8_model_version: str
    posterior_v8_input_bounds_coordinate_semantics: str
    posterior_v8_output_coordinate_semantics: str
    posterior_v8_branch_catalog_version: str
    posterior_v8_component_slots_version: str
    probe_condition: BoundsLocalBranchCondition = field(default_factory=_default_probe_condition)

    @classmethod
    def from_loaded_model(cls, model: object) -> "CallableBoundsLocalProposalModel":
        return cls(
            predictor=model,
            posterior_v8_model_version=getattr(model, "posterior_v8_model_version", None),
            posterior_v8_input_bounds_coordinate_semantics=getattr(
                model,
                "posterior_v8_input_bounds_coordinate_semantics",
                None,
            ),
            posterior_v8_output_coordinate_semantics=getattr(
                model,
                "posterior_v8_output_coordinate_semantics",
                None,
            ),
            posterior_v8_branch_catalog_version=getattr(
                model,
                "posterior_v8_branch_catalog_version",
                None,
            ),
            posterior_v8_component_slots_version=getattr(
                model,
                "posterior_v8_component_slots_version",
                None,
            ),
        )

    def __post_init__(self) -> None:
        if not callable(self.predictor):
            raise TypeError("predictor must be callable")
        if not isinstance(self.probe_condition, BoundsLocalBranchCondition):
            raise TypeError("probe_condition must be BoundsLocalBranchCondition")
        validate_bounds_local_model_port(self)

    def _predict(self, inputs) -> Mapping[str, object]:
        output = self.predictor(inputs)
        if not isinstance(output, Mapping):
            raise TypeError("V3 predictor must return a mapping")
        if set(output) != set(MODEL_OUTPUT_KEYS):
            raise ValueError("V3 predictor output keys do not match the frozen contract")
        return output

    def predict_discrete(self, curve_inputs: Mapping[str, object]) -> DiscreteProposalOutput:
        # The V3 graph guarantees that discrete heads have no branch/bounds path;
        # this legal probe only supplies the required full-model input signature.
        output = self._predict(build_bounds_local_model_inputs(curve_inputs, self.probe_condition))
        return discrete_output_from_mapping(output)

    def predict_continuous(
        self,
        curve_inputs: Mapping[str, object],
        condition: BoundsLocalBranchCondition,
    ) -> ContinuousProposalOutput:
        output = self._predict(build_bounds_local_model_inputs(curve_inputs, condition))
        return continuous_output_from_mapping(output)


@dataclass(frozen=True, kw_only=True)
class LocalProposalSample:
    """A direct sample in the user-bounds codec; no global coordinate exists here."""

    condition: BoundsLocalBranchCondition
    mixture_index: int
    mixture_rank: int
    sample_index: int
    mixture_log_weight: float
    local_unit: tuple[float, ...]
    coordinate_semantics: str = LOCAL_SAMPLE_COORDINATE_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.condition, BoundsLocalBranchCondition):
            raise TypeError("condition must be BoundsLocalBranchCondition")
        if (
            isinstance(self.mixture_index, (bool, np.bool_))
            or not isinstance(self.mixture_index, Integral)
            or int(self.mixture_index) < 0
        ):
            raise ValueError("mixture_index must be a non-negative integer")
        object.__setattr__(self, "mixture_index", int(self.mixture_index))
        for name in ("mixture_rank", "sample_index"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        score = float(self.mixture_log_weight)
        if not np.isfinite(score):
            raise ValueError("mixture_log_weight must be finite")
        if self.coordinate_semantics != LOCAL_SAMPLE_COORDINATE_SEMANTICS:
            raise ValueError("unsupported local sample coordinate semantics")
        local = _finite_vector(self.local_unit, UNIT_CUBE_DIMENSIONS, "local_unit")
        if np.any(local < 0.0) or np.any(local > 1.0):
            raise ValueError("local_unit must lie in [0, 1]")
        varying = np.asarray(self.condition.local_varying_mask, dtype=bool)
        if np.any(local[~varying] != INACTIVE_UNIT_VALUE):
            raise ValueError("fixed and inactive local coordinates must equal 0.5")
        object.__setattr__(self, "mixture_log_weight", score)
        object.__setattr__(self, "local_unit", tuple(float(value) for value in local))

    @property
    def raw_model_log_score(self) -> float:
        return float(self.condition.scored_branch.joint_log_score + self.mixture_log_weight)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)) or not np.any(np.isfinite(values)):
        raise FloatingPointError("mixture logits have no finite support")
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted)))


def rank_canonical_joint_branches(
    output: DiscreteProposalOutput,
    *,
    topology_limit: int,
) -> tuple[JointBranchScore, ...]:
    """Rank only the 418 permutation-quotiented V3/V4 physical branches."""

    if not isinstance(output, DiscreteProposalOutput):
        raise TypeError("output must be DiscreteProposalOutput")
    topology_limit = _positive_integer(
        topology_limit,
        "topology_limit",
        maximum=NUM_TOPOLOGIES,
    )
    topology_scores = _log_softmax(output.topology_logits)
    topology_order = sorted(
        range(NUM_TOPOLOGIES),
        key=lambda index: (-topology_scores[index], index),
    )[:topology_limit]
    ranked = []
    for topology_rank, topology_id in enumerate(topology_order, 1):
        valid = np.asarray(
            CANONICAL_VALID_BRANCH_PATTERN_MASK[topology_id],
            dtype=bool,
        )
        pattern_ids = np.flatnonzero(valid)
        conditional = _log_softmax(output.branch_pattern_logits[topology_id, valid])
        pattern_order = sorted(
            range(pattern_ids.size),
            key=lambda index: (-conditional[index], int(pattern_ids[index])),
        )
        for pattern_rank, local_index in enumerate(pattern_order, 1):
            pattern_id = int(pattern_ids[local_index])
            topology_score = float(topology_scores[topology_id])
            pattern_score = float(conditional[local_index])
            ranked.append(
                JointBranchScore(
                    branch=CompetingBranch(
                        topology_id=topology_id,
                        pattern_id=pattern_id,
                    ),
                    topology_rank=topology_rank,
                    pattern_rank_within_topology=pattern_rank,
                    joint_rank=1,
                    topology_log_score=topology_score,
                    conditional_pattern_log_score=pattern_score,
                    joint_log_score=topology_score + pattern_score,
                )
            )
    ranked.sort(
        key=lambda item: (
            -item.joint_log_score,
            item.branch.topology_id,
            item.branch.pattern_id,
        )
    )
    if sum(sum(row) for row in CANONICAL_VALID_BRANCH_PATTERN_MASK) != (
        MODEL_PHYSICAL_BRANCH_COUNT
    ):  # pragma: no cover
        raise RuntimeError("V3/V4 branch catalog count drifted")
    return tuple(replace(item, joint_rank=rank) for rank, item in enumerate(ranked, 1))


def sample_local_logistic_normal_mixture(
    output: ContinuousProposalOutput,
    condition: BoundsLocalBranchCondition,
    *,
    mixture_limit: int,
    samples_per_mixture: int,
    seed: int,
) -> tuple[LocalProposalSample, ...]:
    """Sample the learned local distribution without clipping or affine remapping."""

    if not isinstance(output, ContinuousProposalOutput):
        raise TypeError("output must be ContinuousProposalOutput")
    if not isinstance(condition, BoundsLocalBranchCondition):
        raise TypeError("condition must be BoundsLocalBranchCondition")
    mixture_limit = _positive_integer(mixture_limit, "mixture_limit")
    samples_per_mixture = _positive_integer(samples_per_mixture, "samples_per_mixture")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    weights = _log_softmax(output.mixture_logits)
    order = sorted(range(weights.size), key=lambda index: (-weights[index], index))[
        : min(mixture_limit, weights.size)
    ]
    varying_indices = np.flatnonzero(condition.local_varying_mask)
    scales = np.exp(output.mixture_logscale)
    generated = []
    for mixture_rank, mixture_index in enumerate(order, 1):
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
            latent = rng.normal(
                loc=output.mixture_loc[mixture_index, varying_indices],
                scale=scales[mixture_index, varying_indices],
                size=(samples_per_mixture, varying_indices.size),
            )
            local[:, varying_indices] = expit(latent)
        if not np.all(np.isfinite(local)) or np.any(local < 0.0) or np.any(local > 1.0):
            raise FloatingPointError("local logistic-normal sample escaped [0, 1]")
        for sample_index in range(samples_per_mixture):
            generated.append(
                LocalProposalSample(
                    condition=condition,
                    mixture_index=mixture_index,
                    mixture_rank=mixture_rank,
                    sample_index=sample_index + 1,
                    mixture_log_weight=float(weights[mixture_index]),
                    local_unit=tuple(local[sample_index]),
                )
            )
    return tuple(generated)


__all__ = [
    "BOUNDS_LOCAL_INFERENCE_VERSION",
    "LOCAL_SAMPLE_COORDINATE_SEMANTICS",
    "BoundsLocalBranchCondition",
    "BoundsLocalProposalModelPort",
    "CallableBoundsLocalProposalModel",
    "LocalProposalSample",
    "build_bounds_local_model_inputs",
    "codec_effective_varying_mask",
    "rank_canonical_joint_branches",
    "sample_local_logistic_normal_mixture",
    "validate_bounds_local_model_port",
]
