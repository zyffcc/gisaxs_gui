"""Paired paper inference over independent RQMC scramble-replicate means.

The estimand is conditional on an exact, predeclared set of trained artifacts.
Each model seed is reported separately.  For a method contrast, complete-block
means are paired by independent scramble replicate and a Student-t interval is
formed across those paired means.  Individual Sobol points are dependent QMC
design points and this API intentionally offers no point-bootstrap operation.
Every method also binds an external inference-seed-set SHA.  A solver-only
baseline can be declared shared, in which case its replicate mean must be
identical across model-seed rows; the generic seed-specific policy explicitly
forbids interpreting those rows as independent baseline replications.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isclose, isfinite, sqrt
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .paper_rqmc_design_v5 import (
    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_RANDOMIZATION_UNIT,
    V5PaperRQMCDesign,
)
from .paper_endpoint_metrics import (
    REFERENCE_QUALIFICATION_STATUSES,
    REFERENCE_STATUS_CERTIFIED_NO_SOLUTION,
    REFERENCE_STATUS_QUALIFIED,
    REFERENCE_STATUS_UNRESOLVED,
)


V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA = "gisaxs.posterior_v8.frozen_training_artifact_set/v2"
V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION = (
    "predeclared_minimum_five_seed_artifacts_conditional_inference_v2"
)
V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE = "formal_minimum_five_predeclared_artifacts"
V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE = (
    "engineering_only_tiny_artifact_set_no_formal_inference"
)
V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS = 5
V5_PAPER_RQMC_REPLICATE_MEAN_SCHEMA = "gisaxs.posterior_v8.paper_rqmc_replicate_method_mean/v3"
V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA = "gisaxs.posterior_v8.paper_rqmc_paired_evaluator/v3"
V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION = (
    "paired_student_t_scramble_means_five_seed_set_qualification_contract_bound_v3"
)
V5_PAPER_RQMC_TRAINING_SCOPE = (
    "conditional_on_frozen_predeclared_training_artifact_set_report_each_seed_and_"
    "paired_frozen_seed_set_mean"
)
V5_PAPER_RQMC_SHARED_BASELINE_POLICY = "shared_across_model_seeds_require_identical"
V5_PAPER_RQMC_SEED_SPECIFIC_BASELINE_POLICY = "seed_specific_no_cross_model_seed_independence_claim"
V5_PAPER_RQMC_BASELINE_POLICIES = (
    V5_PAPER_RQMC_SHARED_BASELINE_POLICY,
    V5_PAPER_RQMC_SEED_SPECIFIC_BASELINE_POLICY,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256") from exc
    if len(raw) != 32:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _budgets(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("exact_forward_budgets must be a sequence")
    result = tuple(_integer(value, "exact_forward_budget", minimum=1) for value in values)
    if not result or any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError("exact_forward_budgets must be non-empty and strictly increasing")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _qualification_status_counts(
    values: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("qualification_status_counts must be a sequence")
    pairs = tuple(values)
    if not all(isinstance(value, tuple) and len(value) == 2 for value in pairs):
        raise TypeError("qualification_status_counts must contain (status, count) pairs")
    if len(pairs) != len(REFERENCE_QUALIFICATION_STATUSES):
        raise ValueError("qualification_status_counts must contain every status exactly once")
    lookup: dict[str, int] = {}
    for raw_status, raw_count in pairs:
        status = _text(raw_status, "qualification status")
        if status not in REFERENCE_QUALIFICATION_STATUSES or status in lookup:
            raise ValueError("qualification_status_counts must contain every status exactly once")
        lookup[status] = _integer(raw_count, f"qualification count {status}")
    if set(lookup) != set(REFERENCE_QUALIFICATION_STATUSES):
        raise ValueError("qualification_status_counts must contain every status exactly once")
    return tuple((status, lookup[status]) for status in REFERENCE_QUALIFICATION_STATUSES)


@dataclass(frozen=True, kw_only=True)
class V5FrozenTrainingArtifact:
    model_seed: int
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_seed", _integer(self.model_seed, "model_seed"))
        object.__setattr__(
            self, "artifact_sha256", _digest(self.artifact_sha256, "artifact_sha256")
        )

    @property
    def artifact_id(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "schema": V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA,
                    "model_seed": self.model_seed,
                    "artifact_sha256": self.artifact_sha256,
                }
            ).encode("utf-8")
        ).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "model_seed": self.model_seed,
            "artifact_sha256": self.artifact_sha256,
            "artifact_id": self.artifact_id,
        }


def _artifact_set_payload(
    artifacts: tuple[V5FrozenTrainingArtifact, ...],
    scope: str,
) -> dict[str, object]:
    formal = scope == V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE
    return {
        "schema": V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA,
        "version": V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION,
        "artifact_set_scope": scope,
        "formal_inference_allowed": formal,
        "minimum_formal_training_seed_count": V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS,
        "inference_scope": V5_PAPER_RQMC_TRAINING_SCOPE,
        "training_seeds_reported_separately": True,
        "frozen_seed_set_mean_reported": formal,
        "training_seed_population_interval_allowed": False,
        "artifacts": [value.payload() for value in artifacts],
    }


@dataclass(frozen=True)
class V5FrozenTrainingArtifactSet:
    artifacts: tuple[V5FrozenTrainingArtifact, ...]
    artifact_set_scope: str
    canonical_json: str
    sha256: str

    @classmethod
    def create(cls, artifacts: Sequence[V5FrozenTrainingArtifact]) -> "V5FrozenTrainingArtifactSet":
        """Create a formal set with at least five predeclared training artifacts."""

        return cls._create(
            artifacts,
            artifact_set_scope=V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE,
        )

    @classmethod
    def create_engineering_only(
        cls, artifacts: Sequence[V5FrozenTrainingArtifact]
    ) -> "V5FrozenTrainingArtifactSet":
        """Create a tiny explicitly non-formal set for engineering diagnostics."""

        return cls._create(
            artifacts,
            artifact_set_scope=V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE,
        )

    @classmethod
    def _create(
        cls,
        artifacts: Sequence[V5FrozenTrainingArtifact],
        *,
        artifact_set_scope: str,
    ) -> "V5FrozenTrainingArtifactSet":
        if isinstance(artifacts, (str, bytes)):
            raise TypeError("artifacts must be a sequence")
        values = tuple(artifacts)
        if not values or not all(isinstance(value, V5FrozenTrainingArtifact) for value in values):
            raise ValueError("artifacts must contain frozen training artifacts")
        values = tuple(sorted(values, key=lambda value: value.model_seed))
        if len({value.model_seed for value in values}) != len(values):
            raise ValueError("model seeds must be unique")
        if len({value.artifact_sha256 for value in values}) != len(values):
            raise ValueError("each model seed must bind a distinct artifact")
        scope = _text(artifact_set_scope, "artifact_set_scope")
        if scope == V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE:
            if len(values) < V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS:
                raise ValueError("formal inference requires at least five training artifacts")
        elif scope != V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE:
            raise ValueError("artifact_set_scope is unsupported")
        canonical = _canonical_json(_artifact_set_payload(values, scope))
        return cls(values, scope, canonical, sha256(canonical.encode("utf-8")).hexdigest())

    def __post_init__(self) -> None:
        values = tuple(self.artifacts)
        if not values or not all(isinstance(value, V5FrozenTrainingArtifact) for value in values):
            raise ValueError("artifacts must contain frozen training artifacts")
        if tuple(sorted(values, key=lambda value: value.model_seed)) != values:
            raise ValueError("training artifacts are not in canonical seed order")
        if len({value.model_seed for value in values}) != len(values):
            raise ValueError("model seeds must be unique")
        if len({value.artifact_sha256 for value in values}) != len(values):
            raise ValueError("each model seed must bind a distinct artifact")
        scope = _text(self.artifact_set_scope, "artifact_set_scope")
        if scope == V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE:
            if len(values) < V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS:
                raise ValueError("formal inference requires at least five training artifacts")
        elif scope != V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE:
            raise ValueError("artifact_set_scope is unsupported")
        canonical = _canonical_json(_artifact_set_payload(values, scope))
        if (
            self.canonical_json != canonical
            or self.sha256 != sha256(canonical.encode("utf-8")).hexdigest()
        ):
            raise ValueError("frozen training-artifact set does not reproduce")

    def to_json(self) -> str:
        payload = json.loads(self.canonical_json)
        payload["artifact_set_sha256"] = self.sha256
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "V5FrozenTrainingArtifactSet":
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid frozen training-artifact JSON") from exc
        expected = {
            "schema",
            "version",
            "artifact_set_scope",
            "formal_inference_allowed",
            "minimum_formal_training_seed_count",
            "inference_scope",
            "training_seeds_reported_separately",
            "frozen_seed_set_mean_reported",
            "training_seed_population_interval_allowed",
            "artifacts",
            "artifact_set_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("unsupported or incomplete training-artifact set")
        if (
            payload.get("schema") != V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA
            or payload.get("version") != V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION
            or payload.get("minimum_formal_training_seed_count")
            != V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS
            or payload.get("inference_scope") != V5_PAPER_RQMC_TRAINING_SCOPE
            or payload.get("training_seeds_reported_separately") is not True
            or payload.get("training_seed_population_interval_allowed") is not False
        ):
            raise ValueError("unsupported training-artifact inference claim")
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ValueError("artifacts must be an array")
        try:
            artifacts = tuple(
                V5FrozenTrainingArtifact(
                    model_seed=value["model_seed"], artifact_sha256=value["artifact_sha256"]
                )
                for value in raw_artifacts
            )
            scope = payload.get("artifact_set_scope")
            if scope == V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE:
                if (
                    payload.get("formal_inference_allowed") is not True
                    or payload.get("frozen_seed_set_mean_reported") is not True
                ):
                    raise ValueError("formal artifact-set claims are inconsistent")
                replay = cls.create(artifacts)
            elif scope == V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE:
                if (
                    payload.get("formal_inference_allowed") is not False
                    or payload.get("frozen_seed_set_mean_reported") is not False
                ):
                    raise ValueError("engineering artifact-set claims are inconsistent")
                replay = cls.create_engineering_only(artifacts)
            else:
                raise ValueError("artifact_set_scope is unsupported")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid training-artifact set") from exc
        supplied_hash = payload.pop("artifact_set_sha256")
        if payload != json.loads(replay.canonical_json) or supplied_hash != replay.sha256:
            raise ValueError("training-artifact payload/hash does not reproduce")
        return replay


@dataclass(frozen=True, kw_only=True)
class V5PaperRQMCReplicateMethodMean:
    design_sha256: str
    study_protocol_sha256: str
    endpoint_metric_contract_sha256: str
    split_name: str
    split_id: str
    replicate_index: int
    replicate_id: str
    block_id: str
    model_seed: int
    training_artifact_sha256: str
    method_id: str
    method_protocol_sha256: str
    inference_seed_set_sha256: str
    preselected_cohort_sha256: str
    qualification_protocol_sha256: str
    qualification_artifact_sha256: str
    qualification_status_counts: tuple[tuple[str, int], ...]
    qualified_endpoint_numerator: float
    qualified_endpoint_denominator: int
    full_cohort_zero_imputed_lower_bound: float
    full_cohort_unresolved_universal_upper_bound: float
    exact_forward_budgets: tuple[int, ...]
    evaluated_point_count: int
    replicate_mean: float

    def __post_init__(self) -> None:
        for name in (
            "design_sha256",
            "study_protocol_sha256",
            "endpoint_metric_contract_sha256",
            "split_id",
            "replicate_id",
            "block_id",
            "training_artifact_sha256",
            "method_protocol_sha256",
            "inference_seed_set_sha256",
            "preselected_cohort_sha256",
            "qualification_protocol_sha256",
            "qualification_artifact_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "split_name", _text(self.split_name, "split_name"))
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        object.__setattr__(
            self, "replicate_index", _integer(self.replicate_index, "replicate_index")
        )
        object.__setattr__(self, "model_seed", _integer(self.model_seed, "model_seed"))
        object.__setattr__(
            self,
            "evaluated_point_count",
            _integer(self.evaluated_point_count, "evaluated_point_count", minimum=1),
        )
        object.__setattr__(self, "exact_forward_budgets", _budgets(self.exact_forward_budgets))
        status_counts = _qualification_status_counts(self.qualification_status_counts)
        object.__setattr__(self, "qualification_status_counts", status_counts)
        counts = dict(status_counts)
        total = sum(counts.values())
        if total != self.evaluated_point_count:
            raise ValueError("qualification status counts must equal evaluated_point_count")
        denominator = _integer(
            self.qualified_endpoint_denominator,
            "qualified_endpoint_denominator",
            minimum=1,
        )
        if denominator != counts[REFERENCE_STATUS_QUALIFIED]:
            raise ValueError("qualified endpoint denominator must equal the qualified status count")
        numerator = _finite(self.qualified_endpoint_numerator, "qualified_endpoint_numerator")
        if not 0.0 <= numerator <= denominator:
            raise ValueError("qualified_endpoint_numerator must be in [0, denominator]")
        mean = _unit_interval(self.replicate_mean, "replicate_mean")
        if not isclose(mean, numerator / denominator, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(
                "replicate_mean is inconsistent with its qualified numerator/denominator"
            )
        lower = _unit_interval(
            self.full_cohort_zero_imputed_lower_bound,
            "full_cohort_zero_imputed_lower_bound",
        )
        upper = _unit_interval(
            self.full_cohort_unresolved_universal_upper_bound,
            "full_cohort_unresolved_universal_upper_bound",
        )
        expected_lower = numerator / total
        expected_upper = (numerator + counts[REFERENCE_STATUS_UNRESOLVED]) / total
        if not isclose(lower, expected_lower, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError("zero-imputed lower bound is inconsistent with qualification counts")
        if not isclose(upper, expected_upper, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(
                "unresolved universal upper bound is inconsistent with qualification counts"
            )
        object.__setattr__(self, "qualified_endpoint_numerator", numerator)
        object.__setattr__(self, "qualified_endpoint_denominator", denominator)
        object.__setattr__(self, "full_cohort_zero_imputed_lower_bound", lower)
        object.__setattr__(self, "full_cohort_unresolved_universal_upper_bound", upper)
        object.__setattr__(self, "replicate_mean", mean)

    def payload(self) -> dict[str, object]:
        return {
            "schema": V5_PAPER_RQMC_REPLICATE_MEAN_SCHEMA,
            "aggregation_unit": V5_PAPER_RQMC_RANDOMIZATION_UNIT,
            **{
                name: list(value)
                if name in {"exact_forward_budgets", "qualification_status_counts"}
                else value
                for name, value in self.__dict__.items()
            },
        }


@dataclass(frozen=True)
class V5PaperRQMCPairedReplicateDifference:
    replicate_index: int
    replicate_id: str
    block_id: str
    baseline_mean: float
    comparison_mean: float
    comparison_minus_baseline: float
    baseline_full_cohort_zero_imputed_lower_bound: float
    comparison_full_cohort_zero_imputed_lower_bound: float
    baseline_full_cohort_unresolved_universal_upper_bound: float
    comparison_full_cohort_unresolved_universal_upper_bound: float


@dataclass(frozen=True)
class V5PaperRQMCQualificationBlockBinding:
    replicate_index: int
    replicate_id: str
    block_id: str
    preselected_cohort_sha256: str
    qualification_protocol_sha256: str
    qualification_artifact_sha256: str
    qualification_status_counts: tuple[tuple[str, int], ...]
    preselected_recipe_count: int
    qualified_recipe_count: int
    unresolved_recipe_count: int
    certified_no_solution_recipe_count: int


@dataclass(frozen=True)
class V5PaperRQMCPairedSeedSummary:
    model_seed: int
    training_artifact_sha256: str
    replicate_count: int
    degrees_of_freedom: int
    baseline_replicate_mean: float
    comparison_replicate_mean: float
    comparison_minus_baseline: float
    standard_error: float
    confidence_interval: tuple[float, float]
    baseline_full_cohort_zero_imputed_lower_mean: float
    comparison_full_cohort_zero_imputed_lower_mean: float
    baseline_full_cohort_unresolved_universal_upper_mean: float
    comparison_full_cohort_unresolved_universal_upper_mean: float
    paired_replicates: tuple[V5PaperRQMCPairedReplicateDifference, ...]


@dataclass(frozen=True)
class V5PaperRQMCPairedFrozenSeedSetSummary:
    """RQMC inference for the mean over the exact frozen model-artifact set."""

    training_seed_count: int
    replicate_count: int
    degrees_of_freedom: int
    baseline_replicate_mean: float
    comparison_replicate_mean: float
    comparison_minus_baseline: float
    standard_error: float
    confidence_interval: tuple[float, float]
    baseline_full_cohort_zero_imputed_lower_mean: float
    comparison_full_cohort_zero_imputed_lower_mean: float
    baseline_full_cohort_unresolved_universal_upper_mean: float
    comparison_full_cohort_unresolved_universal_upper_mean: float
    paired_replicates: tuple[V5PaperRQMCPairedReplicateDifference, ...]


@dataclass(frozen=True)
class V5PaperRQMCPairedInferenceSummary:
    schema: str
    version: str
    design_sha256: str
    artifact_set_sha256: str
    input_sha256: str
    metric_id: str
    endpoint_metric_contract_sha256: str
    study_protocol_sha256: str
    split_name: str
    split_id: str
    baseline_method_id: str
    baseline_method_protocol_sha256: str
    baseline_inference_seed_set_sha256: str
    baseline_model_seed_policy: str
    comparison_method_id: str
    comparison_method_protocol_sha256: str
    comparison_inference_seed_set_sha256: str
    exact_forward_budgets: tuple[int, ...]
    confidence_level: float
    randomization_unit: str
    point_bootstrap_used: bool
    training_scope: str
    qualification_bindings: tuple[V5PaperRQMCQualificationBlockBinding, ...]
    seed_summaries: tuple[V5PaperRQMCPairedSeedSummary, ...]
    frozen_seed_set_summary: V5PaperRQMCPairedFrozenSeedSetSummary


def _paired_summary_values(
    pairs: Sequence[V5PaperRQMCPairedReplicateDifference],
    *,
    confidence_level: float,
) -> dict[str, object]:
    values = tuple(pairs)
    differences = tuple(value.comparison_minus_baseline for value in values)
    count = len(differences)
    mean_difference = fsum(differences) / count
    variance = fsum((value - mean_difference) ** 2 for value in differences) / (count - 1)
    standard_error = sqrt(variance / count)
    critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, df=count - 1))
    half_width = critical * standard_error
    return {
        "replicate_count": count,
        "degrees_of_freedom": count - 1,
        "baseline_replicate_mean": fsum(value.baseline_mean for value in values) / count,
        "comparison_replicate_mean": fsum(value.comparison_mean for value in values) / count,
        "comparison_minus_baseline": mean_difference,
        "standard_error": standard_error,
        "confidence_interval": (mean_difference - half_width, mean_difference + half_width),
        "baseline_full_cohort_zero_imputed_lower_mean": fsum(
            value.baseline_full_cohort_zero_imputed_lower_bound for value in values
        )
        / count,
        "comparison_full_cohort_zero_imputed_lower_mean": fsum(
            value.comparison_full_cohort_zero_imputed_lower_bound for value in values
        )
        / count,
        "baseline_full_cohort_unresolved_universal_upper_mean": fsum(
            value.baseline_full_cohort_unresolved_universal_upper_bound for value in values
        )
        / count,
        "comparison_full_cohort_unresolved_universal_upper_mean": fsum(
            value.comparison_full_cohort_unresolved_universal_upper_bound for value in values
        )
        / count,
        "paired_replicates": values,
    }


def summarize_v5_paired_rqmc_replicates(
    design: V5PaperRQMCDesign,
    artifacts: V5FrozenTrainingArtifactSet,
    cells: Sequence[V5PaperRQMCReplicateMethodMean],
    *,
    metric_id: str,
    split_name: str,
    baseline_method_id: str,
    comparison_method_id: str,
    baseline_model_seed_policy: str,
    confidence_level: float = 0.95,
) -> V5PaperRQMCPairedInferenceSummary:
    """Return seed-specific and frozen-seed-set paired RQMC intervals."""

    if not isinstance(design, V5PaperRQMCDesign) or not isinstance(
        artifacts, V5FrozenTrainingArtifactSet
    ):
        raise TypeError("design/artifacts have invalid types")
    if (
        artifacts.artifact_set_scope != V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE
        or len(artifacts.artifacts) < V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS
    ):
        raise ValueError("formal RQMC inference requires a formal minimum-five artifact set")
    metric = _text(metric_id, "metric_id")
    split = _text(split_name, "split_name")
    baseline = _text(baseline_method_id, "baseline_method_id")
    comparison = _text(comparison_method_id, "comparison_method_id")
    baseline_policy = _text(baseline_model_seed_policy, "baseline_model_seed_policy")
    if baseline == comparison:
        raise ValueError("paired methods must be distinct")
    if baseline_policy not in V5_PAPER_RQMC_BASELINE_POLICIES:
        raise ValueError("baseline_model_seed_policy is unsupported")
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a real number")
    level = float(confidence_level)
    if not isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    rows = tuple(cells)
    if not rows or not all(isinstance(value, V5PaperRQMCReplicateMethodMean) for value in rows):
        raise ValueError("cells must contain complete replicate-method means")
    if {value.method_id for value in rows} != {baseline, comparison}:
        raise ValueError("cells must contain exactly the two paired methods")

    blocks = design.blocks_for_split(split)
    if len(blocks) < V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES:
        raise ValueError("too few independent scramble replicates for formal inference")
    blocks_by_index = {value.replicate_index: value for value in blocks}
    artifacts_by_seed = {value.model_seed: value for value in artifacts.artifacts}
    lookup: dict[tuple[int, str, int], V5PaperRQMCReplicateMethodMean] = {}
    budgets = rows[0].exact_forward_budgets
    shared_contract_binding = (
        rows[0].study_protocol_sha256,
        rows[0].endpoint_metric_contract_sha256,
    )
    method_bindings: dict[str, tuple[str, str]] = {}
    qualification_by_replicate: dict[int, tuple[object, ...]] = {}
    for row in rows:
        if row.design_sha256 != design.sha256 or row.split_name != split:
            raise ValueError("replicate cell escaped the selected design/split")
        block = blocks_by_index.get(row.replicate_index)
        if block is None or (
            row.split_id,
            row.replicate_id,
            row.block_id,
            row.evaluated_point_count,
        ) != (block.split_id, block.replicate_id, block.block_id, block.count):
            raise ValueError("replicate cell is misaligned with its frozen complete block")
        artifact = artifacts_by_seed.get(row.model_seed)
        if artifact is None or row.training_artifact_sha256 != artifact.artifact_sha256:
            raise ValueError("replicate cell escaped the frozen training-artifact set")
        if row.exact_forward_budgets != budgets:
            raise ValueError("paired methods have an exact-forward budget mismatch")
        if (
            row.study_protocol_sha256,
            row.endpoint_metric_contract_sha256,
        ) != shared_contract_binding:
            raise ValueError("study protocol or endpoint metric contract changed across cells")
        binding = (row.method_protocol_sha256, row.inference_seed_set_sha256)
        previous_binding = method_bindings.setdefault(row.method_id, binding)
        if previous_binding != binding:
            raise ValueError("method protocol or inference-seed set changed across paired cells")
        qualification_binding = (
            row.preselected_cohort_sha256,
            row.qualification_protocol_sha256,
            row.qualification_artifact_sha256,
            row.qualification_status_counts,
            row.qualified_endpoint_denominator,
            row.evaluated_point_count,
        )
        previous_qualification = qualification_by_replicate.setdefault(
            row.replicate_index, qualification_binding
        )
        if previous_qualification != qualification_binding:
            raise ValueError(
                "qualification cohort, artifact, statuses, or denominator changed "
                "across paired method/model-seed cells"
            )
        key = (row.model_seed, row.method_id, row.replicate_index)
        if key in lookup:
            raise ValueError("duplicated model-seed/method/replicate cell")
        lookup[key] = row

    expected = {
        (artifact.model_seed, method, block.replicate_index)
        for artifact in artifacts.artifacts
        for method in (baseline, comparison)
        for block in blocks
    }
    if set(lookup) != expected:
        raise ValueError("missing or unexpected model-seed/method/replicate cells")
    qualification_bindings = []
    for block in blocks:
        raw = qualification_by_replicate[block.replicate_index]
        counts = dict(raw[3])
        qualification_bindings.append(
            V5PaperRQMCQualificationBlockBinding(
                replicate_index=block.replicate_index,
                replicate_id=block.replicate_id,
                block_id=block.block_id,
                preselected_cohort_sha256=raw[0],
                qualification_protocol_sha256=raw[1],
                qualification_artifact_sha256=raw[2],
                qualification_status_counts=raw[3],
                preselected_recipe_count=raw[5],
                qualified_recipe_count=counts[REFERENCE_STATUS_QUALIFIED],
                unresolved_recipe_count=counts[REFERENCE_STATUS_UNRESOLVED],
                certified_no_solution_recipe_count=counts[REFERENCE_STATUS_CERTIFIED_NO_SOLUTION],
            )
        )
    if baseline_policy == V5_PAPER_RQMC_SHARED_BASELINE_POLICY:
        for block in blocks:
            means = {
                lookup[(artifact.model_seed, baseline, block.replicate_index)].replicate_mean
                for artifact in artifacts.artifacts
            }
            if len(means) != 1:
                raise ValueError(
                    "shared solver baseline replicate means must be identical across model seeds"
                )

    seed_summaries = []
    for artifact in artifacts.artifacts:
        pairs = []
        for block in blocks:
            left = lookup[(artifact.model_seed, baseline, block.replicate_index)]
            right = lookup[(artifact.model_seed, comparison, block.replicate_index)]
            difference = right.replicate_mean - left.replicate_mean
            pairs.append(
                V5PaperRQMCPairedReplicateDifference(
                    block.replicate_index,
                    block.replicate_id,
                    block.block_id,
                    left.replicate_mean,
                    right.replicate_mean,
                    difference,
                    left.full_cohort_zero_imputed_lower_bound,
                    right.full_cohort_zero_imputed_lower_bound,
                    left.full_cohort_unresolved_universal_upper_bound,
                    right.full_cohort_unresolved_universal_upper_bound,
                )
            )
        seed_summaries.append(
            V5PaperRQMCPairedSeedSummary(
                model_seed=artifact.model_seed,
                training_artifact_sha256=artifact.artifact_sha256,
                **_paired_summary_values(pairs, confidence_level=level),
            )
        )

    frozen_seed_pairs = []
    for block in blocks:
        left_rows = tuple(
            lookup[(artifact.model_seed, baseline, block.replicate_index)]
            for artifact in artifacts.artifacts
        )
        right_rows = tuple(
            lookup[(artifact.model_seed, comparison, block.replicate_index)]
            for artifact in artifacts.artifacts
        )
        seed_count = len(artifacts.artifacts)
        baseline_mean = fsum(value.replicate_mean for value in left_rows) / seed_count
        comparison_mean = fsum(value.replicate_mean for value in right_rows) / seed_count
        frozen_seed_pairs.append(
            V5PaperRQMCPairedReplicateDifference(
                block.replicate_index,
                block.replicate_id,
                block.block_id,
                baseline_mean,
                comparison_mean,
                comparison_mean - baseline_mean,
                fsum(value.full_cohort_zero_imputed_lower_bound for value in left_rows)
                / seed_count,
                fsum(value.full_cohort_zero_imputed_lower_bound for value in right_rows)
                / seed_count,
                fsum(value.full_cohort_unresolved_universal_upper_bound for value in left_rows)
                / seed_count,
                fsum(value.full_cohort_unresolved_universal_upper_bound for value in right_rows)
                / seed_count,
            )
        )
    frozen_seed_set_summary = V5PaperRQMCPairedFrozenSeedSetSummary(
        training_seed_count=len(artifacts.artifacts),
        **_paired_summary_values(frozen_seed_pairs, confidence_level=level),
    )

    sorted_rows = sorted(
        (value.payload() for value in rows),
        key=lambda value: (
            value["model_seed"],
            value["method_id"],
            value["replicate_index"],
        ),
    )
    input_sha = sha256(
        _canonical_json(
            {
                "schema": V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA,
                "version": V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION,
                "design_sha256": design.sha256,
                "artifact_set_sha256": artifacts.sha256,
                "metric_id": metric,
                "study_protocol_sha256": shared_contract_binding[0],
                "endpoint_metric_contract_sha256": shared_contract_binding[1],
                "split_name": split,
                "baseline_method_id": baseline,
                "comparison_method_id": comparison,
                "baseline_model_seed_policy": baseline_policy,
                "confidence_level": level,
                "cells": sorted_rows,
            }
        ).encode("utf-8")
    ).hexdigest()
    return V5PaperRQMCPairedInferenceSummary(
        schema=V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA,
        version=V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION,
        design_sha256=design.sha256,
        artifact_set_sha256=artifacts.sha256,
        input_sha256=input_sha,
        metric_id=metric,
        study_protocol_sha256=shared_contract_binding[0],
        endpoint_metric_contract_sha256=shared_contract_binding[1],
        split_name=split,
        split_id=blocks[0].split_id,
        baseline_method_id=baseline,
        baseline_method_protocol_sha256=method_bindings[baseline][0],
        baseline_inference_seed_set_sha256=method_bindings[baseline][1],
        baseline_model_seed_policy=baseline_policy,
        comparison_method_id=comparison,
        comparison_method_protocol_sha256=method_bindings[comparison][0],
        comparison_inference_seed_set_sha256=method_bindings[comparison][1],
        exact_forward_budgets=budgets,
        confidence_level=level,
        randomization_unit=V5_PAPER_RQMC_RANDOMIZATION_UNIT,
        point_bootstrap_used=False,
        training_scope=V5_PAPER_RQMC_TRAINING_SCOPE,
        qualification_bindings=tuple(qualification_bindings),
        seed_summaries=tuple(seed_summaries),
        frozen_seed_set_summary=frozen_seed_set_summary,
    )


__all__ = [
    "V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE",
    "V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE",
    "V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA",
    "V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION",
    "V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS",
    "V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA",
    "V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION",
    "V5_PAPER_RQMC_REPLICATE_MEAN_SCHEMA",
    "V5_PAPER_RQMC_BASELINE_POLICIES",
    "V5_PAPER_RQMC_SEED_SPECIFIC_BASELINE_POLICY",
    "V5_PAPER_RQMC_SHARED_BASELINE_POLICY",
    "V5_PAPER_RQMC_TRAINING_SCOPE",
    "V5FrozenTrainingArtifact",
    "V5FrozenTrainingArtifactSet",
    "V5PaperRQMCPairedFrozenSeedSetSummary",
    "V5PaperRQMCPairedInferenceSummary",
    "V5PaperRQMCQualificationBlockBinding",
    "V5PaperRQMCPairedReplicateDifference",
    "V5PaperRQMCPairedSeedSummary",
    "V5PaperRQMCReplicateMethodMean",
    "summarize_v5_paired_rqmc_replicates",
]
