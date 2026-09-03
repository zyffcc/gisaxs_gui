"""Framework-neutral metrics for the frozen Posterior V8 paper endpoint.

The primary endpoint is a recipe-macro average of per-recipe recall AUCs.  AUC
is integrated over log2 exact-forward budget so every budget doubling receives
equal weight.  Candidate/reference matching remains the evaluator's concern;
this module owns only the arithmetic and its fail-closed validation.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import fsum, isfinite, log2
from numbers import Integral, Real


PAPER_ENDPOINT_METRICS_VERSION = (
    "posterior_v8_typed_qualification_conditional_actual_emission_representative_budget_auc/v4"
)
REFERENCE_QUALIFICATION_SCHEMA = "gisaxs.posterior_v8.reference_qualification/v1"
REFERENCE_QUALIFICATION_VERSION = (
    "frozen_preselected_cohort_typed_status_universal_unresolved_bounds_v1"
)
REFERENCE_STATUS_QUALIFIED = "qualified"
REFERENCE_STATUS_UNRESOLVED = "unresolved"
REFERENCE_STATUS_CERTIFIED_NO_SOLUTION = "certified_no_solution"
REFERENCE_QUALIFICATION_STATUSES = (
    REFERENCE_STATUS_QUALIFIED,
    REFERENCE_STATUS_UNRESOLVED,
    REFERENCE_STATUS_CERTIFIED_NO_SOLUTION,
)
OUTPUT_CAPS = (1, 4, 8, 16)
PRIMARY_OUTPUT_CAP = 16
EXACT_FORWARD_BUDGETS = (256, 512, 1024, 2048, 4096)
PARAMETER_CLUSTER_DISTANCE_MAX = 0.08
REFERENCE_MATCH_DISTANCE_MAX = 0.10
CURVE_EQUIVALENCE_LOG_RMSE_MAX = 0.02
THRESHOLD_SENSITIVITY_MULTIPLIERS = (0.5, 1.0, 2.0)
PRIMARY_ENDPOINT_NAME = (
    "conditional_on_frozen_reference_search_qualification_reference_discovered_"
    "complete_linkage_diameter_cluster_compatible_parameter_actual_emitted_representative_"
    "recall_at_n16_log2_exact_forward_budget_auc"
)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


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


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_budgets(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("budgets must be a sequence of integers")
    result = tuple(
        _integer(value, f"budgets[{index}]", minimum=1) for index, value in enumerate(values)
    )
    if len(result) < 2:
        raise ValueError("budgets must contain at least two values")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("budgets must be strictly increasing")
    return result


def _recall(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _validated_recall_curve(
    recall_by_budget: Mapping[int, float], budgets: tuple[int, ...]
) -> tuple[float, ...]:
    if not isinstance(recall_by_budget, Mapping):
        raise TypeError("recall_by_budget must be a mapping")
    expected = set(budgets)
    raw_keys = tuple(recall_by_budget)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_keys):
        raise TypeError("recall_by_budget keys must be integer exact-forward budgets")
    actual = {int(value) for value in raw_keys}
    if actual != expected:
        raise ValueError(
            "recall_by_budget keys must exactly equal the frozen budgets; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return tuple(
        _recall(recall_by_budget[budget], f"recall_by_budget[{budget}]") for budget in budgets
    )


def normalized_log2_budget_auc(
    recall_by_budget: Mapping[int, float],
    *,
    budgets: Sequence[int] = EXACT_FORWARD_BUDGETS,
) -> float:
    """Return normalized trapezoidal recall AUC over ``log2(budget)``.

    The result stays in ``[0, 1]``.  For the frozen doubling schedule this is
    ``(.5*r256 + r512 + r1024 + r2048 + .5*r4096) / 4``.
    """

    budget_values = _validated_budgets(budgets)
    recalls = _validated_recall_curve(recall_by_budget, budget_values)
    log_budgets = tuple(log2(value) for value in budget_values)
    span = log_budgets[-1] - log_budgets[0]
    area = fsum(
        0.5 * (left_recall + right_recall) * (right_budget - left_budget)
        for left_recall, right_recall, left_budget, right_budget in zip(
            recalls,
            recalls[1:],
            log_budgets,
            log_budgets[1:],
        )
    )
    return float(area / span)


def completed_candidate_prefix_lengths(
    cumulative_exact_forward_calls: Sequence[int],
    *,
    budgets: Sequence[int] = EXACT_FORWARD_BUDGETS,
) -> tuple[int, ...]:
    """Count candidates fully completed and verified within each budget.

    A candidate whose cumulative completion count exceeds a budget is excluded
    from that budget prefix.  Empty candidate sequences are valid and produce
    an all-zero prefix.
    """

    budget_values = _validated_budgets(budgets)
    if isinstance(cumulative_exact_forward_calls, (str, bytes)):
        raise TypeError("cumulative_exact_forward_calls must be an integer sequence")
    calls = tuple(
        _integer(value, f"cumulative_exact_forward_calls[{index}]", minimum=1)
        for index, value in enumerate(cumulative_exact_forward_calls)
    )
    if any(right < left for left, right in zip(calls, calls[1:])):
        raise ValueError("cumulative exact-forward calls must be non-decreasing")
    return tuple(bisect_right(calls, budget) for budget in budget_values)


@dataclass(frozen=True)
class ReferenceRecall:
    """One qualified-reference recall value and its output-cap ceiling."""

    matched_reference_count: int
    reference_count: int
    output_cap: int
    recall: float
    ceiling: float


def reference_recall(
    matched_reference_count: int,
    reference_count: int,
    *,
    output_cap: int = PRIMARY_OUTPUT_CAP,
) -> ReferenceRecall:
    """Calculate recall without renormalizing when ``M_ref > output_cap``.

    Unqualified, unsaturated, or empty uncertified reference sets must not call
    this function; their handling is frozen separately in the study protocol.
    """

    matched = _integer(matched_reference_count, "matched_reference_count", minimum=0)
    references = _integer(reference_count, "reference_count", minimum=1)
    cap = _integer(output_cap, "output_cap", minimum=1)
    if matched > min(references, cap):
        raise ValueError("matched_reference_count exceeds the output-cap/reference ceiling")
    return ReferenceRecall(
        matched_reference_count=matched,
        reference_count=references,
        output_cap=cap,
        recall=matched / references,
        ceiling=min(cap, references) / references,
    )


def zero_recall_curve(*, budgets: Sequence[int] = EXACT_FORWARD_BUDGETS) -> dict[int, float]:
    """Return the required all-zero curve for a method crash or search failure."""

    return {budget: 0.0 for budget in _validated_budgets(budgets)}


@dataclass(frozen=True)
class SeedEndpointSummary:
    seed_id: str
    recipe_aucs: tuple[tuple[str, float], ...]
    recipe_macro_auc: float


@dataclass(frozen=True)
class SeededEndpointSummary:
    seed_summaries: tuple[SeedEndpointSummary, ...]
    recipe_count: int
    seed_count: int
    seed_mean_recipe_macro_auc: float


@dataclass(frozen=True)
class FrozenPreselectedReferenceCohort:
    """Content-bound recipe inventory frozen before reference search."""

    cohort_id: str
    preselection_protocol_sha256: str
    source_artifact_sha256: str
    recipe_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_id", _identifier(self.cohort_id, "cohort_id"))
        for name in ("preselection_protocol_sha256", "source_artifact_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if isinstance(self.recipe_ids, (str, bytes)):
            raise TypeError("recipe_ids must be a sequence")
        recipes = tuple(_identifier(value, "recipe_id") for value in self.recipe_ids)
        if not recipes:
            raise ValueError("the frozen preselected cohort must contain recipes")
        if len(recipes) != len(set(recipes)):
            raise ValueError("the frozen preselected cohort contains duplicate recipe IDs")
        object.__setattr__(self, "recipe_ids", tuple(sorted(recipes)))

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": REFERENCE_QUALIFICATION_SCHEMA,
            "version": REFERENCE_QUALIFICATION_VERSION,
            "kind": "frozen_preselected_reference_cohort",
            "cohort_id": self.cohort_id,
            "preselection_protocol_sha256": self.preselection_protocol_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "recipe_ids": list(self.recipe_ids),
        }

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class ReferenceQualificationRecord:
    """Method-independent frozen reference-search status for one recipe."""

    recipe_id: str
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe_id", _identifier(self.recipe_id, "recipe_id"))
        status = _identifier(self.status, "qualification status")
        if status not in REFERENCE_QUALIFICATION_STATUSES:
            raise ValueError("qualification status is unsupported")
        object.__setattr__(self, "status", status)


@dataclass(frozen=True)
class FrozenReferenceQualificationArtifact:
    """Complete typed qualification result bound to one preselected cohort."""

    cohort: FrozenPreselectedReferenceCohort
    qualification_protocol_sha256: str
    source_artifact_sha256: str
    records: tuple[ReferenceQualificationRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cohort, FrozenPreselectedReferenceCohort):
            raise TypeError("cohort must be a FrozenPreselectedReferenceCohort")
        for name in ("qualification_protocol_sha256", "source_artifact_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if isinstance(self.records, (str, bytes)):
            raise TypeError("records must be a sequence")
        records = tuple(self.records)
        if not all(isinstance(value, ReferenceQualificationRecord) for value in records):
            raise TypeError("records must contain ReferenceQualificationRecord values")
        recipe_ids = tuple(value.recipe_id for value in records)
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("qualification records contain duplicate recipe IDs")
        expected = set(self.cohort.recipe_ids)
        actual = set(recipe_ids)
        if actual != expected:
            raise ValueError(
                "qualification records must contain every frozen cohort recipe exactly once; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        object.__setattr__(self, "records", tuple(sorted(records)))

    @property
    def preselected_cohort_sha256(self) -> str:
        return self.cohort.sha256

    @property
    def status_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (status, sum(value.status == status for value in self.records))
            for status in REFERENCE_QUALIFICATION_STATUSES
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": REFERENCE_QUALIFICATION_SCHEMA,
            "version": REFERENCE_QUALIFICATION_VERSION,
            "kind": "frozen_reference_qualification_artifact",
            "preselected_cohort_sha256": self.preselected_cohort_sha256,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "records": [asdict(value) for value in self.records],
        }

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReferenceQualificationSummary:
    """Conditional endpoint plus valid universal full-cohort sensitivity bounds.

    Unresolved recipes may have any endpoint value in ``[0, 1]``.  Imputing one
    gives a valid universal upper bound, not a sharp endpoint-specific bound.
    Independently certified no-solution recipes are counted separately and
    contribute only to their separate abstention endpoint.
    """

    preselected_cohort_sha256: str
    qualification_protocol_sha256: str
    qualification_artifact_sha256: str
    qualification_source_artifact_sha256: str
    status_counts: tuple[tuple[str, int], ...]
    preselected_recipe_count: int
    qualified_recipe_count: int
    unresolved_recipe_count: int
    certified_no_solution_recipe_count: int
    qualification_fraction: float
    qualified_endpoint_numerator: float
    qualified_endpoint_denominator: int
    conditional_qualified_mean: float
    full_cohort_zero_imputed_lower_bound: float
    full_cohort_unresolved_universal_upper_bound: float


def summarize_reference_qualification(
    qualified_endpoint_values: Mapping[str, float],
    *,
    cohort: FrozenPreselectedReferenceCohort,
    qualification: FrozenReferenceQualificationArtifact,
) -> ReferenceQualificationSummary:
    """Join exact qualified recipe IDs and report conditional/full-cohort values.

    The input mapping must contain exactly one endpoint value for each
    ``qualified`` record.  Unresolved recipes contribute zero to the lower and
    one to the universal upper sensitivity bound.  Certified no-solution cases
    remain explicit and are not silently relabelled unresolved.
    """

    if not isinstance(cohort, FrozenPreselectedReferenceCohort):
        raise TypeError("cohort must be a FrozenPreselectedReferenceCohort")
    if not isinstance(qualification, FrozenReferenceQualificationArtifact):
        raise TypeError("qualification must be a FrozenReferenceQualificationArtifact")
    if qualification.preselected_cohort_sha256 != cohort.sha256:
        raise ValueError("qualification artifact escaped the frozen preselected cohort")
    if not isinstance(qualified_endpoint_values, Mapping):
        raise TypeError("qualified_endpoint_values must be a recipe-ID mapping")
    value_ids = tuple(
        _identifier(value, "qualified recipe_id") for value in qualified_endpoint_values
    )
    if len(value_ids) != len(set(value_ids)):
        raise ValueError("qualified endpoint mapping contains duplicate normalized recipe IDs")
    expected_ids = {
        value.recipe_id
        for value in qualification.records
        if value.status == REFERENCE_STATUS_QUALIFIED
    }
    actual_ids = set(value_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            "qualified endpoint values must match the frozen qualified recipe IDs exactly; "
            f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}"
        )
    if not expected_ids:
        raise ValueError("at least one qualified recipe is required for the conditional endpoint")
    values = tuple(
        _recall(qualified_endpoint_values[recipe_id], f"qualified_endpoint_values[{recipe_id!r}]")
        for recipe_id in sorted(expected_ids)
    )
    counts = dict(qualification.status_counts)
    preselected = len(cohort.recipe_ids)
    qualified = counts[REFERENCE_STATUS_QUALIFIED]
    unresolved = counts[REFERENCE_STATUS_UNRESOLVED]
    certified = counts[REFERENCE_STATUS_CERTIFIED_NO_SOLUTION]
    qualified_sum = fsum(values)
    return ReferenceQualificationSummary(
        preselected_cohort_sha256=cohort.sha256,
        qualification_protocol_sha256=qualification.qualification_protocol_sha256,
        qualification_artifact_sha256=qualification.sha256,
        qualification_source_artifact_sha256=qualification.source_artifact_sha256,
        status_counts=qualification.status_counts,
        preselected_recipe_count=preselected,
        qualified_recipe_count=qualified,
        unresolved_recipe_count=unresolved,
        certified_no_solution_recipe_count=certified,
        qualification_fraction=qualified / preselected,
        qualified_endpoint_numerator=float(qualified_sum),
        qualified_endpoint_denominator=qualified,
        conditional_qualified_mean=qualified_sum / qualified,
        full_cohort_zero_imputed_lower_bound=qualified_sum / preselected,
        full_cohort_unresolved_universal_upper_bound=(qualified_sum + unresolved) / preselected,
    )


def summarize_seeded_recipe_auc(
    recall_curves: Mapping[str, Mapping[str, Mapping[int, float]]],
    *,
    budgets: Sequence[int] = EXACT_FORWARD_BUDGETS,
) -> SeededEndpointSummary:
    """Macro-average qualified recipes per seed, then average seed macros.

    Every seed must contain exactly the same prequalified recipe IDs.  Missing
    method results therefore fail closed instead of disappearing from the
    denominator; callers must insert :func:`zero_recall_curve` for crashes.
    """

    budget_values = _validated_budgets(budgets)
    if not isinstance(recall_curves, Mapping) or not recall_curves:
        raise ValueError("recall_curves must be a non-empty seed mapping")
    seed_ids = tuple(_identifier(value, "seed_id") for value in recall_curves)
    expected_recipe_ids: set[str] | None = None
    summaries = []
    for seed_id in sorted(seed_ids):
        recipes = recall_curves[seed_id]
        if not isinstance(recipes, Mapping) or not recipes:
            raise ValueError(f"seed {seed_id!r} must contain qualified recipes")
        recipe_ids = {_identifier(value, "recipe_id") for value in recipes}
        if len(recipe_ids) != len(recipes):
            raise ValueError(f"seed {seed_id!r} contains duplicate normalized recipe IDs")
        if expected_recipe_ids is None:
            expected_recipe_ids = recipe_ids
        elif recipe_ids != expected_recipe_ids:
            raise ValueError("every seed must contain exactly the same qualified recipes")
        recipe_aucs = tuple(
            (
                recipe_id,
                normalized_log2_budget_auc(recipes[recipe_id], budgets=budget_values),
            )
            for recipe_id in sorted(recipe_ids)
        )
        macro = fsum(value for _, value in recipe_aucs) / len(recipe_aucs)
        summaries.append(
            SeedEndpointSummary(
                seed_id=seed_id,
                recipe_aucs=recipe_aucs,
                recipe_macro_auc=float(macro),
            )
        )
    seed_mean = fsum(value.recipe_macro_auc for value in summaries) / len(summaries)
    assert expected_recipe_ids is not None
    return SeededEndpointSummary(
        seed_summaries=tuple(summaries),
        recipe_count=len(expected_recipe_ids),
        seed_count=len(summaries),
        seed_mean_recipe_macro_auc=float(seed_mean),
    )


__all__ = [
    "CURVE_EQUIVALENCE_LOG_RMSE_MAX",
    "EXACT_FORWARD_BUDGETS",
    "FrozenPreselectedReferenceCohort",
    "FrozenReferenceQualificationArtifact",
    "OUTPUT_CAPS",
    "PAPER_ENDPOINT_METRICS_VERSION",
    "PARAMETER_CLUSTER_DISTANCE_MAX",
    "PRIMARY_ENDPOINT_NAME",
    "PRIMARY_OUTPUT_CAP",
    "REFERENCE_QUALIFICATION_SCHEMA",
    "REFERENCE_QUALIFICATION_STATUSES",
    "REFERENCE_QUALIFICATION_VERSION",
    "REFERENCE_MATCH_DISTANCE_MAX",
    "REFERENCE_STATUS_CERTIFIED_NO_SOLUTION",
    "REFERENCE_STATUS_QUALIFIED",
    "REFERENCE_STATUS_UNRESOLVED",
    "THRESHOLD_SENSITIVITY_MULTIPLIERS",
    "ReferenceRecall",
    "ReferenceQualificationRecord",
    "ReferenceQualificationSummary",
    "SeedEndpointSummary",
    "SeededEndpointSummary",
    "completed_candidate_prefix_lengths",
    "normalized_log2_budget_auc",
    "reference_recall",
    "summarize_seeded_recipe_auc",
    "summarize_reference_qualification",
    "zero_recall_curve",
]
