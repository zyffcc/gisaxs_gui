"""Verified one-click orchestration above raw neural/retrieval/Sobol rescue."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal, Mapping, Sequence

import numpy as np

from .branch_catalog import branch_pattern_id
from .component_observability import (
    OBSERVABILITY_POLICY_VERSION,
    CandidateObservabilityAssessment,
    ObservabilityPolicy,
    assess_candidate_observability,
)
from .evaluation import (
    EVALUATION_AUDIT_SCHEMA,
    CandidateInput,
    EvaluationReport,
    EvaluationThresholds,
    ObservedCurve,
    evaluate_candidates,
)
from .inference_proposals import ProposalModelPort
from .one_click_inference import InferenceBudget
from .production_bridge import ProductionBranchFactory, ProductionExactRefiner
from .reduced_model_search import BoundedReducedModelSearcher
from .reference_bank import CompetingBranch
from .rescue_inference import (
    RESCUE_INFERENCE_VERSION,
    RawCandidateGenerationResult,
    RescueAudit,
    RescuePolicy,
    RetrievalSeed,
    _run_candidate_generation,
)


VERIFIED_ONE_CLICK_SCHEMA = "gisaxs.posterior_v8.verified_one_click/v3"
VerifiedStatus = Literal[
    "compatible_target_reached",
    "compatible_partial",
    "no_compatible_mode_found_within_budget",
    "no_candidate_found_within_budget",
]


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True, kw_only=True)
class VerifiedOneClickResult:
    """Primary exact-compatibility result plus secondary minimality labels."""

    status: VerifiedStatus
    target_parameter_mode_count: int
    compatible_parameter_mode_count: int
    compatible_representative_candidate_ids: tuple[str, ...]
    effective_parameter_mode_count: int
    effective_representative_candidate_ids: tuple[str, ...]
    raw_generation: RawCandidateGenerationResult
    evaluation_report: EvaluationReport | None
    effective_evaluation_report: EvaluationReport | None
    observability_assessments: tuple[CandidateObservabilityAssessment, ...]
    raw_refinement_forward_evaluations: int
    observability_exact_forward_evaluations: int
    total_forward_evaluations: int
    total_forward_evaluation_limit: int
    observability_forward_evaluation_limit: int
    audit_schema: str = VERIFIED_ONE_CLICK_SCHEMA
    raw_generation_schema: str = RESCUE_INFERENCE_VERSION
    evaluation_audit_schema: str = EVALUATION_AUDIT_SCHEMA
    observability_policy_version: str = OBSERVABILITY_POLICY_VERSION

    def __post_init__(self) -> None:
        target = _integer(
            self.target_parameter_mode_count,
            "target_parameter_mode_count",
            minimum=1,
        )
        if self.audit_schema != VERIFIED_ONE_CLICK_SCHEMA:
            raise ValueError("unsupported verified one-click schema")
        if self.raw_generation_schema != RESCUE_INFERENCE_VERSION:
            raise ValueError("raw-generation schema does not match rescue v3")
        if self.evaluation_audit_schema != EVALUATION_AUDIT_SCHEMA:
            raise ValueError("evaluation schema does not match the authoritative evaluator")
        if self.observability_policy_version != OBSERVABILITY_POLICY_VERSION:
            raise ValueError("unexpected observability policy version")
        if not isinstance(self.raw_generation, RawCandidateGenerationResult):
            raise TypeError("raw_generation must be RawCandidateGenerationResult")
        report = self.evaluation_report
        if report is None:
            if self.raw_generation.candidates:
                raise ValueError("raw candidates require an EvaluationReport")
            expected_compatible_ids: tuple[str, ...] = ()
        else:
            if not isinstance(report, EvaluationReport):
                raise TypeError("evaluation_report must be EvaluationReport or None")
            if report.audit_schema != self.evaluation_audit_schema:
                raise ValueError("nested EvaluationReport uses an unexpected schema")
            if report.proposal_count != len(self.raw_generation.candidates):
                raise ValueError("evaluation report does not cover every raw candidate")
            if tuple(item.candidate_id for item in report.candidates) != tuple(
                item.candidate_id for item in self.raw_generation.candidates
            ):
                raise ValueError("evaluation report candidate IDs do not match raw generation")
            expected_compatible_ids = tuple(
                mode.representative_candidate_id for mode in report.parameter_modes
            )
        effective_report = self.effective_evaluation_report
        if effective_report is None:
            expected_effective_ids: tuple[str, ...] = ()
        else:
            if not isinstance(effective_report, EvaluationReport):
                raise TypeError("effective_evaluation_report must be EvaluationReport or None")
            expected_effective_ids = tuple(
                mode.representative_candidate_id for mode in effective_report.parameter_modes
            )
        assessments = tuple(self.observability_assessments)
        if not all(isinstance(item, CandidateObservabilityAssessment) for item in assessments):
            raise TypeError("observability_assessments contain an invalid value")
        assessment_ids = tuple(item.candidate_id for item in assessments)
        if len(set(assessment_ids)) != len(assessment_ids):
            raise ValueError("observability candidate IDs must be unique")
        accepted_ids = (
            set()
            if report is None
            else {item.candidate_id for item in report.candidates if item.accepted}
        )
        if set(assessment_ids) != accepted_ids:
            raise ValueError("every compatible accepted candidate requires a secondary assessment")
        compatible_ids = tuple(self.compatible_representative_candidate_ids)
        if compatible_ids != expected_compatible_ids:
            raise ValueError(
                "compatible representative IDs must match authoritative parameter modes"
            )
        compatible_count = _integer(
            self.compatible_parameter_mode_count,
            "compatible_parameter_mode_count",
            minimum=0,
        )
        if compatible_count != len(compatible_ids):
            raise ValueError("compatible_parameter_mode_count does not match representative IDs")
        effective_ids = tuple(self.effective_representative_candidate_ids)
        if effective_ids != expected_effective_ids:
            raise ValueError("effective representative IDs must match the secondary evaluation")
        effective_count = _integer(
            self.effective_parameter_mode_count,
            "effective_parameter_mode_count",
            minimum=0,
        )
        if effective_count != len(effective_ids):
            raise ValueError("effective_parameter_mode_count does not match representative IDs")
        expected_status: VerifiedStatus
        if compatible_count >= target:
            expected_status = "compatible_target_reached"
        elif compatible_count:
            expected_status = "compatible_partial"
        elif self.raw_generation.candidates:
            expected_status = "no_compatible_mode_found_within_budget"
        else:
            expected_status = "no_candidate_found_within_budget"
        if self.status != expected_status:
            raise ValueError(f"verified status must be {expected_status!r}")
        object.__setattr__(self, "target_parameter_mode_count", target)
        object.__setattr__(self, "compatible_parameter_mode_count", compatible_count)
        object.__setattr__(self, "compatible_representative_candidate_ids", compatible_ids)
        object.__setattr__(self, "effective_parameter_mode_count", effective_count)
        object.__setattr__(self, "effective_representative_candidate_ids", effective_ids)
        raw_calls = _integer(
            self.raw_refinement_forward_evaluations,
            "raw_refinement_forward_evaluations",
            minimum=0,
        )
        observability_calls = _integer(
            self.observability_exact_forward_evaluations,
            "observability_exact_forward_evaluations",
            minimum=0,
        )
        total = _integer(self.total_forward_evaluations, "total_forward_evaluations", minimum=0)
        total_limit = _integer(
            self.total_forward_evaluation_limit,
            "total_forward_evaluation_limit",
            minimum=1,
        )
        observability_limit = _integer(
            self.observability_forward_evaluation_limit,
            "observability_forward_evaluation_limit",
            minimum=0,
        )
        if raw_calls != self.raw_generation.audit.forward_evaluations_used:
            raise ValueError("raw refinement forward ledger disagrees with rescue audit")
        if observability_calls != sum(item.exact_forward_calls for item in assessments):
            raise ValueError("observability forward ledger disagrees with assessments")
        if total != raw_calls + observability_calls or total > total_limit:
            raise ValueError("unified total forward ledger is inconsistent")
        if observability_calls > observability_limit:
            raise ValueError("observability forward budget was exceeded")
        object.__setattr__(self, "observability_assessments", assessments)

    @property
    def raw_candidates(self) -> tuple[CandidateInput, ...]:
        return self.raw_generation.candidates

    @property
    def verified_parameter_mode_count(self) -> int:
        """Compatibility-verified mode count retained for API compatibility."""

        return self.compatible_parameter_mode_count

    @property
    def verified_representative_candidate_ids(self) -> tuple[str, ...]:
        """Compatibility-verified representatives retained as an API alias."""

        return self.compatible_representative_candidate_ids

    @property
    def audit(self) -> RescueAudit:
        return self.raw_generation.audit


def run_verified_one_click_inference(
    curve_inputs: Mapping[str, object],
    curve: ObservedCurve,
    *,
    model: ProposalModelPort,
    branch_factory: ProductionBranchFactory,
    thresholds: EvaluationThresholds,
    target_parameter_mode_count: int,
    inference_budget: InferenceBudget = InferenceBudget(),
    rescue_policy: RescuePolicy = RescuePolicy(),
    retrieval_seeds: Sequence[RetrievalSeed] = (),
    seed: int = 0,
    refiner: ProductionExactRefiner | None = None,
    observability_policy: ObservabilityPolicy = ObservabilityPolicy(),
    observability_forward_evaluation_limit: int = 64,
) -> VerifiedOneClickResult:
    """Search until distinct accepted parameter modes reach target or budget ends.

    ``RescuePolicy.target_candidate_count`` labels only the nested raw result;
    it never stops this verified search.
    """

    target = _integer(
        target_parameter_mode_count,
        "target_parameter_mode_count",
        minimum=1,
    )
    if not isinstance(thresholds, EvaluationThresholds):
        raise TypeError("thresholds must be EvaluationThresholds")
    if not isinstance(observability_policy, ObservabilityPolicy):
        raise TypeError("observability_policy must be ObservabilityPolicy")
    observability_limit = _integer(
        observability_forward_evaluation_limit,
        "observability_forward_evaluation_limit",
        minimum=0,
    )
    cached_count = -1
    cached_report: EvaluationReport | None = None

    def current_report(values: tuple[CandidateInput, ...]) -> EvaluationReport | None:
        nonlocal cached_count, cached_report
        if not values:
            return None
        if len(values) != cached_count:
            cached_report = evaluate_candidates(
                curve,
                values,
                thresholds=thresholds,
                best_of_n=(len(values),),
            )
            cached_count = len(values)
        return cached_report

    def compatible_target_reached(
        values: tuple[CandidateInput, ...], _raw_forward_used: int
    ) -> bool:
        report = current_report(values)
        return report is not None and len(report.parameter_modes) >= target

    raw = _run_candidate_generation(
        curve_inputs,
        curve,
        model=model,
        branch_factory=branch_factory,
        inference_budget=inference_budget,
        rescue_policy=rescue_policy,
        retrieval_seeds=retrieval_seeds,
        seed=seed,
        refiner=refiner,
        stop_when=compatible_target_reached,
    )
    report = current_report(raw.candidates)
    compatible_ids = (
        ()
        if report is None
        else tuple(mode.representative_candidate_id for mode in report.parameter_modes)
    )
    observability_by_id: dict[str, CandidateObservabilityAssessment] = {}
    observability_calls = 0
    if report is not None:
        by_id = {item.candidate_id: item for item in raw.candidates}
        accepted_ids = tuple(item.candidate_id for item in report.candidates if item.accepted)
        # Representatives receive secondary labels first.  This ordering can
        # improve diagnostic coverage but never changes primary search success.
        representative_set = set(compatible_ids)
        assessment_order = compatible_ids + tuple(
            candidate_id for candidate_id in accepted_ids if candidate_id not in representative_set
        )
        for candidate_id in assessment_order:
            remaining_total = (
                inference_budget.forward_evaluation_limit
                - raw.audit.forward_evaluations_used
                - observability_calls
            )
            remaining_observability = observability_limit - observability_calls
            allowance = max(0, min(remaining_total, remaining_observability))
            candidate = by_id[candidate_id]
            d_present = tuple(item.log_D is not None for item in candidate.components)
            branch = CompetingBranch(
                topology_id=candidate.topology_id,
                pattern_id=branch_pattern_id(
                    d_present + (False,) * (4 - len(d_present)),
                    candidate.resolution is not None,
                ),
            )
            context = branch_factory.context_for(branch)
            reduced_searcher = (
                None
                if context is None
                else BoundedReducedModelSearcher(
                    component_bounds=context.component_bounds,
                    resolution_bounds=context.resolution_bounds,
                    seed=seed + candidate.proposal_rank,
                )
            )
            assessment = assess_candidate_observability(
                curve,
                candidate,
                primary_metric_name=report.primary_gate_metric_name,
                primary_compatibility_threshold=report.primary_gate_threshold,
                policy=observability_policy,
                exact_forward_call_limit=allowance,
                reduced_model_searcher=reduced_searcher,
            )
            observability_by_id[candidate_id] = assessment
            observability_calls += assessment.exact_forward_calls
    assessments = tuple(
        observability_by_id[item.candidate_id]
        for item in (() if report is None else report.candidates)
        if item.candidate_id in observability_by_id
    )
    effective = tuple(
        candidate
        for candidate in raw.candidates
        if candidate.candidate_id in observability_by_id
        and observability_by_id[candidate.candidate_id].confirmed_effective
    )
    contiguous_effective = tuple(
        replace(item, proposal_rank=index) for index, item in enumerate(effective, 1)
    )
    effective_report = (
        None
        if not contiguous_effective
        else evaluate_candidates(
            curve,
            contiguous_effective,
            thresholds=thresholds,
            best_of_n=(len(contiguous_effective),),
        )
    )
    effective_ids = (
        ()
        if effective_report is None
        else tuple(mode.representative_candidate_id for mode in effective_report.parameter_modes)
    )
    if len(compatible_ids) >= target:
        status: VerifiedStatus = "compatible_target_reached"
    elif compatible_ids:
        status = "compatible_partial"
    elif raw.candidates:
        status = "no_compatible_mode_found_within_budget"
    else:
        status = "no_candidate_found_within_budget"
    return VerifiedOneClickResult(
        status=status,
        target_parameter_mode_count=target,
        compatible_parameter_mode_count=len(compatible_ids),
        compatible_representative_candidate_ids=compatible_ids,
        effective_parameter_mode_count=len(effective_ids),
        effective_representative_candidate_ids=effective_ids,
        raw_generation=raw,
        evaluation_report=report,
        effective_evaluation_report=effective_report,
        observability_assessments=assessments,
        raw_refinement_forward_evaluations=raw.audit.forward_evaluations_used,
        observability_exact_forward_evaluations=observability_calls,
        total_forward_evaluations=(raw.audit.forward_evaluations_used + observability_calls),
        total_forward_evaluation_limit=inference_budget.forward_evaluation_limit,
        observability_forward_evaluation_limit=observability_limit,
    )


__all__ = [
    "VERIFIED_ONE_CLICK_SCHEMA",
    "VerifiedOneClickResult",
    "VerifiedStatus",
    "run_verified_one_click_inference",
]
