from __future__ import annotations

import math

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
    FrozenPreselectedReferenceCohort,
    FrozenReferenceQualificationArtifact,
    REFERENCE_STATUS_CERTIFIED_NO_SOLUTION,
    REFERENCE_STATUS_QUALIFIED,
    REFERENCE_STATUS_UNRESOLVED,
    ReferenceQualificationRecord,
    completed_candidate_prefix_lengths,
    normalized_log2_budget_auc,
    reference_recall,
    summarize_reference_qualification,
    summarize_seeded_recipe_auc,
    zero_recall_curve,
)


def _constant_curve(value: float) -> dict[int, float]:
    return {budget: value for budget in EXACT_FORWARD_BUDGETS}


def _cohort(*, suffix: str = "main") -> FrozenPreselectedReferenceCohort:
    return FrozenPreselectedReferenceCohort(
        cohort_id=f"cohort-{suffix}",
        preselection_protocol_sha256="a" * 64,
        source_artifact_sha256="b" * 64,
        recipe_ids=("recipe-5", "recipe-3", "recipe-1", "recipe-4", "recipe-2"),
    )


def _qualification(
    cohort: FrozenPreselectedReferenceCohort,
) -> FrozenReferenceQualificationArtifact:
    return FrozenReferenceQualificationArtifact(
        cohort=cohort,
        qualification_protocol_sha256="c" * 64,
        source_artifact_sha256="d" * 64,
        records=(
            ReferenceQualificationRecord("recipe-4", REFERENCE_STATUS_UNRESOLVED),
            ReferenceQualificationRecord("recipe-2", REFERENCE_STATUS_QUALIFIED),
            ReferenceQualificationRecord("recipe-5", REFERENCE_STATUS_CERTIFIED_NO_SOLUTION),
            ReferenceQualificationRecord("recipe-1", REFERENCE_STATUS_QUALIFIED),
            ReferenceQualificationRecord("recipe-3", REFERENCE_STATUS_QUALIFIED),
        ),
    )


def test_normalized_auc_gives_each_budget_doubling_equal_weight() -> None:
    recalls = {
        256: 0.0,
        512: 0.25,
        1024: 0.5,
        2048: 0.75,
        4096: 1.0,
    }

    assert normalized_log2_budget_auc(recalls) == pytest.approx(0.5)
    assert normalized_log2_budget_auc({1: 0.0, 2: 0.0, 8: 1.0}, budgets=(1, 2, 8)) == pytest.approx(
        1.0 / 3.0
    )


@pytest.mark.parametrize(
    ("recalls", "budgets", "error"),
    [
        ({256: 0.0}, EXACT_FORWARD_BUDGETS, "keys must exactly equal"),
        (
            {256: 0.0, 512: 0.0, 1024: 0.0, 2048: 0.0, 4096: math.nan},
            EXACT_FORWARD_BUDGETS,
            "finite and in",
        ),
        ({1: 0.0, 2: 1.1}, (1, 2), "finite and in"),
        ({1: 0.0, 2: 1.0}, (2, 1), "strictly increasing"),
    ],
)
def test_normalized_auc_rejects_incomplete_or_invalid_curves(recalls, budgets, error) -> None:
    with pytest.raises(ValueError, match=error):
        normalized_log2_budget_auc(recalls, budgets=budgets)


def test_completed_candidate_prefix_requires_full_completion_within_budget() -> None:
    assert completed_candidate_prefix_lengths((10, 256, 257, 1024, 5000)) == (
        2,
        3,
        4,
        4,
        4,
    )
    assert completed_candidate_prefix_lengths(()) == (0, 0, 0, 0, 0)

    with pytest.raises(ValueError, match="non-decreasing"):
        completed_candidate_prefix_lengths((100, 99))


def test_reference_recall_keeps_full_denominator_and_reports_n_ceiling() -> None:
    value = reference_recall(16, 20, output_cap=16)

    assert value.recall == pytest.approx(0.8)
    assert value.ceiling == pytest.approx(0.8)
    assert value.reference_count == 20

    with pytest.raises(ValueError, match="at least 1"):
        reference_recall(0, 0)
    with pytest.raises(ValueError, match="exceeds"):
        reference_recall(17, 20, output_cap=16)


def test_reference_qualification_reports_conditional_mean_and_full_cohort_bounds() -> None:
    cohort = _cohort()
    qualification = _qualification(cohort)
    summary = summarize_reference_qualification(
        {"recipe-1": 0.25, "recipe-2": 0.75, "recipe-3": 1.0},
        cohort=cohort,
        qualification=qualification,
    )

    assert summary.preselected_recipe_count == 5
    assert summary.qualified_recipe_count == 3
    assert summary.unresolved_recipe_count == 1
    assert summary.certified_no_solution_recipe_count == 1
    assert summary.qualification_fraction == pytest.approx(0.6)
    assert summary.conditional_qualified_mean == pytest.approx(2.0 / 3.0)
    assert summary.full_cohort_zero_imputed_lower_bound == pytest.approx(0.4)
    assert summary.full_cohort_unresolved_universal_upper_bound == pytest.approx(0.6)
    assert summary.preselected_cohort_sha256 == cohort.sha256
    assert summary.qualification_artifact_sha256 == qualification.sha256
    assert qualification.status_counts == (
        (REFERENCE_STATUS_QUALIFIED, 3),
        (REFERENCE_STATUS_UNRESOLVED, 1),
        (REFERENCE_STATUS_CERTIFIED_NO_SOLUTION, 1),
    )


def test_reference_qualification_fails_closed_without_a_conditional_cohort() -> None:
    cohort = _cohort()
    no_qualified = FrozenReferenceQualificationArtifact(
        cohort=cohort,
        qualification_protocol_sha256="c" * 64,
        source_artifact_sha256="d" * 64,
        records=tuple(
            ReferenceQualificationRecord(recipe_id, REFERENCE_STATUS_UNRESOLVED)
            for recipe_id in cohort.recipe_ids
        ),
    )
    with pytest.raises(ValueError, match="at least one qualified"):
        summarize_reference_qualification({}, cohort=cohort, qualification=no_qualified)
    qualification = _qualification(cohort)
    with pytest.raises(ValueError, match="match the frozen qualified recipe IDs exactly"):
        summarize_reference_qualification(
            {"recipe-1": 0.1, "recipe-2": 0.2},
            cohort=cohort,
            qualification=qualification,
        )
    with pytest.raises(ValueError, match="finite and in"):
        summarize_reference_qualification(
            {"recipe-1": math.nan, "recipe-2": 0.2, "recipe-3": 0.3},
            cohort=cohort,
            qualification=qualification,
        )


def test_reference_qualification_identities_are_order_invariant_and_complete() -> None:
    cohort = _cohort()
    reordered_cohort = FrozenPreselectedReferenceCohort(
        cohort_id=cohort.cohort_id,
        preselection_protocol_sha256=cohort.preselection_protocol_sha256,
        source_artifact_sha256=cohort.source_artifact_sha256,
        recipe_ids=tuple(reversed(cohort.recipe_ids)),
    )
    qualification = _qualification(cohort)
    reordered_qualification = FrozenReferenceQualificationArtifact(
        cohort=reordered_cohort,
        qualification_protocol_sha256=qualification.qualification_protocol_sha256,
        source_artifact_sha256=qualification.source_artifact_sha256,
        records=tuple(reversed(qualification.records)),
    )
    assert reordered_cohort.sha256 == cohort.sha256
    assert reordered_qualification.sha256 == qualification.sha256

    with pytest.raises(ValueError, match="every frozen cohort recipe exactly once"):
        FrozenReferenceQualificationArtifact(
            cohort=cohort,
            qualification_protocol_sha256="c" * 64,
            source_artifact_sha256="d" * 64,
            records=qualification.records[:-1],
        )
    with pytest.raises(ValueError, match="duplicate recipe IDs"):
        FrozenReferenceQualificationArtifact(
            cohort=cohort,
            qualification_protocol_sha256="c" * 64,
            source_artifact_sha256="d" * 64,
            records=qualification.records[:-1] + (qualification.records[0],),
        )


def test_seeded_summary_is_recipe_macro_then_seed_mean() -> None:
    summary = summarize_seeded_recipe_auc(
        {
            "seed-1": {
                "recipe-a": _constant_curve(0.5),
                "recipe-b": _constant_curve(1.0),
            },
            "seed-2": {
                "recipe-a": _constant_curve(0.25),
                "recipe-b": _constant_curve(0.75),
            },
        }
    )

    assert summary.recipe_count == 2
    assert summary.seed_count == 2
    assert [item.recipe_macro_auc for item in summary.seed_summaries] == pytest.approx([0.75, 0.5])
    assert summary.seed_mean_recipe_macro_auc == pytest.approx(0.625)


def test_missing_recipe_fails_closed_and_crash_must_be_explicit_zero() -> None:
    with pytest.raises(ValueError, match="exactly the same qualified recipes"):
        summarize_seeded_recipe_auc(
            {
                "seed-1": {"recipe-a": _constant_curve(1.0)},
                "seed-2": {
                    "recipe-a": _constant_curve(1.0),
                    "recipe-b": _constant_curve(0.0),
                },
            }
        )

    summary = summarize_seeded_recipe_auc(
        {
            "seed-1": {
                "recipe-a": _constant_curve(1.0),
                "recipe-crash": zero_recall_curve(),
            }
        }
    )
    assert summary.seed_summaries[0].recipe_macro_auc == pytest.approx(0.5)
