from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_rqmc_design_v5 import (
    V5_PAPER_RQMC_SPLITS,
    V5PaperRQMCCoordinateContractIdentity,
    V5PaperRQMCDesign,
    V5PaperRQMCSplitSpec,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_rqmc_evaluator_v5 import (
    V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE,
    V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE,
    V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS,
    V5_PAPER_RQMC_SEED_SPECIFIC_BASELINE_POLICY,
    V5_PAPER_RQMC_SHARED_BASELINE_POLICY,
    V5FrozenTrainingArtifact,
    V5FrozenTrainingArtifactSet,
    V5PaperRQMCReplicateMethodMean,
    summarize_v5_paired_rqmc_replicates,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    REFERENCE_STATUS_CERTIFIED_NO_SOLUTION,
    REFERENCE_STATUS_QUALIFIED,
    REFERENCE_STATUS_UNRESOLVED,
)


_BUDGETS = (256, 512, 1024)
_BASELINE = "solver-only"
_COMPARISON = "neural-plus-refinement"
_BASELINE_PROTOCOL = "b" * 64
_COMPARISON_PROTOCOL = "c" * 64
_BASELINE_INFERENCE_SEEDS = "1" * 64
_COMPARISON_INFERENCE_SEEDS = "2" * 64
_STUDY_PROTOCOL = "3" * 64
_ENDPOINT_METRIC_CONTRACT = "4" * 64


def _sha(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _design() -> V5PaperRQMCDesign:
    return V5PaperRQMCDesign.create(
        coordinate_contract=V5PaperRQMCCoordinateContractIdentity(
            schema="test.coordinates/v1",
            version="test_transform_v1",
            sha256="a" * 64,
            dimension=2,
            coordinate_names=("shape", "radius"),
        ),
        split_specs=tuple(
            V5PaperRQMCSplitSpec(
                split_name=name,
                points_per_replicate=8,
                scramble_seeds=tuple(
                    100 * split_index + value for value in (11, 22, 33, 44, 55, 66, 77, 88)
                ),
            )
            for split_index, name in enumerate(V5_PAPER_RQMC_SPLITS, start=1)
        ),
    )


def _artifacts() -> V5FrozenTrainingArtifactSet:
    return V5FrozenTrainingArtifactSet.create(
        tuple(
            V5FrozenTrainingArtifact(model_seed=seed, artifact_sha256=_sha(f"artifact-{seed}"))
            for seed in (7, 19, 23, 31, 43)
        )
    )


def _cells():
    design = _design()
    artifacts = _artifacts()
    differences = (0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4)
    rows = []
    for artifact in artifacts.artifacts:
        for block, difference in zip(design.blocks_for_split("test"), differences):
            baseline_mean = 0.2 + 0.01 * block.replicate_index
            common = dict(
                design_sha256=design.sha256,
                study_protocol_sha256=_STUDY_PROTOCOL,
                endpoint_metric_contract_sha256=_ENDPOINT_METRIC_CONTRACT,
                split_name="test",
                split_id=block.split_id,
                replicate_index=block.replicate_index,
                replicate_id=block.replicate_id,
                block_id=block.block_id,
                model_seed=artifact.model_seed,
                training_artifact_sha256=artifact.artifact_sha256,
                preselected_cohort_sha256=_sha(f"cohort-{block.replicate_index}"),
                qualification_protocol_sha256=_sha("qualification-protocol"),
                qualification_artifact_sha256=_sha(f"qualification-{block.replicate_index}"),
                qualification_status_counts=(
                    (REFERENCE_STATUS_QUALIFIED, 6),
                    (REFERENCE_STATUS_UNRESOLVED, 1),
                    (REFERENCE_STATUS_CERTIFIED_NO_SOLUTION, 1),
                ),
                qualified_endpoint_denominator=6,
                exact_forward_budgets=_BUDGETS,
                evaluated_point_count=block.count,
            )
            rows.append(
                V5PaperRQMCReplicateMethodMean(
                    **common,
                    method_id=_BASELINE,
                    method_protocol_sha256=_BASELINE_PROTOCOL,
                    inference_seed_set_sha256=_BASELINE_INFERENCE_SEEDS,
                    qualified_endpoint_numerator=baseline_mean * 6,
                    full_cohort_zero_imputed_lower_bound=baseline_mean * 6 / 8,
                    full_cohort_unresolved_universal_upper_bound=(baseline_mean * 6 + 1) / 8,
                    replicate_mean=baseline_mean,
                )
            )
            rows.append(
                V5PaperRQMCReplicateMethodMean(
                    **common,
                    method_id=_COMPARISON,
                    method_protocol_sha256=_COMPARISON_PROTOCOL,
                    inference_seed_set_sha256=_COMPARISON_INFERENCE_SEEDS,
                    qualified_endpoint_numerator=(baseline_mean + difference) * 6,
                    full_cohort_zero_imputed_lower_bound=(baseline_mean + difference) * 6 / 8,
                    full_cohort_unresolved_universal_upper_bound=(
                        (baseline_mean + difference) * 6 + 1
                    )
                    / 8,
                    replicate_mean=baseline_mean + difference,
                )
            )
    return design, artifacts, tuple(rows)


def _summarize(
    design,
    artifacts,
    rows,
    *,
    baseline_policy=V5_PAPER_RQMC_SHARED_BASELINE_POLICY,
):
    return summarize_v5_paired_rqmc_replicates(
        design,
        artifacts,
        rows,
        metric_id="reference_recall_auc_n16",
        split_name="test",
        baseline_method_id=_BASELINE,
        comparison_method_id=_COMPARISON,
        baseline_model_seed_policy=baseline_policy,
    )


def test_paired_student_t_uses_scramble_means_and_reports_every_model_seed() -> None:
    design, artifacts, rows = _cells()
    summary = _summarize(design, artifacts, rows)

    assert [value.model_seed for value in summary.seed_summaries] == [7, 19, 23, 31, 43]
    for seed_summary in summary.seed_summaries:
        assert seed_summary.replicate_count == 8
        assert seed_summary.degrees_of_freedom == 7
        assert seed_summary.comparison_minus_baseline == pytest.approx(0.25)
        assert seed_summary.confidence_interval == pytest.approx(
            (0.1500763876466062, 0.3499236123533938)
        )
        assert [value.comparison_minus_baseline for value in seed_summary.paired_replicates] == (
            pytest.approx((0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4))
        )
    assert summary.point_bootstrap_used is False
    assert summary.randomization_unit == "independent_scramble_replicate_mean"
    assert summary.baseline_model_seed_policy == V5_PAPER_RQMC_SHARED_BASELINE_POLICY
    assert summary.baseline_inference_seed_set_sha256 == _BASELINE_INFERENCE_SEEDS
    assert summary.comparison_inference_seed_set_sha256 == _COMPARISON_INFERENCE_SEEDS
    assert summary.study_protocol_sha256 == _STUDY_PROTOCOL
    assert summary.endpoint_metric_contract_sha256 == _ENDPOINT_METRIC_CONTRACT
    assert len(summary.qualification_bindings) == 8
    assert all(value.qualified_recipe_count == 6 for value in summary.qualification_bindings)
    frozen = summary.frozen_seed_set_summary
    assert frozen.training_seed_count == 5
    assert frozen.comparison_minus_baseline == pytest.approx(0.25)
    assert frozen.confidence_interval == pytest.approx((0.1500763876466062, 0.3499236123533938))


def test_pairing_and_hash_are_invariant_to_input_row_order() -> None:
    design, artifacts, rows = _cells()
    forward = _summarize(design, artifacts, rows)
    reverse = _summarize(design, artifacts, tuple(reversed(rows)))

    assert reverse == forward
    assert reverse.input_sha256 == forward.input_sha256


def test_frozen_seed_set_averages_seed_differences_within_each_scramble() -> None:
    design, artifacts, rows = _cells()
    seed_rank = {
        artifact.model_seed: index - 2 for index, artifact in enumerate(artifacts.artifacts)
    }
    changed = []
    for row in rows:
        if row.method_id != _COMPARISON:
            changed.append(row)
            continue
        offset = seed_rank[row.model_seed] * 0.01 * (row.replicate_index + 1)
        changed_mean = row.replicate_mean + offset
        changed.append(
            replace(
                row,
                replicate_mean=changed_mean,
                qualified_endpoint_numerator=changed_mean * 6,
                full_cohort_zero_imputed_lower_bound=changed_mean * 6 / 8,
                full_cohort_unresolved_universal_upper_bound=(changed_mean * 6 + 1) / 8,
            )
        )

    summary = _summarize(design, artifacts, tuple(changed))

    assert [
        value.comparison_minus_baseline
        for value in summary.frozen_seed_set_summary.paired_replicates
    ] == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4))
    assert [
        value.comparison_minus_baseline for value in summary.seed_summaries[0].paired_replicates
    ] != pytest.approx((0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[:-1], "missing or unexpected"),
        (lambda rows: rows + (rows[0],), "duplicated"),
        (
            lambda rows: (replace(rows[0], block_id="0" * 64),) + rows[1:],
            "misaligned",
        ),
        (
            lambda rows: (replace(rows[0], evaluated_point_count=7),) + rows[1:],
            "qualification status counts",
        ),
        (
            lambda rows: (replace(rows[0], training_artifact_sha256="a" * 64),) + rows[1:],
            "artifact set",
        ),
        (
            lambda rows: (replace(rows[0], exact_forward_budgets=(128, 256)),) + rows[1:],
            "budget mismatch",
        ),
        (
            lambda rows: (replace(rows[0], inference_seed_set_sha256="3" * 64),) + rows[1:],
            "inference-seed set changed",
        ),
        (
            lambda rows: (replace(rows[0], study_protocol_sha256="5" * 64),) + rows[1:],
            "study protocol or endpoint metric contract changed",
        ),
        (
            lambda rows: (replace(rows[0], endpoint_metric_contract_sha256="6" * 64),) + rows[1:],
            "study protocol or endpoint metric contract changed",
        ),
        (
            lambda rows: (replace(rows[0], qualification_artifact_sha256="7" * 64),) + rows[1:],
            "qualification cohort, artifact, statuses, or denominator changed",
        ),
    ],
)
def test_evaluator_fails_closed_on_missing_duplicate_misaligned_or_budget_cells(
    mutation, message
) -> None:
    design, artifacts, rows = _cells()

    with pytest.raises(ValueError, match=message):
        _summarize(design, artifacts, mutation(rows))


def test_shared_solver_baseline_must_be_identical_across_model_seeds() -> None:
    design, artifacts, rows = _cells()
    target = next(
        index
        for index, row in enumerate(rows)
        if row.model_seed == 19 and row.method_id == _BASELINE and row.replicate_index == 0
    )
    changed_mean = 0.99
    changed = (
        rows[:target]
        + (
            replace(
                rows[target],
                replicate_mean=changed_mean,
                qualified_endpoint_numerator=changed_mean * 6,
                full_cohort_zero_imputed_lower_bound=changed_mean * 6 / 8,
                full_cohort_unresolved_universal_upper_bound=(changed_mean * 6 + 1) / 8,
            ),
        )
        + rows[target + 1 :]
    )

    with pytest.raises(ValueError, match="identical across model seeds"):
        _summarize(design, artifacts, changed)

    summary = _summarize(
        design,
        artifacts,
        changed,
        baseline_policy=V5_PAPER_RQMC_SEED_SPECIFIC_BASELINE_POLICY,
    )
    assert summary.baseline_model_seed_policy == V5_PAPER_RQMC_SEED_SPECIFIC_BASELINE_POLICY


def test_frozen_artifact_hash_replays_and_rejects_seed_population_claim() -> None:
    artifacts = _artifacts()
    assert V5FrozenTrainingArtifactSet.from_json(artifacts.to_json()) == artifacts
    assert artifacts.artifact_set_scope == V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE
    assert len(artifacts.artifacts) == V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS
    payload = json.loads(artifacts.to_json())
    payload["training_seed_population_interval_allowed"] = True

    with pytest.raises(ValueError, match="inference claim"):
        V5FrozenTrainingArtifactSet.from_json(json.dumps(payload))


def test_formal_artifact_set_requires_five_and_engineering_set_cannot_infer() -> None:
    formal = _artifacts()
    with pytest.raises(ValueError, match="at least five"):
        V5FrozenTrainingArtifactSet.create(formal.artifacts[:4])

    engineering = V5FrozenTrainingArtifactSet.create_engineering_only(formal.artifacts[:2])
    assert engineering.artifact_set_scope == V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE
    assert V5FrozenTrainingArtifactSet.from_json(engineering.to_json()) == engineering
    design, _, rows = _cells()
    engineering_rows = tuple(row for row in rows if row.model_seed in {7, 19})
    with pytest.raises(ValueError, match="formal minimum-five"):
        _summarize(design, engineering, engineering_rows)


def test_replicate_cell_fails_closed_on_inconsistent_qualification_arithmetic() -> None:
    _, _, rows = _cells()
    with pytest.raises(ValueError, match="replicate_mean is inconsistent"):
        replace(rows[0], qualified_endpoint_numerator=0.0)
    with pytest.raises(ValueError, match="unresolved universal upper bound is inconsistent"):
        replace(rows[0], full_cohort_unresolved_universal_upper_bound=0.99)


def test_qualification_statuses_must_match_across_methods_and_model_seeds() -> None:
    design, artifacts, rows = _cells()
    changed = replace(
        rows[0],
        qualification_status_counts=(
            (REFERENCE_STATUS_QUALIFIED, 6),
            (REFERENCE_STATUS_UNRESOLVED, 0),
            (REFERENCE_STATUS_CERTIFIED_NO_SOLUTION, 2),
        ),
        full_cohort_unresolved_universal_upper_bound=(
            rows[0].qualified_endpoint_numerator / rows[0].evaluated_point_count
        ),
    )
    with pytest.raises(ValueError, match="qualification cohort, artifact, statuses"):
        _summarize(design, artifacts, (changed,) + rows[1:])
