from __future__ import annotations

from dataclasses import replace
import json

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    V5SobolDesign,
    materialize_v5_design_points,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_leakage_audit_v5 import (
    V5LeakageAudit,
    V5LeakageAuditConfig,
    V5LeakageAuditRow,
    V5LeakageViolation,
    V5OODDefinition,
    audit_v5_split_leakage,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_leakage_features_v5 import (
    V5_CLEAN_PHYSICS_LEAKAGE_DIM,
    V5_QUERY_BOUNDS_LEAKAGE_DIM,
)


def _study():
    counts = V5SplitCounts(
        train=2,
        tuning_validation=1,
        calibration=1,
        test=1,
        reference=1,
        ood_topology=1,
        ood_range_width=1,
        ood_weak_component=1,
        ood_acquisition_policy=1,
    )
    plan = V5SplitPlan.create(counts, guard_band=2)
    design = V5SobolDesign(
        coordinate_names=("physical_0", "physical_1", "bound_0", "bound_1"),
        scramble_seed=117,
    )
    definition = V5OODDefinition(
        held_out_topology_ids=(33,),
        held_out_range_width_labels=("extra_narrow",),
        weak_component_max_fraction=0.005,
        held_out_acquisition_policy_ids=("held_acquisition",),
    )
    rows = []
    for point in materialize_v5_design_points(plan, design):
        parameters = tuple(
            (point.unit_coordinates[index % 2] + 0.013 * (index + 1)) % 1.0
            for index in range(V5_CLEAN_PHYSICS_LEAKAGE_DIM)
        )
        query_bounds = tuple(
            (point.unit_coordinates[2 + index % 2] + 0.007 * (index + 1)) % 1.0
            for index in range(V5_QUERY_BOUNDS_LEAKAGE_DIM)
        )
        topology_id = 33 if point.ood_label == "topology_holdout" else 0
        range_label = "extra_narrow" if point.ood_label == "range_width_holdout" else "id_width"
        weakest = 0.001 if point.ood_label == "weak_component_holdout" else 0.2
        policy = (
            "held_acquisition"
            if point.ood_label == "acquisition_policy_holdout"
            else "id_acquisition"
        )
        rows.append(
            V5LeakageAuditRow(
                design_index=point.sobol_index,
                clean_group_id=point.clean_group_id,
                assigned_split=point.assigned_split,
                ood_label=point.ood_label,
                view_index=0,
                normalized_continuous_parameters=parameters,
                normalized_query_bounds=query_bounds,
                topology_id=topology_id,
                branch_pattern_id=0,
                acquisition_policy_id=policy,
                range_width_label=range_label,
                weakest_particle_fraction=weakest,
            )
        )
    return plan, design, definition, rows


def test_complete_grouped_design_passes_and_emits_replayable_json_hash():
    plan, design, definition, rows = _study()
    second_view = replace(rows[0], view_index=1, acquisition_policy_id="id_acquisition_2")

    audit = audit_v5_split_leakage(
        [*rows, second_view],
        plan=plan,
        design=design,
        ood_definition=definition,
    )

    assert audit.passed
    assert audit.payload["population"]["row_count"] == len(rows) + 1
    assert audit.payload["population"]["clean_group_count"] == plan.assigned_count
    assert len(audit.payload["nearest_neighbours"]["split_pairs"]) == 15
    assert V5LeakageAudit.from_json(audit.to_json()) == audit


def test_all_views_are_confined_to_the_clean_parent_split():
    plan, design, definition, rows = _study()
    escaped_view = replace(rows[0], view_index=1, assigned_split="test")

    with pytest.raises(V5LeakageViolation) as captured:
        audit_v5_split_leakage(
            [*rows, escaped_view],
            plan=plan,
            design=design,
            ood_definition=definition,
        )

    codes = {value["code"] for value in captured.value.audit.payload["violations"]["records"]}
    assert "split_or_ood_label_disagrees_with_plan" in codes
    assert "views_do_not_share_clean_parent" in codes


def test_cross_split_physics_duplicate_is_a_fail_closed_leak():
    plan, design, definition, rows = _study()
    train = next(value for value in rows if value.assigned_split == "train")
    test_index = next(index for index, value in enumerate(rows) if value.assigned_split == "test")
    rows[test_index] = replace(
        rows[test_index],
        normalized_continuous_parameters=train.normalized_continuous_parameters,
    )

    with pytest.raises(V5LeakageViolation) as captured:
        audit_v5_split_leakage(
            rows,
            plan=plan,
            design=design,
            ood_definition=definition,
        )

    report = captured.value.audit.payload
    assert report["status"] == "failed"
    assert report["exact_duplicates"]["physics_exact_cross_split"] == 1
    assert any(
        value["code"] == "physics_exact_cross_split" for value in report["violations"]["records"]
    )


def test_configured_near_duplicate_threshold_is_enforced_after_distance_report():
    plan, design, definition, rows = _study()
    train = next(value for value in rows if value.assigned_split == "train")
    test_index = next(index for index, value in enumerate(rows) if value.assigned_split == "test")
    close = list(train.normalized_continuous_parameters)
    close[0] += 1.0e-6 if close[0] < 0.5 else -1.0e-6
    rows[test_index] = replace(
        rows[test_index],
        normalized_continuous_parameters=tuple(close),
    )

    audit = audit_v5_split_leakage(
        rows,
        plan=plan,
        design=design,
        ood_definition=definition,
        config=V5LeakageAuditConfig(
            physics_near_duplicate_rms_threshold=1.0e-4,
            observation_near_duplicate_rms_threshold=0.0,
        ),
        fail_on_violation=False,
    )

    assert any(
        value["code"] == "physics_near_duplicate_cross_split"
        for value in audit.payload["violations"]["records"]
    )


def test_guard_indices_and_ood_labels_fail_closed():
    plan, design, definition, rows = _study()
    guard_index = plan.guard_blocks[0].start
    rows[0] = replace(rows[0], design_index=guard_index)
    rows[-1] = replace(rows[-1], acquisition_policy_id="id_acquisition")

    audit = audit_v5_split_leakage(
        rows,
        plan=plan,
        design=design,
        ood_definition=definition,
        fail_on_violation=False,
    )

    codes = {value["code"] for value in audit.payload["violations"]["records"]}
    assert "unassigned_or_guard_design_index" in codes
    assert "ood_support_not_isolated_or_label_incorrect" in codes


def test_leakage_audit_rejects_old_schema_and_hash_tampering():
    plan, design, definition, rows = _study()
    audit = audit_v5_split_leakage(
        rows,
        plan=plan,
        design=design,
        ood_definition=definition,
    )
    old = json.loads(audit.to_json())
    old["schema"] = "gisaxs.posterior_v8.split_leakage_audit/v0"
    with pytest.raises(ValueError, match="unsupported"):
        V5LeakageAudit.from_json(json.dumps(old))

    tampered = json.loads(audit.to_json())
    tampered["population"]["row_count"] += 1
    with pytest.raises(ValueError, match="hash"):
        V5LeakageAudit.from_json(json.dumps(tampered))
