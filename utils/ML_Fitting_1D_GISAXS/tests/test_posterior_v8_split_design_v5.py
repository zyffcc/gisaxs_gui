from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    V5SobolDesign,
    materialize_v5_design_points,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    MAIN_SPLITS,
    OOD_LABELS,
    V5SplitCounts,
    V5SplitPlan,
)


def _counts() -> V5SplitCounts:
    return V5SplitCounts(
        train=3,
        tuning_validation=2,
        calibration=2,
        test=2,
        reference=2,
        ood_topology=1,
        ood_range_width=1,
        ood_weak_component=1,
        ood_acquisition_policy=1,
    )


def test_scrambled_sobol_design_replays_exactly_with_content_bound_group_ids():
    plan = V5SplitPlan.create(_counts(), start_index=5, guard_band=3)
    design = V5SobolDesign(
        coordinate_names=("query_width", "target_R", "amplitude", "acquisition"),
        scramble_seed=20260903,
    )

    first = materialize_v5_design_points(plan, design)
    replay = materialize_v5_design_points(plan, design)

    assert first == replay
    assert len(first) == plan.assigned_count
    assert len({value.clean_group_id for value in first}) == len(first)
    assert np.all(np.asarray([value.unit_coordinates for value in first]) >= 0.0)
    assert np.all(np.asarray([value.unit_coordinates for value in first]) < 1.0)
    assert V5SplitPlan.from_json(plan.to_json()) == plan


def test_main_blocks_are_contiguous_disjoint_and_separated_by_exact_guard_band():
    plan = V5SplitPlan.create(_counts(), start_index=7, guard_band=4)

    assert tuple(value.name for value in plan.blocks) == MAIN_SPLITS
    assert tuple(value.name for value in plan.ood_blocks) == OOD_LABELS
    for left, guard, right in zip(plan.blocks, plan.guard_blocks, plan.blocks[1:]):
        assert left.stop == guard.start
        assert guard.count == 4
        assert guard.stop == right.start
        with pytest.raises(ValueError, match="guard band"):
            plan.split_for_index(guard.start)
    assigned_sets = [set(range(value.start, value.stop)) for value in plan.blocks]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(assigned_sets)
        for right in assigned_sets[index + 1 :]
    )
    assert sum(value.count for value in plan.ood_blocks) == plan.blocks[-1].count


def test_split_plan_rejects_tampering_and_old_schema():
    plan = V5SplitPlan.create(_counts())
    old = json.loads(plan.to_json())
    old["schema"] = "gisaxs.posterior_v8.sobol_split_plan/v0"
    with pytest.raises(ValueError, match="unsupported"):
        V5SplitPlan.from_json(json.dumps(old))

    tampered = json.loads(plan.to_json())
    tampered["guard_band"] += 1
    with pytest.raises(ValueError, match="does not reproduce"):
        V5SplitPlan.from_json(json.dumps(tampered))
    with pytest.raises(ValueError, match="audit hash"):
        replace(plan, sha256="0" * 64)
