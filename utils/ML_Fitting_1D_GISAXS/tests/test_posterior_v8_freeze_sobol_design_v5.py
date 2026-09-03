from __future__ import annotations

from hashlib import sha256
import json

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.freeze_sobol_design_v5 import (
    FREEZE_RECEIPT_FILENAME,
    SOBOL_DESIGN_FILENAME,
    SPLIT_PLAN_FILENAME,
    freeze_v5_sobol_design,
    main,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import V5SobolDesign
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)


def _counts() -> V5SplitCounts:
    return V5SplitCounts(
        train=12,
        tuning_validation=4,
        calibration=3,
        test=3,
        reference=2,
        ood_topology=1,
        ood_range_width=1,
        ood_weak_component=1,
        ood_acquisition_policy=1,
    )


def test_freeze_creates_exclusive_runtime_bound_round_trip(tmp_path):
    target = tmp_path / "pilot-design"
    receipt = freeze_v5_sobol_design(
        target,
        experiment_id="v5.1-all34-pilot",
        counts=_counts(),
        scramble_seed=20260903,
        start_index=17,
        guard_band=8,
    )

    plan_text = (target / SPLIT_PLAN_FILENAME).read_text(encoding="utf-8")
    design_text = (target / SOBOL_DESIGN_FILENAME).read_text(encoding="utf-8")
    stored = json.loads((target / FREEZE_RECEIPT_FILENAME).read_text(encoding="utf-8"))
    plan = V5SplitPlan.from_json(plan_text)
    design = V5SobolDesign.from_json(design_text)

    assert stored == receipt
    assert plan.counts == _counts()
    assert plan.start_index == 17
    assert plan.guard_band == 8
    assert design.coordinate_names == V5_SOBOL_RECIPE_COORDINATE_NAMES
    assert receipt["split_plan_contract_sha256"] == plan.sha256
    assert receipt["sobol_design_contract_sha256"] == design.sha256
    assert receipt["split_plan_file_sha256"] == sha256(plan_text.encode()).hexdigest()
    assert receipt["sobol_design_file_sha256"] == sha256(design_text.encode()).hexdigest()
    assert receipt["runtime"]["scipy"] == design.payload()["scipy_version"]
    assert receipt["heavy_compute_performed"] is False

    with pytest.raises(FileExistsError):
        freeze_v5_sobol_design(
            target,
            experiment_id="must-not-overwrite",
            counts=_counts(),
            scramble_seed=1,
        )


def test_cli_requires_explicit_counts_and_prints_receipt(tmp_path, capsys):
    target = tmp_path / "cli-design"
    result = main(
        [
            "--output-directory",
            str(target),
            "--experiment-id",
            "smoke",
            "--scramble-seed",
            "17",
            "--guard-band",
            "2",
            "--train",
            "3",
            "--tuning-validation",
            "2",
            "--calibration",
            "1",
            "--test",
            "1",
            "--reference",
            "1",
            "--ood-topology",
            "1",
            "--ood-range-width",
            "1",
            "--ood-weak-component",
            "1",
            "--ood-acquisition-policy",
            "1",
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["experiment_id"] == "smoke"
    assert printed["assigned_recipe_count"] == 12
