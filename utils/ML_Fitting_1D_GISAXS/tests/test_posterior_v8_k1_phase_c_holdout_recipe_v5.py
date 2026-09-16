from __future__ import annotations

import json

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_holdout_recipe_v5 import (
    decode_v5_k1_phase_c_holdout_recipe_identity,
    materialize_v5_k1_phase_c_holdout_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
    build_v5_k1_phase_c_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_stress_v5 import (
    build_v5_k1_phase_c_observation_design,
    force_v5_k1_phase_c_range_stress,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_branch_forcing_v5 import (
    force_v5_k1_branch_coordinates,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_design_v5 import (
    materialize_v5_unit_coordinates_for_indices,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    v5_sobol_recipe_design,
)


@pytest.fixture(scope="module")
def plan():
    return build_v5_k1_phase_c_plan(formal=True)


def _recipe(plan, branch_ordinal: int, index: int):
    block = plan.sobol_blocks[branch_ordinal]
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    point = materialize_v5_unit_coordinates_for_indices(design, (index,))[0]
    return materialize_v5_k1_phase_c_holdout_recipe(
        plan=plan,
        block=block,
        sobol_index=index,
        original_unit_coordinates=point,
    )


@pytest.mark.parametrize("index,expected", tuple(enumerate(K1_PHASE_C_RANGE_STRESS_STRATA)))
def test_all_range_strata_materialize_and_replay(plan, index, expected):
    recipe = _recipe(plan, 0, index)
    identity = decode_v5_k1_phase_c_holdout_recipe_identity(
        recipe.canonical_json,
        expected_sha256=recipe.sha256,
    )
    assert identity.range_stress_stratum == expected
    assert identity.branch_id == plan.sobol_blocks[0].branch_id
    assert identity.clean_group_id == recipe.clean_group_id


@pytest.mark.parametrize(
    "offset,expected", tuple(enumerate(K1_PHASE_C_OBSERVATION_STRESS_STRATA))
)
def test_all_observation_strata_are_one_factor(plan, offset, expected):
    recipe = _recipe(plan, 1, offset * 5)
    assert recipe.observation_design.stress_stratum == expected
    assert recipe.observation_design.audit_payload()["generated_before_curve"] is True


def test_stress_only_changes_active_range_decisions(plan):
    block = plan.sobol_blocks[0]
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    point = materialize_v5_unit_coordinates_for_indices(design, (0,))[0]
    branch = force_v5_k1_branch_coordinates(point, branch_id=block.branch_id)
    stress = force_v5_k1_phase_c_range_stress(branch, range_stress_stratum="full")
    changed = {
        row["coordinate_name"] for row in stress.audit_payload()["replacements"]
    }
    assert changed
    assert all(name.endswith((".regime", ".placement")) for name in changed)


def test_observation_strata_do_not_share_one_identity():
    values = [
        build_v5_k1_phase_c_observation_design(
            clean_group_id="1" * 64,
            observation_stress_stratum=name,
        )
        for name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
    ]
    assert len({value.sha256 for value in values}) == len(values)
    assert values[0].noise.relative_sigma == 0.0
    assert values[1].noise.relative_sigma == 0.03
    assert values[2].point_keep_probability == 0.82
    assert values[3].crop_fraction == (0.08, 0.08)
    assert values[4].grid.kind == "linear"


def test_decoder_rejects_persisted_stress_drift(plan):
    recipe = _recipe(plan, 2, 7)
    payload = json.loads(recipe.canonical_json)
    payload["source"]["range_stress"]["range_stress_stratum"] = "full"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with pytest.raises(ValueError, match="range stress"):
        decode_v5_k1_phase_c_holdout_recipe_identity(encoded)


def test_decoder_rejects_wrong_recipe_hash(plan):
    recipe = _recipe(plan, 0, 0)
    with pytest.raises(ValueError, match="SHA-256"):
        decode_v5_k1_phase_c_holdout_recipe_identity(
            recipe.canonical_json,
            expected_sha256="0" * 64,
        )
