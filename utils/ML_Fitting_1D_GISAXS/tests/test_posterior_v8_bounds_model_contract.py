from __future__ import annotations

import subprocess
import sys

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    BOUNDS_EMBEDDING_VERSION,
    LOCAL_TARGET_SEMANTICS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_model_contract import (
    BOUNDS_MODEL_COORDINATE_CONTRACT,
    BOUNDS_PROPOSAL_MODEL_NAME,
    BOUNDS_PROPOSAL_MODEL_VERSION,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
    MODEL_FIXED_DIMENSIONS,
    MODEL_INPUT_KEYS,
    MODEL_OUTPUT_KEYS,
    MODEL_PHYSICAL_BRANCH_COUNT,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_branch_catalog import (
    CANONICAL_BRANCH_CATALOG_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_component_slots import (
    CANONICAL_COMPONENT_SLOTS_VERSION,
)


def test_v3_contract_names_real_physical_bounds_and_local_output():
    assert BOUNDS_PROPOSAL_MODEL_NAME.endswith("local_proposal_v3")
    assert BOUNDS_PROPOSAL_MODEL_VERSION.endswith("logistic_normal_v3")
    assert MODEL_INPUT_KEYS == (
        "x",
        "point_mask",
        "global_features",
        "branch_topology",
        "branch_d_present",
        "branch_resolution_present",
        "bounds_embedding",
        "active_dimension_mask",
        "varying_dimension_mask",
    )
    assert "branch_low" not in MODEL_INPUT_KEYS
    assert "branch_high" not in MODEL_INPUT_KEYS
    assert MODEL_FIXED_DIMENSIONS["bounds_embedding_dim"] == 78
    assert MODEL_FIXED_DIMENSIONS["local_coordinate_dim"] == 26
    assert MODEL_BRANCH_CATALOG_VERSION == CANONICAL_BRANCH_CATALOG_VERSION
    assert MODEL_PHYSICAL_BRANCH_COUNT == 418
    assert MODEL_COMPONENT_SLOTS_VERSION == CANONICAL_COMPONENT_SLOTS_VERSION
    assert BOUNDS_MODEL_COORDINATE_CONTRACT["model_input_bounds"] == (
        BOUNDS_EMBEDDING_VERSION
    )
    assert BOUNDS_MODEL_COORDINATE_CONTRACT["model_output"] == LOCAL_TARGET_SEMANTICS
    assert BOUNDS_MODEL_COORDINATE_CONTRACT[
        "simple_global_26d_box_is_gui_bounds"
    ] is False
    assert MODEL_OUTPUT_KEYS == (
        "topology_logits",
        "branch_pattern_logits",
        "mixture_logits",
        "mixture_loc",
        "mixture_logscale",
    )


def test_v3_contract_import_does_not_import_tensorflow():
    command = (
        "import sys; "
        "import utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_model_contract; "
        "assert 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
