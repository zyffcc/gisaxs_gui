"""TensorFlow-free contract for the bounds-conditioned V3 proposal model."""

from __future__ import annotations

from types import MappingProxyType

from .bounds_first_contract import (
    BOUNDS_EMBEDDING_DIM,
    BOUNDS_EMBEDDING_VERSION,
    LOCAL_TARGET_SEMANTICS,
)
from .branch_codec import UNIT_CUBE_DIMENSIONS
from .canonical_branch_catalog import (
    CANONICAL_BRANCH_CATALOG_VERSION,
    CANONICAL_BRANCH_COUNT,
)
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES


# These dimensions are part of this independent serialized-input contract.
# A TensorFlow-side invariant in model_v3 prevents drift from the curve encoder.
CURVE_POINT_FEATURE_DIM = 3
CURVE_GLOBAL_FEATURE_DIM = 5


BOUNDS_PROPOSAL_MODEL_VERSION = "posterior_v8_bounds_conditioned_logistic_normal_v3"
BOUNDS_PROPOSAL_MODEL_NAME = "posterior_v8_bounds_conditioned_local_proposal_v3"
MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS = BOUNDS_EMBEDDING_VERSION
MODEL_OUTPUT_COORDINATE_SEMANTICS = LOCAL_TARGET_SEMANTICS
MODEL_BRANCH_CATALOG_VERSION = CANONICAL_BRANCH_CATALOG_VERSION
MODEL_PHYSICAL_BRANCH_COUNT = CANONICAL_BRANCH_COUNT
MODEL_COMPONENT_SLOTS_VERSION = CANONICAL_COMPONENT_SLOTS_VERSION
MODEL_ACTIVE_MASK_SEMANTICS = "semantic_branch_presence_in_local_user_codec_26d_v1"
MODEL_VARYING_MASK_SEMANTICS = "codec_effective_varying_dimensions_in_local_user_codec_26d_v1"
BOUNDS_EMBEDDING_LAYOUT = (
    "interleaved_normalized_gui_physical_low_high_presence_for_24_component_"
    "axes_then_2_resolution_axes_v1"
)

MODEL_INPUT_KEYS = (
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
MODEL_OUTPUT_KEYS = (
    "topology_logits",
    "branch_pattern_logits",
    "mixture_logits",
    "mixture_loc",
    "mixture_logscale",
)
MODEL_FIXED_DIMENSIONS = MappingProxyType(
    {
        "point_feature_dim": CURVE_POINT_FEATURE_DIM,
        "global_feature_dim": CURVE_GLOBAL_FEATURE_DIM,
        "topology_dim": NUM_TOPOLOGIES,
        "d_presence_dim": MAX_COMPONENTS,
        "resolution_presence_dim": 1,
        "bounds_embedding_dim": BOUNDS_EMBEDDING_DIM,
        "local_coordinate_dim": UNIT_CUBE_DIMENSIONS,
    }
)
BOUNDS_MODEL_COORDINATE_CONTRACT = MappingProxyType(
    {
        "model_input_bounds": MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS,
        "model_input_bounds_layout": BOUNDS_EMBEDDING_LAYOUT,
        "model_input_active_mask": MODEL_ACTIVE_MASK_SEMANTICS,
        "model_input_varying_mask": MODEL_VARYING_MASK_SEMANTICS,
        "model_output": MODEL_OUTPUT_COORDINATE_SEMANTICS,
        "component_slot_canonicalization": MODEL_COMPONENT_SLOTS_VERSION,
        "fixed_and_inactive_output_value": 0.5,
        "simple_global_26d_box_is_gui_bounds": False,
    }
)

if BOUNDS_EMBEDDING_DIM != 78 or UNIT_CUBE_DIMENSIONS != 26:  # pragma: no cover
    raise RuntimeError("bounds-conditioned V3 tensor dimensions changed unexpectedly")


__all__ = [
    "BOUNDS_EMBEDDING_LAYOUT",
    "BOUNDS_MODEL_COORDINATE_CONTRACT",
    "BOUNDS_PROPOSAL_MODEL_NAME",
    "BOUNDS_PROPOSAL_MODEL_VERSION",
    "CURVE_GLOBAL_FEATURE_DIM",
    "CURVE_POINT_FEATURE_DIM",
    "MODEL_ACTIVE_MASK_SEMANTICS",
    "MODEL_BRANCH_CATALOG_VERSION",
    "MODEL_COMPONENT_SLOTS_VERSION",
    "MODEL_FIXED_DIMENSIONS",
    "MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS",
    "MODEL_INPUT_KEYS",
    "MODEL_OUTPUT_COORDINATE_SEMANTICS",
    "MODEL_OUTPUT_KEYS",
    "MODEL_PHYSICAL_BRANCH_COUNT",
    "MODEL_VARYING_MASK_SEMANTICS",
]
