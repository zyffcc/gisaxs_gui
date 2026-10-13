"""Fitting 的 framework-neutral scientific API。"""

from .constraints import (
    default_global_search_bounds,
    default_global_search_selected,
    default_refine_bounds,
    default_refine_selected,
)
from .ai_curve import AiCurve, ai_q_key, prepare_ai_curve
from .candidates import (
    CandidateParameterMapping,
    candidate_parameter_mapping,
    verify_and_rank_candidates,
)
from .curve_transformations import (
    filter_axis,
    filter_for_display,
    interpolate_series,
    normalize_intensity,
    q_values_for_display,
    q_values_for_model,
    sort_filter_pairs,
    valid_y_values_for_limits,
)
from .manual_refinement import run_manual_refinement
from .insitu_recipe import (
    InSituFittingPolicy,
    InSituProcessingRecipe,
    InSituTrackingPolicy,
)
from .manual_fit import ManualFitRequest, ManualFitResult
from .models import CurveData, CutResult, CutSelection, FittingParameterSet, ParameterValue
from .physical_constraints import (
    ConstraintSet,
    ConstraintViolation,
    constraint_registry,
    exclusion_size,
    normalize_geometry,
)
from .scoring import chi_square, log_rmse, log_residuals, optimize_scale_factor
from .signed_q import (
    QBranch,
    QCombination,
    SignedQPreparation,
    prepare_signed_q_curve,
)

__all__ = [
    "CurveData",
    "AiCurve",
    "CandidateParameterMapping",
    "CutResult",
    "CutSelection",
    "ConstraintSet",
    "ConstraintViolation",
    "FittingParameterSet",
    "ManualFitRequest",
    "ManualFitResult",
    "InSituFittingPolicy",
    "InSituProcessingRecipe",
    "InSituTrackingPolicy",
    "ParameterValue",
    "QBranch",
    "QCombination",
    "SignedQPreparation",
    "ai_q_key",
    "candidate_parameter_mapping",
    "chi_square",
    "constraint_registry",
    "default_refine_bounds",
    "default_refine_selected",
    "default_global_search_bounds",
    "default_global_search_selected",
    "exclusion_size",
    "filter_axis",
    "filter_for_display",
    "interpolate_series",
    "log_residuals",
    "log_rmse",
    "normalize_intensity",
    "normalize_geometry",
    "optimize_scale_factor",
    "q_values_for_display",
    "q_values_for_model",
    "prepare_ai_curve",
    "prepare_signed_q_curve",
    "run_manual_refinement",
    "sort_filter_pairs",
    "valid_y_values_for_limits",
    "verify_and_rank_candidates",
]
