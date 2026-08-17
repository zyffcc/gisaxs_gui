"""Fitting 的 framework-neutral scientific API。"""

from .constraints import default_refine_bounds, default_refine_selected
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
from .cut_math import extract_pixel_profile, extract_q_profile, sample_q_mesh_line
from .image_transforms import (
    apply_input_image_options,
    apply_threshold_mask,
    finite_log_profiles,
    finite_mean_axis,
    mirror_fill_detector_gaps,
)
from .manual_refinement import run_manual_refinement
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
    "ParameterValue",
    "apply_input_image_options",
    "apply_threshold_mask",
    "ai_q_key",
    "candidate_parameter_mapping",
    "chi_square",
    "constraint_registry",
    "default_refine_bounds",
    "default_refine_selected",
    "extract_pixel_profile",
    "extract_q_profile",
    "exclusion_size",
    "filter_axis",
    "filter_for_display",
    "finite_log_profiles",
    "finite_mean_axis",
    "interpolate_series",
    "log_residuals",
    "log_rmse",
    "mirror_fill_detector_gaps",
    "normalize_intensity",
    "normalize_geometry",
    "optimize_scale_factor",
    "q_values_for_display",
    "q_values_for_model",
    "prepare_ai_curve",
    "run_manual_refinement",
    "sample_q_mesh_line",
    "sort_filter_pairs",
    "valid_y_values_for_limits",
    "verify_and_rank_candidates",
]
