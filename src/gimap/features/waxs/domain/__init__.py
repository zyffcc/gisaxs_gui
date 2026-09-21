"""WAXS domain public API。"""

from .geometry import (
    UNSET_Q_LIMIT,
    compute_q_maps,
    cut_image_by_q_range,
    q_range_mask,
)
from .curve_preprocessing import (
    aligned_detector_distance,
    locate_reference_peak,
    peak_normalization_factor,
)
from .masking import (
    estimate_display_limits,
    percentile_limits,
    prepare_display_array,
)
from .integration import (
    angle_between,
    circle_cut_profile,
    integrate_image,
    line_cut_profile,
    normalize_angle_deg,
    smooth_curve,
)
from .background import subtract_background

__all__ = [
    "UNSET_Q_LIMIT",
    "compute_q_maps",
    "cut_image_by_q_range",
    "q_range_mask",
    "aligned_detector_distance",
    "locate_reference_peak",
    "peak_normalization_factor",
    "estimate_display_limits",
    "percentile_limits",
    "prepare_display_array",
    "angle_between",
    "circle_cut_profile",
    "integrate_image",
    "line_cut_profile",
    "normalize_angle_deg",
    "smooth_curve",
    "subtract_background",
]
