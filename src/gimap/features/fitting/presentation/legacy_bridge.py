"""Deprecated Fitting controller name kept for external callers."""

from . import view_binding as _implementation

AsyncImageLoader = _implementation.AsyncImageLoader
FittingController = _implementation.FittingViewBinding
FittingViewBinding = _implementation.FittingViewBinding
apply_input_image_options = _implementation.apply_input_image_options
apply_threshold_mask = _implementation.apply_threshold_mask
finite_log_profiles = _implementation.finite_log_profiles
finite_mean_axis = _implementation.finite_mean_axis


def __getattr__(name):
    """Preserve less-common legacy imports without duplicating implementation."""
    return getattr(_implementation, name)


__all__ = [
    "AsyncImageLoader",
    "FittingController",
    "FittingViewBinding",
    "apply_input_image_options",
    "apply_threshold_mask",
    "finite_log_profiles",
    "finite_mean_axis",
]
