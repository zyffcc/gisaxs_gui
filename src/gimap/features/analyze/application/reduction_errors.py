"""The same pixel uncertainty inputs for line cuts, rings and standard calibration."""

from ..domain.curve_errors import PixelErrors
from .frame_preprocessing import photon_counts


def reduction_errors(corrections, frame):
    model = frame.get("error_model_used")
    if model is None:
        counts = photon_counts(frame.get("metadata"), corrections, mirror_filled=frame.get("filled_pixels") is not None)
        model = "poisson" if counts else "scatter"
    return PixelErrors(
        model=model, variance=frame.get("pixel_variance"), scale=frame.get("intensity_scale"),
        extra_variance=frame.get("extra_variance"), pooled_scatter=corrections.pooled_scatter,
        poisson_variance=corrections.poisson_variance,
    )
