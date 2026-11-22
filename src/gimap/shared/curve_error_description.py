"""Statistical error descriptions shared by reduction exports and analysis reports."""

ERROR_MODELS = {
    "poisson": "Poisson counting statistics of each bin",
    "scaled_poisson": "Poisson counting statistics propagated through the reduction",
    "pixel_scatter": "the scatter of the corrected pixels in each bin (a pixel-scatter uncertainty, not counting noise)",
    "sigma_frame": "the supplied pixel standard deviations propagated through the reduction",
}


def curve_error_description(model, corrections):
    """Describe the recorded selection, including records that predate the optional estimators."""
    description = ERROR_MODELS.get(model, "the statistical errors reported with the curve")
    if model == "pixel_scatter" and corrections.get("pooled_scatter"):
        return description + ", with pooled variance for points containing fewer than ten pixels"
    if model in ("poisson", "scaled_poisson") and corrections.get("poisson_variance") == "smoothed":
        return description + ", with the sample shot variance estimated from neighbouring bins"
    return description
