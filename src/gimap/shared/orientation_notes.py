"""Per-ring selection notes for the pole window and the text and HTML reports."""


def _number(value):
    return f"{float(value):#.3g}".removesuffix(".").replace("-", "−")


def _limits_text(limits):
    return f"[{_number(limits[0])}, {_number(limits[1])}]"


def measured_f_text(value, sigma=None, limits=None, *, stat_label="stat", isotropic=None):
    """Measured moment, statistical error, systematic limits and same-coverage reference."""
    text = _number(value)
    if sigma is not None:
        text += f" ± {_number(sigma)} ({stat_label})"
    if limits is not None:
        text += " " + _limits_text(limits)
    if isotropic is not None:
        text += ", f_iso = " + _number(isotropic)
    return text


COVERAGE_WARNING = "The orientation factor is not determined by the measured χ range (systematic range {limits})."


def coverage_warning(systematics, format_text=str.format):
    limits = systematics.get("range")
    if limits and limits[1] > limits[0] + 0.05:
        return format_text(COVERAGE_WARNING, limits=_limits_text(limits))
    return ""


def orientation_notes(settings, systematics, format_text=str.format):
    lines = []
    bounds = settings.get("horizon_excluded_deg")
    if bounds:
        lines.append(format_text(
            "Excluded |χ| {low}–{high}° next to the exit-angle cut at {fraction} of the median net ring intensity.",
            low=f"{bounds[0]:g}", high=f"{bounds[1]:g}", fraction=f"{settings['horizon_fraction']:.0%}",
        ))
    sources = systematics.get("range_sources") or {}
    for end, template in (("min", "Lower limit from {scenario} (exit cut ×{cut}, horizon threshold {threshold}, b = {background})."),
                          ("max", "Upper limit from {scenario} (exit cut ×{cut}, horizon threshold {threshold}, b = {background}).")):
        source = sources.get(end)
        if source:
            lines.append(format_text(template, scenario=format_text(source["scenario"]),
                                     cut=f"{source['threshold_multiplier']:g}", threshold=f"{source['horizon_fraction']:.0%}",
                                     background=f"{source['background']:.5g}"))
    if "range_sources" not in systematics and systematics.get("model_fill_allowed") is False:
        lines.append(format_text(
            "The pole fit is undetermined. The systematic range uses measured ±χ partners only and leaves the missing wedge empty."
        ))
    return lines
