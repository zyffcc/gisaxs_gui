"""Per-ring selection notes for the pole window and the text and HTML reports."""


def orientation_notes(settings, systematics, format_text=str.format):
    lines = []
    bounds = settings.get("horizon_excluded_deg")
    if bounds:
        lines.append(format_text(
            "Excluded |χ| {low}–{high}° next to the exit-angle cut at {fraction} of the median net ring intensity.",
            low=f"{bounds[0]:g}", high=f"{bounds[1]:g}", fraction=f"{settings['horizon_fraction']:.0%}",
        ))
    if systematics.get("model_fill_allowed") is False:
        lines.append(format_text(
            "The pole fit is undetermined. The systematic range uses measured ±χ partners only and leaves the missing wedge empty."
        ))
    return lines
