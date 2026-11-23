"""Display frame-derived incidence and its comparison with the angle entered in Analyze."""

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.numbers import format_number


def measured_incidence(record):
    return trf("Measured αi  {estimate}°", estimate=format_number(record["deg"], record.get("error_deg")))


def incidence_text(record, kind):
    if not record:
        return ""
    observed = []
    for name, label in (("horizon", "Shadow edge"), ("specular", "Specular spot"), ("yoneda", "Yoneda band")):
        if name == "yoneda" and kind != "gisaxs":
            continue
        estimate = record.get(name)
        if estimate:
            value = format_number(estimate["deg"], estimate.get("error_deg")) + "°"
        elif name == "yoneda" and record.get("material_optics_error"):
            value = tr("Outside the optical table")
        else:
            value = tr("Needs a material αc" if name == "yoneda" and not record.get("critical_angle_deg") else "Unavailable")
        observed.append(f"{tr(label)}  {value}")
    if record.get("deg") is not None:
        observed.insert(0, measured_incidence(record))
        dominant = record.get("dominant_estimator")
        if dominant:
            label = {"specular": "Single specular-like spot", "horizon": "Shadow edge", "yoneda": "Yoneda band"}[dominant]
            observed.append(trf("{feature} supplies {weight}% of the incidence weight.", feature=tr(label),
                                weight=f"{100 * record['estimator_weights'][dominant]:.2f}"))
    else:
        observed.insert(0, tr("Incidence estimates disagree." if record.get("method") == "disagreement"
                              else "No incidence angle could be measured."))
    reference = record.get("override_deg", record.get("reference_deg"))
    decision = record.get("decision")
    if decision == "disagreement":
        observed.append(trf("Using measured αi. The reference ({value}°) differs by more than 3σ.",
                            value=f"{reference:g}"))
    elif decision == "agreement":
        observed.append(trf("Using reference αi ({value}°), within 3σ of the measurement.", value=f"{reference:g}"))
    elif decision == "user_override":
        if reference is not None and record.get("deg") is not None and abs(reference - record["deg"]) > 3 * record["error_deg"]:
            observed.append(trf("Keeping αi at {value}° by request. It differs from the measurement by more than 3σ.",
                                value=f"{reference:g}"))
        else:
            observed.append(tr("Using the entered αi."))
    return "\n".join(observed)
