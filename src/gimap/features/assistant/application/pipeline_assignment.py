"""Phase-assignment results for the automatic GIWAXS report."""

from __future__ import annotations


def axis_text(axis: dict) -> str:
    kind = axis.get("kind")
    if kind == "transmission":
        return "Assignment used transmission |q| = 2k sin θ."
    if kind not in ("kinematic", "refracted_transmitted"):
        return "The source of the radial q axis was not recorded."
    incidence = axis["incidence_deg"]
    critical = axis.get("critical_angle_deg")
    scale = axis["k_sin_alpha_i_inv_A"]
    if kind == "kinematic":
        basis = f"Assignment used kinematic |q| (αi = {incidence:g}°, no film refraction correction)."
    else:
        basis = f"Assignment used transmitted-channel |q| with refracted qz (αi = {incidence:g}°, αc = {critical:g}°)."
    consequence = f" Near the horizon the refraction shift in qz can be of order k sin αi = {scale:.4g} Å⁻¹."
    if critical is None or incidence > critical:
        consequence += " Above the film critical angle, transmitted and reflected channels can give two rings for one hkl."
    return basis + consequence


def assignment_text(item: dict) -> str:
    """Measured and reference q, relative spacing difference and match status of a peak."""
    if not item.get("assigned"):
        return f"q = {item['q']:.4f} Å⁻¹ unassigned (window ±{item['tolerance']:.4f} Å⁻¹)"
    relative = item.get("relative_d")
    spacing = "" if relative is None else f", Δd/d = {relative:+.3%}"
    text = (f"q = {item['q']:.4f} → {item['phase']} {item['hkl']} (q_hkl = {item['q_hkl']:.4f}, "
            f"Δq = {item['delta_q']:+.4f} Å⁻¹ within ±{item['tolerance']:.4f}{spacing})")
    if item.get("ambiguous") or item.get("alternatives"):
        text += " — ambiguous"
    if item.get("tentative"):
        text += " — tentative (positions only)"
    return text


def assignment_lines(report: dict) -> list[str]:
    """The Phase assignment section of report.md (empty without phases)."""
    found = report.get("phase_assignment")
    if not found:
        return []
    names = ", ".join(phase.get("name", "?") for phase in found.get("phases") or ())
    lines = [f"Phases given: {names}. Rule: {found.get('rule')}.", axis_text(found.get("q_axis") or {}), ""]
    for phase in found.get("phases") or ():
        if phase.get("warnings"):
            lines.append(f"- ⚠ {phase.get('name', '?')}: " + " ".join(phase["warnings"]))
    if lines[-1]:
        lines.append("")
    for item in found.get("assignments") or []:
        others = []
        for other in item.get("alternatives") or ():
            spacing = other.get("relative_d")
            d = "" if spacing is None else f", Δd/d = {spacing:+.3%}"
            others.append(f"{other['phase']} {other['hkl']} (Δq = {other['delta_q']:+.4f} Å⁻¹{d})")
        lines.append(f"- {assignment_text(item)}" + (f"; also within the window: {'; '.join(others)}" if others else ""))
    return lines
