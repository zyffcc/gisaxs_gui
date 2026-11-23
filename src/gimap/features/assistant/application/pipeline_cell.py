"""Human-readable counterpart of the numerical cell_refinement record."""

from src.gimap.shared.numbers import format_number


def _estimate(value, sigma, scale=1, *, compact=False):
    if value is None:
        return "—"
    return format_number(value * scale, None if sigma is None else sigma * scale, compact=compact)


def cell_lines(records):
    lines = ["The fit uses peak-centre errors and checks the reflection windows again at the fitted metrics "
             "and offsets. Unresolved families share an unknown intensity ratio on a textured frame and "
             "are excluded. The fit requires independent metric constraints. A free q shift requires two "
             "more peaks than metric parameters and an independent shift column.", ""]
    for phase in records:
        lines += [f"**{phase['phase']}** ({phase['system']})", ""]
        if phase["ok"]:
            lines += ["| Parameter | Refined ± 1σ | Reference | Relative change (%) |",
                      "|---|---:|---:|---:|"]
            for key, value in phase["values"].items():
                unit = "Å³" if key == "V" else "°" if key in ("alpha", "beta", "gamma") else "Å"
                delta = _estimate(phase["strain"].get(key), phase["strain_errors"].get(key), 100, compact=True)
                lines.append(f"| {key} ({unit}) | {format_number(value, phase['errors'][key], compact=True)} | "
                             f"{format_number(phase['reference'][key], phase.get('reference_errors', {}).get(key), compact=True, digits=5)} | {delta} |")
            shift = (_estimate(phase["zero_shift"], phase["zero_shift_error"])
                     if phase["zero_shift_error"] is not None else "0 (fixed)")
            lines += ["", f"χ²red = {format_number(phase['chi2_red'])}, {phase['dof']} degrees of freedom. "
                      f"Additive q shift = {shift} Å⁻¹.", "",
                      "| h k l | Measured q ± σ (Å⁻¹) | Observed − fitted (Å⁻¹) |",
                      "|---|---:|---:|"]
            for peak in phase["residuals"]:
                hkl = " ".join(map(str, peak["hkl"]))
                measured = format_number(peak["q"], peak["sigma"], compact=True)
                residual = format_number(peak["residual"], peak["sigma"], value_only=True)
                lines.append(f"| {hkl} | {measured} | {residual} |")
        else:
            lines.append(phase["reason"])
            available = [f"({' '.join(map(str, s['hkl']))}) at {format_number(s['q'], s['sigma'], compact=True)}"
                         for s in phase.get("seeds", ())]
            if available:
                lines.append("Available singleton centres (Å⁻¹): " + ", ".join(available) + ".")
        axis = phase.get("q_axis") or {}
        critical = axis.get("critical_angle_deg")
        lines += ["", (f"Transmitted-channel q, film αc = {critical:g}°."
                       if critical else "Kinematic q.")]
        if "metric_rank" in phase:
            lines.append(f"Metric rank {phase['metric_rank']}/{phase['metric_parameters']}.")
        sensitivity = phase.get("assignment_sensitivity", {})
        if sensitivity:
            bounds = ", ".join(f"{k} = {lo:.6g}–{hi:.6g} Å" for k, (lo, hi) in sensitivity["range_A"].items())
            unavailable = sum(bool(r["reason"]) for r in sensitivity["fits"])
            lines.append(f"Omitting one assigned centre at a time: {bounds or 'no determined fits'}. "
                         f"Undetermined subsets: {unavailable}/{len(sensitivity['fits'])}.")
        optics = phase.get("refraction_sensitivity", {})
        if optics.get("available"):
            for key, angle in (("off", 0.), ("on", optics["critical_angle_deg"])):
                trial = optics[key]
                values = ", ".join(f"{k} = {v:.6g} Å" for k, v in trial["values"].items() if k in ("a", "b", "c"))
                lines.append(f"Film αc = {angle:.5g}° ({len(trial['seeds'])} selected centres): "
                             f"{values or trial['reason']}")
        elif optics:
            lines.append(optics["reason"])
        for item in phase.get("excluded", ()):
            candidates = ", ".join(f"({c['hkl']}) {c['q']:.6g}" for c in item.get("candidates", ()))
            detail = f" Candidate positions (Å⁻¹): {candidates}." if candidates else ""
            lines.append(f"- q = {format_number(item.get('q'))} Å⁻¹ excluded. {item['reason']}{detail}")
        fibre = phase.get("fibre_strain")
        if fibre:
            lines.append("")
            if fibre["ok"]:
                axis = " ".join(map(str, fibre["axis_hkl"]))
                for key, value in fibre["values"].items():
                    lines.append(f"({axis}) fibre, {key.replace('_', '-')} strain "
                                 f"{_estimate(value, fibre['errors'][key], 100)}%.")
            else:
                lines.append(f"Directional strain unavailable. {fibre['reason']}")
        lines.append("")
    return lines
