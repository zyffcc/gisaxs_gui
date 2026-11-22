"""Human-readable counterpart of the numerical cell_refinement record."""

from .report_numbers import _number


def _estimate(value, sigma, scale=1):
    if value is None:
        return "—"
    text = f"{scale*value:.6g}"
    return text if sigma is None else text + f" ± {scale*sigma:.3g}"


def cell_lines(records):
    lines = ["The fit uses peak-centre errors. A competing reflection is accepted only when its CIF "
             "family intensity is at least three times the combined alternatives from the same phase. "
             "Cross-phase overlaps are excluded. A free q shift requires at least two more peaks than cell "
             "parameters and an independent shift column.", ""]
    for phase in records:
        lines += [f"**{phase['phase']}** ({phase['system']})", ""]
        if phase["ok"]:
            lines += ["| Parameter | Refined ± 1σ | Reference | Relative change (%) |",
                      "|---|---:|---:|---:|"]
            for key, value in phase["values"].items():
                unit = "Å³" if key == "V" else "°" if key in ("alpha", "beta", "gamma") else "Å"
                delta = _estimate(phase["strain"].get(key), phase["strain_errors"].get(key), 100)
                lines.append(f"| {key} ({unit}) | {_estimate(value, phase['errors'][key])} | "
                             f"{phase['reference'][key]:.6g} | {delta} |")
            shift = (_estimate(phase["zero_shift"], phase["zero_shift_error"])
                     if phase["zero_shift_error"] is not None else "0 (fixed)")
            lines += ["", f"χ²red = {_number(phase['chi2_red'])}, {phase['dof']} degrees of freedom. "
                      f"Additive q shift = {shift} Å⁻¹.", "",
                      "| h k l | Measured q ± σ (Å⁻¹) | Observed − fitted (Å⁻¹) |",
                      "|---|---:|---:|"]
            for peak in phase["residuals"]:
                hkl = " ".join(map(str, peak["hkl"]))
                lines.append(f"| {hkl} | {_estimate(peak['q'], peak['sigma'])} | {peak['residual']:+.5g} |")
        else:
            lines.append(phase["reason"])
        axis = phase.get("q_axis") or {}
        critical = axis.get("critical_angle_deg")
        lines += ["", (f"Transmitted-channel q, film αc = {critical:g}°."
                       if critical else "Kinematic q. No film critical angle was supplied.")]
        for item in phase.get("excluded", ()):
            lines.append(f"- q = {_number(item.get('q'))} Å⁻¹ excluded. {item['reason']}")
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
