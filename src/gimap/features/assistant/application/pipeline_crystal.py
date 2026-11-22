"""Crystal orientation estimates, competing models and measured ring moments in the report."""

from .report_numbers import _number, _pm


def compact_crystal(phase):
    """Keep the numerical results and family weights in the summary; profiles are in report.json."""
    rings = []
    for ring in phase.get("rings", []):
        curves = {"chi_deg", "observed", "sigma", "measured", "fit_mask", "fitted", "components",
                  "exit_transmission", "film_intensity_factor", "exit_transmission_times_absorption"}
        row = {key: value for key, value in ring.items() if key not in curves}
        row["components"] = [{k: v for k, v in c.items() if k != "values"} for c in ring["components"]]
        rings.append(row)
    return {**phase, "rings": rings}


def crystal_lines(phases: list[dict]) -> list[str]:
    def axes_text(axes):
        if axes is None:
            return "unavailable"
        return " + ".join("(" + " ".join(map(str, hkl)) + ")" for hkl in axes or ()) or "isotropic"

    lines = []
    for phase in phases:
        status = "determined among the candidate axes" if phase["determined"] else "not determined"
        reasons = ", ".join(phase["reasons"])
        lines += [f"**{phase['phase']}** — {status}" + (f" ({reasons})." if reasons else "."), ""]
        if phase["ok"]:
            lines.append(f"Best model {axes_text(phase['axes'])}, runner-up {axes_text(phase['runner_up_axes'])}, "
                         f"ΔBIC = {_number(phase.get('delta_bic'))}.")
            if phase.get("runner_up_tied_axes"):
                tied = ", ".join(axes_text(axes) for axes in phase["runner_up_tied_axes"])
                lines.append(f"The model {tied} ties with the runner-up.")
            if phase["axes"]:
                if "populations" in phase:
                    for population in phase["populations"]:
                        plane = axes_text([population["axis"]])
                        width = _pm(population["fwhm_deg"], population["fwhm_err_deg"], 2)
                        fraction = _pm(population["fraction"], population["fraction_err"], 3)
                        lines.append(f"Fitted {plane} planes parallel to the substrate, mosaic FWHM {width}°, "
                                     f"fraction of this phase {fraction}.")
                else:
                    # Archived reports used a common mosaic width for all populations.
                    lines.append(f"Shared mosaic FWHM {_pm(phase['fwhm_deg'], phase['fwhm_err_deg'], 2)}°.")
                    if len(phase["axes"]) == 2:
                        shares = phase["axis_fractions"]
                        errors = phase.get("axis_fractions_err", [None, None])
                        ratios = [_pm(s, e, 3) for s, e in zip(shares, errors)]
                        lines.append(f"The fitted axes account for fractions {' and '.join(ratios)} "
                                     "of the oriented population.")
                lines.append(f"Total oriented fraction {_pm(phase['oriented_fraction'], phase['oriented_fraction_err'], 3)}, "
                             f"isotropic fraction {_pm(phase['fraction_iso'], phase['fraction_iso_err'], 3)}. "
                             f"The fit uses {phase['family_weights']} reflection-family weights.")
            lines += [f"Crystal f = {_number(phase.get('herman_crystal'))}, "
                      f"joint χ²_red = {_number(phase.get('reduced_chi2'))}.", ""]
            optics = phase.get("film_optics")
            if optics:
                if optics.get("model") in ("refracted waves", "frame corrected with refracted waves"):
                    context = ("The frame was corrected for" if optics["model"].startswith("frame corrected")
                               else "The orientation model includes")
                    lines += [f"{context} refraction and attenuation of the incident and exit waves in the film "
                              f"(δ = {_number(optics['delta'])}, β = {_number(optics['beta'])}, "
                              f"{optics['depth_source']}).", ""]
                elif optics["available"]:
                    lines += [f"The fit includes the film's exit transmission |T|² "
                              f"(δ = {_number(optics['delta'])}, β = {_number(optics['beta'])}).", ""]
                else:
                    lines += [f"Film optics unavailable. {optics.get('reason', '')}".rstrip(), ""]
            if phase.get("search_settings", {}).get("noncollinear_poles_available") is False:
                lines += ["Only one pole direction constrains this phase. Two distinct fibre axes cannot be established.", ""]
            elif len(phase["rings"]) == 1:
                lines += ["Only one ring constrains this phase. Its polar angles provide the axis comparison.", ""]
            for ring in phase["rings"]:
                line = (f"- q = {_number(ring['q'])} Å⁻¹, χ²_red = {_number(ring['reduced_chi2'])}, "
                        f"measured ring f = {_number(ring['herman_measured'])}.")
                residual = ring.get("largest_residual")
                if residual and ring["reduced_chi2"] is not None and ring["reduced_chi2"] > phase["max_reduced_chi2"]:
                    line += (f" Largest observed minus fitted difference {residual['observed_minus_fit_sigma']:+.1f}σ "
                             f"at χ = {residual['chi_deg']:.1f}°.")
                lines.append(line)
        for item in phase.get("excluded_rings", []):
            lines.append(f"- q = {_number(item['q'])} Å⁻¹ excluded. {item['reason']}")
        lines.append("")
    return lines
