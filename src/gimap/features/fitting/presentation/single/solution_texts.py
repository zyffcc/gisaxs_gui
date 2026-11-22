"""What the single-curve page says of solutions and of the steps of Undo, in the interface language:

* a solution's name, numbered where two have the same composition (``Vertical cylinder + Vertical cylinder #2``);
* its flags in short (``model error; at bound: σ_q``), for the solution selector;
* where the values of a fit came from after Refine These Values (``refined_text``);
* the ± of a value of a Predict solution: half the width of its 68 % interval (``predict_plus``);
* the name of a step of Undo (``step_text``).
"""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared import fit_names

from ...application.solution_export import half_width, intervals_by_path

SHORT_FLAGS = {
    "unconstrained": "not determined: {name}", "at_bound": "at bound: {name}", "correlated": "correlated: {name}",
    "low_ess": "few samples", "bic": "compared by BIC", "model_error": "model error", "large": "large: {name}",
    "qzR_cov": "interval coverage: {name}", "qzR_fail": "level 2 fails: {name}",
    "cyl_sphere": "possible sphere: {name}",
    "small": "small: {name}", "qzR": "q_z·R high: {name}", "absent": "population {name} unused",
}
"""A flag in a few words (the selector of solutions; Fit details has the sentences)."""
INCOMPLETE = "incomplete"
SELECTOR = "{number}: {label} · χ²ᵣ {chi2}"
REFINED_ENTRY = "Refined from solution {number} ({method}) · χ²ᵣ {chi2}"
REFINED_FROM = ("Values from a new refinement ({method}) of solution {number}, {label}{edited}: not the values Predict "
                "gave; those and their intervals stay with solution {number} below.")
EDITED = ", edited in Model before the fit"
PLUS_TIP = "± half the width of the 68 % posterior interval [{low}, {high}], extended to include the least-squares value."
PLUS_TIP_BIC = "± half the width of the 68 % interval [{low}, {high}] from the least-squares errors (this solution is compared by BIC)."
NO_INTERVAL = "No interval: Predict parameters solves the scales and the background for every sample."
PLUS_HEADER_TIP = ("After Refine or Search, ± gives the least-squares error. For Predict parameters, "
                   "± gives half the width of the reported 68 % interval.")
METHODS = {"local": "Refine", "global": "Search the ranges", "predict": "Predict parameters"}


def numbered_labels(solutions) -> list[str]:
    """The solutions' names, ``#1``, ``#2`` … added where two have the same (``Solution.label``)."""
    labels = [solution.label for solution in solutions]
    out, seen = [], {}
    for label in labels:
        if labels.count(label) > 1:
            seen[label] = seen.get(label, 0) + 1
            out.append(f"{label} #{seen[label]}")
        else:
            out.append(label)
    return out


def short_flags(flags, many: bool) -> str:
    """The flags in a few words each, one per kind (``at bound: σ_q, Tail ν; model error``)."""
    order, names = [], {}
    for flag in flags or ():
        kind, values = fit_names.flag_names(flag)
        if kind not in SHORT_FLAGS:
            continue
        if kind not in names:
            order.append(kind)
            names[kind] = []
        joiner = " ~ " if kind == "correlated" else ""
        shown = values[0] if kind == "absent" and values else joiner.join(
            fit_names.display_name(name, many, tr) for name in values)
        if shown and shown not in names[kind]:
            names[kind].append(shown)
    return "; ".join(trf(SHORT_FLAGS[kind], name=", ".join(names[kind])) for kind in order)


def selector_text(index: int, solution, label: str) -> str:
    """An entry of the solution selector: number, name, χ²ᵣ, p, flags and whether it is incomplete."""
    chi2 = f"{solution.chi2:.3g}" if math.isfinite(solution.chi2) else "—"
    parts = [trf(SELECTOR, number=index + 1, label=label, chi2=chi2)]
    if solution.probability is not None:
        label = tr(fit_names.BIC_WEIGHT) if "bic" in solution.flags else "p"
        parts.append(f"{label} {solution.probability:.2f}")
    if getattr(solution, "incomplete", False):
        parts.append(tr(INCOMPLETE))
    flags = short_flags(solution.flags, len(solution.model.components) > 1)
    if flags:
        parts.append("⚠ " + flags)
    return " · ".join(parts)


def refined_entry(refined: dict, result) -> str:
    return trf(REFINED_ENTRY, number=refined["solution"], method=tr(METHODS.get(refined.get("method"), "Refine")),
               chi2=f"{result.chi2_reduced:.3g}")


def refined_text(refined: Optional[dict], solutions) -> str:
    """After a fit that started from a solution: where its values come from (``""`` otherwise)."""
    if not refined or not 1 <= int(refined.get("solution") or 0) <= len(solutions):
        return ""
    number = int(refined["solution"])
    label = numbered_labels(solutions)[number - 1]
    return trf(REFINED_FROM, method=tr(METHODS.get(refined.get("method"), "Refine")), number=number, label=label,
               edited=tr(EDITED) if refined.get("edited") else "")


def predict_plus(solution, path) -> tuple[str, str]:
    """(“± half-width”, its tooltip) of a value of a Predict solution; ``("", "")`` when it is no such solution."""
    if solution is None or not getattr(solution, "intervals", ()):
        return "", ""
    values = intervals_by_path(solution).get(path)
    if values is None:
        if path[0] == "globals" and (sampled := solution.posterior_summary.get(path[1])):
            low, high = sampled["low68"], sampled["high68"]
            return f"± {0.5 * (high - low):.2g}", trf("Sampled central 68 % interval [{low}, {high}].",
                                                    low=f"{low:.4g}", high=f"{high:.4g}")
        return "–", tr(NO_INTERVAL)
    half = half_width(values)
    if half is None:
        return "–", tr(NO_INTERVAL)
    kind = fit_names.reported_interval_kind(solution.flags, values[6], solution.diagnostics.get("posterior_kind"))
    tip = PLUS_TIP_BIC if kind == "bic" else PLUS_TIP
    if kind == "network_proposal":
        tip = "± half the width of the reported 68 % interval [{low}, {high}] from the network proposals."
    return f"± {half:.2g}", trf(tip, low=f"{values[2]:.4g}", high=f"{values[3]:.4g}")


def step_text(label) -> str:
    """The name of a step of Undo (``Snapshot.label``) in the interface language."""
    if not label:
        return tr("the last change")
    template, values = label
    return trf(template, **{key: tr(value) if isinstance(value, str) else value for key, value in values.items()})


__all__ = ["EDITED", "INCOMPLETE", "METHODS", "NO_INTERVAL", "PLUS_HEADER_TIP", "PLUS_TIP", "PLUS_TIP_BIC",
           "REFINED_ENTRY", "REFINED_FROM", "SELECTOR", "SHORT_FLAGS", "numbered_labels", "predict_plus",
           "refined_entry", "refined_text", "selector_text", "short_flags", "step_text"]
