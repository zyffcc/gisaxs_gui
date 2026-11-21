"""One name per fitted quantity, and one sentence per flag of a Predict-parameters solution, wherever a person reads
them: Fitting's Fit details and tooltips, the 1D Predict rows, the automatic analysis's report and window.

The names are those of Fitting's Model table (``Tail ν``, ``σ_q``, ``σR/R``); a value of one of several populations
carries its number in brackets (``σR/R (1)``, ``Scale (2)``; the populations are numbered as in Model). The keys of the
saved tables and records stay as they are (``R_1``, ``sigma_R_1``, ``res_amplitude``, ``sigma_Res`` …): ``legend`` lists
each with its name. No Qt and no feature is imported here; the interface translates the templates.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

NAMES = {
    "R": "R", "sigma_R": "σR/R", "h": "h", "sigma_h": "σh/h", "D": "D", "sigma_nn": "σnn/D", "sigma_D": "σD/D",
    "Int": "Scale", "amplitude": "Scale",
    "background": "Background",
    "res_amplitude": "Tail A", "resolution_amplitude": "Tail A",
    "res_width": "Tail w", "tail_width": "Tail w", "sigma_Res": "Tail w",
    "res_exponent": "Tail ν", "tail_exponent": "Tail ν", "nu_Res": "Tail ν",
    "q_resolution": "σ_q", "k": "Factor k", "refl_amp": "|r|", "refl_phase": "arg r",
}
"""Every key a fit, a row or a record uses for a quantity → its name in the Model table."""
UNITS = {"R": "nm", "h": "nm", "D": "nm", "Tail w": "nm⁻¹", "σ_q": "nm⁻¹"}
"""The unit of a quantity, by its display name."""
POPULATION = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn", "sigma_D", "Int", "amplitude")
"""The keys of a population's own values; every other key is of the whole curve (background, beam tail, resolution)."""
TAIL = ("res_width", "tail_width", "sigma_Res", "res_exponent", "tail_exponent", "nu_Res")
"""The beam tail's width and exponent: they change nothing when Tail A is 0."""
TAIL_AMPLITUDE = ("res_amplitude", "resolution_amplitude")
GENERAL = ("low_ess", "bic", "model_error")
"""Flags of the whole solution (no value named)."""

SENTENCES = {
    "unconstrained": "the curve says little about {name} (its 68 % interval spans most of its allowed range)",
    "at_bound": "{name} stopped at a bound of its range",
    "correlated": "{name} are strongly correlated",
    "low_ess": "few effective samples of the posterior: the intervals are rough, and wider than a converged sampling "
               "would give",
    "bic": "the posterior's sampling did not converge: the solutions are compared by χ² and their number of values "
           "(BIC), and the intervals are the least-squares errors",
    "model_error": "the values are those of a fit with the model error added to σ: the fit with the curve's σ alone "
                   "lay outside the posterior",
    "large": "{name}: its Guinier region lies at or below the curve's lowest q, so the curve shows only its decay, "
             "which roughness or a rise at low q produce as well",
    "small": "{name}: smaller than the curve resolves, it acts as a smooth background",
    "absent": "population {name} has Scale 0: this solution does not use it",
}
"""Flag kind → the sentence a person reads (``{name}``: the display names; templates, translated by the interface)."""
EDGE = ("the refined value lies at the edge of the posterior: the model describes this curve only within the model "
        "error ({names})")
"""A refined value at an end of its own interval that is not at a bound of its range."""
CHI2_DEFINITION = ("χ²ᵣ = Σ((I − model)² / σ²) / (N − number of free values), over the N fitted points, with the "
                   "curve's σ")
RELATIVE = "probabilities are relative to the compositions that were tried"
ONLY_ONE = "only this composition was tried"
POINTS = "{fitted} of {given} points"
BELOW_WINDOW = "{count} below the network's q window"
UNUSABLE = "{count} without a usable value"
MORE = "Background, beam tail and resolution: {notes}"
"""The second paragraph of a solution's notes: the flags of values that are not the populations'."""

_NUMBERED = re.compile(r"^(?P<key>.+)_(?P<slot>\d+)$")


def split_name(name: str) -> tuple[str, Optional[int]]:
    """``("sigma_R", 1)`` of ``sigma_R_1``; ``("res_amplitude", None)`` of a value of the whole curve."""
    name = str(name)
    if name in NAMES:
        return name, None
    found = _NUMBERED.match(name)
    if found and found.group("key") in NAMES:
        return found.group("key"), int(found.group("slot"))
    return name, None


def display_name(name: str, many: bool = True, say=str) -> str:
    """The Model table's name of a key: ``σR/R (1)`` of ``sigma_R_1``, ``Tail ν`` of ``res_exponent`` /
    ``tail_exponent`` / ``nu_Res``. ``many`` False: a model of one population, whose values carry no number.
    ``say``: the interface's translation of a name (the Model table shows its names translated)."""
    key, slot = split_name(name)
    shown = say(NAMES.get(key, key))
    return shown if slot is None or not many else f"{shown} ({slot})"


def unit_of(name: str) -> str:
    return UNITS.get(NAMES.get(split_name(name)[0], ""), "")


def flag_names(flag: str) -> tuple[str, list[str]]:
    """``(kind, [internal names])`` of a flag (``correlated:R_2~sigma_R_2`` names two)."""
    kind, _, rest = str(flag).partition(":")
    return kind, [part for part in rest.split("~") if part]


def of_population(flag: str) -> bool:
    """Whether a flag concerns the populations' values or the solution as a whole (shown first); the others are of
    the background, the beam tail and the resolution (shown after them, in one paragraph)."""
    kind, names = flag_names(flag)
    if kind in GENERAL or kind in ("large", "small", "absent"):
        return True
    return any(split_name(name)[0] in POPULATION for name in names)


def sentence(flag: str, many: bool = True, joiner: str = " and ", say=str) -> Optional[tuple[str, dict]]:
    """``(template, {"name": …})`` of a flag (None for a kind without a sentence): the interface translates the
    template and fills it."""
    kind, names = flag_names(flag)
    template = SENTENCES.get(kind)
    if template is None:
        return None
    shown = names[0] if kind == "absent" and names else joiner.join(display_name(name, many, say) for name in names)
    return template, {"name": shown}


def grouped(flags: Iterable[str], many: bool = True, joiner: str = " and ", say=str) -> list[tuple[str, dict]]:
    """The flags as ``(template, {"name": …})``, one per kind: the values at a bound in one sentence (``σR/R, Tail ν
    stopped at a bound of its range``), the correlated pairs in one (``R and σR/R; D and σnn/D are strongly
    correlated``), so that a solution has a line per kind of remark, not per value."""
    order, names = [], {}
    for flag in flags:
        found = sentence(flag, many, joiner, say)
        if found is None:
            continue
        kind = flag_names(flag)[0]
        if kind not in names:
            order.append(kind)
            names[kind] = []
        if found[1]["name"] and found[1]["name"] not in names[kind]:
            names[kind].append(found[1]["name"])
    return [(SENTENCES[kind], {"name": ("; " if kind == "correlated" else ", ").join(names[kind])}) for kind in order]


def sentences(flags: Iterable[str], many: bool = True, joiner: str = " and ") -> list[str]:
    """The flags as English sentences, one per kind (records, the command line's report)."""
    return [template.format(**values) for template, values in grouped(flags, many, joiner)]


def split_flags(flags: Iterable[str]) -> tuple[list[str], list[str]]:
    """``(flags of the populations and of the solution, flags of background, tail and resolution)``."""
    first, second = [], []
    for flag in flags:
        (first if of_population(flag) else second).append(flag)
    return first, second


def clean_flags(flags: Iterable[str], amplitudes: dict) -> tuple:
    """The flags without those of a term that is absent: with Tail A = 0 (``amplitudes["res_amplitude"]``) nothing is
    said of Tail A at its bound, nor of Tail w and Tail ν, which then change nothing; a background at 0 is no flag;
    a population whose Scale is 0 (``amplitudes["Int_n"]``) gets one ``absent:n`` in place of the flags of its
    values. ``amplitudes``: the linear terms' values by their keys."""
    def zero(key: str) -> bool:
        value = amplitudes.get(key)
        return value is not None and math.isfinite(float(value)) and float(value) <= 0.0

    no_tail = zero("res_amplitude")
    gone = {int(key.rsplit("_", 1)[1]) for key in amplitudes if key.startswith("Int_") and zero(key)}

    def absent(name: str) -> bool:
        key, slot = split_name(name)
        if key in TAIL_AMPLITUDE or key == "background":
            return zero("res_amplitude") if key in TAIL_AMPLITUDE else zero("background")
        if key in TAIL:
            return no_tail
        return slot in gone

    kept = []
    for flag in flags:
        kind, names = flag_names(flag)
        if names and kind in ("at_bound", "unconstrained", "correlated", "large", "small") and any(map(absent, names)):
            continue
        kept.append(flag)
    return tuple(kept) + tuple(f"absent:{slot}" for slot in sorted(gone))


def interval_digits(best: float, low: float, high: float, significant: int = 3, most: int = 8) -> int:
    """Decimals with which ``best [low, high]`` is printed: ``significant`` digits of the value, and more until the
    three numbers differ where they differ (``6.00 [5.99, 6.01]``, not ``6 [5.99, 6.01]``)."""
    scale = abs(best) if best else max(abs(low), abs(high), 1e-300)
    decimals = max(0, significant - 1 - int(math.floor(math.log10(scale)))) if scale > 0 else significant
    decimals = min(decimals, most)
    while decimals < most:
        texts = [f"{value:.{decimals}f}" for value in (low, best, high)]
        if (low == best or texts[0] != texts[1]) and (high == best or texts[2] != texts[1]):
            break
        decimals += 1
    return decimals


def interval_text(best: float, low: float, high: float, unit: str = "") -> str:
    """``6.00 [5.99, 6.01] nm``: a value with its interval, all three with the same decimals."""
    decimals = interval_digits(float(best), float(low), float(high))
    return (f"{best:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]" + (f" {unit}" if unit else ""))


def at_edge(best: float, low: float, high: float, tolerance: float = 1e-6) -> bool:
    """The value sits at an end of its interval (which is wider than a point)."""
    width = high - low
    return width > 0 and (best - low <= tolerance * width or high - best <= tolerance * width)


def sizes_text(components, equals: str = " = ", decimals: Optional[dict] = None) -> str:
    """R of every population and D of those with a paracrystal, in the order of the radii: ``R = 3.06 / 6.02 nm,
    D = 49.4 / – nm`` (one population: ``R = 3.06 nm, D = 49.4 nm``; no D when no population has one).
    ``components``: dicts with ``R`` and, with a paracrystal, ``D``. ``decimals``: fixed decimals per key, else three
    significant digits."""
    def shown(value, key) -> str:
        if value is None:
            return "–"
        return f"{float(value):.{decimals[key]}f}" if decimals else f"{float(value):.3g}"

    ordered = sorted((c for c in components if c.get("R") is not None), key=lambda c: float(c["R"]))
    if not ordered:
        return f"R{equals}–"
    text = f"R{equals}{' / '.join(shown(c['R'], 'R') for c in ordered)} nm"
    if any(c.get("D") is not None for c in ordered):
        text += f", D{equals}{' / '.join(shown(c.get('D'), 'D') for c in ordered)} nm"
    return text


def points_text(points: Optional[dict]) -> str:
    """``596 of 598 points; 2 below the network's q window`` (English; "" without the counts)."""
    if not points or points.get("fitted") is None:
        return ""
    parts = [POINTS.format(fitted=points["fitted"], given=points.get("given", points["fitted"]))]
    if points.get("below_window"):
        parts.append(BELOW_WINDOW.format(count=points["below_window"]))
    if points.get("unusable"):
        parts.append(UNUSABLE.format(count=points["unusable"]))
    return "; ".join(parts)


def legend(populations: int = 1) -> str:
    """The keys of the saved tables and records with the names the texts use (one line)."""
    many = populations > 1
    pairs = [("R_n", "R (n)"), ("sigma_R_n", "σR/R (n)"), ("h_n", "h (n)"), ("sigma_h_n", "σh/h (n)"),
             ("D_n", "D (n)"), ("sigma_nn_n", "σnn/D (n)"), ("sigma_D", "σD/D = σnn/D / √(2π)"),
             ("amplitude / Int_n", "Scale (n)")] if many else [
        ("R_1", "R"), ("sigma_R_1", "σR/R"), ("h_1", "h"), ("sigma_h_1", "σh/h"), ("D_1", "D"),
        ("sigma_nn_1", "σnn/D"), ("sigma_D", "σD/D = σnn/D / √(2π)"), ("amplitude / Int_1", "Scale")]
    pairs += [("background", "Background"), ("res_amplitude / resolution_amplitude", "Tail A"),
              ("res_width / tail_width / sigma_Res", "Tail w"), ("res_exponent / tail_exponent / nu_Res", "Tail ν"),
              ("q_resolution", "σ_q"), ("chi2", "χ²ᵣ")]
    return "; ".join(f"{key}: {name}" for key, name in pairs)


__all__ = [
    "BELOW_WINDOW", "CHI2_DEFINITION", "EDGE", "MORE", "NAMES", "ONLY_ONE", "POINTS", "RELATIVE", "SENTENCES", "UNUSABLE",
    "at_edge", "clean_flags", "display_name", "flag_names", "grouped", "interval_digits", "interval_text", "legend", "of_population",
    "points_text", "sentence", "sentences", "sizes_text", "split_flags", "split_name", "unit_of",
]
