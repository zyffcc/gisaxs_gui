"""Size/structure table of an in-situ sputter series and what each value is worth.

Pure Python (no Qt): rows come from the toolkit's ``structure*.csv`` and
``structure*_summary.json``. Grades: A = converged physical refinement with a
unique solution, B = converged refinement with an equally good alternative,
C = network estimate only, P = network estimate on the physical path (one
stage path through time, no switching of the source from window to window),
Lit = the literature method (R from the nominal thickness and D = 2π/q* by mass
balance, as in the P03 in-situ GISAXS studies; ``literature.py``).
Typical errors are simulation-based (see summary ``reliability``); they
describe the hemisphere-equivalent model, not a proof that the sample has this
shape.

The reliability report is written from English templates; ``text(template,
**values)`` gives each sentence (the interface fills the translated template,
the default fills the English one), so the report reads the same in both
languages and the values stay as they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import math

GRADES = {
    "A": "refined, unique",
    "B": "refined, alternative exists",
    "C": "network only",
    "S": "time-consistent, data-supported",
    "T": "time-consistent, constraint-driven",
    "L": "learned selector",
    "P": "network estimate, physical path",
    "Lit": "literature method (mass balance)",
}
NUMERIC = {
    "time_s", "q_star_nm_inv", "D_corr_nm", "network_R_nm", "network_D_nm", "network_mode1_probability",
    "refined_R_nm", "refined_D_nm", "refined_deviance_per_dof_5pct", "coequal_alternatives",
    "R_nm", "R_typical_rel_error", "R_lo_nm", "R_hi_nm", "R_P90_rel_error", "R_order_accuracy",
    "D_nm", "D_typical_rel_error", "D_lo_nm", "D_hi_nm", "D_P90_rel_error", "D_order_accuracy",
    "L_TS_nm", "burial_nm_conditional", "thickness_nm_conditional", "D_over_direct", "sigma_log", "spacing_disorder",
    "TS_metal_fraction_conditional", "TS_film_thickness_nm_conditional", "sequence_data_cost_nats",
    "sequence_component_weight", "unconstrained_R_nm", "unconstrained_D_nm", "selector_probability",
    "network_top_R_nm", "network_top_D_nm",
    # the physical path (structure_physical.csv)
    "R_trend_nm", "D_trend_nm", "t_eq_nm", "t_eq_lo_nm", "t_eq_hi_nm", "nominal_thickness_nm",
    # the literature method (same file)
    "delta_nm", "D_lit_nm", "D_lit_trend_nm", "R_lit_nm", "R_lit_trend_nm", "two_R_over_D_lit", "two_R_over_D_lit_trend",
    "coverage_lit", "two_R_over_D_net", "coverage_net", "eta_net",
}
FLAGS = {"film", "path_changed"}


def number(value):
    """A finite float, or None for empty/reason text."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def flag(value) -> Optional[bool]:
    """``true`` / ``false`` of the toolkit's tables (also 1/0, yes/no); None when empty."""
    if isinstance(value, bool) or value is None:
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def parse_row(raw: dict) -> dict:
    row = {}
    for key, value in raw.items():
        if key in NUMERIC:
            row[key] = number(value)
        elif key in FLAGS:
            row[key] = flag(value)
        else:
            row[key] = None if value in ("", "None") else value
    return row


def is_film(row: dict) -> bool:
    """A film window: on the physical path its ``film`` flag, otherwise the per-window TS model."""
    if row.get("film") is not None:
        return bool(row["film"])
    return (row.get("module_path") or row.get("module")) == "TS_layer"


@dataclass
class Series:
    """One parameter over time for one grade (for plotting)."""
    time: list = field(default_factory=list)
    value: list = field(default_factory=list)
    low: list = field(default_factory=list)
    high: list = field(default_factory=list)


@dataclass
class StructureResult:
    folder: Path
    rows: list
    summary: dict
    variant: str = "window"   # "literature", "physical", "window", "selected" or "sequence"
    scan_folder: Optional[Path] = None
    """The scan the result was predicted from (``gimap_request.json``), None when the folder does not say."""

    @property
    def metal(self) -> str:
        return str(self.summary.get("metal") or (self.rows[0].get("metal") if self.rows else ""))

    def series(self, name: str, grade: str) -> Series:
        s = Series()
        for r in self.rows:
            if r.get(f"{name}_grade") != grade or r.get("time_s") is None or r.get(f"{name}_nm") is None:
                continue
            s.time.append(r["time_s"]); s.value.append(r[f"{name}_nm"])
            s.low.append(r.get(f"{name}_lo_nm") or r[f"{name}_nm"]); s.high.append(r.get(f"{name}_hi_nm") or r[f"{name}_nm"])
        return s

    def column(self, key: str, *, low: str = "", high: str = "") -> Series:
        """One numeric column over time (windows without a value are left out)."""
        s = Series()
        for r in self.rows:
            if r.get(key) is None or r.get("time_s") is None:
                continue
            s.time.append(r["time_s"]); s.value.append(r[key])
            s.low.append(r.get(low) if low and r.get(low) is not None else r[key])
            s.high.append(r.get(high) if high and r.get(high) is not None else r[key])
        return s

    def direct_spacing(self) -> Series:
        return self.column("D_corr_nm")

    def inconsistent_spacing(self) -> Series:
        """D values whose ratio to the direct 2π/q* lies outside the simulated 90% range."""
        s = Series()
        for r in self.rows:
            check = r.get("D_direct_check")
            if check and check != "consistent" and r.get("time_s") is not None and r.get("D_nm") is not None:
                s.time.append(r["time_s"]); s.value.append(r["D_nm"])
        return s

    def ts_length(self) -> Series:
        return self.column("L_TS_nm")

    def film_spans(self) -> list:
        """(first, last) time of each run of consecutive film windows."""
        spans, start, last = [], None, None
        for r in self.rows:
            t = r.get("time_s")
            if t is None:
                continue
            if is_film(r):
                start = t if start is None else start
                last = t
            elif start is not None:
                spans.append((start, last)); start = None
        if start is not None:
            spans.append((start, last))
        return spans

    def has(self, key: str) -> bool:
        return any(r.get(key) is not None for r in self.rows)

    def counts(self, name: str) -> dict:
        grades = [r.get(f"{name}_grade") for r in self.rows]
        return {g: grades.count(g) for g in GRADES} | {"missing": sum(g is None for g in grades)}


Text = Callable[..., str]


def english(template: str, **values) -> str:
    return template.format(**values)


def as_is(sentence: str) -> str:
    return sentence


READ_POINTS = ("<b>How to read this.</b> In the trend each window's value is a faint point and the line is their "
               "smoothed trend; the points are not error bars. The typical error of each value (simulation-based, "
               "about a 68% band) is in the Windows table.")
READ_ERROR_BARS = ("<b>How to read this.</b> Error bars in the trend are the simulation-based typical error of the "
                   "value's source (about a 68% band).")
READ_COMMON = ("About 10–20% of values carry a larger error (P90 ≈ 50–65%) because another structure fits the counts "
               "equally well; grade B marks windows where such an alternative was found. Trends are more reliable "
               "than absolute values: two windows whose true R differ by 20% are put in the right order in about "
               "92–96% of simulated cases. The window-to-window scatter is measured on your data from neighbouring "
               "windows with independent photons.")


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _grades_table(result: StructureResult, text: Text) -> list:
    s = result.summary
    lines = ["<table border='1' cellspacing='0' cellpadding='4'><tr>"
             + "".join(f"<th>{text(h)}</th>" for h in (
                 "Value", "Windows with a value", "By grade", "Typical error of these values",
                 "Order of two windows correct (≥1.2×)", "Window-to-window scatter (data)")) + "</tr>"]
    for name, label in (("R", "R, equivalent hemisphere radius"), ("D", "D, centre-to-centre spacing")):
        c = result.counts(name)
        by_grade = " · ".join(f"{g} {c[g]}" for g in GRADES if c[g]) or "–"
        errors = sorted(r[f"{name}_typical_rel_error"] for r in result.rows
                        if r.get(f"{name}_nm") is not None and r.get(f"{name}_typical_rel_error") is not None)
        orders = sorted(r[f"{name}_order_accuracy"] for r in result.rows
                        if r.get(f"{name}_nm") is not None and r.get(f"{name}_order_accuracy") is not None)
        typical = f"±{_pct(errors[len(errors) // 2])}" if errors else "–"
        if errors and errors[0] != errors[-1]:
            typical = f"±{_pct(errors[0])} – ±{_pct(errors[-1])}"
        order = _pct(orders[len(orders) // 2]) if orders else "–"
        scatter = (s.get(name) or {}).get("window_to_window_scatter")
        lines.append(f"<tr><td>{text(label)}</td><td>{sum(c[g] for g in GRADES)}</td><td>{by_grade}</td>"
                     f"<td>{typical}</td><td>{order}</td><td>{_pct(scatter)}</td></tr>")
    lines.append("</table>")
    return lines


def reliability_report(result: StructureResult, text: Optional[Text] = None,
                       note: Optional[Callable[[str], str]] = None) -> str:
    """Plain HTML: how much each value can be trusted, from simulation and from the data itself. ``text`` fills
    a template of this module, ``note`` gives a sentence of the toolkit itself (both in the interface language)."""
    from .literature import literature_paragraphs
    from .physical import physical_paragraphs
    from .variants import selector_paragraphs, sequence_paragraphs

    text = text or english
    note = note or as_is
    s = result.summary; rel = (s.get("reliability") or {}).get("parameters", {})
    lines = [f"<h3>{text('{name}: how reliable are the sizes?', name=result.folder.name)}</h3>"]
    lines.append("<p>" + text("{windows} time windows; {film} late windows use the TS film model (characteristic "
                              "length L instead of island R/D).", windows=len(result.rows),
                              film=s.get("TS_windows", 0)) + "</p>")
    lines += _grades_table(result, text)
    lines.append("<p>" + text("Grades:") + " " + "; ".join(f"<b>{g}</b> {text(t)}" for g, t in GRADES.items()) + ".</p>")
    if result.variant == "literature":
        lines += literature_paragraphs(result, text, note)
    if result.variant in ("physical", "literature"):
        lines += physical_paragraphs(result, text, note)
    if result.variant == "sequence":
        lines += sequence_paragraphs(result, text)
    if result.variant == "selected":
        lines += selector_paragraphs(result, text)
    d = s.get("D") or {}
    if d.get("checked_against_direct"):
        band = (rel.get("D_over_direct") or {}).get("band_90", [0.59, 1.75])
        lines.append("<p>" + text(
            "<b>Check against the data.</b> {outside} of {checked} D values differ from the direct 2π/q* by more than "
            "the simulated 90% range (D/(2π/q*) outside {low}–{high}); they are circled in red and should not be used.",
            outside=d.get("outside_direct_band", 0), checked=d["checked_against_direct"], low=f"{band[0]:.2f}",
            high=f"{band[1]:.2f}") + "</p>")
    lines.append("<p>" + text(READ_POINTS if result.variant in ("physical", "literature") else READ_ERROR_BARS) + " "
                 + text(READ_COMMON) + "</p>")
    lines.append("<p>" + text(
        "<b>Model conditions.</b> R is the base radius of a flat-based hemisphere (height h = R). Real clusters may "
        "be taller (h/R between 1 and 2); absolute R depends on this condition, trends much less. D is given only "
        "where a correlation peak is resolved; × marks the direct value 2π/q*. Burial depth and polymer thickness are "
        "not reliable (order accuracy 52–80%) and are only listed as conditional values.") + "</p>")
    if result.metal == "Cu":
        lines.append("<p><b>" + text("Cu: approximated with the Ag model, for reference only.") + "</b> " + text(
            "The networks were trained for Ag and Au; a Cu series is read with the Ag model, so its sizes are a "
            "reference, not a measurement.") + "</p>")
    return "\n".join(lines)


__all__ = ["FLAGS", "GRADES", "READ_COMMON", "READ_ERROR_BARS", "READ_POINTS", "Series", "StructureResult", "as_is",
           "english", "flag", "is_film", "number", "parse_row", "reliability_report"]
