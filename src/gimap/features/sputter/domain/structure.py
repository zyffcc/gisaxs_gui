"""Size/structure table of an in-situ sputter series and what each value is worth.

Pure Python (no Qt): rows come from the toolkit's ``structure.csv`` and
``structure_summary.json``. Grades: A = converged physical refinement with a
unique solution, B = converged refinement with an equally good alternative,
C = network estimate only. Typical errors are simulation-based (see summary
``reliability``); they describe the hemisphere-equivalent model, not a proof
that the sample has this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

GRADES = {
    "A": "refined, unique",
    "B": "refined, alternative exists",
    "C": "network only",
    "S": "time-consistent, data-supported",
    "T": "time-consistent, constraint-driven",
    "L": "learned selector",
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
}


def number(value):
    """A finite float, or None for empty/reason text."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_row(raw: dict) -> dict:
    row = {}
    for key, value in raw.items():
        if key in NUMERIC:
            row[key] = number(value)
        else:
            row[key] = None if value in ("", "None") else value
    return row


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
    variant: str = "window"   # "window": per-window choice; "sequence": time-consistent path

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

    def direct_spacing(self) -> Series:
        s = Series()
        for r in self.rows:
            if r.get("D_corr_nm") is not None and r.get("time_s") is not None:
                s.time.append(r["time_s"]); s.value.append(r["D_corr_nm"])
        return s

    def inconsistent_spacing(self) -> Series:
        """D values whose ratio to the direct 2π/q* lies outside the simulated 90% range."""
        s = Series()
        for r in self.rows:
            check = r.get("D_direct_check")
            if check and check != "consistent" and r.get("time_s") is not None and r.get("D_nm") is not None:
                s.time.append(r["time_s"]); s.value.append(r["D_nm"])
        return s

    def ts_length(self) -> Series:
        s = Series()
        for r in self.rows:
            if r.get("L_TS_nm") is not None and r.get("time_s") is not None:
                s.time.append(r["time_s"]); s.value.append(r["L_TS_nm"])
        return s

    def counts(self, name: str) -> dict:
        grades = [r.get(f"{name}_grade") for r in self.rows]
        return {g: grades.count(g) for g in GRADES} | {"missing": sum(g is None for g in grades)}


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def reliability_report(result: StructureResult) -> str:
    """Plain HTML: how much each value can be trusted, from simulation and from the data itself."""
    s = result.summary; rel = (s.get("reliability") or {}).get("parameters", {})
    metal = "Ag" if result.metal == "Cu" else (result.metal or "Ag")
    lines = [f"<h3>{result.folder.name}: how reliable are the sizes?</h3>"]
    lines.append(f"<p>{len(result.rows)} time windows; {s.get('TS_windows', 0)} late windows use the TS film model "
                 "(characteristic length L instead of island R/D).</p>")
    lines.append("<table border='1' cellspacing='0' cellpadding='4'>"
                 "<tr><th>Value</th><th>Windows with a value</th><th>By grade</th><th>Typical error of these values</th>"
                 "<th>Order of two windows correct (≥1.2×)</th><th>Window-to-window scatter (data)</th></tr>")
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
        lines.append(f"<tr><td>{label}</td><td>{sum(c[g] for g in GRADES)}</td><td>{by_grade}</td><td>{typical}</td>"
                     f"<td>{order}</td><td>{_pct(scatter)}</td></tr>")
    lines.append("</table>")
    lines.append("<p>Grades: " + "; ".join(f"<b>{g}</b> {text}" for g, text in GRADES.items()) + ".</p>")
    seq = s.get("sequence")
    if result.variant == "sequence" and seq:
        cR = result.counts("R")
        first = seq.get("first_film_time_s")
        lines.append(
            f"<p><b>Time-consistent path.</b> All network explanations of every window were considered and one path "
            f"through time was chosen with registered physical rules (sizes and spacing do not shrink quickly; a film does "
            f"not revert to islands; D should agree with 2π/q*). {seq.get('changed_from_unconstrained', 0)} of "
            f"{len(result.rows)} windows changed from the per-window choice. Grade <b>S</b> ({cR['S']} R values): the data "
            f"support the chosen explanation (within 2 nats of the best). Grade <b>T</b> ({cR['T']} R values): the rules "
            f"imposed it against the data — read these as the story, not as a measurement. Film from "
            f"{'%.1f s' % first if first is not None else 'never'} on this path; when the observation-rule labels switch "
            f"back and forth, this time is uncertain. The path can only choose among the network's explanations: where none "
            f"fits (e.g. a steep drop that every explanation shows) the drop stays.</p>"
            f"<p><b>Exploratory — not more accurate.</b> In the registered test on 900 simulated growth windows this path "
            f"was <i>not</i> more accurate than the per-window choice (R wrong by &gt;30%: 26% vs 28% of windows; median R "
            f"error 16% vs 15%; D unchanged), and it erased a real 25% dip of R in 93% of the dip windows. Use it to see "
            f"one consistent picture, not as a more reliable measurement. The correct explanation was among the network's "
            f"candidates in about 92% of windows; a better way of choosing it is being developed.</p>")
    sel = s.get("selector")
    if result.variant == "selected" and sel is not None:
        test = sel.get("test") or {}
        lines.append("<p><b>Learned selector.</b> For every window the network offers four explanations (mixture "
                     "components); a model trained on simulated growth series picks one from their network weights, the "
                     "agreement of D with 2π/q* and how well each agrees with the explanations of the three windows before "
                     f"and after. {sel.get('low_confidence', 0)} windows have a selection probability below 0.5.</p>")
        rows = []
        for key, label in (("R", "R"), ("D", "D (peak windows)")):
            b = test.get(key) or {}
            top, chosen = b.get("top") or {}, b.get("sel") or {}
            if top and chosen:
                rows.append(f"<tr><td>{label}</td><td>{_pct(top.get('wrong_branch_frac'))} / {_pct(top.get('median_rel_error'))}</td>"
                            f"<td>{_pct(chosen.get('wrong_branch_frac'))} / {_pct(chosen.get('median_rel_error'))}</td>"
                            f"<td>{_pct((b.get('oracle') or {}).get('wrong_branch_frac'))}</td></tr>")
        if rows:
            parts = []
            for key in ("R", "D"):
                b = test.get(key) or {}
                if "pass" in b:
                    parts.append(f"{key}: {'passed' if b['pass'] else '<b>not passed</b>'}")
            verdict = "; ".join(parts) or ("passed" if test.get("passed") else "<b>not passed</b>")
            lines.append("<table border='1' cellspacing='0' cellpadding='4'><tr><th>Independent simulated TEST</th>"
                         "<th>Network top: wrong &gt;30% / median error</th><th>Selector: wrong &gt;30% / median error</th>"
                         "<th>Best possible (oracle)</th></tr>" + "".join(rows) + "</table>")
            lines.append(f"<p>Registered test (wrong-branch share cut by at least 30% and median error not larger): {verdict}. "
                         "The selector is therefore offered as an option, not as the default, unless both pass. Its typical "
                         "errors in the trend come from that test. On real scans compare its D with 2π/q* (red circles).</p>")
    d = s.get("D") or {}
    if d.get("checked_against_direct"):
        band = (rel.get("D_over_direct") or {}).get("band_90", [0.59, 1.75])
        lines.append(f"<p><b>Check against the data.</b> {d.get('outside_direct_band', 0)} of {d['checked_against_direct']} D "
                     f"values differ from the direct 2π/q* by more than the simulated 90% range (D/(2π/q*) outside "
                     f"{band[0]:.2f}–{band[1]:.2f}); they are circled in red and should not be used.</p>")
    lines.append("<p><b>How to read this.</b> Error bars in the trend are the simulation-based typical error of the "
                 "value's source (about a 68% band). About 10–20% of values carry a larger error (P90 ≈ 50–65%) because "
                 "another structure fits the counts equally well; grade B marks windows where such an alternative was "
                 "found. Trends are more reliable than absolute values: two windows whose true R differ by 20% are put in "
                 "the right order in about 92–96% of simulated cases. The window-to-window scatter is measured on your "
                 "data from neighbouring windows with independent photons.</p>")
    lines.append("<p><b>Model conditions.</b> R is the base radius of a flat-based hemisphere (height h = R). Real "
                 "clusters may be taller (h/R between 1 and 2); absolute R depends on this condition, trends much less. "
                 "D is given only where a correlation peak is resolved; × marks the direct value 2π/q*. Burial depth and "
                 "polymer thickness are not reliable (order accuracy 52–80%) and are only listed as conditional values. "
                 "Cu is approximated by the Ag model.</p>")
    return "\n".join(lines)


__all__ = ["GRADES", "Series", "StructureResult", "parse_row", "reliability_report", "number"]
