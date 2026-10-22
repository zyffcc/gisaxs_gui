"""Reliability paragraphs of the optional explanations: the time-consistent path and the learned selector.

Written from English templates like the rest of the report (``structure.reliability_report``).
"""

from __future__ import annotations


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def sequence_paragraphs(result, text) -> list:
    seq = result.summary.get("sequence")
    if not seq:
        return []
    cR = result.counts("R")
    first = seq.get("first_film_time_s")
    film = text("{time} s", time=f"{first:.1f}") if first is not None else text("never")
    return [
        "<p>" + text(
            "<b>Time-consistent path.</b> All network explanations of every window were considered and one path "
            "through time was chosen with registered physical rules (sizes and spacing do not shrink quickly; a film "
            "does not revert to islands; D should agree with 2π/q*). {changed} of {windows} windows changed from the "
            "per-window choice. Grade <b>S</b> ({supported} R values): the data support the chosen explanation (within "
            "2 nats of the best). Grade <b>T</b> ({imposed} R values): the rules imposed it against the data — read "
            "these as the story, not as a measurement. Film from {film} on this path; when the observation-rule labels "
            "switch back and forth, this time is uncertain. The path can only choose among the network's explanations: "
            "where none fits (e.g. a steep drop that every explanation shows) the drop stays.",
            changed=seq.get("changed_from_unconstrained", 0), windows=len(result.rows), supported=cR["S"],
            imposed=cR["T"], film=film) + "</p>",
        "<p>" + text(
            "<b>Exploratory — not more accurate.</b> In the registered test on 900 simulated growth windows this path "
            "was <i>not</i> more accurate than the per-window choice (R wrong by &gt;30%: 26% vs 28% of windows; median "
            "R error 16% vs 15%; D unchanged), and it erased a real 25% dip of R in 93% of the dip windows. Use it to "
            "see one consistent picture, not as a more reliable measurement. The correct explanation was among the "
            "network's candidates in about 92% of windows; a better way of choosing it is being developed.") + "</p>",
    ]


def selector_paragraphs(result, text) -> list:
    sel = result.summary.get("selector")
    if sel is None:
        return []
    test = sel.get("test") or {}
    lines = ["<p>" + text(
        "<b>Learned selector.</b> For every window the network offers four explanations (mixture components); a model "
        "trained on simulated growth series picks one from their network weights, the agreement of D with 2π/q* and "
        "how well each agrees with the explanations of the three windows before and after. {low} windows have a "
        "selection probability below 0.5.", low=sel.get("low_confidence", 0)) + "</p>"]
    rows = []
    for key, label in (("R", "R"), ("D", text("D (peak windows)"))):
        b = test.get(key) or {}
        top, chosen = b.get("top") or {}, b.get("sel") or {}
        if top and chosen:
            rows.append(f"<tr><td>{label}</td><td>{_pct(top.get('wrong_branch_frac'))} / "
                        f"{_pct(top.get('median_rel_error'))}</td><td>{_pct(chosen.get('wrong_branch_frac'))} / "
                        f"{_pct(chosen.get('median_rel_error'))}</td>"
                        f"<td>{_pct((b.get('oracle') or {}).get('wrong_branch_frac'))}</td></tr>")
    if not rows:
        return lines
    parts = []
    for key in ("R", "D"):
        b = test.get(key) or {}
        if "pass" in b:
            parts.append(f"{key}: " + (text("passed") if b["pass"] else "<b>" + text("not passed") + "</b>"))
    verdict = "; ".join(parts) or (text("passed") if test.get("passed") else "<b>" + text("not passed") + "</b>")
    lines.append("<table border='1' cellspacing='0' cellpadding='4'><tr>" + "".join(f"<th>{text(h)}</th>" for h in (
        "Independent simulated TEST", "Network top: wrong &gt;30% / median error",
        "Selector: wrong &gt;30% / median error", "Best possible (oracle)")) + "</tr>" + "".join(rows) + "</table>")
    lines.append("<p>" + text(
        "Registered test (wrong-branch share cut by at least 30% and median error not larger): {verdict}. The selector "
        "is therefore offered as an option, not as the default, unless both pass. Its typical errors in the trend come "
        "from that test. On real scans compare its D with 2π/q* (red circles).", verdict=verdict) + "</p>")
    return lines


__all__ = ["selector_paragraphs", "sequence_paragraphs"]
