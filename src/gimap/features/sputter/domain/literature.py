"""The literature method of a sputter series and its comparison with published growth (pure Python).

The toolkit writes it with a deposition rate into ``structure_physical.csv`` (``delta_nm``, ``R_lit_nm``,
``D_lit_nm``, their trends, ``two_R_over_D_*``, ``coverage_*``, ``eta_net``, ``lit_status``, ``lit_reason``) and
``structure_physical_summary.json`` (``literature``: the percolation, the published band, reference curves and the
checks). As in the P03 in-situ GISAXS studies, R is not fitted: it follows from the nominal thickness δ and the
spacing D = 2π/q* of the direct peak by mass balance for hemispheres (h = R) on a hexagonal lattice,
δ = 2.418·R³/D². δ = rate × (t − t0) up to the end of the deposition t_end when it is given, constant after it. It
agrees with the literature by construction; the independent tests are D(δ), the percolation δ against the published
band and the network's coverage and mass balance.

The page's method is the network: its physical path is always the default explanation. The ``literature``
explanation is a check of it, offered second (and in the Literature tab), never the default. It shows these values
as R and D (grade ``Lit``); after the percolation the windows count as film: D = 2π/q* stays, R is not given. That
is this page's choice (molecular dynamics, Abr16, finds that R
loses its island meaning once a network forms); the P03 studies go on reporting R and 2R/D after it. The
percolation's interval holds the scatter of the windows and the errors of the rate and t0 (the toolkit's
``rate_error`` and ``t0_error_s``); when 2R/D may reach 1 within them but need not, the percolation is not
determined. The report is written from English templates (``text``); the toolkit's own sentences go through
``note``.
"""

from __future__ import annotations

from html import escape
from typing import Optional

LITERATURE_GRADE = "Lit"
VERDICT_MARKS = {"consistent": "✓", "inconsistent": "×", "not testable": "–"}
VERDICT_COLOURS = {"consistent": "#2a8a3e", "inconsistent": "#c0392b", "not testable": "#7f7f7f"}
LITERATURE_COLUMNS = {"R_nm": "R_lit_nm", "D_nm": "D_lit_nm", "R_trend_nm": "R_lit_trend_nm",
                      "D_trend_nm": "D_lit_trend_nm"}
NOT_LITERATURE = ("film", "L_TS_nm", "burial_nm_conditional", "thickness_nm_conditional", "sigma_log", "spacing_disorder",
                  "D_over_direct", "D_direct_check")
METHOD_PARAGRAPH = (
    "<b>Literature method (mass balance), a check of the network.</b> As in the P03 in-situ GISAXS studies, R is not "
    "fitted: it follows from the nominal thickness δ = rate × (t − t0) (constant after the end of the deposition) and "
    "the spacing D = 2π/q* of the direct peak by mass balance for hemispheres (h = R) on a hexagonal lattice, "
    "δ = 2.418·R³/D² (grade <b>Lit</b>). It agrees with the literature by "
    "construction; the Literature tab shows the independent tests (D against δ, the percolation against the published "
    "band, the network's 2R/D and mass balance). The simulation-based typical errors do not apply: D = 2π/q* can "
    "be off by more than 40% for D &gt; 10 nm once coalescence starts, and R follows D as D^(2/3) and the rate as "
    "rate^(1/3).")
PERCOLATION_AT = ("Percolation (2R/D = 1) at δ = {delta} nm, t = {time} s (68% interval {low}–{high} nm from the "
                  "scatter of the windows, the rate ±{rate_error}% and t0 ±{t0_error} s; a systematic error of 2π/q* "
                  "is not in it); after it D = 2π/q* stays and R is not given (film).")
PERCOLATION_UNDETERMINED = ("Percolation not determined: within the scatter of the windows, the rate ±{rate_error}% "
                            "and t0 ±{t0_error} s, 2R/D reaches 1 from δ = {low} nm, or not before the peak is lost "
                            "at δ = {last} nm. R is given throughout.")
PERCOLATION_FIRST = ("2R/D is at or above 1 from the first window with a peak after nucleation: percolation at "
                     "δ ≤ {delta} nm (t = {time} s).")
PERCOLATION_NONE = ("2R/D stays below 1 (at most {ratio}) up to δ = {delta} nm, where the peak is lost, also with a "
                    "{rate_error}% higher rate and t0 {t0_error} s earlier: no percolation while a peak is resolved.")
SHORT_AT = "percolation (literature method) at δ = {delta} nm, t = {time} s"
SHORT_FIRST = "percolation (literature method) at δ ≤ {delta} nm"
SHORT_UNDETERMINED = "percolation (literature method) not determined: from δ = {low} nm or later"
SHORT_NONE = "2R/D below 1 up to δ = {delta} nm (no percolation while a peak is resolved)"
CHECK_COUNTS = ("Checks against the literature: {consistent} consistent, {inconsistent} inconsistent, {open} not "
                "testable (Literature tab).")
NEEDS_RATE_PARAGRAPH = ("<b>Literature method.</b> It needs the deposition rate: enter it under More options ▸ "
                        "Deposition rate (nm/s). The files are rewritten without reading the frames again.")
CHECK_HEADERS = ("What is compared", "Verdict", "Measured", "Expected (literature)", "References")
VERDICTS = {"consistent": "consistent", "inconsistent": "inconsistent", "not testable": "not testable"}
REFERENCES_TITLE = "References"
NOTES_TITLE = "Caveats and notes"


def block(summary: dict) -> dict:
    value = (summary or {}).get("literature")
    return value if isinstance(value, dict) else {}


def literature_available(summary: dict) -> bool:
    """The literature method has values (a rate was given and enough windows show the metal's peak): it is offered
    as a check of the network, after the physical path."""
    return block(summary).get("status") == "ok"


def _float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def percolation_state(perc: dict) -> str:
    """``reached``, ``first_window``, ``not_determined``, ``not_reached`` or "" (also for a result written before
    the toolkit named the state)."""
    state = perc.get("state")
    if state:
        return str(state)
    high = (list(perc.get("interval_delta_nm") or []) + [None, None])[1]
    if perc.get("delta_nm") is not None:
        return "first_window" if perc.get("bound") == "upper" else ("reached" if high is not None else "not_determined")
    return "not_reached" if perc.get("bound") == "lower" else ""


def percolation(summary: dict) -> tuple[Optional[float], Optional[tuple]]:
    """Time (s) of the literature-method percolation and its 68% interval (low, high).

    The time is None when the percolation is not reached or not determined; one that is not determined still has
    its interval (low, None): 2R/D may reach 1 from ``low`` on, or not at all. An open end is None, never the time
    of the percolation itself."""
    perc = block(summary).get("percolation") or {}
    state = percolation_state(perc)
    when = _float(perc.get("time_s")) if state in ("reached", "first_window") else None
    interval = perc.get("interval_time_s") or None
    try:
        low = _float(interval[0]) if interval else None
        high = _float(interval[1]) if interval else None
    except (TypeError, IndexError):
        low = high = None
    if low is None or state not in ("reached", "first_window", "not_determined"):
        return when, None
    return when, (low, high)


def literature_rows(rows: list, summary: Optional[dict] = None) -> list:
    """The rows of ``structure_physical.csv`` as the literature explanation shows them: R_lit/D_lit as R/D (grade
    Lit), film after the percolation (every window after its time, also one without a peak or with a peak the
    method does not use), no network shape values (the method knows none)."""
    when, _interval = percolation(summary or {})
    out = []
    for r in rows:
        row = dict(r)
        for key, source in LITERATURE_COLUMNS.items():
            row[key] = r.get(source)
        for name in ("R", "D"):
            has = row[f"{name}_nm"] is not None
            row.update({f"{name}_grade": LITERATURE_GRADE if has else None,
                        f"{name}_source": "literature method" if has else None,
                        f"{name}_reason": "" if has else (r.get("lit_reason") or ""),
                        f"{name}_typical_rel_error": None, f"{name}_lo_nm": None, f"{name}_hi_nm": None,
                        f"{name}_P90_rel_error": None, f"{name}_order_accuracy": None})
        for key in NOT_LITERATURE:
            row[key] = None
        time = _float(r.get("time_s"))
        row["film"] = r.get("lit_status") == "after_percolation" or (
            when is not None and time is not None and time > when and r.get("lit_status") != "no_data")
        if row["D_nm"] is not None:
            row["D_direct_check"] = "consistent"  # D is the direct 2π/q* itself
        out.append(row)
    return out


def onset_of(result) -> tuple[Optional[float], Optional[tuple], str]:
    """(time, interval, "percolation" or "film"): what the trend marks as the start of the film."""
    from .physical import film_onset

    if result.variant == "literature":
        when, interval = percolation(result.summary)
        return when, interval, "percolation"
    when, interval = film_onset(result.summary)
    return when, interval, "film"


def _fmt(value) -> str:
    value = _float(value)
    return "–" if value is None else f"{value:.3g}"


def errors(perc: dict) -> dict:
    """The rate error (%) and t0 error (s) the percolation interval holds (0 for a result made without them)."""
    rate_error = _float(perc.get("rate_error")) or 0.0
    return {"rate_error": _fmt(100 * rate_error), "t0_error": _fmt(_float(perc.get("t0_error_s")) or 0.0)}


def percolation_last_time(summary: dict) -> Optional[float]:
    """Time (s) of the last window of the peak phase the percolation was looked for in (None when not known)."""
    return _float((block(summary).get("percolation") or {}).get("last_time_s"))


def percolation_errors(summary: dict) -> dict:
    """``errors`` of a summary's literature-method percolation (for a label)."""
    return errors(block(summary).get("percolation") or {})


def percolation_sentence(summary: dict, text) -> str:
    perc = block(summary).get("percolation") or {}
    state = percolation_state(perc)
    low, high = (list(perc.get("interval_delta_nm") or []) + [None, None])[:2]
    if state == "first_window":
        return text(PERCOLATION_FIRST, delta=_fmt(perc.get("delta_nm")), time=_fmt(perc.get("time_s")))
    if state == "reached":
        return text(PERCOLATION_AT, delta=_fmt(perc.get("delta_nm")), time=_fmt(perc.get("time_s")), low=_fmt(low),
                    high=_fmt(high), **errors(perc))
    if state == "not_determined":
        return text(PERCOLATION_UNDETERMINED, low=_fmt(low), last=_fmt(perc.get("last_delta_nm")), **errors(perc))
    if state == "not_reached":
        return text(PERCOLATION_NONE, ratio=_fmt(perc.get("max_two_R_over_D_trend")),
                    delta=_fmt(perc.get("last_delta_nm")), **errors(perc))
    return ""


def percolation_short(summary: dict, text) -> str:
    """The percolation of the literature method in a few words (the result line)."""
    perc = block(summary).get("percolation") or {}
    state = percolation_state(perc)
    if state == "first_window":
        return text(SHORT_FIRST, delta=_fmt(perc.get("delta_nm")))
    if state == "reached":
        return text(SHORT_AT, delta=_fmt(perc.get("delta_nm")), time=_fmt(perc.get("time_s")))
    if state == "not_determined":
        return text(SHORT_UNDETERMINED, low=_fmt((list(perc.get("interval_delta_nm") or []) + [None])[0]))
    if state == "not_reached":
        return text(SHORT_NONE, delta=_fmt(perc.get("last_delta_nm")))
    return ""


def verdict_counts(summary: dict) -> dict:
    checks = block(summary).get("checks") or []
    return {v: sum(1 for c in checks if c.get("verdict") == v) for v in VERDICTS}


def literature_paragraphs(result, text, note=lambda sentence: sentence) -> list:
    """The Reliability tab's part about the literature method (its explanation is shown)."""
    summary = result.summary
    if not literature_available(summary):
        return ["<p>" + text(NEEDS_RATE_PARAGRAPH) + "</p>"]
    lines = ["<p>" + text(METHOD_PARAGRAPH) + "</p>"]
    sentence = percolation_sentence(summary, text)
    if sentence:
        lines.append("<p>" + sentence + "</p>")
    counts = verdict_counts(summary)
    lines.append("<p>" + text(CHECK_COUNTS, consistent=counts["consistent"], inconsistent=counts["inconsistent"],
                              open=counts["not testable"]) + "</p>")
    notes = [words for words in block(summary).get("notes") or [] if isinstance(words, str) and words]
    if notes:
        lines.append("<ul>" + "".join(f"<li>{escape(note(words))}</li>" for words in notes) + "</ul>")
    return lines


def checks_html(summary: dict, text, note=lambda sentence: sentence) -> str:
    """The checks as a table, the notes and the references with their DOI (plain HTML)."""
    lit = block(summary)
    head = "".join(f"<th>{text(h)}</th>" for h in CHECK_HEADERS)
    rows = []
    for check in lit.get("checks") or []:
        verdict = check.get("verdict") or "not testable"
        mark = VERDICT_MARKS.get(verdict, "?")
        colour = VERDICT_COLOURS.get(verdict, "#7f7f7f")
        notes = " ".join(escape(note(n)) for n in check.get("notes") or [] if n)
        measured = escape(note(check.get("measured") or ""))
        rows.append(
            f"<tr><td><b>{escape(note(check.get('title') or check.get('id') or ''))}</b></td>"
            f"<td style='color:{colour}'><b>{mark} {escape(text(VERDICTS.get(verdict, verdict)))}</b></td>"
            f"<td>{measured}" + (f"<br><i>{notes}</i>" if notes else "") + "</td>"
            f"<td>{escape(note(check.get('expected') or ''))}</td>"
            f"<td>{escape(', '.join(check.get('references') or []))}</td></tr>")
    html = [f"<table border='1' cellspacing='0' cellpadding='4'><tr>{head}</tr>{''.join(rows)}</table>"]
    notes = [n for n in lit.get("notes") or [] if isinstance(n, str) and n]
    if notes:
        html.append(f"<h4>{text(NOTES_TITLE)}</h4><ul>"
                    + "".join(f"<li>{escape(note(n))}</li>" for n in notes) + "</ul>")
    refs = lit.get("references") or {}
    if refs:
        items = []
        for key, ref in sorted(refs.items()):
            link = ref.get("doi") and f"https://doi.org/{ref['doi']}" or ref.get("url") or ""
            items.append(f"<li><b>{escape(key)}</b> {escape(ref.get('citation') or '')}"
                         + (f" — <a href='{escape(link)}'>{escape(ref.get('doi') or link)}</a>" if link else "") + "</li>")
        html.append(f"<h4>{text(REFERENCES_TITLE)}</h4><ul>{''.join(items)}</ul>")
    return "\n".join(html)


TEMPLATES = (METHOD_PARAGRAPH, PERCOLATION_AT, PERCOLATION_UNDETERMINED, PERCOLATION_FIRST, PERCOLATION_NONE,
             SHORT_AT, SHORT_FIRST, SHORT_UNDETERMINED, SHORT_NONE, CHECK_COUNTS,
             NEEDS_RATE_PARAGRAPH, *CHECK_HEADERS, *VERDICTS.values(), REFERENCES_TITLE, NOTES_TITLE)
"""The texts this module writes (the interface translates them)."""

__all__ = ["LITERATURE_GRADE", "TEMPLATES", "VERDICT_COLOURS", "VERDICT_MARKS", "block", "checks_html",
           "literature_available", "literature_paragraphs", "literature_rows", "onset_of",
           "percolation", "percolation_errors", "percolation_last_time", "percolation_sentence", "percolation_short",
           "percolation_state", "verdict_counts"]
