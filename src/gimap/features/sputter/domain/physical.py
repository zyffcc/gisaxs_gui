"""The physical path of a sputter series: what it assumes, what it changed and what the thickness values mean.

``structure_physical.csv`` and ``structure_physical_summary.json`` (written by the toolkit's ``analyze``,
``table`` and ``physical`` commands) hold one stage path through time (pre-nucleation → islands → coalescence
→ film; a film does not revert to islands), the network estimate of R and D in every island window (grade P;
no switching between refined and network values from window to window), smoothed trend lines, the
island-volume equivalent thickness t_eq, the nominal thickness rate·(t − t0) when a deposition rate is given,
the film onset with its interval and the period of a periodic intensity modulation (sample rotation).
Written from English templates (``text``) like the rest of the reliability report; the toolkit's own notes go
through ``note`` (the interface gives the sentences it knows in its language, the others as they are).
"""

from __future__ import annotations

from html import escape
from typing import Optional

REVERSAL_ROWS = (("stage", "Stage steps back"), ("module_revert", "Film → islands"),
                 ("R_zigzag", "R zig-zags"), ("D_zigzag", "D zig-zags"))
FILM_NOT_DETERMINED = "film: thickness not determined"


def _seconds(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.1f}"


def film_onset(summary: dict) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """The film onset (s) of the physical path and its interval, when the path reaches a film."""
    onset = summary.get("film_onset_s")
    interval = summary.get("film_onset_interval_s")
    try:
        onset = None if onset is None else float(onset)
    except (TypeError, ValueError):
        onset = None
    try:
        interval = None if not interval else (float(interval[0]), float(interval[1]))
    except (TypeError, ValueError, IndexError):
        interval = None
    return onset, interval


def rotation_period(summary: dict) -> Optional[float]:
    try:
        value = summary.get("rotation_period_s")
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _reversal_table(reversals: dict, text) -> list:
    before, after = reversals.get("before") or {}, reversals.get("after") or {}
    if not before and not after:
        return []
    head = "".join(f"<th>{text(h)}</th>" for h in ("Back and forth", "Per-window rule", "Physical path"))
    rows = "".join(f"<tr><td>{text(label)}</td><td>{before.get(key, '–')}</td><td>{after.get(key, '–')}</td></tr>"
                   for key, label in REVERSAL_ROWS)
    return [f"<table border='1' cellspacing='0' cellpadding='4'><tr>{head}</tr>{rows}</table>"]


def _thickness(summary: dict, text, note) -> list:
    thickness = summary.get("thickness") or {}
    lines = ["<p>" + text(
        "<b>Metal thickness.</b> t_eq is the island-volume equivalent thickness: the metal in the visible islands, "
        "for hemispheres (h = R) on a hexagonal arrangement with spacing D. It is a conditional value with an error "
        "of about a factor 2–3, and it is not given in film windows.") + "</p>"]
    rate = thickness.get("rate_nm_s")
    if rate is not None:
        t0 = thickness.get("t0_s")
        lines.append("<p>" + text(
            "Nominal thickness = rate × (t − t0) with rate {rate} nm/s and t0 = {t0} s: what was deposited, not what "
            "is measured.", rate=f"{float(rate):.4g}", t0=_seconds(None if t0 is None else float(t0))) + "</p>")
    else:
        lines.append("<p>" + text("Give the deposition rate (More options) to draw the nominal thickness "
                                  "rate × (t − t0).") + "</p>")
    lines.append("<p>" + text(
        "In film windows the thickness is not determined: the current model cannot measure the thickness of a "
        "continuous film (in simulation its estimate equals the prior). A model of a film with particles on it is "
        "planned.") + "</p>")
    mass = thickness.get("mass_continuity")
    if isinstance(mass, dict) and mass.get("ratio") is not None:
        lines.append("<p>" + text(
            "Mass check at the film onset ({time} s): island-volume thickness {teq} nm against the nominal {nominal} "
            "nm (ratio {ratio}). A ratio far below 1 means that much of the metal is not in the visible islands "
            "(smaller clusters, embedded metal, a wetting layer) or that the hemisphere condition does not hold.",
            time=_seconds(mass.get("time_s")), teq=f"{float(mass.get('t_eq_nm') or 0):.3g}",
            nominal=f"{float(mass.get('nominal_nm') or 0):.3g}", ratio=f"{float(mass['ratio']):.2g}") + "</p>")
    notes = [thickness.get(key) for key in ("t_eq_note", "nominal_note", "film_thickness_note")]
    notes = [words for words in notes if isinstance(words, str) and words]
    if notes:
        lines.append("<ul>" + "".join(f"<li>{escape(note(words))}</li>" for words in notes) + "</ul>")  # the model's
    return lines


def physical_paragraphs(result, text, note=lambda sentence: sentence) -> list:
    summary = result.summary
    changed = sum(1 for r in result.rows if r.get("path_changed"))
    lines = ["<p>" + text(
        "<b>Physical path (default).</b> One path of stages through time — pre-nucleation, islands, coalescence, "
        "film — in which a film does not revert to islands. R and D are the network estimate in every island window "
        "(grade P): the source does not switch between refined and network values from window to window (refined "
        "values stay in the Windows table). The trend lines are smoothed over neighbouring island windows. {changed} "
        "of {windows} windows differ from the per-window rule.", changed=changed, windows=len(result.rows)) + "</p>"]
    onset, interval = film_onset(summary)
    if onset is not None and interval is not None:
        lines.append("<p>" + text("Film from {onset} s on this path (onset between {low} and {high} s).",
                                  onset=_seconds(onset), low=_seconds(interval[0]), high=_seconds(interval[1])) + "</p>")
    elif onset is not None:
        lines.append("<p>" + text("Film from {onset} s on this path.", onset=_seconds(onset)) + "</p>")
    else:
        lines.append("<p>" + text("No film window on this path.") + "</p>")
    reversals = summary.get("reversals") or {}
    if reversals:
        lines.append("<p>" + text("Back-and-forth changes before (per-window rule) and after the physical path:") + "</p>")
        lines += _reversal_table(reversals, text)
    period = rotation_period(summary)
    detection = summary.get("rotation_detection") or {}
    if period is not None:
        acf, cycles = detection.get("acf_peak"), detection.get("cycles")
        lines.append("<p>" + text(
            "The intensity is modulated with a period of about {period} s (autocorrelation peak {acf}, {cycles} "
            "cycles): sample rotation. A single window shorter than this period sees part of a turn; the physical "
            "path and the trend lines average over it.", period=f"{period:.3g}",
            acf="–" if acf is None else f"{float(acf):.2f}", cycles="–" if cycles is None else cycles) + "</p>")
    else:
        lines.append("<p>" + text("No periodic intensity modulation (sample rotation) was found.") + "</p>")
    lines += _thickness(summary, text, note)
    notes = [words for words in (summary.get("notes") or []) if isinstance(words, str) and words]
    if notes:
        lines.append("<p>" + text("Notes of the model:") + "</p><ul>"
                     + "".join(f"<li>{escape(note(words))}</li>" for words in notes) + "</ul>")
    return lines


__all__ = ["FILM_NOT_DETERMINED", "REVERSAL_ROWS", "film_onset", "physical_paragraphs", "rotation_period"]
