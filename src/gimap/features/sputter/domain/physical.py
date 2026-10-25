"""The physical path of a sputter series: what it assumes, what it changed and what the thickness values mean.

``structure_physical.csv`` and ``structure_physical_summary.json`` (written by the toolkit's ``analyze``,
``table`` and ``physical`` commands) hold one stage path through time (pre-nucleation → islands → coalescence
→ film; a film does not revert to islands), the network estimate of R and D in every island window (grade P;
no switching between refined and network values from window to window), smoothed trend lines, the
island-volume equivalent thickness t_eq, the nominal thickness rate·(t − t0) when a deposition rate is given
(constant after the end of the deposition t_end when it is given: the shutter is closed by hand),
the film onset with its interval and the period of a periodic intensity modulation (probably the back-and-forth
translation of the sample stage during the deposition, which keeps the beam from damaging one spot: near the ends of
its travel the beam can reach the sample edge or leave the sample). Windows whose intensity dips far below that of
their neighbours are flagged ``beam_off_sample`` (the beam is probably partly or fully off the sample), a whole dip
at a time: its deepest windows and the windows next to them that are still well below the level. They keep their
values, but cast no stage vote and are left out of the trend lines, the back-and-forth counts, the mass check at the
film onset and the literature method (its trend of D and its checks of the network). The summary's
``modulation_period_s`` is the neutral name of ``rotation_period_s`` (results of older toolkits have only the
latter).
Written from English templates (``text``) like the rest of the reliability report; the toolkit's own notes go
through ``note`` (the interface gives the sentences it knows in its language, the others as they are).
"""

from __future__ import annotations

from html import escape
from typing import Optional

REVERSAL_ROWS = (("stage", "Stage steps back"), ("module_revert", "Film → islands"),
                 ("R_zigzag", "R zig-zags"), ("D_zigzag", "D zig-zags"))
FILM_NOT_DETERMINED = "film: thickness not determined"
NOMINAL = ("Nominal thickness = rate × (t − t0) with rate {rate} nm/s and t0 = {t0} s: what was deposited, not what "
           "is measured.")
NOMINAL_ENDED = ("Nominal thickness = rate × (t − t0) with rate {rate} nm/s and t0 = {t0} s until the deposition ended "
                 "at t_end = {t_end} s, then constant at {thickness} nm: what was deposited, not what is measured.")
MODULATION = ("The intensity is modulated with a period of about {period} s (autocorrelation peak {acf}, {cycles} cycles), "
              "probably by the back-and-forth translation of the sample stage during the deposition (it keeps the beam "
              "from damaging one spot; near the ends of its travel the beam can reach the sample edge or leave the "
              "sample). A single window shorter than this period sees part of one modulation cycle; the physical path "
              "and the trend lines average over it.")
BEAM_OFF = ("<b>Beam off the sample.</b> In {count} of {windows} windows the beam has probably moved partly or fully "
            "off the sample while the stage was translated: {core} of them have an intensity below {limit}% of its "
            "level within one modulation period before and after, the others lie next to them in the same dip, still "
            "below {release}%. These windows keep their values in the Windows table (column Beam may be off the "
            "sample, with the reason) and are drawn as hollow grey points in Size Trend, but they cast no stage vote "
            "(the path carries the stage of their neighbours) and are left out of the trend lines, the back-and-forth "
            "counts, the mass check at the film onset and the literature method (its trend of D and its checks of "
            "the network).")
BEAM_OFF_EXTREME = ("<b>Beam off the sample.</b> No periodic intensity modulation was found, so only extreme dips are "
                    "flagged: in {count} of {windows} windows the beam was probably off the sample or interrupted. "
                    "{core} of them have an intensity below {limit}% of its level within {span} s before and after, the "
                    "others lie next to them in the same dip, still below {release}%. These windows keep their values "
                    "in the Windows table and are drawn as hollow grey points in Size Trend, but they cast no stage vote "
                    "(the path carries the stage of their neighbours) and are left out of the trend lines, the "
                    "back-and-forth counts, the mass check at the film onset and the literature method (its trend of "
                    "D and its checks of the network).")
BEAM_ON = ("No window has an intensity below {limit}% of its level within one modulation period before and after: "
           "nothing suggests that the beam left the sample.")
BEAM_ON_EXTREME = ("No window has an intensity below {limit}% of its level within {span} s before and after; without a "
                   "periodic modulation smaller dips are not flagged.")


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


def film_onset_established(summary: dict) -> bool:
    """The physical path's film onset is established: a path without any film window fits clearly worse (the
    toolkit's ``film_onset_no_TS_within_slack`` is not true). Without an onset there is nothing established."""
    onset, _interval = film_onset(summary)
    return onset is not None and summary.get("film_onset_no_TS_within_slack") is not True


def deposition_end(summary: dict) -> Optional[float]:
    """The end of the deposition t_end (s) the physical files were drawn with (None: not given, the deposition
    lasts to the end of the scan)."""
    try:
        value = (summary.get("thickness") or {}).get("t_end_s")
        return None if value is None else float(value)
    except (AttributeError, TypeError, ValueError):
        return None


def modulation_period(summary: dict) -> Optional[float]:
    """The period (s) of the intensity modulation: ``modulation_period_s``, or ``rotation_period_s`` of a result of an
    older toolkit (the same value under the name it had)."""
    try:
        value = summary.get("modulation_period_s", summary.get("rotation_period_s"))
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def beam_off_record(summary: dict) -> dict:
    """The toolkit's record of the windows where the beam may be off the sample ({} for an older result)."""
    record = summary.get("beam_off_sample")
    return record if isinstance(record, dict) else {}


def beam_off_count(result) -> int:
    """How many windows of a result may have the beam off the sample (``beam_off_sample`` = true)."""
    return sum(1 for r in result.rows if r.get("beam_off_sample") is True)


def _beam_off(result, text) -> list:
    """What the toolkit flagged as beam off the sample (nothing for a result of an older toolkit)."""
    record = beam_off_record(result.summary)
    if not record:
        return []
    count, modulated = beam_off_count(result), record.get("rule") == "modulation"
    try:
        limit = f"{100 * float(record.get('limit_fraction')):g}"
        release = f"{100 * float(record.get('release_fraction', record.get('limit_fraction'))):g}"
        span = f"{float(record.get('span_s')):.3g}"
        core = int(record.get("core_windows", count))
    except (TypeError, ValueError):
        return []
    if count:
        return ["<p>" + text(BEAM_OFF if modulated else BEAM_OFF_EXTREME, count=count, windows=len(result.rows),
                             core=core, limit=limit, release=release, span=span) + "</p>"]
    return ["<p>" + text(BEAM_ON if modulated else BEAM_ON_EXTREME, limit=limit, span=span) + "</p>"]


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
        t0 = None if thickness.get("t0_s") is None else float(thickness["t0_s"])
        t_end = deposition_end(summary)
        if t_end is None:
            lines.append("<p>" + text(NOMINAL, rate=f"{float(rate):.4g}", t0=_seconds(t0)) + "</p>")
        else:
            lines.append("<p>" + text(NOMINAL_ENDED, rate=f"{float(rate):.4g}", t0=_seconds(t0), t_end=_seconds(t_end),
                                      thickness=f"{float(rate) * max(0.0, t_end - (t0 or 0.0)):.3g}") + "</p>")
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
    notes = [thickness.get(key) for key in ("t_eq_note", "nominal_note", "t_end_note", "film_thickness_note")]
    notes = [words for words in notes if isinstance(words, str) and words]
    if notes:
        lines.append("<ul>" + "".join(f"<li>{escape(note(words))}</li>" for words in notes) + "</ul>")  # the model's
    return lines


def physical_paragraphs(result, text, note=lambda sentence: sentence) -> list:
    summary = result.summary
    changed = sum(1 for r in result.rows if r.get("path_changed"))
    lines = ["<p>" + text(
        "<b>Physical path of the network.</b> One path of stages through time — pre-nucleation, islands, coalescence, "
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
    period = modulation_period(summary)
    detection = summary.get("modulation_detection") or summary.get("rotation_detection") or {}
    if period is not None:
        acf, cycles = detection.get("acf_peak"), detection.get("cycles")
        lines.append("<p>" + text(MODULATION, period=f"{period:.3g}", acf="–" if acf is None else f"{float(acf):.2f}",
                                  cycles="–" if cycles is None else cycles) + "</p>")
    else:
        lines.append("<p>" + text("No periodic intensity modulation was found.") + "</p>")
    lines += _beam_off(result, text)
    lines += _thickness(summary, text, note)
    notes = [words for words in (summary.get("notes") or []) if isinstance(words, str) and words]
    if notes:
        lines.append("<p>" + text("Notes of the model:") + "</p><ul>"
                     + "".join(f"<li>{escape(note(words))}</li>" for words in notes) + "</ul>")
    return lines


TEMPLATES = (NOMINAL, NOMINAL_ENDED, MODULATION, BEAM_OFF, BEAM_OFF_EXTREME, BEAM_ON, BEAM_ON_EXTREME)
"""Sentences the report says from a name (the interface translates them)."""

__all__ = ["BEAM_OFF", "BEAM_OFF_EXTREME", "BEAM_ON", "BEAM_ON_EXTREME", "FILM_NOT_DETERMINED", "MODULATION",
           "NOMINAL", "NOMINAL_ENDED", "REVERSAL_ROWS", "TEMPLATES", "beam_off_count", "beam_off_record",
           "deposition_end", "film_onset", "film_onset_established", "modulation_period", "physical_paragraphs"]
