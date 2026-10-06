"""The standard pipeline's report as Markdown: open questions first, then the numbers.

Written for people and for simple agents alike: every open question names the
option that answers it, every number comes from the tools, and every value
that is not available says why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..domain import assessment
from .gisaxs_report import gisaxs_batch_table, gisaxs_sections

OPTION_FLAGS = {
    "calibration": "--calibration PATH",
    "standard": "--standard agbh|lab6|ceo2|lab6_ceo2",
    "energy_kev": "--energy-kev E",
    "incidence_deg": "--incidence-deg A",
    "pixel_size_um": "--pixel-size-um P",
    "technique": "--technique giwaxs|gisaxs",
}
"""The command-line flag that answers an open question (``needs_attention[].option``; MCP: the argument of that name)."""


def compact_report(report: dict) -> dict:
    """The report without the full tables, the calibration record and the fitted arrays (kept in report.json)."""
    compact = {key: value for key, value in report.items() if key not in ("tables", "calibration")}
    fit = (compact.get("gisaxs") or {}).get("fit")
    if fit:
        compact["gisaxs"] = {**compact["gisaxs"], "fit": {
            key: value for key, value in fit.items() if key not in ("data", "best_curve", "curves", "native")}}
    if compact.get("peaks"):
        compact["peaks"] = [
            {**peak, "fit": {key: value for key, value in peak["fit"].items() if key not in ("x", "y", "sigma")}}
            if peak.get("fit") else peak for peak in compact["peaks"]
        ]
    return compact


def calibration_quality(result: Optional[dict]) -> Optional[dict]:
    """How good the calibration a run used is: its assessment, the standard and the line check."""
    if not result:
        return None
    if result.get("from_file"):
        return {"assessment": "a saved calibration file (not re-checked against a standard image)", "warnings": []}
    check = result.get("line_check") or {}
    return {
        "assessment": assessment(result),
        "standard": result.get("standard"),
        "lines_checked": check.get("lines_checked"),
        "mean_line_q_error_percent": None if check.get("mean_relative") is None else round(100 * check["mean_relative"], 3),
        "matched_rings": result.get("matched_rings"),
        "warnings": list(result.get("warnings") or ()),
    }


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _geometry_lines(report: dict) -> list[str]:
    geometry = report.get("geometry") or {}
    if not geometry.get("distance_mm"):
        return ["No geometry: nothing below is in q."]
    centre = geometry.get("beam_center_px") or [None, None]
    lines = [
        f"- source: {geometry.get('source') or 'instrument profile'}",
        f"- distance {_number(geometry.get('distance_mm'))} mm, beam centre ({_number(centre[0])}, "
        f"{_number(centre[1])}) px, λ {_number(geometry.get('wavelength_A'))} Å, "
        f"αi {_number(geometry.get('incidence_deg'))}°",
    ]
    quality = report.get("calibration_quality")
    if quality:
        lines.append(f"- quality: {quality.get('assessment')}")
        confirmed = str(quality.get("assessment", "")).startswith("good") and quality.get("lines_checked")
        for warning in quality.get("warnings") or ():
            # The fit engine's warnings predate the line check; when the lines confirm the q scale, say so.
            lines.append(f"- note: {warning} (the line check above confirms the q scale)" if confirmed else f"- warning: {warning}")
    return lines


def _peak_table(peaks: list[dict]) -> list[str]:
    if not peaks:
        return ["No peak found."]
    lines = [
        "| q (Å⁻¹) | d (Å) | FWHM (Å⁻¹) | SNR | in-/out-of-plane | size (nm) | caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for peak in peaks:
        size = peak.get("size_nm")
        size_text = "—" if size is None else ("≥ " if peak.get("size_is_lower_bound") else "") + _number(size, 3)
        if size is None and peak.get("size_note"):
            size_text = "n/a"
        lines.append(
            f"| {_number(peak.get('q'))} | {_number(peak.get('d_A'))} | {_number(peak.get('fwhm'), 3)} | "
            f"{_number(peak.get('snr'), 3)} | {peak.get('orientation') or '—'} | {size_text} | "
            f"{peak.get('caveat') or ', '.join(peak.get('flags') or ()) or ''} |"
        )
    notes = [
        f"- q = {_number(peak.get('q'))}: {peak['size_note']}" for peak in peaks if peak.get("size_note")
    ]
    return lines + ([""] + notes if notes else [])


def _ring_lines(rings: list[dict]) -> list[str]:
    if not rings:
        return ["No ring analysed."]
    lines = []
    for ring in rings:
        head = f"- q ≈ {_number(ring.get('q'))} Å⁻¹, |χ| coverage {_number(ring.get('coverage'), 2)}: "
        if ring.get("herman") is None:
            lines.append(head + (ring.get("reason") or ring.get("texture") or "no result"))
            lines += [f"  - {note}" for note in ring.get("notes") or () if "shadowed" in note]
            continue
        reference = ring.get("herman_isotropic")
        compare = f" (a random ring over the same |χ|: {_number(reference, 2)})" if reference is not None and abs(reference) > 0.005 else ""
        lines.append(head + f"f = {_number(ring.get('herman'), 3)}{compare} — {ring.get('texture')}")
        for note in ring.get("notes") or ():
            if "missing wedge" in note or "shadowed" in note:
                lines.append(f"  - {note}")
    return lines


def pipeline_markdown(report: dict, *, title: Optional[str] = None) -> str:
    frame = report.get("frame") or ""
    name = title or Path(frame).name or "frame"
    attention = report.get("needs_attention") or []
    state = "OK" if report.get("ok") and not attention else ("NEEDS INPUT" if attention else "FAILED")
    if report.get("failed"):  # ended early (the frame changed, the file could not be read): its point says why
        state = "FAILED"
    gisaxs = report.get("procedure") == "gisaxs"
    undecided = _no_technique(report)
    heading = "GISAXS" if gisaxs else ("GIMaP" if undecided else "GIWAXS")
    lines = [f"# {heading} — {name}", "", f"Status: **{state}**  ", f"File: `{frame}`", ""]
    if attention:
        lines += ["## 需要处理 / Needs attention (answer, then run again)", ""]
        for number, item in enumerate(attention, 1):
            option = OPTION_FLAGS.get(item.get("option") or "", "")
            fix = f" → `{option}`" if option else ""
            lines.append(f"{number}. **{item['item']}** — {item['why']}{fix}  ")
            lines.append(f"   {item.get('hint', '')}")
        lines.append("")
    frames = report.get("frames") or {}
    lines += ["## 几何 / Geometry", "", *_geometry_lines(report), ""]
    if frames.get("total") and frames["total"] > 1:
        lines += [
            "## 帧 / Frames", "",
            f"- series of {frames['total']} frames; analysed from frame {frames.get('first')}, "
            f"{frames.get('summed')} summed", "",
        ]
    if report.get("ok") and gisaxs:
        lines += gisaxs_sections(report)
    elif report.get("ok") and report.get("procedure") != "geometry":
        lines += ["## 峰 / Peaks (radial I(q))", "", *_peak_table(report.get("peaks") or []), ""]
        hints = report.get("series_hints") or []
        if hints:
            lines += ["Series hints (reliable peaks only): " + "; ".join(hints), ""]
        lines += ["## 取向 / Ring orientation", "", *_ring_lines(report.get("rings") or []), ""]
    lines += ["## 决策 / Decisions", ""]
    lines += [f"- **{item['what']}**: {item['decision']} — {item['why']}" for item in report.get("decisions") or []]
    lines += ["", "## 步骤 / Steps", ""]
    lines += [
        f"{number}. `{step['tool']}` {'ERROR ' if step.get('error') else ''}{step.get('summary', '')}"
        for number, step in enumerate(report.get("steps") or [], 1)
    ]
    if not undecided:
        lines += ["", (
            "q in Å⁻¹, sizes and distances in nm. The fit compares particle families; the model is a choice, "
            "not a measurement." if gisaxs else
            "Sizes are Scherrer lower bounds (no instrumental width subtracted). Phases are not assigned: "
            "name the material to compare q with known lines."
        )]
    return "\n".join(lines) + "\n"


def _no_technique(report: dict) -> bool:
    """Neither procedure ran and none was chosen: Auto had no geometry to tell the technique from, or the run
    ended at the frame itself (not read, not open). Such a report is titled "GIMaP", not "GIWAXS"."""
    if report.get("ok") or report.get("procedure") == "geometry":
        return False
    decided = [item for item in report.get("decisions") or () if item.get("what") == "technique"]
    if decided:
        return decided[-1].get("decision") == "none"
    return any(item.get("item") == "frame" for item in report.get("needs_attention") or ())


SAME_Q = 0.003
"""Relative q difference within which peaks of different frames count as the same line."""


def _frames_text(report: dict) -> str:
    frames = report.get("frames") or {}
    total, first, summed = frames.get("total"), frames.get("first"), frames.get("summed")
    if not total or total <= 1 or first is None:
        return "1"
    summed = summed or 1
    return f"{first}–{first + summed - 1} of {total}" if summed > 1 else f"{first} of {total}"


def common_peaks(reports: list[dict]) -> list[dict]:
    """Peaks of the first frame found at the same q (±0.3 %) in every other frame."""
    analysed = [report for report in reports if report.get("ok")]
    if len(analysed) < 2:
        return []
    shared = []
    for peak in analysed[0].get("peaks") or []:
        q = peak.get("q")
        if q and all(any(abs(other["q"] - q) <= SAME_Q * q for other in report.get("peaks") or []) for report in analysed[1:]):
            shared.append(peak)
    return shared


def batch_markdown(reports: list[dict]) -> str:
    """One table for many frames, the lines they share and the questions still open."""
    def state_of(report: dict) -> str:
        attention = report.get("needs_attention") or []
        if report.get("error"):
            return f"error: {report['error']}"
        if report.get("failed"):
            return f"failed: {report['failed']}"
        return "OK" if report.get("ok") and not attention else ("needs input" if attention else "failed")

    gisaxs = any(not report.get("error") for report in reports) and all(
        report.get("procedure") == "gisaxs" or report.get("error") for report in reports
    )
    lines = [f"# {'GISAXS' if gisaxs else 'GIWAXS'} — batch", "", f"{len(reports)} frame(s); each has its own report.md.", ""]
    if gisaxs:
        questions = {item["item"]: item for report in reports for item in report.get("needs_attention") or []}
        lines += gisaxs_batch_table(reports, state_of) + [""]
        if questions:
            lines += ["## 需要处理 / Open questions", ""]
            lines += [f"- **{item['item']}** — {item['why']}" for item in questions.values()]
            lines.append("")
        return "\n".join(lines) + "\n"
    lines += [
        "| frame | status | frames | geometry | peaks q (Å⁻¹) | ring f | open questions |",
        "|---|---|---|---|---|---|---|",
    ]
    questions: dict[str, dict] = {}
    for report in reports:
        attention = report.get("needs_attention") or []
        for item in attention:
            questions.setdefault(item["item"], item)
        state = state_of(report)
        peaks = ", ".join(
            f"{peak['q']:.3f}" + ("*" if peak.get("caveat") else "") for peak in report.get("peaks") or [] if peak.get("q")
        ) or "—"
        rings = "; ".join(
            f"{ring['q']:.3g}: {ring['herman']:.2f}" for ring in report.get("rings") or [] if ring.get("herman") is not None
        ) or "—"
        lines.append(
            f"| {Path(report.get('frame') or '').name} | {state} | {_frames_text(report)} | "
            f"{(report.get('geometry') or {}).get('source') or '—'} | {peaks} | {rings} | "
            f"{', '.join(item['item'] for item in attention) or '—'} |"
        )
    lines += ["", "Peaks marked * have a caveat (spike, halo, failed fit or weak): see that frame's report.md.", ""]
    shared = common_peaks(reports)
    if shared:
        listed = ", ".join(f"{peak['q']:.4g}" + (f" ({peak['caveat'].split(':')[0]})" if peak.get("caveat") else "") for peak in shared)
        lines += [
            f"Lines at the same q (±{SAME_Q:.1%}) in every frame: {listed}. A line every sample shares is "
            "either a common phase or instrumental (a window, the substrate, hot pixels): compare with the "
            "calibration image or an empty substrate before interpreting it.",
            "",
        ]
    if questions:
        lines += ["## 需要处理 / Open questions", ""]
        for item in questions.values():
            option = OPTION_FLAGS.get(item.get("option") or "", "")
            lines.append(f"- **{item['item']}** — {item['why']}" + (f" → `{option}`" if option else ""))
        lines.append("")
    return "\n".join(lines) + "\n"


SAME_LINE_FWHM = 0.5
"""Two peaks closer than this fraction of the wider one's FWHM are the same line (it may have moved)."""
NEARBY_Q = 0.01
"""A line gone and one new within this relative q are shown together: perhaps one line that moved."""


def series_changes(start: dict, end: dict, tolerance: float = SAME_Q) -> list[dict]:
    """How the lines change from the start to the end of a series, sorted by q.

    Only reliable peaks count as present.  Two peaks are the same line when they are closer than
    half the wider one's FWHM (at least ``tolerance`` in relative q); one that moved further than
    ``tolerance`` is "shifted".  A weak (tentative) peak on the other side makes a line "grew" or
    "faded" rather than "appeared" or "disappeared"; a line gone next to a new one (within
    ``NEARBY_Q``) becomes one "moved?" row.  ``start`` / ``end``: "yes", "weak" or "—".
    """
    def pick(report: dict, weak: bool) -> list[dict]:
        return [
            peak for peak in report.get("peaks") or [] if peak.get("q")
            and (str(peak.get("caveat") or "").startswith("weak") if weak else not peak.get("caveat"))
        ]

    def nearest(peak: dict, others: list[dict]) -> Optional[dict]:
        def same(other: dict) -> bool:
            width = max(float(peak.get("fwhm") or 0.0), float(other.get("fwhm") or 0.0))
            return abs(other["q"] - peak["q"]) <= max(tolerance * peak["q"], SAME_LINE_FWHM * width)

        return min((other for other in others if same(other)), key=lambda other: abs(other["q"] - peak["q"]), default=None)

    first, last = pick(start, False), pick(end, False)
    rows, matched = [], []
    for peak in last:
        other = nearest(peak, [item for item in first if all(item is not used for used in matched)])
        if other is not None:
            matched.append(other)
            shift = peak["q"] - other["q"]
            change = "present at both" if abs(shift) <= tolerance * peak["q"] else (
                f"shifted {shift:+.3f} Å⁻¹ ({shift / other['q']:+.1%}) from {other['q']:.4g}"
            )
            rows.append({"q": peak["q"], "start": "yes", "end": "yes", "change": change})
        elif nearest(peak, pick(start, True)) is not None:
            rows.append({"q": peak["q"], "start": "weak", "end": "yes", "change": "grew (weak at the start)"})
        else:
            rows.append({"q": peak["q"], "start": "—", "end": "yes", "change": "appeared"})
    for peak in first:
        if any(peak is used for used in matched):
            continue
        if nearest(peak, pick(end, True)) is not None:
            rows.append({"q": peak["q"], "start": "yes", "end": "weak", "change": "faded (weak at the end)"})
        else:
            rows.append({"q": peak["q"], "start": "yes", "end": "—", "change": "disappeared"})
    # A line that disappeared next to one that appeared: possibly one line that moved further than its width.
    gone = [row for row in rows if row["change"] == "disappeared"]
    for row in [row for row in rows if row["change"] == "appeared"]:
        near = min(gone, key=lambda other: abs(other["q"] - row["q"]), default=None)
        if near is not None and abs(near["q"] - row["q"]) <= NEARBY_Q * row["q"]:
            gone.remove(near)
            rows.remove(near)
            shift = (row["q"] - near["q"]) / near["q"]
            row.update(start="yes", change=(
                f"moved? {near['q']:.4g} → {row['q']:.4g} ({shift:+.1%}): one line shifting further than its "
                "width, or one line replacing another"
            ))
    return sorted(rows, key=lambda row: row["q"])


def series_markdown(start: dict, end: dict) -> str:
    """The start-versus-end comparison of a series as a Markdown section (empty without both reports)."""
    if not start.get("ok") or not end.get("ok"):
        return ""
    def frames(report: dict) -> str:
        info = report.get("frames") or {}
        first, summed = int(info.get("first") or 1), int(info.get("summed") or 1)
        return f"{first}–{first + summed - 1}"

    lines = [
        "## 序列 / Start versus end of the series", "",  # bilingual like the report's other sections
        f"Frames {frames(start)} compared with frames {frames(end)}. Reliable peaks only; peaks closer than "
        "half their width are the same line.", "",
        "| q (Å⁻¹) | at the start | at the end | change |", "|---|---|---|---|",
    ]
    for row in series_changes(start, end):
        lines.append(f"| {row['q']:.4g} | {row['start']} | {row['end']} | {row['change']} |")
    return "\n".join(lines) + "\n"


def peak_markers(report: dict) -> list[dict]:
    """The report's peaks as I(q) markers (``curve_png(markers=…)``): reliable ones and flagged ones."""
    return [
        {"q": peak["q"], "label": f"{peak['q']:.3g}", "reliable": not peak.get("caveat")}
        for peak in report.get("peaks") or () if peak.get("q")
    ]


def ring_overlays(report: dict) -> list[dict]:
    """The analysed rings of a report, to draw on the q map (``preview_png(rings=…)``)."""
    return [
        {"q": ring["q"], "shadowed": ring.get("shadowed") or (), "label": f"{ring['q']:.3g}"}
        for ring in report.get("rings") or () if ring.get("q")
    ]


__all__ = [
    "OPTION_FLAGS", "SAME_Q", "batch_markdown", "calibration_quality", "common_peaks", "compact_report", "peak_markers",
    "pipeline_markdown", "ring_overlays", "series_changes", "series_markdown",
]
