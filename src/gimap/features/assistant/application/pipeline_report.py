"""The standard pipeline's report as Markdown: open questions first, then the numbers.

Written for people and for simple agents alike: every open question names the
option that answers it, every number comes from the tools, and every value
that is not available says why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..domain import AT_EDGE, assessment, herman_decimals, signed
from .gisaxs_report import gisaxs_batch_table, gisaxs_sections
from .pipeline_geometry import SEARCHED

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


def _fixed(value, decimals: int) -> str:
    """A number with a fixed number of decimals ("—" when there is none): one precision per table column."""
    return "—" if value is None else f"{float(value):.{decimals}f}"


PIXEL_DECIMALS = 2
"""A detector position (the beam centre) to 0.01 px, as the cuts and decisions give it."""
PEAK_DECIMALS = {"q": 4, "d_A": 3, "fwhm": 4, "snr": 1, "size_nm": 1}
"""The decimals of each column of the peak table, the same in every row (1.9000 next to 0.2236, not 1.9)."""
FLAG_WORDS = {
    "at_edge": AT_EDGE,
    "resolution_limited": "narrower than 3 bins (its width is set by the binning)",
    "overlap": "overlaps a neighbouring peak",
}
"""A flag without a caveat in the words of the peak table (report.json keeps the flags themselves)."""


def reliable_row(peak: dict) -> bool:
    """A peak row the analysis builds on: no caveat and not at the end of the data (``domain.reliable``)."""
    return not peak.get("caveat") and "at_edge" not in (peak.get("flags") or ())


def herman_texts(ring: dict) -> Optional[dict]:
    """f ± σ_f, f_iso and f − f_iso of a ring row as text, each with the decimals that show σ_f to two significant
    digits and the Unicode minus: ``{"f": "0.1392 ± 0.0027", "iso": "−0.0014", "excess": "+0.1405", "decimals": 4}``
    (``None`` without an f)."""
    if ring.get("herman") is None:
        return None
    sigma = ring.get("herman_err")
    decimals = herman_decimals(sigma)
    texts = {"f": signed(ring["herman"], decimals) + (f" ± {float(sigma):.{decimals}f}" if sigma else ""),
             "decimals": decimals}
    if ring.get("herman_isotropic") is not None:
        texts["iso"] = signed(ring["herman_isotropic"], decimals)
    if ring.get("herman_excess") is not None:
        texts["excess"] = signed(ring["herman_excess"], decimals, plus=True)
    return texts


def found_by_search(report: dict) -> Optional[dict]:
    """The decision that says the run's calibration was found by the automatic search (``decision``: its full
    path, ``why``: where it lies seen from the frame); ``None`` when the person named it or a profile was kept."""
    return next((item for item in report.get("decisions") or () if item.get("what") == SEARCHED), None)


def _geometry_lines(report: dict) -> list[str]:
    geometry = report.get("geometry") or {}
    if not geometry.get("distance_mm"):
        return ["No geometry: nothing below is in q."]
    centre = geometry.get("beam_center_px") or [None, None]
    lines = [
        f"- source: {geometry.get('source') or 'instrument profile'}",
        f"- distance {_number(geometry.get('distance_mm'))} mm, beam centre ({_fixed(centre[0], PIXEL_DECIMALS)}, "
        f"{_fixed(centre[1], PIXEL_DECIMALS)}) px, λ {_number(geometry.get('wavelength_A'))} Å, "
        f"αi {_number(geometry.get('incidence_deg'))}°",
    ]
    flat = geometry.get("flat_detector")
    if flat:  # a .poni tilt the flat geometry leaves out (Needs attention or Decisions says whether it matters)
        error = flat.get("q_error_max")
        lines.append(
            f"- detector tilt left out: {_number(flat.get('dropped_tilt_deg'), 3)}° (GIMaP's detector is flat and "
            f"perpendicular to the beam); |q| off by up to {'—' if error is None else _number(100 * error, 2)} % "
            "over the valid detector area"
        )
    searched = found_by_search(report)
    if searched:  # nobody named it: the quality below says it fits its own standard, not whose measurement it is
        lines.insert(1, f"- found by the automatic search, named by nobody: `{searched['decision']}` ({searched['why']})")
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
    decimals = PEAK_DECIMALS
    for peak in peaks:
        size = peak.get("size_nm")
        size_text = "—" if size is None else ("≥ " if peak.get("size_is_lower_bound") else "") + _fixed(size, decimals["size_nm"])
        if size is None and peak.get("size_note"):
            size_text = "n/a"
        flags = "; ".join(FLAG_WORDS.get(flag, flag) for flag in peak.get("flags") or () if flag in FLAG_WORDS)
        lines.append(
            "| " + " | ".join(_fixed(peak.get(key), decimals[key]) for key in ("q", "d_A", "fwhm", "snr")) + " | "
            f"{peak.get('orientation') or '—'} | {size_text} | {peak.get('caveat') or flags} |"
        )
    notes = [
        f"- q = {_fixed(peak.get('q'), decimals['q'])}: {peak['size_note']}" for peak in peaks if peak.get("size_note")
    ]
    return lines + ([""] + notes if notes else [])


def _ring_lines(rings: list[dict]) -> list[str]:
    if not rings:
        return ["No ring analysed."]
    lines = []
    for ring in rings:
        head = f"- q ≈ {_number(ring.get('q'))} Å⁻¹, |χ| coverage {_fixed(ring.get('coverage'), 2)}: "
        texts = herman_texts(ring)
        if texts is None:
            lines.append(head + (ring.get("reason") or ring.get("texture") or "no result"))
            lines += [f"  - {note}" for note in ring.get("notes") or () if "shadowed" in note]
            continue
        # f ± σ_f, f_iso and f − f_iso on every ring (Physics and Methods 7.7), with the decimals of σ_f.
        compare = f" (a random ring over the same |χ|: f_iso = {texts['iso']})" if "iso" in texts else ""
        excess = f", f − f_iso = {texts['excess']}" if "excess" in texts else ""
        lines.append(head + f"f = {texts['f']}{compare}{excess} — {ring.get('texture')}")
        for note in ring.get("notes") or ():
            if any(word in note for word in RING_NOTES_SHOWN):
                lines.append(f"  - {note}")
    return lines


RING_NOTES_SHOWN = ("missing wedge", "shadowed", "next to the sample horizon", "Without those bins")
"""The notes of a ring report.md lists under it (all are in report.json)."""


def _hints_title(report: dict) -> str:
    """What the series hints were taken from: the reliable peaks, of a list cut at the cap when it was."""
    search = report.get("peak_search") or {}
    if search.get("truncated"):
        return (f"Series hints (reliable peaks among the {search.get('max_peaks')} most prominent of "
                f"{search.get('candidates')} candidates only)")
    return "Series hints (reliable peaks only)"


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
        search = report.get("peak_search") or {}
        if search.get("truncated"):
            lines += [f"Only the {search.get('max_peaks')} most prominent of {search.get('candidates')} candidate peaks "
                      "were fitted (see Decisions: peak search).", ""]
        hints = report.get("series_hints") or []
        if hints:
            lines += [_hints_title(report) + ": " + "; ".join(hints), ""]
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
    """Neither procedure ran: the run ended at the frame itself (not read, not open) or at the geometry (none
    found, declined, stopped there) — also when a technique was asked for, since ``procedure`` is "giwaxs"
    until one starts. Such a report is titled "GIMaP", not "GIWAXS". A procedure that started has its
    ``technique`` decision, written before its first step."""
    if report.get("ok") or report.get("procedure") == "geometry":
        return False
    decided = [item for item in report.get("decisions") or () if item.get("what") == "technique"]
    return not decided or decided[-1].get("decision") == "none"


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


def technique_run(report: dict) -> Optional[str]:
    """``"giwaxs"`` or ``"gisaxs"``: the procedure that ran on a frame; ``None`` when none did (the frame was not
    read, there was no geometry to tell the technique from, or only the geometry was looked for)."""
    if report.get("error") or report.get("procedure") == "geometry" or _no_technique(report):
        return None
    return "gisaxs" if report.get("procedure") == "gisaxs" else "giwaxs"


def _state(report: dict) -> str:
    attention = report.get("needs_attention") or []
    if report.get("error"):
        return f"error: {report['error']}"
    if report.get("failed"):
        return f"failed: {report['failed']}"
    return "OK" if report.get("ok") and not attention else ("needs input" if attention else "failed")


def _asked(report: dict) -> str:
    return ", ".join(item["item"] for item in report.get("needs_attention") or []) or "—"


def _giwaxs_batch_table(reports: list[dict]) -> list[str]:
    lines = [
        "| frame | status | frames | geometry | peaks q (Å⁻¹) | ring f | open questions |",
        "|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        peaks = ", ".join(
            f"{peak['q']:.3f}" + ("" if reliable_row(peak) else "*") for peak in report.get("peaks") or [] if peak.get("q")
        ) or "—"
        rings = "; ".join(
            f"{ring['q']:.3g}: {ring['herman']:.2f}" for ring in report.get("rings") or [] if ring.get("herman") is not None
        ) or "—"
        lines.append(
            f"| {Path(report.get('frame') or '').name} | {_state(report)} | {_frames_text(report)} | "
            f"{(report.get('geometry') or {}).get('source') or '—'} | {peaks} | {rings} | {_asked(report)} |"
        )
    lines += ["", "Peaks marked * have a caveat (spike, halo, failed fit or weak) or lie at the end of the data: see that "
              "frame's report.md.", ""]
    cut = [report for report in reports if (report.get("peak_search") or {}).get("truncated")]
    if cut:
        lines += [
            "Peak lists cut at the cap (the most prominent candidates fitted; their series hints use these only): "
            + ", ".join(f"{Path(report.get('frame') or '').name} ({report['peak_search'].get('max_peaks')} of "
                        f"{report['peak_search'].get('candidates')})" for report in cut)
            + ". `--max-peaks N` fits more.", "",
        ]
    shared = common_peaks(reports)
    if shared:
        listed = ", ".join(f"{peak['q']:.4g}" + (f" ({peak['caveat'].split(':')[0]})" if peak.get("caveat") else "") for peak in shared)
        lines += [
            f"Lines at the same q (±{SAME_Q:.1%}) in every frame: {listed}. A line every sample shares is "
            "either a common phase or instrumental (a window, the substrate, hot pixels): compare with the "
            "calibration image or an empty substrate before interpreting it.",
            "",
        ]
    return lines


def _unanalysed_table(reports: list[dict]) -> list[str]:
    """Frames no procedure ran on: why (the status) and what is asked."""
    lines = ["| frame | status | geometry | open questions |", "|---|---|---|---|"]
    lines += [
        f"| {Path(report.get('frame') or '').name} | {_state(report)} | "
        f"{(report.get('geometry') or {}).get('source') or '—'} | {_asked(report)} |" for report in reports
    ]
    return lines + [""]


def _searched_notes(reports: list[dict]) -> list[str]:
    """One line per calibration the automatic search found and a frame of the batch used (information only)."""
    found: dict[str, list[dict]] = {}
    for report in reports:
        searched = found_by_search(report)
        if searched:
            found.setdefault(searched["decision"], []).append(searched)
    return [line for path, uses in found.items() for line in (
        f"Calibration found by the automatic search, named by nobody: `{path}` ({uses[0]['why']}); the geometry of "
        f"{len(uses)} frame(s). `--calibration PATH` gives the one of the measurement instead.", "",
    )]


def batch_markdown(reports: list[dict]) -> str:
    """The frames of a batch in tables by the procedure that ran, the lines the GIWAXS frames share and the
    questions still open. Titled by what ran: GIWAXS, GISAXS, or GIMaP when both did or neither."""
    groups: dict[Optional[str], list[dict]] = {"giwaxs": [], "gisaxs": [], None: []}
    for report in reports:
        groups[technique_run(report)].append(report)
    ran = [key for key in ("giwaxs", "gisaxs") if groups[key]]
    mixed = len(ran) == 2
    if len(ran) == 1:  # one procedure: the frames it did not run on are rows of its table
        groups = {**groups, ran[0]: list(reports), None: []}
    heading = ran[0].upper() if len(ran) == 1 else "GIMaP"
    lines = [f"# {heading} — batch", "", f"{len(reports)} frame(s); each has its own report.md.", ""]
    if groups["giwaxs"]:
        lines += (["## GIWAXS", ""] if mixed else []) + _giwaxs_batch_table(groups["giwaxs"])
    if groups["gisaxs"]:
        lines += (["## GISAXS", ""] if mixed else []) + gisaxs_batch_table(groups["gisaxs"], _state) + [""]
    if groups[None]:
        lines += (["## 未分析 / Not analysed", ""] if mixed else []) + _unanalysed_table(groups[None])
    lines += _searched_notes(reports)
    questions: dict[tuple, dict] = {}  # one per item and procedure: αi is asked in the words of each
    for report in reports:
        for item in report.get("needs_attention") or []:
            questions.setdefault((item["item"], technique_run(report)), item)
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

    Only reliable peaks (``reliable_row``) count as present.  Two peaks are the same line when they are closer than
    half the wider one's FWHM (at least ``tolerance`` in relative q); one that moved further than
    ``tolerance`` is "shifted".  A weak (tentative) peak on the other side makes a line "grew" or
    "faded" rather than "appeared" or "disappeared"; a line gone next to a new one (within
    ``NEARBY_Q``) becomes one "moved?" row.  ``start`` / ``end``: "yes", "weak" or "—".
    """
    def pick(report: dict, weak: bool) -> list[dict]:
        return [
            peak for peak in report.get("peaks") or [] if peak.get("q")
            and (str(peak.get("caveat") or "").startswith("weak") if weak else reliable_row(peak))
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
        {"q": peak["q"], "label": f"{peak['q']:.3g}", "reliable": reliable_row(peak)}
        for peak in report.get("peaks") or () if peak.get("q")
    ]


def ring_overlays(report: dict) -> list[dict]:
    """The analysed rings of a report, to draw on the q map (``preview_png(rings=…)``)."""
    return [
        {"q": ring["q"], "shadowed": ring.get("shadowed") or (), "label": f"{ring['q']:.3g}"}
        for ring in report.get("rings") or () if ring.get("q")
    ]


__all__ = [
    "FLAG_WORDS", "OPTION_FLAGS", "PEAK_DECIMALS", "SAME_Q", "batch_markdown", "calibration_quality", "common_peaks",
    "compact_report", "found_by_search", "herman_texts", "peak_markers", "pipeline_markdown", "reliable_row", "ring_overlays",
    "series_changes", "series_markdown", "technique_run",
]
