"""Short report from the recorded peak assignments and ring measurements."""

from math import isclose

from src.gimap.shared.orientation_notes import coverage_warning


WARNING_TOPICS = {
    "orientation factor": "sensitivity of the orientation factor to analysis settings",
    "pole figure": "undetermined pole-figure fits",
}


def result_lines(report, word=str):
    peaks = report.get("peaks") or []
    matched = [p for p in peaks if p.get("phase")]
    ambiguous = sum(bool(p.get("ambiguous") or p.get("alternatives")) for p in matched)
    lines = [word("{matched} of {total} peaks indexed to candidate phases ({ambiguous} ambiguous).").format(
        matched=len(matched), total=len(peaks), ambiguous=ambiguous)]
    if not matched:
        lines.append(word("No phase matched the measured peaks.") if report.get("phase_assignment") else
                     word("No phase assignments recorded."))
    rings = report.get("rings") or []
    for name in dict.fromkeys(p["phase"] for p in matched):
        results = []
        phase_peaks = [p for p in matched if p["phase"] == name]
        for peak in phase_peaks:
            ring = next((r for r in rings if r.get("q") is not None and peak.get("q") is not None
                         and isclose(r["q"], peak["q"], rel_tol=1e-6, abs_tol=1e-9)), None)
            if ring is None:
                continue
            pole = ring.get("pole_figure") or {}
            if peak.get("ambiguous") or peak.get("alternatives") or peak.get("tentative"):
                orientation = word("phase assignment uncertain")
            elif ring.get("herman") is None or pole.get("quality_reasons") or coverage_warning(pole.get("systematics") or {}):
                orientation = word("orientation undetermined")
            else:
                orientation = word(ring.get("texture") or "measured ring") + f" (f = {ring['herman']:.3g})"
            results.append(f"{peak.get('hkl') or ''} · {ring['q']:.4g} Å⁻¹ · {orientation}")
        if not results:
            results.append(word("no ring orientation measured"))
            for peak in phase_peaks:
                if (peak.get("orientation") and not any(peak.get(k) for k in ("ambiguous", "alternatives", "tentative", "caveat"))
                        and "at_edge" not in (peak.get("flags") or ())):
                    results.append(word("sector comparison at {hkl}: {preference}").format(
                        hkl=peak.get("hkl") or "", preference=word(peak["orientation"])))
                    break
        lines.append(f"{name} — " + " / ".join(results))
    topics = dict.fromkeys(p["item"] for p in report.get("needs_attention") or () if p.get("item"))
    warnings = [word(WARNING_TOPICS.get(topic, topic)) for topic in topics]
    tentative = sum(bool(p.get("tentative")) for p in matched)
    if tentative:
        warnings.append(word("{count} matches based on peak positions alone").format(count=tentative))
    flagged = sum(bool(p.get("flags") or p.get("caveat")) for p in peaks)
    if flagged:
        warnings.append(word("{count} flagged peak(s)").format(count=flagged))
    if warnings:
        lines.append(word("Check {warnings}.").format(warnings=", ".join(warnings)))
    return lines
