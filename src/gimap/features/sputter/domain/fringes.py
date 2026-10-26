"""Island height H from the moving q_z fringes (``utils/sputter_fringes``): its outputs as the page uses them.

The analysis (a separate tool, run in the background) writes ``fringe_heights.csv`` (one row per time group: H and its
68% interval, or a status saying why there is none) and ``fringe_summary.json`` (alpha_c used and where it comes from,
the clean q_z range, the growth fit, notes, the SHA-256 of its inputs). H is the vertical size of the islands or of
the layer: the height of a uniform layer whose order-n minimum lies where the moving minimum is, refraction-corrected
inside the metal. It equals the film thickness only once a continuous film has formed, and it can be measured only
while a minimum is inside the detector's q_z range.

Pure Python: parsing, the group nearest to a time, and the Reliability paragraph from English templates (``text``
fills them in the interface language, as in ``structure.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .structure import english, number

FRINGE_FOLDER = "fringes_v1"
HEIGHTS_FILE = "fringe_heights.csv"
SUMMARY_FILE = "fringe_summary.json"
STATUS_NAMES = {
    "measured": "H measured",
    "one_side_only": "seen on one side of the beam only",
    "sides_disagree": "the two sides of the beam disagree",
    "ambiguous_order": "moving minima of unknown order",
    "below_range": "below the range: the first minimum is above the top of the detector's q_z range",
    "not_detected": "no moving minimum",
}
"""What a group's status means (the page translates them)."""
MEASURED = ("measured", "one_side_only")

REPORT_TITLE = "<b>Island height H from the q_z fringes.</b>"
REPORT_WHAT = ("H is the vertical size of the islands or of the layer, from the positions of the minima of the "
               "vertical cuts at low q_y that move to lower q_z while the metal grows (H = 2π n / Q, Q = kz(αi) + kz(αf) inside "
               "the metal). It equals the film thickness only after a continuous film has formed. It can be measured "
               "only while a minimum lies inside the detector's q_z range ({low}–{high} nm⁻¹ inside the metal); before "
               "the first minimum enters it, H is below {bound} nm.")
REPORT_MEASURED = ("H is measured in {measured} of {groups} time groups ({first}–{last} s; {one_side} of them on one "
                   "side only).")
REPORT_NONE = "No minimum moves through the range like the fringe of a growing layer: H is not measured."
REPORT_AMBIGUOUS = "{count} time groups show moving minima whose order cannot be told: H is not given there."
REPORT_ALPHA_C = "Refraction-corrected with αc = {alpha_c}° ± {error}° ({source})."
REPORT_GROWTH = ("A straight line through the measured groups: dH/dt = {v} ± {error} nm/s; it reaches H = 0 at "
                 "t = {t0} s (the deposition start the fringes imply, for information).")
REPORT_CAVEAT = ("The intervals hold the minima's position errors, the error of αc and a systematic error that grows "
                 "when less than about one fringe period is in the range; the slab phase assumed (minima at "
                 "H Q = 2π n) may leave a constant offset of about 1 nm.")
ALPHA_C_SOURCES = {"phase-locked fits": "fitted to the fringes of the accepted time groups",
                   "fit of coexisting orders": "fitted so that coexisting orders give the same H",
                   "given": "given by the user"}
BULK_SOURCE = "bulk value of {metal}"
TEMPLATES = (REPORT_TITLE, REPORT_WHAT, REPORT_MEASURED, REPORT_NONE, REPORT_AMBIGUOUS, REPORT_ALPHA_C, REPORT_GROWTH,
             REPORT_CAVEAT, BULK_SOURCE, *ALPHA_C_SOURCES.values(), *STATUS_NAMES.values())
"""Texts the page translates (the Reliability paragraph, the table's tooltips)."""


@dataclass(frozen=True)
class FringeGroup:
    time_s: float
    t_lo_s: float
    t_hi_s: float
    status: str
    H_nm: Optional[float] = None
    H_lo_nm: Optional[float] = None
    H_hi_nm: Optional[float] = None
    bound_nm: Optional[float] = None
    method: str = ""

    @property
    def measured(self) -> bool:
        return self.H_nm is not None and self.status in MEASURED


@dataclass
class FringeHeights:
    folder: Path
    groups: list
    summary: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.summary.get("status") or "")

    def measured(self) -> list:
        return [g for g in self.groups if g.measured]

    def nearest(self, time) -> Optional[FringeGroup]:
        """The group whose time span holds ``time``, else the nearest within one group length (None when none)."""
        if time is None or not self.groups:
            return None
        inside = [g for g in self.groups if g.t_lo_s - 1e-9 <= time <= g.t_hi_s + 1e-9]
        if inside:
            return min(inside, key=lambda g: abs(g.time_s - time))
        best = min(self.groups, key=lambda g: abs(g.time_s - time))
        limit = float(self.summary.get("group_seconds") or 1.0)
        return best if abs(best.time_s - time) <= limit else None


def parse_heights(folder: Path, rows: list, summary: dict) -> FringeHeights:
    groups = []
    for row in rows:
        time = number(row.get("time_s"))
        if time is None:
            continue
        groups.append(FringeGroup(
            time_s=time, t_lo_s=number(row.get("t_lo_s")) or time, t_hi_s=number(row.get("t_hi_s")) or time,
            status=str(row.get("status") or ""), H_nm=number(row.get("H_nm")), H_lo_nm=number(row.get("H_lo68_nm")),
            H_hi_nm=number(row.get("H_hi68_nm")), bound_nm=number(row.get("H_upper_bound_nm")),
            method=str(row.get("method") or "")))
    groups.sort(key=lambda g: g.time_s)
    return FringeHeights(Path(folder), groups, summary if isinstance(summary, dict) else {})


def _fmt(value, digits: int = 3) -> str:
    return "–" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}g}"


def alpha_c_source(summary: dict, text: Callable[..., str] = english) -> str:
    record = summary.get("alpha_c") or {}
    source = str(record.get("source") or "")
    if source.startswith("bulk"):
        return text(BULK_SOURCE, metal=summary.get("metal") or "?")
    return text(ALPHA_C_SOURCES[source]) if source in ALPHA_C_SOURCES else source


def fringe_paragraph(heights: Optional[FringeHeights], text: Callable[..., str] = english) -> str:
    """HTML of the Reliability tab's part about H ("" without an analysis)."""
    if heights is None:
        return ""
    s = heights.summary
    ranges = s.get("ranges") or {}
    span = [pair for side in (ranges.get("clean_q_inside_nm_inv") or {}).values() for pair in side]
    low = min((a for a, _b in span), default=None)
    high = max((b for _a, b in span), default=None)
    parts = [text(REPORT_TITLE), text(REPORT_WHAT, low=_fmt(low), high=_fmt(high),
                                      bound=_fmt(ranges.get("H_upper_bound_below_range_nm")))]
    measured = heights.measured()
    if measured:
        parts.append(text(REPORT_MEASURED, measured=len(measured), groups=len(heights.groups),
                          first=f"{measured[0].time_s:.1f}", last=f"{measured[-1].time_s:.1f}",
                          one_side=sum(g.status == "one_side_only" for g in measured)))
    else:
        parts.append(text(REPORT_NONE))
    ambiguous = sum(g.status == "ambiguous_order" for g in heights.groups)
    if ambiguous:
        parts.append(text(REPORT_AMBIGUOUS, count=ambiguous))
    record = s.get("alpha_c") or {}
    if record.get("used_deg") is not None:
        parts.append(text(REPORT_ALPHA_C, alpha_c=f"{float(record['used_deg']):.3f}",
                          error=f"{float(record.get('error_deg') or 0):.3f}", source=alpha_c_source(s, text)))
    growth = s.get("growth") or {}
    if measured and growth.get("fitted"):
        parts.append(text(REPORT_GROWTH, v=f"{growth['v_nm_s']:.4f}", error=f"{growth['v_error_nm_s']:.4f}",
                          t0=f"{growth['t0_s']:.1f}"))
    if measured:
        parts.append(text(REPORT_CAVEAT))
    return "<p>" + " ".join(parts) + "</p>"


__all__ = ["ALPHA_C_SOURCES", "BULK_SOURCE", "FRINGE_FOLDER", "FringeGroup", "FringeHeights",
           "HEIGHTS_FILE", "MEASURED", "STATUS_NAMES", "SUMMARY_FILE", "TEMPLATES", "alpha_c_source",
           "fringe_paragraph", "parse_heights"]
