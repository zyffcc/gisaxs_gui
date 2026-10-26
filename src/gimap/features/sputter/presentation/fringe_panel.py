"""Size Trend's panel of the island height H from the moving q_z fringes, and the Windows table's cells of it.

H is drawn with its 68% interval as error bars (filled: both sides of the beam or the phase-locked fit; hollow: one
side only), the nominal thickness δ = rate × (t − t0) as a thin dashed line when the result has one (for comparison:
H is the islands' vertical size, not the deposited amount), and, before the first minimum enters the detector's q_z
range, the bound H stays below as a dotted line. The panel is shown whenever H is measured in some time group, also in
the simple view (it is a direct measurement); its axis is linear from 0 up to the measured values.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr

from ..application import FRINGE_STATUS_NAMES
from .texts import text

H_TITLE = "H, island height from q_z fringes (nm)"
H_LABEL = "H from q_z fringes (68%)"
H_ONE_SIDE = "H, one side of the beam only"
H_NOMINAL = "nominal δ = rate × (t − t0)"
H_BELOW = "below the range: H < {bound} nm"
H_COLOR = "#2a7f62"
"""Readable on the light and the dark plot background."""
BELOW_CELL = "< {bound}"
METHODS = {"minima": "from the minima of known order", "phase-locked fit": "from the phase-locked fit"}
TEMPLATES = (H_TITLE, H_LABEL, H_ONE_SIDE, H_NOMINAL, H_BELOW, *METHODS.values())


def has_heights(fringes) -> bool:
    return fringes is not None and bool(fringes.measured())


def draw_heights(ax, fringes, result) -> None:
    """H with its interval, the nominal thickness of the result (if any) and the bound below the range."""
    for status, label, face in (("measured", tr(H_LABEL), H_COLOR), ("one_side_only", tr(H_ONE_SIDE), "none")):
        groups = [g for g in fringes.measured() if g.status == status]
        if not groups:
            continue
        times = [g.time_s for g in groups]
        values = [g.H_nm for g in groups]
        low = [g.H_nm - (g.H_lo_nm if g.H_lo_nm is not None else g.H_nm) for g in groups]
        high = [(g.H_hi_nm if g.H_hi_nm is not None else g.H_nm) - g.H_nm for g in groups]
        ax.errorbar(times, values, yerr=[low, high], fmt="o", ms=3, lw=.7, color=H_COLOR, mfc=face, mec=H_COLOR,
                    label=label)
    below = [g for g in fringes.groups if g.status == "below_range" and g.bound_nm is not None]
    if below:
        ax.plot([below[0].t_lo_s, below[-1].t_hi_s], [below[0].bound_nm] * 2, ":", lw=1, color=H_COLOR,
                label=text(H_BELOW, bound=f"{below[0].bound_nm:.2f}"))
    nominal = result.column("nominal_thickness_nm") if result is not None else None
    if nominal is not None and nominal.time:
        ax.plot(nominal.time, nominal.value, "--", lw=.8, color="#7f7f7f", label=tr(H_NOMINAL))
    tops = [g.H_hi_nm or g.H_nm for g in fringes.measured()] + [g.bound_nm for g in below]
    ax.set_ylim(0, 1.15 * max(tops))  # linear, from 0, up to the measured values (a long δ line is cut)


def height_cells(fringes, time) -> tuple:
    """((H, tooltip), (68% interval, tooltip)) of the time group nearest to ``time`` (empty without one)."""
    group = fringes.nearest(time) if fringes is not None else None
    if group is None:
        return ("", ""), ("", "")
    meaning = tr(FRINGE_STATUS_NAMES.get(group.status, group.status)) if group.status else ""
    if group.measured:
        tip = f"{meaning} · {tr(METHODS[group.method])}" if group.method in METHODS else meaning
        interval = (f"{group.H_lo_nm:.3g}–{group.H_hi_nm:.3g}"
                    if group.H_lo_nm is not None and group.H_hi_nm is not None else "")
        return (f"{group.H_nm:.3g}", tip), (interval, tip)
    if group.status == "below_range" and group.bound_nm is not None:
        return (text(BELOW_CELL, bound=f"{group.bound_nm:.3g}"), meaning), ("", meaning)
    return ("", meaning), ("", meaning)


__all__ = ["H_COLOR", "H_TITLE", "TEMPLATES", "draw_heights", "has_heights", "height_cells"]
