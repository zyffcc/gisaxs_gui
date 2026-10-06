"""Texts about stages and odd frames (``shared/series_stages``) in the interface language — the same
words in Analyze ▸ Series, Fitting ▸ In-situ series and Compare. The kernel itself stays English
(its texts go into exported tables and records)."""

from __future__ import annotations

from typing import Callable

from .i18n import tr


def axis_symbol(x_label: str) -> str:
    """The axis a series runs along, for ``odd_reason`` and ``change_text``: “q (Å⁻¹)” → “q”,
    “χ (°)” → “χ” (as ``CurvePlot.set_labels`` reads it)."""
    return str(x_label).split(" (")[0].split(" or ")[0].strip().strip("|") or "x"


def odd_reason(frame, axis: str = "q") -> str:
    """Why a frame is odd; ``axis`` names the x of the series (“q”, “χ” …; ``frame.q`` is on it)."""
    if frame.narrow:
        return tr("differs only near {axis} {q}: probably the detector, not the sample").format(
            axis=axis, q=f"{frame.q:.4g}")
    return tr("differs from the frames before and after it ({times}× their step), most near {axis} {q}").format(
        times=f"{frame.z:.0f}", axis=axis, q=f"{frame.q:.4g}")


def change_text(change, axis: str = "q") -> str:
    """What grows and falls from one stage to the next; ``axis`` names the x of the series."""
    grows = ", ".join(f"{q:.4g} (+{percent:.0f} %)" for q, percent in change.rises)
    drops = ", ".join(f"{q:.4g} ({percent:.0f} %)" for q, percent in change.falls)
    parts = [tr("grows most at {axis} {list}").format(axis=axis, list=grows) if grows else "",
             tr("falls most at {axis} {list}").format(axis=axis, list=drops) if drops else ""]
    return tr("Stage {a} → {b}: {what}").format(a=change.stage - 1, b=change.stage,
                                               what="; ".join(part for part in parts if part))


def stages_summary(stages, frame: Callable[[int], str] = lambda row: str(row + 1)) -> str:
    """“3 stages: frames 1–52, 53–170, 171–403 · 3 odd frames”; ``frame(row)`` names a frame."""
    if stages.count > 1:
        ranges = ", ".join(f"{frame(first)}–{frame(last)}" for first, last in stages.ranges())
        text = tr("{count} stages: frames {ranges}").format(count=stages.count, ranges=ranges)
    else:
        text = tr("One stage: no change of course")
    if stages.odd:
        text += " · " + (tr("1 odd frame") if len(stages.odd) == 1 else tr("{n} odd frames").format(n=len(stages.odd)))
    return text


__all__ = ["axis_symbol", "change_text", "odd_reason", "stages_summary"]
