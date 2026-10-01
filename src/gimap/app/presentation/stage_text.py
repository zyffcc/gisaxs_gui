"""Texts about stages and odd frames (``shared/series_stages``) in the interface language — the same
words in Analyze ▸ Series, Fitting ▸ In-situ series and Compare. The kernel itself stays English
(its texts go into exported tables and records)."""

from __future__ import annotations

from typing import Callable

from .i18n import tr


def odd_reason(frame) -> str:
    if frame.narrow:
        return tr("differs only near q {q}: probably the detector, not the sample").format(q=f"{frame.q:.4g}")
    return tr("differs from the frames before and after it ({times}× their step), most near q {q}").format(
        times=f"{frame.z:.0f}", q=f"{frame.q:.4g}")


def change_text(change) -> str:
    grows = ", ".join(f"{q:.4g} (+{percent:.0f} %)" for q, percent in change.rises)
    drops = ", ".join(f"{q:.4g} ({percent:.0f} %)" for q, percent in change.falls)
    parts = [tr("grows most at q {list}").format(list=grows) if grows else "",
             tr("falls most at q {list}").format(list=drops) if drops else ""]
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


__all__ = ["change_text", "odd_reason", "stages_summary"]
