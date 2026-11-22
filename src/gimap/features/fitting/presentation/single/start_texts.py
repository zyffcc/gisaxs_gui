"""Where a frame's kept fit started, in the interface language: the Start column of the In-situ series table.

The record keeps English (``FrameFit.start_from``, ``start_frame``, ``chain``, ``predict_choice``; the templates of
``application/series_chain``). ``chain_text`` finds the template a text was made from and fills the same values into
its translation, so a cell is said again in the other language after a switch; a text made from no template (a record
of an older run) is translated when it is a key, else shown as it is.

* “continued from frame n”: refined from the result of frame n (the frame before; the frame after in the backward
  pass);
* “restarted from Predict (why)”: the chain broke there, and the fit from Predict's solution was the better one;
* “… (the chain broke: why)”: the chain broke, and the continued fit was still the better one;
* “from the model in Single analysis”: the first frame, a new stage, a frame after one that did not converge.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.gimap.app.presentation.i18n import tr, trf

from ...application.series_chain import BREAKS, CHOICES

CONTINUED = "continued from frame {n}"
CONTINUED_BACKWARD = "continued from frame {n} (backward pass)"
CONTINUED_BEFORE = "continued from the frame before"
CONTINUED_AFTER = "continued from the frame after (backward pass)"
FROM_SINGLE = "from the model in Single analysis"
RESTARTED = "restarted from Predict ({reason})"
BROKE = "{start} (the chain broke: {reason})"
PREDICTED = "Predict parameters"
"""Predict each frame: every frame from a solution of Predict parameters (no chain)."""
CHOSEN = "Predict's solution: {rule}"
START = "Started from"
START_TIP = ("Where the frame's kept fit started: the result of frame n (the frame before; the frame after in the "
             "backward pass), Predict parameters where the chain broke (a failed or unconverged fit, a new stage, a "
             "jump of χ²ᵣ), or the model of Single analysis")
TEXTS = (CONTINUED, CONTINUED_BACKWARD, CONTINUED_BEFORE, CONTINUED_AFTER, FROM_SINGLE, RESTARTED, BROKE, PREDICTED,
         CHOSEN, START, START_TIP, *BREAKS, *CHOICES)
"""Every English text of this module, and the templates it translates (the Chinese table has them all)."""
_FIELD = re.compile(r"\{(\w+)\}")


@lru_cache(maxsize=None)
def _pattern(template: str) -> re.Pattern:
    """``template`` as a pattern that matches the texts made from it (each ``{value}`` as a group)."""
    parts, last = [], 0
    for field in _FIELD.finditer(template):
        parts += [re.escape(template[last:field.start()]), f"(?P<{field.group(1)}>.+?)"]
        last = field.end()
    parts.append(re.escape(template[last:]))
    return re.compile("".join(parts), re.DOTALL)


def chain_text(text: str, templates=BREAKS) -> str:
    """``text`` (made from one of ``templates``, English) in the interface language; the reason of a failed fit is
    translated too (``why``), numbers and names stay as they are."""
    for template in templates:
        found = _pattern(template).fullmatch(str(text))
        if found is None or "{" not in template:
            continue
        values = found.groupdict()
        if "why" in values:
            values["why"] = tr(values["why"])
        return trf(template, **values)
    return tr(str(text))


def start_text(frame) -> str:
    """The Start cell of a frame's row (module docstring); empty for a frame without a fit or a run before the chain
    said where frames start."""
    if not frame.ok:
        return ""
    origin, reason = frame.start_from, chain_text(frame.chain) if frame.chain else ""
    if origin == "predict":
        return trf(RESTARTED, reason=reason) if reason else tr(PREDICTED)
    if origin in ("previous", "next"):
        backward = origin == "next"
        start = (trf(CONTINUED_BACKWARD if backward else CONTINUED, n=frame.start_frame) if frame.start_frame else
                 tr(CONTINUED_AFTER if backward else CONTINUED_BEFORE))
    elif origin == "single":
        start = tr(FROM_SINGLE)
    else:
        return reason
    return trf(BROKE, start=start, reason=reason) if reason else start


def start_tip(frame) -> str:
    """The Start cell's tooltip: the rule that chose Predict's solution when the fit started from one, else what the
    column says."""
    if frame.ok and frame.start_from == "predict" and frame.predict_choice:
        return trf(CHOSEN, rule=chain_text(frame.predict_choice, CHOICES))
    return tr(START_TIP)


__all__ = ["BROKE", "CHOSEN", "CONTINUED", "CONTINUED_AFTER", "CONTINUED_BACKWARD", "CONTINUED_BEFORE", "FROM_SINGLE",
           "PREDICTED", "RESTARTED", "START", "START_TIP", "TEXTS", "chain_text", "start_text", "start_tip"]
