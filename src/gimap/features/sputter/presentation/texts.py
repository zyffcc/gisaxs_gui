"""The Sputter page's run-time texts in the interface language.

The application and the domain write English from templates (problems of a request, the analysis package's engine,
the result reader). A message that reaches the page as text only (a failed background task) is matched against those
templates again and the same values are filled into the translated template (``trf``), as Compare does
(``compare/presentation/errors.py``); file names, folders and numbers stay as they are. Anything else is translated
when it is an exact key of the table, else shown as it is.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.gimap.app.presentation.i18n import tr, trf

from ..application import (CHECK_SOURCES, CLIENT_PROBLEMS, CLIENT_TEMPLATES, HINT_TEXT, NO_FRAMES, PROBLEMS,
                           SOURCE_TEXT, STATIC_TEMPLATES, Source)

TEMPLATES = (NO_FRAMES, *PROBLEMS, *CLIENT_TEMPLATES, *STATIC_TEMPLATES)
SOURCE_TIPS = {
    "missing": "Not in the frame headers or the scan's files: enter it.",
    "auto": "Empty: the beam centre is estimated from the frames. A centre you give is in Analyze's pixel coordinates.",
    "last": "The value used last, for another folder: check that it holds for this scan.",
    "assumed": "Not in the files: 0 is used. Enter the sample roll if it was not 0.",
}
"""The rest of a source line, in its tooltip (and in the tooltip of its field)."""
HINT_TIPS = {
    "header": "The frame header gives {energy} keV, which differs from the value used: check which is the beam's energy.",
    "nexus": "The scan's NeXus file gives {energy} keV, which differs from the value used: check which is the beam's "
             "energy.",
    "flat_field": "The E in the flat-field file name is the detector's calibration energy, not necessarily the beam's: "
                  "it is never filled in. Enter the beam energy.",
}
"""Why an energy of the files is shown beside the value used."""
SUMMARY_TIPS = ("S/N {serial}", "{pixel} µm pixels", "first frame {first_frame}", "flat field {flat_field}")
"""The parts of the scan's line needed once (matching a preset): its tooltip."""
LABEL_TEMPLATES = (*SOURCE_TEXT.values(), *HINT_TEXT.values(), *SOURCE_TIPS.values(), *HINT_TIPS.values(),
                   *SUMMARY_TIPS)
"""Where a value comes from and what the files say about the energy (shown beside the fields)."""
_FIELD = re.compile(r"\{(\w+)\}")


@lru_cache(maxsize=None)
def _pattern(template: str) -> re.Pattern:
    parts, last = [], 0
    for found in _FIELD.finditer(template):
        parts += [re.escape(template[last:found.start()]), f"(?P<{found.group(1)}>.+?)"]
        last = found.end()
    parts.append(re.escape(template[last:]))
    return re.compile("".join(parts), re.DOTALL)


def text(template: str, **values) -> str:
    """``trf`` that never fails on a template the table holds differently (the English is used then)."""
    try:
        return trf(template, **values)
    except (KeyError, IndexError, ValueError):
        return template.format(**values)


def message_text(message) -> str:
    """``message`` (English made from one of the templates, or several problems of a request) in the
    interface language."""
    message = str(message)
    for template in TEMPLATES:
        if "{" in template:
            found = _pattern(template).fullmatch(message)
            if found is not None:
                return text(template, **found.groupdict())
        elif message == template:
            return tr(template)
    parts = [problem for problem in (*PROBLEMS, *CLIENT_PROBLEMS) if problem in message]
    if parts and " ".join(parts) == message:
        return " ".join(tr(problem) for problem in parts)
    return tr(message)


def source_text(source: Source) -> str:
    """Where a geometry value comes from (“from preset …”), with a different energy of the files; a value to check
    starts with ⚠."""
    words = text(SOURCE_TEXT.get(source.kind, ""), name=source.name) if source.kind in SOURCE_TEXT else ""
    if words and source.kind in CHECK_SOURCES:
        words = "⚠ " + words
    if source.hint_kev:
        hint = HINT_TEXT.get(source.hint_source or "header", HINT_TEXT["header"])
        words += " · " + text(hint, energy=f"{source.hint_kev:.4g}")
    return words


def source_warns(source: Source) -> bool:
    """The source line is a warning: a guessed value to check, or an energy of the files that differs."""
    return source.kind in CHECK_SOURCES or bool(source.hint_kev)


def source_tip(source: Source) -> str:
    """The tooltip of a source line: what it means and what to do (the line itself when it says it all)."""
    tips = [tr(SOURCE_TIPS[source.kind])] if source.kind in SOURCE_TIPS else []
    if source.hint_kev:
        hint = HINT_TIPS.get(source.hint_source or "header", HINT_TIPS["header"])
        tips.append(text(hint, energy=f"{source.hint_kev:.4g}"))
    return "\n".join(tips) or source_text(source)


__all__ = ["HINT_TIPS", "LABEL_TEMPLATES", "SOURCE_TIPS", "SUMMARY_TIPS", "TEMPLATES", "message_text", "source_text",
           "source_tip", "source_warns", "text"]
