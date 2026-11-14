"""The Sputter page's run-time texts in the interface language.

The application and the domain write English from templates (problems of a request, PyTorch's installer,
the toolkit, the result reader, the sketch's notes, the reliability report). A message that reaches the page
as text only (a failed background task) is matched against those templates again and the same values are
filled into the translated template (``trf``), as Compare does (``compare/presentation/errors.py``); file
names, folders and numbers stay as they are. Anything else is translated when it is an exact key of the
table, else shown as it is. The toolkit's own sentences (notes, reasons, the literature checks) are matched the
same way against its templates (``toolkit_texts.TOOLKIT_TEMPLATES``): ``note_text``.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.gimap.app.presentation.i18n import tr, trf

from ..application import (CHECK_SOURCES, CLIENT_TEMPLATES, FRINGES_FAILED, FRINGES_MISSING, HINT_TEXT,
                           INSTALL_MESSAGES, INSTALL_STAGES, NO_FRAMES, NO_TABLE, PROBLEMS,
                           SKETCH_TEMPLATES, SOURCE_TEXT, STATIC_TEMPLATES, STOPPED, TOOLKIT_FAILED, TOOLKIT_MISSING,
                           WINDOW_TEMPLATES, InstallProgress, Source)
from .toolkit_texts import TOOLKIT_TEMPLATES

TEMPLATES = (*INSTALL_MESSAGES.values(), TOOLKIT_MISSING, TOOLKIT_FAILED, STOPPED, NO_FRAMES, NO_TABLE,
             *SKETCH_TEMPLATES, *WINDOW_TEMPLATES, *PROBLEMS, FRINGES_FAILED, FRINGES_MISSING, *CLIENT_TEMPLATES,
             *STATIC_TEMPLATES)
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
STAGE_NAMES = {
    "pre_nucleation": "pre-nucleation", "islands": "islands", "coalescence": "coalescence", "film": "film",
    "pre_nucleation_candidate": "pre-nucleation (candidate)",
    "separate_islands_candidate": "separate islands (candidate)",
    "coalescence_candidate": "coalescence (candidate)",
    "coalescence_or_film_candidate": "coalescence or film (candidate)",
    "continuous_film_candidate": "continuous film (candidate)",
}
"""The toolkit's stage names (on the physical path, and the per-window rule's candidates) as the page says them."""
_FIELD = re.compile(r"\{(\w+)\}")
SYMBOLS = ((re.compile(r"\(2\*pi/sqrt\(3\)\)"), "(2π/√3)"), (re.compile(r"2\*pi/q\*"), "2π/q*"),
           (re.compile(r"\bdelta_c\b"), "δc"), (re.compile(r"\bdelta\b"), "δ"), (re.compile(r"\bchi\b"), "χ"),
           (re.compile(r"\brho\b"), "ρ"), (re.compile(r"\beta\b"), "η"), (re.compile(r"<="), "≤"),
           (re.compile(r">="), "≥"))
"""The toolkit writes symbols in ASCII (``delta``, ``2*pi/q*``); the page shows them as symbols."""


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
    parts = [problem for problem in PROBLEMS if problem in message]
    if parts and " ".join(parts) == message:
        return " ".join(tr(problem) for problem in parts)
    return tr(message)


@lru_cache(maxsize=4096)
def _toolkit_match(sentence: str):
    """(template, values) of the toolkit's template a sentence was made from, or None."""
    for template in TOOLKIT_TEMPLATES:
        if "{" not in template:
            if sentence == template:
                return template, ()
            continue
        found = _pattern(template).fullmatch(sentence)
        if found is not None:
            return template, tuple(found.groupdict().items())
    return None


def note_text(sentence) -> str:
    """A sentence of the toolkit (a note, a reason, a literature check) in the interface language; one it words
    otherwise stays as it is."""
    sentence = "" if sentence is None else str(sentence)
    if not sentence:
        return sentence
    found = _toolkit_match(sentence)
    if found is None:
        return message_text(sentence)
    template, values = found
    return symbols(text(template, **dict(values)) if values else tr(template))


def symbols(sentence: str) -> str:
    """The toolkit's ASCII symbols as symbols (δ, 2π/q*, χ, ρ, η, ≤, ≥); column names stay as they are."""
    for pattern, symbol in SYMBOLS:
        sentence = pattern.sub(symbol, sentence)
    return sentence


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


def stage_name(stage) -> str:
    """A stage of the toolkit in the interface language (an unknown one as it is)."""
    return tr(STAGE_NAMES[stage]) if stage in STAGE_NAMES else (stage or "")


def install_text(progress: InstallProgress) -> str:
    template = INSTALL_STAGES.get(progress.stage)
    return text(template, **progress.fields()) if template else ""


__all__ = ["HINT_TIPS", "LABEL_TEMPLATES", "SOURCE_TIPS", "STAGE_NAMES", "SUMMARY_TIPS", "TEMPLATES", "TOOLKIT_TEMPLATES",
           "install_text", "message_text", "note_text", "source_text", "source_tip", "source_warns", "stage_name",
           "symbols", "text"]
