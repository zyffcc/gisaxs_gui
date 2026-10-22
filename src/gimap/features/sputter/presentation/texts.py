"""The Sputter page's run-time texts in the interface language.

The application and the domain write English from templates (problems of a request, PyTorch's installer,
the toolkit, the result reader, the sketch's notes, the reliability report). A message that reaches the page
as text only (a failed background task) is matched against those templates again and the same values are
filled into the translated template (``trf``), as Compare does (``compare/presentation/errors.py``); file
names, folders and numbers stay as they are. Anything else is translated when it is an exact key of the
table, else shown as it is (the toolkit's own words).
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.gimap.app.presentation.i18n import tr, trf

from ..application import (HINT_TEXT, INSTALL_MESSAGES, INSTALL_STAGES, NO_FRAMES, NO_TABLE, PROBLEMS,
                           SKETCH_TEMPLATES, SOURCE_TEXT, STOPPED, SUMMARY_HINT, TOOLKIT_FAILED, TOOLKIT_MISSING,
                           InstallProgress, Source)

TEMPLATES = (*INSTALL_MESSAGES.values(), TOOLKIT_MISSING, TOOLKIT_FAILED, STOPPED, NO_FRAMES, NO_TABLE,
             *SKETCH_TEMPLATES, *PROBLEMS)
LABEL_TEMPLATES = (*SOURCE_TEXT.values(), *HINT_TEXT.values(), *SUMMARY_HINT.values())
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


def source_text(source: Source) -> str:
    """Where a geometry value comes from (“from preset …”), with a different energy of the files."""
    words = text(SOURCE_TEXT.get(source.kind, ""), name=source.name) if source.kind in SOURCE_TEXT else ""
    if source.hint_kev:
        hint = HINT_TEXT.get(source.hint_source or "header", HINT_TEXT["header"])
        words += " · " + text(hint, energy=f"{source.hint_kev:.4g}")
    return words


def stage_name(stage) -> str:
    """A stage of the toolkit in the interface language (an unknown one as it is)."""
    return tr(STAGE_NAMES[stage]) if stage in STAGE_NAMES else (stage or "")


def install_text(progress: InstallProgress) -> str:
    template = INSTALL_STAGES.get(progress.stage)
    return text(template, **progress.fields()) if template else ""


__all__ = ["LABEL_TEMPLATES", "STAGE_NAMES", "TEMPLATES", "install_text", "message_text", "source_text",
           "stage_name", "text"]
