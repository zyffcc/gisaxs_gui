"""Compare's errors in the interface language.

The domain and the application raise ``ValueError`` whose English is made from the templates in ``ERRORS``
(“Fewer than two curve files (.dat, …) in run_7.”); a comparison that fails in the background reaches the page
as that text only (``TaskRunner``). ``error_text`` finds the template again and fills the same values into the
translated template (``trf``): names, folders, file suffixes and axis labels stay as they are. Any other text — a
reason of the shared series stages (“widen the q range”), of the curve reader, of the operating system — is
translated when it is an exact key, else shown as it is. The page keeps the English and calls ``error_text``
whenever it shows the error, so a switch of the language says it again in the new one.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.gimap.app.presentation.i18n import tr, trf

from ..application import ERRORS, TOO_FEW_READ

_FIELD = re.compile(r"\{(\w+)\}")
_UNREAD = re.compile(r"(?P<name>.+) \((?P<error>.+)\)", re.DOTALL)
"""One file of ``TOO_FEW_READ``: ``UNREAD_FILE`` (“run_3.dat (fewer than three points …)”); the name may hold
brackets of its own, so it takes all but the last pair."""


@lru_cache(maxsize=None)
def _pattern(template: str) -> re.Pattern:
    """``template`` as a pattern that matches the texts made from it (each ``{value}`` as a group)."""
    parts, last = [], 0
    for field in _FIELD.finditer(template):
        parts += [re.escape(template[last:field.start()]), f"(?P<{field.group(1)}>.+?)"]
        last = field.end()
    parts.append(re.escape(template[last:]))
    return re.compile("".join(parts), re.DOTALL)


def _unread(text: str) -> str:
    found = _UNREAD.fullmatch(text)
    return f"{found['name']} ({error_text(found['error'])})" if found else text


def error_text(message) -> str:
    """``message`` (an error's English) in the interface language."""
    message = str(message)
    for template in ERRORS:
        if "{" not in template:
            continue  # an exact key: ``tr`` below
        found = _pattern(template).fullmatch(message)
        if found is None:
            continue
        values = found.groupdict()
        if template == TOO_FEW_READ:
            values["files"] = "; ".join(_unread(part) for part in values["files"].split("; "))
        return trf(template, **values)
    return tr(message)


__all__ = ["error_text"]
