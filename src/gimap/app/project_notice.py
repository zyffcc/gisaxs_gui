"""What is said after a project opened with files gone or parts not restored (text only, no dialogs).

``src.gimap.app.menus`` re-exports these names (``opened_notice``, ``NOTE_CHARS``, ``SHOWN_NAMES``, ``SHOWN_NOTES``).
"""

from __future__ import annotations

from pathlib import Path

from .presentation.i18n import tr, trf

SHOWN_NAMES = 3
"""How many of the files a project no longer finds its notice names; the others are counted."""
SHOWN_NOTES = 2
NOTE_CHARS = 120
"""A note in the notice is one line of at most this many characters; the whole of it is in Details…"""


def _short(note: str) -> str:
    line = " ".join(str(note).split())
    return line if len(line) <= NOTE_CHARS else line[: NOTE_CHARS - 1].rstrip() + "…"


def _by_folder(paths) -> str:
    """``paths`` as their folders, each with the names in it (the order kept): a beamtime folder that moved
    is one line, not the same long path in front of every file."""
    folders: dict[str, list[str]] = {}
    for item in paths:
        folders.setdefault(str(Path(item).parent), []).append(Path(item).name)
    return "\n".join("\n".join([folder, *(f"    {name}" for name in names)]) for folder, names in folders.items())


def opened_notice(name: str, missing, notes) -> tuple[str, str, str]:
    """What is said after the project ``name`` opened with files gone (``missing``: their paths) or parts not
    restored (``notes``, as the pages wrote them): the short notice, and for Details… its summary and the
    whole list — every missing file under its folder, every note in full."""
    missing, notes = [str(item) for item in missing], [str(note) for note in notes]
    if len(missing) == 1:
        text = trf("Opened the project {name}. {file} is no longer there.", name=name, file=Path(missing[0]).name)
    elif missing:
        names = ", ".join(Path(item).name for item in missing[:SHOWN_NAMES])
        if len(missing) > SHOWN_NAMES:
            names = trf("{names} and {n} more", names=names, n=len(missing) - SHOWN_NAMES)
        text = trf("Opened the project {name}. {count} files or folders it names are no longer there: {names}",
                   name=name, count=len(missing), names=names)
    else:
        shown = "; ".join(_short(note) for note in notes[:SHOWN_NOTES])
        if len(notes) > SHOWN_NOTES:
            shown = trf("{notes} … and {n} more", notes=shown, n=len(notes) - SHOWN_NOTES)
        text = trf("Opened the project {name}: {notes}", name=name, notes=shown)
    summary, parts = text, []
    if missing:
        summary += "\n\n" + tr("The project keeps the paths as they were saved. If the data moved, open it from its new "
                               "place and save the project; saving now would write the project without what is missing.")
        parts.append(trf("No longer there ({count}):", count=len(missing)) + "\n" + _by_folder(missing))
    if notes:
        parts.append(tr("What the pages could not restore:") + "\n" + "\n".join(notes))
    return text, summary, "\n\n".join(parts)


__all__ = ["NOTE_CHARS", "SHOWN_NAMES", "SHOWN_NOTES", "opened_notice"]
