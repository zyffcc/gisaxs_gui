"""What Fitting's pages say of a curve as it was taken in (``application.curve_intake``): an implausible q range in
a unit the file did not state, the σ estimated for a file without one, the negative and zero intensities. The same
sentences on the Fit page (Curve card, Fit step, Results) and in the 1D Predict window, in the interface language."""

from __future__ import annotations

from src.gimap.app.presentation.i18n import trf

from ..application.curve_intake import IntakeNote, note_from_dict


def note_text(note) -> str:
    """An ``IntakeNote`` — or a record's entry made of one — in the interface language; a record's entry that is
    none any more (another version's) as its own English text."""
    found = note if isinstance(note, IntakeNote) else note_from_dict(note)
    if found is None:
        return str(note.get("text", "")) if isinstance(note, dict) else str(note)
    return trf(found.template, **found.values)


def warning_lines(notes) -> list[str]:
    """Each note as a warning line (“⚠ …”)."""
    return ["⚠ " + text for text in (note_text(note) for note in notes) if text]


def row_note_lines(rows) -> list[str]:
    """The notes of the candidate rows of a run, each curve half once: “⚠ name / side: …”."""
    seen, lines = set(), []
    for row in rows:
        for note in row.get("curve_notes") or ():
            key = (row.get("file"), row.get("side"), note.get("kind") if isinstance(note, dict) else str(note))
            if key in seen:
                continue
            seen.add(key)
            where = " / ".join(str(part) for part in (row.get("file"), row.get("side")) if part)
            lines.append(f"⚠ {where}: {note_text(note)}" if where else f"⚠ {note_text(note)}")
    return lines


__all__ = ["note_text", "row_note_lines", "warning_lines"]
