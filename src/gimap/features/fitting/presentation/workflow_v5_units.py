"""The unit of q 1D Predict reads each curve in, as its window says it (``WorkflowV5Dialog``).

A file whose header states the unit of q (Analyze's ``_fit_input.dat``: 1/A) is read in that unit; the setting
“q unit of files without one in their header” applies to the others. The rule is
``application.workflow_v5.q_unit_of_file``, which the job follows too, so the line under the buttons says before a
run what the run does, and each candidate row says what was done. The headers are read once, when the files are
added (``stated_q_unit``: the curve reader of Fitting); a change of the setting only changes the files that state
no unit. A window made without a reader (``stated_q_unit=None``) says the rule instead of the units.
"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf

from ..application.workflow_v5 import FROM_FILE, q_unit_of_file
from .unit_texts import UNIT_LABEL, UNIT_SHOWN, read_in

UNIT_TIP = ("A file whose header states the unit of q is read in that unit, whatever is set here: Analyze's "
            "_fit_input.dat files state 1/A. This unit is for files that state none. Use current cut is converted "
            "by itself.")
STATED = "q in {unit}, as their header states: {names}"
SETTING = "q in {unit}, no unit in their header (Parameters…): {names}"
UNCLEAR = "Not fitted, the unit of q cannot be read: {name} — {reason}"
RULE = "q in the unit a file's header states, else in {unit} (Parameters…)."
NAMES_SHOWN = 4
"""Names a line lists before “… (+n)”."""
UNCLEAR_SHOWN = 3


def names_text(paths) -> str:
    names = [Path(path).name for path in paths]
    if len(names) <= NAMES_SHOWN:
        return ", ".join(names)
    return ", ".join(names[:NAMES_SHOWN]) + f", … (+{len(names) - NAMES_SHOWN})"


class WorkflowV5UnitsMixin:
    """Needs ``self.files``, ``self.unit`` (the combo of the setting) and ``self.stated_q_unit`` (a reader or None)."""

    def _read_units(self) -> None:
        """Each listed file's header, read once: its unit, ``None`` or why it cannot be told."""
        self._stated = {}
        if self.stated_q_unit is None:
            return
        for path in self.files:
            try:
                self._stated[path] = self.stated_q_unit(path)
            except (OSError, ValueError) as exc:
                self._stated[path] = exc

    def _stated_unit(self, path):
        stated = self._stated[path]
        if isinstance(stated, Exception):
            raise stated
        return stated

    def _units_lines(self) -> list[str]:
        """The lines under the file names: which files are read in which unit, and which cannot be."""
        setting = {"q_unit": self.unit.currentData()}
        if self.stated_q_unit is None or set(self._stated) != set(self.files):
            return [trf(RULE, unit=UNIT_SHOWN[setting["q_unit"]])]
        groups, unclear = {}, []
        for path in self.files:
            read = q_unit_of_file(self._stated_unit, path, setting)
            if read.get("error"):
                unclear.append((path, read["error"]))
            else:
                groups.setdefault((read["q_unit_from"] == FROM_FILE, read["q_unit_read"]), []).append(path)
        lines = [trf(STATED if stated else SETTING, unit=UNIT_SHOWN[unit], names=names_text(paths))
                 for (stated, unit), paths in sorted(groups.items(), key=lambda item: (not item[0][0], item[0][1]))]
        lines += [trf(UNCLEAR, name=Path(path).name, reason=tr(reason)) for path, reason in unclear[:UNCLEAR_SHOWN]]
        if len(unclear) > UNCLEAR_SHOWN:
            lines.append(f"… (+{len(unclear) - UNCLEAR_SHOWN})")
        return lines

    @staticmethod
    def _row_unit(row: dict) -> tuple[str, str]:
        """``(after the name in the table, line of its tooltip)`` for a candidate: the unit its curve's q was read in
        (a file's only after the name; the current cut is always nm⁻¹, converted by Fitting)."""
        tip = read_in(row.get("q_unit_read"), row.get("q_unit_from") or "")
        if not tip:
            return "", ""
        return ("" if row.get("file") == "Current curve" else f" · {UNIT_SHOWN[row['q_unit_read']]}"), tip


__all__ = ["UNIT_LABEL", "UNIT_SHOWN", "UNIT_TIP", "WorkflowV5UnitsMixin", "names_text"]
