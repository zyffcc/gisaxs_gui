"""The unit of q of the curves of the In-situ series, said in the Curves step.

Every curve file is read in the unit its header states (Analyze's ``_fit_input.dat``: 1/A), else in Single analysis'
setting “q unit of files without one in their header” — the one at Start once the series has started
(``application.curve_units``, the rule of all of Fitting). The Curves step counts the listed curves by unit and by
where it came from, and names a file whose unit cannot be told (its header states both units): such a file is not
fitted, its frame says why. The headers are read once per file (``view_model.stated_q_unit``). Detector frames are
cut in Å⁻¹ and the frames card says how: no line here.
"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf

from ...application.curve_units import FROM_FILE, q_unit_read
from ..unit_texts import UNIT_SHOWN

STATED = "{count} curves: q in {unit}, as their header states"
SETTING = "{count} curves: q in {unit}, no unit in their header (Single analysis ▸ Curve ▸ File)"
UNCLEAR = "{count} curves are not fitted, the unit of q cannot be read: {name} — {reason}"


class FitSeriesUnitsMixin:
    """Needs the series page: ``paths``, ``view_model``, ``units_note``, ``_listed()``, ``_preview_unit()``,
    ``frames_mode()``."""

    _stated_units: dict = None
    """The unit each file's header states, ``None`` (none) or why it cannot be told, by path."""

    def _stated_unit(self, path: str):
        if self._stated_units is None:
            self._stated_units = {}
        if path not in self._stated_units:
            read = getattr(self.view_model, "stated_q_unit", None)
            try:
                self._stated_units[path] = None if read is None else read(path)
            except (OSError, ValueError) as exc:
                self._stated_units[path] = exc
        stated = self._stated_units[path]
        if isinstance(stated, Exception):
            raise stated
        return stated

    def _units_lines(self) -> list[str]:
        setting = self._preview_unit()
        groups, unclear = {}, []
        for index in self._listed():
            path = self.paths[index]
            try:
                unit, origin = q_unit_read(self._stated_unit, path, setting)
            except (OSError, ValueError) as exc:
                unclear.append((path, str(exc)))
                continue
            groups.setdefault((origin != FROM_FILE, unit), []).append(path)
        lines = [trf(SETTING if from_setting else STATED, count=len(paths), unit=UNIT_SHOWN[unit])
                 for (from_setting, unit), paths in sorted(groups.items())]
        if unclear:
            name, reason = unclear[0]
            lines.append(trf(UNCLEAR, count=len(unclear), name=Path(name).name, reason=tr(reason)))
        return lines

    def _show_units(self) -> None:
        """The Curves step's line about the unit of q of the listed curves (refreshed with the step)."""
        lines = [] if self.frames_mode() or not self.paths else self._units_lines()
        self.units_note.setText("\n".join(lines))
        self.units_note.setVisible(bool(lines))


__all__ = ["FitSeriesUnitsMixin", "SETTING", "STATED", "UNCLEAR"]
