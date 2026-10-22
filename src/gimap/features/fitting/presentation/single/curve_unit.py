"""The unit of q of the curve on Single analysis: the one its file's header states (Analyze's ``_fit_input.dat``:
1/A), else the setting Curve ▸ File ▸ “q unit of files without one in their header” (``application.curve_units``,
the rule of all of Fitting; ``LoadCurve`` follows it). The Curve step says which unit and where it came from. A
change of the setting opens again only a curve whose file states no unit; for the other it says why nothing
changes. In-situ series reads the files that state none in this setting too (``unit_setting``).

A curve saved before (the last session, a project) is opened with the unit it was read in then, as the setting
for a file without one: such a file opens as before. A file whose header states another unit than the one saved
is read in the stated unit — the saved one was the setting, not the file — and the status line says so.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.gimap.app.presentation.i18n import trf

from ...application.curve_units import FROM_FILE
from ..unit_texts import UNIT_SHOWN, read_in

SETTING_KEY = "unit_setting"
"""The setting, remembered for the next session (and kept with a project)."""
NOT_FOLLOWED = "{name} states q in {unit} in its header: the setting is for files without one."
SAVED_OTHER = "{name} states q in {stated} in its header: read in {stated}, not in {saved} as saved before."


class CurveUnitMixin:
    """Needs the Single analysis page: ``unit_combo``, ``session``, ``open_curve``, ``_status``, ``_remember``,
    ``_remembered``."""

    def setup_curve_unit(self) -> None:
        self.set_unit_setting(self._remembered(SETTING_KEY))
        self.unit_combo.activated.connect(lambda _index: self._unit_setting_changed())

    def unit_setting(self) -> str:
        """The unit of q of files whose header states none (``angstrom`` / ``nm``)."""
        return self.unit_combo.currentData() or "angstrom"

    def set_unit_setting(self, unit) -> None:
        index = self.unit_combo.findData(unit)
        if index >= 0:
            self.unit_combo.setCurrentIndex(index)

    def _unit_setting_changed(self) -> None:
        self._remember(**{SETTING_KEY: self.unit_setting()})
        curve = self.session.curve
        if curve is None or not curve.path:
            return
        if curve.unit_from == FROM_FILE:
            name, unit = curve.name, UNIT_SHOWN[curve.source_unit]
            self._status(lambda: trf(NOT_FOLLOWED, name=name, unit=unit))
        elif self.unit_setting() != curve.source_unit:
            self.open_curve(curve.path, self.session.side)

    def _follow_saved_unit(self) -> None:
        """A file without a unit, opened in a saved unit: the setting shows that unit, as when it was saved."""
        curve = self.session.curve
        if curve is not None and curve.unit_from != FROM_FILE:
            self.set_unit_setting(curve.source_unit)

    def _unit_line(self) -> str:
        """The Curve step's line: “q read in Å⁻¹: stated in the file”."""
        curve = self.session.curve
        return "" if curve is None else read_in(curve.source_unit, curve.unit_from)

    def _saved_unit_note(self, saved: Optional[str]) -> Optional[Callable[[], str]]:
        """When the curve just opened states another unit than the one it was saved with: what is said of it."""
        curve = self.session.curve
        if curve is None or saved not in UNIT_SHOWN or curve.unit_from != FROM_FILE or saved == curve.source_unit:
            return None
        name, stated, before = curve.name, UNIT_SHOWN[curve.source_unit], UNIT_SHOWN[saved]
        return lambda: trf(SAVED_OTHER, name=name, stated=stated, saved=before)


__all__ = ["CurveUnitMixin", "NOT_FOLLOWED", "SAVED_OTHER", "SETTING_KEY"]
