"""αi in the command bar: the instrument profile's value, unless the user sets one for this measurement.

A value set by the user is kept for every frame and remembered for the next session, so it must
never go unnoticed: the field is marked (``override`` QSS property), its tooltip names both values,
its context menu and the notice at the start of a session go back to the profile's value in one step.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QMenu

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme import set_state

from ..views.analyze_page_view import INCIDENCE_FROM_PROFILE, INCIDENCE_TIP

OVERRIDE_TIP = "αi set by you: {value}° · profile: {profile}°"
RESTORED_NOTICE = "αi {value}° from your last session (profile: {profile}°)"


def _degrees(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value):g}"


class IncidenceMixin:
    """Needs ``incidence_spin``, ``incidence_reset_action``, ``view_model``, ``run_analysis``, ``_remember``, ``_status``."""

    _incidence_notice_pending = False
    """A value restored from the last session that the user has not been told about yet."""

    def _connect_incidence(self) -> None:
        self.incidence_menu = QMenu(self.incidence_spin)  # one menu, shown again on every right-click
        self.incidence_menu.addAction(self.incidence_reset_action)
        self.incidence_spin.customContextMenuRequested.connect(self._incidence_menu)
        self.incidence_reset_action.triggered.connect(self.reset_incidence)
        self._show_incidence_state()

    def _incidence_menu(self, position) -> None:
        self.incidence_menu.exec_(self.incidence_spin.mapToGlobal(position))

    def _profile_incidence(self) -> Optional[float]:
        analysis = self.view_model.state.analysis
        profile = analysis.resolution.profile if analysis is not None and analysis.resolution is not None else None
        return profile.geometry.incidence_deg if profile is not None else None

    # -- changes -----------------------------------------------------------------------------

    def _update_incidence(self, analysis) -> None:
        """A frame is shown: the field shows the αi in use (the profile's, unless one is set)."""
        if self.view_model.state.incidence_deg is None and analysis.geometry is not None:
            with QSignalBlocker(self.incidence_spin):
                self.incidence_spin.setValue(analysis.geometry.incidence_deg)
        self._show_incidence_state()
        self._tell_restored_incidence(analysis)

    def _incidence_changed(self, value: float) -> None:
        self.view_model.set_incidence(None if value <= INCIDENCE_FROM_PROFILE + 1e-9 else value)
        self._incidence_notice_pending = False
        self._show_incidence_state()
        self._remember()
        self.run_analysis()

    def reset_incidence(self) -> None:
        """Back to the instrument profile's αi (for this and every following frame)."""
        self.view_model.set_incidence(None)
        self._incidence_notice_pending = False
        profile = self._profile_incidence()
        with QSignalBlocker(self.incidence_spin):
            self.incidence_spin.setValue(INCIDENCE_FROM_PROFILE if profile is None else profile)
        self._show_incidence_state()
        self._remember()
        self.run_analysis()

    # -- what the field shows ----------------------------------------------------------------

    def _show_incidence_state(self) -> None:
        value = self.view_model.state.incidence_deg
        set_state(self.incidence_spin, "override", value is not None)
        self.incidence_reset_action.setEnabled(value is not None)
        if value is None:
            self.incidence_spin.setToolTip(tr(INCIDENCE_TIP))
            return
        tip = tr(OVERRIDE_TIP).format(value=_degrees(value), profile=_degrees(self._profile_incidence()))
        self.incidence_spin.setToolTip(tip + "\n" + tr("Right-click: Back to Profile αi"))

    def _tell_restored_incidence(self, analysis) -> None:
        """Once, with the first frame that has a profile: αi comes from the last session, not from the profile."""
        if not self._incidence_notice_pending:
            return
        value = self.view_model.state.incidence_deg
        profile = analysis.resolution.profile if analysis.resolution is not None else None
        if value is None:
            self._incidence_notice_pending = False
            return
        if profile is None:
            return  # no profile to compare with yet: tell it with the first frame that has one
        self._incidence_notice_pending = False
        text = tr(RESTORED_NOTICE).format(value=_degrees(value), profile=_degrees(profile.geometry.incidence_deg))
        show_toast(self.window(), text, level="info", action=(tr("Back to Profile"), self.reset_incidence),
                   timeout_ms=15000)


__all__ = ["IncidenceMixin", "OVERRIDE_TIP", "RESTORED_NOTICE"]
