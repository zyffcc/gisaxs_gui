"""αi in the command bar: the instrument profile's value, unless the user sets one for this measurement.

A value set by the user is kept for every frame and remembered for the next session, so it must
never go unnoticed: the field is marked (``override`` QSS property), its tooltip names both values,
its context menu and the notice at the start of a session go back to the profile's value in one step.

The value of the last session belongs to the instrument profile it was set with. A first frame of another
profile (or of a detector without one) does not use it: the field goes back to the profile's value, a notice
offers the old value in one step, and the automatic analysis asks for αi with it as a suggestion
(``status()["incidence_suggestion"]``).
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QMenu

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_state

from ..views.analyze_page_view import INCIDENCE_FROM_PROFILE, INCIDENCE_TIP
from .incidence_text import measured_incidence

OVERRIDE_TIP = "αi set by you: {value}° · profile: {profile}°"
RESTORED_NOTICE = "αi {value}° from your last session (profile: {profile}°)"
NOT_USED_NOTICE = (
    "αi {value}° of your last session is not used: it was set with another instrument profile "
    "(this profile: {profile}°)"
)
NOT_USED_NO_PROFILE = "αi {value}° of your last session is not used: this detector has no instrument profile yet"
USE_IT = "Use {value}°"


def _degrees(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value):g}"


class IncidenceMixin:
    """Needs ``incidence_spin``, ``incidence_reset_action``, ``view_model``, ``run_analysis``, ``_remember``, ``_status``."""

    _incidence_notice_pending = False
    """A value restored from the last session that the user has not been told about yet."""
    _incidence_profile: Optional[str] = None
    """The instrument profile in use when the αi in the field was set (remembered with it)."""
    _incidence_suggestion: Optional[float] = None
    """The last session's αi that was not used for this session's frames (another profile's)."""

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
        if analysis.geometry is not None:
            with QSignalBlocker(self.incidence_spin):
                self.incidence_spin.setValue(analysis.geometry.incidence_deg)
        self._show_incidence_state()
        self._tell_restored_incidence(analysis)
        self._keep_incidence_profile(analysis)

    def _forget_incidence(self) -> None:
        """No frame is shown (Clear, a frame that could not be read): the field no longer shows the αi of the
        frame before. A value set by hand stays: it is the person's, for every frame."""
        value = self.view_model.state.incidence_deg
        with QSignalBlocker(self.incidence_spin):
            self.incidence_spin.setValue(INCIDENCE_FROM_PROFILE if value is None else value)
        self._show_incidence_state()

    def _adopt_incidence(self, degrees: Optional[float]) -> None:
        """αi set in this session (by hand, a notice's button, the automatic analysis): no longer the last
        session's, and it belongs to the profile of the frame shown."""
        self.view_model.set_incidence(degrees)
        self._incidence_notice_pending = False
        self._incidence_suggestion = None
        self._incidence_profile = self._profile_name(self.view_model.state.analysis) if degrees is not None else None

    @staticmethod
    def _profile_name(analysis) -> Optional[str]:
        profile = analysis.resolution.profile if analysis is not None and analysis.resolution is not None else None
        return profile.name if profile is not None else None

    def _keep_incidence_profile(self, analysis) -> None:
        """A value of this session goes with the profile it was last used with."""
        name = self._profile_name(analysis)
        if (self.view_model.state.incidence_deg is not None and not self._incidence_notice_pending
                and name is not None and name != self._incidence_profile):
            self._incidence_profile = name
            self._remember()

    def _drop_carried_incidence(self, analysis) -> bool:
        """The first frame of this session was analysed with the last session's αi, which was set with another
        instrument profile: the value is taken out, said, and offered. ``True``: the frame is being analysed
        again without it (its first analysis is not shown)."""
        value = self.view_model.state.incidence_deg
        if not self._incidence_notice_pending or value is None:
            return False
        name = self._profile_name(analysis)
        if analysis.resolution is None or (name is not None and name == self._incidence_profile):
            return False
        profile = analysis.resolution.profile
        self.view_model.set_incidence(None)
        self._incidence_notice_pending = False
        self._incidence_suggestion, self._incidence_profile = float(value), None
        with QSignalBlocker(self.incidence_spin):
            self.incidence_spin.setValue(INCIDENCE_FROM_PROFILE if profile is None else profile.geometry.incidence_deg)
        self._show_incidence_state()
        self._remember()
        text = (trf(NOT_USED_NO_PROFILE, value=_degrees(value)) if profile is None else
                trf(NOT_USED_NOTICE, value=_degrees(value), profile=_degrees(profile.geometry.incidence_deg)))
        show_toast(self.window(), text, level="warning", timeout_ms=20000,
                   action=(trf(USE_IT, value=_degrees(value)), lambda: self._use_incidence(float(value))))
        if analysis.geometry is None:
            return False  # nothing was computed with it
        self.run_analysis()
        return True

    def _use_incidence(self, value: float) -> None:
        with QSignalBlocker(self.incidence_spin):
            self.incidence_spin.setValue(value)
        self._incidence_changed(value)

    def _incidence_changed(self, value: float) -> None:
        degrees = None if value <= INCIDENCE_FROM_PROFILE + 1e-9 else value
        request = self.view_model.request()
        record = request.incidence_measurement if request else None
        self._adopt_incidence(degrees)
        if record and degrees is not None:
            self.view_model.state.incidence_locked = True
            self.view_model.set_incidence_measurement(dict(
                record, applied_deg=None, decision="user_override", override_deg=degrees, override_source="entered"))
        self._show_incidence_state()
        self._remember()
        self.run_analysis()

    def reset_incidence(self) -> None:
        """Back to the instrument profile's αi (for this and every following frame)."""
        self._adopt_incidence(None)
        self.view_model.state.incidence_locked = False
        analysis = self.view_model.state.analysis
        header = (analysis.metadata or {}).get("incidence_deg") if analysis else None
        profile = self._profile_incidence() if header is None else header
        with QSignalBlocker(self.incidence_spin):
            self.incidence_spin.setValue(INCIDENCE_FROM_PROFILE if profile is None else profile)
        self._show_incidence_state()
        self._remember()
        self.run_analysis()

    # -- what the field shows ----------------------------------------------------------------

    def _show_incidence_state(self) -> None:
        value = self.view_model.state.incidence_deg
        analysis = self.view_model.state.analysis
        measured = analysis.incidence_measurement if analysis else None
        header = (analysis.metadata or {}).get("incidence_deg") if analysis else None
        self.incidence_reset_action.setText(tr("Use frame-header αi" if header is not None else "Back to Profile αi"))
        self.incidence_reset_action.setToolTip(
            tr("Use frame-header αi" if header is not None else "Use the grazing angle of the instrument profile again"))
        active = bool(measured and measured.get("applied_deg") is not None)
        set_state(self.incidence_spin, "override", value is not None or active)
        self.incidence_reset_action.setEnabled(value is not None or bool(measured))
        if active:
            self.incidence_spin.setToolTip(measured_incidence(measured))
            return
        if header is not None:
            self.incidence_spin.setToolTip(
                trf("Frame header αi  {value}°", value=_degrees(header)) if value is None
                else trf("Entered αi  {value}° · frame header  {header}°", value=_degrees(value), header=_degrees(header)))
            return
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
        text = trf(RESTORED_NOTICE, value=_degrees(value), profile=_degrees(profile.geometry.incidence_deg))
        show_toast(self.window(), text, level="info", action=(tr("Back to Profile"), self.reset_incidence),
                   timeout_ms=15000)


__all__ = ["IncidenceMixin", "NOT_USED_NOTICE", "NOT_USED_NO_PROFILE", "OVERRIDE_TIP", "RESTORED_NOTICE", "USE_IT"]
