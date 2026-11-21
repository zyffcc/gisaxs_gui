"""Undo and Redo of the single-curve page, said: the status line names the step taken back or brought again
(“Undone: the fit (Refine).”), and the buttons' tooltips name the step they would take (``FitSession`` keeps a label
with every step: ``Snapshot.label``)."""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import model_to_dict
from .solution_texts import step_text

UNDONE = "Undone: {step}."
REDONE = "Redone: {step}."
UNDO_TIP = "Undo: {step} (Ctrl+Z)"
REDO_TIP = "Redo: {step} (Ctrl+Shift+Z)"
UNDO_COVERS = "Undo the last change of the model, the fitting range or the left-out points, or the last fit (Ctrl+Z)"
REDO_COVERS = "Redo (Ctrl+Shift+Z)"
"""The buttons' tooltips while there is nothing to take back (the view's own): what Undo covers."""


class HistoryMixin:
    """Needs the page's ``session``, ``undo_button``, ``redo_button``, ``_status``, ``_remember``, ``_remember_curve``,
    ``_render_all``."""

    def undo(self) -> None:
        """The model, the fitting range and the left-out points before the last change; back at a fit's
        own, the fit holds again (its quality and errors)."""
        if self.session.undo():
            self._history_moved(UNDONE)

    def redo(self) -> None:
        if self.session.redo():
            self._history_moved(REDONE)

    def _history_moved(self, template: str) -> None:
        self._remember(model=model_to_dict(self.session.model))
        self._remember_curve()
        self._render_all()
        step = self.session.last_step
        self._status(lambda: trf(template, step=step_text(step)), "info")

    def undo_names(self) -> tuple[str, str]:
        """What the next Undo and Redo take back or bring again, in the interface language (``""``: nothing): Edit ▸
        Undo / Redo."""
        session = self.session
        return (step_text(session.undo_label()) if session.can_undo() else "",
                step_text(session.redo_label()) if session.can_redo() else "")

    def refresh_history_buttons(self) -> None:
        session = self.session
        self.undo_button.setEnabled(session.can_undo())
        self.redo_button.setEnabled(session.can_redo())
        self.undo_button.setToolTip(trf(UNDO_TIP, step=step_text(session.undo_label())) if session.can_undo()
                                    else tr(UNDO_COVERS))
        self.redo_button.setToolTip(trf(REDO_TIP, step=step_text(session.redo_label())) if session.can_redo()
                                    else tr(REDO_COVERS))


__all__ = ["REDONE", "REDO_COVERS", "REDO_TIP", "UNDONE", "UNDO_COVERS", "UNDO_TIP", "HistoryMixin"]
