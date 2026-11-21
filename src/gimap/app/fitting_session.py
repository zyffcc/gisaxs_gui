"""Application-level lifecycle for the legacy-compatible Fitting session."""

from __future__ import annotations

import logging
import os

from PyQt5.QtCore import QTimer

logger = logging.getLogger(__name__)


class FittingSessionCoordinator:
    """Save and restore Fitting session data without owning UI navigation."""

    def __init__(self, settings, fitting_runtime):
        self.settings = settings
        self.fitting_runtime = fitting_runtime

    def load_last_session(self) -> None:
        try:
            session = self.settings.get("fitting", "last_session", {})
            if not session:
                logger.info("No last Fitting session")
                return
            last_file = session.get("last_opened_file")
            if not last_file or not os.path.exists(last_file):
                logger.info("The file of the last Fitting session is gone: not restored")
                return
            logger.info("Restoring the last Fitting session: %s", os.path.basename(last_file))
            QTimer.singleShot(2000, lambda: self.restore(session))
        except Exception as exc:
            logger.exception("The last Fitting session could not be read: %s", exc)

    def restore(self, session: dict) -> None:
        try:
            self.fitting_runtime.restore_session(session)
            logger.info("Last Fitting session restored")
        except Exception as exc:
            logger.exception("The last Fitting session could not be restored: %s", exc)

    def save(self) -> None:
        try:
            session = self.fitting_runtime.get_session_data()
            if session:
                self.settings.set("fitting", "last_session", session)
                self.settings.save()
                logger.info("Fitting session saved")
        except Exception as exc:
            logger.exception("The Fitting session could not be saved: %s", exc)


__all__ = ["FittingSessionCoordinator"]
