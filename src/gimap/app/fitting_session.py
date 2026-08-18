"""Application-level lifecycle for the legacy-compatible Fitting session."""

from __future__ import annotations

import os

from PyQt5.QtCore import QTimer


class FittingSessionCoordinator:
    """Save and restore Fitting session data without owning UI navigation."""

    def __init__(self, settings, fitting_runtime):
        self.settings = settings
        self.fitting_runtime = fitting_runtime

    def load_last_session(self) -> None:
        try:
            session = self.settings.get("fitting", "last_session", {})
            if not session:
                print("Application runtime: No last session data found")
                return
            last_file = session.get("last_opened_file")
            if not last_file or not os.path.exists(last_file):
                print("Application runtime: Last session file does not exist, skipping restore")
                return
            print(f"Preparing to restore last session: {os.path.basename(last_file)}")
            QTimer.singleShot(2000, lambda: self.restore(session))
        except Exception as exc:
            print(f"Application runtime: Failed to load last session: {exc}")

    def restore(self, session: dict) -> None:
        try:
            self.fitting_runtime.restore_session(session)
            print("Application runtime: Last session restored")
        except Exception as exc:
            print(f"Application runtime: Failed to restore last session: {exc}")

    def save(self) -> None:
        try:
            session = self.fitting_runtime.get_session_data()
            if session:
                self.settings.set("fitting", "last_session", session)
                self.settings.save()
                print("Application runtime: Current session saved")
        except Exception as exc:
            print(f"Application runtime: Failed to save current session: {exc}")


__all__ = ["FittingSessionCoordinator"]
