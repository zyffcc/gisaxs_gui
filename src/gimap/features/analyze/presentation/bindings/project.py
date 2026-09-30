"""The Analyze part of a GIMaP project: the frames listed, the one on screen and the whole set-up
(``settings_record``: mode, geometry, αi, masks, corrections, cuts), to reopen a sample as it was left."""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr

from ...application import settings_from_record, settings_record


class ProjectMixin:
    """Needs ``view_model``, ``file_list``, ``add_paths``, ``clear_files``, ``run_analysis``."""

    def project_state(self) -> dict:
        state = self.view_model.state
        return {
            "files": [str(path) for path in state.files],
            "current": int(self.file_list.currentRow()),
            "frame": int(getattr(state, "frame_index", 0) or 0),
            "settings": settings_record(self.view_model.current_settings()),
        }

    def apply_project_state(self, data: dict) -> list[str]:
        """The frames and set-up of a project; returns notes (files or masks no longer there)."""
        notes: list[str] = []
        self.clear_files()
        files = [str(path) for path in data.get("files") or ()]
        present = [path for path in files if Path(path).exists()]
        missing = len(files) - len(present)
        if missing:
            notes.append(tr("{count} frames of the project are no longer there").format(count=missing))
        if data.get("settings"):
            try:
                notes += self.view_model.apply_settings(settings_from_record(data["settings"]))
                self._sync_settings_widgets()
            except (ValueError, KeyError, TypeError) as exc:
                notes.append(tr("the Analyze set-up could not be read: {reason}").format(reason=exc))
        if present:
            self.add_paths(present)
            current = int(data.get("current", 0) or 0)
            if 0 <= current < self.file_list.count() and current != self.file_list.currentRow():
                self.file_list.setCurrentRow(current)
            frame = int(data.get("frame", 0) or 0)
            if frame:
                self.view_model.set_frame(frame)
            self.run_analysis()
        return notes


__all__ = ["ProjectMixin"]
