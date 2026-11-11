"""The Analyze part of a GIMaP project: the frames listed, the one on screen and the whole set-up
(``settings_record``: mode, geometry, αi, masks, corrections, cuts), to reopen a sample as it was left."""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf

from ...application import settings_from_record, settings_record


class ProjectMixin:
    """Needs ``view_model``, ``file_list``, ``add_paths``, ``clear_files``, ``run_analysis``."""

    def project_state(self) -> dict:
        state = self.view_model.state
        return {
            "files": [str(path) for path in state.files],
            "current": int(self.file_list.currentRow()),
            "frame": int(getattr(state, "frame_index", 0) or 0),
            "import_selections": self.view_model.export_import_selections(),
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
            self._import_applying = True
            try:
                self.add_paths(present, preserve_order=True)
                notes += self.view_model.apply_import_selections(data.get("import_selections") or {}, sources=files)
            finally:
                self._import_applying = False
            sources = self.view_model.state.files
            current = min(max(0, int(data.get("current", 0) or 0)), len(files) - 1)
            saved_source = files[current]
            found = self.view_model.locate_source_frame(saved_source, int(data.get("frame", 0) or 0))
            if found is None and sources:
                notes.append(trf("The previously displayed source {name} is unavailable; opened the first imported frame of another source.",
                                 name=Path(saved_source).name))
                found = next(((source, indices[0]) for source in sources
                              if len(indices := self.view_model.selected_frame_indices(source))), None)
            if found is not None:
                source, frame = found
                indices = self.view_model.selected_frame_indices(source)
                if len(indices):
                    self.show_frame(sources.index(source), frame if frame in indices else indices[0])
            self.refresh_frame_list()
        return notes


__all__ = ["ProjectMixin"]
