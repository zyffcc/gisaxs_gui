"""Files and folders dropped on the Format Converter become inputs."""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr


def dropped_paths(mime) -> list[Path]:
    """The local files and folders of a drop."""
    if mime is None or not mime.hasUrls():
        return []
    return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]


class DropMixin:
    """Own drag-and-drop of detector files and folders."""

    def _converting(self) -> bool:
        thread = self._conversion_thread
        return thread is not None and thread.isRunning()

    def _droppable(self, paths: list[Path]) -> bool:
        return any(
            path.is_dir() or (path.is_file() and self.view_model.supports_input_path(str(path)))
            for path in paths
        )

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self._converting() and self._droppable(dropped_paths(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = dropped_paths(event.mimeData())
        if self._converting() or not self._droppable(paths):
            event.ignore()
            return
        event.acceptProposedAction()
        self.add_dropped(paths)

    def add_dropped(self, paths: list[Path]) -> None:
        """Supported files as they are; a folder adds its detector files (not those of subfolders)."""
        inputs = [str(path) for path in paths if path.is_file() and self.view_model.supports_input_path(str(path))]
        for folder in (path for path in paths if path.is_dir()):
            inputs.extend(self.view_model.scan_folder(str(folder)))
        if not inputs:
            self.input_note.setText(tr("Nothing to add: no NXS, CBF or TIFF files in what was dropped."))
            return
        self.view_model.remember_folder(str(paths[0]))
        if self._many_files_confirmed(len(inputs)):
            self.add_paths(inputs)
