"""Export of the frame on screen, figures, the Fit hand-off and folder watching for Analyze (batches: ``batch_export.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QListWidgetItem

from src.gimap.shared.figures import FIGURE_FILE_FILTER


class BatchWatchMixin:
    """Own everything that writes results or follows new files."""

    def export_current(self) -> None:
        """Export next to the data; if that folder is not writable, ask for another one."""
        try:
            written = self.view_model.export()
        except ValueError as exc:
            self._status(f"Export failed: {exc}", "error")
            return
        except OSError as exc:
            folder = QFileDialog.getExistingDirectory(
                self, f"Cannot write next to the data ({exc.strerror or exc}); export to…",
                self._last_folder,
            )
            if not folder:
                self._status(f"Export failed: {exc}", "error")
                return
            try:
                written = self.view_model.export(Path(folder))
            except (ValueError, OSError) as retry_error:
                self._status(f"Export failed: {retry_error}", "error")
                return
        self.notify_written(f"Exported {len(written)} files to {written[0].parent}", written[0].parent)

    # -- figures ---------------------------------------------------------------------

    FIGURE_FILTER = FIGURE_FILE_FILTER

    def _figure_path(self, title: str, suffix: str) -> Optional[Path]:
        analysis = self.view_model.state.analysis
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        stem = analysis.path.stem if analysis is not None else "figure"
        path, _ = QFileDialog.getSaveFileName(
            self, title, str(folder / f"{stem}_{suffix}.png"), self.FIGURE_FILTER
        )
        return Path(path) if path else None

    def save_image(self) -> None:
        state = self.detector_view.display_state()
        if state is None:
            self._status("Open a frame first.", "warning")
            return
        path = self._figure_path("Save Image", ("detector", "qmap", "cake")[self.view_combo.currentIndex()])
        if path is None:
            return
        image = state.pop("image")
        try:
            written = self.view_model.save_image_figure(path, image, **state)
        except (ValueError, OSError) as exc:
            self._status(f"Could not save the image: {exc}", "error")
            return
        self.notify_written(f"Saved {written.name}", written.parent)

    def save_view_data(self) -> None:
        """The q map or the cake as a CSV table with its axes (the view shown)."""
        index = self.view_combo.currentIndex()
        if index == 1:
            self.save_map_data()
            return
        cached = getattr(self, "_cake", None)
        analysis = self.view_model.state.analysis
        if index != 2 or cached is None or cached[0] is not analysis:
            self._status("Show the q map or the cake to save it as data.", "warning")
            return
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        path, _ = QFileDialog.getSaveFileName(self, "Save Cake Data", str(folder / f"{analysis.path.stem}_cake.csv"), "CSV table (*.csv)")
        if not path:
            return
        try:
            written = self.view_model.export_cake(analysis, cached[1], Path(path))
        except (ValueError, OSError) as exc:
            self._status(f"Could not save the cake: {exc}", "error")
            return
        self.notify_written(f"Saved {written.name}", written.parent)

    def save_plot_data(self, which: str) -> None:
        """The curves of the upper or lower plot, one CSV each (with the JSON record)."""
        keys = self._plot_keys.get("top" if which == "upper" else "bottom", [])
        destination = self.view_model.default_export_dir()
        if not keys or destination is None:
            self._status("The plot shows no curve.", "warning")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save the Curves of the Plot", str(destination))
        if not folder:
            return
        try:
            written = self.view_model.export_curves(keys, Path(folder))
        except (ValueError, OSError) as exc:
            self._status(f"Could not save the curves: {exc}", "error")
            return
        self.notify_written(f"Saved {len(written)} files", written[0].parent)

    def save_plot(self, plot, suffix: str) -> None:
        state = plot.figure_state()
        path = self._figure_path("Save Plot", suffix)
        if path is None:
            return
        curves = state.pop("curves")
        try:
            written = self.view_model.save_curves_figure(path, curves, **state)
        except (ValueError, OSError) as exc:
            self._status(f"Could not save the plot: {exc}", "error")
            return
        self.notify_written(f"Saved {written.name}", written.parent)

    def fit_current(self) -> None:
        if self._send_to_fitting is None:
            return
        try:
            path = self.view_model.write_fit_input()
            self._send_to_fitting(path, self.view_model.fit_side)
        except (ValueError, OSError, RuntimeError) as exc:
            self._status(f"Could not open the curve in Fitting: {exc}", "error")
            return
        self._status(f"Opened {path.name} in Fitting.", "ok")

    # -- watch -------------------------------------------------------------------------

    def _watch_clicked(self, checked: bool) -> None:
        if not checked:
            self.stop_watch()
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Watch a folder for new frames", self._last_folder
        )
        if folder:
            self._remember(last_folder=folder)
            self.start_watch(Path(folder))
        else:
            self.watch_button.setChecked(False)

    def start_watch(self, folder: Path) -> None:
        before = len(self.view_model.state.files)
        self.view_model.start_watch(folder)
        self._append_list_items(self.view_model.state.files[before:])
        self.watch_button.setChecked(True)
        self._watch_timer.start()
        self._status(f"Watching {folder} for new frames …")

    def stop_watch(self) -> None:
        self._watch_timer.stop()
        self.view_model.stop_watch()
        self.watch_button.setChecked(False)
        self._status("Stopped watching.")

    def poll_watch(self) -> list[Path]:
        added = self.view_model.poll_watch()
        if not added:
            return added
        self._append_list_items(added)
        # Follow the newest frame, the usual need during an in-situ run.
        self.file_list.setCurrentRow(len(self.view_model.state.files) - 1)
        destination = self.view_model.default_export_dir()
        if self.auto_export_check.isChecked() and destination is not None:
            # With frame summing only complete groups of N new frames are exported.
            for request in self.view_model.watch_requests(added):
                self.tasks.submit(
                    f"export:{request.path}:{request.frame_index}",
                    lambda request=request: self.view_model.analyze_and_export(request, destination),
                    on_error=lambda message, _details, name=request.path.name: self._status(
                        f"Auto-export of {name} failed: {message}", "error"
                    ),
                )
        return added

    def _append_list_items(self, paths) -> None:
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self.file_list.addItem(item)


__all__ = ["BatchWatchMixin"]
