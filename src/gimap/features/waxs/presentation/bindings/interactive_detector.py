"""WAXS adapter for the application-owned pixel inspection window."""

from types import SimpleNamespace

from PyQt5.QtWidgets import QMessageBox
from src.gimap.app.presentation import ScientificImageViewer


class InteractiveDetectorMixin:
    def _open_interactive_detector(self):
        window = getattr(self.viewer, "interactive_window", None)
        if window is None:
            try:
                window = ScientificImageViewer(self)
            except ImportError:
                QMessageBox.information(
                    self, "Interactive detector", "Install pyqtgraph to enable this viewer."
                )
                return
            self.viewer.interactive_window = window
            window.log_changed.connect(self.display_log.setChecked)
            window.levels_changed.connect(self._interactive_levels_changed)
            window.region_selected.connect(self._interactive_region_selected)
            window.frame_requested.connect(self._interactive_frame_requested)
        window.show()
        window.raise_()
        if self._active_view == "2d":
            self.refresh_view()
        else:
            window.set_unavailable("Select the 2D detector tab to inspect pixels.")

    def _interactive_levels_changed(self, low, high):
        self._set_values_without_refresh(((self.vmin_spin, low), (self.vmax_spin, high)))
        self.display_auto_scale.blockSignals(True)
        self.display_auto_scale.setChecked(False)
        self.display_auto_scale.blockSignals(False)
        self.refresh_view()

    def _interactive_region_selected(self, bounds):
        if self.coordinate_mode_combo.currentText() == "q space":
            return
        x0, x1, y0, y1 = bounds
        self._on_roi_selected(
            SimpleNamespace(xdata=x0, ydata=y0), SimpleNamespace(xdata=x1, ydata=y1)
        )

    def _interactive_frame_requested(self, index):
        if self._loader_thread is not None and self._loader_thread.isRunning():
            # A frame may arrive before its loader thread has finished shutting down.
            # Acknowledge the current position so a rejected request can retry next tick.
            window = getattr(self.viewer, "interactive_window", None)
            if window is not None:
                window.set_frame_position(self.frame_spin.value() - 1, self.current_frame_count)
            return
        self._start_loader(self.current_file, index)

    def _sync_interactive_detector_overlays(self):
        window = getattr(self.viewer, "interactive_window", None)
        if window is None or not window.isVisible():
            return
        window.set_frame_position(self.frame_spin.value() - 1, self.current_frame_count)
        center = None
        paths = []
        if self.show_center_check.isChecked():
            center = (self.center_x_spin.value(), self.center_y_spin.value())
        if self.show_cut_region_check.isChecked():
            for artist in self.viewer._overlay_artists:
                if hasattr(artist, "get_patch_transform"):
                    paths.extend(artist.get_path().to_polygons(artist.get_patch_transform()))
        window.set_overlay(center=center, paths=paths)
