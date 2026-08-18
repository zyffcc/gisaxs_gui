"""Fit Graphics Events for fitting presentation."""

from __future__ import annotations

import os


from PyQt5.QtCore import Qt, QTimer, QEvent

from PyQt5.QtWidgets import (
    QMessageBox,
    QGraphicsScene,
)


from src.gimap.shared.file_paths import normalize_path


class FitGraphicsEventsMixin:
    """Own fit graphics events behavior."""

    def _expand_right_card(self, card_attr: str) -> None:
        try:
            card = getattr(self.ui, card_attr, None)
            if card is not None and hasattr(card, "set_expanded"):
                card.set_expanded(True)
        except Exception:
            pass

    def _setup_fit_graphics_scene(self):
        """itGraphicsView"""
        try:
            self._expand_right_card("fittingPlotCard")
            if not hasattr(self.ui, "fitGraphicsView"):
                return None

            if not hasattr(self, "_fit_graphics_scene") or self._fit_graphics_scene is None:
                self._fit_graphics_scene = QGraphicsScene()
                self.ui.fitGraphicsView.setScene(self._fit_graphics_scene)
                # Configure the view for a fixed-size, scroll-less canvas
                try:
                    from PyQt5.QtWidgets import QGraphicsView, QFrame

                    view = self.ui.fitGraphicsView
                    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    view.setDragMode(QGraphicsView.NoDrag)
                    view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
                    view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
                    view.setInteractive(False)
                    view.setFrameShape(QFrame.NoFrame)
                    from PyQt5.QtGui import QPainter

                    view.setRenderHint(QPainter.Antialiasing, False)
                    view.setRenderHint(QPainter.SmoothPixmapTransform, True)
                    view.setRenderHint(QPainter.TextAntialiasing, True)
                except Exception:
                    pass
            else:
                self._fit_graphics_scene.clear()

            return self._fit_graphics_scene

        except Exception as e:
            self.status_updated.emit(f"Failed to setup fit graphics scene: {str(e)}")
            return None

    def eventFilter(self, watched, event):
        """Refit preview canvases after users resize their splitter regions."""
        try:
            preview_view = getattr(self.ui, "gisaxsInputGraphicsView", None)
            preview_targets = (
                preview_view,
                preview_view.viewport() if preview_view is not None else None,
            )
            if watched in preview_targets and event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if self._detector_path_from_drop_event(event):
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
            if watched in preview_targets and event.type() == QEvent.Drop:
                file_path = self._detector_path_from_drop_event(event)
                if file_path:
                    event.acceptProposedAction()
                    self._apply_imported_gisaxs_file(file_path, show_image=True)
                    return True
                event.ignore()
                return True
            if event.type() == QEvent.Resize and watched in (
                preview_view,
                getattr(self.ui, "fitGraphicsView", None),
            ):
                if not self._preview_resize_refit_pending:
                    self._preview_resize_refit_pending = True
                    QTimer.singleShot(0, self._refit_resized_preview_canvases)
        except Exception:
            pass
        return super().eventFilter(watched, event)

    @staticmethod
    def _detector_path_from_drop_event(event):
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return ""
        supported = {".cbf", ".nxs", ".tif", ".tiff"}
        for url in mime_data.urls():
            file_path = normalize_path(url.toLocalFile())
            if file_path and os.path.splitext(file_path)[1].lower() in supported:
                return file_path
        return ""

    def _refit_resized_preview_canvases(self):
        self._preview_resize_refit_pending = False
        for view, item in (
            (
                getattr(self.ui, "gisaxsInputGraphicsView", None),
                getattr(self, "_preview_proxy_widget", None),
            ),
            (
                getattr(self.ui, "fitGraphicsView", None),
                self._current_fit_proxy_item(),
            ),
        ):
            if view is not None and item is not None:
                self._fit_view_to_item(view, item, keep_aspect=True)

    def _current_fit_proxy_item(self):
        scene = getattr(self, "_fit_graphics_scene", None)
        if scene is None:
            return None
        try:
            items = scene.items()
            return items[0] if items else None
        except Exception:
            return None

    def _fit_view_to_item(self, graphics_view, item, keep_aspect=True):
        """Fit the view to the given item bounds; disable scrollbars by sizing the scene to the item."""
        try:
            scene = graphics_view.scene()
            if scene is None or item is None:
                return
            scene.setSceneRect(item.sceneBoundingRect())
            # Always discard the transform inherited from the previous image.
            # Otherwise a large canvas followed by a smaller one can retain a
            # stale scale and leave the new preview tiny inside the viewport.
            graphics_view.resetTransform()
            if keep_aspect:
                graphics_view.fitInView(item, Qt.KeepAspectRatio)
            else:
                graphics_view.fitInView(item)
            graphics_view.update()
        except Exception:
            pass

    def _clear_fit_graphics_view(self):
        """fitGraphicsView"""
        try:
            if not hasattr(self.ui, "fitGraphicsView"):
                return

            scene = self._setup_fit_graphics_scene()
            if scene is not None:
                scene.clear()

            self.status_updated.emit("Fit graphics view cleared")

        except Exception as e:
            self.status_updated.emit(f"Failed to clear fit graphics view: {str(e)}")

    def _start_fitting(self):
        """No description."""
        if not self.current_parameters.get("imported_gisaxs_file"):
            QMessageBox.warning(
                self.parent, "Warning", "Please import a GISAXS file before processing."
            )
            return

        try:
            self.status_updated.emit("Start Cut Fitting Processing...")
            self.progress_updated.emit(0)

            self._run_fitting_process()

            self.progress_updated.emit(100)
            self.status_updated.emit("Cut Fitting processing complete!")

        except Exception as e:
            QMessageBox.critical(self.parent, "Cut Fitting Error", f"Cut fitting failed:\n{str(e)}")

    def _run_fitting_process(self):
        """No description."""
        pass

    def _reset_fitting(self):
        """No description."""
        self._set_default_parameters()
