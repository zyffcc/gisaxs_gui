"""The fitting curve canvas: scene set-up, resize refit and the independent fit window."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtWidgets import QGraphicsScene, QMessageBox

from src.gimap.app.presentation.layout_metrics import move_window_to_cursor_screen
from src.gimap.app.presentation.components import MplBoxZoom

from ..binding_primitives import (
    IndependentFitWindow,
    _qobject_is_alive,
    is_matplotlib_available,
)


class FitGraphicsEventsMixin:
    """Own the embedded fitting canvas and its larger independent window."""

    def _expand_right_card(self, card_attr: str) -> None:
        try:
            card = getattr(self.ui, card_attr, None)
            if card is not None and hasattr(card, "set_expanded"):
                card.set_expanded(True)
        except Exception:
            pass

    def _active_curve_graphics_view(self):
        return getattr(self.ui, "fitGraphicsView", None)

    def _setup_fit_graphics_scene(self):
        """itGraphicsView"""
        try:
            self._expand_right_card("fittingPlotCard")
            view = self._active_curve_graphics_view()
            if view is None:
                return None
            scene = getattr(self, "_curve_graphics_scene", None)
            if scene is None:
                scene = QGraphicsScene()
                self._curve_graphics_scene = scene
                view.setScene(scene)
                # Configure the view for a fixed-size, scroll-less canvas
                try:
                    from PyQt5.QtWidgets import QGraphicsView, QFrame

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

            return scene

        except Exception as e:
            self.status_updated.emit(f"Failed to setup fit graphics scene: {str(e)}")
            return None

    def _ensure_curve_canvas(self):
        """Create the scene-owned canvas once; explicit clear invalidates all references."""
        scene = self._setup_fit_graphics_scene()
        if scene is None:
            return None
        canvas = getattr(self, "_current_fit_canvas", None)
        proxy = getattr(self, "_curve_canvas_proxy", None)
        if canvas is None or proxy is None:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            scene.clear()
            figure = Figure(figsize=(8.0, 5.0), dpi=90)
            canvas = FigureCanvasQTAgg(figure)
            figure.add_subplot(111)
            proxy = scene.addWidget(canvas)
            self._current_fit_figure = figure
            self._current_fit_canvas = canvas
            self._curve_zoom = MplBoxZoom(canvas)  # drag a rectangle to zoom, double-click to reset
            self._curve_canvas_proxy = proxy
        figure = self._current_fit_figure
        return figure, canvas, figure.axes[0], proxy

    def eventFilter(self, watched, event):
        """Refit the curve canvas after the user resizes its splitter region."""
        try:
            if event.type() == QEvent.Resize and watched is getattr(self.ui, "fitGraphicsView", None):
                if not self._preview_resize_refit_pending:
                    self._preview_resize_refit_pending = True
                    QTimer.singleShot(0, self._refit_resized_preview_canvases)
        except Exception:
            pass
        return super().eventFilter(watched, event)

    def _refit_resized_preview_canvases(self):
        self._preview_resize_refit_pending = False
        view = getattr(self.ui, "fitGraphicsView", None)
        item = self._current_curve_proxy_item()
        if view is not None and item is not None:
            self._fit_view_to_item(view, item, keep_aspect=True)

    def _current_curve_proxy_item(self):
        scene = getattr(self, "_curve_graphics_scene", None)
        if scene is None:
            return None
        try:
            items = scene.items()
            return items[0] if items else None
        except Exception:
            return None

    def _current_fit_proxy_item(self):
        return self._current_curve_proxy_item()

    def _fit_view_to_item(self, graphics_view, item, keep_aspect=True):
        """Size the plot canvas to the view, 1:1: the plot fills it and text keeps its size.

        Scaling a fixed-size canvas into the view (``fitInView``) left a small plot
        with tiny labels; resizing the canvas lets Matplotlib lay out the real size.
        A view too small to hold a readable plot still gets the scaled canvas.
        """
        try:
            scene = graphics_view.scene()
            if scene is None or item is None:
                return
            # Always discard the transform inherited from the previous canvas.
            graphics_view.resetTransform()
            canvas = item.widget() if hasattr(item, "widget") else None
            viewport = graphics_view.viewport().size()
            if canvas is not None and viewport.width() >= 200 and viewport.height() >= 150:
                if canvas.size() != viewport:
                    canvas.resize(viewport)
                    figure = getattr(canvas, "figure", None)
                    if figure is not None:
                        try:
                            figure.tight_layout()
                        except Exception:
                            pass
                    canvas.draw_idle()
                scene.setSceneRect(item.sceneBoundingRect())
            else:
                scene.setSceneRect(item.sceneBoundingRect())
                graphics_view.fitInView(item, Qt.KeepAspectRatio if keep_aspect else Qt.IgnoreAspectRatio)
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
                self._current_fit_figure = None
                self._current_fit_canvas = None
                self._curve_canvas_proxy = None

            self.status_updated.emit("Fit graphics view cleared")

        except Exception as e:
            self.status_updated.emit(f"Failed to clear fit graphics view: {str(e)}")

    def _reset_fitting(self):
        """No description."""
        self._set_default_parameters()

    def _on_fit_graphics_view_double_click(self, event):
        """No description."""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib",
                )
                return

            if self.q is None or self.I is None:
                QMessageBox.information(
                    self.main_window, "No Data", "No data available for display."
                )
                return
            try:
                q_snapshot = np.asarray(self.q, dtype=float).reshape(-1)
                i_snapshot = np.asarray(self.I, dtype=float).reshape(-1)
                n_snapshot = min(q_snapshot.size, i_snapshot.size)
                if n_snapshot <= 0 or not np.any(
                    np.isfinite(q_snapshot[:n_snapshot]) & np.isfinite(i_snapshot[:n_snapshot])
                ):
                    QMessageBox.information(
                        self.main_window, "No Data", "No finite fitting plot data available."
                    )
                    return
            except Exception:
                QMessageBox.information(
                    self.main_window, "No Data", "Fitting plot data is not ready yet."
                )
                return

            if not _qobject_is_alive(self.independent_fit_window):
                self.independent_fit_window = None

            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                self.independent_fit_window.setAttribute(Qt.WA_DeleteOnClose, True)
                self.independent_fit_window.destroyed.connect(
                    lambda _obj=None: setattr(self, "independent_fit_window", None)
                )
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
                self.independent_fit_window.view_state_changed.connect(
                    self._on_independent_curve_view_state_changed
                )
                if hasattr(self.independent_fit_window, "input_point_delete_requested"):
                    self.independent_fit_window.input_point_delete_requested.connect(
                        self._exclude_ai_input_point_from_plot
                    )
                try:
                    self.independent_fit_window.set_curve_view_state(
                        self._current_curve_view_state(sync_window=False)
                    )
                except Exception:
                    pass

                move_window_to_cursor_screen(self.independent_fit_window)
                self.independent_fit_window.show()
                self.independent_fit_window.raise_()
                self.independent_fit_window.activateWindow()

            mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            try:
                if (
                    hasattr(self, "_is_in_fitting_mode")
                    and callable(self._is_in_fitting_mode)
                    and self._is_in_fitting_mode()
                ):
                    mode = "fitting"
            except Exception:
                pass
            try:
                has_fit = bool(
                    getattr(self, "has_fitting_data", False)
                    and getattr(self, "I_fitting", None) is not None
                )
                if mode == "fitting" and not has_fit:
                    mode = "normal"
            except Exception:
                pass

            if mode == "fitting":
                try:
                    self._update_gui_fitting_display()
                except Exception:
                    pass
                self._update_outside_window("fitting")
            else:
                self._update_outside_window(mode)

            if hasattr(self.independent_fit_window, "canvas"):
                self.independent_fit_window.canvas.setFocus()
                self.independent_fit_window.canvas.draw_idle()

            self.status_updated.emit(f"{mode.capitalize()} mode independent window updated")

        except Exception as e:
            self.status_updated.emit(f"Fit double-click error: {str(e)}")


__all__ = ["FitGraphicsEventsMixin"]
