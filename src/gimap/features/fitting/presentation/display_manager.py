"""Display Manager primitives for fitting presentation."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import QTimer


from .scientific_commands import (
    is_matplotlib_available,
)
from .curve_plotting import plot_cut_data_with_log_handling


class UnifiedDisplayManager:
    """Manage unified plot display updates."""

    # 函数说明：初始化对象状态和相关资源。
    def __init__(self, controller):
        self.controller = controller
        self.ui = controller.ui

    # 函数说明：绘制1d 数据。
    def plot_1d_data(
        self,
        q,
        intensity,
        err=None,
        title="",
        source_info="",
        log_x=False,
        log_y=False,
        normalize=False,
    ):
        """Plot 1D data."""
        try:
            plot_q = np.array(q)
            plot_I = np.array(intensity)
            plot_err = np.array(err) if err is not None else None

            if normalize and len(plot_I) > 0:
                max_I = np.max(plot_I)
                if max_I > 0:
                    plot_I = plot_I / max_I
                    if plot_err is not None:
                        plot_err = plot_err / max_I

            if log_y and len(plot_I) > 0 and not np.all(plot_I > 0):
                min_positive = np.min(plot_I[plot_I > 0]) if np.any(plot_I > 0) else 1e-10
                plot_I = np.where(plot_I <= 0, min_positive, plot_I)
                if plot_err is not None:
                    plot_err = np.where(plot_I <= min_positive, min_positive * 0.1, plot_err)

            self._update_gui_1d_display(plot_q, plot_I, plot_err, title, log_x, log_y, normalize)

            self._update_independent_1d_display(
                plot_q, plot_I, plot_err, title, log_x, log_y, normalize
            )

            y_label = "Intensity" + (" (normalized)" if normalize else "")
            mode_str = f"Log-X: {log_x}, Log-Y: {log_y}, Norm: {normalize}"
            self.controller.status_updated.emit(f"1D data displayed: {title} [{mode_str}]")

        except Exception as e:
            self.controller.status_updated.emit(f"Failed to plot 1D data: {str(e)}")

    # 函数说明：更新界面 1d 显示。
    def _update_gui_1d_display(self, q, intensity, err, title, log_x, log_y, normalize):
        """Render the 1D fitting data in the GUI."""
        try:
            if not hasattr(self.ui, "fitGraphicsView") or not is_matplotlib_available():
                return

            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            figure = Figure(figsize=(8, 6))
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111)

            self._unified_plot_1d_data(ax, q, intensity, err, title, log_x, log_y, normalize)

            figure.tight_layout()

            canvas.draw()

            scene = self.controller._setup_fit_graphics_scene()
            if scene is not None:
                proxy_widget = scene.addWidget(canvas)
                self.controller._fit_view_to_item(
                    self.ui.fitGraphicsView, proxy_widget, keep_aspect=True
                )

                self.controller._current_fit_canvas = canvas
                self.controller._current_fit_figure = figure

                try:
                    if log_x and hasattr(self.controller, "_adjust_roi_bounds_for_log_x"):
                        QTimer.singleShot(0, self.controller._adjust_roi_bounds_for_log_x)
                except Exception:
                    pass

        except Exception as e:
            self.controller.status_updated.emit(f"Failed to update GUI 1D display: {str(e)}")

    # 函数说明：实现 unified 图表 1d 数据 相关逻辑。
    def _unified_plot_1d_data(self, ax, q, intensity, err, title, log_x, log_y, normalize):
        """No description."""
        try:
            q_plot = self.controller._convert_q_values_for_display(q)

            if err is not None:
                ax.errorbar(
                    q_plot,
                    intensity,
                    yerr=err,
                    fmt="o-",
                    markersize=3,
                    linewidth=1,
                    capsize=2,
                    alpha=0.8,
                    label="Data with error bars",
                )
            else:
                plot_cut_data_with_log_handling(
                    ax, q_plot, intensity, log_x, markersize=3, linewidth=1
                )

            has_negative = np.any(np.isfinite(np.asarray(q)) & (np.asarray(q) < 0))
            ax.set_xlabel(
                self.controller._build_q_axis_label(absolute=(log_x and has_negative)), fontsize=13
            )
            ax.set_ylabel("Intensity" + (" (normalized)" if normalize else ""), fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

            if log_x:
                ax.set_xscale("log")
            else:
                ax.set_xscale("linear")

            if log_y:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")

            try:
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(1.8)
                ax.tick_params(axis="both", which="both", width=1.6, labelsize=12)
            except Exception:
                pass

        except Exception as e:
            q_plot = self.controller._convert_q_values_for_display(q)
            if err is not None:
                ax.errorbar(
                    q_plot, intensity, yerr=err, fmt="o-", markersize=3, linewidth=1, capsize=2
                )
            else:
                ax.plot(q_plot, intensity, "o-", markersize=3, linewidth=1)
            has_negative = np.any(np.isfinite(np.asarray(q)) & (np.asarray(q) < 0))
            ax.set_xlabel(
                self.controller._build_q_axis_label(absolute=(log_x and has_negative)), fontsize=13
            )
            ax.set_ylabel("Intensity" + (" (normalized)" if normalize else ""), fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

    # 函数说明：更新独立 1d 显示。
    def _update_independent_1d_display(self, q, intensity, err, title, log_x, log_y, normalize):
        """D"""
        try:
            if self.controller.independent_fit_window and hasattr(
                self.controller.independent_fit_window, "update_plot"
            ):
                y_label = "Intensity" + (" (normalized)" if normalize else "")
                q_internal_nm = self.controller._convert_q_values_for_model(
                    q, source=getattr(self.controller, "data_source", None)
                )
                self.controller.independent_fit_window.update_plot(
                    q_internal_nm,
                    intensity,
                    self.controller._build_q_axis_label(),
                    y_label,
                    title,
                    log_x,
                    log_y,
                    normalize,
                    err,
                )

        except Exception as e:
            self.controller.status_updated.emit(
                f"Failed to update independent 1D display: {str(e)}"
            )

    # 函数说明：获取显示 options。
    def get_display_options(self):
        """No description."""
        return {
            "log_x": hasattr(self.ui, "fitLogXCheckBox") and self.ui.fitLogXCheckBox.isChecked(),
            "log_y": hasattr(self.ui, "fitLogYCheckBox") and self.ui.fitLogYCheckBox.isChecked(),
            "normalize": hasattr(self.ui, "fitNormCheckBox")
            and self.ui.fitNormCheckBox.isChecked(),
        }


def _qobject_is_alive(obj) -> bool:
    if obj is None:
        return False
    try:
        import sip

        if sip.isdeleted(obj):
            return False
    except Exception:
        pass
    try:
        obj.objectName()
    except RuntimeError:
        return False
    except Exception:
        pass
    return True
