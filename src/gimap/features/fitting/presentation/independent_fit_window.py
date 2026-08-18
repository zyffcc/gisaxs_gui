"""Independent Fit Window primitives for fitting presentation."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import pyqtSignal, Qt

from PyQt5.QtWidgets import (
    QMainWindow,
)


from src.gimap.app.presentation.responsive_layout import (
    apply_density_profile,
    install_adaptive_window_profile,
)


from .views import IndependentFitWindowView


from src.gimap.app.presentation.responsive_layout import (
    apply_density_profile,
    install_adaptive_window_profile,
)


from .views import IndependentFitWindowView


from .scientific_commands import (
    is_matplotlib_available,
)
from .curve_plotting import plot_cut_data_with_log_handling


class IndependentFitWindow(QMainWindow, IndependentFitWindowView):
    """Independent Matplotlib window for cut/fitting results."""

    status_updated = pyqtSignal(str)
    display_unit_changed = pyqtSignal(str)
    input_point_delete_requested = pyqtSignal(float)

    # 函数说明：初始化对象状态和相关资源。
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax = None
        self._delete_input_points_enabled = False
        self._delete_raw_q = np.array([], dtype=float)
        self._delete_plot_x = np.array([], dtype=float)
        self._delete_plot_y = np.array([], dtype=float)
        try:
            if is_matplotlib_available():
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import (
                    NavigationToolbar2QT as NavigationToolbar,
                )

                self.figure = Figure(figsize=(10, 6), dpi=100)
                self.canvas = FigureCanvas(self.figure)
                self.toolbar = NavigationToolbar(self.canvas, self)
        except Exception:
            pass

        self._bind_control_widgets()

        if self.toolbar is not None:
            self.toolbarHostLayout.addWidget(self.toolbar)
        if self.canvas is not None:
            self.canvasHostLayout.addWidget(self.canvas)

        if self.figure is not None:
            self.ax = self.figure.add_subplot(111)

        self.setFocusPolicy(Qt.StrongFocus)
        if self.canvas is not None:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)

        if self.figure is not None and self.canvas is not None and self.ax is not None:
            self._setup_empty_plot()
        install_adaptive_window_profile(
            self, self._apply_screen_profile, apply_window_minimum=False
        )

    # 函数说明：应用screen profile。
    def _apply_screen_profile(self, profile, screen):
        apply_density_profile(self, profile)

    # 函数说明：配置empty 图表。
    def _setup_empty_plot(self):
        """No description."""
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            "Perform a cut operation to see results here.\nDouble-click the Fitting Plot to open a larger window.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=12,
            alpha=0.7,
        )
        self.ax.set_xlabel("Position")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("GIMaP Cut Analysis Results")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    # 函数说明：创建control buttons。
    def _bind_control_widgets(self):
        """Connect behavior to the fixed controls owned by the Python View."""
        self.show_positive_cb.toggled.connect(self._on_show_positive_toggled)
        self.show_negative_cb.toggled.connect(self._on_show_negative_toggled)
        self.q_unit_combo.setItemData(0, "angstrom")
        self.q_unit_combo.setItemData(1, "nm")
        self.q_unit_combo.setCurrentIndex(1)
        self.q_unit_combo.currentTextChanged.connect(self._on_q_unit_changed)
        self.y_range_combo.setItemData(0, "experimental")
        self.y_range_combo.setItemData(1, "fitting")
        self.y_range_combo.setItemData(2, "all")
        self.y_range_combo.setCurrentIndex(2)
        self.y_range_combo.currentTextChanged.connect(self._on_y_range_changed)
        self.delete_input_points_cb.toggled.connect(self._on_delete_input_points_toggled)

    # 函数说明：处理正值 toggled事件。
    def _on_show_positive_toggled(self, checked):
        """No description."""
        if checked and hasattr(self, "show_negative_cb") and self.show_negative_cb.isChecked():
            self.show_negative_cb.blockSignals(True)
            self.show_negative_cb.setChecked(False)
            self.show_negative_cb.blockSignals(False)
        self.status_updated.emit(f"Positive Only mode: {'enabled' if checked else 'disabled'}")

    # 函数说明：处理负值 toggled事件。
    def _on_show_negative_toggled(self, checked):
        """No description."""
        if checked and hasattr(self, "show_positive_cb") and self.show_positive_cb.isChecked():
            self.show_positive_cb.blockSignals(True)
            self.show_positive_cb.setChecked(False)
            self.show_positive_cb.blockSignals(False)
        self.status_updated.emit(f"Negative Only mode: {'enabled' if checked else 'disabled'}")

    # 函数说明：获取Q unit key。
    def _get_q_unit_key(self):
        """No description."""
        try:
            if hasattr(self, "q_unit_combo"):
                unit = self.q_unit_combo.currentData()
                if isinstance(unit, str) and unit.lower() in ("angstrom", "nm"):
                    return unit.lower()
        except Exception:
            pass
        return "nm"

    # 函数说明：获取Q unit 缩放 factor。
    def _get_q_unit_scale_factor(self):
        """No description."""
        return 0.1 if self._get_q_unit_key() == "angstrom" else 1.0

    # 函数说明：实现 format Q 坐标轴 label 相关逻辑。
    def _format_q_axis_label(self, filter_mode="all", absolute=False):
        """No description."""
        unit_text = "nm$^{-1}$" if self._get_q_unit_key() == "nm" else r"$\AA^{-1}$"
        base = "|q|" if absolute or filter_mode == "negative" else "q"
        suffix = ""
        if filter_mode == "positive":
            suffix = " [Positive Only]"
        elif filter_mode == "negative":
            suffix = " [Negative Only]"
        return f"{base} ({unit_text}){suffix}"

    # 函数说明：处理Q unit changed事件。
    def _on_q_unit_changed(self, _text):
        """No description."""
        unit_text = "nm^-1" if self._get_q_unit_key() == "nm" else "Angstrom^-1"
        self.status_updated.emit(f"q unit changed to {unit_text}")
        self.display_unit_changed.emit(unit_text)

    # 函数说明：处理y 范围 changed事件。
    def _on_y_range_changed(self, _text):
        """No description."""
        mode = self._get_y_range_mode()
        label = {
            "experimental": "experimental data",
            "fitting": "fitting data",
            "all": "all visible data",
        }.get(mode, "all visible data")
        self.status_updated.emit(f"Y range based on {label}")

    # 函数说明：获取y 范围 模式。
    def _get_y_range_mode(self):
        """No description."""
        try:
            if hasattr(self, "y_range_combo"):
                mode = self.y_range_combo.currentData()
                if mode in ("experimental", "fitting", "all"):
                    return mode
        except Exception:
            pass
        return "all"

    # 函数说明：处理input points toggled事件。
    def _on_delete_input_points_toggled(self, checked):
        """Enable point deletion mode for AI fitting input outliers."""
        self._delete_input_points_enabled = bool(checked)
        if self.canvas is not None:
            self.canvas.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)
        self.status_updated.emit(
            "Delete Points mode enabled: left-click a data point to exclude it."
            if checked
            else "Delete Points mode disabled."
        )

    # 函数说明：设置deletable points。
    def set_deletable_points(self, raw_q, plot_x, plot_y):
        """Register visible points that can be clicked to exclude from AI fitting input."""
        try:
            raw_q = np.asarray(raw_q, dtype=float).reshape(-1)
            plot_x = np.asarray(plot_x, dtype=float).reshape(-1)
            plot_y = np.asarray(plot_y, dtype=float).reshape(-1)
            n = min(raw_q.size, plot_x.size, plot_y.size)
            if n <= 0:
                self.clear_deletable_points()
                return
            raw_q, plot_x, plot_y = raw_q[:n], plot_x[:n], plot_y[:n]
            mask = np.isfinite(raw_q) & np.isfinite(plot_x) & np.isfinite(plot_y)
            self._delete_raw_q = raw_q[mask]
            self._delete_plot_x = plot_x[mask]
            self._delete_plot_y = plot_y[mask]
        except Exception:
            self.clear_deletable_points()

    # 函数说明：清除deletable points。
    def clear_deletable_points(self):
        self._delete_raw_q = np.array([], dtype=float)
        self._delete_plot_x = np.array([], dtype=float)
        self._delete_plot_y = np.array([], dtype=float)

    # 函数说明：处理画布 按钮 press事件。
    def _on_canvas_button_press(self, event):
        """Delete the nearest registered data point when delete mode is active."""
        try:
            if not getattr(self, "_delete_input_points_enabled", False):
                return
            if event.button != 1 or event.inaxes is None or self.ax is None:
                return
            if self.toolbar is not None and getattr(self.toolbar, "mode", ""):
                return
            if self._delete_raw_q.size == 0:
                self.status_updated.emit("No deletable input points are registered for this plot.")
                return

            points = np.column_stack([self._delete_plot_x, self._delete_plot_y])
            pixel_points = self.ax.transData.transform(points)
            click = np.array([event.x, event.y], dtype=float)
            distances = np.hypot(pixel_points[:, 0] - click[0], pixel_points[:, 1] - click[1])
            if distances.size == 0:
                return
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) > 16.0:
                self.status_updated.emit("Click closer to a data point to delete it.")
                return
            q_value = float(self._delete_raw_q[nearest])
            self.input_point_delete_requested.emit(q_value)
        except Exception as exc:
            self.status_updated.emit(f"Failed to delete input point: {exc}")

    # 函数说明：更新plot。
    def update_plot(
        self,
        x_coords,
        y_intensity,
        x_label,
        y_label,
        title,
        log_x=False,
        log_y=False,
        normalize=False,
        y_errors=None,
    ):
        """No description."""
        try:
            if self.ax is None or self.canvas is None or self.figure is None:
                self.status_updated.emit("Independent fit window is not ready for plotting.")
                return
            x_data = np.asarray(x_coords, dtype=float).reshape(-1)
            y_data = np.asarray(y_intensity, dtype=float).reshape(-1)
            n = min(x_data.size, y_data.size)
            if n <= 0:
                self.status_updated.emit("No finite data available for independent fit window.")
                return
            x_data = x_data[:n]
            y_data = y_data[:n]

            err_data = None
            if y_errors is not None:
                err_data = np.asarray(y_errors, dtype=float).reshape(-1)[:n]

            finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if err_data is not None:
                finite_mask &= np.isfinite(err_data)
            x_data = x_data[finite_mask]
            y_data = y_data[finite_mask]
            if err_data is not None:
                err_data = err_data[finite_mask]
            if x_data.size == 0 or y_data.size == 0:
                self.status_updated.emit("No finite data available for independent fit window.")
                return

            is_q_axis = isinstance(x_label, str) and "q" in x_label.lower()

            if hasattr(self, "show_positive_cb") and self.show_positive_cb.isChecked():
                mask = x_data > 0
                x_data = x_data[mask]
                y_data = y_data[mask]
                if err_data is not None:
                    err_data = err_data[mask]
            elif hasattr(self, "show_negative_cb") and self.show_negative_cb.isChecked():
                mask = x_data < 0
                x_data = np.abs(x_data[mask])
                y_data = y_data[mask]
                if err_data is not None:
                    err_data = err_data[mask]
                if x_data.size > 0:
                    sort_idx = np.argsort(x_data)
                    x_data = x_data[sort_idx]
                    y_data = y_data[sort_idx]
                    if err_data is not None:
                        err_data = err_data[sort_idx]

            if is_q_axis:
                x_data = x_data * self._get_q_unit_scale_factor()
                if hasattr(self, "show_positive_cb") and self.show_positive_cb.isChecked():
                    x_label = self._format_q_axis_label(filter_mode="positive")
                elif hasattr(self, "show_negative_cb") and self.show_negative_cb.isChecked():
                    x_label = self._format_q_axis_label(filter_mode="negative")
                else:
                    original_x = np.asarray(x_coords)
                    has_negative = np.any(np.isfinite(original_x) & (original_x < 0))
                    x_label = self._format_q_axis_label(absolute=(log_x and has_negative))

            if normalize:
                max_intensity = np.max(y_data) if y_data.size > 0 else 0.0
                if max_intensity > 0:
                    y_data = y_data / float(max_intensity)
                    if err_data is not None:
                        err_data = err_data / float(max_intensity)
                    y_label = "Normalized Intensity"

            if log_x:
                positive_x = x_data > 0
                if not np.any(positive_x):
                    log_x = False
                else:
                    x_data = x_data[positive_x]
                    y_data = y_data[positive_x]
                    if err_data is not None:
                        err_data = err_data[positive_x]

            if log_y:
                positive_y = y_data > 0
                if not np.any(positive_y):
                    log_y = False
                else:
                    x_data = x_data[positive_y]
                    y_data = y_data[positive_y]
                    if err_data is not None:
                        err_data = err_data[positive_y]

            if x_data.size == 0 or y_data.size == 0:
                self.status_updated.emit("No plottable data available for independent fit window.")
                return

            self.ax.clear()

            if err_data is not None:
                self.ax.errorbar(
                    x_data,
                    y_data,
                    yerr=err_data,
                    fmt="o-",
                    markersize=4,
                    linewidth=1.5,
                    capsize=3,
                    alpha=0.8,
                    label="Data with error bars",
                )
            else:
                try:
                    plot_cut_data_with_log_handling(
                        self.ax, x_data, y_data, log_x, markersize=6, linewidth=2
                    )
                except:
                    self.ax.plot(
                        x_data, y_data, "o-", markersize=4, linewidth=1.5, alpha=0.8, label="Data"
                    )

            try:
                self.set_deletable_points(
                    x_data / self._get_q_unit_scale_factor() if is_q_axis else x_data,
                    x_data,
                    y_data,
                )
            except Exception:
                self.clear_deletable_points()

            try:
                # 函数说明：实现 to mathtext 相关逻辑。
                def _to_mathtext(label: str) -> str:
                    if not isinstance(label, str):
                        return label
                    return (
                        label.replace("A^-1", r"$\AA^{-1}$")
                        .replace("Ang^-1", r"$\AA^{-1}$")
                        .replace("(A^-1)", r"($\AA^{-1}$)")
                        .replace("(Ang^-1)", r"($\AA^{-1}$)")
                    )

                x_lbl = _to_mathtext(x_label)
                y_lbl = _to_mathtext(y_label)
            except Exception:
                x_lbl, y_lbl = x_label, y_label

            self.ax.set_xlabel(x_lbl, fontsize=13)
            self.ax.set_ylabel(y_lbl, fontsize=13)
            self.ax.set_title(title, fontsize=15)

            if log_x:
                self.ax.set_xscale("log")
            else:
                self.ax.set_xscale("linear")

            if log_y:
                self.ax.set_yscale("log")
            else:
                self.ax.set_yscale("linear")

            self.ax.grid(True, alpha=0.4, linestyle="--")
            try:
                for axis in ["top", "bottom", "left", "right"]:
                    self.ax.spines[axis].set_linewidth(1.8)
                self.ax.tick_params(axis="both", which="both", width=1.6, labelsize=12)
            except Exception:
                pass

            # stats_text = f'Points: {len(x_data)}\nMax: {np.max(y_data):.2e}\nMin: {np.min(y_data):.2e}'
            # self.ax.text(0.02, 0.88, stats_text, transform=self.ax.transAxes,
            #             verticalalignment='bottom', fontsize=10,
            #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.figure.tight_layout()

            self.canvas.draw_idle()

            self.setWindowTitle(f"GIMaP Cut Analysis - {title}")

            self.status_updated.emit(f"Independent fit window updated: {title}")

        except Exception as e:
            self.status_updated.emit(f"Failed to update independent fit window: {str(e)}")

    # 函数说明：处理窗口关闭事件。
    def closeEvent(self, event):
        """No description."""
        try:
            if self.figure is not None:
                self.figure.clear()
            self.ax = None
            self.canvas = None
            self.toolbar = None
        except Exception:
            pass
        super().closeEvent(event)
