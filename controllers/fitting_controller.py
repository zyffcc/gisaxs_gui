"""
Cut Fitting ?????- ???GISAXS?????????????
"""

import os
import json
import re
import time
import datetime
import copy
import sys
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QTimer, QPoint, QProcess, QUrl
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QGraphicsScene,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMainWindow,
    QMenu,
    QAction,
    QTextBrowser,
    QSizePolicy,
    QDialog,
    QInputDialog,
    QComboBox,
    QStackedWidget,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

# ?????????????
from ui.detector_parameters_dialog import DetectorParametersDialog
from ui.responsive_layout import (
    apply_density_profile,
    install_adaptive_window_profile,
    move_window_to_cursor_screen,
)

# ??????????????
from config.model_parameters_manager import ModelParametersManager

# ?????????????????
from utils.universal_parameter_trigger_manager import UniversalParameterTriggerManager
from utils.path_utils import normalize_path
from utils.ai_fitting_models import (
    ModelInfo,
    default_ai_fitting_model_base_dirs,
    discover_ai_fitting_models,
    discover_model_in_path,
)
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence, QDesktopServices

"""Heavy libraries (matplotlib/fabio) are lazy-loaded to speed up startup."""
# Lazy availability flags (None means unchecked yet)
MATPLOTLIB_AVAILABLE = None
FABIO_AVAILABLE = None


COMPONENT_FORMULA_TOOLTIPS = {
    "None": (
        "Component: None\n\n"
        "No component is used."
    ),
    "Sphere": (
        "Component: Sphere\n\n"
        "Formula:\n"
        "F(q,R) = 3 * [sin(qR) - qR cos(qR)] / (qR)^3\n"
        "I(q) = Int * <F(q,R)^2> * S(q; D, sigma_D)\n\n"
        "Parameters:\n"
        "R = radius in nm\n"
        "sigma_R = radius distribution width\n"
        "D = structure spacing in nm\n"
        "sigma_D = structure disorder"
    ),
    "Cylinder": (
        "Component: Cylinder\n\n"
        "Formula:\n"
        "F(q,R,h,alpha) = [2 J1(qR sin(alpha)) / (qR sin(alpha))] "
        "* sinc(qh cos(alpha)/2)\n"
        "I(q) = Int * <F(q,R,h,alpha)^2>_{R,h,alpha} * S(q; D, sigma_D)\n\n"
        "This is the existing isotropic/random-orientation cylinder."
    ),
    "Vertical Cylinder": (
        "Component: Vertical Cylinder\n\n"
        "Formula from gisaxs_fit_v3.1_4structures.py:\n"
        "I(q) = Int * <(R * J1(qR) / q)^2>_R * S(q; D, sigma_D)\n\n"
        "Parameters:\n"
        "R = cylinder radius in nm\n"
        "sigma_R = fractional radius distribution width\n"
        "D = structure spacing in nm\n"
        "sigma_D = structure disorder"
    ),
}


COMPONENT_PARAMETER_SCHEMAS = {
    "Sphere": [
        ("intensity", "Int", "Intensity", 1.0, 6, 0.1),
        ("radius", "R", "Radius (nm)", 10.0, 3, 0.1),
        ("sigma_radius", "SigmaR", "sigma Radius", 0.1, 4, 0.01),
        ("diameter", "D", "D spacing (nm)", 20.0, 3, 0.1),
        ("sigma_diameter", "SigmaD", "sigma D", 0.1, 4, 0.01),
    ],
    "Cylinder": [
        ("intensity", "Int", "Intensity", 1.0, 6, 0.1),
        ("radius", "R", "Radius (nm)", 10.0, 3, 0.1),
        ("sigma_radius", "SigmaR", "sigma Radius", 0.1, 4, 0.01),
        ("height", "h", "Height (nm)", 20.0, 3, 0.1),
        ("sigma_height", "Sigmah", "sigma Height", 0.1, 4, 0.01),
        ("diameter", "D", "D spacing (nm)", 20.0, 3, 0.1),
        ("sigma_diameter", "SigmaD", "sigma D", 0.1, 4, 0.01),
    ],
    "Vertical Cylinder": [
        ("intensity", "Int", "Intensity", 1.0, 6, 0.1),
        ("radius", "R", "Radius (nm)", 10.0, 3, 0.1),
        ("sigma_radius", "SigmaR", "sigma Radius / R", 0.3, 4, 0.01),
        ("diameter", "D", "D spacing (nm)", 20.0, 3, 0.1),
        ("sigma_diameter", "SigmaD", "sigma D", 0.1, 4, 0.01),
    ],
}

COMPONENT_ORDER = ("None", "Sphere", "Cylinder", "Vertical Cylinder")


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e) -> None:
        if e is not None:
            e.ignore()


class CurrentPageHeightStackedWidget(QStackedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.currentChanged.connect(lambda _index: QTimer.singleShot(0, self.sync_current_height))

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_current_height()

    def sync_current_height(self):
        current_widget = self.currentWidget()
        hint = current_widget.sizeHint() if current_widget is not None else super().sizeHint()
        height = max(1, hint.height())
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def sizeHint(self):
        current_widget = self.currentWidget()
        if current_widget is not None:
            return current_widget.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        current_widget = self.currentWidget()
        if current_widget is not None:
            return current_widget.minimumSizeHint()
        return super().minimumSizeHint()


def is_matplotlib_available():
    """Lazy-check matplotlib availability and cache the result.

    Returns True if matplotlib can be imported, else False. This avoids importing it at
    process start; the first call may pay the cost, which is acceptable when needed.
    """
    global MATPLOTLIB_AVAILABLE
    if MATPLOTLIB_AVAILABLE is None:
        try:
            # Import minimal submodules needed later; do not configure backend here
            import matplotlib  # noqa: F401
            MATPLOTLIB_AVAILABLE = True
        except Exception:
            MATPLOTLIB_AVAILABLE = False
    return MATPLOTLIB_AVAILABLE


def is_fabio_available():
    """Lazy-check fabio availability and cache the result."""
    global FABIO_AVAILABLE
    if FABIO_AVAILABLE is None:
        try:
            import fabio  # noqa: F401
            FABIO_AVAILABLE = True
        except Exception:
            FABIO_AVAILABLE = False
    return FABIO_AVAILABLE


try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ManualAutoRefineWorker(QObject):
    """Run manual Auto Refine outside the GUI thread."""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, controller, setup, selected, options):
        super().__init__()
        self.controller = controller
        self.setup = setup
        self.selected = selected
        self.options = options
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            result = self.controller._run_manual_auto_refine(
                self.setup,
                self.selected,
                self.options,
                progress_callback=self.progress.emit,
                stop_callback=lambda: self._stop_requested,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class IndependentMatplotlibWindow(QMainWindow):
    """No description."""

    DEFAULT_TITLE = "GIMaP Image Viewer - Independent Window (right-click to select)"
    SELECTION_TITLE = "GIMaP Image Viewer - Selection Mode (drag to select, Esc to exit)"

    # ???????????????????
    region_selected = pyqtSignal(dict)  # ?????????????
    status_updated = pyqtSignal(str)  # ?????????????

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.DEFAULT_TITLE)
        self.setGeometry(100, 100, 900, 700)

        # ?????widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # ??????
        layout = QVBoxLayout(central_widget)

        # ???matplotlib??????????????
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax = None
        try:
            if is_matplotlib_available():
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                self.figure = Figure(figsize=(10, 8), dpi=100)
                self.canvas = FigureCanvas(self.figure)
                self.toolbar = NavigationToolbar(self.canvas, self)
                layout.addWidget(self.toolbar)
                layout.addWidget(self.canvas)
                self.ax = self.figure.add_subplot(111)
        except Exception:
            # ?????????????????????????????????
            pass

        self.current_image = None
        self.colorbar = None

        # ??????????
        self.current_xlim = None
        self.current_ylim = None
        self.last_image_shape = None
        self._last_use_log = None
        self._last_show_q_axis = None

        # Q??????
        self._q_detector = None
        self._q_cache_key = None  # ????????????????
        self._qy_mesh = None
        self._qz_mesh = None

        # ?????????????
        self.selection_mode = False
        self.selection_start = None
        self.selection_rect = None
        self.current_selection = None
        self.parameter_selection = None  # ?????????????
        self.parameter_selection_center = None
        self.parameter_selection_info = None

        # ????????????????
        self.setFocusPolicy(Qt.StrongFocus)
        if self.canvas is not None:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
        central_widget.setFocusPolicy(Qt.StrongFocus)

        # ????????????
        if self.ax is not None:
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)

        # ????????????????
        if self.canvas is not None:
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.canvas.mpl_connect('key_press_event', self._on_key_press)
        install_adaptive_window_profile(self, self._apply_screen_profile, apply_window_minimum=False)

    def _apply_screen_profile(self, profile, screen):
        apply_density_profile(self, profile)

    def _on_xlim_changed(self, ax):
        """No description."""
        self.current_xlim = ax.get_xlim()

    def _on_ylim_changed(self, ax):
        """No description."""
        self.current_ylim = ax.get_ylim()

    def _on_mouse_press(self, event):
        """No description."""
        if event.button == 3:  # ???
            if not self.selection_mode:
                self.selection_mode = True
                self.setWindowTitle(self.SELECTION_TITLE)
                self.canvas.setCursor(Qt.CrossCursor)
                self.status_updated.emit("Selection mode activated - Drag to select, Right-click again to exit")
            else:
                self._exit_selection_mode()
            return

        if event.button == 1 and self.selection_mode and event.inaxes == self.ax:  # ?????xes??
            self.selection_start = (event.xdata, event.ydata)
            if self.selection_rect:
                self.selection_rect.remove()
                self.selection_rect = None
            self.status_updated.emit("Selection started - drag to define region")

    def _on_mouse_move(self, event):
        """No description."""
        if (self.selection_mode and self.selection_start and
            event.inaxes == self.ax and event.xdata and event.ydata):

            # ?????????
            if self.selection_rect:
                self.selection_rect.remove()

            # ?????????
            x0, y0 = self.selection_start
            x1, y1 = event.xdata, event.ydata

            width = abs(x1 - x0)
            height = abs(y1 - y0)
            x_min = min(x0, x1)
            y_min = min(y0, y1)

            # ?????????
            from matplotlib.patches import Rectangle
            self.selection_rect = Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
            )
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        """No description."""
        if (self.selection_mode and self.selection_start and
            event.button == 1 and event.inaxes == self.ax and
            event.xdata and event.ydata):

            # ?????????????
            x0, y0 = self.selection_start
            x1, y1 = event.xdata, event.ydata

            # ???????????Q??????
            show_q_axis = self._should_show_q_axis()

            # ????????????????????????
            min_size_threshold = 0.001 if show_q_axis else 5  # Q????????????????
            if abs(x1 - x0) > min_size_threshold and abs(y1 - y0) > min_size_threshold:
                # ?????????
                width = abs(x1 - x0)
                height = abs(y1 - y0)
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2

                # ??????????????
                image_shape = getattr(self, 'current_image_shape', (1, 1))
                img_height, img_width = image_shape

                if show_q_axis:
                    # Q?????????????????????????????????????
                    selection_info = {
                        'center_x': center_x,      # Q??? (qy)
                        'center_y': center_y,      # Q??? (qz)
                        'width': width,            # Q??????
                        'height': height,          # Q??????
                        'is_q_space': True,        # ??????????
                        'bounds': {
                            'x_min': min(x0, x1),
                            'x_max': max(x0, x1),
                            'y_min': min(y0, y1),
                            'y_max': max(y0, y1)
                        }
                    }

                    # ????????Q??????
                    self.setWindowTitle(
                        f"GIMaP Image Viewer - Q selection: "
                        f"center=({center_x:.6f}, {center_y:.6f}) nm^-1, "
                        f"size=({width:.6f} x {height:.6f}) nm^-1"
                    )
                else:
                    # ???????????????????????flipud????????rigin='lower'???
                    # ????atplotlib????????????????????
                    original_center_y = center_y

                    # ?????????
                    selection_info = {
                        'center_x': center_x,
                        'center_y': center_y,  # matplotlib???
                        'width': width,
                        'height': height,
                        'pixel_center_x': int(center_x),
                        'pixel_center_y': int(original_center_y),  # ?????????????????
                        'pixel_width': int(width),
                        'pixel_height': int(height),
                        'is_q_space': False,       # ?????????????
                        'bounds': {
                            'x_min': min(x0, x1),
                            'x_max': max(x0, x1),
                            'y_min': min(y0, y1),
                            'y_max': max(y0, y1)
                        }
                    }

                    # ?????????????????????????????
                    self.setWindowTitle(
                        f"GIMaP Image Viewer - Pixel selection: "
                        f"center=({selection_info['pixel_center_x']}, {selection_info['pixel_center_y']}), "
                        f"size=({selection_info['pixel_width']} x {selection_info['pixel_height']}) px"
                    )

                self.current_selection = selection_info

                # ??????????
                self.region_selected.emit(selection_info)

            # ??????????
            self.selection_start = None

    def _on_key_press(self, event):
        """No description."""
        if event.key == 'escape':
            self._exit_selection_mode()
        elif event.key == 'delete' or event.key == 'backspace':
            self._clear_selection()

    def keyPressEvent(self, event):
        """Qt?????????????atplotlib"""
        try:
            if event.key() == Qt.Key_Escape:
                self._exit_selection_mode()
            elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                self._clear_selection()
            else:
                super().keyPressEvent(event)
        except Exception:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Qt"""
        # ??canvas??????????????
        self.canvas.setFocus()
        super().mousePressEvent(event)

    def _exit_selection_mode(self):
        """Exit selection mode."""
        self.selection_mode = False
        self.selection_start = None
        self.canvas.unsetCursor()
        self.setWindowTitle(self.DEFAULT_TITLE)
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()

    def _clear_selection(self):
        """No description."""
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()
        self.current_selection = None
        if self.selection_mode:
            self.setWindowTitle(self.SELECTION_TITLE)
        else:
            self.setWindowTitle(self.DEFAULT_TITLE)

    def update_image(self, image_data, vmin=None, vmax=None, use_log=True):
        """No description."""
        try:
            # ????????????????
            t_total_update = time.perf_counter()
            current_shape = image_data.shape
            shape_changed = (self.last_image_shape is None or
                           self.last_image_shape != current_shape)

            # ?????????????????????????
            self.current_image_shape = current_shape

            if shape_changed:
                # ?????????????????
                self.current_xlim = None
                self.current_ylim = None
                self.last_image_shape = current_shape

            # ????????- ?????????????
            saved_xlim = self.current_xlim
            saved_ylim = self.current_ylim
            preserve_view = (not shape_changed and saved_xlim is not None and saved_ylim is not None)
            can_reuse_artist = (
                self.current_image is not None and
                not shape_changed and
                self._last_use_log == bool(use_log) and
                self._last_show_q_axis == self._should_show_q_axis()
            )
            if can_reuse_artist:
                t_total = time.perf_counter()
                if use_log:
                    t0 = time.perf_counter()
                    safe_data = np.where(image_data > 0, image_data, 0.001)
                    processed_data = np.log(safe_data, dtype=np.float32)
                    print(f"[Timing] log transform: {(time.perf_counter() - t0) * 1000:.2f} ms (independent window)")
                else:
                    processed_data = image_data.astype(np.float32)
                if vmin is None or vmax is None:
                    t0 = time.perf_counter()
                    auto_vmin = np.percentile(processed_data, 1)
                    auto_vmax = np.percentile(processed_data, 99)
                    vmin = vmin if vmin is not None else auto_vmin
                    vmax = vmax if vmax is not None else auto_vmax
                    print(f"[Timing] autoscale calculation: {(time.perf_counter() - t0) * 1000:.2f} ms (independent window)")
                processed_data = np.flipud(processed_data)
                self.current_image.set_data(processed_data)
                self.current_image.set_clim(vmin, vmax)
                if self.colorbar is not None:
                    try:
                        self.colorbar.update_normal(self.current_image)
                    except Exception:
                        pass
                if preserve_view:
                    self.ax.set_xlim(saved_xlim)
                    self.ax.set_ylim(saved_ylim)
                self._redraw_parameter_selection()
                render_start = time.perf_counter()
                self.canvas.draw()
                print(f"[Timing] Matplotlib rendering: {(time.perf_counter() - render_start) * 1000:.2f} ms (independent window)")
                print(f"[Timing] independent window rendering: {(time.perf_counter() - t_total) * 1000:.2f} ms")
                return

            # ??????????????
            try:
                xlim_cid = None
                ylim_cid = None

                try:
                    for cid, func in self.ax.callbacks.callbacks['xlim_changed'].items():
                        if func.func == self._on_xlim_changed:
                            xlim_cid = cid
                            break
                    for cid, func in self.ax.callbacks.callbacks['ylim_changed'].items():
                        if func.func == self._on_ylim_changed:
                            ylim_cid = cid
                            break

                    if xlim_cid is not None:
                        self.ax.callbacks.disconnect(xlim_cid)
                    if ylim_cid is not None:
                        self.ax.callbacks.disconnect(ylim_cid)

                except (AttributeError, KeyError):
                    try:
                        self.ax.callbacks.disconnect('xlim_changed', self._on_xlim_changed)
                        self.ax.callbacks.disconnect('ylim_changed', self._on_ylim_changed)
                    except TypeError:
                        pass

            except Exception:
                pass

            # ???????????olorbar
            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception:
                    pass
                finally:
                    self.colorbar = None

            # ???axes
            self.ax.clear()

            # ???????????????
            if use_log:
                t0 = time.perf_counter()
                safe_data = np.where(image_data > 0, image_data, 0.001)
                processed_data = np.log(safe_data, dtype=np.float32)
                print(f"[Timing] log transform: {(time.perf_counter() - t0) * 1000:.2f} ms")
                scale_text = "Log Scale"
                colorbar_label = "Log Intensity"
            else:
                processed_data = image_data.astype(np.float32)
                scale_text = "Linear Scale"
                colorbar_label = "Intensity"

            # ?????????vmin/vmax????????
            if vmin is None or vmax is None:
                t0 = time.perf_counter()
                auto_vmin = np.percentile(processed_data, 1)
                auto_vmax = np.percentile(processed_data, 99)
                vmin = vmin if vmin is not None else auto_vmin
                vmax = vmax if vmax is not None else auto_vmax
                print(f"[Timing] autoscale calculation: {(time.perf_counter() - t0) * 1000:.2f} ms (independent window)")

            # ???????????????????????
            processed_data = np.flipud(processed_data)

            # ???????????????
            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                # ???Q?????????extent
                extent = self._get_q_axis_extent(image_data.shape)

                # ?????????????????????extent
                qy_mesh, qz_mesh = self._get_cached_q_meshgrids()

                if qy_mesh is not None and qz_mesh is not None:
                    # ???Q??????????extent [left, right, bottom, top]
                    qy_min, qy_max = qy_mesh.min(), qy_mesh.max()
                    qz_min, qz_max = qz_mesh.min(), qz_mesh.max()
                    q_extent = [qy_min, qy_max, qz_min, qz_max]

                    # ???imshow???Q?????
                    self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal',
                                                      origin='lower', interpolation='nearest',
                                                      vmin=vmin, vmax=vmax, extent=q_extent)
                else:
                    # ???Q?????????????????xtent
                    self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal',
                                                      origin='lower', interpolation='nearest',
                                                      vmin=vmin, vmax=vmax, extent=extent)

                # ???Q?????
                self.ax.set_xlabel(r'$q_y$ (nm$^{-1}$)')
                self.ax.set_ylabel(r'$q_z$ (nm$^{-1}$)')
            else:
                # ?????????????????
                self.current_image = self.ax.imshow(processed_data, cmap='viridis', aspect='equal',
                                                  origin='lower', interpolation='nearest',
                                                  vmin=vmin, vmax=vmax)
                # ???????????
                self.ax.set_xlabel('Pixels (Horizontal)')
                self.ax.set_ylabel('Pixels (Vertical)')

            # ?????
            coord_info = "Q-space" if show_q_axis else "Pixel coordinates"
            self.ax.set_title(
                f'GISAXS Image ({scale_text}) - {image_data.shape[1]} x {image_data.shape[0]} ({coord_info})\n'
                f'Vmin: {vmin:.3f}, Vmax: {vmax:.3f}'
            )

            # ???????????
            if show_q_axis:
                self.ax.set_aspect('equal')  # Q????????qual aspect?????????
            else:
                self.ax.set_aspect('equal')  # ?????????equal aspect

            # ???????????
            try:
                self.colorbar = self.figure.colorbar(self.current_image, ax=self.ax)
                self.colorbar.set_label(colorbar_label)
            except Exception:
                self.colorbar = None

            # ???????????
            self.figure.tight_layout()

            # ????????????????????/????????
            if preserve_view:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
                self.current_xlim = saved_xlim
                self.current_ylim = saved_ylim
            else:
                # ????????????????????
                if show_q_axis:
                    # Q????????atplotlib???????????extent?????
                    self.ax.autoscale()

                else:
                    # ???????????????????
                    self.ax.set_xlim(-0.5, processed_data.shape[1] - 0.5)
                    self.ax.set_ylim(-0.5, processed_data.shape[0] - 0.5)

                self.current_xlim = self.ax.get_xlim()
                self.current_ylim = self.ax.get_ylim()

            self._redraw_parameter_selection()

            # ???????????????
            try:
                self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
                self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            except Exception:
                pass

            # ??????
            self._last_use_log = bool(use_log)
            self._last_show_q_axis = show_q_axis
            render_start = time.perf_counter()
            self.canvas.draw()
            print(f"[Timing] Matplotlib rendering: {(time.perf_counter() - render_start) * 1000:.2f} ms (independent window)")
            print(f"[Timing] independent window rendering: {(time.perf_counter() - t_total_update) * 1000:.2f} ms")

            # ??????????????????anvas??????
            if preserve_view:
                def final_view_check():
                    current_xlim_after_draw = self.ax.get_xlim()
                    current_ylim_after_draw = self.ax.get_ylim()
                    if (abs(current_xlim_after_draw[0] - saved_xlim[0]) > 0.01 or
                        abs(current_xlim_after_draw[1] - saved_xlim[1]) > 0.01 or
                        abs(current_ylim_after_draw[0] - saved_ylim[0]) > 0.01 or
                        abs(current_ylim_after_draw[1] - saved_ylim[1]) > 0.01):
                        self.ax.set_xlim(saved_xlim)
                        self.ax.set_ylim(saved_ylim)
                        self.current_xlim = saved_xlim
                        self.current_ylim = saved_ylim
                        self.canvas.draw_idle()

                QTimer.singleShot(50, final_view_check)

        except Exception as e:
            pass

    def _convert_q_to_pixel_coordinates(self, center_qy, center_qz, width_q, height_q):
        """No description."""
        try:
            # ??????????????
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()

            if qy_mesh is None or qz_mesh is None:
                # ???Q???????????????
                return {'center_x': 0, 'center_y': 0, 'width': 100, 'height': 100}

            # ????????
            if hasattr(self, 'current_stack_data') and self.current_stack_data is not None:
                img_height, img_width = self.current_stack_data.shape
            else:
                img_height, img_width = qy_mesh.shape

            # ??????????Q???????????
            qy_diff = np.abs(qy_mesh - center_qy)
            qz_diff = np.abs(qz_mesh - center_qz)
            combined_diff = qy_diff + qz_diff
            center_idx = np.unravel_index(np.argmin(combined_diff), qy_mesh.shape)
            center_pixel_y, center_pixel_x = center_idx

            # ???Q??????????????????
            qy_range = qy_mesh.max() - qy_mesh.min()
            qz_range = qz_mesh.max() - qz_mesh.min()
            pixel_x_range = img_width
            pixel_y_range = img_height

            qy_to_pixel_ratio = pixel_x_range / qy_range
            qz_to_pixel_ratio = pixel_y_range / qz_range

            # ??????????
            width_pixel = width_q * qy_to_pixel_ratio
            height_pixel = height_q * qz_to_pixel_ratio

            result = {
                'center_x': int(center_pixel_x),
                'center_y': int(center_pixel_y),
                'width': int(width_pixel),
                'height': int(height_pixel)
            }

            return result

        except Exception as e:
            return {'center_x': 0, 'center_y': 0, 'width': 100, 'height': 100}

    def _update_cutline_labels_units(self):
        """No description."""
        try:
            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                # Q???????????(nm??? ???
                unit_suffix = " (nm^-1)"
            else:
                # ??????????????(pixel) ???
                unit_suffix = " (pixel)"

            # ???Center??
            if hasattr(self.ui, 'gisaxsInputCenterVerticalLabel'):
                self.ui.gisaxsInputCenterVerticalLabel.setText(f"Vertical.{unit_suffix}")

            if hasattr(self.ui, 'gisaxsInputCenterParallelLabel'):
                self.ui.gisaxsInputCenterParallelLabel.setText(f"Parallel.{unit_suffix}")

            # ???Cut Line??
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalLabel'):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(f"Vertical.{unit_suffix}")

            if hasattr(self.ui, 'gisaxsInputCutLineParallelLabel'):
                self.ui.gisaxsInputCutLineParallelLabel.setText(f"Parallel.{unit_suffix}")

        except Exception:
            pass

    def _should_show_q_axis(self):
        """No description."""
        try:
            from core.global_params import GlobalParameterManager
            global_params = GlobalParameterManager()
            return global_params.get_parameter('fitting', 'detector.show_q_axis', False)
        except Exception:
            return False

    def _get_q_axis_extent(self, image_shape):
        """Q???extent"""
        try:
            from core.global_params import GlobalParameterManager
            global_params = GlobalParameterManager()

            # ????????? - ??itting????????????
            height, width = image_shape
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)  # micrometers
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)  # micrometers
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)  # Fitting??????beam center
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)  # Fitting??????beam center
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)  # mm
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)  # nm

            # Q-axis calculation parameters

            # ????????
            cache_key = f"{width}x{height}_{pixel_size_x}_{pixel_size_y}_{beam_center_x}_{beam_center_y}_{distance}_{theta_in_deg}_{wavelength}"

            # ???????????????
            if self._q_cache_key != cache_key or self._q_detector is None:
                # ??????????????Q???????????
                from utils.q_space_calculator import create_detector_from_image_and_params
                self._q_detector = create_detector_from_image_and_params(
                    image_shape=(height, width),
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    beam_center_x=beam_center_x,
                    beam_center_y=beam_center_y,
                    distance=distance,
                    theta_in_deg=theta_in_deg,
                    wavelength=wavelength,
                    crop_params=None  # ??????????????????????????????
                )

                # ???Q??????
                self._qy_mesh, self._qz_mesh = self._q_detector.get_qy_qz_meshgrids()
                self._q_cache_key = cache_key



            from utils.q_space_calculator import get_q_axis_labels_and_extents
            _, _, extent = get_q_axis_labels_and_extents(self._q_detector)
            return extent

        except Exception:
            # ???????????extent
            height, width = image_shape
            return [-0.5, width - 0.5, -0.5, height - 0.5]

    def _get_cached_q_meshgrids(self):
        """No description."""
        return self._qy_mesh, self._qz_mesh

    def closeEvent(self, event):
        """Clean up the figure when the window closes."""
        # ???colorbar
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None

        # ??????
        try:
            self.figure.clear()
        except Exception:
            pass


        super().closeEvent(event)

    def update_parameter_selection(self, center_v, center_p, cutline_v, cutline_p):
        """No description."""
        if center_v == 0 and center_p == 0 and cutline_v == 0 and cutline_p == 0:
            self.clear_parameter_selection()
            return

        x_start = center_p - cutline_p / 2
        x_end = center_p + cutline_p / 2
        y_start = center_v - cutline_v / 2
        y_end = center_v + cutline_v / 2

        self.set_parameter_selection({
            'bounds': {
                'x_min': x_start,
                'x_max': x_end,
                'y_min': y_start,
                'y_max': y_end,
            },
            'pixel_center_x': center_p,
            'pixel_center_y': center_v,
            'pixel_width': cutline_p,
            'pixel_height': cutline_v,
            'is_q_space': False,
            'is_parameter_based': True,
        })

    def set_parameter_selection(self, selection_info):
        """No description."""
        self.parameter_selection_info = dict(selection_info) if selection_info else None
        self._redraw_parameter_selection()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _redraw_parameter_selection(self):
        """No description."""
        try:
            if self.parameter_selection is not None:
                try:
                    self.parameter_selection.remove()
                except Exception:
                    pass
                finally:
                    self.parameter_selection = None
            if self.parameter_selection_center is not None:
                try:
                    self.parameter_selection_center.remove()
                except Exception:
                    pass
                finally:
                    self.parameter_selection_center = None

            if not self.parameter_selection_info or self.ax is None:
                return

            bounds = self.parameter_selection_info.get('bounds', {})
            x_min = bounds.get('x_min', 0)
            x_max = bounds.get('x_max', 0)
            y_min = bounds.get('y_min', 0)
            y_max = bounds.get('y_max', 0)
            if x_min == x_max or y_min == y_max:
                return

            from matplotlib.patches import Rectangle
            self.parameter_selection = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                alpha=0.85,
            )
            self.ax.add_patch(self.parameter_selection)
            center_lines = self.ax.plot((x_min + x_max) / 2, (y_min + y_max) / 2,
                                        'r+', markersize=10, markeredgewidth=2)
            self.parameter_selection_center = center_lines[0] if center_lines else None
        except Exception:
            pass

    def clear_parameter_selection(self):
        """No description."""
        self.parameter_selection_info = None
        if self.parameter_selection is not None:
            try:
                self.parameter_selection.remove()
            except Exception:
                pass
            finally:
                self.parameter_selection = None
        if self.parameter_selection_center is not None:
            try:
                self.parameter_selection_center.remove()
            except Exception:
                pass
            finally:
                self.parameter_selection_center = None
        self.canvas.draw()

    def closeEvent(self, event):
        """No description."""
        # ???colorbar
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None

        # ??????
        try:
            self.figure.clear()
        except Exception:
            pass

        super().closeEvent(event)


class IndependentFitWindow(QMainWindow):
    """atplotlib??????????????ut"""

    # ??????????
    status_updated = pyqtSignal(str)
    display_unit_changed = pyqtSignal(str)
    input_point_delete_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GIMaP Cut Analysis - Independent Fit Window")
        self.setGeometry(150, 150, 800, 600)

        # ?????widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # ??????
        layout = QVBoxLayout(central_widget)

        # ???matplotlib???????????
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
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                self.figure = Figure(figsize=(10, 6), dpi=100)
                self.canvas = FigureCanvas(self.figure)
                self.toolbar = NavigationToolbar(self.canvas, self)
        except Exception:
            pass

        # ?????????????
        control_layout = self._create_control_buttons()

        # ?????????????anvas?????
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addLayout(control_layout)  # ???????
        if self.canvas is not None:
            layout.addWidget(self.canvas)

        # ???axes
        if self.figure is not None:
            self.ax = self.figure.add_subplot(111)

        # ????????????????
        self.setFocusPolicy(Qt.StrongFocus)
        if self.canvas is not None:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.mpl_connect('button_press_event', self._on_canvas_button_press)

        # ???????
        if self.figure is not None and self.canvas is not None and self.ax is not None:
            self._setup_empty_plot()
        install_adaptive_window_profile(self, self._apply_screen_profile, apply_window_minimum=False)

    def _apply_screen_profile(self, profile, screen):
        apply_density_profile(self, profile)

    def _setup_empty_plot(self):
        """No description."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Perform a cut operation to see results here.\nDouble-click the Fitting Plot to open a larger window.',
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, alpha=0.7)
        self.ax.set_xlabel('Position')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title('GIMaP Cut Analysis Results')
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def _create_control_buttons(self):
        """No description."""
        from PyQt5.QtWidgets import QHBoxLayout, QCheckBox, QLabel, QComboBox
        control_layout = QHBoxLayout()

        # ?????
        control_layout.addWidget(QLabel("Data Filter:"))

        # ????????
        self.show_positive_cb = QCheckBox("Positive Only")
        self.show_positive_cb.toggled.connect(self._on_show_positive_toggled)
        control_layout.addWidget(self.show_positive_cb)

        # ??????????????????????|q|
        self.show_negative_cb = QCheckBox("Negative Only")
        self.show_negative_cb.toggled.connect(self._on_show_negative_toggled)
        control_layout.addWidget(self.show_negative_cb)

        control_layout.addSpacing(12)
        control_layout.addWidget(QLabel("q Unit:"))

        self.q_unit_combo = QComboBox()
        self.q_unit_combo.addItem("q (Angstrom^-1)", "angstrom")
        self.q_unit_combo.addItem("q (nm^-1)", "nm")
        self.q_unit_combo.setCurrentIndex(1)
        self.q_unit_combo.currentTextChanged.connect(self._on_q_unit_changed)
        control_layout.addWidget(self.q_unit_combo)

        control_layout.addSpacing(12)
        control_layout.addWidget(QLabel("Y Range:"))

        self.y_range_combo = QComboBox()
        self.y_range_combo.addItem("Experiment", "experimental")
        self.y_range_combo.addItem("Fit", "fitting")
        self.y_range_combo.addItem("All", "all")
        self.y_range_combo.setCurrentIndex(2)
        self.y_range_combo.currentTextChanged.connect(self._on_y_range_changed)
        control_layout.addWidget(self.y_range_combo)

        control_layout.addSpacing(12)
        self.delete_input_points_cb = QCheckBox("Delete Points")
        self.delete_input_points_cb.setToolTip("Enable, then left-click a plotted input point to exclude it from AI fitting input.")
        self.delete_input_points_cb.toggled.connect(self._on_delete_input_points_toggled)
        control_layout.addWidget(self.delete_input_points_cb)

        control_layout.addStretch(1)
        return control_layout

    def _on_show_positive_toggled(self, checked):
        """No description."""
        if checked and hasattr(self, 'show_negative_cb') and self.show_negative_cb.isChecked():
            self.show_negative_cb.blockSignals(True)
            self.show_negative_cb.setChecked(False)
            self.show_negative_cb.blockSignals(False)
        self.status_updated.emit(f"Positive Only mode: {'enabled' if checked else 'disabled'}")

    def _on_show_negative_toggled(self, checked):
        """No description."""
        if checked and hasattr(self, 'show_positive_cb') and self.show_positive_cb.isChecked():
            self.show_positive_cb.blockSignals(True)
            self.show_positive_cb.setChecked(False)
            self.show_positive_cb.blockSignals(False)
        self.status_updated.emit(f"Negative Only mode: {'enabled' if checked else 'disabled'}")

    def _get_q_unit_key(self):
        """No description."""
        try:
            if hasattr(self, 'q_unit_combo'):
                unit = self.q_unit_combo.currentData()
                if isinstance(unit, str) and unit.lower() in ('angstrom', 'nm'):
                    return unit.lower()
        except Exception:
            pass
        return 'nm'

    def _get_q_unit_scale_factor(self):
        """No description."""
        return 0.1 if self._get_q_unit_key() == 'angstrom' else 1.0

    def _format_q_axis_label(self, filter_mode='all', absolute=False):
        """No description."""
        unit_text = 'nm$^{-1}$' if self._get_q_unit_key() == 'nm' else r'$\AA^{-1}$'
        base = '|q|' if absolute or filter_mode == 'negative' else 'q'
        suffix = ''
        if filter_mode == 'positive':
            suffix = ' [Positive Only]'
        elif filter_mode == 'negative':
            suffix = ' [Negative Only]'
        return f'{base} ({unit_text}){suffix}'

    def _on_q_unit_changed(self, _text):
        """No description."""
        unit_text = 'nm^-1' if self._get_q_unit_key() == 'nm' else 'Angstrom^-1'
        self.status_updated.emit(f"q unit changed to {unit_text}")
        self.display_unit_changed.emit(unit_text)

    def _on_y_range_changed(self, _text):
        """No description."""
        mode = self._get_y_range_mode()
        label = {
            'experimental': 'experimental data',
            'fitting': 'fitting data',
            'all': 'all visible data',
        }.get(mode, 'all visible data')
        self.status_updated.emit(f"Y range based on {label}")

    def _get_y_range_mode(self):
        """No description."""
        try:
            if hasattr(self, 'y_range_combo'):
                mode = self.y_range_combo.currentData()
                if mode in ('experimental', 'fitting', 'all'):
                    return mode
        except Exception:
            pass
        return 'all'

    def _on_delete_input_points_toggled(self, checked):
        """Enable point deletion mode for AI fitting input outliers."""
        self._delete_input_points_enabled = bool(checked)
        if self.canvas is not None:
            self.canvas.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)
        self.status_updated.emit(
            "Delete Points mode enabled: left-click a data point to exclude it."
            if checked else
            "Delete Points mode disabled."
        )

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

    def clear_deletable_points(self):
        self._delete_raw_q = np.array([], dtype=float)
        self._delete_plot_x = np.array([], dtype=float)
        self._delete_plot_y = np.array([], dtype=float)

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

    def update_plot(self, x_coords, y_intensity, x_label, y_label, title, log_x=False, log_y=False, normalize=False, y_errors=None):
        """No description."""
        try:
            # ???????
            x_data = np.array(x_coords)
            y_data = np.array(y_intensity)


            # ??????????
            err_data = None
            if y_errors is not None:
                err_data = np.array(y_errors)

            is_q_axis = isinstance(x_label, str) and 'q' in x_label.lower()

            # ?????????????????
            if hasattr(self, 'show_positive_cb') and self.show_positive_cb.isChecked():
                mask = x_data > 0
                x_data = x_data[mask]
                y_data = y_data[mask]
                if err_data is not None:
                    err_data = err_data[mask]
            elif hasattr(self, 'show_negative_cb') and self.show_negative_cb.isChecked():
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
                if hasattr(self, 'show_positive_cb') and self.show_positive_cb.isChecked():
                    x_label = self._format_q_axis_label(filter_mode='positive')
                elif hasattr(self, 'show_negative_cb') and self.show_negative_cb.isChecked():
                    x_label = self._format_q_axis_label(filter_mode='negative')
                else:
                    original_x = np.asarray(x_coords)
                    has_negative = np.any(np.isfinite(original_x) & (original_x < 0))
                    x_label = self._format_q_axis_label(absolute=(log_x and has_negative))

            # ???????????????????
            if normalize:
                max_intensity = np.max(y_data) if y_data.size > 0 else 0.0
                if max_intensity > 0:
                    y_data = y_data / float(max_intensity)
                    if err_data is not None:
                        err_data = err_data / float(max_intensity)
                    y_label = "Normalized Intensity"

            # ????????
            self.ax.clear()

            # ?????????????
            if err_data is not None:
                # ?????????rrorbar
                self.ax.errorbar(x_data, y_data, yerr=err_data, fmt='o-',
                               markersize=4, linewidth=1.5, capsize=3,
                               alpha=0.8, label='Data with error bars')
            else:
                # ???????????????plot???????????????
                try:
                    FittingController._plot_cut_data_with_log_handling(self.ax, x_data, y_data, log_x, markersize=6, linewidth=2)
                except:
                    # ???????????????????????
                    self.ax.plot(x_data, y_data, 'o-', markersize=4, linewidth=1.5, alpha=0.8, label='Data')

            try:
                self.set_deletable_points(x_data / self._get_q_unit_scale_factor() if is_q_axis else x_data, x_data, y_data)
            except Exception:
                self.clear_deletable_points()

            # ??????????????????????mathtext ???????????????
            try:
                # ?????? '???? ??'A^-1' ???????mathtext ???
                def _to_mathtext(label: str) -> str:
                    if not isinstance(label, str):
                        return label
                    return (label
                            .replace('A^-1', r'$\AA^{-1}$')
                            .replace('Ang^-1', r'$\AA^{-1}$')
                            .replace('(A^-1)', r'($\AA^{-1}$)')
                            .replace('(Ang^-1)', r'($\AA^{-1}$)'))
                x_lbl = _to_mathtext(x_label)
                y_lbl = _to_mathtext(y_label)
            except Exception:
                x_lbl, y_lbl = x_label, y_label

            self.ax.set_xlabel(x_lbl, fontsize=13)
            self.ax.set_ylabel(y_lbl, fontsize=13)
            self.ax.set_title(title, fontsize=15)

            # ????????????
            if log_x:
                self.ax.set_xscale('log')
            else:
                self.ax.set_xscale('linear')

            if log_y:
                self.ax.set_yscale('log')
            else:
                self.ax.set_yscale('linear')

            # ????????
            self.ax.grid(True, alpha=0.4, linestyle='--')
            try:
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.ax.spines[axis].set_linewidth(1.8)
                self.ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)
            except Exception:
                pass

            # ???????????????????
            # stats_text = f'Points: {len(x_data)}\nMax: {np.max(y_data):.2e}\nMin: {np.min(y_data):.2e}'
            # self.ax.text(0.02, 0.88, stats_text, transform=self.ax.transAxes,
            #             verticalalignment='bottom', fontsize=10,
            #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # ??????
            self.figure.tight_layout()

            # ??????
            self.canvas.draw()

            # ????????
            self.setWindowTitle(f"GIMaP Cut Analysis - {title}")

            self.status_updated.emit(f"Independent fit window updated: {title}")

        except Exception as e:
            self.status_updated.emit(f"Failed to update independent fit window: {str(e)}")

    def closeEvent(self, event):
        """No description."""
        try:
            self.figure.clear()
        except Exception:
            pass
        super().closeEvent(event)


class UnifiedDisplayManager:
    """Manage unified plot display updates."""

    def __init__(self, controller):
        self.controller = controller
        self.ui = controller.ui

    def plot_1d_data(self, q, intensity, err=None, title="", source_info="",
                     log_x=False, log_y=False, normalize=False):
        """Plot 1D data."""
        try:
            # ???????
            plot_q = np.array(q)
            plot_I = np.array(intensity)
            plot_err = np.array(err) if err is not None else None

            # ???????
            if normalize and len(plot_I) > 0:
                max_I = np.max(plot_I)
                if max_I > 0:
                    plot_I = plot_I / max_I
                    if plot_err is not None:
                        plot_err = plot_err / max_I

            # Log-Y????
            if log_y and len(plot_I) > 0 and not np.all(plot_I > 0):
                min_positive = np.min(plot_I[plot_I > 0]) if np.any(plot_I > 0) else 1e-10
                plot_I = np.where(plot_I <= 0, min_positive, plot_I)
                if plot_err is not None:
                    plot_err = np.where(plot_I <= min_positive, min_positive * 0.1, plot_err)

            # ???GUI?????
            self._update_gui_1d_display(plot_q, plot_I, plot_err, title,
                                       log_x, log_y, normalize)

            # ???????????
            self._update_independent_1d_display(plot_q, plot_I, plot_err, title,
                                               log_x, log_y, normalize)

            # ???????
            y_label = 'Intensity' + (' (normalized)' if normalize else '')
            mode_str = f"Log-X: {log_x}, Log-Y: {log_y}, Norm: {normalize}"
            self.controller.status_updated.emit(f"1D data displayed: {title} [{mode_str}]")

        except Exception as e:
            self.controller.status_updated.emit(f"Failed to plot 1D data: {str(e)}")

    def _update_gui_1d_display(self, q, intensity, err, title, log_x, log_y, normalize):
        """GUI???1D"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView') or not is_matplotlib_available():
                return

            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            # ???figure??:3 ????????400x300 ?????
            figure = Figure(figsize=(8, 6))
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111)

            # ??????????????
            self._unified_plot_1d_data(ax, q, intensity, err, title, log_x, log_y, normalize)

            figure.tight_layout()

            # ????????????
            canvas.draw()

            # ????????
            scene = self.controller._setup_fit_graphics_scene()
            if scene is not None:
                proxy_widget = scene.addWidget(canvas)
                self.controller._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)

                # ??????
                self.controller._current_fit_canvas = canvas
                self.controller._current_fit_figure = figure

                # ??????????????????og-x????????OI???/????????????????????????????
                try:
                    if log_x and hasattr(self.controller, '_adjust_roi_bounds_for_log_x'):
                        # ???????????????t?????????????????lim
                        QTimer.singleShot(0, self.controller._adjust_roi_bounds_for_log_x)
                except Exception:
                    pass

        except Exception as e:
            self.controller.status_updated.emit(f"Failed to update GUI 1D display: {str(e)}")

    def _unified_plot_1d_data(self, ax, q, intensity, err, title, log_x, log_y, normalize):
        """No description."""
        try:
            q_plot = self.controller._convert_q_values_for_display(q)

            # ?????? - ?????????????????????
            if err is not None:
                # ?????????rrorbar
                ax.errorbar(q_plot, intensity, yerr=err, fmt='o-',
                           markersize=3, linewidth=1, capsize=2,
                           alpha=0.8, label='Data with error bars')
            else:
                # ???????????????Log-X??????
                FittingController._plot_cut_data_with_log_handling(
                    ax, q_plot, intensity, log_x, markersize=3, linewidth=1
                )

            # ?????????????? mathtext ??? superscript minus ????????????
            has_negative = np.any(np.isfinite(np.asarray(q)) & (np.asarray(q) < 0))
            ax.set_xlabel(self.controller._build_q_axis_label(absolute=(log_x and has_negative)), fontsize=13)
            ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ''), fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

            # ????????- ??????????????
            if log_x:
                ax.set_xscale('log')
            else:
                ax.set_xscale('linear')

            if log_y:
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')

            # ????????????????????????????
            try:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(1.8)
                ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)
            except Exception:
                pass

        except Exception as e:
            # ??????????
            q_plot = self.controller._convert_q_values_for_display(q)
            if err is not None:
                ax.errorbar(q_plot, intensity, yerr=err, fmt='o-', markersize=3, linewidth=1, capsize=2)
            else:
                ax.plot(q_plot, intensity, 'o-', markersize=3, linewidth=1)
            has_negative = np.any(np.isfinite(np.asarray(q)) & (np.asarray(q) < 0))
            ax.set_xlabel(self.controller._build_q_axis_label(absolute=(log_x and has_negative)), fontsize=13)
            ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ''), fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

    def _update_independent_1d_display(self, q, intensity, err, title, log_x, log_y, normalize):
        """D"""
        try:
            if (self.controller.independent_fit_window and
                hasattr(self.controller.independent_fit_window, 'update_plot')):

                y_label = 'Intensity' + (' (normalized)' if normalize else '')
                q_internal_nm = self.controller._convert_q_values_for_model(
                    q, source=getattr(self.controller, 'data_source', None)
                )
                self.controller.independent_fit_window.update_plot(
                    q_internal_nm, intensity, self.controller._build_q_axis_label(), y_label, title,
                    log_x, log_y, normalize, err
                )

        except Exception as e:
            self.controller.status_updated.emit(f"Failed to update independent 1D display: {str(e)}")

    def get_display_options(self):
        """No description."""
        return {
            'log_x': hasattr(self.ui, 'fitLogXCheckBox') and self.ui.fitLogXCheckBox.isChecked(),
            'log_y': hasattr(self.ui, 'fitLogYCheckBox') and self.ui.fitLogYCheckBox.isChecked(),
            'normalize': hasattr(self.ui, 'fitNormCheckBox') and self.ui.fitNormCheckBox.isChecked()
        }


class AsyncImageLoader(QThread):
    """No description."""

    image_loaded = pyqtSignal(np.ndarray, str)  # ??????, ?????
    progress_updated = pyqtSignal(int, str)  # ???, ???????
    error_occurred = pyqtSignal(str)  # ?????

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.stack_count = 1
        self._image_cache = OrderedDict()
        self._image_cache_limit = 8

    def load_image(self, file_path, stack_count=1):
        """No description."""
        self.file_path = file_path
        self.stack_count = stack_count
        self.start()

    def run(self):
        """No description."""
        try:
            if not is_fabio_available():
                self.error_occurred.emit("fabio library is required for CBF file processing")
                return

            self.progress_updated.emit(10, "Loading file...")

            file_ext = os.path.splitext(self.file_path)[1].lower()

            if file_ext != '.cbf':
                self.error_occurred.emit("Only CBF files are supported currently")
                return

            cache_key = (normalize_path(self.file_path), int(self.stack_count))
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache.move_to_end(cache_key)
                self.progress_updated.emit(90, "Using cached image data...")
                print(f"[Timing] fabio read: 0.00 ms (cache hit: {os.path.basename(self.file_path)})")
                self.image_loaded.emit(cached, self.file_path)
                self.progress_updated.emit(100, "Done")
                return

            read_start = time.perf_counter()
            if self.stack_count == 1:
                # ????????
                self.progress_updated.emit(50, "Loading single CBF file...")
                image_data = self._load_single_cbf_file(self.file_path)
            else:
                # ????????
                self.progress_updated.emit(30, f"Loading and stacking {self.stack_count} files...")
                image_data = self._load_multiple_cbf_files(self.file_path, self.stack_count)
            print(f"[Timing] fabio read: {(time.perf_counter() - read_start) * 1000:.2f} ms ({os.path.basename(self.file_path)})")

            if image_data is not None:
                self._image_cache[cache_key] = image_data
                self._image_cache.move_to_end(cache_key)
                while len(self._image_cache) > self._image_cache_limit:
                    self._image_cache.popitem(last=False)
                self.progress_updated.emit(90, "Processing image data...")
                self.image_loaded.emit(image_data, self.file_path)
                self.progress_updated.emit(100, "Done")
            else:
                self.error_occurred.emit("Failed to load image data")

        except Exception as e:
            self.error_occurred.emit(f"Error loading image: {str(e)}")

    def _load_single_cbf_file(self, cbf_file):
        """CBF"""
        try:
            import fabio
            cbf_file = normalize_path(cbf_file)
            cbf_image = fabio.open(cbf_file)
            data = cbf_image.data

            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)

            return data

        except Exception as e:
            return None

    def _load_multiple_cbf_files(self, start_file, stack_count):
        """BF"""
        try:
            start_file = normalize_path(start_file)
            file_dir = os.path.dirname(start_file)
            base_name = os.path.basename(start_file)

            cbf_files = [f for f in os.listdir(file_dir) if f.lower().endswith('.cbf')]
            cbf_files.sort()

            try:
                start_index = cbf_files.index(base_name)
            except ValueError:
                return None

            available_files = len(cbf_files) - start_index
            if stack_count > available_files:
                return None

            summed_data = None
            files_to_stack = cbf_files[start_index:start_index + stack_count]

            import fabio
            for i, file_name in enumerate(files_to_stack):
                file_path = os.path.join(file_dir, file_name)
                progress = 40 + int((i / len(files_to_stack)) * 40)
                self.progress_updated.emit(progress, f"Processing file {i+1}/{len(files_to_stack)}: {file_name}")

                try:
                    cbf_image = fabio.open(file_path)
                    data = cbf_image.data.astype(np.float32, copy=False) if cbf_image.data.dtype != np.float32 else cbf_image.data

                    if summed_data is None:
                        summed_data = data.copy() if data.dtype == np.float32 else data.astype(np.float32)
                    else:
                        summed_data += data

                except Exception as e:
                    continue

            return summed_data

        except Exception as e:
            return None


class FittingController(QObject):
    """Cut Fitting????????GISAXS"""

    # ???????
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    fitting_completed = pyqtSignal(dict)

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent  # ?????????
        # ???????????
        self.main_window = parent.parent if hasattr(parent, 'parent') else None

        # ???????????
        self.q = None  # q??????
        self.I = None  # I??????
        self.I_fitting = None  # ?????????

        # ???????????
        self.display_mode = 'normal'  # 'normal' ??'fitting'
        self.data_source = None  # 'cut' ??'1d'
        # ROI ????????
        self._q_full_min = None
        self._q_full_max = None
        self._roi_min = None
        self._roi_max = None
        self.q_ROI = None
        self.I_ROI = None
        self._updating_roi_controls = False
        self._roi_controls_enabled = True
        self._last_axis_filter_mode = 'all'
        self._slider_is_source = False
        self._points_num_default = 50
        self._points_num_current = 50
        self._interp_method_default = 'Linear'

        # FittingTextBrowser????????????????????????
        self._fitting_messages_max_lines = 500
        self._detached_fitting_dialog = None
        self._detached_append = None
        self._fitting_browser_original_height = None
        self.has_fitting_data = False  # ??????????

        # ??????
        self.current_parameters = {}

        # ??????
        self.fitting_results = {}

        # ?????????
        self.current_stack_data = None
        self.current_file_list = []
        self._folder_image_files = []
        self._folder_image_index = -1
        self._previous_image_button = None
        self._next_image_button = None
        # ??????????????????????
        self.data = None              # ????????????????????????????
        self.summed_data = None       # ???????????????????????
        self.cut = None               # ??????ut??D??? {'q':..., 'I':..., meta}
        self.fitting = None           # ??????itting??D??? {'q':..., 'I':..., meta}
        # Q?????????????????????????????????????
        self.qy_matrix = None
        self.qz_matrix = None
        self.qr_matrix = None
        self._q_mesh_cache_key = None

        # ??matplotlib???
        self.independent_window = None

        # ???????????
        self.independent_fit_window = None

        # ???????????? (???????????)
        self.current_cut_data = None

        # 1D?????????
        self.current_1d_data = None  # ????????D?????? {q, I, err}
        self.current_1d_file_path = None  # ???????????D?????
        self._imported_1d_q_unit = 'angstrom'  # 1D????????^-1 ?????????????????? nm^-1

        # ???????????????????
        self._last_q_mode = None

        # ??????????????????
        self._graphics_scene = None
        self._figure_cache = None
        self._canvas_cache = None
        self._preview_ax = None
        self._preview_image_artist = None
        self._preview_proxy_widget = None
        self._preview_shape = None
        self._preview_show_q_axis = None
        self._preview_colorbar = None
        self._image_display_cache = OrderedDict()
        self._image_display_cache_limit = 12

        # ????????????
        try:
            # ?????global_params ?????? meta ?????????????????user_settings
            default_points = None
            try:
                from core.global_params import global_params
                gp_val = global_params.get_parameter('fitting', 'fit.points_num', None)
                if gp_val is not None:
                    default_points = int(gp_val)
            except Exception:
                default_points = None
            try:
                from core.user_settings import user_settings
                us_val = int(user_settings.get('fit.points_num', 50))
            except Exception:
                us_val = 50
            self._points_num_default = int(default_points if default_points is not None else us_val)
            self._points_num_current = self._points_num_default
            # ???????? user_settings ???
            try:
                from core.user_settings import user_settings as _us
                self._interp_method_default = _us.get('fit.interp_method', 'Linear')
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._setup_fitting_region_controls()
        except Exception:
            pass

        # ???????????????
        self._fit_graphics_scene = None
        self._current_fit_canvas = None
        self._current_fit_figure = None

        # ???????????
        self._display_manager = UnifiedDisplayManager(self)

        # ?????????
        self._current_vmin = None
        self._current_vmax = None
        self._updating_color_scale_ui = False
        self._has_displayed_image = False  # ???????????????

        # ???????
        self._initialized = False
        self._initializing = True  # ??????????

        # ???????????
        self.async_image_loader = AsyncImageLoader()
        self.async_image_loader.image_loaded.connect(self._on_image_loaded)
        self.async_image_loader.progress_updated.connect(self._on_image_loading_progress)
        self.async_image_loader.error_occurred.connect(self._on_image_loading_error)

        # ??????????
        self.current_parameter_selection = None

        # ?????????
        self._display_mode = 'normal'  # 'normal' ??'fitting'
        self._has_fitting_data = False  # ??????????
        self._fitting_mode_active = False  # ???????????
        self._last_active_particle_ids = []  # ??????????????????????
        self._particle_widgets = {}
        self._particle_parameter_meta_ids = defaultdict(list)
        self._recycled_particle_ids = []
        self._particle_widget_style_template = ''
        self._particle_widget_style_source_name = ''
        self._particle_widget_style_fallback = (
            "background-color: #ffffff;"
            "border: 1px solid #d8dee8;"
            "border-radius: 7px;"
        )
        self._particle_container_layout = None
        self._particle_add_button = None
        self._particle_show_checkboxes = {}
        self._dynamic_show_layout = None
        self._dynamic_show_container = None
        self._particle_checkbox_host_name = ''
        self._ai_process = None
        self._ai_output_dir = None
        self._ai_input_csv = None
        self._ai_action_buttons = []
        self._ai_stop_button = None
        self._ai_open_output_button = None
        self._ai_export_output_button = None
        self._ai_results_dialog = None
        self._ai_candidate_rows = []
        self._ai_log_lines = []
        self._ai_excluded_input_q = set()
        self._ai_input_data_dialog = None
        self._ai_input_data_table = None
        self._ai_input_data_summary = None
        self._ai_input_dialog_arrays = None

        # ????????ingle/Stack/In-situ?????????
        self.load_mode = 'Single'
        self._insitu_timer = None
        self._insitu_last_file = None

        # ?????????????????finished??????????????changed??
        self._default_signal_mode = 'finished'
        self._signal_mode_overrides = {
            # Fitting region ???????????
            'fitFittingRegionSlider': 'changed',
            # Detector Para. ?? Beam Center?????????????????????????????????
            # ??????????????????:
            # 'detectorBeamCenterX': 'changed',
            # 'detectorBeamCenterY': 'changed',
        }

        # ??????????
        self.detector_params_dialog = None

        # ???????????
        self.model_params_manager = ModelParametersManager()

        # ???????????????????? UniversalParameterTriggerManager?????? FittingParameterTriggerManager ????
        self.param_trigger_manager = UniversalParameterTriggerManager(self)

        # ???????????????????????
        self._loading_parameters = False  # ????????????????
        self._initializing = False  # ??????????????

        # ==========================
        # ???????????????????? meta registry ?????
        # ==========================
        # ???????????????????????
        self._param_debounce_ms = 280
        # ???????+?????
        self._param_abs_eps = 1e-12
        self._param_rel_eps = 1e-10
        # ROI????????????????????????????????????????
        self._roi_debounce_ms = 140
        self._roi_update_timer = None

        # ??K????????
        self._auto_k_enabled = False  # ??????K?????

        # ???????????
        self._setup_particle_shape_connector()

        # ??????????????uto-K?????
        self._load_auto_k_enabled()

    def initialize(self):
        """No description."""
        if self._initialized:
            return

        # ??????UI??????????????
        self._initialize_ui()
        self._setup_folder_navigation_ui()
        # ????????????
        self._setup_connections()
        # ???????????????????????
        self._initialized = True
        self._initializing = False  # ???????
        # ??????????
        self._setup_meta_debug_shortcut()
        # ????????????????????????????????????????????????????????
        try:
            if hasattr(self.ui, 'gisaxsInputModelCombox'):
                mode_now = self.ui.gisaxsInputModelCombox.currentText()
                self.load_mode = mode_now or getattr(self, 'load_mode', 'Single')
                # ???????????????????
                self._update_stack_controls_visibility()
                # ??? In-situ???????????
                if self.load_mode == 'In-situ' and hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setVisible(False)
        except Exception:
            pass

    def _setup_meta_debug_shortcut(self):
        """No description."""
        try:
            sc = QShortcut(QKeySequence("Ctrl+Alt+M"), self.ui)
            def _dump():
                snap = self.param_trigger_manager.debug_dump_meta(verbose=False)
                self._add_fitting_message("==== META SNAPSHOT ====", "INFO")
                for wid, data in snap.items():
                    self._add_fitting_message(f"{wid}: {data}", "INFO")
            sc.activated.connect(_dump)
            self._add_fitting_message("Meta debug shortcut Ctrl+Alt+M registered", "INFO")
        except Exception as e:
            print(f"Failed to register meta debug shortcut: {e}")

    def _setup_folder_navigation_ui(self):
        """Add Previous/Next buttons beside the current GISAXS file field."""
        try:
            if self._previous_image_button is not None:
                return

            parent = getattr(self.ui, 'gisaxsInputBox', None)
            self._previous_image_button = QPushButton("Previous", parent)
            self._next_image_button = QPushButton("Next", parent)
            self._previous_image_button.setObjectName("gisaxsInputPreviousImageButton")
            self._next_image_button.setObjectName("gisaxsInputNextImageButton")
            self._previous_image_button.setToolTip("No previous image")
            self._next_image_button.setToolTip("No next image")
            self._previous_image_button.setEnabled(False)
            self._next_image_button.setEnabled(False)

            nav_layout = getattr(self.ui, 'gisaxsInputFileNavigationLayout', None)
            if nav_layout is not None:
                self._previous_image_button.setParent(getattr(self.ui, 'gisaxsInputFileNavigationWidget', parent))
                self._next_image_button.setParent(getattr(self.ui, 'gisaxsInputFileNavigationWidget', parent))
                nav_layout.addWidget(self._previous_image_button)
                nav_layout.addWidget(self._next_image_button)
            elif hasattr(self.ui, 'gridLayout_23'):
                self.ui.gridLayout_23.addWidget(self._previous_image_button, 0, 5, 1, 1)
                self.ui.gridLayout_23.addWidget(self._next_image_button, 0, 6, 1, 1)
        except Exception as e:
            self.status_updated.emit(f"Failed to set up image navigation: {str(e)}")

    def _supported_folder_image_extensions(self):
        return ('.cbf',)

    def _natural_sort_key(self, path):
        name = os.path.basename(path)
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]

    def _scan_folder_images_for_file(self, file_path):
        """Scan the selected file's folder and update navigation state."""
        try:
            file_path = normalize_path(file_path)
            if not file_path or not os.path.exists(file_path):
                self._folder_image_files = []
                self._folder_image_index = -1
                self._update_folder_navigation_buttons()
                self.status_updated.emit("File does not exist")
                return

            folder = os.path.dirname(file_path)
            current_norm = os.path.normcase(os.path.abspath(file_path))
            files = []
            for name in os.listdir(folder):
                candidate = os.path.join(folder, name)
                if os.path.isfile(candidate) and os.path.splitext(name)[1].lower() in self._supported_folder_image_extensions():
                    files.append(normalize_path(candidate))
            files.sort(key=self._natural_sort_key)

            self._folder_image_files = files
            norm_files = [os.path.normcase(os.path.abspath(p)) for p in files]
            self._folder_image_index = norm_files.index(current_norm) if current_norm in norm_files else -1
            if self._folder_image_index < 0 and files:
                self.status_updated.emit("Current image is not in the scanned folder list")
            self._update_folder_navigation_buttons()
        except Exception as e:
            self._folder_image_files = []
            self._folder_image_index = -1
            self._update_folder_navigation_buttons()
            self.status_updated.emit(f"Folder scan failed: {str(e)}")

    def _update_folder_navigation_buttons(self):
        try:
            count = len(self._folder_image_files)
            index = self._folder_image_index
            has_previous = count > 1 and index > 0
            has_next = count > 1 and 0 <= index < count - 1
            if self._previous_image_button is not None:
                self._previous_image_button.setEnabled(has_previous)
                self._previous_image_button.setToolTip("Previous" if has_previous else "No previous image")
            if self._next_image_button is not None:
                self._next_image_button.setEnabled(has_next)
                self._next_image_button.setToolTip("Next" if has_next else "No next image")
        except Exception:
            pass

    def _show_previous_folder_image(self):
        self._show_folder_image_at_offset(-1)

    def _show_next_folder_image(self):
        self._show_folder_image_at_offset(1)

    def _show_folder_image_at_offset(self, offset):
        try:
            if not self._folder_image_files:
                self.status_updated.emit("No previous image" if offset < 0 else "No next image")
                self._update_folder_navigation_buttons()
                return

            current_file = self.current_parameters.get('imported_gisaxs_file', '')
            if self._folder_image_index < 0 and current_file:
                self._scan_folder_images_for_file(current_file)

            target_index = self._folder_image_index + offset
            if target_index < 0:
                self.status_updated.emit("No previous image")
                self._update_folder_navigation_buttons()
                return
            if target_index >= len(self._folder_image_files):
                self.status_updated.emit("No next image")
                self._update_folder_navigation_buttons()
                return

            self._select_folder_image(self._folder_image_files[target_index])
        except Exception as e:
            self.status_updated.emit(f"Image navigation failed: {str(e)}")

    def _select_folder_image(self, file_path):
        try:
            file_path = normalize_path(file_path)
            if not os.path.exists(file_path):
                QMessageBox.warning(self.main_window, "File Error", f"File does not exist:\n{file_path}")
                self._scan_folder_images_for_file(self.current_parameters.get('imported_gisaxs_file', ''))
                return

            self.current_parameters['imported_gisaxs_file'] = file_path
            if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                self.ui.gisaxsInputImportButtonValue.setText(os.path.basename(file_path))

            self._scan_folder_images_for_file(file_path)
            self._update_stack_display()
            self.parameters_changed.emit(self.current_parameters)
            if hasattr(self.parent, 'save_current_session'):
                self.parent.save_current_session()

            self.status_updated.emit(f"Current image: {os.path.basename(file_path)}")
            self._show_image()
        except Exception as e:
            QMessageBox.warning(self.main_window, "Image Navigation Error", f"Failed to load image:\n{str(e)}")

    # ---------------- ROI helpers for plotting -----------------
    def _roi_active(self) -> bool:
        if not getattr(self, '_roi_controls_enabled', True):
            return False
        return (
            self._roi_min is not None and self._roi_max is not None and
            self._q_full_min is not None and self._q_full_max is not None and
            (abs(self._roi_min - self._q_full_min) > 1e-12 or abs(self._roi_max - self._q_full_max) > 1e-12)
        )

    def _get_roi_active_arrays(self):
        """Return (q_plot, I_plot) using ROI subset if active, else full arrays."""
        if self.q is None or self.I is None:
            return None, None
        # Helper: filter out non-finite pairs to avoid None/inf/NaN issues
        def _filter_finite(q_arr, I_arr):
            q_arr = np.asarray(q_arr)
            I_arr = np.asarray(I_arr)
            mask = np.isfinite(q_arr) & np.isfinite(I_arr)
            if not np.any(mask):
                return q_arr[:0], I_arr[:0]
            return q_arr[mask], I_arr[mask]

        if self._roi_active() and self.q_ROI is not None and self.I_ROI is not None and len(self.q_ROI) > 0:
            return _filter_finite(self.q_ROI, self.I_ROI)
        return _filter_finite(self.q, self.I)

    def _draw_roi_guides_if_active(self, ax):
        try:
            if not getattr(self, '_roi_controls_enabled', True):
                return
            if not self._roi_active():
                return
            q_bounds = self._convert_q_values_for_display(
                np.array([float(self._roi_min), float(self._roi_max)], dtype=float),
                source=getattr(self, 'data_source', None)
            )
            if q_bounds.size >= 2:
                try:
                    if self._get_independent_axis_filter_mode() == 'negative':
                        q_bounds = np.abs(q_bounds)
                except Exception:
                    pass
                q_bounds = np.sort(q_bounds)
                ax.axvline(float(q_bounds[0]), color='red', linestyle='--', linewidth=1.2, alpha=0.8)
                ax.axvline(float(q_bounds[1]), color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        except Exception:
            pass

    def _current_q_has_negative_values(self) -> bool:
        try:
            q = np.asarray(self.q) if self.q is not None else None
            if q is None or q.size == 0:
                return False
            return bool(np.any(np.isfinite(q) & (q < 0)))
        except Exception:
            return False

    def _roi_editing_should_be_enabled(self) -> bool:
        """ROI is ambiguous only when log-x folds positive and negative q together."""
        try:
            return not (
                self._is_fit_log_x_enabled() and
                self._current_q_has_negative_values() and
                self._get_independent_axis_filter_mode() == 'all'
            )
        except Exception:
            return True

    def _set_roi_controls_enabled(self, enabled: bool):
        self._roi_controls_enabled = bool(enabled)
        for name in ('fitFittingRegionSlider', 'fitFittingRegionMinValue', 'fitFittingRegionMaxValue'):
            try:
                if hasattr(self.ui, name):
                    widget = getattr(self.ui, name)
                    widget.setEnabled(bool(enabled))
                    if enabled:
                        widget.setToolTip('')
                    else:
                        widget.setToolTip(
                            'Log-X with both +q and -q is ambiguous. Select Positive Only or Negative Only first.'
                        )
            except Exception:
                pass
        try:
            if hasattr(self.ui, 'fitRegionEditHintLabel'):
                self.ui.fitRegionEditHintLabel.setVisible(not bool(enabled))
        except Exception:
            pass

    def _get_roi_domain_bounds(self):
        if self.q is None or self.I is None:
            return None
        q_all = np.asarray(self.q)
        I_all = np.asarray(self.I)
        valid = np.isfinite(q_all) & np.isfinite(I_all)
        if not np.any(valid):
            return None

        q_valid = q_all[valid]
        log_x = self._is_fit_log_x_enabled()
        filter_mode = self._get_independent_axis_filter_mode()
        if filter_mode == 'positive':
            q_valid = q_valid[q_valid > 0]
        elif filter_mode == 'negative':
            q_valid = q_valid[q_valid < 0]
        elif log_x and not self._current_q_has_negative_values():
            q_valid = q_valid[q_valid > 0]

        if q_valid.size == 0:
            return None
        return float(np.min(q_valid)), float(np.max(q_valid))

    def _roi_controls_use_abs_negative(self) -> bool:
        try:
            return self._get_independent_axis_filter_mode() == 'negative'
        except Exception:
            return False

    def _roi_data_to_control_range(self, q_min: float, q_max: float):
        if self._roi_controls_use_abs_negative():
            vals = np.sort(np.abs(np.array([q_min, q_max], dtype=float)))
            return float(vals[0]), float(vals[1])
        return float(q_min), float(q_max)

    def _roi_data_to_control_values(self, q_min: float, q_max: float):
        return self._roi_data_to_control_range(q_min, q_max)

    def _roi_control_to_data_values(self, vmin: float, vmax: float):
        if self._roi_controls_use_abs_negative():
            lo, hi = sorted((abs(float(vmin)), abs(float(vmax))))
            return -hi, -lo
        return float(vmin), float(vmax)

    def _nearest_roi_control_value(self, value: float):
        try:
            q = np.asarray(self.q) if self.q is not None else None
            if q is None or q.size == 0:
                return float(value)
            finite = q[np.isfinite(q)]
            if finite.size == 0:
                return float(value)
            if self._roi_controls_use_abs_negative():
                finite = np.abs(finite[finite < 0])
            if finite.size == 0:
                return float(value)
            return float(finite[np.argmin(np.abs(finite - value))])
        except Exception:
            return float(value)

    def _sync_roi_controls_to_current_display(self, reset_to_domain: bool = False):
        """Update ROI bounds/editability to match the current Fitting Plot display."""
        enabled = self._roi_editing_should_be_enabled()
        self._set_roi_controls_enabled(enabled)

        bounds = self._get_roi_domain_bounds()
        if bounds is None:
            return
        q_min, q_max = bounds

        self._q_full_min, self._q_full_max = q_min, q_max
        if reset_to_domain or self._roi_min is None or self._roi_max is None or not enabled:
            self._roi_min, self._roi_max = q_min, q_max
        else:
            self._roi_min = max(q_min, min(float(self._roi_min), q_max))
            self._roi_max = max(self._roi_min, min(float(self._roi_max), q_max))

        self._updating_roi_controls = True
        try:
            control_min, control_max = self._roi_data_to_control_range(q_min, q_max)
            control_roi_min, control_roi_max = self._roi_data_to_control_values(self._roi_min, self._roi_max)
            if hasattr(self.ui, 'fitFittingRegionSlider'):
                s = self.ui.fitFittingRegionSlider
                s.setRangeF(control_min, control_max)
                s.setMinValueF(control_roi_min)
                s.setMaxValueF(control_roi_max)
            if hasattr(self.ui, 'fitFittingRegionMinValue'):
                self.ui.fitFittingRegionMinValue.setRange(control_min, control_max)
                self.ui.fitFittingRegionMinValue.setValue(control_roi_min)
            if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                self.ui.fitFittingRegionMaxValue.setRange(control_min, control_max)
                self.ui.fitFittingRegionMaxValue.setValue(control_roi_max)
        finally:
            self._updating_roi_controls = False

    # ---------------- ROI & Interpolation Controls -----------------
    def _setup_fitting_region_controls(self):
        """Wire up ROI slider/spinboxes and interpolation widgets.
        - Initialize defaults from user settings
        - Connect slider (live) and spinboxes (editingFinished)
        - Defer actual ROI initialization to first import/cut
        """
        # Slider
        if hasattr(self.ui, 'fitFittingRegionSlider'):
            try:
                self.ui.fitFittingRegionSlider.setDecimals(4)
            except Exception:
                pass
            try:
                self.ui.fitFittingRegionSlider.rangeChangedF.connect(self._on_roi_slider_changed)
            except Exception:
                if hasattr(self.ui.fitFittingRegionSlider, 'rangeChanged'):
                    self.ui.fitFittingRegionSlider.rangeChanged.connect(self._on_roi_slider_changed_int)

        # Min/Max spinboxes
        if hasattr(self.ui, 'fitFittingRegionMinValue'):
            try:
                self.ui.fitFittingRegionMinValue.setDecimals(4)
            except Exception:
                pass
            self.ui.fitFittingRegionMinValue.editingFinished.connect(self._on_roi_spin_finished)
        if hasattr(self.ui, 'fitFittingRegionMaxValue'):
            try:
                self.ui.fitFittingRegionMaxValue.setDecimals(4)
            except Exception:
                pass
            self.ui.fitFittingRegionMaxValue.editingFinished.connect(self._on_roi_spin_finished)

        # Points number
        if hasattr(self.ui, 'fitDataPointsNumValue'):
            try:
                self.ui.fitDataPointsNumValue.setRange(10, 5000)
                self.ui.fitDataPointsNumValue.setSingleStep(1)
                # ?????????????????? global_params/user_settings
                self.ui.fitDataPointsNumValue.setValue(int(max(10, self._points_num_current)))
            except Exception:
                pass
            # ??? meta ?????????????????????
            try:
                def _dp_after_commit(info, value):
                    try:
                        self._points_num_current = int(value)
                    except Exception:
                        self._points_num_current = int(self._points_num_default)
                    # ???????????????????
                    was_fitting = self._is_in_fitting_mode() if hasattr(self, '_is_in_fitting_mode') else False

                    # ????????????????????
                    if getattr(self, 'data_source', None) == 'cut':
                        self._perform_cut(points_override=int(self._points_num_current))
                    elif getattr(self, 'data_source', None) == '1d':
                        self._resample_1d(n_points=int(self._points_num_current))

                    # ????????????????????????????perform_manual_fitting?????????????????????
                    if was_fitting:
                        self._perform_manual_fitting()

                mode = self._signal_mode_overrides.get('fitDataPointsNumValue', self._default_signal_mode)
                self.param_trigger_manager.register_parameter_widget(
                    widget=self.ui.fitDataPointsNumValue,
                    widget_id='meta_fit_points_num',
                    category='fit_controls',
                    immediate_handler=lambda v: None,
                    delayed_handler=None,
                    connect_signals=True,
                    meta={
                        'persist': 'global_params',
                        'key_path': ('fitting', 'fit.points_num'),
                        'debounce_ms': 0,
                        'epsilon_abs': 0,
                        'epsilon_rel': 0,
                        'after_commit': _dp_after_commit,
                        'trigger_fit': False,
                        'connect_mode': mode,
                    }
                )
                # ??meta ????????connect_mode ?????
            except Exception:
                # ??????????
                self.ui.fitDataPointsNumValue.editingFinished.connect(self._on_points_num_finished)

        # Interpolation method
        if hasattr(self.ui, 'fitInterpolationMethodValue'):
            combo = self.ui.fitInterpolationMethodValue
            try:
                combo.clear(); combo.addItems(['Linear', 'Quadratic', 'Spline'])
                idx = combo.findText(self._interp_method_default)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            except Exception:
                pass
            combo.currentTextChanged.connect(self._on_interp_method_changed)

        # ROI min/max ??? editingFinished ???????????????????? editingFinished
        # ROI slider ?????? valueChanged??angeChangedF/int???????????? manager ?????

    def _initialize_roi_from_current_q(self, force_full: bool = False):
        """Initialize or refresh ROI bounds from current q/I arrays.

        - force_full=True resets ROI to full valid range regardless of previous ROI.
        - q_min/q_max are computed from pairs where both q and I are finite (exclude None/inf/NaN).
        """
        if self.q is None or self.I is None:
            return
        q_all = np.asarray(self.q)
        I_all = np.asarray(self.I)
        valid = np.isfinite(q_all) & np.isfinite(I_all)
        if not np.any(valid):
            return
        q_valid = q_all[valid]
        q_min, q_max = float(np.min(q_valid)), float(np.max(q_valid))
        # Update full range tracking to valid bounds
        prev_full = (self._q_full_min, self._q_full_max)
        self._q_full_min, self._q_full_max = q_min, q_max
        # Decide whether to reset ROI to full
        if force_full or self._roi_min is None or self._roi_max is None:
            self._roi_min, self._roi_max = q_min, q_max
        else:
            # If previous ROI is outside new bounds or full bounds changed notably, clamp or reset
            changed = (prev_full[0] is None or prev_full[1] is None or
                       abs(prev_full[0] - q_min) > 1e-12 or abs(prev_full[1] - q_max) > 1e-12)
            if changed:
                self._roi_min, self._roi_max = q_min, q_max
            else:
                # Clamp ROI into new bounds
                self._roi_min = max(q_min, min(self._roi_min, q_max))
                self._roi_max = max(self._roi_min, min(self._roi_max, q_max))
        # Update UI controls
        self._updating_roi_controls = True
        try:
            if hasattr(self.ui, 'fitFittingRegionSlider'):
                s = self.ui.fitFittingRegionSlider
                s.setRangeF(q_min, q_max)
                s.setMinValueF(self._roi_min)
                s.setMaxValueF(self._roi_max)
            if hasattr(self.ui, 'fitFittingRegionMinValue'):
                self.ui.fitFittingRegionMinValue.setRange(q_min, q_max)
                self.ui.fitFittingRegionMinValue.setValue(self._roi_min)
            if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                self.ui.fitFittingRegionMaxValue.setRange(q_min, q_max)
                self.ui.fitFittingRegionMaxValue.setValue(self._roi_max)
        finally:
            self._updating_roi_controls = False
        self._sync_roi_controls_to_current_display(reset_to_domain=force_full)

    def _on_roi_slider_changed_int(self, imin, imax):
        s = self.ui.fitFittingRegionSlider
        dec = 2
        try:
            dec = s.decimals()
        except Exception:
            pass
        scale = 10 ** dec
        self._on_roi_slider_changed(imin / scale, imax / scale)

    def _on_roi_slider_changed(self, vmin: float, vmax: float):
        if self._updating_roi_controls:
            return
        if not getattr(self, '_roi_controls_enabled', True):
            self._sync_roi_controls_to_current_display(reset_to_domain=True)
            return
        self._slider_is_source = True
        try:
            vmin = self._nearest_roi_control_value(float(vmin))
            vmax = self._nearest_roi_control_value(float(vmax))
            control_min, control_max = self._roi_data_to_control_range(
                self._q_full_min if self._q_full_min is not None else vmin,
                self._q_full_max if self._q_full_max is not None else vmax,
            )
            vmin = max(control_min, min(vmin, vmax))
            vmax = min(control_max, max(vmax, vmin))
            self._roi_min, self._roi_max = self._roi_control_to_data_values(vmin, vmax)
            # Update spinboxes
            self._updating_roi_controls = True
            if hasattr(self.ui, 'fitFittingRegionMinValue'):
                self.ui.fitFittingRegionMinValue.setValue(vmin)
            if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                self.ui.fitFittingRegionMaxValue.setValue(vmax)
        finally:
            self._updating_roi_controls = False
            self._slider_is_source = False
            # ?????????????????????????????????????????????????
            self._schedule_roi_refresh()

    def _on_roi_spin_finished(self):
        if self._updating_roi_controls:
            return
        if not getattr(self, '_roi_controls_enabled', True):
            self._sync_roi_controls_to_current_display(reset_to_domain=True)
            return
        vmin = float(self.ui.fitFittingRegionMinValue.value()) if hasattr(self.ui, 'fitFittingRegionMinValue') else self._roi_min
        vmax = float(self.ui.fitFittingRegionMaxValue.value()) if hasattr(self.ui, 'fitFittingRegionMaxValue') else self._roi_max
        control_min, control_max = self._roi_data_to_control_range(
            self._q_full_min if self._q_full_min is not None else vmin,
            self._q_full_max if self._q_full_max is not None else vmax,
        )
        vmin = max(control_min, vmin)
        vmax = min(control_max, vmax)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        self._roi_min, self._roi_max = self._roi_control_to_data_values(vmin, vmax)
        # Update slider
        self._updating_roi_controls = True
        try:
            if hasattr(self.ui, 'fitFittingRegionSlider'):
                s = self.ui.fitFittingRegionSlider
                if self._q_full_min is not None and self._q_full_max is not None:
                    s.setRangeF(control_min, control_max)
                s.setMinValueF(vmin)
                s.setMaxValueF(vmax)
        finally:
            self._updating_roi_controls = False
            # ?????????????????????????????
            self._apply_roi_to_data_and_refresh()

    def _schedule_roi_refresh(self):
        """No description."""
        try:
            from PyQt5.QtCore import QTimer
            if self._roi_update_timer is None:
                self._roi_update_timer = QTimer()
                self._roi_update_timer.setSingleShot(True)
                self._roi_update_timer.timeout.connect(self._apply_roi_to_data_and_refresh)
            # ?????????????????????????
            delay = int(getattr(self, '_roi_debounce_ms', 140))
            self._roi_update_timer.start(max(20, delay))
        except Exception:
            # ?????????????????????
            self._apply_roi_to_data_and_refresh()

    def _apply_roi_to_data_and_refresh(self):
        if self.q is None or self.I is None:
            return
        self._sync_roi_controls_to_current_display(reset_to_domain=False)
        q = np.asarray(self.q); I = np.asarray(self.I)
        # Always drop non-finite pairs before ROI masking
        valid = np.isfinite(q) & np.isfinite(I)
        q = q[valid]; I = I[valid]
        if not getattr(self, '_roi_controls_enabled', True) or self._roi_min is None or self._roi_max is None:
            self.q_ROI, self.I_ROI = q, I
        else:
            mask = (q >= self._roi_min) & (q <= self._roi_max)
            if not np.any(mask):
                self.q_ROI, self.I_ROI = q, I
            else:
                self.q_ROI = q[mask]
                self.I_ROI = I[mask]
        # Redraw displays
        try:
            # ????????????????????????????????????ormal???????????Fitting
            current_mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            if current_mode == 'fitting' or (hasattr(self, '_is_in_fitting_mode') and self._is_in_fitting_mode()):
                self._update_gui_fitting_display()
                current_mode = 'fitting'
            # ??????????????????????
            self._update_GUI_image(current_mode)
            self._update_outside_window(current_mode)
        except Exception:
            pass

    # ---------------- ROI bounds adjustment for log-x -----------------
    def _get_current_fit_axes(self):
        """Try to get the current Matplotlib Axes used by the in-GUI 1D plot."""
        try:
            fig = getattr(self, '_current_fit_figure', None)
            if fig is None:
                return None
            axes = getattr(fig, 'axes', None)
            if not axes:
                return None
            return axes[0] if len(axes) > 0 else None
        except Exception:
            return None

    def _compute_display_xmin_for_log(self) -> float:
        """Compute a safe lower X bound to use when log-x is enabled.

        Priority:
        1) If current Axes exist, use its left xlim (must be > 0).
        2) Else, use min positive finite q from current data (full q preferred).
        3) Fallback to a tiny positive epsilon.
        """
        # 1) Try current axes xlim
        try:
            ax = self._get_current_fit_axes()
            if ax is not None:
                x0, _ = ax.get_xlim()
                if np.isfinite(x0) and x0 > 0:
                    return float(x0)
        except Exception:
            pass
        # 2) Use data-based min positive q
        try:
            q_all = None
            if self.q is not None:
                q_all = np.asarray(self.q)
            elif self.current_cut_data is not None and 'x_coords' in self.current_cut_data:
                q_all = np.asarray(self.current_cut_data['x_coords'])
            elif self.current_1d_data is not None and 'q' in self.current_1d_data:
                q_all = np.asarray(self.current_1d_data['q'])
            if q_all is not None and q_all.size > 0:
                pos = q_all[np.isfinite(q_all) & (q_all > 0)]
                if pos.size > 0:
                    return float(np.min(pos))
        except Exception:
            pass
        # 3) Fallback
        return 1e-12

    def _adjust_roi_bounds_for_log_x(self):
        """When log-x is enabled, ensure ROI slider/spin ranges start from the displayed x-axis min (>0).

        Also clamp current ROI values to the new bounds.
        """
        try:
            self._sync_roi_controls_to_current_display(reset_to_domain=False)
            return
        except Exception:
            pass

        try:
            log_x = self._is_fit_log_x_enabled()
        except Exception:
            log_x = False

        if not hasattr(self.ui, 'fitFittingRegionSlider'):
            return

        try:
            s = self.ui.fitFittingRegionSlider
            # Determine full max from existing tracking or data
            q_max = None
            if self._q_full_max is not None:
                q_max = float(self._q_full_max)
            else:
                try:
                    q_all = np.asarray(self.q) if self.q is not None else None
                    if q_all is not None and q_all.size > 0:
                        q_max = float(np.nanmax(q_all[np.isfinite(q_all)]))
                except Exception:
                    q_max = None
            if q_max is None:
                return

            if log_x:
                xmin = self._compute_display_xmin_for_log()
                # Ensure xmin < q_max
                if not np.isfinite(xmin) or xmin <= 0 or xmin >= q_max:
                    xmin = min(max(1e-12, (q_max * 1e-6)), q_max * 0.5)  # conservative fallback
                new_min, new_max = float(xmin), float(q_max)
            else:
                # Restore to full valid data bounds if known
                q_min = float(self._q_full_min) if self._q_full_min is not None else None
                if q_min is None:
                    try:
                        q_all = np.asarray(self.q) if self.q is not None else None
                        if q_all is not None and q_all.size > 0:
                            q_min = float(np.nanmin(q_all[np.isfinite(q_all)]))
                    except Exception:
                        q_min = None
                if q_min is None:
                    return
                new_min, new_max = float(q_min), float(q_max)

            # Update control ranges and clamp current ROI values
            self._updating_roi_controls = True
            try:
                s.setRangeF(new_min, new_max)
                # Clamp current ROI values
                if self._roi_min is None or self._roi_max is None:
                    cur_min, cur_max = new_min, new_max
                else:
                    cur_min = max(new_min, min(float(self._roi_min), new_max))
                    cur_max = max(cur_min, min(float(self._roi_max), new_max))
                self._roi_min, self._roi_max = cur_min, cur_max
                s.setMinValueF(cur_min)
                s.setMaxValueF(cur_max)
                # Update spinbox ranges to match
                if hasattr(self.ui, 'fitFittingRegionMinValue'):
                    self.ui.fitFittingRegionMinValue.setRange(new_min, new_max)
                    self.ui.fitFittingRegionMinValue.setValue(cur_min)
                if hasattr(self.ui, 'fitFittingRegionMaxValue'):
                    self.ui.fitFittingRegionMaxValue.setRange(new_min, new_max)
                    self.ui.fitFittingRegionMaxValue.setValue(cur_max)
            finally:
                self._updating_roi_controls = False
        except Exception:
            pass

    # ---------------- Interpolation -----------------
    def _on_points_num_finished(self):
        n = None
        try:
            if hasattr(self.ui, 'fitDataPointsNumValue'):
                n = int(self.ui.fitDataPointsNumValue.value())
        except Exception:
            n = None
        if n is None:
            return
        if n < 10:
            n = 10
        # Keep a stable in-controller cache for repeated cuts
        try:
            self._points_num_current = int(n)
        except Exception:
            self._points_num_current = int(self._points_num_default)
        # Persist
        try:
            from core.user_settings import user_settings
            user_settings.set('fit.points_num', int(n)); user_settings.save_settings()
        except Exception:
            pass
        # ???????????????????
        was_fitting = self._is_in_fitting_mode() if hasattr(self, '_is_in_fitting_mode') else False

        # Apply ??????/?????
        if getattr(self, 'data_source', None) == 'cut':
            self._perform_cut(points_override=n)
        elif getattr(self, 'data_source', None) == '1d':
            self._resample_1d(n_points=n)

        # ???????????????????????????_perform_manual_fitting???????????????
        if was_fitting:
            self._perform_manual_fitting()

    def _on_interp_method_changed(self, method: str):
        meth = method or 'Linear'
        try:
            from core.user_settings import user_settings
            user_settings.set('fit.interp_method', meth); user_settings.save_settings()
        except Exception:
            pass
        if self.data_source == '1d' and self.q is not None:
            self._resample_1d(n_points=len(self.q), method=meth, keep_same_count=True)
        elif self.data_source == 'cut':
            self._perform_cut()

    def _resample_1d(self, n_points: int, method: str = None, keep_same_count: bool = False):
        if self.current_1d_data is None or self.q is None or self.I is None:
            return
        q0 = np.asarray(self.current_1d_data.get('q', self.q))
        I0 = np.asarray(self.current_1d_data.get('I', self.I))
        if q0.size < 2:
            return
        method = method or (self.ui.fitInterpolationMethodValue.currentText() if hasattr(self.ui, 'fitInterpolationMethodValue') else 'Linear')
        if keep_same_count:
            n_points = len(self.q)
        # Fallback to stable current points if not valid
        try:
            n_points = int(max(10, n_points))
        except Exception:
            n_points = int(max(10, getattr(self, '_points_num_current', self._points_num_default)))
        q_new = np.linspace(q0.min(), q0.max(), int(max(10, n_points)))
        I_new = self._interpolate_series(q0, I0, q_new, method)
        self.q, self.I = q_new, I_new
        if self._q_full_min is None or self._q_full_max is None:
            self._initialize_roi_from_current_q()
        self._apply_roi_to_data_and_refresh()

    def _interpolate_series(self, x, y, x_new, method: str):
        m = (method or 'Linear').lower()
        if m == 'linear':
            return np.interp(x_new, x, y)
        if m == 'quadratic':
            try:
                from scipy.interpolate import interp1d
                f = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                return f(x_new)
            except Exception:
                coeff = np.polyfit(x, y, deg=min(2, max(1, len(x)-1)))
                return np.polyval(coeff, x_new)
        if m == 'spline':
            try:
                from scipy.interpolate import CubicSpline
                return CubicSpline(x, y, extrapolate=True)(x_new)
            except Exception:
                return np.interp(x_new, x, y)
        return np.interp(x_new, x, y)

    def _setup_connections(self):
        """No description."""
        # ???GISAXS?????????
        if hasattr(self.ui, 'gisaxsInputImportButton'):
            self.ui.gisaxsInputImportButton.clicked.connect(self._import_gisaxs_file)

        if self._previous_image_button is not None:
            self._previous_image_button.clicked.connect(self._show_previous_folder_image)
        if self._next_image_button is not None:
            self._next_image_button.clicked.connect(self._show_next_folder_image)

        # ?????????????????????
        if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
            self.ui.gisaxsInputImportButtonValue.returnPressed.connect(self._on_import_value_changed)

        # ???Stack?????????????????
        if hasattr(self.ui, 'gisaxsInputStackValue'):
            self.ui.gisaxsInputStackValue.returnPressed.connect(self._on_stack_value_changed)

        # ???Show???
        if hasattr(self.ui, 'gisaxsInputShowButton'):
            self.ui.gisaxsInputShowButton.clicked.connect(self._show_image)

        # ???AutoShow????
        if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox'):
            self.ui.gisaxsInputAutoShowCheckBox.toggled.connect(self._on_auto_show_changed)

        # ??????????????
        if hasattr(self.ui, 'gisaxsInputModelCombox'):
            try:
                self.ui.gisaxsInputModelCombox.currentTextChanged.connect(self._on_load_mode_changed)
            except Exception:
                pass

        # ???Log????
        if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
            self.ui.gisaxsInputIntLogCheckBox.toggled.connect(self._on_log_changed)

        # ???AutoScale????
        if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
            self.ui.gisaxsInputAutoScaleCheckBox.toggled.connect(self._on_auto_scale_changed)

        # ???Q?????????
        if hasattr(self.ui, 'gisaxsInputDisplayModeQ'):
            self.ui.gisaxsInputDisplayModeQ.toggled.connect(self._on_q_mode_changed)
        if hasattr(self.ui, 'gisaxsInputDisplayModePixel'):
            self.ui.gisaxsInputDisplayModePixel.toggled.connect(self._on_q_mode_changed)

        if hasattr(self.ui, 'gisaxsInputVminValue'):
            self.ui.gisaxsInputVminValue.editingFinished.connect(self._on_color_scale_value_committed)
        if hasattr(self.ui, 'gisaxsInputVmaxValue'):
            self.ui.gisaxsInputVmaxValue.editingFinished.connect(self._on_color_scale_value_committed)

        # Vmin/Vmax?????????????????????
        # ??_connect_cutline_parameter_signals() ???

        # ???AutoFinding???
        if hasattr(self.ui, 'gisaxsInputCenterAutoFindingButton'):
            self.ui.gisaxsInputCenterAutoFindingButton.clicked.connect(self._auto_find_center)

        # ???Cut???
        if hasattr(self.ui, 'gisaxsInputCutButton'):
            self.ui.gisaxsInputCutButton.clicked.connect(lambda _checked=False: self._perform_cut())

        # ???Detector Parameters???
        if hasattr(self.ui, 'gisaxsInputDetectorParaButton'):
            self.ui.gisaxsInputDetectorParaButton.clicked.connect(self._show_detector_parameters)

        # ???GraphicsView??????
        if hasattr(self.ui, 'gisaxsInputGraphicsView'):
            self.ui.gisaxsInputGraphicsView.setToolTip("Double-click to open a larger independent image window.")
            self.ui.gisaxsInputGraphicsView.mouseDoubleClickEvent = self._on_graphics_view_double_click

        # ???fitGraphicsView??????
        if hasattr(self.ui, 'fitGraphicsView'):
            self.ui.fitGraphicsView.setToolTip("Double-click to open a larger independent fit window.")
            self.ui.fitGraphicsView.mouseDoubleClickEvent = self._on_fit_graphics_view_double_click

        # ????????????????I????????
        if hasattr(self.ui, 'fitStartButton'):
            self.ui.fitStartButton.clicked.connect(self._start_fitting)

        # ???Clear Fitting???
        if hasattr(self.ui, 'FittingClearFittingButton_2'):
            self.ui.FittingClearFittingButton_2.clicked.connect(self._clear_fitting_data)

        # ?????????log???????
        if hasattr(self.ui, 'fitLogXCheckBox'):
            self.ui.fitLogXCheckBox.toggled.connect(self._on_fit_log_changed)
        if hasattr(self.ui, 'fitLogYCheckBox'):
            self.ui.fitLogYCheckBox.toggled.connect(self._on_fit_log_changed)

        # ????????????????????????????????????
        for _name in ['fitBGShowCheckBox', 'fitResShowCheckBox']:
            if hasattr(self.ui, _name):
                try:
                    getattr(self.ui, _name).toggled.connect(self._on_component_checkbox_changed)
                except Exception:
                    pass

        # ???Normalize????
        if hasattr(self.ui, 'OthersNormalizeCheckBox'):
            self.ui.OthersNormalizeCheckBox.toggled.connect(self._on_normalize_changed)
        if hasattr(self.ui, 'fitNormCheckBox'):
            self.ui.fitNormCheckBox.toggled.connect(self._on_normalize_changed)

        # ????????? Positive Only ?????????????????
        if hasattr(self.ui, 'PositiveOnlyCheckBox'):
            self.ui.PositiveOnlyCheckBox.toggled.connect(self._on_positive_only_changed)
        if hasattr(self.ui, 'fitRegionPositiveOnlyCheckBox'):
            self.ui.fitRegionPositiveOnlyCheckBox.toggled.connect(self._on_positive_only_changed)
        if hasattr(self.ui, 'fitRegionNegativeOnlyCheckBox'):
            self.ui.fitRegionNegativeOnlyCheckBox.toggled.connect(self._on_positive_only_changed)

        if hasattr(self.ui, 'fitResetButton'):
            self.ui.fitResetButton.clicked.connect(self._reset_fitting)

        # ???fitImport1dFileButton???
        if hasattr(self.ui, 'fitImport1dFileButton'):
            self.ui.fitImport1dFileButton.clicked.connect(self._import_1d_file)

        # ???fitImport1dFileValue????????????
        if hasattr(self.ui, 'fitImport1dFileValue'):
            self.ui.fitImport1dFileValue.returnPressed.connect(self._on_1d_file_value_changed)

        # ???FittingExportButton???
        if hasattr(self.ui, 'FittingExportButton'):
            self.ui.FittingExportButton.clicked.connect(self._export_fitting_data)

        # ???FittingManualFittingButton???
        if hasattr(self.ui, 'FittingManualFittingButton'):
            self.ui.FittingManualFittingButton.clicked.connect(self._perform_manual_fitting)

        if hasattr(self.ui, 'FittingAutoRefineButton'):
            self.ui.FittingAutoRefineButton.clicked.connect(self._show_manual_auto_refine_dialog)

        if hasattr(self.ui, 'FittingAutoFittingButton'):
            self.ui.FittingAutoFittingButton.clicked.connect(self.open_ai_fitting_workspace)
        if hasattr(self.ui, 'aiFittingRefreshButton'):
            self.ui.aiFittingRefreshButton.clicked.connect(self._refresh_ai_fitting_models)
        if hasattr(self.ui, 'aiFittingOpenWorkspaceButton'):
            self.ui.aiFittingOpenWorkspaceButton.clicked.connect(self.open_ai_fitting_workspace)
        if hasattr(self.ui, 'aiFittingExportOutputButton'):
            self.ui.aiFittingExportOutputButton.clicked.connect(self._export_ai_prediction_output)
        if hasattr(self.ui, 'aiFittingModelComboBox'):
            self.ui.aiFittingModelComboBox.currentIndexChanged.connect(self._on_ai_model_selected)
        if hasattr(self.ui, 'aiFittingConstraintComboBox'):
            self.ui.aiFittingConstraintComboBox.currentTextChanged.connect(self._on_ai_constraint_mode_changed)
        if hasattr(self.ui, 'aiFittingFixedKComboBox'):
            self.ui.aiFittingFixedKComboBox.currentTextChanged.connect(
                lambda text: self._on_ai_fixed_k_changed(text)
            )
        if hasattr(self.ui, 'aiFittingCombinationButton'):
            self.ui.aiFittingCombinationButton.clicked.connect(self._show_ai_fixed_combination_dialog)
        if hasattr(self.ui, 'aiFittingFastPredictButton'):
            self.ui.aiFittingFastPredictButton.clicked.connect(lambda: self._start_ai_prediction("fast"))
        if hasattr(self.ui, 'aiFittingFullAutoFitButton'):
            self.ui.aiFittingFullAutoFitButton.clicked.connect(lambda: self._start_ai_prediction("full"))
        if hasattr(self.ui, 'aiFittingStopButton'):
            self.ui.aiFittingStopButton.clicked.connect(self._stop_ai_fitting_process)
        if hasattr(self.ui, 'aiFittingAdvancedConstraintsButton'):
            self.ui.aiFittingAdvancedConstraintsButton.clicked.connect(self._show_advanced_constraints_dialog)
        self._connect_ai_fitting_settings_widgets()

        # ???FittingAutoKButton???
        if hasattr(self.ui, 'FittingAutoKButton'):
            self.ui.FittingAutoKButton.clicked.connect(self._on_auto_k_button_clicked)

        # ????????????????
        if hasattr(self.ui, 'fitCurrentDataCheckBox'):
            self.ui.fitCurrentDataCheckBox.toggled.connect(self._on_current_data_checkbox_changed)

        # ???Cut Line??enter????????????????????
        self._connect_cutline_parameter_signals(
            mode=self._default_signal_mode,
            overrides=self._signal_mode_overrides,
        )

        # ??????????????????????????
        self._connect_parameter_widgets()

        # ???FittingTextBrowser????????
        self._setup_fitting_text_browser()
        self._setup_fitting_parameters_context_menu()
        self._refresh_ai_fitting_models()
        self._restore_main_ai_settings()

    def _connect_cutline_parameter_signals(self, mode: str = 'changed', overrides: dict = None):
        """Cut Line/Center/Vmin/Vmax?????meta ??? & ?????? global_params
        ???:
            mode: 'changed' ??'finished'??
                  - 'changed': ??? valueChanged/textChanged/currentTextChanged ??? meta ??????
                  - 'finished': editingFinished/returnPressed ????????????
            overrides: ???ict?????????????????????????('changed'|'finished')?????????? mode??
        ????????????ode='changed'??
        """
        from functools import partial
        mapping = [
            ('gisaxsInputCenterVerticalValue', 'center_vertical'),
            ('gisaxsInputCenterParallelValue', 'center_parallel'),
            ('gisaxsInputCutLineVerticalValue', 'cutline_vertical'),
            ('gisaxsInputCutLineParallelValue', 'cutline_parallel'),
            ('gisaxsInputVminValue', 'vmin'),
            ('gisaxsInputVmaxValue', 'vmax'),
        ]
        overrides = overrides or {}
        for widget_name, param_key in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            w = getattr(self.ui, widget_name)
            def _after_commit(info, value, p=param_key):
                try:
                    if p in ('vmin', 'vmax'):
                        self._on_color_scale_value_committed()
                        self._add_fitting_message(f"Meta commit GISAXS {p} = {value}", "INFO")
                        return
                    self._on_parameter_display_changed()
                    self._add_fitting_message(f"Meta commit GISAXS {p} = {value}", "INFO")
                except Exception:
                    pass
            widget_mode = overrides.get(widget_name, mode)
            meta = {
                'persist': 'global_params',
                'key_path': ('fitting', f'gisaxs_input.{param_key}'),
                'trigger_fit': False,
                'debounce_ms': self._param_debounce_ms,
                'epsilon_abs': self._param_abs_eps,
                'epsilon_rel': self._param_rel_eps,
                'after_commit': _after_commit,
                'connect_mode': widget_mode,
            }
            self.param_trigger_manager.register_parameter_widget(
                widget=w,
                widget_id=f'meta_gisaxs_{param_key}',
                category='gisaxs_input',
                immediate_handler=lambda v: None,
                delayed_handler=None,
                connect_signals=True,
                meta=meta
            )
            # ??meta ????????connect_mode ????????????????

    def _connect_parameter_widgets(self):
        """No description."""
        # ????????UI????????????????
        # ????????????????????

        # ?????????????????????????I?????????
        self._setup_particle_connections()

    def _initialize_ui(self):
        """No description."""
        # ??????????????
        if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
            self.ui.gisaxsInputImportButtonValue.clear()

        # ???Stack????
        if hasattr(self.ui, 'gisaxsInputStackValue'):
            self.ui.gisaxsInputStackValue.setText("1")

        # ???stack?????
        if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
            self.ui.gisaxsInputStackDisplayLabel.setText("")

        # ????????????????????
        if hasattr(self.ui, 'gisaxsInputModelCombox'):
            try:
                from core.global_params import GlobalParameterManager
                gp = GlobalParameterManager()
                last_mode = gp.get_parameter('fitting', 'gisaxs_input.load_mode', 'Single')
                idx = self.ui.gisaxsInputModelCombox.findText(last_mode)
                self.ui.gisaxsInputModelCombox.setCurrentIndex(idx if idx >= 0 else 0)
                self.load_mode = self.ui.gisaxsInputModelCombox.currentText()
            except Exception:
                self.load_mode = 'Single'
            # ???????UI?????????????In-situ???????????? Stack ??????????????LineEdit ?????
            try:
                if self.load_mode == 'In-situ' and hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setVisible(False)
            except Exception:
                pass
            # ?????????Stack??????????
            self._update_stack_controls_visibility()

        # ???Log?????????
        if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
            self.ui.gisaxsInputIntLogCheckBox.setChecked(True)

        # ???AutoScale?????????
        if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox'):
            self.ui.gisaxsInputAutoScaleCheckBox.setChecked(True)

        # ????????????????????????????????????
        self._initialize_fit_checkboxes()

        # ????min/Vmax???0??????????????????????
        if hasattr(self.ui, 'gisaxsInputVminValue'):
            self.ui.gisaxsInputVminValue.setValue(0.0)
            # ???????????????????????????
            self.ui.gisaxsInputVminValue.setDecimals(6)
            self.ui.gisaxsInputVminValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVminValue.setSingleStep(0.1)
            self.ui.gisaxsInputVminValue.setKeyboardTracking(True)
            self._setup_smart_display(self.ui.gisaxsInputVminValue)

        if hasattr(self.ui, 'gisaxsInputVmaxValue'):
            self.ui.gisaxsInputVmaxValue.setValue(0.0)
            # ???????????????????????????
            self.ui.gisaxsInputVmaxValue.setDecimals(6)
            self.ui.gisaxsInputVmaxValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVmaxValue.setSingleStep(0.1)
            self.ui.gisaxsInputVmaxValue.setKeyboardTracking(True)
            self._setup_smart_display(self.ui.gisaxsInputVmaxValue)

        # ???Cut Line Center?????????????????????????
        if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
            self.ui.gisaxsInputCenterVerticalValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCenterVerticalValue.setDecimals(2)
            self.ui.gisaxsInputCenterVerticalValue.setValue(0.0)
            self.ui.gisaxsInputCenterVerticalValue.setKeyboardTracking(True)

        if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
            self.ui.gisaxsInputCenterParallelValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCenterParallelValue.setDecimals(2)
            self.ui.gisaxsInputCenterParallelValue.setValue(0.0)
            self.ui.gisaxsInputCenterParallelValue.setKeyboardTracking(True)

        # ???Cut Line Vertical/Parallel?????????????????????????
        if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
            self.ui.gisaxsInputCutLineVerticalValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCutLineVerticalValue.setDecimals(2)
            self.ui.gisaxsInputCutLineVerticalValue.setValue(10.0)
            self.ui.gisaxsInputCutLineVerticalValue.setKeyboardTracking(True)

        if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
            self.ui.gisaxsInputCutLineParallelValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCutLineParallelValue.setDecimals(2)
            self.ui.gisaxsInputCutLineParallelValue.setValue(10.0)
            self.ui.gisaxsInputCutLineParallelValue.setKeyboardTracking(True)

        # ???Cut Line??????????????????????
        self._update_cutline_step_sizes()

        # ????????????????????
        if hasattr(self, '_on_q_mode_changed'):
            # ??????????I???????
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self._update_cutline_step_sizes)

        # ????????
        self._set_default_parameters()

        # ????ut Line???????
        self._update_cutline_labels_units()

        # ??????????????????????????????
        self._initialize_q_mode_state()

        # ????????
        self._check_dependencies()

        # ??In-situ ????????????
        try:
            if getattr(self, 'load_mode', 'Single') == 'In-situ' and self._is_auto_show_enabled():
                self._start_insitu_timer()
            # ?????????????????????????????????????????????????
            self._enforce_insitu_visibility_once()
        except Exception:
            pass

    def _initialize_fit_checkboxes(self):
        """No description."""
        try:
            # ????itCurrentDataCheckBox???????????
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(False)
                self.ui.fitCurrentDataCheckBox.blockSignals(False)

            # ????itLogXCheckBox???????????
            if hasattr(self.ui, 'fitLogXCheckBox'):
                self.ui.fitLogXCheckBox.blockSignals(True)
                self.ui.fitLogXCheckBox.setChecked(False)
                self.ui.fitLogXCheckBox.blockSignals(False)

            # ????itLogYCheckBox???????????
            if hasattr(self.ui, 'fitLogYCheckBox'):
                self.ui.fitLogYCheckBox.blockSignals(True)
                self.ui.fitLogYCheckBox.setChecked(False)
                self.ui.fitLogYCheckBox.blockSignals(False)

            # ????itNormCheckBox???????????
            if hasattr(self.ui, 'fitNormCheckBox'):
                self.ui.fitNormCheckBox.blockSignals(True)
                self.ui.fitNormCheckBox.setChecked(False)
                self.ui.fitNormCheckBox.blockSignals(False)

        except Exception as e:
            pass

    def _restore_fit_checkboxes(self, session_data):
        """No description."""
        try:
            # ??fitCurrentDataCheckBox
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(session_data.get('fit_current_data', False))
                self.ui.fitCurrentDataCheckBox.blockSignals(False)

            # ??fitLogXCheckBox
            if hasattr(self.ui, 'fitLogXCheckBox'):
                self.ui.fitLogXCheckBox.blockSignals(True)
                self.ui.fitLogXCheckBox.setChecked(session_data.get('fit_log_x', False))
                self.ui.fitLogXCheckBox.blockSignals(False)

            # ??fitLogYCheckBox
            if hasattr(self.ui, 'fitLogYCheckBox'):
                self.ui.fitLogYCheckBox.blockSignals(True)
                self.ui.fitLogYCheckBox.setChecked(session_data.get('fit_log_y', False))
                self.ui.fitLogYCheckBox.blockSignals(False)

            # ??fitNormCheckBox
            if hasattr(self.ui, 'fitNormCheckBox'):
                self.ui.fitNormCheckBox.blockSignals(True)
                self.ui.fitNormCheckBox.setChecked(session_data.get('fit_norm', False))
                self.ui.fitNormCheckBox.blockSignals(False)

        except Exception as e:
            pass

    def _initialize_q_mode_state(self):
        """No description."""
        try:
            # ??????Q????????????????????
            current_q_mode = self._should_show_q_axis()
            self._last_q_mode = current_q_mode
        except Exception as e:
            # ???????????????????????????
            self._last_q_mode = False

    def _setup_smart_display(self, spinbox):
        """No description."""
        try:
            # ?????????????????????
            spinbox.valueChanged.connect(lambda value: self._update_spinbox_format(spinbox, value))
            spinbox.editingFinished.connect(lambda: self._update_spinbox_format(spinbox, spinbox.value()))
            self._update_spinbox_format(spinbox, spinbox.value())
        except Exception:
            spinbox.setDecimals(2)

    def _update_spinbox_format(self, spinbox, value):
        """No description."""
        try:
            is_log_mode = False
            if hasattr(self.ui, 'gisaxsInputIntLogCheckBox'):
                is_log_mode = self.ui.gisaxsInputIntLogCheckBox.isChecked()

            if is_log_mode:
                spinbox.setDecimals(2)
            else:
                if abs(value - round(value)) < 1e-9:
                    spinbox.setDecimals(0)
                else:
                    value_str = f"{value:.6f}".rstrip('0').rstrip('.')
                    if '.' in value_str:
                        decimal_places = len(value_str.split('.')[1])
                        decimal_places = min(decimal_places, 6)
                        decimal_places = max(decimal_places, 1)
                        spinbox.setDecimals(decimal_places)
                    else:
                        spinbox.setDecimals(0)
        except Exception:
            try:
                if hasattr(self.ui, 'gisaxsInputIntLogCheckBox') and self.ui.gisaxsInputIntLogCheckBox.isChecked():
                    spinbox.setDecimals(2)
                else:
                    spinbox.setDecimals(0)
            except:
                spinbox.setDecimals(2)

    def _refresh_vmin_vmax_display(self):
        """No description."""
        try:
            if hasattr(self.ui, 'gisaxsInputVminValue'):
                self._update_spinbox_format(self.ui.gisaxsInputVminValue, self.ui.gisaxsInputVminValue.value())
            if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                self._update_spinbox_format(self.ui.gisaxsInputVmaxValue, self.ui.gisaxsInputVmaxValue.value())
        except Exception:
            pass

    def _check_dependencies(self):
        """No description."""
        if not is_fabio_available():
            self.status_updated.emit("Warning: fabio library not available. CBF processing will be disabled.")
        if not is_matplotlib_available():
            self.status_updated.emit("Warning: matplotlib not available. Image display will be disabled.")

    def _is_q_space_mode(self):
        """Q-space"""
        try:
            # ?????should_show_q_axis()????????
            return self._should_show_q_axis()
        except Exception:
            return False

    def _delayed_cut_update(self):
        """No description."""
        try:
            # ????????ut???????????
            if hasattr(self, '_cut_data') and self._cut_data is not None:
                # ?????Cut???????????
                self._execute_cut()
        except Exception as e:
            pass

    def _on_parameter_display_changed(self):
        """No description."""
        try:
            # ???????????
            if getattr(self, '_initializing', False):
                return
            # ?????????/??????????????
            if hasattr(self, '_update_stack_display'):
                self._update_stack_display()

            # ????????????????????Cut???
            center_x = 0
            center_y = 0
            width = 0
            height = 0

            # ???????????
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                width = self.ui.gisaxsInputCutLineParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                height = self.ui.gisaxsInputCutLineVerticalValue.value()

            # ?????????????????????
            if width > 0 and height > 0:
                selection_info = self._create_selection_from_parameters(center_x, center_y, width, height)
                self._update_parameter_selection_display(selection_info)

            # ??????????????????????
            self._trigger_delayed_cut_update()

        except Exception as e:
            pass

    def _trigger_delayed_cut_update(self):
        """ut"""
        try:
            if not hasattr(self, '_cut_update_timer'):
                from PyQt5.QtCore import QTimer
                self._cut_update_timer = QTimer()
                self._cut_update_timer.setSingleShot(True)
                self._cut_update_timer.timeout.connect(self._delayed_cut_image_update)

            # ?????????????
            self._cut_update_timer.stop()
            self._cut_update_timer.start(300)  # 300ms?????????

        except Exception as e:
            pass

    def _delayed_cut_image_update(self):
        """Cut"""
        try:
            # ???????????Cut???????????????????ut???
            if (self.current_cut_data is not None and
                hasattr(self, 'current_stack_data') and self.current_stack_data is not None):

                # ?????????
                center_x = 0
                center_y = 0
                width = 0
                height = 0

                if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                    center_x = self.ui.gisaxsInputCenterParallelValue.value()
                if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                    center_y = self.ui.gisaxsInputCenterVerticalValue.value()
                if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                    width = self.ui.gisaxsInputCutLineParallelValue.value()
                if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                    height = self.ui.gisaxsInputCutLineVerticalValue.value()

                # ?????Cut???
                self._perform_cut()
                self.status_updated.emit(f"Auto-updated cut with new parameters: Center({center_x}, {center_y}), Size({width} x {height})")

        except Exception as e:
            pass

    def _on_cutline_parameters_immediate_update(self):
        """No description."""
        try:
            # ??????????????????
            if hasattr(self, '_cut_update_timer'):
                self._cut_update_timer.stop()

            # ???????????
            self._delayed_cut_image_update()

        except Exception as e:
            pass

    def _update_cutline_step_sizes(self):
        """No description."""
        try:
            # ??????????
            is_q_mode = self._is_q_space_mode()

            # ????????????
            if is_q_mode:
                # Q-space????????.01???
                step_size = 0.01
            else:
                # Pixel????????.0???
                step_size = 1.0

            # ???????????ut Line??????
            cutline_controls = [
                'gisaxsInputCenterVerticalValue',
                'gisaxsInputCenterParallelValue',
                'gisaxsInputCutLineVerticalValue',
                'gisaxsInputCutLineParallelValue'
            ]

            for control_name in cutline_controls:
                if hasattr(self.ui, control_name):
                    control = getattr(self.ui, control_name)
                    control.setSingleStep(step_size)

            cutline_step_controls = [
                'gisaxsInputCutLineVerticalStep',
                'gisaxsInputCutLineParallelStep',
                'gisaxsInputCenterVerticalStep',
                'gisaxsInputCenterParallelStep',
            ]
            for control_name in cutline_step_controls:
                if hasattr(self.ui, control_name):
                    control = getattr(self.ui, control_name)
                    control.setProperty('defaultStepValue', step_size)
                    control.blockSignals(True)
                    control.setSingleStep(step_size)
                    control.setValue(step_size)
                    control.blockSignals(False)

            self.status_updated.emit(f"Cut Line step size updated to {step_size} ({'Q-space' if is_q_mode else 'Pixel'} mode)")

        except Exception as e:
            self.status_updated.emit(f"Error updating cut line step sizes: {str(e)}")

    def _update_cutline_labels_units(self):
        """No description."""
        try:
            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                # Q???????????(nm??? ???
                unit_suffix = " (q)"
            else:
                # ??????????????(pixel) ???
                unit_suffix = " (px)"

            # ???Center??
            if hasattr(self.ui, 'gisaxsInputCenterVerticalLabel'):
                self.ui.gisaxsInputCenterVerticalLabel.setText(f"Center Vertical{unit_suffix}")

            if hasattr(self.ui, 'gisaxsInputCenterParallelLabel'):
                self.ui.gisaxsInputCenterParallelLabel.setText(f"Center Parallel{unit_suffix}")

            # ???Cut Line??
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalLabel'):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(f"Vertical{unit_suffix}")

            if hasattr(self.ui, 'gisaxsInputCutLineParallelLabel'):
                self.ui.gisaxsInputCutLineParallelLabel.setText(f"Parallel{unit_suffix}")

        except Exception as e:
            pass

    def _should_show_q_axis(self):
        """No description."""
        try:
            from core.global_params import GlobalParameterManager
            global_params = GlobalParameterManager()
            return global_params.get_parameter('fitting', 'detector.show_q_axis', False)
        except Exception:
            return False

    def _get_cached_q_meshgrids(self):
        """ - FittingController"""
        try:
            # ????????????????????????????
            if (self.independent_window is not None and
                hasattr(self.independent_window, '_qy_mesh') and
                self.independent_window._qy_mesh is not None):
                return self.independent_window._qy_mesh, self.independent_window._qz_mesh

            # ?????????Q???
            if hasattr(self, 'current_stack_data') and self.current_stack_data is not None:
                from core.global_params import GlobalParameterManager
                from utils.q_space_calculator import create_detector_from_image_and_params

                global_params = GlobalParameterManager()
                height, width = self.current_stack_data.shape

                pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
                pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
                beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
                beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
                distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
                theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
                wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)

                detector = create_detector_from_image_and_params(
                    image_shape=(height, width),
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    beam_center_x=beam_center_x,
                    beam_center_y=beam_center_y,
                    distance=distance,
                    theta_in_deg=theta_in_deg,
                    wavelength=wavelength,
                    crop_params=None
                )

                return detector.get_qy_qz_meshgrids()

            return None, None

        except Exception as e:
            return None, None

    def _set_default_parameters(self):
        """No description."""
        self.current_parameters = {
            'imported_gisaxs_file': '',  # ?????ISAXS???
            'stack_count': 1,  # ??????
            'cut_region': {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            'fitting_params': {}
        }

    def get_parameters(self):
        """No description."""
        return self.current_parameters.copy()

    def set_parameters(self, parameters):
        """No description."""
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)

    def get_imported_file(self):
        """ISAXS"""
        return self.current_parameters.get('imported_gisaxs_file', '')

    def get_session_data(self):
        """Return the lightweight fitting session data used by MainController."""
        session_data = {}

        gisaxs_file = self.current_parameters.get('imported_gisaxs_file', '')
        if gisaxs_file:
            gisaxs_file = normalize_path(gisaxs_file)
            session_data['last_opened_file'] = gisaxs_file
            session_data['imported_gisaxs_file'] = gisaxs_file
            session_data['last_directory'] = os.path.dirname(gisaxs_file)

        one_d_file = getattr(self, 'current_1d_file_path', None)
        if one_d_file:
            one_d_file = normalize_path(one_d_file)
            session_data['last_1d_file'] = one_d_file
            session_data['last_1d_directory'] = os.path.dirname(one_d_file)

        session_data['load_mode'] = getattr(self, 'load_mode', 'Single')
        session_data['display_mode'] = getattr(self, 'display_mode', 'normal')
        session_data['stack_value'] = self._get_stack_value_text()
        session_data['stack_count'] = self.current_parameters.get('stack_count', 1)
        session_data['insitu_range'] = self.current_parameters.get('insitu_range', '')
        session_data['fit_current_data'] = self._get_checkbox_state('fitCurrentDataCheckBox', False)
        session_data['fit_log_x'] = self._get_checkbox_state('fitLogXCheckBox', False)
        session_data['fit_log_y'] = self._get_checkbox_state('fitLogYCheckBox', False)
        session_data['fit_norm'] = self._get_checkbox_state('fitNormCheckBox', False)
        session_data['auto_show'] = self._is_auto_show_enabled()
        session_data['load_mode'] = getattr(self, 'load_mode', 'Single')
        return session_data

    def restore_session(self, session_data):
        """Restore the last opened fitting session with the current UI pathways."""
        if not isinstance(session_data, dict):
            return

        last_file = session_data.get('last_opened_file') or session_data.get('imported_gisaxs_file')
        if last_file:
            last_file = normalize_path(last_file)

        if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox'):
            try:
                self.ui.gisaxsInputAutoShowCheckBox.blockSignals(True)
                self.ui.gisaxsInputAutoShowCheckBox.setChecked(bool(session_data.get('auto_show', self._is_auto_show_enabled())))
                self.ui.gisaxsInputAutoShowCheckBox.blockSignals(False)
            except Exception:
                pass

        load_mode = str(session_data.get('load_mode', '')).strip()
        if load_mode:
            for combo_name in ('gisaxsInputModelCombox', 'gisaxsInputModeValue'):
                if not hasattr(self.ui, combo_name):
                    continue
                try:
                    combo = getattr(self.ui, combo_name)
                    index = combo.findText(load_mode)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                    break
                except Exception:
                    pass

        stack_value = str(session_data.get('stack_value', '') or session_data.get('insitu_range', '')).strip()
        if stack_value:
            self._set_stack_value_text(stack_value)

        try:
            self._sync_ui_to_parameters()
        except Exception:
            pass

        self._restore_fit_checkboxes(session_data)

        if last_file and os.path.exists(last_file):
            self.current_parameters['imported_gisaxs_file'] = last_file
            if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                self.ui.gisaxsInputImportButtonValue.setText(os.path.basename(last_file))

            try:
                self._scan_folder_images_for_file(last_file)
            except Exception:
                pass

            try:
                self._validate_imported_file(last_file)
            except Exception:
                pass

            try:
                self._update_stack_display()
            except Exception:
                pass

            try:
                self._refresh_vmin_vmax_display()
            except Exception:
                pass

            try:
                if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                    self._show_image()
            except Exception:
                pass

            self.parameters_changed.emit(self.current_parameters)
            self.status_updated.emit(f"Session restored: {os.path.basename(last_file)}")

        if session_data.get('display_mode') == 'normal':
            try:
                self._switch_to_normal_display_mode()
            except Exception:
                pass

        one_d_file = session_data.get('last_1d_file')
        if one_d_file:
            try:
                one_d_file = normalize_path(one_d_file)
                self.current_1d_file_path = one_d_file
                if hasattr(self.ui, 'fitImport1dFileValue'):
                    self.ui.fitImport1dFileValue.setText(one_d_file)
            except Exception:
                pass

    # ========== ?????????????==========

    def _import_gisaxs_file(self):
        """GISAXS"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import GISAXS",
            "",
            "GISAXS Files (*.tif *.tiff *.dat *.txt *.h5 *.hdf5 *.jpg *.png *.bmp *.cbf);;TIF Files (*.tif *.tiff);;Data Files (*.dat *.txt);;HDF5 Files (*.h5 *.hdf5 *cbf);;Image Files (*.jpg *.png *.bmp);;All Files (*)"
        )

        if file_path:
            file_path = normalize_path(file_path)
            # ??????
            self.current_parameters['imported_gisaxs_file'] = file_path

            # ???UI??????????
            if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                file_name = os.path.basename(file_path)
                self.ui.gisaxsInputImportButtonValue.setText(file_name)

            self._scan_folder_images_for_file(file_path)

            # ?????????????
            self.status_updated.emit(f"Imported GISAXS file: {os.path.basename(file_path)}")
            self.parameters_changed.emit(self.current_parameters)

            # ???????????????
            if hasattr(self.parent, 'save_current_session'):
                self.parent.save_current_session()

            # ??????
            self._validate_imported_file(file_path)

            # ?????????
            self._update_stack_display()

            # ???AutoShow????????????
            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                self._show_image()

    def _validate_imported_file(self, file_path):
        """ISAXS"""
        try:
            file_path = normalize_path(file_path)
            if not os.path.exists(file_path):
                QMessageBox.warning(self.main_window, "File Error", f"File does not exist: {file_path}")
                return False

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(self.main_window, "File Error", "File is empty")
                return False

            file_ext = os.path.splitext(file_path)[1].lower()
            supported_extensions = ['.tif', '.tiff', '.dat', '.txt', '.h5', '.hdf5', '.jpg', '.png', '.bmp', '.cbf']

            if file_ext not in supported_extensions:
                reply = QMessageBox.question(
                    self.main_window,
                    "File Format Warning",
                    f"The file extension '{file_ext}' may not be supported.\nSupported formats: {', '.join(supported_extensions)}\n\nContinue import?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False

            self.status_updated.emit(f"File validation passed - {os.path.basename(file_path)}")
            return True

        except Exception as e:
            QMessageBox.critical(self.main_window, "File Validation Error", f"Error validating file:\n{str(e)}")
            return False

    def _on_import_value_changed(self):
        """No description."""
        try:
            if not hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                return

            file_path_input = self.ui.gisaxsInputImportButtonValue.text().strip()

            if not file_path_input:
                self.status_updated.emit("Please enter a valid file path")
                return

            # ?????????????????????????????????????
            if not os.path.isabs(file_path_input):
                current_file = self.current_parameters.get('imported_gisaxs_file', '')
                if current_file and os.path.exists(current_file):
                    current_dir = os.path.dirname(current_file)
                    file_path_input = os.path.join(current_dir, file_path_input)
                else:
                    file_path_input = os.path.abspath(file_path_input)

            # ???????????
            if not os.path.exists(file_path_input):
                self.status_updated.emit(f"File does not exist: {os.path.basename(file_path_input)}")
                QMessageBox.warning(self.main_window, "File Error", f"File does not exist:\n{file_path_input}")
                return

            # ??????
            self.current_parameters['imported_gisaxs_file'] = file_path_input

            # ???UI?????????
            file_name = os.path.basename(file_path_input)
            self.ui.gisaxsInputImportButtonValue.setText(file_name)

            # ??????
            if self._validate_imported_file(file_path_input):
                self._scan_folder_images_for_file(file_path_input)
                self.status_updated.emit(f"Updated GISAXS file: {file_name}")
                self.parameters_changed.emit(self.current_parameters)

                # ???????????????
                if hasattr(self.parent, 'save_current_session'):
                    self.parent.save_current_session()

                self._update_stack_display()
                self._refresh_vmin_vmax_display()

                if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                    self._show_image()
                else:
                    self.status_updated.emit("File updated. Click 'Show' to display the image")

        except Exception as e:
            self.status_updated.emit(f"Import value processing error: {str(e)}")
            QMessageBox.critical(self.main_window, "Processing Error", f"Error handling the imported file path:\n{str(e)}")

    # ========== Stack ?????? ==========

    def _on_stack_value_changed(self):
        """No description."""
        try:
            stack_text = self.ui.gisaxsInputStackValue.text() if hasattr(self.ui, 'gisaxsInputStackValue') else "1"

            try:
                stack_count = int(stack_text)
            except ValueError:
                if hasattr(self.ui, 'gisaxsInputStackValue'):
                    self.ui.gisaxsInputStackValue.setText("1")
                stack_count = 1

            if stack_count < 1:
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText("At least 1")
                return

            self.current_parameters['stack_count'] = stack_count
            self._update_stack_display()
            self._refresh_vmin_vmax_display()

            should_reload_image = False

            if hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked():
                should_reload_image = True
            elif self.current_stack_data is not None:
                imported_file = self.current_parameters.get('imported_gisaxs_file', '')
                if imported_file and os.path.splitext(imported_file)[1].lower() == '.cbf':
                    should_reload_image = True

            if should_reload_image:
                self._show_image()
            else:
                self.status_updated.emit(f"Stack count updated to {stack_count}")

        except Exception as e:
            self.status_updated.emit(f"Stack value processing error: {str(e)}")

    def _update_stack_display(self):
        """No description."""
        try:
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                return

            file_ext = os.path.splitext(imported_file)[1].lower()
            mode = getattr(self, 'load_mode', 'Single')
            stack_count = self.current_parameters.get('stack_count', 1)

            if file_ext != '.cbf':
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"File: {os.path.basename(imported_file)}")
                return

            if mode == 'Single' or stack_count == 1:
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    self.ui.gisaxsInputStackDisplayLabel.setText(f"Single: {os.path.basename(imported_file)}")
                return

            if mode == 'Stack':
                file_dir = os.path.dirname(imported_file)
                base_name = os.path.basename(imported_file)
                cbf_files = [f for f in os.listdir(file_dir) if f.lower().endswith('.cbf')]
                cbf_files.sort()
                try:
                    start_index = cbf_files.index(base_name)
                    available_files = len(cbf_files) - start_index
                    if stack_count > available_files:
                        if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                            self.ui.gisaxsInputStackDisplayLabel.setText(f"Maximum available: {available_files}")
                    else:
                        start_file = cbf_files[start_index]
                        end_file = cbf_files[start_index + stack_count - 1]
                        start_name = os.path.splitext(start_file)[0]
                        end_name = os.path.splitext(end_file)[0]
                        if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                            self.ui.gisaxsInputStackDisplayLabel.setText(f"Stack: {start_name} - {end_name}")
                except ValueError:
                    if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                        self.ui.gisaxsInputStackDisplayLabel.setText("File not found in directory")
                return

            if mode == 'In-situ':
                dir_path = os.path.dirname(imported_file)
                sv = ''
                try:
                    if hasattr(self.ui, 'gisaxsInputStackValue'):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ''
                latest = self._find_latest_cbf(dir_path)
                if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                    if sv == '' or sv.endswith('-'):
                        self.ui.gisaxsInputStackDisplayLabel.setText(
                            f"In-situ: latest -> {os.path.splitext(os.path.basename(latest or ''))[0]}"
                        )
                    elif '-' in sv:
                        self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ range: {sv}")
                    else:
                        self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ index: {sv}")
                return

        except Exception as e:
            self.status_updated.emit(f"Display update error: {str(e)}")

    # ========== ?????????????==========

    def _sync_ui_to_parameters(self):
        """UI"""
        try:
            # ??????? - ?????????
            if hasattr(self.ui, 'gisaxsInputImportButtonValue'):
                file_input = self.ui.gisaxsInputImportButtonValue.text().strip()
                if file_input:
                    # ??????????????????????
                    if os.path.isabs(file_input) and os.path.exists(file_input):
                        self.current_parameters['imported_gisaxs_file'] = file_input
                    # ?????????????????????????????????
                    elif not os.path.isabs(file_input):
                        file_found = False

                        # ????????????????????
                        current_file = self.current_parameters.get('imported_gisaxs_file', '')
                        if current_file and os.path.dirname(current_file):
                            # ?????????????
                            new_path = os.path.join(os.path.dirname(current_file), file_input)
                            if os.path.exists(new_path):
                                self.current_parameters['imported_gisaxs_file'] = new_path
                                file_found = True

                        # ?????????????????xperiment_data??????
                        if not file_found:
                            experiment_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Experiment_data')
                            if os.path.exists(experiment_dir):
                                new_path = os.path.join(experiment_dir, file_input)
                                if os.path.exists(new_path):
                                    self.current_parameters['imported_gisaxs_file'] = new_path
                                    file_found = True

                        # ????????????????????????????
                        if not file_found:
                            self.status_updated.emit(f"Error: File '{file_input}' not found in any expected location")
                            # ?????imported_gisaxs_file ?????????????????????????
                            return  # ???????????????

                    # ???????????????????
                    elif os.path.isabs(file_input) and not os.path.exists(file_input):
                        self.status_updated.emit(f"Error: File '{file_input}' does not exist")
                        # ??????????????????
                        return  # ???????????????

            # ??Stack/????????????
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                sv = self.ui.gisaxsInputStackValue.text().strip()
                if getattr(self, 'load_mode', 'Single') == 'Single':
                    self.current_parameters['stack_count'] = 1
                elif self.load_mode == 'Stack':
                    try:
                        self.current_parameters['stack_count'] = max(1, int(sv or '1'))
                    except Exception:
                        self.current_parameters['stack_count'] = 1
                elif self.load_mode == 'In-situ':
                    self.current_parameters['insitu_range'] = sv

        except Exception as e:
            self.status_updated.emit(f"Failed to sync UI parameters: {str(e)}")

    def _show_image(self):
        """No description."""
        try:
            # ????I?????????????????
            self._sync_ui_to_parameters()

            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                self.status_updated.emit("No file imported to show")
                return

            if not os.path.exists(imported_file):
                self.status_updated.emit("File does not exist")
                QMessageBox.warning(self.main_window, "File Error", f"File does not exist:\n{imported_file}")
                self._scan_folder_images_for_file(imported_file)
                return

            self._scan_folder_images_for_file(imported_file)

            # ????????
            if not is_fabio_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "fabio library is required for CBF file processing.\nPlease install it using: pip install fabio")
                return

            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for image display.\nPlease install it using: pip install matplotlib")
                return

            # ??????????????????
            file_ext = os.path.splitext(imported_file)[1].lower()
            if file_ext != '.cbf':
                self.status_updated.emit("Image display only supports CBF files currently")
                return

            mode = getattr(self, 'load_mode', 'Single')
            if mode == 'Single':
                self.status_updated.emit("Please wait while the image starts loading (Single)...")
                self.async_image_loader.load_image(imported_file, 1)
            elif mode == 'Stack':
                stack_count = self.current_parameters.get('stack_count', 1)
                self.status_updated.emit(f"Please wait while stacking {stack_count} files...")
                self.async_image_loader.load_image(imported_file, stack_count)
            else:
                # In-situ ??????????????????
                sv = ''
                try:
                    if hasattr(self.ui, 'gisaxsInputStackValue'):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ''
                dir_path = os.path.dirname(imported_file)
                target = self._resolve_insitu_target(dir_path, imported_file, sv)
                if not target:
                    self.status_updated.emit("No CBF file found for In-situ mode")
                    return
                self._insitu_last_file = target
                self._show_image_insitu(target)

        except Exception as e:
            self.status_updated.emit(f"Show image error: {str(e)}")

    def _on_load_mode_changed(self, text: str):
        """No description."""
        try:
            self.load_mode = text or 'Single'
            # ?????
            try:
                from core.global_params import GlobalParameterManager
                gp = GlobalParameterManager()
                gp.set_parameter('fitting', 'gisaxs_input.load_mode', self.load_mode)
            except Exception:
                pass
            # ??????
            self._update_stack_controls_visibility()
            # ???????
            if self.load_mode == 'In-situ':
                if self._is_auto_show_enabled():
                    self._start_insitu_timer()
            else:
                self._stop_insitu_timer()
            # ????????
            self._update_stack_display()
        except Exception:
            pass

    def _update_stack_controls_visibility(self):
        """No description."""
        try:
            base_widget = getattr(self.ui, 'gisaxsInputStackValue', None)
            editor_widget = getattr(self.ui, 'gisaxsInputStackEditorWidget', None)
            editor_layout = getattr(self.ui, 'gisaxsInputStackEditorLayout', None)
            stack_widget = getattr(self.ui, 'gisaxsInputStackWidget', None)
            stack_layout = stack_widget.layout() if stack_widget is not None else None
            try:
                from PyQt5.QtWidgets import QLineEdit
                from PyQt5.QtGui import QIntValidator, QRegularExpressionValidator
                from PyQt5.QtCore import QRegularExpression
                if self.load_mode == 'In-situ':
                    # ???????????????????????? 1-??
                    if base_widget is not None:
                        base_widget.setVisible(False)
                    if editor_widget is not None:
                        editor_widget.setVisible(True)
                    # ???????????In-situ LineEdit
                    if not hasattr(self, '_insitu_lineedit') or self._insitu_lineedit is None:
                        # ??????????????????
                        parent = None
                        try:
                            parent = editor_widget if editor_widget is not None else (base_widget.parent() if base_widget is not None else self.ui.gisaxsInputStackDisplayLabel.parent())
                        except Exception:
                            parent = None
                        from PyQt5.QtWidgets import QLineEdit as _QLE
                        self._insitu_lineedit = _QLE(parent)
                        # ??????????????????????????????????
                        try:
                            layout = editor_layout if editor_layout is not None else (parent.layout() if parent is not None else None)
                            if layout is not None:
                                # ??????????????????????????????????
                                try:
                                    disp_label = getattr(self.ui, 'gisaxsInputStackDisplayLabel', None)
                                    if disp_label is not None:
                                        idx = layout.indexOf(disp_label)
                                        if idx >= 0:
                                            layout.insertWidget(idx, self._insitu_lineedit)
                                        else:
                                            layout.addWidget(self._insitu_lineedit)
                                    else:
                                        layout.addWidget(self._insitu_lineedit)
                                except Exception:
                                    layout.addWidget(self._insitu_lineedit)
                        except Exception:
                            pass
                        # ??????? N | A-B | A-
                        regex = QRegularExpression(r"^\s*(?:\d+|\d+\s*-\s*\d+|\d+\s*-)\s*$")
                        self._insitu_lineedit.setValidator(QRegularExpressionValidator(regex, self._insitu_lineedit))
                        # ????
                        self._insitu_lineedit.setText('1-')
                        self._insitu_lineedit.setPlaceholderText('e.g. 1-, 1-10, 5')
                        # ?????????????????
                        try:
                            self._insitu_lineedit.returnPressed.connect(self._on_stack_value_changed)
                            self._insitu_lineedit.editingFinished.connect(self._on_stack_value_changed)
                        except Exception:
                            pass
                    self._insitu_lineedit.setVisible(True)
                else:
                    # ??In-situ ????????LineEdit?????????
                    if hasattr(self, '_insitu_lineedit') and self._insitu_lineedit is not None:
                        self._insitu_lineedit.setVisible(False)
                    if base_widget is not None:
                        base_widget.setVisible(self.load_mode != 'Single')
                        if editor_widget is not None:
                            editor_widget.setVisible(self.load_mode != 'Single')
                        if self.load_mode == 'Stack':
                            base_widget.setValidator(QIntValidator(1, 9999, base_widget))
                            base_widget.setPlaceholderText('e.g. 5')
                        else:
                            base_widget.setValidator(None)
                            base_widget.setPlaceholderText('')
            except Exception:
                # ??? PyQt5 ???????
                if base_widget is not None:
                    base_widget.setVisible(self.load_mode != 'Single')
                if editor_widget is not None:
                    editor_widget.setVisible(self.load_mode != 'Single')
            if stack_layout is not None:
                top_margin = 0
                if self.load_mode == 'Single':
                    combo = getattr(self.ui, 'gisaxsInputModelCombox', None)
                    label = getattr(self.ui, 'gisaxsInputStackDisplayLabel', None)
                    combo_height = combo.sizeHint().height() if combo is not None else 0
                    text_height = label.fontMetrics().height() if label is not None else 0
                    top_margin = max(0, (combo_height - text_height) // 2)
                left, _top, right, bottom = stack_layout.getContentsMargins()
                stack_layout.setContentsMargins(left, top_margin, right, bottom)
            if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                self.ui.gisaxsInputStackDisplayLabel.setVisible(True)
        except Exception:
            pass

    def _enforce_insitu_visibility_once(self):
        """No description."""
        try:
            mode = getattr(self, 'load_mode', 'Single')
            base_widget = getattr(self.ui, 'gisaxsInputStackValue', None)
            editor_widget = getattr(self.ui, 'gisaxsInputStackEditorWidget', None)
            insitu_edit = getattr(self, '_insitu_lineedit', None)
            if mode == 'In-situ':
                if base_widget is not None:
                    base_widget.setVisible(False)
                if editor_widget is not None:
                    editor_widget.setVisible(True)
                if insitu_edit is not None:
                    insitu_edit.setVisible(True)
            elif mode == 'Stack':
                if insitu_edit is not None:
                    insitu_edit.setVisible(False)
                if base_widget is not None:
                    base_widget.setVisible(True)
                if editor_widget is not None:
                    editor_widget.setVisible(True)
            else:
                # Single
                if insitu_edit is not None:
                    insitu_edit.setVisible(False)
                if base_widget is not None:
                    base_widget.setVisible(False)
                if editor_widget is not None:
                    editor_widget.setVisible(False)
        except Exception:
            pass

    # ????????????????????????????????????

    def _is_auto_show_enabled(self) -> bool:
        return hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked()

    def _start_insitu_timer(self):
        try:
            if self._insitu_timer is None:
                from PyQt5.QtCore import QTimer
                self._insitu_timer = QTimer()
                self._insitu_timer.setSingleShot(False)
                self._insitu_timer.timeout.connect(self._insitu_poll_latest)
            self._insitu_timer.start(2000)
        except Exception:
            pass

    def _stop_insitu_timer(self):
        try:
            if self._insitu_timer is not None:
                self._insitu_timer.stop()
        except Exception:
            pass

    def _insitu_poll_latest(self):
        try:
            if self.load_mode != 'In-situ':
                return
            imported_file = self.current_parameters.get('imported_gisaxs_file', '')
            if not imported_file:
                return
            dir_path = os.path.dirname(imported_file)
            latest = self._find_latest_cbf(dir_path)
            if latest and latest != self._insitu_last_file:
                sv = ''
                try:
                    if hasattr(self.ui, 'gisaxsInputStackValue'):
                        sv = self.ui.gisaxsInputStackValue.text().strip()
                except Exception:
                    sv = ''
                if (sv.endswith('-') or sv.strip() == '' or sv.strip().endswith('-')) and self._is_auto_show_enabled():
                    self._insitu_last_file = latest
                    self._show_image_insitu(latest)
        except Exception:
            pass

    def _get_stack_value_text(self) -> str:
        try:
            if getattr(self, 'load_mode', 'Single') == 'In-situ' and hasattr(self, '_insitu_lineedit') and self._insitu_lineedit is not None:
                return self._insitu_lineedit.text().strip()
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                return self.ui.gisaxsInputStackValue.text().strip()
        except Exception:
            pass
        return ''

    def _set_stack_value_text(self, value: str):
        try:
            stack_text = str(value).strip()
            if getattr(self, 'load_mode', 'Single') == 'In-situ' and hasattr(self, '_insitu_lineedit') and self._insitu_lineedit is not None:
                self._insitu_lineedit.setText(stack_text or '1-')
                return
            if hasattr(self.ui, 'gisaxsInputStackValue'):
                self.ui.gisaxsInputStackValue.setText(stack_text)
        except Exception:
            pass

    def _resolve_insitu_target(self, dir_path: str, imported_file: str, sv_text: str) -> str:
        try:
            files = [f for f in os.listdir(dir_path) if f.lower().endswith('.cbf')]
            if not files:
                return None
            files.sort()
            def _index_from_name(name: str):
                base = os.path.splitext(name)[0]
                import re
                m = re.search(r'(\d+)$', base)
                return int(m.group(1)) if m else None
            indexed = [(fn, _index_from_name(fn)) for fn in files]
            latest_file = files[-1]
            t = (sv_text or '').strip()
            if t == '':
                return os.path.join(dir_path, latest_file)
            if '-' in t:
                parts = t.split('-')
                try:
                    a = int(parts[0]) if parts[0] != '' else None
                except Exception:
                    a = None
                b = None
                try:
                    if len(parts) > 1 and parts[1] != '':
                        b = int(parts[1])
                except Exception:
                    b = None
                if b is None:
                    return os.path.join(dir_path, latest_file)
                for fn, idx in indexed:
                    if idx == b:
                        return os.path.join(dir_path, fn)
                return os.path.join(dir_path, latest_file)
            else:
                try:
                    n = int(t)
                    for fn, idx in indexed:
                        if idx == n:
                            return os.path.join(dir_path, fn)
                except Exception:
                    pass
                return os.path.join(dir_path, latest_file)
        except Exception:
            return None

    def _find_latest_cbf(self, dir_path: str) -> str:
        try:
            files = [f for f in os.listdir(dir_path) if f.lower().endswith('.cbf')]
            if not files:
                return None
            files.sort()
            return os.path.join(dir_path, files[-1])
        except Exception:
            return None

    def _show_image_insitu(self, target_path: str):
        try:
            if not target_path:
                return
            self.async_image_loader.load_image(target_path, 1)
            if hasattr(self.ui, 'gisaxsInputStackDisplayLabel'):
                base = os.path.basename(target_path)
                self.ui.gisaxsInputStackDisplayLabel.setText(f"In-situ: {os.path.splitext(base)[0]}")
        except Exception:
            pass

    def _on_image_loaded(self, image_data, file_path):
        """No description."""
        try:
            self.status_updated.emit(f"Image loading complete: {os.path.basename(file_path)}")
            self._display_image(image_data)
        except Exception as e:
            self.status_updated.emit(f"Error while displaying image: {str(e)}")

    def _on_image_loading_progress(self, progress, status):
        """No description."""
        try:
            self.status_updated.emit(f"Image loading... {progress}% - {status}")
            self.progress_updated.emit(progress)
        except Exception as e:
            self.status_updated.emit(f"Progress update error: {str(e)}")

    def _on_image_loading_error(self, error_message):
        """No description."""
        QMessageBox.critical(self.main_window, "Image loading error", error_message)

    def _display_image(self, image_data):
        """No description."""
        try:
            # ?????????
            self.current_stack_data = image_data
            # ?????????????????????
            self.data = image_data
            try:
                sc = int(self.current_parameters.get('stack_count', 1))
            except Exception:
                sc = 1
            self.summed_data = image_data if sc and sc > 1 else None
            # ???????????????????y, qz, qr??
            self._compute_q_meshgrids_and_store()

            # ????????????
            self._handle_color_scale(image_data)

            # ???Cut Line???????
            self._update_cutline_labels_units()

            # ????????raphicsView
            if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                self._update_graphics_view(image_data)

            # ???????????????????????????
            if self.independent_window is not None and self.independent_window.isVisible():
                is_log = self._is_log_mode_enabled()
                self.independent_window.update_image(image_data, self._current_vmin, self._current_vmax, use_log=is_log)
                self._sync_independent_window_selection()

            # ???????
            window_status = " (+ Independent window)" if (self.independent_window and self.independent_window.isVisible()) else ""
            vmin_vmax_info = f" [Vmin: {self._current_vmin:.3f}, Vmax: {self._current_vmax:.3f}]" if self._current_vmin is not None and self._current_vmax is not None else ""
            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            self.status_updated.emit(f"{mode_text} image displayed: {image_data.shape}{vmin_vmax_info}{window_status}")

        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")

    def _compute_q_meshgrids_and_store(self):
        """No description."""
        try:
            if self.current_stack_data is None:
                return
            from core.global_params import GlobalParameterManager
            from utils.q_space_calculator import create_detector_from_image_and_params
            global_params = GlobalParameterManager()
            height, width = self.current_stack_data.shape
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)
            cache_key = (
                height, width,
                float(pixel_size_x), float(pixel_size_y),
                float(beam_center_x), float(beam_center_y),
                float(distance), float(theta_in_deg), float(wavelength),
            )
            if self._q_mesh_cache_key == cache_key and self.qy_matrix is not None and self.qz_matrix is not None:
                return
            t0 = time.perf_counter()
            detector = create_detector_from_image_and_params(
                image_shape=(height, width),
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                distance=distance,
                theta_in_deg=theta_in_deg,
                wavelength=wavelength,
                crop_params=None
            )
            qy_mesh, qz_mesh = detector.get_qy_qz_meshgrids()
            self.qy_matrix = qy_mesh
            self.qz_matrix = qz_mesh
            try:
                self.qr_matrix = np.sqrt(np.square(qy_mesh) + np.square(qz_mesh))
            except Exception:
                self.qr_matrix = None
            self._q_mesh_cache_key = cache_key
            print(f"[Timing] q-space mesh calculation: {(time.perf_counter() - t0) * 1000:.2f} ms")
        except Exception:
            # ????????????????????
            self.qy_matrix = None
            self.qz_matrix = None
            self.qr_matrix = None
            self._q_mesh_cache_key = None

    def _update_graphics_view(self, image_data):
        """GraphicsView"""
        self._update_graphics_view_with_selection(
            image_data,
            getattr(self, 'current_parameter_selection', None)
        )

    def _prepare_image_data_for_display(self, image_data):
        """No description."""
        try:
            is_log = self._is_log_mode_enabled()
            cache_key = (id(image_data), bool(is_log))
            cached = self._image_display_cache.get(cache_key)
            if cached is not None:
                self._image_display_cache.move_to_end(cache_key)
                if is_log:
                    print("[Timing] log transform: 0.00 ms (cache hit)")
                return cached, is_log

            if is_log:
                t0 = time.perf_counter()
                # Log???????????????????????????
                safe_data = np.where(image_data > 0, image_data, 0.001)
                processed_data = np.log(safe_data, dtype=np.float32)
                print(f"[Timing] log transform: {(time.perf_counter() - t0) * 1000:.2f} ms")
            else:
                # ???????????????????
                processed_data = image_data.astype(np.float32)
            self._image_display_cache[cache_key] = processed_data
            self._image_display_cache.move_to_end(cache_key)
            while len(self._image_display_cache) > self._image_display_cache_limit:
                self._image_display_cache.popitem(last=False)

            return processed_data, is_log

        except Exception:
            # ??????????????Log???
            return image_data.astype(np.float32), True

    def _refresh_image_display(self):
        """No description."""
        try:
            if self.current_stack_data is not None:
                # ????????raphicsView
                if hasattr(self.ui, 'gisaxsInputGraphicsView'):
                    self._update_graphics_view(self.current_stack_data)

                # ???????????????????????????
                if self.independent_window is not None and self.independent_window.isVisible():
                    is_log = self._is_log_mode_enabled()
                    self.independent_window.update_image(self.current_stack_data,
                                                       self._current_vmin, self._current_vmax,
                                                       use_log=is_log)
                    self._sync_independent_window_selection()
        except Exception as e:
            self.status_updated.emit(f"Refresh display error: {str(e)}")

    def _on_graphics_view_double_click(self, event):
        """GraphicsView"""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib")
                return

            # ???????????????????????
            if self.current_stack_data is None:
                QMessageBox.information(self.main_window, "No Image", "Please import and display an image first.")
                return

            # ??????????????
            self._show_independent_window()

        except Exception as e:
            self.status_updated.emit(f"Double-click error: {str(e)}")

    def _show_independent_window(self):
        """atplotlib"""
        try:
            # ?????????????????????????
            if self.independent_window is None or not self.independent_window.isVisible():
                self.independent_window = IndependentMatplotlibWindow(self.main_window)
                # ????????????
                self.independent_window.region_selected.connect(self._on_region_selected)
                # ?????????????
                self.independent_window.status_updated.connect(self.status_updated.emit)

            # ????????????????????vmin/vmax??og?????
            if self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                # ?????????????
                self.independent_window.current_image_shape = self.current_stack_data.shape
                self.independent_window.update_image(self.current_stack_data,
                                                   self._current_vmin, self._current_vmax,
                                                   use_log=is_log)
                self._sync_independent_window_selection()

            # ??????????????
            if not self.independent_window.isVisible():
                move_window_to_cursor_screen(self.independent_window)
            self.independent_window.show()
            self.independent_window.raise_()
            self.independent_window.activateWindow()

            # ????????anvas??????????????
            self.independent_window.canvas.setFocus()

            self.status_updated.emit("Independent window opened - Right-click to activate selection, ESC to exit selection mode")

        except Exception as e:
            self.status_updated.emit(f"Independent window error: {str(e)}")

    def _on_region_selected(self, selection_info):
        """ut Line"""
        try:
            # ?????????????????????????
            is_q_space = selection_info.get('is_q_space', False)

            # ???Cut Line??enter???
            updated_controls = []

            if is_q_space:
                # Q?????????????????Q???????????
                center_qy = selection_info.get('center_x', 0)  # Q?????x???qy
                center_qz = selection_info.get('center_y', 0)  # Q?????y???qz
                width_q = selection_info.get('width', 0)
                height_q = selection_info.get('height', 0)

                self.status_updated.emit(
                    f"Q-space region selected: center=({center_qy:.6f}, {center_qz:.6f}) nm^-1, "
                    f"size=({width_q:.6f} x {height_q:.6f}) nm^-1"
                )

                try:
                    # ??????????????Q??????
                    if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                        self.ui.gisaxsInputCenterVerticalValue.setValue(center_qz)  # Vertical???qz???????
                        updated_controls.append('gisaxsInputCenterVerticalValue')

                    if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                        self.ui.gisaxsInputCenterParallelValue.setValue(center_qy)  # Parallel???qy???????
                        updated_controls.append('gisaxsInputCenterParallelValue')

                    if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                        self.ui.gisaxsInputCutLineVerticalValue.setValue(height_q)  # Vertical?????height_q
                        updated_controls.append('gisaxsInputCutLineVerticalValue')

                    if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                        self.ui.gisaxsInputCutLineParallelValue.setValue(width_q)  # Parallel?????width_q
                        updated_controls.append('gisaxsInputCutLineParallelValue')

                    # ????????????????????????Q?????????????
                    pixel_coords = self._convert_q_to_pixel_coordinates(center_qy, center_qz, width_q, height_q)
                    center_x = pixel_coords['center_x']
                    center_y = pixel_coords['center_y']
                    width = pixel_coords['width']
                    height = pixel_coords['height']

                except Exception as e:
                    self.status_updated.emit(f"Q-space parameter update failure: {str(e)}")
                    return
            else:
                # ???????????????????????
                center_x = selection_info.get('pixel_center_x', 0)
                center_y = selection_info.get('pixel_center_y', 0)
                width = selection_info.get('pixel_width', 0)
                height = selection_info.get('pixel_height', 0)

                self.status_updated.emit(
                    f"Pixel region selected: Center({center_x}, {center_y}), Size({width} x {height})"
                )

                # ????????????????????
                # Vertical???Y???????????????Parallel???X??????????????
                # ???Vertical????-> Y???
                if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                    self.ui.gisaxsInputCenterVerticalValue.setValue(center_y)
                    updated_controls.append('gisaxsInputCenterVerticalValue')

                # ???Parallel????-> X???
                if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                    self.ui.gisaxsInputCenterParallelValue.setValue(center_x)
                    updated_controls.append('gisaxsInputCenterParallelValue')

                # ???Cut Line??ertical??arallel????????????
                # Vertical??????????????Parallel?????????????
                if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                    self.ui.gisaxsInputCutLineVerticalValue.setValue(height)
                    updated_controls.append('gisaxsInputCutLineVerticalValue')

                if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                    self.ui.gisaxsInputCutLineParallelValue.setValue(width)
                    updated_controls.append('gisaxsInputCutLineParallelValue')

            # ??????GraphicsView???????????
            if is_q_space:
                # Q??????????????????????????
                main_view_selection_info = {
                    'bounds': {
                        'x_min': center_qy - width_q / 2,
                        'x_max': center_qy + width_q / 2,
                        'y_min': center_qz - height_q / 2,
                        'y_max': center_qz + height_q / 2
                    },
                    'center_x': center_qy,
                    'center_y': center_qz,
                    'width': width_q,
                    'height': height_q,
                    'is_q_space': True
                }
            else:
                # ????????????????????
                main_view_selection_info = {
                    'bounds': {
                        'x_min': center_x - width / 2,
                        'x_max': center_x + width / 2,
                        'y_min': center_y - height / 2,
                        'y_max': center_y + height / 2
                    },
                    'pixel_center_x': center_x,
                    'pixel_center_y': center_y,
                    'pixel_width': width,
                    'pixel_height': height,
                    'is_q_space': False
                }

            self._draw_selection_on_main_view(main_view_selection_info)

            # ??????????????
            if updated_controls:
                coord_mode = "Q-space" if is_q_space else "pixel"
                self.status_updated.emit(f"Updated Cut Line parameters ({coord_mode}): {', '.join(updated_controls)}")
                # ????????????????????
                if self.independent_window and self.independent_window.isVisible():
                    if is_q_space:
                        self.independent_window.setWindowTitle(
                            f"GIMaP Image Viewer - Q selection "
                            f"center=({center_qy:.6f}, {center_qz:.6f}) nm^-1, "
                            f"size=({width_q:.6f} x {height_q:.6f}) nm^-1"
                        )
                    else:
                        self.independent_window.setWindowTitle(
                            f"GIMaP Image Viewer - pixel selection "
                            f"center=({center_x}, {center_y}), size=({width} x {height})"
                        )
            else:
                self.status_updated.emit("No matching Cut Line controls found for parameter update")

        except Exception as e:
            self.status_updated.emit(f"Error updating Cut Line parameters: {str(e)}")

    @staticmethod
    def _plot_cut_data_with_log_handling(ax, x_coords, y_intensity, is_log_x, markersize=4, linewidth=1.5):
        """No description."""
        try:
            x_array = np.array(x_coords)
            y_array = np.array(y_intensity)

            if is_log_x:
                # ??????????????
                positive_mask = x_array > 0
                x_positive = x_array[positive_mask]
                y_positive = y_array[positive_mask]

                # ???????????????
                negative_mask = x_array < 0
                x_negative_abs = np.abs(x_array[negative_mask])
                y_negative = y_array[negative_mask]

                # ???????????????
                zero_mask = x_array == 0
                x_zero = x_array[zero_mask]
                y_zero = y_array[zero_mask]

                # ???????????????
                if len(x_positive) > 0:
                    ax.plot(x_positive, y_positive, 'bo-', markersize=markersize, linewidth=linewidth,
                           markerfacecolor='lightblue', alpha=0.8, label='Positive coordinates')

                # ????????????????????????
                if len(x_negative_abs) > 0:
                    ax.plot(x_negative_abs, y_negative, 'ro--', markersize=markersize, linewidth=linewidth,
                           markerfacecolor='lightcoral', alpha=0.8, label='Negative coordinates (|x|)')

                # ??????????????????
                if len(x_zero) > 0:
                    # ?????????????????????????????
                    min_positive = min(np.min(x_positive) if len(x_positive) > 0 else 1e-6,
                                     np.min(x_negative_abs) if len(x_negative_abs) > 0 else 1e-6)
                    x_zero_replacement = np.full_like(x_zero, min_positive * 0.1)
                    ax.plot(x_zero_replacement, y_zero, 'go^', markersize=markersize+2,
                           markerfacecolor='lightgreen', alpha=0.8, label='Zero coordinates (approximated)')

                # ??????
                ax.legend(loc='best', fontsize=max(8, markersize*2))

            else:
                # ??og-X??????????
                ax.plot(x_array, y_array, 'bo-', markersize=markersize, linewidth=linewidth,
                       markerfacecolor='lightblue', alpha=0.8)

        except Exception as e:
            raise Exception(f"Plot data error: {str(e)}")

    def _on_fit_graphics_view_double_click(self, event):
        """No description."""
        try:
            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for independent window.\nPlease install it using: pip install matplotlib")
                return

            # ???????????
            if self.q is None or self.I is None:
                QMessageBox.information(self.main_window, "No Data", "No data available for display.")
                return

            # ?????????????????
            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
                # ??????????????????????????
                self.independent_fit_window.show_positive_cb.toggled.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'show_negative_cb'):
                    self.independent_fit_window.show_negative_cb.toggled.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'q_unit_combo'):
                    self.independent_fit_window.q_unit_combo.currentTextChanged.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'y_range_combo'):
                    self.independent_fit_window.y_range_combo.currentTextChanged.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'input_point_delete_requested'):
                    self.independent_fit_window.input_point_delete_requested.connect(self._exclude_ai_input_point_from_plot)
                try:
                    self._sync_axis_filter_controls()
                except Exception:
                    pass

                # ??????
                move_window_to_cursor_screen(self.independent_fit_window)
                self.independent_fit_window.show()
                self.independent_fit_window.raise_()
                self.independent_fit_window.activateWindow()

            # ?????????????????????????????????
            # ????????????????????????????????????????????????itting
            mode = (self.display_mode if hasattr(self, 'display_mode') else 'normal')
            try:
                if hasattr(self, '_is_in_fitting_mode') and callable(self._is_in_fitting_mode) and self._is_in_fitting_mode():
                    mode = 'fitting'
            except Exception:
                pass
            # ???????????????????????????normal ?????????????????????????????
            try:
                has_fit = bool(getattr(self, 'has_fitting_data', False) and getattr(self, 'I_fitting', None) is not None)
                if mode == 'fitting' and not has_fit:
                    mode = 'normal'
            except Exception:
                pass

            # ???????????????????????GUI?????????????????????ROI/Normalize???????????????
            if mode == 'fitting':
                try:
                    self._update_gui_fitting_display()
                except Exception:
                    pass
                self._update_outside_window('fitting')
            else:
                self._update_outside_window(mode)

            # ???????????
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.setFocus()
                # ?????????
                self.independent_fit_window.canvas.draw()

            self.status_updated.emit(f"{mode.capitalize()} mode independent window updated")

        except Exception as e:
            self.status_updated.emit(f"Fit double-click error: {str(e)}")



    def _on_cutline_parameters_changed(self):
        """No description."""
        try:
            # ??????????????????
            if getattr(self, '_initializing', True):
                return

            # ???????????????????????????
            if not hasattr(self, '_cutline_update_timer'):
                from PyQt5.QtCore import QTimer
                self._cutline_update_timer = QTimer()
                self._cutline_update_timer.setSingleShot(True)
                self._cutline_update_timer.timeout.connect(self._delayed_cutline_update)

            # ?????????????????????
            self._cutline_update_timer.stop()
            self._cutline_update_timer.start(150)  # 150ms????????????????

        except Exception as e:
            pass

    def _delayed_cutline_update(self):
        """No description."""
        try:
            # ????????ut Line?????
            center_x = 0
            center_y = 0
            width = 0
            height = 0

            # ???Center???
            if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()

            # ???Cut Line?????
            if hasattr(self.ui, 'gisaxsInputCutLineParallelValue'):
                width = self.ui.gisaxsInputCutLineParallelValue.value()
            if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue'):
                height = self.ui.gisaxsInputCutLineVerticalValue.value()

            # ??????????????????????????
            if center_x == 0 and center_y == 0 and width == 0 and height == 0:
                self._clear_parameter_selection()
                return

            # ????????????????????????
            if width <= 0 or height <= 0:
                self._clear_parameter_selection()
                return

            # ????????????
            selection_info = self._create_selection_from_parameters(center_x, center_y, width, height)

            # ?????????
            self._update_parameter_selection_display(selection_info)

            # ???????????Cut??????????????????????ut???
            if (self.current_cut_data is not None and
                hasattr(self, 'current_stack_data') and self.current_stack_data is not None):

                # ???????Cut???
                self._perform_cut()
                self.status_updated.emit(f"Auto-updated cut with new parameters: Center({center_x}, {center_y}), Size({width} x {height})")
            else:
                # ??????????
                self.status_updated.emit(f"Parameter selection: Center({center_x}, {center_y}), Size({width} x {height}) - Perform Cut to see results")

        except Exception as e:
            self.status_updated.emit(f"Error updating parameter selection: {str(e)}")

    def _create_selection_from_parameters(self, center_x, center_y, width, height):
        """No description."""
        is_q_space = self._should_show_q_axis()
        # ??????????????
        half_width = width / 2
        half_height = height / 2

        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height

        # ?????????????????????????
        selection_info = {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'pixel_center_x': int(center_x),
            'pixel_center_y': int(center_y),
            'pixel_width': int(width),
            'pixel_height': int(height),
            'bounds': {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            },
            'is_q_space': is_q_space,
            'is_parameter_based': True  # ????????????????
        }

        return selection_info

    def _update_parameter_selection_display(self, selection_info):
        """No description."""
        try:
            # ?????????????????
            self.current_parameter_selection = selection_info

            # ???????????????
            if self.current_stack_data is not None:
                self._update_graphics_view_with_selection(self.current_stack_data, selection_info)

            # ?????????????????
            self._sync_independent_window_selection()

        except Exception as e:
            self.status_updated.emit(f"Error updating parameter selection display: {str(e)}")

    def _sync_independent_window_selection(self):
        """No description."""
        try:
            if self.independent_window is None or not self.independent_window.isVisible():
                return
            selection_info = getattr(self, 'current_parameter_selection', None)
            if selection_info:
                if hasattr(self.independent_window, 'set_parameter_selection'):
                    self.independent_window.set_parameter_selection(selection_info)
            else:
                self.independent_window.clear_parameter_selection()
        except Exception as e:
            self.status_updated.emit(f"Error syncing independent window selection: {str(e)}")

    def _refresh_current_parameter_selection_from_ui(self):
        """No description."""
        try:
            if not all(hasattr(self.ui, name) for name in (
                'gisaxsInputCenterParallelValue',
                'gisaxsInputCenterVerticalValue',
                'gisaxsInputCutLineParallelValue',
                'gisaxsInputCutLineVerticalValue',
            )):
                return

            center_x = self.ui.gisaxsInputCenterParallelValue.value()
            center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            width = self.ui.gisaxsInputCutLineParallelValue.value()
            height = self.ui.gisaxsInputCutLineVerticalValue.value()

            if width > 0 and height > 0:
                self.current_parameter_selection = self._create_selection_from_parameters(
                    center_x, center_y, width, height
                )
            else:
                self.current_parameter_selection = None
        except Exception:
            pass

    def _clear_parameter_selection(self):
        """No description."""
        try:
            # ?????????????????
            self.current_parameter_selection = None

            # ???????????????????????
            if self.current_stack_data is not None:
                self._update_graphics_view_with_selection(self.current_stack_data, None)

            # ???????????????
            if self.independent_window is not None and self.independent_window.isVisible():
                self.independent_window.clear_parameter_selection()

            self.status_updated.emit("Parameter selection cleared")

        except Exception as e:
            self.status_updated.emit(f"Error clearing parameter selection: {str(e)}")

    def _draw_selection_on_main_view(self, selection_info):
        """raphicsView"""
        try:
            if not hasattr(self.ui, 'gisaxsInputGraphicsView') or self.current_stack_data is None:
                return

            self.current_parameter_selection = selection_info
            # ??????????????????????????????
            self._update_graphics_view_with_selection(self.current_stack_data, selection_info)
            self._sync_independent_window_selection()

        except Exception as e:
            self.status_updated.emit(f"Error drawing selection on main view: {str(e)}")

    def _downsample_for_preview(self, data, max_pixels=700_000):
        try:
            h, w = data.shape
            pixels = max(1, h * w)
            step = max(1, int(np.ceil(np.sqrt(pixels / max_pixels))))
            return data[::step, ::step], step
        except Exception:
            return data, 1

    def _preview_extent(self, image_shape, show_q_axis):
        if show_q_axis:
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            if qy_mesh is not None and qz_mesh is not None:
                return [qy_mesh.min(), qy_mesh.max(), qz_mesh.min(), qz_mesh.max()], True
            return None, False
        height, width = image_shape
        return [-0.5, width - 0.5, -0.5, height - 0.5], False

    def _draw_preview_selection(self, ax, selection_info):
        try:
            for artist in getattr(self, '_preview_selection_artists', []):
                try:
                    artist.remove()
                except Exception:
                    pass
            self._preview_selection_artists = []
            if not selection_info:
                return
            bounds = selection_info.get('bounds', {})
            x_min = bounds.get('x_min', 0)
            x_max = bounds.get('x_max', 0)
            y_min = bounds.get('y_min', 0)
            y_max = bounds.get('y_max', 0)
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            marker = ax.plot((x_min + x_max) / 2, (y_min + y_max) / 2,
                             'r+', markersize=10, markeredgewidth=2)[0]
            self._preview_selection_artists = [rect, marker]
        except Exception:
            pass

    def _try_update_cached_preview(self, image_data, selection_info=None):
        try:
            if not is_matplotlib_available():
                return False
            t_total = time.perf_counter()
            graphics_view = self.ui.gisaxsInputGraphicsView
            processed_data, _ = self._prepare_image_data_for_display(image_data)
            processed_data = np.flipud(processed_data)
            preview_data, _ = self._downsample_for_preview(processed_data)
            show_q_axis = self._should_show_q_axis()
            extent, q_ok = self._preview_extent(image_data.shape, show_q_axis)
            if show_q_axis and not q_ok:
                show_q_axis = False
                extent, _ = self._preview_extent(image_data.shape, False)
            vmin = self._current_vmin if self._current_vmin is not None else np.min(processed_data)
            vmax = self._current_vmax if self._current_vmax is not None else np.max(processed_data)
            needs_create = (
                self._figure_cache is None or self._canvas_cache is None or
                self._preview_ax is None or self._preview_image_artist is None
            )
            mode_changed = self._preview_shape != image_data.shape or self._preview_show_q_axis != show_q_axis

            if needs_create:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                img_height, img_width = image_data.shape
                aspect_ratio = img_width / img_height
                base_size = 6
                fig_width = max(base_size if aspect_ratio > 1 else base_size * aspect_ratio, 3)
                fig_height = max(base_size / aspect_ratio if aspect_ratio > 1 else base_size, 2.5)
                self._figure_cache = Figure(figsize=(fig_width, fig_height), dpi=72)
                self._canvas_cache = FigureCanvas(self._figure_cache)
                self._preview_ax = self._figure_cache.add_subplot(111)
                if self._graphics_scene is None:
                    self._graphics_scene = QGraphicsScene()
                    graphics_view.setScene(self._graphics_scene)
                else:
                    self._graphics_scene.clear()
                self._preview_proxy_widget = self._graphics_scene.addWidget(self._canvas_cache)

            ax = self._preview_ax
            if needs_create or mode_changed:
                ax.clear()
                self._preview_selection_artists = []
                self._preview_image_artist = ax.imshow(preview_data, cmap='viridis', aspect='equal',
                                                       origin='lower', interpolation='nearest',
                                                       vmin=vmin, vmax=vmax, extent=extent)
                if show_q_axis:
                    ax.set_xlabel(r'$q_y$ (nm$^{-1}$)')
                    ax.set_ylabel(r'$q_z$ (nm$^{-1}$)')
                    ax.autoscale()
                else:
                    ax.set_xlabel('Pixels (Horizontal)')
                    ax.set_ylabel('Pixels (Vertical)')
                    ax.axis('off')
                    ax.set_xlim(-0.5, image_data.shape[1] - 0.5)
                    ax.set_ylim(-0.5, image_data.shape[0] - 0.5)
                self._figure_cache.tight_layout(pad=0.05)
            else:
                self._preview_image_artist.set_data(preview_data)
                self._preview_image_artist.set_extent(extent)
                self._preview_image_artist.set_clim(vmin, vmax)

            self._draw_preview_selection(ax, selection_info)
            render_start = time.perf_counter()
            self._canvas_cache.draw()
            print(f"[Timing] Matplotlib rendering: {(time.perf_counter() - render_start) * 1000:.2f} ms (Detector Preview)")
            if self._preview_proxy_widget is not None:
                self._fit_view_to_item(graphics_view, self._preview_proxy_widget, keep_aspect=True)
            self._preview_shape = image_data.shape
            self._preview_show_q_axis = show_q_axis
            print(f"[Timing] preview rendering: {(time.perf_counter() - t_total) * 1000:.2f} ms")
            return True
        except Exception as e:
            self.status_updated.emit(f"Preview cache update failed: {str(e)}")
            return False

    def _update_graphics_view_with_selection(self, image_data, selection_info=None):
        """GraphicsView"""
        try:
            self._expand_right_card('detectorPreviewCard')
            if not is_matplotlib_available():
                self.status_updated.emit("matplotlib not available for image display")
                return

            graphics_view = self.ui.gisaxsInputGraphicsView
            if self._try_update_cached_preview(image_data, selection_info):
                return

            # ?????????scene?????????????
            if self._graphics_scene is None:
                self._graphics_scene = QGraphicsScene()
                graphics_view.setScene(self._graphics_scene)
            else:
                self._graphics_scene.clear()

            # ??????????????
            img_height, img_width = image_data.shape
            aspect_ratio = img_width / img_height

            # ???figure?????
            base_size = 6
            if aspect_ratio > 1:
                fig_width = base_size
                fig_height = base_size / aspect_ratio
            else:
                fig_height = base_size
                fig_width = base_size * aspect_ratio

            # ??????????
            fig_width = max(fig_width, 3)
            fig_height = max(fig_height, 2.5)

            # ???figure?????PI??????????????atplotlib??
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                # ??????matplotlib????????
                return
            fig = Figure(figsize=(fig_width, fig_height), dpi=72)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # ????????????????????
            processed_data, is_log = self._prepare_image_data_for_display(image_data)

            # ???????????????????????
            processed_data = np.flipud(processed_data)

            # ???????????????
            show_q_axis = self._should_show_q_axis()

            # ????????????????min/vmax
            vmin = self._current_vmin if self._current_vmin is not None else np.min(processed_data)
            vmax = self._current_vmax if self._current_vmax is not None else np.max(processed_data)

            if show_q_axis:
                # Q?????????????xtent?????
                try:
                    # ?????????????????????extent
                    qy_mesh, qz_mesh = self._get_cached_q_meshgrids()

                    if qy_mesh is not None and qz_mesh is not None:
                        # ???Q??????????extent [left, right, bottom, top]
                        qy_min, qy_max = qy_mesh.min(), qy_mesh.max()
                        qz_min, qz_max = qz_mesh.min(), qz_mesh.max()
                        q_extent = [qy_min, qy_max, qz_min, qz_max]

                        im = ax.imshow(processed_data, cmap='viridis', aspect='equal', origin='lower',
                                      interpolation='nearest', vmin=vmin, vmax=vmax, extent=q_extent)

                        # ???Q?????
                        ax.set_xlabel(r'$q_y$ (nm$^{-1}$)')
                        ax.set_ylabel(r'$q_z$ (nm$^{-1}$)')
                    else:
                        # ???Q????????????????????
                        show_q_axis = False
                except Exception as e:
                    pass
                    show_q_axis = False

            if not show_q_axis:
                # ?????????
                im = ax.imshow(processed_data, cmap='viridis', aspect='equal', origin='lower',
                              interpolation='nearest', vmin=vmin, vmax=vmax)
                # ???????????
                ax.set_xlabel('Pixels (Horizontal)')
                ax.set_ylabel('Pixels (Vertical)')

            # ?????????????????????????
            if selection_info:
                bounds = selection_info.get('bounds', {})
                x_min = bounds.get('x_min', 0)
                x_max = bounds.get('x_max', 0)
                y_min = bounds.get('y_min', 0)
                y_max = bounds.get('y_max', 0)

                # ?????????
                from matplotlib.patches import Rectangle
                selection_rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
                )
                ax.add_patch(selection_rect)

                # ??????????
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                ax.plot(center_x, center_y, 'r+', markersize=10, markeredgewidth=2)

            if not show_q_axis:
                ax.axis('off')

            # ??????????
            fig.tight_layout(pad=0.05)

            # ?????????
            if show_q_axis:
                # Q?????matplotlib???????????extent?????
                ax.autoscale()
            else:
                # ???????????????????
                ax.set_xlim(-0.5, processed_data.shape[1] - 0.5)
                ax.set_ylim(-0.5, processed_data.shape[0] - 0.5)

            # ???canvas
            canvas.draw()

            # ????????
            proxy_widget = self._graphics_scene.addWidget(canvas)

            # ??????????????????????????
            self._fit_view_to_item(graphics_view, proxy_widget, keep_aspect=True)

            mode_text = "Log" if self._is_log_mode_enabled() else "Linear"
            coord_mode = "Q-space" if show_q_axis else "Pixel coordinates"
            selection_text = " with selection" if selection_info else ""
            self.status_updated.emit(f"{mode_text} image displayed ({coord_mode}){selection_text} (Double-click to open independent window)")

        except Exception as e:
            self.status_updated.emit(f"Display error: {str(e)}")

    # ========== ?????????????????==========

    def _calculate_vmin_vmax(self, image_data, use_log=True):
        """min??max???1%??9%"""
        try:
            t0 = time.perf_counter()
            if use_log:
                safe_data = np.where(image_data > 0, image_data, 0.001)
                log_data = np.log(safe_data)
                vmin = np.percentile(log_data, 1)
                vmax = np.percentile(log_data, 99)
            else:
                vmin = np.percentile(image_data, 1)
                vmax = np.percentile(image_data, 99)

            print(f"[Timing] autoscale calculation: {(time.perf_counter() - t0) * 1000:.2f} ms")
            return vmin, vmax
        except Exception:
            return None, None

    def _update_vmin_vmax_ui(self, vmin, vmax):
        """No description."""
        try:
            if vmin is not None and vmax is not None:
                self._updating_color_scale_ui = True
                try:
                    if hasattr(self.ui, 'gisaxsInputVminValue'):
                        self.ui.gisaxsInputVminValue.setValue(float(vmin))
                    if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                        self.ui.gisaxsInputVmaxValue.setValue(float(vmax))
                finally:
                    self._updating_color_scale_ui = False

                self._current_vmin = float(vmin)
                self._current_vmax = float(vmax)
                self._refresh_vmin_vmax_display()
        except Exception:
            pass

    def _get_vmin_vmax_from_ui(self):
        """No description."""
        try:
            vmin = None
            vmax = None

            if hasattr(self.ui, 'gisaxsInputVminValue'):
                vmin = self.ui.gisaxsInputVminValue.value()
            if hasattr(self.ui, 'gisaxsInputVmaxValue'):
                vmax = self.ui.gisaxsInputVmaxValue.value()

            return vmin, vmax
        except Exception:
            return None, None

    def _handle_color_scale(self, image_data):
        """No description."""
        try:
            is_auto_scale = self._is_auto_scale_enabled()
            is_first_image = not self._has_displayed_image
            is_log = self._is_log_mode_enabled()

            if is_first_image:
                # ?????????????AutoScale????????????
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                self._has_displayed_image = True

            elif is_auto_scale:
                # ???????????AutoScale?????????
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)

            else:
                # ???????????AutoScale???????UI???????
                vmin, vmax = self._get_vmin_vmax_from_ui()
                self._current_vmin = vmin
                self._current_vmax = vmax

        except Exception:
            # ??????????????????
            try:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
            except Exception:
                pass

    def _is_auto_scale_enabled(self):
        """No description."""
        return self._get_checkbox_state('gisaxsInputAutoScaleCheckBox', True)

    def _is_log_mode_enabled(self):
        """og"""
        return self._get_checkbox_state('gisaxsInputIntLogCheckBox', True)

    def _on_color_scale_value_committed(self, *args):
        """Apply manually edited vmin/vmax values to all image views."""
        try:
            if self._updating_color_scale_ui or getattr(self, '_initializing', False):
                return

            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmin is None or vmax is None:
                return

            vmin = float(vmin)
            vmax = float(vmax)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                self.status_updated.emit("Invalid color scale values")
                return
            if vmax <= vmin:
                self.status_updated.emit("Invalid color scale: vmax must be greater than vmin")
                return

            if hasattr(self.ui, 'gisaxsInputAutoScaleCheckBox') and self.ui.gisaxsInputAutoScaleCheckBox.isChecked():
                self.ui.gisaxsInputAutoScaleCheckBox.blockSignals(True)
                self.ui.gisaxsInputAutoScaleCheckBox.setChecked(False)
                self.ui.gisaxsInputAutoScaleCheckBox.blockSignals(False)

            self._current_vmin = vmin
            self._current_vmax = vmax
            self._refresh_vmin_vmax_display()
            if self.current_stack_data is not None:
                self._refresh_image_display()
            self.status_updated.emit(f"Color scale updated: Vmin={vmin:.3f}, Vmax={vmax:.3f}")
        except Exception as e:
            self.status_updated.emit(f"Color scale update error: {str(e)}")

    def _on_auto_scale_changed(self):
        """No description."""
        try:
            is_enabled = self._is_auto_scale_enabled()
            self.status_updated.emit(f"AutoScale {'enabled' if is_enabled else 'disabled'}")

            if is_enabled and self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(self.current_stack_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                    self._refresh_image_display()
        except Exception as e:
            self.status_updated.emit(f"AutoScale change error: {str(e)}")

    def _on_vmin_value_changed(self):
        """No description."""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmin is not None:
                self._current_vmin = vmin
                if self.current_stack_data is not None:
                    self._refresh_image_display()
        except Exception:
            pass

    def _on_vmax_value_changed(self):
        """No description."""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmax is not None:
                self._current_vmax = vmax
                if self.current_stack_data is not None:
                    self._refresh_image_display()
        except Exception:
            pass

    def _on_auto_show_changed(self):
        """AutoShow"""
        auto_show = hasattr(self.ui, 'gisaxsInputAutoShowCheckBox') and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
        self.status_updated.emit(f"AutoShow {'enabled' if auto_show else 'disabled'}")
        # In-situ ????????
        try:
            if getattr(self, 'load_mode', 'Single') == 'In-situ':
                if auto_show:
                    self._start_insitu_timer()
                else:
                    self._stop_insitu_timer()
        except Exception:
            pass

    def _on_log_changed(self):
        """Log???????????? - """
        try:
            is_log = self._is_log_mode_enabled()

            # ???Vmin/Vmax???????????
            self._refresh_vmin_vmax_display()

            # ????????????????????vmin/vmax????????
            if self.current_stack_data is not None:
                if self._is_auto_scale_enabled():
                    vmin, vmax = self._calculate_vmin_vmax(self.current_stack_data, use_log=is_log)
                    if vmin is not None and vmax is not None:
                        self._update_vmin_vmax_ui(vmin, vmax)

                # ?????????
                self._refresh_image_display()

            self.status_updated.emit(f"*** DISPLAY MODE CHANGED TO: {'LOG' if is_log else 'LINEAR'} ***")

        except Exception as e:
            self.status_updated.emit(f"Log mode change error: {str(e)}")

    def _on_fit_display_option_changed(self):
        """No description."""
        try:
            # ??????????????????
            if getattr(self, '_initializing', True):
                return

            # ???itCurrentDataCheckBox?????
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # fitCurrentDataCheckBox?????????????ut???
                self._perform_cut()
                self.status_updated.emit("Fit display options changed - Cut results updated")
            else:
                # fitCurrentDataCheckBox??????????????????????D???
                if self.current_1d_data is not None and hasattr(self, 'q') and self.q is not None:
                    mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
                    self._update_GUI_image(mode)
                    self._update_outside_window(mode)
                    self.status_updated.emit("Fit display options changed - 1D data display updated")
                else:
                    self.status_updated.emit("Fit display options changed - no data to update")

        except Exception as e:
            self.status_updated.emit(f"Fit display option change error: {str(e)}")

    def _on_current_data_checkbox_changed(self, checked):
        """No description."""
        try:
            # ??????????????????
            if getattr(self, '_initializing', True):
                return

            if checked:
                # ???????????Cut????????GISAXS????????
                if self.current_stack_data is not None:
                    # ??ISAXS????????ut???
                    self._perform_cut()
                    self.status_updated.emit("Current Data enabled - Cut operation performed")
                else:
                    # ???GISAXS???????????
                    self.status_updated.emit("Current Data enabled - No GISAXS data available for cut operation")
            else:
                # ?????????????D????????????????
                if self.current_1d_data is not None:
                    # ??1D?????,I?????
                    self.q = self.current_1d_data['q']
                    self.I = self.current_1d_data['I']
                    self.data_source = '1d'
                    self.display_mode = 'normal'

                    # ?????????
                    self._update_GUI_image('normal')
                    self._update_outside_window('normal')
                    self.status_updated.emit("Current Data disabled - 1D data restored")
                else:
                    # ???1D??????????????
                    self._clear_fit_graphics_view()
                    # ???????????????????
                    if hasattr(self, 'independent_fit_window') and self.independent_fit_window is not None and self.independent_fit_window.isVisible():
                        self.independent_fit_window.ax.clear()
                        self.independent_fit_window.canvas.draw()
                    self.status_updated.emit("Current Data disabled - No 1D data available")

        except Exception as e:
            self.status_updated.emit(f"Current Data checkbox change error: {str(e)}")

    # ========== ????????? ==========

    def _import_1d_file(self):
        """No description."""
        try:
            # ???????????????????
            from core.global_params import global_params
            fitting_session = global_params.get_parameter('fitting', 'last_session', {})
            last_1d_directory = fitting_session.get('last_1d_directory')

            # ??????
            if last_1d_directory and os.path.exists(last_1d_directory):
                start_directory = last_1d_directory
            else:
                start_directory = os.getcwd()  # ????????

            # ??????????????
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Select 1D SAXS Data File",
                start_directory,
                "Data Files (*.dat *.txt);;All Files (*)"
            )

            # ??????????????
            if not file_path:
                return
            file_path = normalize_path(file_path)

            # ????????????????
            self.current_1d_file_path = file_path

            # ?????????????
            current_directory = os.path.dirname(file_path)
            fitting_session['last_1d_directory'] = current_directory
            global_params.set_parameter('fitting', 'last_session', fitting_session)

            # ??????
            self._load_1d_data(file_path)

        except Exception as e:
            self.status_updated.emit(f"Failed to import 1D file: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to import 1D file:\n{str(e)}"
            )

    def _on_1d_file_value_changed(self):
        """No description."""
        try:
            if not hasattr(self.ui, 'fitImport1dFileValue'):
                return

            file_path_input = self.ui.fitImport1dFileValue.text().strip()

            if not file_path_input:
                self.status_updated.emit("No file path entered")
                return

            # ???global_params
            from core.global_params import global_params

            # ????????????????????????????????????
            if not os.path.isabs(file_path_input):
                # ???????????????????
                fitting_session = global_params.get_parameter('fitting', 'last_session', {})
                last_1d_directory = fitting_session.get('last_1d_directory')

                if last_1d_directory and os.path.exists(last_1d_directory):
                    file_path_input = os.path.join(last_1d_directory, file_path_input)
                else:
                    file_path_input = os.path.join(os.getcwd(), file_path_input)

            # ???????????
            if not os.path.exists(file_path_input):
                QMessageBox.warning(
                    self.main_window,
                    "File Not Found",
                    f"File does not exist:\n{file_path_input}"
                )
                return

            # ???????????
            file_ext = os.path.splitext(file_path_input)[1].lower()
            if file_ext not in ['.dat', '.txt']:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid File Type",
                    f"Only .dat and .txt files are supported.\nSelected: {file_ext}"
                )
                return

            # ?????????????????
            self.ui.fitImport1dFileValue.setText(file_path_input)

            # ????????
            self.current_1d_file_path = file_path_input

            # ?????????????
            current_directory = os.path.dirname(file_path_input)
            fitting_session = global_params.get_parameter('fitting', 'last_session', {})
            fitting_session['last_1d_directory'] = current_directory
            global_params.set_parameter('fitting', 'last_session', fitting_session)

            # ??????
            self._load_1d_data(file_path_input)

        except Exception as e:
            self.status_updated.emit(f"Failed to process 1D file path: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to process 1D file path:\n{str(e)}"
            )

    def _load_1d_data(self, file_path):
        """No description."""
        try:
            # ?????????
            from utils.load_SAXS_data import load_xy_any

            # ??????
            self.status_updated.emit(f"Loading 1D data from {os.path.basename(file_path)}...")
            data = load_xy_any(file_path)

            # ????????q,I ?????
            self.q = data.q
            self.I = data.I

            # ???????????????
            self.current_1d_data = {
                'q': data.q,
                'I': data.I,
                'err': getattr(data, 'err', None) if hasattr(data, 'err') else None,
                'file_path': file_path,
                'q_source_unit': self._imported_1d_q_unit
            }

            # ?????D???????????????normal
            self.data_source = '1d'
            self.display_mode = 'normal'
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(False)
                self.ui.fitCurrentDataCheckBox.blockSignals(False)

            # ???fitImport1dFileValue???
            if hasattr(self.ui, 'fitImport1dFileValue'):
                self.ui.fitImport1dFileValue.setText(file_path)

            # ????????ROI???????????
            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass
            self._apply_roi_to_data_and_refresh()
            # ??????
            self._update_GUI_image('normal')
            self._update_outside_window('normal')

            self.status_updated.emit(f"Successfully loaded 1D data: {os.path.basename(file_path)} ({len(self.q)} points)")

        except Exception as e:
            self.status_updated.emit(f"Failed to load 1D data: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Failed to load 1D data from {os.path.basename(file_path)}:\n{str(e)}")



    def _expand_right_card(self, card_attr: str) -> None:
        try:
            card = getattr(self.ui, card_attr, None)
            if card is not None and hasattr(card, 'set_expanded'):
                card.set_expanded(True)
        except Exception:
            pass

    def _setup_fit_graphics_scene(self):
        """itGraphicsView"""
        try:
            self._expand_right_card('fittingPlotCard')
            if not hasattr(self.ui, 'fitGraphicsView'):
                return None

            # ??????????
            if not hasattr(self, '_fit_graphics_scene') or self._fit_graphics_scene is None:
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
                # ????????????????
                self._fit_graphics_scene.clear()

            return self._fit_graphics_scene

        except Exception as e:
            self.status_updated.emit(f"Failed to setup fit graphics scene: {str(e)}")
            return None

    def _fit_view_to_item(self, graphics_view, item, keep_aspect=True):
        """Fit the view to the given item bounds; disable scrollbars by sizing the scene to the item."""
        try:
            scene = graphics_view.scene()
            if scene is None or item is None:
                return
            scene.setSceneRect(item.sceneBoundingRect())
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
            if not hasattr(self.ui, 'fitGraphicsView'):
                return

            scene = self._setup_fit_graphics_scene()
            if scene is not None:
                scene.clear()

            self.status_updated.emit("Fit graphics view cleared")

        except Exception as e:
            self.status_updated.emit(f"Failed to clear fit graphics view: {str(e)}")

    def _start_fitting(self):
        """No description."""
        if not self.current_parameters.get('imported_gisaxs_file'):
            QMessageBox.warning(
                self.parent,
                "Warning",
                "Please import a GISAXS file before processing."
            )
            return

        try:
            self.status_updated.emit("Start Cut Fitting Processing...")
            self.progress_updated.emit(0)

            # TODO: ??????????????
            self._run_fitting_process()

            self.progress_updated.emit(100)
            self.status_updated.emit("Cut Fitting processing complete!")

        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Cut Fitting Error",
                f"Cut fitting failed:\n{str(e)}"
            )

    def _run_fitting_process(self):
        """No description."""
        # TODO: ??????????????
        # ??????????
        # 1. ??????
        # 2. ????
        # 3. ??????
        # 4. ??????
        pass

    def _reset_fitting(self):
        """No description."""
        self._set_default_parameters()

    def _auto_find_center(self):
        """GISAXS"""
        # ????I?????????????????
        self._sync_ui_to_parameters()

        # ???????????????????
        if self.current_stack_data is None:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "Please import an image first."
            )
            return

        # ?????????????????
        if not self._has_displayed_image:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "Please display the image first by clicking the Show button."
            )
            return

        try:
            self.status_updated.emit("Searching for the center point automatically...")

            # ??????????????
            data = np.log10(np.maximum(self.current_stack_data, 1))

            # 1. ?????(center_y): ?????????????????????
            vertical_profile = np.sum(data, axis=1)  # ????????
            raw_center_y = np.argmax(vertical_profile)  # ????????
            # ???????????flipud???????????
            height = data.shape[0]
            pixel_center_y = float(height - 1 - raw_center_y)

            # 2. ????(center_x): ??????????????????????????
            horizontal_profile = np.sum(data, axis=0)  # ????????
            pixel_center_x = float(np.sum(np.arange(len(horizontal_profile)) * horizontal_profile) / np.sum(horizontal_profile))

            # 3. ????????0pixel
            pixel_cutline_height = 20.0

            # 4. Estimate cut-line width from the horizontal intensity profile.
            pixel_cutline_width = float(self._calculate_95_percent_width(horizontal_profile))

            self.ui.gisaxsInputCenterVerticalValue.setValue(pixel_center_y)
            self.ui.gisaxsInputCenterParallelValue.setValue(pixel_center_x)
            self.ui.gisaxsInputCutLineVerticalValue.setValue(pixel_cutline_height)
            self.ui.gisaxsInputCutLineParallelValue.setValue(pixel_cutline_width)

            selection_info = self._create_selection_from_parameters(
                pixel_center_x,
                pixel_center_y,
                pixel_cutline_width,
                pixel_cutline_height,
            )
            self._update_parameter_selection_display(selection_info)

            self.status_updated.emit(
                f"Auto center found: center=({pixel_center_x:.1f}, {pixel_center_y:.1f}), "
                f"cut size=({pixel_cutline_width:.1f}, {pixel_cutline_height:.1f})"
            )

            # 5. ?????ut??????????????? normal????????????????
            self.data_source = 'cut'
            # ????????Normal ???????????????????????
            try:
                if hasattr(self, '_switch_to_normal_display_mode') and callable(self._switch_to_normal_display_mode):
                    self._switch_to_normal_display_mode()
                else:
                    # ?????????????????
                    self.display_mode = 'normal'
                    if hasattr(self, '_display_mode'):
                        self._display_mode = 'normal'
                    if hasattr(self, '_fitting_mode_active'):
                        self._fitting_mode_active = False
                # ????????????????? display_mode ????? normal
                self.display_mode = 'normal'
            except Exception:
                # ???????????????display_mode ??normal
                self.display_mode = 'normal'
            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                try:
                    self.ui.fitCurrentDataCheckBox.blockSignals(True)
                    self.ui.fitCurrentDataCheckBox.setChecked(True)
                finally:
                    try:
                        self.ui.fitCurrentDataCheckBox.blockSignals(False)
                    except Exception:
                        pass

            # ????????ROI????????????????????
            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass
            self._apply_roi_to_data_and_refresh()
            self._update_GUI_image('normal')
            self._update_outside_window('normal')

        except Exception as e:
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Cut operation failed:\n{str(e)}")

    def _calculate_95_percent_width(self, profile):
        """Return the profile width containing roughly 95 percent of the intensity."""
        profile = np.asarray(profile, dtype=float)
        if profile.size == 0:
            return 50.0

        total_intensity = float(np.sum(profile))
        if total_intensity <= 0:
            return 50.0

        center_idx = int(np.argmax(profile))
        target_intensity = total_intensity * 0.95
        left_idx = center_idx
        right_idx = center_idx
        current_intensity = float(profile[center_idx])

        while current_intensity < target_intensity and (left_idx > 0 or right_idx < profile.size - 1):
            left_val = profile[left_idx - 1] if left_idx > 0 else 0
            right_val = profile[right_idx + 1] if right_idx < profile.size - 1 else 0

            if left_val >= right_val and left_idx > 0:
                left_idx -= 1
                current_intensity += float(profile[left_idx])
            elif right_idx < profile.size - 1:
                right_idx += 1
                current_intensity += float(profile[right_idx])
            else:
                break

        width = right_idx - left_idx + 1
        min_width = 20.0
        max_width = profile.size * 0.8
        return float(max(min_width, min(width, max_width)))

    def _show_detector_parameters(self):
        """Show the detector parameters dialog."""
        try:
            if getattr(self, 'detector_params_dialog', None) is not None:
                if self.detector_params_dialog.isVisible():
                    self.detector_params_dialog.raise_()
                    self.detector_params_dialog.activateWindow()
                    return

            self.detector_params_dialog = DetectorParametersDialog(self.main_window)
            self.detector_params_dialog.parameters_changed.connect(self._on_detector_parameters_changed)
            self.detector_params_dialog.finished.connect(self._on_detector_dialog_finished)
            self.detector_params_dialog.show()
            self.detector_params_dialog.raise_()
            self.detector_params_dialog.activateWindow()

            self.status_updated.emit("Detector Parameters dialog opened")

        except Exception as e:
            self.status_updated.emit(f"Failed to display Detector Parameters dialog: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Detector Parameters dialog cannot be displayed:\n{str(e)}"
            )

    def _on_detector_dialog_finished(self):
        """Clear detector dialog reference after close."""
        try:
            self.detector_params_dialog = None
            self.status_updated.emit("Detector Parameters dialog closed")
        except Exception as e:
            self.status_updated.emit(f"Failed to clear detector dialog: {str(e)}")

    def _on_detector_parameters_changed(self, parameters=None):
        """Handle detector parameter changes from the dialog."""
        try:
            self._update_cutline_labels_units()
            self._update_cutline_step_sizes()

            if hasattr(self, '_update_parameter_values_for_q_axis'):
                self._update_parameter_values_for_q_axis()

            try:
                self._compute_q_meshgrids_and_store()
            except Exception:
                pass

            if (self.current_cut_data is not None and
                    getattr(self, 'current_stack_data', None) is not None):
                self._perform_cut()
                self.status_updated.emit("Detector parameters updated; Cut results recalculated")
            else:
                self.status_updated.emit("Detector parameters updated and saved")

        except Exception as e:
            self.status_updated.emit(f"Failed to process detector parameter change: {str(e)}")

    def _update_GUI_image(self, mode):
        """UI"""
        try:
            if not self._has_valid_data():
                return

            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            filter_mode = self._get_independent_axis_filter_mode()
            positive_only = (filter_mode == 'positive')
            negative_only = (filter_mode == 'negative')

            # ???????????OI???????????
            q_data, I_data = self._get_roi_active_arrays()
            if q_data is None or I_data is None:
                return
            q_data, I_data = self._filter_ai_excluded_points_for_display(q_data, I_data)

            q_data, q_plot, I_data, _ = self._filter_q_data_for_independent_display(q_data, I_data)
            if q_data.size == 0 or I_data is None or I_data.size == 0:
                return
            q_data_display = self._convert_q_values_for_display(q_data)
            q_plot = self._convert_q_values_for_display(q_plot)

            # ?????????????????????????????????????????????????????
            norm_factor = 1.0
            if normalize:
                max_I = np.max(I_data) if I_data.size > 0 else 0.0
                if max_I > 0:
                    norm_factor = float(max_I)
                    I_data = I_data / norm_factor

            # ??????
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return

            # ??? 4:3 ????????????
            fig = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            fitting_y_for_limits = None
            extra_y_for_limits = []

            # Normal?????????????????????
            if mode == 'normal' and log_x and not positive_only and not negative_only:
                # ????????????
                positive_mask = q_data > 0
                negative_mask = q_data < 0
                zero_mask = q_data == 0

                # ????????????????????
                if np.any(positive_mask):
                    ax.plot(q_data_display[positive_mask], I_data[positive_mask], 'o-',
                           color='blue', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q>0)' if self.data_source else 'Data (q>0)', zorder=2)

                # ???????????????????????q|??
                if np.any(negative_mask):
                    ax.plot(np.abs(q_data_display[negative_mask]), I_data[negative_mask], 'o-',
                           color='red', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q<0, |q|)' if self.data_source else 'Data (q<0, |q|)', zorder=2)

                # ???q=0??????????
                if np.any(zero_mask):
                    ax.plot(q_data_display[zero_mask], I_data[zero_mask], 'o',
                           color='green', markersize=6, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q=0)' if self.data_source else 'Data (q=0)', zorder=3)
            else:
                # ???????????????
                ax.scatter(q_plot, I_data, s=30, alpha=0.7, color='blue',
                          label=f'{self.data_source.upper()} Data' if self.data_source else 'Data', zorder=2)
            # ROI ?????
            self._draw_roi_guides_if_active(ax)

            # ???????????G??es??articles??????????????????????????????fitting????????
            try:
                show_bg = self._get_checkbox_state('fitBGShowCheckBox', False)
                show_res = self._get_checkbox_state('fitResShowCheckBox', False)
                particle_flags = self._get_particle_sequence_flags()
                show_any = show_bg or show_res or any(particle_flags.values())
            except Exception:
                particle_flags = {}
                show_any = False

            # ???????????????????
            norm_divisor = norm_factor if normalize and norm_factor > 0 else None
            if mode == 'fitting' and show_any:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if shapes and params_list:
                    try:
                        from utils.fitting import mixed_model_components
                        q_model = self._convert_q_values_for_model(q_data, source=self.data_source)
                        comp = mixed_model_components(shapes, q_model, params_list)
                        # BG
                        if show_bg and comp.get('BG_total') is not None:
                            y_bg = comp['BG_total'] / norm_divisor if norm_divisor else comp['BG_total']
                            ax.plot(q_plot, y_bg, linestyle='--', color='#666666', linewidth=1.5, label='bg', zorder=2)
                            extra_y_for_limits.append(y_bg)
                        # Resolution function
                        if show_res and comp.get('resolution') is not None:
                            y_res = comp['resolution'] / norm_divisor if norm_divisor else comp['resolution']
                            ax.plot(q_plot, y_res, linestyle='--', color='#8E44AD', linewidth=1.5, label='Res.', zorder=2)
                            extra_y_for_limits.append(y_res)
                        # Particles
                        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
                        for item in comp.get('particles', []):
                            idx = int(item.get('index', 0))
                            if particle_flags.get(idx, False):
                                yv = item.get('I')
                                if yv is not None:
                                    shape_name = str(item.get('shape', 'Particle')).capitalize()
                                    widget_id = self._sequence_index_to_widget_id(idx)
                                    color_key = widget_id if widget_id is not None else idx
                                    color = colors[(color_key - 1) % len(colors)] if color_key else colors[(idx-1) % len(colors)]
                                    yv_plot = yv / norm_divisor if norm_divisor else yv
                                    label_id = f"{shape_name} {widget_id}" if widget_id is not None else f"{shape_name} {idx}"
                                    ax.plot(q_plot, yv_plot, linestyle='--', color=color, linewidth=1.5, label=label_id, zorder=2)
                                    extra_y_for_limits.append(yv_plot)
                        # ?????????????
                        ax.legend()
                    except Exception:
                        pass

            #??????????????_plot_fitting_result ??????????????????????????

            # ?????itting??????????????????????????OI???????????
            if mode == 'fitting' and self.has_fitting_data and self.I_fitting is not None:
                I_fitting_arr = np.asarray(self.I_fitting)
                q_full = np.asarray(self.q)

                mask_full = np.isfinite(q_full)
                if self._roi_active():
                    mask_full &= (q_full >= self._roi_min) & (q_full <= self._roi_max)
                if positive_only:
                    mask_full &= (q_full > 0)
                elif negative_only:
                    mask_full &= (q_full < 0)

                q_fit_raw = q_full[mask_full]
                I_fitting_data = I_fitting_arr[mask_full]
                _, q_fit_plot, I_fitting_data, _ = self._filter_q_data_for_independent_display(q_fit_raw, I_fitting_data)
                q_fit_plot = self._convert_q_values_for_display(q_fit_plot)

                if normalize and norm_factor > 0 and I_fitting_data.size > 0:
                    I_fitting_data = I_fitting_data / norm_factor

                plot_len = min(len(q_fit_plot), len(I_fitting_data))
                if plot_len > 0:
                    fitting_y_for_limits = I_fitting_data[:plot_len]
                    ax.plot(q_fit_plot[:plot_len], I_fitting_data[:plot_len], color='red', linewidth=2,
                           label='Fitting', zorder=3)

            # ??????????
            x_label = self._build_q_axis_label(filter_mode=filter_mode)
            if mode == 'normal' and log_x and not positive_only and not negative_only and np.any(q_data < 0):
                x_label = self._build_q_axis_label(filter_mode='all', absolute=True)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Normalized Intensity' if normalize else 'Intensity (a.u.)')
            ax.set_title(f'{mode.capitalize()} Mode - {self.data_source.upper() if self.data_source else "Data"}')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # ?????????
            self._apply_log_scales(ax, log_x, log_y)
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=I_data,
                fitting_y=fitting_y_for_limits,
                extra_y_values=extra_y_for_limits,
                log_y=log_y,
            )

            fig.tight_layout()

            # ????????
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)

            # ??????
            self._current_fit_figure = fig
            self._current_fit_canvas = canvas

        except Exception:
            pass

    def _update_outside_window(self, mode):
        """No description."""
        try:
            if not hasattr(self, 'independent_fit_window') or self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                return

            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            # ??????????????????????
            filter_mode = self._get_independent_axis_filter_mode()
            positive_only = (filter_mode == 'positive')
            negative_only = (filter_mode == 'negative')

            # ?????Fitting Plot ????????????????OI???????????????????
            q_data, I_data = self._get_roi_active_arrays()

            if q_data is None or I_data is None or len(q_data) == 0 or len(I_data) == 0:
                return
            q_data, I_data = self._filter_ai_excluded_points_for_display(q_data, I_data)

            q_data, q_plot, I_data, _ = self._filter_q_data_for_independent_display(q_data, I_data)
            if q_data.size == 0 or I_data is None or I_data.size == 0:
                return
            q_data_display = self._convert_q_values_for_display(q_data)
            q_plot = self._convert_q_values_for_display(q_plot)

            # ?????????????????????????????????????????????
            norm_factor = 1.0
            if normalize:
                max_I = np.max(I_data) if I_data.size > 0 else 0.0
                if max_I > 0:
                    norm_factor = float(max_I)
                    I_data = I_data / norm_factor

            ax = self.independent_fit_window.ax
            ax.clear()
            fitting_y_for_limits = None
            extra_y_for_limits = []

            # Normal??????????????????????????????????????
            if mode == 'normal' and log_x and not positive_only and not negative_only:
                # ????????????
                positive_mask = q_data > 0
                negative_mask = q_data < 0
                zero_mask = q_data == 0

                # ????????????????????
                if np.any(positive_mask):
                    ax.plot(q_data_display[positive_mask], I_data[positive_mask], 'o-',
                           color='blue', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q>0)' if self.data_source else 'Data (q>0)', zorder=2)

                # ???????????????????????q|??
                if np.any(negative_mask):
                    ax.plot(np.abs(q_data_display[negative_mask]), I_data[negative_mask], 'o-',
                           color='red', markersize=4, linewidth=1, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q<0, |q|)' if self.data_source else 'Data (q<0, |q|)', zorder=2)

                # ???q=0??????????
                if np.any(zero_mask):
                    ax.plot(q_data_display[zero_mask], I_data[zero_mask], 'o',
                           color='green', markersize=6, alpha=0.8,
                           label=f'{self.data_source.upper()} Data (q=0)' if self.data_source else 'Data (q=0)', zorder=3)
            else:
                # ???????????????
                ax.scatter(q_plot, I_data, s=30, alpha=0.7, color='blue',
                          label=f'{self.data_source.upper()} Data' if self.data_source else 'Data', zorder=2)
            # ROI ?????
            self._draw_roi_guides_if_active(ax)

            # ???????????G??es??articles??????fitting??????
            try:
                show_bg = self._get_checkbox_state('fitBGShowCheckBox', False)
                show_res = self._get_checkbox_state('fitResShowCheckBox', False)
                particle_flags = self._get_particle_sequence_flags()
                show_any = show_bg or show_res or any(particle_flags.values())
            except Exception:
                particle_flags = {}
                show_any = False

            if mode == 'fitting' and show_any:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if shapes and params_list:
                    try:
                        from utils.fitting import mixed_model_components
                        q_model = self._convert_q_values_for_model(q_data, source=self.data_source)
                        comp = mixed_model_components(shapes, q_model, params_list)
                        norm_divisor = norm_factor if normalize and norm_factor > 0 else None
                        # BG
                        if show_bg and comp.get('BG_total') is not None:
                            y_bg = comp['BG_total'] / norm_divisor if norm_divisor else comp['BG_total']
                            ax.plot(q_plot, y_bg, linestyle='--', color='#666666', linewidth=1.5, label='bg', zorder=2)
                            extra_y_for_limits.append(y_bg)
                        # Resolution function
                        if show_res and comp.get('resolution') is not None:
                            y_res = comp['resolution'] / norm_divisor if norm_divisor else comp['resolution']
                            ax.plot(q_plot, y_res, linestyle='--', color='#8E44AD', linewidth=1.5, label='Res.', zorder=2)
                            extra_y_for_limits.append(y_res)
                        # Particles
                        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
                        for item in comp.get('particles', []):
                            idx = int(item.get('index', 0))
                            if particle_flags.get(idx, False):
                                yv = item.get('I')
                                if yv is not None:
                                    shape_name = str(item.get('shape', 'Particle')).capitalize()
                                    widget_id = self._sequence_index_to_widget_id(idx)
                                    color_key = widget_id if widget_id is not None else idx
                                    color = colors[(color_key - 1) % len(colors)] if color_key else colors[(idx-1) % len(colors)]
                                    yv_plot = yv / norm_divisor if norm_divisor else yv
                                    label_id = f"{shape_name} {widget_id}" if widget_id is not None else f"{shape_name} {idx}"
                                    ax.plot(q_plot, yv_plot, linestyle='--', color=color, linewidth=1.5, label=label_id, zorder=2)
                                    extra_y_for_limits.append(yv_plot)
                        ax.legend()
                    except Exception:
                        pass

            # ?????itting????????????????????????????????????????????
            if mode == 'fitting' and self.has_fitting_data and self.I_fitting is not None:
                I_fitting_arr = np.asarray(self.I_fitting)
                q_full = np.asarray(self.q)
                # Build mask to align with displayed q_data (ROI + axis filter)
                mask_full = np.isfinite(q_full)
                if self._roi_active():
                    mask_full &= (q_full >= self._roi_min) & (q_full <= self._roi_max)
                if positive_only:
                    mask_full &= (q_full > 0)
                elif negative_only:
                    mask_full &= (q_full < 0)

                q_fit_raw = q_full[mask_full]
                I_fitting_data = I_fitting_arr[mask_full]
                q_fit_raw, q_fit_plot, I_fitting_data, _ = self._filter_q_data_for_independent_display(q_fit_raw, I_fitting_data)
                q_fit_plot = self._convert_q_values_for_display(q_fit_plot)

                # ?????????????????????
                if normalize and norm_factor > 0 and I_fitting_data.size > 0:
                    I_fitting_data = I_fitting_data / norm_factor

                # Trim/pad safety: align length with plotted q values
                plot_len = min(len(q_fit_plot), len(I_fitting_data))
                if plot_len > 0:
                    fitting_y_for_limits = I_fitting_data[:plot_len]
                    ax.plot(q_fit_plot[:plot_len], I_fitting_data[:plot_len], color='red', linewidth=2.5,
                           label='Fitting', zorder=3)

            # ??????????
            x_label = self._build_q_axis_label(filter_mode=filter_mode)
            if mode == 'normal' and log_x and not positive_only and not negative_only and np.any(np.array(self.q) < 0):
                x_label = self._build_q_axis_label(filter_mode='all', absolute=True)

            ax.set_xlabel(x_label)
            ax.set_ylabel('Normalized Intensity' if normalize else 'Intensity (a.u.)')
            ax.set_title(f'{mode.capitalize()} Mode - {self.data_source.upper() if self.data_source else "Data"}')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # ???????????
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)

            # ?????????
            self._apply_log_scales(ax, log_x, log_y)
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=I_data,
                fitting_y=fitting_y_for_limits,
                extra_y_values=extra_y_for_limits,
                log_y=log_y,
            )

            try:
                if hasattr(self.independent_fit_window, 'set_deletable_points'):
                    self.independent_fit_window.set_deletable_points(q_data, q_plot, I_data)
            except Exception:
                pass

            # ??????
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.draw()

        except Exception:
            pass

    def _get_cut_center_coordinates(self):
        """No description."""
        center_x = 0.0
        center_y = 0.0

        if hasattr(self.ui, 'gisaxsInputCenterParallelValue'):
            center_x = self.ui.gisaxsInputCenterParallelValue.value()
        if hasattr(self.ui, 'gisaxsInputCenterVerticalValue'):
            center_y = self.ui.gisaxsInputCenterVerticalValue.value()

        return center_x, center_y

    def _resolve_cut_points(self, points_override: int = None) -> int:
        """Resolve the target cut point count from override, UI, cache, or settings."""
        if points_override is not None and not isinstance(points_override, bool):
            try:
                return max(10, int(points_override))
            except Exception:
                pass

        try:
            if hasattr(self.ui, 'fitDataPointsNumValue'):
                widget = self.ui.fitDataPointsNumValue
                if hasattr(widget, 'lineEdit') and widget.lineEdit() is not None:
                    text_value = widget.lineEdit().text().strip()
                    if text_value:
                        parsed = int(float(text_value))
                        if parsed >= 10:
                            return parsed
                if hasattr(widget, 'interpretText'):
                    widget.interpretText()
                if hasattr(widget, 'value'):
                    return max(10, int(widget.value()))
                if hasattr(widget, 'text') and str(widget.text()).strip():
                    return max(10, int(float(str(widget.text()).strip())))
        except Exception:
            pass

        current = getattr(self, '_points_num_current', None)
        if isinstance(current, (int, float)):
            try:
                return max(10, int(current))
            except Exception:
                pass

        try:
            from core.user_settings import user_settings
            return max(10, int(user_settings.get('fit.points_num', self._points_num_default)))
        except Exception:
            return max(10, int(getattr(self, '_points_num_default', 300)))

    def _perform_cut(self, points_override: int = None):
        """Execute the current Cut operation using existing horizontal/vertical cut logic."""
        try:
            if self.current_stack_data is None:
                QMessageBox.warning(self.main_window, "Warning", "Please import an image first.")
                return

            vertical_value = self.ui.gisaxsInputCutLineVerticalValue.value() if hasattr(self.ui, 'gisaxsInputCutLineVerticalValue') else 0.0
            parallel_value = self.ui.gisaxsInputCutLineParallelValue.value() if hasattr(self.ui, 'gisaxsInputCutLineParallelValue') else 0.0

            if vertical_value <= 0 or parallel_value <= 0:
                QMessageBox.warning(self.main_window, "Warning", "Please select a valid region.")
                return

            n_points_cut = self._resolve_cut_points(points_override)
            self._points_num_current = int(n_points_cut)

            if vertical_value <= parallel_value:
                self._perform_horizontal_cut(vertical_value, parallel_value, points_override=n_points_cut)
                self.status_updated.emit(
                    f"Horizontal cut performed: Vertical={vertical_value:.2f}, "
                    f"Parallel={parallel_value:.2f}, Points={n_points_cut}"
                )
            else:
                self._perform_vertical_cut(vertical_value, parallel_value, points_override=n_points_cut)
                self.status_updated.emit(
                    f"Vertical cut performed: Vertical={vertical_value:.2f}, "
                    f"Parallel={parallel_value:.2f}, Points={n_points_cut}"
                )

            self.data_source = 'cut'
            try:
                self._switch_to_normal_display_mode()
            except Exception:
                self.display_mode = 'normal'
                if hasattr(self, '_display_mode'):
                    self._display_mode = 'normal'
                if hasattr(self, '_fitting_mode_active'):
                    self._fitting_mode_active = False

            if hasattr(self.ui, 'fitCurrentDataCheckBox'):
                try:
                    self.ui.fitCurrentDataCheckBox.blockSignals(True)
                    self.ui.fitCurrentDataCheckBox.setChecked(True)
                finally:
                    self.ui.fitCurrentDataCheckBox.blockSignals(False)

            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass

            self._apply_roi_to_data_and_refresh()
            self._update_GUI_image('normal')
            self._update_outside_window('normal')

        except Exception as e:
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Cut operation failed:\n{str(e)}")

    def _perform_cut_operation(self, vertical_value, parallel_value, cut_type: str, points_override: int = None):
        """

        Args:
            vertical_value: ??????????????
            parallel_value: ?????????????
            cut_type: ????????horizontal' ??'vertical'
        """
        try:
            # ???????????????
            center_x, center_y = self._get_cut_center_coordinates()

            # ????????Q??????
            show_q_axis = self._should_show_q_axis()

            # ???????????????
            if cut_type == 'horizontal':
                q_mode_method = self._extract_horizontal_cut_q_mode
                pixel_mode_method = self._extract_horizontal_cut_pixel_mode
                pixel_to_q_method = self._convert_pixel_to_qy
                x_label = r'$q_y$ (nm$^{-1}$)'
                title = "Horizontal Cut"
            elif cut_type == 'vertical':
                q_mode_method = self._extract_vertical_cut_q_mode
                pixel_mode_method = self._extract_vertical_cut_pixel_mode
                pixel_to_q_method = self._convert_pixel_to_qz
                x_label = r'$q_z$ (nm$^{-1}$)'
                title = "Vertical Cut"
            else:
                raise Exception(f"Unknown cut type: {cut_type}")

            if show_q_axis:
                # Q?????????????????
                cut_data, q_coords = q_mode_method(
                    center_x, center_y, vertical_value, parallel_value, points_override=points_override
                )
                x_coordinates = q_coords
            else:
                # ???????????????????????????
                cut_data, pixel_coords = pixel_mode_method(
                    center_x, center_y, vertical_value, parallel_value, points_override=points_override
                )
                # ?????????????
                x_coordinates = pixel_to_q_method(pixel_coords)

            # ??????
            self._plot_cut_result(x_coordinates, cut_data, x_label, "Intensity (a.u.)", title)

        except Exception as e:
            raise Exception(f"{cut_type.capitalize()} cut failed: {str(e)}")

    def _perform_horizontal_cut(self, vertical_value, parallel_value, points_override: int = None):
        """No description."""
        self._perform_cut_operation(vertical_value, parallel_value, 'horizontal', points_override=points_override)

    def _perform_vertical_cut(self, vertical_value, parallel_value, points_override: int = None):
        """No description."""
        self._perform_cut_operation(vertical_value, parallel_value, 'vertical', points_override=points_override)

    def _extract_cut_q_mode(self, center_qy, center_qz, height_q, width_q, cut_type: str, points_override: int = None):
        """Q?????????????????????

        Args:
            center_qy, center_qz: Q????????
            height_q, width_q: Q????????
            cut_type: ????????horizontal' ??'vertical'
        """
        try:
            # ???Q???
            qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
            if qy_mesh is None or qz_mesh is None:
                raise Exception("Q-space meshgrids not available")

            # ????????????
            qy_min = center_qy - width_q / 2
            qy_max = center_qy + width_q / 2
            qz_min = center_qz - height_q / 2
            qz_max = center_qz + height_q / 2

            # ?????????
            mask = ((qy_mesh >= qy_min) & (qy_mesh <= qy_max) &
                    (qz_mesh >= qz_min) & (qz_mesh <= qz_max))

            # ???????????
            region_data = np.where(mask, self.current_stack_data, 0)

            # ??????????????????????
            if cut_type == 'horizontal':
                # ?????????????????
                intensity_sum = np.sum(region_data, axis=0)
                q_line = qy_mesh[0, :]  # ???????qy??
            elif cut_type == 'vertical':
                # ?????????????????
                intensity_sum = np.sum(region_data, axis=1)
                q_line = qz_mesh[:, 0]  # ???????qz??
            else:
                raise Exception(f"Unknown cut type: {cut_type}")

            # ???????????
            valid_indices = np.isfinite(intensity_sum) & (intensity_sum > 0)
            if not np.any(valid_indices):
                raise Exception("No valid data in the selected region")

            valid_q = q_line[valid_indices]
            valid_intensity = intensity_sum[valid_indices]

            n_points = self._resolve_cut_points(points_override)
            q_interp = np.linspace(valid_q.min(), valid_q.max(), n_points)
            method = None
            try:
                method = self.ui.fitInterpolationMethodValue.currentText() if hasattr(self.ui, 'fitInterpolationMethodValue') else self._interp_method_default
            except Exception:
                method = self._interp_method_default
            intensity_interp = self._interpolate_series(valid_q, valid_intensity, q_interp, method)
            try:
                self.status_updated.emit(f"Cut(Q) extracted points: {len(q_interp)} (method={method})")
            except Exception:
                pass
            return intensity_interp, q_interp

        except Exception as e:
            raise Exception(f"Q-mode {cut_type} cut extraction failed: {str(e)}")

    def _extract_horizontal_cut_q_mode(self, center_qy, center_qz, height_q, width_q, points_override: int = None):
        """Q"""
        return self._extract_cut_q_mode(center_qy, center_qz, height_q, width_q, 'horizontal', points_override=points_override)

    def _extract_vertical_cut_q_mode(self, center_qy, center_qz, height_q, width_q, points_override: int = None):
        """Q"""
        return self._extract_cut_q_mode(center_qy, center_qz, height_q, width_q, 'vertical', points_override=points_override)

    def _extract_cut_pixel_mode(self, center_x, center_y, height, width, cut_type: str, points_override: int = None):
        """

        Args:
            center_x, center_y: ?????
            height, width: ?????
            cut_type: ????????horizontal' ??'vertical'
        """
        try:
            img_height, img_width = self.current_stack_data.shape

            # ?????????
            x_min = max(0, int(center_x - width / 2))
            x_max = min(img_width, int(center_x + width / 2))
            y_min = max(0, int(center_y - height / 2))
            y_max = min(img_height, int(center_y + height / 2))

            # ???????????flipud???????????
            y_min_adj = img_height - 1 - y_max
            y_max_adj = img_height - 1 - y_min
            y_min_adj, y_max_adj = max(0, y_min_adj), min(img_height, y_max_adj)

            # ?????????
            region_data = self.current_stack_data[y_min_adj:y_max_adj+1, x_min:x_max+1]

            if region_data.size == 0:
                raise Exception("Empty region selected")

            # ??????????????
            if cut_type == 'horizontal':
                # ?????????????????
                intensity_sum = np.sum(region_data, axis=0)
                pixel_coords = np.arange(x_min, x_min + len(intensity_sum))
            elif cut_type == 'vertical':
                # ?????????????????
                intensity_sum = np.sum(region_data, axis=1)
                pixel_coords = np.arange(y_min, y_min + len(intensity_sum))
            else:
                raise Exception(f"Unknown cut type: {cut_type}")

            # ???????????????????????
            if len(pixel_coords) > 1:
                n_points = self._resolve_cut_points(points_override)
                # ??????
                finite_mask = np.isfinite(pixel_coords) & np.isfinite(intensity_sum)
                pixel_coords_f = pixel_coords[finite_mask]
                intensity_sum_f = intensity_sum[finite_mask]
                if len(pixel_coords_f) < 2:
                    pixel_coords_f = pixel_coords
                    intensity_sum_f = intensity_sum
                pixel_interp = np.linspace(pixel_coords_f.min(), pixel_coords_f.max(), n_points)
                method = None
                try:
                    method = self.ui.fitInterpolationMethodValue.currentText() if hasattr(self.ui, 'fitInterpolationMethodValue') else self._interp_method_default
                except Exception:
                    method = self._interp_method_default
                intensity_interp = self._interpolate_series(pixel_coords_f, intensity_sum_f, pixel_interp, method)
                try:
                    self.status_updated.emit(f"Cut(Pixel) extracted points: {len(pixel_interp)} (method={method})")
                except Exception:
                    pass
            else:
                pixel_interp = pixel_coords
                intensity_interp = intensity_sum

            return intensity_interp, pixel_interp

        except Exception as e:
            raise Exception(f"Pixel-mode {cut_type} cut extraction failed: {str(e)}")

    def _extract_horizontal_cut_pixel_mode(self, center_x, center_y, height, width, points_override: int = None):
        """No description."""
        return self._extract_cut_pixel_mode(center_x, center_y, height, width, 'horizontal', points_override=points_override)

    def _extract_vertical_cut_pixel_mode(self, center_x, center_y, height, width, points_override: int = None):
        """No description."""
        return self._extract_cut_pixel_mode(center_x, center_y, height, width, 'vertical', points_override=points_override)

    def _get_detector_for_pixel_conversion(self):
        """No description."""
        try:
            from utils.q_space_calculator import create_detector_from_image_and_params
            from core.global_params import GlobalParameterManager

            global_params = GlobalParameterManager()
            height, width = self.current_stack_data.shape

            # ???????????
            pixel_size_x = global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0)
            pixel_size_y = global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0)
            beam_center_x = global_params.get_parameter('fitting', 'detector.beam_center_x', width / 2.0)
            beam_center_y = global_params.get_parameter('fitting', 'detector.beam_center_y', height / 2.0)
            distance = global_params.get_parameter('fitting', 'detector.distance', 2565.0)
            theta_in_deg = global_params.get_parameter('beam', 'grazing_angle', 0.4)
            wavelength = global_params.get_parameter('beam', 'wavelength', 0.1045)

            return create_detector_from_image_and_params(
                image_shape=(height, width),
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                beam_center_x=beam_center_x,
                beam_center_y=beam_center_y,
                distance=distance,
                theta_in_deg=theta_in_deg,
                wavelength=wavelength,
                crop_params=None
            )
        except Exception:
            return None

    def _convert_pixel_coords_to_q(self, pixel_coords, conversion_type: str):
        """

        Args:
            pixel_coords: ?????????
            conversion_type: ???????qy' ??'qz'
        """
        try:
            detector = self._get_detector_for_pixel_conversion()
            if detector is None:
                raise Exception("Failed to create detector")

            height, width = self.current_stack_data.shape
            q_coords = []

            if conversion_type == 'qy':
                # ????y???????????????y?????
                center_y = height / 2.0
                for px in pixel_coords:
                    _, qy, _ = detector.pixel_to_q_space(px, center_y)
                    q_coords.append(qy)
            elif conversion_type == 'qz':
                # ????z???????????????x?????
                center_x = width / 2.0
                for py in pixel_coords:
                    _, _, qz = detector.pixel_to_q_space(center_x, py)
                    q_coords.append(qz)
            else:
                raise Exception(f"Unknown conversion type: {conversion_type}")

            return np.array(q_coords)

        except Exception as e:
            # ?????????????????????????
            self.status_updated.emit(f"Pixel to {conversion_type} conversion failed: {str(e)}")
            return (pixel_coords - pixel_coords.mean()) / pixel_coords.std()

    def _convert_pixel_to_qy(self, pixel_coords):
        """qy"""
        return self._convert_pixel_coords_to_q(pixel_coords, 'qy')

    def _convert_pixel_to_qz(self, pixel_coords):
        """qz"""
        return self._convert_pixel_coords_to_q(pixel_coords, 'qz')

    def _plot_cut_result(self, x_coords, y_intensity, x_label, y_label, title):
        """No description."""
        try:
            # ?????????????q,I ?????????????????
            x_arr = np.asarray(x_coords)
            y_arr = np.asarray(y_intensity)
            finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_arr = x_arr[finite_mask]
            y_arr = y_arr[finite_mask]
            self.q = x_arr
            self.I = y_arr

            # ??????????????????????
            self.current_cut_data = {
                'x_coords': x_arr.copy() if hasattr(x_arr, 'copy') else list(x_arr),
                'y_intensity': y_arr.copy() if hasattr(y_arr, 'copy') else list(y_arr),
                'x_label': x_label,
                'y_label': y_label,
                'title': title,
                'q_source_unit': 'nm'
            }
            # ????????Cut?????????????q?????????????
            try:
                import time
                self.cut = {
                    'q': x_arr.copy() if hasattr(x_arr, 'copy') else np.array(x_arr),
                    'I': y_arr.copy() if hasattr(y_arr, 'copy') else np.array(y_arr),
                    'meta': {
                        'x_label': x_label,
                        'y_label': y_label,
                        'title': title,
                        'timestamp': time.time(),
                        'source': 'cut',
                        'q_source_unit': 'nm',
                        'q_model_unit': 'nm'
                    }
                }
            except Exception:
                # ??????????????????
                self.cut = {'q': x_arr, 'I': y_arr, 'meta': {'source': 'cut'}}

            if not is_matplotlib_available():
                QMessageBox.warning(self.main_window, "Missing Library",
                                  "matplotlib library is required for plotting.\nPlease install it using: pip install matplotlib")
                return

            # ?????????
            options = self._display_manager.get_display_options()

            # ??????????????
            # ???????ut?????_coords?????q????????
            if "q" in x_label.lower():
                # ???x???q????????D?????????
                self._display_manager.plot_1d_data(
                    x_coords, y_intensity, None, title, "cut_data",
                    options['log_x'], options['log_y'], options['normalize']
                )
            else:
                # ???x???????????????????????????
                self._plot_cut_data_legacy(x_coords, y_intensity, x_label, y_label, title, options)

            self.status_updated.emit(f"Cut result plotted: {title}")

        except Exception as e:
            self.status_updated.emit(f"Plot failed: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Plot Error",
                f"Failed to plot cut result:\n{str(e)}"
            )

    def _plot_cut_data_legacy(self, x_coords, y_intensity, x_label, y_label, title, options):
        """No description."""
        try:
            # ??????????
            y_data = np.array(y_intensity)
            if options['normalize']:
                max_intensity = np.max(y_data)
                if max_intensity > 0:
                    y_data = y_data / max_intensity
                    y_label = "Normalized Intensity"

            # ??????
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            # ???????????????
            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return

            # ???matplotlib???
            fig = Figure(figsize=(8, 6), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # ???????????????
            self._plot_cut_data_with_log_handling(ax, x_coords, y_data, options['log_x'], markersize=4, linewidth=1.5)

            # ????????
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_label, fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

            # ???????????
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)

            # ????????????
            if options['log_x']:
                ax.set_xscale('log')
            if options['log_y']:
                ax.set_yscale('log')

            # ??????
            fig.tight_layout()

            # ????????
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)

            # ?????????????????????????????
            if self.independent_fit_window is not None and self.independent_fit_window.isVisible():
                self.independent_fit_window.update_plot(
                    x_coords, y_intensity, x_label, y_label, title,
                    log_x=options['log_x'],
                    log_y=options['log_y'],
                    normalize=options['normalize']
                )

            self.status_updated.emit(f"Cut result plotted: {title}")

        except Exception as e:
            self.status_updated.emit(f"Legacy plot cut data error: {str(e)}")

    def _get_checkbox_state(self, checkbox_name: str, default_value: bool = False) -> bool:
        """No description."""
        try:
            if hasattr(self.ui, checkbox_name):
                checkbox = getattr(self.ui, checkbox_name)
                return checkbox.isChecked()
            return default_value
        except Exception:
            return default_value

    def _is_fit_log_x_enabled(self):
        """No description."""
        return self._get_checkbox_state('fitLogXCheckBox', False)

    def _is_fit_log_y_enabled(self):
        """No description."""
        return self._get_checkbox_state('fitLogYCheckBox', False)

    def _is_fit_norm_enabled(self):
        """No description."""
        # ??itNormCheckBox
        return self._get_checkbox_state('fitNormCheckBox', False)

    def _get_q_display_unit(self):
        """No description."""
        try:
            if (hasattr(self, 'independent_fit_window') and
                self.independent_fit_window is not None and
                hasattr(self.independent_fit_window, 'q_unit_combo')):
                unit = self.independent_fit_window.q_unit_combo.currentData()
                if isinstance(unit, str):
                    unit = unit.lower()
                    if unit in ('angstrom', 'nm'):
                        return unit
                text = str(self.independent_fit_window.q_unit_combo.currentText()).lower()
                if 'ang' in text or 'a^-1' in text:
                    return 'angstrom'
        except Exception:
            pass
        return 'nm'

    def _get_q_source_unit(self, source=None):
        """No description."""
        try:
            if isinstance(source, dict):
                unit = str(source.get('q_source_unit', '')).lower()
                if unit in ('nm', 'angstrom'):
                    return unit
                source = source.get('data_source')

            if source is None:
                source = getattr(self, 'data_source', None)

            source_text = str(source or '').lower()
            if 'cut' in source_text:
                return 'nm'
            if 'fit' in source_text and isinstance(getattr(self, 'fitting', None), dict):
                meta = self.fitting.get('meta', {})
                unit = str(meta.get('q_source_unit', '')).lower()
                if unit in ('nm', 'angstrom'):
                    return unit
            if '1d' in source_text:
                return getattr(self, '_imported_1d_q_unit', 'angstrom')
        except Exception:
            pass
        return getattr(self, '_imported_1d_q_unit', 'angstrom')

    def _get_q_display_scale(self):
        """No description."""
        return 0.1 if self._get_q_display_unit() == 'angstrom' else 1.0

    def _get_q_unit_label(self, mathtext: bool = True):
        """No description."""
        if self._get_q_display_unit() == 'nm':
            return 'nm$^{-1}$' if mathtext else 'nm^-1'
        return r'$\AA^{-1}$' if mathtext else 'Angstrom^-1'

    def _convert_q_values_for_model(self, q_values, source=None):
        """No description."""
        q_arr = np.asarray([] if q_values is None else q_values, dtype=float)
        if q_arr.size == 0:
            return q_arr
        return q_arr * 10.0 if self._get_q_source_unit(source) == 'angstrom' else q_arr

    def _convert_q_values_for_display(self, q_values, source=None):
        """No description."""
        q_nm = self._convert_q_values_for_model(q_values, source=source)
        if q_nm.size == 0:
            return q_nm
        return q_nm * self._get_q_display_scale()

    def _build_q_axis_label(self, filter_mode: str = 'all', absolute: bool = False, mathtext: bool = True):
        """No description."""
        unit_label = self._get_q_unit_label(mathtext=mathtext)
        base = '|q|' if absolute or filter_mode == 'negative' else 'q'
        suffix = ''
        if filter_mode == 'positive':
            suffix = ' [Positive Only]'
        elif filter_mode == 'negative':
            suffix = ' [Negative Only]'
        return f'{base} ({unit_label}){suffix}'

    def _is_positive_only_enabled(self):
        """ositive Only????????????q"""
        for owner, name in (
            (self.ui, 'fitRegionPositiveOnlyCheckBox'),
            (self.ui, 'PositiveOnlyCheckBox'),
            (getattr(self, 'independent_fit_window', None), 'show_positive_cb'),
        ):
            try:
                if owner is not None and hasattr(owner, name) and getattr(owner, name).isChecked():
                    return True
            except Exception:
                pass
        return False

    def _is_negative_only_enabled(self):
        """No description."""
        for owner, name in (
            (self.ui, 'fitRegionNegativeOnlyCheckBox'),
            (getattr(self, 'independent_fit_window', None), 'show_negative_cb'),
        ):
            try:
                if owner is not None and hasattr(owner, name) and getattr(owner, name).isChecked():
                    return True
            except Exception:
                pass
        return False

    def _get_independent_axis_filter_mode(self):
        """No description."""
        if self._is_negative_only_enabled():
            return 'negative'
        if self._is_positive_only_enabled():
            return 'positive'
        return 'all'

    def _sync_axis_filter_controls(self):
        """No description."""
        if getattr(self, '_syncing_axis_filter', False):
            return

        self._syncing_axis_filter = True
        try:
            sender = self.sender()
            mode = self._get_independent_axis_filter_mode()

            positive_widgets = [
                (self.ui, 'fitRegionPositiveOnlyCheckBox'),
                (self.ui, 'PositiveOnlyCheckBox'),
                (getattr(self, 'independent_fit_window', None), 'show_positive_cb'),
            ]
            negative_widgets = [
                (self.ui, 'fitRegionNegativeOnlyCheckBox'),
                (getattr(self, 'independent_fit_window', None), 'show_negative_cb'),
            ]

            if sender is not None:
                for owner, name in positive_widgets:
                    if owner is not None and hasattr(owner, name) and sender is getattr(owner, name):
                        mode = 'positive' if sender.isChecked() else 'all'
                        break
                for owner, name in negative_widgets:
                    if owner is not None and hasattr(owner, name) and sender is getattr(owner, name):
                        mode = 'negative' if sender.isChecked() else 'all'
                        break

            def _set_checked(owner, name, checked):
                try:
                    if owner is None or not hasattr(owner, name):
                        return
                    widget = getattr(owner, name)
                    widget.blockSignals(True)
                    widget.setChecked(bool(checked))
                    widget.blockSignals(False)
                except Exception:
                    pass

            for owner, name in positive_widgets:
                _set_checked(owner, name, mode == 'positive')
            for owner, name in negative_widgets:
                _set_checked(owner, name, mode == 'negative')
        finally:
            self._syncing_axis_filter = False

    def _filter_q_data_for_independent_display(self, q_data, y_data=None):
        """No description."""
        q_arr = np.asarray([] if q_data is None else q_data)
        y_arr = None if y_data is None else np.asarray(y_data)

        finite_mask = np.isfinite(q_arr)
        if y_arr is not None:
            finite_mask &= np.isfinite(y_arr)

        q_arr = q_arr[finite_mask]
        if y_arr is not None:
            y_arr = y_arr[finite_mask]

        filter_mode = self._get_independent_axis_filter_mode()
        if filter_mode == 'positive':
            axis_mask = q_arr > 0
        elif filter_mode == 'negative':
            axis_mask = q_arr < 0
        else:
            axis_mask = np.ones(q_arr.shape, dtype=bool)

        q_raw = q_arr[axis_mask]
        if y_arr is not None:
            y_arr = y_arr[axis_mask]

        q_plot = np.abs(q_raw) if filter_mode == 'negative' else np.array(q_raw, copy=True)
        if q_plot.size > 0 and filter_mode == 'negative':
            sort_idx = np.argsort(q_plot)
            q_raw = q_raw[sort_idx]
            q_plot = q_plot[sort_idx]
            if y_arr is not None:
                y_arr = y_arr[sort_idx]

        return q_raw, q_plot, y_arr, filter_mode

    def _get_fit_y_range_mode(self):
        """No description."""
        try:
            if (hasattr(self, 'independent_fit_window') and
                self.independent_fit_window is not None and
                hasattr(self.independent_fit_window, '_get_y_range_mode')):
                return self.independent_fit_window._get_y_range_mode()
        except Exception:
            pass
        return 'all'

    def _valid_y_values_for_limits(self, y_values, log_y=False):
        """No description."""
        try:
            arr = np.asarray([] if y_values is None else y_values, dtype=float).ravel()
            if arr.size == 0:
                return arr
            mask = np.isfinite(arr)
            if log_y:
                mask &= arr > 0
            return arr[mask]
        except Exception:
            return np.asarray([], dtype=float)

    def _apply_fit_y_axis_limits(self, ax, experimental_y=None, fitting_y=None,
                                 extra_y_values=None, log_y=False):
        """No description."""
        try:
            mode = self._get_fit_y_range_mode()
            y_sources = []

            if mode == 'experimental':
                y_sources.append(experimental_y)
            elif mode == 'fitting':
                y_sources.append(fitting_y)
            else:
                y_sources.extend([experimental_y, fitting_y])
                if extra_y_values:
                    y_sources.extend(extra_y_values)

            valid_parts = [
                self._valid_y_values_for_limits(values, log_y=log_y)
                for values in y_sources
            ]
            valid_parts = [values for values in valid_parts if values.size > 0]
            if not valid_parts:
                return

            values = np.concatenate(valid_parts)
            y_min = float(np.min(values))
            y_max = float(np.max(values))
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                return

            if log_y:
                if y_min <= 0 or y_max <= 0:
                    return
                if y_min == y_max:
                    ax.set_ylim(y_min / 1.5, y_max * 1.5)
                else:
                    ax.set_ylim(y_min / 1.08, y_max * 1.08)
                return

            if y_min == y_max:
                pad = abs(y_min) * 0.08 if y_min != 0 else 1.0
            else:
                pad = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - pad, y_max + pad)
        except Exception:
            pass

    def _normalize_intensity_data(self, I_data):
        """No description."""
        if len(I_data) == 0:
            return I_data
        max_I = np.max(I_data)
        if max_I > 0:
            return I_data / max_I
        return I_data

    def _apply_log_scales(self, ax, log_x=False, log_y=False):
        """No description."""
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

    def _has_valid_data(self):
        """,I"""
        return (hasattr(self, 'q') and hasattr(self, 'I') and
                self.q is not None and self.I is not None and
                len(self.q) > 0 and len(self.I) > 0)

    # =========================================================================
    # FittingTextBrowser ?????????
    # =========================================================================

    def _setup_fitting_text_browser(self):
        """No description."""
        if hasattr(self.ui, 'FittingTextBrowser'):
            # ???status_updated?????ittingTextBrowser
            self.status_updated.connect(self._update_fitting_text_browser)

            # ????ittingTextBrowser
            self.ui.FittingTextBrowser.clear()
            self._add_fitting_message("Fitting Controller initialized", "INFO")
            self._init_fitting_textbrowser_enhancements()

    def _init_fitting_textbrowser_enhancements(self):
        """Initialize FittingTextBrowser enhancements: fixed height, context menu, detachable window."""
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        # Fixed height at 100px per new requirement
        tb.setMinimumHeight(100)
        tb.setMaximumHeight(100)
        if self._fitting_browser_original_height is None:
            self._fitting_browser_original_height = tb.height()
        # ????????????
        tb.setContextMenuPolicy(Qt.CustomContextMenu)
        tb.customContextMenuRequested.connect(self._show_fitting_browser_menu)

    def _show_fitting_browser_menu(self, pos: QPoint):
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        menu = QMenu(tb)
        act_copy_all = QAction("Copy All", menu)
        act_clear = QAction("Clear", menu)
        act_save = QAction("Save Log...", menu)
        act_detach = QAction("Open Detached Window", menu)
        act_set_max_lines = QAction(f"Set Max Lines (Current: {self._fitting_messages_max_lines})", menu)
        menu.addAction(act_copy_all)
        menu.addAction(act_clear)
        menu.addSeparator()
        menu.addAction(act_save)
        menu.addSeparator()
        menu.addAction(act_detach)
        menu.addSeparator()
        menu.addAction(act_set_max_lines)
        chosen = menu.exec_(tb.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_copy_all:
            tb.selectAll(); tb.copy()
        elif chosen == act_clear:
            self.clear_fitting_messages()
        elif chosen == act_save:
            path, _ = QFileDialog.getSaveFileName(tb, "Save Fitting Log", "fitting_log.txt", "Text Files (*.txt)")
            if path:
                self.save_fitting_log(path)
        elif chosen == act_detach:
            self._open_detached_fitting_browser()
        elif chosen == act_set_max_lines:
            val, ok = QInputDialog.getInt(tb, "Max Lines", "Set maximum display lines:", self._fitting_messages_max_lines, 50, 5000, 50)
            if ok:
                self._fitting_messages_max_lines = val
                self._trim_fitting_messages_if_needed()

    def _open_detached_fitting_browser(self):
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        if self._detached_fitting_dialog is not None:
            try:
                self._detached_fitting_dialog.raise_(); self._detached_fitting_dialog.activateWindow(); return
            except Exception:
                self._detached_fitting_dialog = None
        dlg = QDialog(tb)
        dlg.setWindowTitle("Fitting Log (Detached)")
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser(dlg)
        browser.setHtml(tb.toHtml())
        layout.addWidget(browser)
        dlg.resize(640, 420)
        self._detached_fitting_dialog = dlg
        def sync(msg_html):
            try: browser.append(msg_html)
            except Exception: pass
        self._detached_append = sync
        # Allow reopen after close
        dlg.finished.connect(self._on_detached_closed)
        dlg.show()

    def _on_detached_closed(self, *_):
        self._detached_fitting_dialog = None
        self._detached_append = None

    def _trim_fitting_messages_if_needed(self):
        tb = getattr(self.ui, 'FittingTextBrowser', None)
        if tb is None:
            return
        doc = tb.document()
        blocks = doc.blockCount()
        if blocks <= self._fitting_messages_max_lines:
            return
        remove_count = blocks - self._fitting_messages_max_lines
        cursor = tb.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(remove_count):
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        # Trimming notice in English (sync detached if exists)
        notice = f'<span style="color:#888;">(Log trimmed, keeping last {self._fitting_messages_max_lines} lines)</span>'
        tb.append(notice)
        if self._detached_append:
            self._detached_append(notice)


    def _update_fitting_text_browser(self, message: str):
        """No description."""
        if hasattr(self.ui, 'FittingTextBrowser'):
            self._add_fitting_message(message, "STATUS")

    def _add_fitting_message(self, message: str, msg_type: str = "INFO"):
        """ittingTextBrowser"""
        if not hasattr(self.ui, 'FittingTextBrowser'):
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # ???????????????
        color_map = {
            "INFO": "#2E86AB",      # ???
            "STATUS": "#28A745",    # ???
            "WARNING": "#FD7E14",   # ???
            "ERROR": "#DC3545",     # ???
            "SUCCESS": "#198754",   # ?????
            "PARTICLE": "#6F42C1"   # ???????????
        }

        color = color_map.get(msg_type, "#333333")

        # ????????
        formatted_message = f'<span style="color: {color};">[{timestamp}] {msg_type}: {message}</span>'

        # ?????ittingTextBrowser
        self.ui.FittingTextBrowser.append(formatted_message)
        # ??????????
        if self._detached_append:
            try:
                self._detached_append(formatted_message)
            except Exception:
                pass
        # ??????
        self._trim_fitting_messages_if_needed()

        # ??????????
        cursor = self.ui.FittingTextBrowser.textCursor()
        cursor.movePosition(cursor.End)
        self.ui.FittingTextBrowser.setTextCursor(cursor)

    def _add_fitting_warning(self, message: str):
        """No description."""
        self._add_fitting_message(message, "WARNING")

    def _add_fitting_error(self, message: str):
        """No description."""
        self._add_fitting_message(message, "ERROR")

    def _add_fitting_success(self, message: str):
        """No description."""
        self._add_fitting_message(message, "SUCCESS")

    def _add_particle_message(self, message: str):
        """No description."""
        self._add_fitting_message(message, "PARTICLE")

    def clear_fitting_messages(self):
        """Clear fitting messages in both embedded and detached browser."""
        if hasattr(self.ui, 'FittingTextBrowser'):
            self.ui.FittingTextBrowser.clear()
            self._add_fitting_message("Messages cleared", "INFO")
            # Sync detached window
            if self._detached_fitting_dialog is not None:
                for child in self._detached_fitting_dialog.children():
                    if isinstance(child, QTextBrowser):
                        child.clear()
                        child.append('<span style="color:#2E86AB;">[INFO] Messages cleared</span>')

    def get_fitting_messages(self) -> str:
        """No description."""
        if hasattr(self.ui, 'FittingTextBrowser'):
            return self.ui.FittingTextBrowser.toPlainText()
        return ""

    def save_fitting_log(self, filepath: str) -> bool:
        """No description."""
        try:
            content = self.get_fitting_messages()
            if content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._add_fitting_success(f"Fitting log saved to: {filepath}")
                return True
            else:
                self._add_fitting_warning("No messages to save")
                return False
        except Exception as e:
            self._add_fitting_error(f"Failed to save fitting log: {str(e)}")
            return False

    # =========================================================================
    # ???????????- ?????????????????
    # =========================================================================

    def _setup_particle_shape_connector(self):
        """No description."""
        self._initialize_particle_ui_registry()
        if not getattr(self, 'particle_shape_configs', None):
            self.particle_shape_configs = {}

        # ????????? - ???????????????
        self.particle_control_types = {
            shape: [field[1] for field in schema]
            for shape, schema in COMPONENT_PARAMETER_SCHEMAS.items()
        }

        # ??????
        self._setup_particle_connections()

        # ????????????
        self._setup_particle_parameter_connections()

        # ???????????????
        self._setup_global_parameter_connections()

        # ?????????????????????
        self._setup_parameter_ranges()

        # ?????????????????????????
        self._initialize_particle_states()

        # ??????????????
        self._initialize_global_parameters()

        self._add_fitting_success("Particle Shape Connector initialized")

    def _iter_particle_widget_ids(self):
        """No description."""
        return sorted(self.particle_shape_configs.keys()) if getattr(self, 'particle_shape_configs', None) else []

    def _collect_active_particles(self):
        """No description."""
        active_shapes = []
        widget_order = []
        for widget_id in self._iter_particle_widget_ids():
            combo_name = f'fitParticleShapeCombox_{widget_id}'
            if not hasattr(self.ui, combo_name):
                continue
            combobox = getattr(self.ui, combo_name)
            current_text = combobox.currentText().strip() if combobox.currentText() else ''
            if current_text and current_text.lower() != 'none':
                active_shapes.append(current_text.lower())
                widget_order.append(widget_id)
        return active_shapes, widget_order

    def _get_particle_sequence_flags(self):
        """No description."""
        flags = {}
        sequence = getattr(self, '_last_active_particle_ids', []) or []
        for idx, widget_id in enumerate(sequence, 1):
            checkbox_name = f'fitParticle{widget_id}ShowCheckBox'
            flags[idx] = self._get_checkbox_state(checkbox_name, False)
        return flags

    def _sequence_index_to_widget_id(self, seq_index: int):
        sequence = getattr(self, '_last_active_particle_ids', []) or []
        if 1 <= seq_index <= len(sequence):
            return sequence[seq_index - 1]
        return None

    def _initialize_particle_ui_registry(self):
        """No description."""
        try:
            if hasattr(self.ui, 'scrollAreaWidgetContents'):
                self._particle_scroll_container = self.ui.scrollAreaWidgetContents
                self._particle_container_layout = self._particle_scroll_container.layout()
        except Exception:
            self._particle_scroll_container = None
            self._particle_container_layout = None

        add_button = None
        for name in ('addModelButton', 'fitAddModelButton', 'pushButton'):
            if hasattr(self.ui, name):
                add_button = getattr(self.ui, name)
                break
        if add_button:
            self._particle_add_button = add_button
            if not add_button.toolTip():
                add_button.setToolTip("Add particle model")
            if not getattr(add_button, '_particle_handler_connected', False):
                add_button.clicked.connect(self._on_add_particle_clicked)
                add_button._particle_handler_connected = True

        self._prepare_dynamic_show_checkbox_area()

        self.particle_shape_configs = {}
        idx = 1
        while hasattr(self.ui, f'fitParticleWidget_{idx}') and hasattr(self.ui, f'fitParticleShapeCombox_{idx}'):
            self._register_existing_particle_widget(idx)
            idx += 1
        self._next_particle_candidate = idx
        self._schedule_model_parameters_region_refresh()

    def _prepare_dynamic_show_checkbox_area(self):
        if self._dynamic_show_layout is not None:
            return

        preferred_names = ('ParticlesNumWidget', 'fitParticlesNumWidget')
        for name in preferred_names:
            host = getattr(self.ui, name, None)
            if isinstance(host, QWidget):
                layout = host.layout()
                if layout is None:
                    layout = QVBoxLayout(host)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                self._dynamic_show_container = host
                self._dynamic_show_layout = layout
                self._particle_checkbox_host_name = name
                return

        host_widget = getattr(self.ui, 'fitFittingShowWidget', None)
        if host_widget is None:
            return
        base_layout = host_widget.layout()
        if base_layout is None:
            from PyQt5.QtWidgets import QGridLayout
            base_layout = QGridLayout(host_widget)

        self._dynamic_show_container = QWidget(host_widget)
        layout = QVBoxLayout(self._dynamic_show_container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        self._dynamic_show_layout = layout
        self._particle_checkbox_host_name = 'fitFittingShowWidget'

        # ?????????????????????????????????
        if hasattr(base_layout, 'addWidget'):
            if hasattr(base_layout, 'rowCount'):
                row_index = base_layout.rowCount()
                try:
                    base_layout.addWidget(self._dynamic_show_container, row_index, 0, 1, 2)
                    return
                except Exception:
                    pass
            base_layout.addWidget(self._dynamic_show_container)

    def _register_existing_particle_widget(self, widget_id: int):
        widget = getattr(self.ui, f'fitParticleWidget_{widget_id}', None)
        if widget is None:
            return
        self._rebuild_particle_widget_editor(widget, widget_id)
        self._particle_widgets[widget_id] = widget
        if not self._particle_widget_style_template:
            self._particle_widget_style_template = widget.styleSheet()
            self._particle_widget_style_source_name = widget.objectName() or ''
        self._apply_particle_widget_style(widget, widget_id)
        self.particle_shape_configs[widget_id] = self._build_particle_config(widget_id)
        self._install_particle_context_menu(widget, widget_id)

        checkbox = getattr(self.ui, f'fitParticle{widget_id}ShowCheckBox', None)
        if checkbox is not None:
            self._register_particle_show_checkbox(widget_id, checkbox)

    def _build_particle_config(self, widget_id: int) -> dict:
        pages = {
            index: {'name': shape_name, 'page_index': index}
            for index, shape_name in enumerate(COMPONENT_ORDER)
        }
        return {
            'combobox': f'fitParticleShapeCombox_{widget_id}',
            'stack_widget': f'fitParticleStackWidget_{widget_id}',
            'pages': pages
        }

    def _register_particle_show_checkbox(self, widget_id: int, checkbox: QCheckBox = None):
        if checkbox is None:
            checkbox = self._create_particle_show_checkbox(widget_id)
        if checkbox is None:
            return None
        checkbox_name = checkbox.objectName() or f'fitParticle{widget_id}ShowCheckBox'
        checkbox.setObjectName(checkbox_name)
        checkbox.setProperty('particleCheckboxId', widget_id)
        if not hasattr(self.ui, checkbox_name):
            setattr(self.ui, checkbox_name, checkbox)
        if widget_id not in self._particle_show_checkboxes:
            checkbox.toggled.connect(self._on_component_checkbox_changed)
        checkbox.setText(checkbox.text() or f'Particle {widget_id}')
        self._particle_show_checkboxes[widget_id] = checkbox
        return checkbox

    def _create_particle_show_checkbox(self, widget_id: int):
        self._prepare_dynamic_show_checkbox_area()
        parent = self._dynamic_show_container or getattr(self.ui, 'fitFittingShowWidget', None)
        if parent is None:
            return None
        checkbox = QCheckBox(f'Particle {widget_id}', parent)
        checkbox.setObjectName(f'fitParticle{widget_id}ShowCheckBox')
        checkbox.setProperty('particleCheckboxId', widget_id)
        if self._dynamic_show_layout is not None:
            self._insert_particle_checkbox_widget(checkbox, widget_id)
        elif hasattr(parent, 'layout') and parent.layout() is not None:
            parent.layout().addWidget(checkbox)
        return checkbox

    def _insert_particle_checkbox_widget(self, checkbox: QCheckBox, widget_id: int):
        layout = self._dynamic_show_layout
        if layout is None:
            return
        can_insert = hasattr(layout, 'insertWidget')
        inserted = False
        for pos in range(layout.count()):
            item = layout.itemAt(pos)
            if item is None:
                continue
            existing = item.widget()
            if existing is None:
                continue
            existing_id = existing.property('particleCheckboxId')
            if existing_id is None:
                continue
            if widget_id < existing_id and can_insert:
                layout.insertWidget(pos, checkbox)
                inserted = True
                break
        if not inserted:
            layout.addWidget(checkbox)

    def _shape_key(self, shape_name: str) -> str:
        return str(shape_name).strip().lower().replace("-", "_").replace(" ", "_")

    def _shape_object_token(self, shape_name: str) -> str:
        return ''.join(part.capitalize() for part in self._shape_key(shape_name).split('_'))

    def _shape_display_name(self, shape_name: str) -> str:
        shape_key = self._shape_key(shape_name)
        for candidate in COMPONENT_ORDER:
            if self._shape_key(candidate) == shape_key:
                return candidate
        return str(shape_name)

    def _parameter_key_from_alias(self, shape_name: str, param_name: str) -> str:
        """Map fitting-template names (Int, sigma_R, h) to stored parameter keys."""
        alias = str(param_name)
        alias_map = {
            "Int": "intensity",
            "R": "radius",
            "sigma_R": "sigma_radius",
            "h": "height",
            "sigma_h": "sigma_height",
            "D": "diameter",
            "sigma_D": "sigma_diameter",
        }
        if alias in alias_map:
            return alias_map[alias]
        for schema_shape, schema in COMPONENT_PARAMETER_SCHEMAS.items():
            if self._shape_key(schema_shape) != self._shape_key(shape_name):
                continue
            for param_key, suffix, _label, _default, _decimals, _step in schema:
                if alias in (param_key, suffix):
                    return param_key
        return alias

    def _rebuild_particle_widget_editor(self, container: QWidget, widget_id: int) -> None:
        old_layout = container.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                child = item.widget()
                if child is not None:
                    child.setParent(None)
                    child.deleteLater()
            layout = old_layout
        else:
            layout = QVBoxLayout(container)
        container.setMinimumHeight(0)
        container.setMaximumHeight(16777215)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QWidget(container)
        header.setObjectName(f'fitParticleHeader_{widget_id}')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(10)
        title = QLabel(f"Component {widget_id}", header)
        title.setObjectName(f'fitParticleTitleLabel_{widget_id}')
        title.setMinimumWidth(88)
        title.setStyleSheet("font-weight: 600; color: #1f2937;")
        type_group = QWidget(header)
        type_group.setObjectName(f'fitParticleTypeGroup_{widget_id}')
        type_layout = QHBoxLayout(type_group)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.setSpacing(4)
        type_label = QLabel("Type", type_group)
        type_label.setObjectName(f'fitParticleTypeLabel_{widget_id}')
        combo = QComboBox(type_group)
        combo.setObjectName(f'fitParticleShapeCombox_{widget_id}')
        combo.setMinimumWidth(158)
        combo.setMaximumWidth(236)
        for shape_name in COMPONENT_ORDER:
            combo.addItem(shape_name)
            combo.setItemData(combo.count() - 1, COMPONENT_FORMULA_TOOLTIPS[shape_name], Qt.ToolTipRole)
        combo.setToolTip(COMPONENT_FORMULA_TOOLTIPS["None"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(combo)
        remove_button = QPushButton("Remove", header)
        remove_button.setObjectName(f"fitParticleRemoveButton_{widget_id}")
        remove_button.setToolTip("Remove this component")
        remove_button.setMinimumWidth(84)
        remove_button.setMaximumWidth(96)
        remove_button.clicked.connect(lambda _checked=False, wid=widget_id: self._remove_particle_widget(wid))
        header_layout.addWidget(title)
        header_layout.addWidget(type_group)
        header_layout.addStretch(1)
        header_layout.addWidget(remove_button)
        layout.addWidget(header)

        stack = CurrentPageHeightStackedWidget(container)
        stack.setObjectName(f'fitParticleStackWidget_{widget_id}')
        stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        none_page = QWidget(stack)
        none_layout = QVBoxLayout(none_page)
        none_layout.setContentsMargins(4, 6, 4, 6)
        none_label = QLabel("No component selected.", none_page)
        none_label.setToolTip(COMPONENT_FORMULA_TOOLTIPS["None"])
        none_layout.addWidget(none_label)
        none_page.setMaximumHeight(38)
        stack.addWidget(none_page)
        for shape_name in COMPONENT_ORDER[1:]:
            stack.addWidget(self._create_particle_parameter_page(stack, widget_id, shape_name))
        layout.addWidget(stack, 0)
        container.setMinimumSize(420, 0)
        container.setMaximumWidth(16777215)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._register_ui_children(container)
        QTimer.singleShot(0, lambda widget=container: self._sync_particle_widget_height(widget))

    def _create_particle_parameter_page(self, parent: QWidget, widget_id: int, shape_name: str) -> QWidget:
        page = QWidget(parent)
        page.setObjectName(f"fitParticle{self._shape_object_token(shape_name)}Page_{widget_id}")
        grid = QGridLayout(page)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        header_labels = (QLabel("Parameter", page), QLabel("Value", page), QLabel("Step", page))
        for col, header_label in enumerate(header_labels):
            header_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
            grid.addWidget(header_label, 0, col)
        for row, (param_key, suffix, label_text, default_value, decimals, step) in enumerate(COMPONENT_PARAMETER_SCHEMAS[shape_name], 1):
            label = QLabel(label_text, page)
            label.setMinimumHeight(24)
            value = QDoubleSpinBox(page)
            value.setObjectName(f"fitParticle{self._shape_object_token(shape_name)}{suffix}Value_{widget_id}")
            value.setDecimals(decimals)
            value.setRange(-1e10, 1e10)
            value.setSingleStep(step)
            value.setValue(default_value)
            value.setMinimumHeight(26)
            value.setMaximumHeight(28)
            step_box = NoWheelDoubleSpinBox(page)
            step_box.setObjectName(f"fitParticle{self._shape_object_token(shape_name)}{suffix}Step_{widget_id}")
            step_box.setDecimals(6)
            step_box.setRange(1e-9, 1e9)
            step_box.setSingleStep(step)
            step_box.setValue(step)
            step_box.setMinimumHeight(26)
            step_box.setMaximumHeight(28)
            step_box.setMaximumWidth(86)
            step_box.valueChanged.connect(lambda new_step, spin=value: spin.setSingleStep(float(new_step)))
            tooltip = COMPONENT_FORMULA_TOOLTIPS[shape_name]
            label.setToolTip(tooltip)
            value.setToolTip(tooltip)
            step_box.setToolTip(f"Single-step increment for {label_text}")
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            grid.addWidget(step_box, row, 2)
        grid.setColumnStretch(1, 1)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        return page

    def _create_particle_widget(self, widget_id: int) -> QWidget:
        parent = getattr(self, '_particle_scroll_container', getattr(self.ui, 'scrollAreaWidgetContents', self.ui))
        container = QWidget(parent)
        container.setObjectName(f'fitParticleWidget_{widget_id}')
        self._rebuild_particle_widget_editor(container, widget_id)
        self._apply_particle_widget_style(container, widget_id)
        return container

    def _apply_particle_widget_style(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        widget.setStyleSheet(
            "QWidget {"
            "background-color: #ffffff;"
            "color: #172033;"
            "}"
            f"QWidget#{widget.objectName()} {{"
            "background-color: #ffffff;"
            "border: 1px solid #d6deea;"
            "border-radius: 12px;"
            "}"
            f"QWidget#fitParticleHeader_{widget_id} {{"
            "background-color: #f8fbff;"
            "border: 1px solid #e5edf6;"
            "border-radius: 10px;"
            "}"
            f"QWidget#fitParticleTypeGroup_{widget_id} {{"
            "background-color: #ffffff;"
            "border: 1px solid #dbe4f0;"
            "border-radius: 8px;"
            "}"
            f"QLabel#fitParticleTitleLabel_{widget_id} {{"
            "background-color: transparent;"
            "border: none;"
            "color: #1f2937;"
            "font-weight: 700;"
            "padding: 0 2px 0 0;"
            "}"
            f"QLabel#fitParticleTypeLabel_{widget_id} {{"
            "background-color: transparent;"
            "border: none;"
            "color: #526070;"
            "font-weight: 600;"
            "padding-left: 8px;"
            "padding-right: 2px;"
            "}"
            f"QComboBox#fitParticleShapeCombox_{widget_id} {{"
            "border: none;"
            "background-color: transparent;"
            "padding-left: 2px;"
            "padding-right: 24px;"
            "min-height: 28px;"
            "}"
            f"QComboBox#fitParticleShapeCombox_{widget_id}::drop-down {{"
            "border: none;"
            "background-color: transparent;"
            "width: 22px;"
            "subcontrol-origin: padding;"
            "subcontrol-position: top right;"
            "}"
            f"QComboBox#fitParticleShapeCombox_{widget_id}::down-arrow {{"
            "width: 10px;"
            "height: 10px;"
            "}"
            f"QPushButton#fitParticleRemoveButton_{widget_id} {{"
            "background-color: #f3f7fb;"
            "border: 1px solid #cfd9e6;"
            "border-radius: 8px;"
            "color: #334155;"
            "font-weight: 600;"
            "padding: 4px 10px;"
            "}"
            f"QPushButton#fitParticleRemoveButton_{widget_id}:hover {{"
            "background-color: #e8f0f8;"
            "border-color: #b8c7d9;"
            "}"
        )

    def _register_ui_children(self, widget: QWidget):
        if widget is None:
            return
        for child in widget.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self.ui, name, child)
        name = widget.objectName()
        if name:
            setattr(self.ui, name, widget)

    def _on_add_particle_clicked(self):
        try:
            widget_id = self._allocate_particle_id()
            particle_key = f'particle_{widget_id}'
            self.model_params_manager.ensure_particle_entry('fitting', particle_key, shape='Sphere')
            new_widget = self._create_particle_widget(widget_id)
            self._attach_particle_widget(new_widget, widget_id)
            self._add_fitting_success(f"Particle {widget_id} added")
        except Exception as e:
            self._add_fitting_error(f"Failed to add particle widget: {e}")

    def _allocate_particle_id(self) -> int:
        if self._recycled_particle_ids:
            self._recycled_particle_ids.sort()
            return self._recycled_particle_ids.pop(0)
        candidate = getattr(self, '_next_particle_candidate', 1)
        while candidate in self.particle_shape_configs:
            candidate += 1
        self._next_particle_candidate = candidate + 1
        return candidate

    def _attach_particle_widget(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        if self._particle_container_layout is not None and self._particle_add_button is not None:
            index = self._particle_container_layout.indexOf(self._particle_add_button)
            if index == -1:
                self._particle_container_layout.addWidget(widget)
            else:
                self._particle_container_layout.insertWidget(index, widget)
        elif self._particle_container_layout is not None:
            self._particle_container_layout.addWidget(widget)

        self._particle_widgets[widget_id] = widget
        self.particle_shape_configs[widget_id] = self._build_particle_config(widget_id)
        self._install_particle_context_menu(widget, widget_id)
        self._register_particle_show_checkbox(widget_id)

        self._setup_particle_connections([widget_id])
        self._setup_particle_parameter_connections([widget_id])
        self._setup_parameter_ranges([widget_id])
        self._initialize_particle_states([widget_id])
        self._schedule_model_parameters_region_refresh()

    def _sync_particle_widget_height(self, widget: QWidget):
        if widget is None:
            return
        for stack in widget.findChildren(CurrentPageHeightStackedWidget):
            stack.sync_current_height()
        layout = widget.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        height = max(1, widget.sizeHint().height())
        widget.setMinimumHeight(height)
        widget.setMaximumHeight(height)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        widget.updateGeometry()

    def _schedule_model_parameters_region_refresh(self):
        try:
            QTimer.singleShot(0, self._refresh_model_parameters_region_height)
        except Exception:
            self._refresh_model_parameters_region_height()

    def _refresh_model_parameters_region_height(self):
        model_card = self.ui.gisaxsFittingPage.findChild(QWidget, 'ModelParameterCard') if hasattr(self.ui, 'gisaxsFittingPage') else None
        work_splitter = self.ui.gisaxsFittingPage.findChild(QWidget, 'gisaxsMainWorkSplitter') if hasattr(self.ui, 'gisaxsFittingPage') else None
        fixed_controls_stack = self.ui.gisaxsFittingPage.findChild(QWidget, 'gisaxsFixedControlsStack') if hasattr(self.ui, 'gisaxsFittingPage') else None
        work_area_contents = self.ui.gisaxsFittingPage.findChild(QWidget, 'gisaxsWorkAreaContents') if hasattr(self.ui, 'gisaxsFittingPage') else None
        particle_container = getattr(self, '_particle_scroll_container', None)

        if self._particle_container_layout is not None:
            self._particle_container_layout.activate()
        if particle_container is not None:
            particle_container.updateGeometry()
            particle_container.adjustSize()
            for widget in particle_container.findChildren(QWidget):
                if widget.objectName().startswith('fitParticleWidget_'):
                    self._sync_particle_widget_height(widget)

        if model_card is not None:
            base_min_height = model_card.property('baseMinHeight')
            if base_min_height is None:
                base_min_height = model_card.minimumHeight()
                model_card.setProperty('baseMinHeight', base_min_height)
            model_card.layout().activate() if model_card.layout() is not None else None
            model_card.updateGeometry()
            model_min_height = max(int(base_min_height), model_card.sizeHint().height())
            model_card.setMinimumHeight(model_min_height)
            model_card.setMaximumHeight(model_min_height)
            model_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            model_min_height = 0

        if fixed_controls_stack is not None:
            fixed_controls_stack.updateGeometry()
            fixed_controls_stack.adjustSize()
            fixed_min_height = max(fixed_controls_stack.minimumHeight(), fixed_controls_stack.sizeHint().height())
            fixed_controls_stack.setMinimumHeight(fixed_min_height)
        else:
            fixed_min_height = 0

        if work_splitter is not None:
            handle_width = work_splitter.handleWidth() if hasattr(work_splitter, 'handleWidth') else 0
            work_splitter.setMinimumHeight(fixed_min_height + model_min_height + handle_width)
            work_splitter.updateGeometry()
            current_sizes = work_splitter.sizes() if hasattr(work_splitter, 'sizes') else []
            if len(current_sizes) == 2 and current_sizes[1] < model_min_height:
                work_splitter.setSizes([max(current_sizes[0], fixed_min_height), model_min_height])

        if work_area_contents is not None and work_splitter is not None:
            margins = work_area_contents.layout().contentsMargins() if work_area_contents.layout() is not None else None
            vertical_margins = (margins.top() + margins.bottom()) if margins is not None else 0
            work_area_contents.setMinimumHeight(work_splitter.minimumHeight() + vertical_margins)
            work_area_contents.updateGeometry()

    def _install_particle_context_menu(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        widget.setProperty('particle_id', widget_id)
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        try:
            widget.customContextMenuRequested.connect(self._handle_particle_context_menu_request)
        except Exception:
            pass

    def _handle_particle_context_menu_request(self, pos: QPoint):
        widget = self.sender()
        if widget is None:
            return
        widget_id = widget.property('particle_id')
        if widget_id is None:
            return
        global_pos = widget.mapToGlobal(pos)
        self._show_particle_context_menu(int(widget_id), global_pos)

    def _show_particle_context_menu(self, widget_id: int, global_pos: QPoint):
        menu = QMenu(self.ui)
        remove_action = menu.addAction('Remove Particle')
        if len(self._iter_particle_widget_ids()) <= 1:
            remove_action.setEnabled(False)
        action = menu.exec_(global_pos)
        if action == remove_action:
            self._remove_particle_widget(widget_id)

    def _remove_particle_widget(self, widget_id: int):
        if widget_id not in self.particle_shape_configs:
            return
        if len(self._iter_particle_widget_ids()) <= 1:
            self._add_fitting_warning('At least one particle widget must remain')
            return

        widget = self._particle_widgets.pop(widget_id, None)
        if widget is not None:
            try:
                widget.customContextMenuRequested.disconnect(self._handle_particle_context_menu_request)
            except Exception:
                pass
            if self._particle_container_layout is not None:
                self._particle_container_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

        checkbox = self._particle_show_checkboxes.pop(widget_id, None)
        if checkbox is not None:
            checkbox.setParent(None)
            checkbox.deleteLater()
            cb_name = checkbox.objectName()
            if cb_name and hasattr(self.ui, cb_name):
                delattr(self.ui, cb_name)

        meta_ids = self._particle_parameter_meta_ids.pop(widget_id, [])
        for meta_id in meta_ids:
            try:
                self.param_trigger_manager.unregister_widget(meta_id)
            except Exception:
                pass

        self._cleanup_particle_ui_attributes(widget_id)
        self.particle_shape_configs.pop(widget_id, None)
        self._recycled_particle_ids.append(widget_id)
        self.model_params_manager.remove_particle('fitting', f'particle_{widget_id}')
        self.model_params_manager.save_parameters()
        self._last_active_particle_ids = [wid for wid in self._last_active_particle_ids if wid != widget_id]

        try:
            self._update_GUI_image('fitting' if self._is_in_fitting_mode() else 'normal')
        except Exception:
            pass
        self._schedule_model_parameters_region_refresh()
        self._add_fitting_success(f"Particle {widget_id} removed")

    def _cleanup_particle_ui_attributes(self, widget_id: int):
        names = [
            f'fitParticleWidget_{widget_id}',
            f'fitParticleShapeCombox_{widget_id}',
            f'fitParticleStackWidget_{widget_id}',
        ]
        for shape in COMPONENT_PARAMETER_SCHEMAS:
            mapping = self._get_parameter_widget_mapping(widget_id, shape)
            names.extend(mapping.values())
            for widget_name in mapping.values():
                if widget_name.endswith(f"_{widget_id}"):
                    names.append(widget_name.replace("Value_", "Step_"))
        for name in names:
            if hasattr(self.ui, name):
                try:
                    attr = getattr(self.ui, name)
                    if hasattr(attr, 'deleteLater'):
                        attr.deleteLater()
                except Exception:
                    pass
                try:
                    delattr(self.ui, name)
                except Exception:
                    pass

    def _setup_particle_connections(self, widget_ids=None):
        """widget"""
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            config = self.particle_shape_configs[widget_id]
            if hasattr(self.ui, config['combobox']):
                combobox = getattr(self.ui, config['combobox'])

                # ???ComboBox???????????
                combobox.currentIndexChanged.connect(
                    lambda index, wid=widget_id: self._on_particle_shape_changed(wid, index)
                )
                combobox.currentTextChanged.connect(
                    lambda text, combo=combobox: combo.setToolTip(
                        COMPONENT_FORMULA_TOOLTIPS.get(text, COMPONENT_FORMULA_TOOLTIPS["None"])
                    )
                )

                self._add_fitting_message(f"Connected Particle Widget {widget_id}: {config['combobox']} -> {config['stack_widget']}", "INFO")

    def _setup_parameter_ranges(self, widget_ids=None):
        """No description."""
        # ???????????????????????ython float??????
        min_value = -1e10  # ??????
        max_value = 1e10   # ??????
        decimals = 2       # 2????????

        # ???????????
        widgets_set = 0

        # Configure component parameter controls from the dynamic schema.
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            for shape_name, schema in COMPONENT_PARAMETER_SCHEMAS.items():
                mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
                decimals_by_param = {param_key: param_decimals for param_key, _suffix, _label, _default, param_decimals, _step in schema}
                step_by_param = {param_key: step for param_key, _suffix, _label, _default, _decimals, step in schema}
                for param_key, widget_name in mapping.items():
                    if hasattr(self.ui, widget_name):
                        widget = getattr(self.ui, widget_name)
                        widget.setRange(min_value, max_value)
                        widget.setDecimals(decimals_by_param.get(param_key, decimals))
                        widget.setSingleStep(step_by_param.get(param_key, 0.1))
                        widgets_set += 1

        # ??????????????????
        if hasattr(self.ui, 'fitBGValue'):
            self.ui.fitBGValue.setRange(min_value, max_value)
            self.ui.fitBGValue.setDecimals(6)
            self.ui.fitBGValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, 'fitSigmaResValue'):
            self.ui.fitSigmaResValue.setRange(min_value, max_value)
            self.ui.fitSigmaResValue.setDecimals(6)  # Br ???????
            self.ui.fitSigmaResValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, 'fitNuResValue'):
            self.ui.fitNuResValue.setRange(min_value, max_value)
            self.ui.fitNuResValue.setDecimals(4)
            self.ui.fitNuResValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, 'fitIntResValue'):
            self.ui.fitIntResValue.setRange(min_value, max_value)
            self.ui.fitIntResValue.setDecimals(6)
            self.ui.fitIntResValue.setSingleStep(0.01)
            widgets_set += 1

        if hasattr(self.ui, 'fitKValue'):
            self.ui.fitKValue.setRange(min_value, max_value)
            self.ui.fitKValue.setDecimals(4)  # k?????????
            self.ui.fitKValue.setSingleStep(0.1)  # k????????1
            widgets_set += 1

        self._add_fitting_success(f"Set ranges for {widgets_set} parameter widgets: [{min_value}, {max_value}] with {decimals} decimals")

    def _setup_particle_parameter_connections(self, widget_ids=None):
        """No description."""
        from functools import partial
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            for shape_name in COMPONENT_PARAMETER_SCHEMAS:
                mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
                shape_lower = self._shape_key(shape_name)
                for param_key, widget_name in mapping.items():
                    if not hasattr(self.ui, widget_name):
                        continue
                    w = getattr(self.ui, widget_name)
                    def _after_commit(info, value, wid=widget_id, shp=shape_lower, p=param_key):
                        try:
                            self._add_particle_message(f"Meta commit {wid}.{shp}.{p} = {value}")
                            # ???????????????????????????????????????????
                            # ?????????????????(cut ??1d)?????????????
                            has_data = (hasattr(self, 'current_cut_data') and self.current_cut_data is not None) or \
                                       (hasattr(self, 'current_1d_data') and self.current_1d_data is not None)
                            if has_data:
                                # ???????????????????????????????????????
                                try:
                                    self.display_mode = 'fitting'
                                    self._display_mode = 'fitting'
                                    self._fitting_mode_active = True
                                except Exception:
                                    pass
                                # ???????????????????????? 1D ???
                                self._perform_manual_fitting()
                        except Exception:
                            pass
                    widget_mode = self._signal_mode_overrides.get(widget_name, self._default_signal_mode)
                    meta = {
                        'persist': 'model_particle',
                        'particle_id': f'particle_{widget_id}',
                        'shape': shape_lower,
                        'param': param_key,
                        'trigger_fit': True,
                        'debounce_ms': self._param_debounce_ms,
                        'epsilon_abs': self._param_abs_eps,
                        'epsilon_rel': self._param_rel_eps,
                        'after_commit': _after_commit,
                        'connect_mode': widget_mode,
                    }
                    meta_id = f'meta_particle_{widget_id}_{shape_lower}_{param_key}'
                    self.param_trigger_manager.register_parameter_widget(
                        widget=w,
                        widget_id=meta_id,
                        category='fitting_particles',
                        immediate_handler=lambda v: None,
                        delayed_handler=None,
                        connect_signals=True,
                        meta=meta
                    )
                    self._particle_parameter_meta_ids[widget_id].append(meta_id)
                    # ??meta ????????connect_mode ?????

    def _setup_global_parameter_connections(self):
        """No description."""
        from functools import partial
        mapping = [
            ('fitBGValue', 'background'),
            ('fitSigmaResValue', 'sigma_res'),
            ('fitNuResValue', 'nu_res'),
            ('fitIntResValue', 'int_res'),
            ('fitKValue', 'k_value'),
        ]
        for widget_name, param_key in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            w = getattr(self.ui, widget_name)
            def _after_commit(info, value, p=param_key):
                try:
                    self._add_particle_message(f"Meta commit global {p} = {value}")
                    has_data = (hasattr(self, 'current_cut_data') and self.current_cut_data is not None) or \
                               (hasattr(self, 'current_1d_data') and self.current_1d_data is not None)
                    if has_data:
                        # ???????????????????????
                        try:
                            self.display_mode = 'fitting'
                            self._display_mode = 'fitting'
                            self._fitting_mode_active = True
                        except Exception:
                            pass
                        self._perform_manual_fitting()
                except Exception:
                    pass
            widget_mode = self._signal_mode_overrides.get(widget_name, self._default_signal_mode)
            meta = {
                'persist': 'model_global',
                'param': param_key,
                'trigger_fit': True,
                'debounce_ms': self._param_debounce_ms,
                'epsilon_abs': self._param_abs_eps,
                'epsilon_rel': self._param_rel_eps,
                'after_commit': _after_commit,
                'connect_mode': widget_mode,
            }
            self.param_trigger_manager.register_parameter_widget(
                widget=w,
                widget_id=f'meta_global_{param_key}',
                category='fitting_global',
                immediate_handler=lambda v: None,
                delayed_handler=None,
                connect_signals=True,
                meta=meta
            )
            # ??meta ????????connect_mode ?????
            self._add_fitting_message(f"Connected (meta, mode={self._signal_mode_overrides.get(widget_name, self._default_signal_mode)}) {widget_name}", "INFO")


    def _initialize_global_parameters(self):
        """No description."""
        try:
            if hasattr(self.ui, 'fitBGValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'background', 0.0)
                self.ui.fitBGValue.blockSignals(True)
                self.ui.fitBGValue.setValue(saved_value)
                self.ui.fitBGValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitBGValue to {saved_value}", "INFO")

            # ????sigma_res
            if hasattr(self.ui, 'fitSigmaResValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'sigma_res', 0.1)
                self.ui.fitSigmaResValue.blockSignals(True)
                self.ui.fitSigmaResValue.setValue(saved_value)
                self.ui.fitSigmaResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitSigmaResValue to {saved_value}", "INFO")

            # ????nu_res
            if hasattr(self.ui, 'fitNuResValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'nu_res', 5.0)
                self.ui.fitNuResValue.blockSignals(True)
                self.ui.fitNuResValue.setValue(saved_value)
                self.ui.fitNuResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitNuResValue to {saved_value}", "INFO")

            # ????int_res
            if hasattr(self.ui, 'fitIntResValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'int_res', 0.0)
                self.ui.fitIntResValue.blockSignals(True)
                self.ui.fitIntResValue.setValue(saved_value)
                self.ui.fitIntResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitIntResValue to {saved_value}", "INFO")

            # ????k_value
            if hasattr(self.ui, 'fitKValue'):
                saved_value = self.model_params_manager.get_global_parameter('fitting', 'k_value', 1.0)
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(saved_value)
                self.ui.fitKValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitKValue to {saved_value}", "INFO")

        except Exception as e:
            self._add_fitting_error(f"Failed to initialize global parameters: {e}")

    def _initialize_particle_states(self, widget_ids=None):
        """SON"""
        try:
            # ????????????????????
            self._initializing = True

            target_ids = widget_ids or self._iter_particle_widget_ids()
            for widget_id in target_ids:
                particle_id = f"particle_{widget_id}"

                # ??SON??????????????????
                saved_shape = self.model_params_manager.get_particle_shape('fitting', particle_id)
                is_enabled = self.model_params_manager.get_particle_enabled('fitting', particle_id)

                self._add_fitting_message(f"Initializing {particle_id}: shape={saved_shape}, enabled={is_enabled}", "INFO")

                # ???ComboBox????????????????????
                config = self.particle_shape_configs[widget_id]
                if hasattr(self.ui, config['combobox']):
                    combobox = getattr(self.ui, config['combobox'])

                    # ????????ombo index
                    combo_index = None
                    for index, page_config in config['pages'].items():
                        if page_config['name'] == saved_shape:
                            combo_index = index
                            break

                    if combo_index is not None:
                        # ???????????
                        combobox.blockSignals(True)
                        combobox.setCurrentIndex(combo_index)
                        combobox.blockSignals(False)

                        # ??????
                        page_config = config['pages'][combo_index]
                        self._switch_particle_page(widget_id, page_config, saved_shape)

                        # ?????????????
                        if not is_enabled or saved_shape == 'None':
                            # ??????
                            self._freeze_particle_controls(widget_id)
                            self._add_fitting_message(f"{particle_id} controls frozen (disabled/None)", "INFO")
                        else:
                            # ??????????????
                            self._unfreeze_particle_controls(widget_id, saved_shape)
                            self._load_particle_parameters_from_json(widget_id, saved_shape)
                            self._add_fitting_message(f"{particle_id} controls active with {saved_shape} parameters", "INFO")

        except Exception as e:
            self._add_fitting_error(f"Failed to initialize particle states: {e}")
            import traceback
            self._add_fitting_error(f"Traceback: {traceback.format_exc()}")
        finally:
            # ??????????
            self._initializing = False
            self._schedule_model_parameters_region_refresh()

    def _set_particle_page_and_state(self, widget_id: int, combo_index: int, shape_name: str):
        """No description."""
        config = self.particle_shape_configs[widget_id]
        page_config = config['pages'][combo_index]

        if hasattr(self.ui, config['stack_widget']):
            stack_widget = getattr(self.ui, config['stack_widget'])

            # ??????
            stack_widget.setCurrentIndex(page_config['page_index'])

            # ??????????
            if shape_name == 'None':
                self._set_particle_none_state(widget_id)
            else:
                self._set_particle_active_state(widget_id, shape_name)

    def _load_particle_parameters(self, widget_id: int, shape_name: str):
        """UI"""
        if shape_name == 'None':
            return

        try:
            particle_id = f"particle_{widget_id}"
            shape_key = self._shape_key(shape_name)

            # ?????????
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)

            # ??????????????????????
            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)

                    # ?????????????
                    value = self.model_params_manager.get_particle_parameter(
                        'fitting', particle_id, shape_key, param_key
                    )

                    if value is not None:
                        # ???????????????????
                        widget.blockSignals(True)
                        widget.setValue(value)
                        widget.blockSignals(False)

                        self._add_particle_message(f"Loaded {param_key}={value} for particle {widget_id} ({shape_name})")
                    else:
                        self._add_particle_message(f"No value found for {param_key} in particle {widget_id} ({shape_name})")

        except Exception as e:
            self._add_fitting_error(f"Failed to load parameters for particle {widget_id}: {e}")

    def _get_parameter_widget_mapping(self, widget_id: int, shape_name: str) -> dict:
        """Return parameter key to spinbox object-name mapping for a component."""
        for schema_shape, schema in COMPONENT_PARAMETER_SCHEMAS.items():
            if self._shape_key(schema_shape) == self._shape_key(shape_name):
                token = self._shape_object_token(schema_shape)
                return {
                    param_key: f"fitParticle{token}{suffix}Value_{widget_id}"
                    for param_key, suffix, _label, _default, _decimals, _step in schema
                }
        return {}
    def _on_particle_shape_changed(self, widget_id: int, combo_index: int):
        """ - ??????????????SON"""
        config = self.particle_shape_configs[widget_id]
        page_config = config['pages'][combo_index]

        if hasattr(self.ui, config['stack_widget']):
            stack_widget = getattr(self.ui, config['stack_widget'])
            shape_name = page_config['name']

            # ?????????????
            particle_id = f"particle_{widget_id}"
            current_shape = self.model_params_manager.get_particle_shape('fitting', particle_id)

            # ????????????????
            if current_shape == shape_name:
                self._add_particle_message(f"??? Particle {widget_id} already in {shape_name} state, skipping")
                return

            self._add_particle_message(f"?? Particle {widget_id} shape changing: {current_shape} -> {shape_name}")

            # 1. ???JSON?????????????
            if shape_name == 'None':
                # ?????one????disabled????
                self.model_params_manager.set_particle_shape('fitting', particle_id, 'None')
                self.model_params_manager.set_particle_enabled('fitting', particle_id, False)
                self._add_particle_message(f"?? Saved {particle_id} as None (disabled)")
            else:
                # ?????????????????nabled????
                self.model_params_manager.set_particle_shape('fitting', particle_id, shape_name)
                self.model_params_manager.set_particle_enabled('fitting', particle_id, True)
                self._add_particle_message(f"??Saved {particle_id} as {shape_name} (enabled)")

            # ???JSON???
            self.model_params_manager.save_parameters()

            # 2. ???UI???
            self._switch_particle_page(widget_id, page_config, shape_name)

            # 3. ?????????????????
            if shape_name == 'None':
                # None???????????????
                self._freeze_particle_controls(widget_id)
                self._add_particle_message(f"??? Particle {widget_id} controls frozen (None state)")
            else:
                # ?????????????????????
                self._unfreeze_particle_controls(widget_id, shape_name)
                self._load_particle_parameters_from_json(widget_id, shape_name)
                self._add_particle_message(f"?? Particle {widget_id} controls unfrozen ({shape_name} state)")

            self._schedule_model_parameters_region_refresh()

    def _switch_particle_page(self, widget_id: int, page_config: dict, shape_name: str):
        """No description."""
        config = self.particle_shape_configs[widget_id]
        if hasattr(self.ui, config['stack_widget']):
            stack_widget = getattr(self.ui, config['stack_widget'])
            target_page_index = page_config['page_index']
            current_page_index = stack_widget.currentIndex()

            self._add_particle_message(f"?? Switching page: {current_page_index} -> {target_page_index} for {shape_name}")

            # ?????????????????????????
            if target_page_index == current_page_index:
                # ???????????????????????UI???
                temp_page_index = 1 if target_page_index == 0 else 0
                stack_widget.setCurrentIndex(temp_page_index)
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
                stack_widget.setCurrentIndex(target_page_index)
                self._add_particle_message(f"?? Forced refresh: temp({temp_page_index}) -> {target_page_index}")
            else:
                stack_widget.setCurrentIndex(target_page_index)

            if isinstance(stack_widget, CurrentPageHeightStackedWidget):
                stack_widget.sync_current_height()
            stack_widget.updateGeometry()
            parent_widget = stack_widget.parentWidget()
            if parent_widget is not None:
                self._sync_particle_widget_height(parent_widget)
            self._schedule_model_parameters_region_refresh()

            # ????????????
            final_page_index = stack_widget.currentIndex()
            if final_page_index == target_page_index:
                self._add_particle_message(f"??Page switch confirmed: {final_page_index}")
            else:
                self._add_particle_message(f"??Page switch failed! Expected {target_page_index}, got {final_page_index}")

    def _freeze_particle_controls(self, widget_id: int):
        """None"""
        for shape_name in COMPONENT_PARAMETER_SCHEMAS:
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setEnabled(False)

    def _unfreeze_particle_controls(self, widget_id: int, active_shape: str):
        """No description."""
        for shape_name in COMPONENT_PARAMETER_SCHEMAS:
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            is_active = (self._shape_key(shape_name) == self._shape_key(active_shape))

            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setEnabled(is_active)

    def _load_particle_parameters_from_json(self, widget_id: int, shape_name: str):
        """SON??????????????I"""
        try:
            # ????????????????????
            self._loading_parameters = True

            particle_id = f"particle_{widget_id}"
            shape_key = self._shape_key(shape_name)
            shape_params = self.model_params_manager.get_particle_parameter('fitting', particle_id, shape_key)

            if not shape_params:
                self._add_particle_message(f"No parameters found in JSON for {particle_id}.{shape_name}")
                return

            # ?????????
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            loaded_count = 0

            # ???????????I???
            for param_key, widget_name in param_mapping.items():
                if param_key in shape_params and hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    value = shape_params[param_key]
                    widget.setValue(value)
                    loaded_count += 1
                    self._add_particle_message(f"?? Loaded {param_key}={value} for {particle_id}.{shape_name}")

            self._add_particle_message(f"??Loaded {loaded_count} parameters from JSON for {particle_id}.{shape_name}")

        except Exception as e:
            self._add_fitting_error(f"Failed to load parameters from JSON: {e}")
        finally:
            # ?????????
            self._loading_parameters = False









    def set_particle_shape(self, widget_id: int, shape_name: str):
        """
        ????????????

        Args:
            widget_id: ?????? (1, 2, 3)
            shape_name: ?????? ('Sphere', 'Cylinder', 'None')
        """
        config = self.particle_shape_configs.get(widget_id)
        if not config:
            self._add_fitting_warning(f"Particle Widget {widget_id} not found")
            return False

        # ????????ombo index
        combo_index = None
        for index, page_config in config['pages'].items():
            if page_config['name'] == shape_name:
                combo_index = index
                break

        if combo_index is None:
            self._add_fitting_warning(f"Shape {shape_name} not found for particle widget {widget_id}")
            return False

        # ???ComboBox
        if hasattr(self.ui, config['combobox']):
            combobox = getattr(self.ui, config['combobox'])
            combobox.setCurrentIndex(combo_index)
            return True

        return False

    def get_particle_shape(self, widget_id: int) -> str:
        """
        ????????????

        Args:
            widget_id: ?????? (1, 2, 3)

        Returns:
            ?????????
        """
        config = self.particle_shape_configs.get(widget_id)
        if not config:
            return 'None'

        if hasattr(self.ui, config['combobox']):
            combobox = getattr(self.ui, config['combobox'])
            current_index = combobox.currentIndex()

            page_config = config['pages'].get(current_index)
            if page_config:
                return page_config['name']

        return 'None'

    def get_particles_status(self) -> dict:
        """No description."""
        status = {}
        for widget_id in self._iter_particle_widget_ids():
            status[widget_id] = self.get_particle_shape(widget_id)
        return status

    def reset_all_particles(self):
        """No description."""
        for widget_id in self._iter_particle_widget_ids():
            self.set_particle_shape(widget_id, 'None')
        self._add_fitting_success("All particles reset to None state")

    def add_new_particle_shape(self, shape_name: str, control_types: list):
        """
        ??????????????? - ??????

        Args:
            shape_name: ???????? 'Ellipsoid'
            control_types: ??????????? ['Int', 'Ra', 'Rb', 'Rc', 'D', 'BG']
        """
        self.particle_control_types[shape_name] = control_types

        # ?????idget????????????
        for widget_id in self._iter_particle_widget_ids():
            pages = self.particle_shape_configs[widget_id]['pages']
            new_index = len(pages) - 1  # None??????????????????????????

            # ????????????None???????
            none_config = pages.pop(len(pages) - 1)  # ???None

            # ????????
            pages[new_index] = {
                'name': shape_name,
                'page_index': new_index,  # ??????????????????
            }

            # ??????None
            pages[len(pages)] = none_config

        self._add_fitting_success(f"Added new particle shape: {shape_name} with {len(control_types)} controls")
        self._add_fitting_warning("Note: You need to add corresponding UI pages and ComboBox items manually")

    def get_all_particle_parameters(self) -> dict:
        """No description."""
        return self.model_params_manager.get_all_particles('fitting')

    def save_particle_parameters(self) -> bool:
        """No description."""
        return self.model_params_manager.save_parameters()

    def reload_particle_parameters(self) -> bool:
        """I"""
        success = self.model_params_manager.load_parameters()
        if success:
            self._initialize_particle_states()
            self._initialize_global_parameters()
            self._add_fitting_success("Particle and global parameters reloaded from file")
        else:
            self._add_fitting_error("Failed to reload particle parameters")
        return success

    def export_particle_parameters(self, filepath: str) -> bool:
        """No description."""
        try:
            import shutil
            shutil.copy2(self.model_params_manager.config_file, filepath)
            self._add_fitting_success(f"Particle parameters exported to: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to export parameters: {e}")
            return False

    def import_particle_parameters(self, filepath: str) -> bool:
        """No description."""
        try:
            import shutil
            shutil.copy2(filepath, self.model_params_manager.config_file)
            success = self.reload_particle_parameters()
            if success:
                self._add_fitting_success(f"Particle parameters imported from: {filepath}")
            return success
        except Exception as e:
            self._add_fitting_error(f"Failed to import parameters: {e}")
            return False

    def _build_fitting_parameter_snapshot(self) -> dict:
        """Return a portable fitting-parameter snapshot."""
        try:
            self.save_particle_parameters()
        except Exception:
            pass
        model_section = {}
        try:
            model_section = copy.deepcopy(self.model_params_manager.get_parameter('fitting', None, {}))
        except Exception:
            model_section = {}
        return {
            "schema": "gimap_fitting_parameters_v1",
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "fitting": copy.deepcopy(self.get_parameters()),
            "model_parameters": {
                "fitting": model_section,
            },
        }

    def save_fitting_parameters_to_file(self, filepath: str) -> bool:
        """Save only Cut/Fitting parameters, including particle/global model params."""
        try:
            filepath = normalize_path(filepath)
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(self._build_fitting_parameter_snapshot(), fh, indent=4, ensure_ascii=False)
            self._add_fitting_success(f"Fitting parameters saved to: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to save fitting parameters: {e}")
            return False

    def load_fitting_parameters_from_file(self, filepath: str) -> bool:
        """Load a Cut/Fitting parameter snapshot and refresh the fitting UI."""
        try:
            filepath = normalize_path(filepath)
            with open(filepath, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

            fitting_params = payload.get("fitting") if isinstance(payload, dict) else None
            if isinstance(fitting_params, dict):
                self.set_parameters(fitting_params)

            model_fitting = None
            if isinstance(payload, dict):
                model_params = payload.get("model_parameters")
                if isinstance(model_params, dict):
                    model_fitting = model_params.get("fitting")
                if model_fitting is None and ("particles" in payload or "global_parameters" in payload):
                    model_fitting = payload
                if model_fitting is None and "fitting" in payload and isinstance(payload.get("fitting"), dict):
                    maybe = payload["fitting"]
                    if "particles" in maybe or "global_parameters" in maybe:
                        model_fitting = maybe

            if isinstance(model_fitting, dict):
                if not hasattr(self.model_params_manager, "_parameters") or not isinstance(self.model_params_manager._parameters, dict):
                    self.model_params_manager._parameters = {}
                self.model_params_manager._parameters["fitting"] = copy.deepcopy(model_fitting)
                self.model_params_manager.save_parameters()
                self.reload_particle_parameters()

            self.parameters_changed.emit(self.current_parameters)
            self._add_fitting_success(f"Fitting parameters loaded from: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to load fitting parameters: {e}")
            return False

    def save_fitting_parameters_dialog(self) -> bool:
        filepath, _ = QFileDialog.getSaveFileName(
            self.main_window or self.ui,
            "Save Fitting Parameters",
            "config/fitting_parameters.json",
            "JSON Files (*.json);;All Files (*)",
        )
        return self.save_fitting_parameters_to_file(filepath) if filepath else False

    def load_fitting_parameters_dialog(self) -> bool:
        filepath, _ = QFileDialog.getOpenFileName(
            self.main_window or self.ui,
            "Load Fitting Parameters",
            "config/",
            "JSON Files (*.json);;All Files (*)",
        )
        return self.load_fitting_parameters_from_file(filepath) if filepath else False

    def _setup_fitting_parameters_context_menu(self) -> None:
        names = (
            "FittingControlsCard",
            "ModelParameterCard",
            "gisaxsFixedControlsStack",
            "gisaxsWorkAreaContents",
            "sampleParametersBox",
            "fitBox",
            "gisaxsFittingPageScrollAreaWidgetContents",
        )
        for name in names:
            widget = getattr(self.ui, name, None)
            if widget is None:
                root = getattr(self.ui, "centralwidget", None)
                widget = root.findChild(QWidget, name) if root is not None else None
            if widget is None:
                continue
            try:
                widget.setContextMenuPolicy(Qt.CustomContextMenu)
                widget.customContextMenuRequested.connect(self._show_fitting_parameters_context_menu)
            except RuntimeError:
                # Some generated containers are intentionally replaced by the runtime layout wrapper.
                # PyQt keeps the Python attribute even after Qt deletes the C++ object.
                continue
            except Exception:
                pass

    def _show_fitting_parameters_context_menu(self, pos: QPoint) -> None:
        widget = self.sender()
        if widget is None:
            return
        try:
            global_pos = widget.mapToGlobal(pos)
        except RuntimeError:
            return
        menu = QMenu(widget)
        save_action = menu.addAction("Save Fitting Parameters...")
        load_action = menu.addAction("Load Fitting Parameters...")
        menu.addSeparator()
        export_particles_action = menu.addAction("Export Particle Parameters Only...")
        import_particles_action = menu.addAction("Import Particle Parameters Only...")
        reload_action = menu.addAction("Reload Parameters from Config")
        menu.addSeparator()
        ai_action = menu.addAction("Open AI Fitting Workspace...")
        action = menu.exec_(global_pos)
        if action == save_action:
            self.save_fitting_parameters_dialog()
        elif action == load_action:
            self.load_fitting_parameters_dialog()
        elif action == export_particles_action:
            filepath, _ = QFileDialog.getSaveFileName(
                self.main_window or self.ui,
                "Export Particle Parameters",
                "config/model_parameters_fitting.json",
                "JSON Files (*.json);;All Files (*)",
            )
            if filepath:
                self.export_particle_parameters(filepath)
        elif action == import_particles_action:
            filepath, _ = QFileDialog.getOpenFileName(
                self.main_window or self.ui,
                "Import Particle Parameters",
                "config/",
                "JSON Files (*.json);;All Files (*)",
            )
            if filepath:
                self.import_particle_parameters(filepath)
        elif action == reload_action:
            self.reload_particle_parameters()
        elif action == ai_action:
            self.open_ai_fitting_workspace()

    def _ai_fitting_settings(self) -> dict:
        try:
            from core.user_settings import user_settings
            settings = user_settings.get("ai_fitting", {})
            return settings if isinstance(settings, dict) else {}
        except Exception:
            return {}

    def _save_ai_fitting_settings(self, **updates) -> None:
        try:
            from core.user_settings import user_settings
            settings = self._ai_fitting_settings()
            settings.update(updates)
            user_settings.set("ai_fitting", settings)
            user_settings.save_settings()
        except Exception:
            pass

    def _default_ai_run_settings(self) -> dict:
        return {
            "full_num_samples": 2000,
            "full_top_k": 20,
            "full_refine_top_n": 5,
            "full_refine_max_nfev": 80,
            "full_refine_progress_interval": 20,
            "full_refine_ftol": 1e-8,
            "full_refine_xtol": 1e-8,
            "full_refine_gtol": 1e-8,
            "full_refine_stall_patience": 0,
            "full_refine_stall_tol": 1e-4,
            "full_refine_target_logrmse": 0.08,
            "full_sampling_std": 0.005,
            "fast_num_samples": 128,
            "fast_top_k": 20,
            "fast_progress_interval": 16,
            "parameter_constraints": {},
        }

    def _ai_run_settings(self) -> dict:
        settings = self._default_ai_run_settings()
        stored = self._ai_fitting_settings()
        for key in settings:
            if key in stored:
                settings[key] = stored[key]
        return settings

    def _connect_ai_fitting_settings_widgets(self) -> None:
        self._restore_ai_run_settings_to_widgets()
        widget_map = {
            "aiFittingSamplesSpinBox": "full_num_samples",
            "aiFittingRefineTopNSpinBox": "full_refine_top_n",
            "aiFittingRefineMaxEvalSpinBox": "full_refine_max_nfev",
            "aiFittingProgressEverySpinBox": "full_refine_progress_interval",
            "aiFittingRefineFtolSpinBox": "full_refine_ftol",
            "aiFittingRefineXtolSpinBox": "full_refine_xtol",
            "aiFittingRefineGtolSpinBox": "full_refine_gtol",
            "aiFittingSamplingStdSpinBox": "full_sampling_std",
            "aiFittingTargetLogRmseSpinBox": "full_refine_target_logrmse",
        }
        for widget_name, setting_key in widget_map.items():
            widget = getattr(self.ui, widget_name, None)
            if widget is None or widget.property("aiSettingConnected"):
                continue
            widget.valueChanged.connect(lambda value, key=setting_key: self._save_ai_fitting_settings(**{key: value}))
            widget.setProperty("aiSettingConnected", True)

    def _restore_ai_run_settings_to_widgets(self) -> None:
        settings = self._ai_run_settings()
        widget_map = {
            "aiFittingSamplesSpinBox": "full_num_samples",
            "aiFittingRefineTopNSpinBox": "full_refine_top_n",
            "aiFittingRefineMaxEvalSpinBox": "full_refine_max_nfev",
            "aiFittingProgressEverySpinBox": "full_refine_progress_interval",
            "aiFittingRefineFtolSpinBox": "full_refine_ftol",
            "aiFittingRefineXtolSpinBox": "full_refine_xtol",
            "aiFittingRefineGtolSpinBox": "full_refine_gtol",
            "aiFittingSamplingStdSpinBox": "full_sampling_std",
            "aiFittingTargetLogRmseSpinBox": "full_refine_target_logrmse",
        }
        for widget_name, setting_key in widget_map.items():
            widget = getattr(self.ui, widget_name, None)
            if widget is None:
                continue
            try:
                widget.blockSignals(True)
                widget.setValue(settings[setting_key])
            finally:
                widget.blockSignals(False)

    def _sync_workspace_ai_run_widgets(self) -> None:
        settings = self._ai_run_settings()
        workspace_map = {
            "_ai_full_samples_spin": "full_num_samples",
            "_ai_refine_top_n_spin": "full_refine_top_n",
            "_ai_refine_max_nfev_spin": "full_refine_max_nfev",
            "_ai_progress_every_spin": "full_refine_progress_interval",
            "_ai_refine_ftol_spin": "full_refine_ftol",
            "_ai_refine_xtol_spin": "full_refine_xtol",
            "_ai_refine_gtol_spin": "full_refine_gtol",
            "_ai_stall_patience_spin": "full_refine_stall_patience",
            "_ai_stall_tol_spin": "full_refine_stall_tol",
            "_ai_sampling_std_spin": "full_sampling_std",
            "_ai_target_logrmse_spin": "full_refine_target_logrmse",
        }
        for attr, key in workspace_map.items():
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            try:
                widget.blockSignals(True)
                widget.setValue(settings[key])
            finally:
                widget.blockSignals(False)

    def _ai_fitting_base_dirs(self) -> list:
        settings = self._ai_fitting_settings()
        stored = settings.get("model_base_dirs")
        dirs = []
        if isinstance(stored, list):
            dirs.extend(Path(p) for p in stored if isinstance(p, str) and p.strip())
        dirs.extend(default_ai_fitting_model_base_dirs(Path.cwd()))
        extra = settings.get("extra_model_paths")
        if isinstance(extra, list):
            dirs.extend(Path(p) for p in extra if isinstance(p, str) and p.strip())
        unique = []
        seen = set()
        for path in dirs:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _scan_ai_fitting_models(self) -> list[ModelInfo]:
        return discover_ai_fitting_models(self._ai_fitting_base_dirs())

    def open_ai_fitting_workspace(self) -> None:
        if getattr(self, "_ai_fitting_dialog", None) is not None:
            self._refresh_ai_fitting_models()
            self._ai_fitting_dialog.show()
            self._ai_fitting_dialog.raise_()
            self._ai_fitting_dialog.activateWindow()
            return

        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("AI Fitting Workspace")
        dialog.resize(760, 460)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("AI Model:", dialog))
        self._ai_model_combo = QComboBox(dialog)
        self._ai_model_combo.setMinimumWidth(360)
        model_row.addWidget(self._ai_model_combo, 1)
        refresh_btn = QPushButton("Refresh", dialog)
        browse_btn = QPushButton("Browse...", dialog)
        model_row.addWidget(refresh_btn)
        model_row.addWidget(browse_btn)
        layout.addLayout(model_row)

        constraint_row = QHBoxLayout()
        constraint_row.addWidget(QLabel("Constraint Mode:", dialog))
        self._ai_constraint_combo = QComboBox(dialog)
        self._ai_constraint_combo.addItems(["Free", "Fixed K", "Fixed Combination", "Current Manual Model"])
        constraint_row.addWidget(self._ai_constraint_combo)
        self._ai_constraint_k_combo = QComboBox(dialog)
        self._ai_constraint_k_combo.addItems(["1", "2", "3", "4"])
        constraint_row.addWidget(QLabel("K:", dialog))
        constraint_row.addWidget(self._ai_constraint_k_combo)
        self._ai_constraint_combination_button = QPushButton("Choose Combination...", dialog)
        self._ai_constraint_combination_button.setVisible(False)
        constraint_row.addWidget(self._ai_constraint_combination_button)
        constraint_row.addStretch(1)
        layout.addLayout(constraint_row)

        self._ai_status_label = QLabel("Status: Ready", dialog)
        self._ai_progress = QProgressBar(dialog)
        self._ai_progress.setRange(0, 100)
        self._ai_progress.setValue(0)
        layout.addWidget(self._ai_status_label)
        layout.addWidget(self._ai_progress)

        settings_grid = QGridLayout()
        settings_grid.addWidget(QLabel("Full samples:", dialog), 0, 0)
        self._ai_full_samples_spin = QSpinBox(dialog)
        self._ai_full_samples_spin.setRange(1, 1_000_000)
        settings_grid.addWidget(self._ai_full_samples_spin, 0, 1)
        settings_grid.addWidget(QLabel("Refine top:", dialog), 0, 2)
        self._ai_refine_top_n_spin = QSpinBox(dialog)
        self._ai_refine_top_n_spin.setRange(0, 100)
        settings_grid.addWidget(self._ai_refine_top_n_spin, 0, 3)
        settings_grid.addWidget(QLabel("Max eval:", dialog), 0, 4)
        self._ai_refine_max_nfev_spin = QSpinBox(dialog)
        self._ai_refine_max_nfev_spin.setRange(1, 100000)
        settings_grid.addWidget(self._ai_refine_max_nfev_spin, 0, 5)
        settings_grid.addWidget(QLabel("Progress every:", dialog), 0, 6)
        self._ai_progress_every_spin = QSpinBox(dialog)
        self._ai_progress_every_spin.setRange(0, 10000)
        settings_grid.addWidget(self._ai_progress_every_spin, 0, 7)
        settings_grid.addWidget(QLabel("Sample std:", dialog), 1, 0)
        self._ai_sampling_std_spin = QDoubleSpinBox(dialog)
        self._ai_sampling_std_spin.setDecimals(5)
        self._ai_sampling_std_spin.setRange(0.00001, 10.0)
        self._ai_sampling_std_spin.setSingleStep(0.001)
        settings_grid.addWidget(self._ai_sampling_std_spin, 1, 1)
        settings_grid.addWidget(QLabel("Target logRMSE:", dialog), 1, 2)
        self._ai_target_logrmse_spin = QDoubleSpinBox(dialog)
        self._ai_target_logrmse_spin.setDecimals(8)
        self._ai_target_logrmse_spin.setRange(0.0, 10.0)
        self._ai_target_logrmse_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_target_logrmse_spin, 1, 3)
        settings_grid.addWidget(QLabel("ftol:", dialog), 2, 0)
        self._ai_refine_ftol_spin = QDoubleSpinBox(dialog)
        self._ai_refine_ftol_spin.setDecimals(10)
        self._ai_refine_ftol_spin.setRange(0.0, 1.0)
        self._ai_refine_ftol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_refine_ftol_spin, 2, 1)
        settings_grid.addWidget(QLabel("xtol:", dialog), 2, 2)
        self._ai_refine_xtol_spin = QDoubleSpinBox(dialog)
        self._ai_refine_xtol_spin.setDecimals(10)
        self._ai_refine_xtol_spin.setRange(0.0, 1.0)
        self._ai_refine_xtol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_refine_xtol_spin, 2, 3)
        settings_grid.addWidget(QLabel("gtol:", dialog), 2, 4)
        self._ai_refine_gtol_spin = QDoubleSpinBox(dialog)
        self._ai_refine_gtol_spin.setDecimals(10)
        self._ai_refine_gtol_spin.setRange(0.0, 1.0)
        self._ai_refine_gtol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_refine_gtol_spin, 2, 5)
        settings_grid.addWidget(QLabel("Stall patience:", dialog), 3, 0)
        self._ai_stall_patience_spin = QSpinBox(dialog)
        self._ai_stall_patience_spin.setRange(0, 100000)
        self._ai_stall_patience_spin.setToolTip("0 disables stall early stop.")
        settings_grid.addWidget(self._ai_stall_patience_spin, 3, 1)
        settings_grid.addWidget(QLabel("Stall tol:", dialog), 3, 2)
        self._ai_stall_tol_spin = QDoubleSpinBox(dialog)
        self._ai_stall_tol_spin.setDecimals(10)
        self._ai_stall_tol_spin.setRange(0.0, 1.0)
        self._ai_stall_tol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_stall_tol_spin, 3, 3)
        layout.addLayout(settings_grid)

        action_row = QHBoxLayout()
        self._ai_action_buttons = []
        for text in ("Fast Predict", "Full Auto Fit", "Show Input Data", "Show Results", "Advanced Constraints"):
            btn = QPushButton(text, dialog)
            btn.setMinimumHeight(28)
            if text == "Fast Predict":
                btn.clicked.connect(lambda _checked=False: self._start_ai_prediction("fast"))
            elif text == "Full Auto Fit":
                btn.clicked.connect(lambda _checked=False: self._start_ai_prediction("full"))
            elif text == "Show Input Data":
                btn.clicked.connect(lambda _checked=False: self._show_ai_input_data_dialog())
            elif text == "Show Results":
                btn.clicked.connect(lambda _checked=False: self._show_ai_candidate_table())
            else:
                btn.clicked.connect(lambda _checked=False, label=text: self._ai_workspace_placeholder(label))
            action_row.addWidget(btn)
            self._ai_action_buttons.append(btn)
        self._ai_stop_button = QPushButton("Stop", dialog)
        self._ai_stop_button.setEnabled(False)
        self._ai_stop_button.clicked.connect(self._stop_ai_fitting_process)
        action_row.addWidget(self._ai_stop_button)
        layout.addLayout(action_row)

        self._ai_log_browser = QTextBrowser(dialog)
        self._ai_log_browser.setMinimumHeight(180)
        self._ai_log_browser.setPlaceholderText("AI fitting log")
        if getattr(self, "_ai_log_lines", None):
            self._ai_log_browser.setPlainText("\n".join(self._ai_log_lines))
        layout.addWidget(self._ai_log_browser, 1)

        close_row = QHBoxLayout()
        self._ai_open_output_button = QPushButton("Open Output Folder", dialog)
        self._ai_open_output_button.setEnabled(bool(getattr(self, "_ai_output_dir", None)))
        self._ai_open_output_button.clicked.connect(self._open_ai_output_folder)
        close_row.addWidget(self._ai_open_output_button)
        self._ai_export_output_button = QPushButton("Export Output...", dialog)
        self._ai_export_output_button.setEnabled(bool(getattr(self, "_ai_output_dir", None)))
        self._ai_export_output_button.clicked.connect(self._export_ai_prediction_output)
        close_row.addWidget(self._ai_export_output_button)
        close_row.addStretch(1)
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        refresh_btn.clicked.connect(self._refresh_ai_fitting_models)
        browse_btn.clicked.connect(self._browse_ai_fitting_model)
        self._ai_model_combo.currentIndexChanged.connect(self._on_ai_model_selected)
        self._ai_constraint_combo.currentTextChanged.connect(self._on_ai_constraint_mode_changed)
        self._ai_constraint_k_combo.currentTextChanged.connect(lambda text: self._on_ai_fixed_k_changed(text))
        self._ai_constraint_combination_button.clicked.connect(self._show_ai_fixed_combination_dialog)
        self._sync_workspace_ai_run_widgets()
        workspace_setting_map = {
            self._ai_full_samples_spin: "full_num_samples",
            self._ai_refine_top_n_spin: "full_refine_top_n",
            self._ai_refine_max_nfev_spin: "full_refine_max_nfev",
            self._ai_progress_every_spin: "full_refine_progress_interval",
            self._ai_refine_ftol_spin: "full_refine_ftol",
            self._ai_refine_xtol_spin: "full_refine_xtol",
            self._ai_refine_gtol_spin: "full_refine_gtol",
            self._ai_stall_patience_spin: "full_refine_stall_patience",
            self._ai_stall_tol_spin: "full_refine_stall_tol",
            self._ai_sampling_std_spin: "full_sampling_std",
            self._ai_target_logrmse_spin: "full_refine_target_logrmse",
        }
        for widget, key in workspace_setting_map.items():
            widget.valueChanged.connect(lambda value, setting_key=key: self._save_ai_fitting_settings(**{setting_key: value}))
        dialog.finished.connect(lambda _result: setattr(self, "_ai_fitting_dialog", None))
        self._ai_fitting_dialog = dialog
        self._refresh_ai_fitting_models()
        self._restore_ai_workspace_settings()
        dialog.show()

    def _refresh_ai_fitting_models(self) -> None:
        models = self._scan_ai_fitting_models()
        self._ai_fitting_models = models
        for combo in (getattr(self, "_ai_model_combo", None), getattr(self.ui, "aiFittingModelComboBox", None)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            for model in models:
                combo.addItem(model.display_name, str(model.artifact_path))
            combo.blockSignals(False)
        self._restore_ai_model_selection()
        if models:
            self._set_ai_workspace_status(f"Found {len(models)} AI fitting model(s).", 0)
        else:
            self._set_ai_workspace_status("No AI fitting model found in modules/Fitting_1D_Model/ or modules/Fitting_1D_model/", 0)

    def _browse_ai_fitting_model(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main_window or self.ui,
            "Select AI Fitting Model Folder",
            os.path.join(os.getcwd(), "modules", "Fitting_1D_Model"),
        )
        if not folder:
            return
        found = discover_model_in_path(Path(folder))
        if not found:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting Model",
                "Selected folder must contain a .keras artifact or a TensorFlow SavedModel root/subfolder.",
            )
            return
        settings = self._ai_fitting_settings()
        extra = settings.get("extra_model_paths")
        extra = extra if isinstance(extra, list) else []
        if folder not in extra:
            extra.append(folder)
        self._save_ai_fitting_settings(extra_model_paths=extra, last_selected_model=str(found[0].artifact_path))
        self._refresh_ai_fitting_models()
        self._set_ai_workspace_status(f"Selected model: {found[0].artifact_path}", 0)

    def _restore_ai_model_selection(self) -> None:
        selected = self._ai_fitting_settings().get("last_selected_model")
        if not selected:
            return
        for combo in (getattr(self, "_ai_model_combo", None), getattr(self.ui, "aiFittingModelComboBox", None)):
            if combo is None:
                continue
            for i in range(combo.count()):
                if combo.itemData(i) == selected:
                    combo.setCurrentIndex(i)
                    break

    def _restore_ai_workspace_settings(self) -> None:
        mode = str(self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(" Prediction", "")
        combo = getattr(self, "_ai_constraint_combo", None)
        if combo is not None:
            idx = combo.findText(str(mode))
            combo.blockSignals(True)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        k_value = str(self._ai_fitting_settings().get("fixed_k", 1))
        k_combo = getattr(self, "_ai_constraint_k_combo", None)
        if k_combo is not None:
            idx = k_combo.findText(k_value)
            k_combo.blockSignals(True)
            k_combo.setCurrentIndex(idx if idx >= 0 else 0)
            k_combo.blockSignals(False)
        self._sync_ai_constraint_controls(str(mode))

    def _restore_main_ai_settings(self) -> None:
        mode = str(self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(" Prediction", "")
        combo = getattr(self.ui, "aiFittingConstraintComboBox", None)
        if combo is not None:
            label = "Free Prediction" if mode == "Free" else str(mode)
            idx = combo.findText(label)
            combo.blockSignals(True)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        k_value = str(self._ai_fitting_settings().get("fixed_k", 1))
        k_combo = getattr(self.ui, "aiFittingFixedKComboBox", None)
        if k_combo is not None:
            idx = k_combo.findText(k_value)
            k_combo.blockSignals(True)
            k_combo.setCurrentIndex(idx if idx >= 0 else 0)
            k_combo.blockSignals(False)
        self._sync_ai_constraint_controls(str(mode))

    def _on_ai_model_selected(self, index: int) -> None:
        combo = self.sender()
        if combo is None or index < 0:
            return
        path = combo.itemData(index)
        if path:
            self._save_ai_fitting_settings(last_selected_model=str(path))
            self._sync_ai_model_combos(str(path))

    def _sync_ai_model_combos(self, selected_path: str) -> None:
        for combo in (getattr(self.ui, "aiFittingModelComboBox", None), getattr(self, "_ai_model_combo", None)):
            if combo is None:
                continue
            for i in range(combo.count()):
                if combo.itemData(i) == selected_path:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    break

    def _on_ai_constraint_mode_changed(self, mode: str) -> None:
        mode = str(mode).replace(" Prediction", "")
        self._save_ai_fitting_settings(last_constraint_mode=mode)
        self._sync_ai_constraint_combos(mode)
        self._sync_ai_constraint_controls(mode)
        if mode == "Fixed Combination" and not self._ai_fixed_combination():
            QTimer.singleShot(0, self._show_ai_fixed_combination_dialog)

    def _on_ai_fixed_k_changed(self, text: str) -> None:
        try:
            value = int(text)
        except Exception:
            value = 1
        self._save_ai_fitting_settings(fixed_k=value)
        for combo in (getattr(self.ui, "aiFittingFixedKComboBox", None), getattr(self, "_ai_constraint_k_combo", None)):
            if combo is None:
                continue
            idx = combo.findText(str(value))
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

    def _sync_ai_constraint_combos(self, mode: str) -> None:
        for combo, free_label in (
            (getattr(self.ui, "aiFittingConstraintComboBox", None), "Free Prediction"),
            (getattr(self, "_ai_constraint_combo", None), "Free"),
        ):
            if combo is None:
                continue
            label = free_label if mode == "Free" else str(mode)
            idx = combo.findText(label)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

    def _sync_ai_constraint_controls(self, mode: str | None = None) -> None:
        mode = str(mode or self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(" Prediction", "")
        show_k = mode == "Fixed K"
        show_combo = mode == "Fixed Combination"
        for widget in (getattr(self.ui, "aiFittingFixedKComboBox", None), getattr(self, "_ai_constraint_k_combo", None)):
            if widget is not None:
                widget.setVisible(show_k)
        label = self._ai_fixed_combination_label()
        for widget in (getattr(self.ui, "aiFittingCombinationButton", None), getattr(self, "_ai_constraint_combination_button", None)):
            if widget is not None:
                widget.setVisible(show_combo)
                widget.setText(label)

    def _ai_fixed_combination(self) -> list[str]:
        constraints = self._ai_run_settings().get("parameter_constraints", {})
        components = constraints.get("components") if isinstance(constraints, dict) else None
        return [str(c) for c in components] if isinstance(components, list) else []

    def _ai_fixed_combination_label(self) -> str:
        components = self._ai_fixed_combination()
        if not components:
            return "Choose Combination..."
        display = [str(c).replace("_", " ").title() for c in components]
        return " + ".join(display)

    def _save_ai_fixed_combination(self, components: list[str]) -> None:
        settings_constraints = self._ai_run_settings().get("parameter_constraints", {})
        constraints_payload = dict(settings_constraints) if isinstance(settings_constraints, dict) else {}
        if components:
            constraints_payload["components"] = components
        else:
            constraints_payload.pop("components", None)
        self._save_ai_fitting_settings(parameter_constraints=constraints_payload)
        self._sync_ai_constraint_controls("Fixed Combination")

    def _show_ai_fixed_combination_dialog(self) -> None:
        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("Fixed Combination")
        dialog.resize(420, 260)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the component sequence for Fixed Combination:", dialog))
        current = self._ai_fixed_combination()
        choices = [
            ("None", ""),
            ("Sphere", "sphere"),
            ("Cylinder", "cylinder"),
            ("Vertical Cylinder", "vertical_cylinder"),
        ]
        combos = []
        for idx in range(4):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Slot {idx + 1}:", dialog))
            combo = QComboBox(dialog)
            for label, value in choices:
                combo.addItem(label, value)
            if idx < len(current):
                found = combo.findData(current[idx])
                combo.setCurrentIndex(found if found >= 0 else 0)
            row.addWidget(combo, 1)
            layout.addLayout(row)
            combos.append(combo)

        buttons = QHBoxLayout()
        save = QPushButton("Save", dialog)
        clear = QPushButton("Clear", dialog)
        cancel = QPushButton("Cancel", dialog)
        buttons.addWidget(save)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        def selected_components() -> list[str]:
            return [str(combo.currentData()) for combo in combos if combo.currentData()]

        def save_selection() -> None:
            components = selected_components()
            if not components:
                QMessageBox.information(dialog, "Fixed Combination", "Select at least one component.")
                return
            self._save_ai_fixed_combination(components)
            self._set_ai_workspace_status(f"Fixed combination: {self._ai_fixed_combination_label()}", None)
            dialog.accept()

        def clear_selection() -> None:
            self._save_ai_fixed_combination([])
            dialog.accept()

        save.clicked.connect(save_selection)
        clear.clicked.connect(clear_selection)
        cancel.clicked.connect(dialog.reject)
        dialog.exec_()

    def _set_ai_workspace_status(self, text: str, progress: int = None) -> None:
        main_label = getattr(self.ui, "aiFittingStatusLabel", None) or getattr(self.ui, "fitMethodInfoLabel", None)
        if main_label is not None:
            main_label.setText(f"Status: {text}")
        label = getattr(self, "_ai_status_label", None)
        if label is not None:
            label.setText(f"Status: {text}")
        bar = getattr(self, "_ai_progress", None)
        if bar is not None and progress is not None:
            bar.setValue(int(progress))
        browser = getattr(self, "_ai_log_browser", None)
        if browser is not None:
            browser.append(text)

    def _ai_workspace_placeholder(self, action_name: str) -> None:
        if action_name == "Advanced Constraints":
            self._show_advanced_constraints_dialog()
            return
        if action_name == "Show Results":
            self._show_ai_candidate_table()
            return
        self._set_ai_workspace_status(
            f"{action_name} is available after a prediction run.",
            0,
        )

    def _selected_ai_model_path(self) -> Path | None:
        for combo in (getattr(self, "_ai_model_combo", None), getattr(self.ui, "aiFittingModelComboBox", None)):
            if combo is None or combo.currentIndex() < 0:
                continue
            data = combo.itemData(combo.currentIndex())
            if data:
                return Path(str(data))
        selected = self._ai_fitting_settings().get("last_selected_model")
        return Path(str(selected)) if selected else None

    def _current_ai_curve_arrays(self, apply_exclusions: bool = True):
        filter_mode = self._get_independent_axis_filter_mode()

        def apply_axis_filter(q_arr, i_arr, sigma_arr=None):
            q_arr = np.asarray(q_arr, dtype=np.float64).reshape(-1)
            i_arr = np.asarray(i_arr, dtype=np.float64).reshape(-1)
            n = min(q_arr.size, i_arr.size)
            q_arr, i_arr = q_arr[:n], i_arr[:n]
            if sigma_arr is not None:
                sigma_arr = np.asarray(sigma_arr, dtype=np.float64).reshape(-1)[:n]

            if filter_mode == "positive":
                axis_mask = q_arr > 0
                q_arr = q_arr[axis_mask]
                i_arr = i_arr[axis_mask]
                if sigma_arr is not None:
                    sigma_arr = sigma_arr[axis_mask]
            elif filter_mode == "negative":
                axis_mask = q_arr < 0
                q_arr = np.abs(q_arr[axis_mask])
                i_arr = i_arr[axis_mask]
                if sigma_arr is not None:
                    sigma_arr = sigma_arr[axis_mask]
                if q_arr.size > 0:
                    order = np.argsort(q_arr)
                    q_arr = q_arr[order]
                    i_arr = i_arr[order]
                    if sigma_arr is not None:
                        sigma_arr = sigma_arr[order]
            return q_arr, i_arr, sigma_arr

        def clean(q_arr, i_arr, sigma_arr=None):
            q_arr = np.asarray(q_arr, dtype=np.float64).reshape(-1)
            i_arr = np.asarray(i_arr, dtype=np.float64).reshape(-1)
            n = min(q_arr.size, i_arr.size)
            q_arr, i_arr = q_arr[:n], i_arr[:n]
            if sigma_arr is None:
                sigma_arr = np.maximum(0.05 * np.maximum(i_arr, 1e-30), 1e-30)
            else:
                sigma_arr = np.asarray(sigma_arr, dtype=np.float64).reshape(-1)[:n]
            mask = np.isfinite(q_arr) & np.isfinite(i_arr) & np.isfinite(sigma_arr) & (q_arr > 0) & (i_arr > 0) & (sigma_arr > 0)
            if np.sum(mask) < 16:
                return None
            return q_arr[mask], i_arr[mask], sigma_arr[mask]

        def apply_fit_region(q_arr, i_arr, sigma_arr=None):
            q_arr = np.asarray(q_arr, dtype=np.float64).reshape(-1)
            i_arr = np.asarray(i_arr, dtype=np.float64).reshape(-1)
            n = min(q_arr.size, i_arr.size)
            q_arr, i_arr = q_arr[:n], i_arr[:n]
            if sigma_arr is not None:
                sigma_arr = np.asarray(sigma_arr, dtype=np.float64).reshape(-1)[:n]
            q_arr, i_arr, sigma_arr = apply_axis_filter(q_arr, i_arr, sigma_arr)
            if getattr(self, "_roi_controls_enabled", True) and self._roi_min is not None and self._roi_max is not None:
                lo = min(float(self._roi_min), float(self._roi_max))
                hi = max(float(self._roi_min), float(self._roi_max))
                region_mask = np.isfinite(q_arr) & (q_arr >= lo) & (q_arr <= hi)
                if np.sum(region_mask) >= 16:
                    q_arr = q_arr[region_mask]
                    i_arr = i_arr[region_mask]
                    if sigma_arr is not None:
                        sigma_arr = sigma_arr[region_mask]
            return clean(q_arr, i_arr, sigma_arr)

        def apply_excluded_q(result):
            if not apply_exclusions or result is None:
                return result
            excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
            if not excluded:
                return result
            q_arr, i_arr, sigma_arr = result
            keep = np.array([self._ai_q_key(q_val) not in excluded for q_val in q_arr], dtype=bool)
            if int(np.sum(keep)) < 16:
                self._add_fitting_error("AI input outlier filter would leave fewer than 16 points; using unfiltered data.")
                return result
            return q_arr[keep], i_arr[keep], sigma_arr[keep]

        try:
            if self.q_ROI is not None and self.I_ROI is not None:
                q_arr, i_arr, sigma_arr = apply_axis_filter(self.q_ROI, self.I_ROI)
                result = clean(q_arr, i_arr, sigma_arr)
                if result is not None:
                    return apply_excluded_q(result)
        except Exception:
            pass
        if self.q is not None and self.I is not None:
            result = apply_fit_region(self.q, self.I)
            if result is not None:
                return apply_excluded_q(result)
        if isinstance(getattr(self, "current_1d_data", None), dict):
            data = self.current_1d_data
            result = apply_fit_region(data.get("q", []), data.get("I", []), data.get("err"))
            if result is not None:
                return apply_excluded_q(result)
        if isinstance(getattr(self, "cut", None), dict):
            result = apply_fit_region(self.cut.get("q", []), self.cut.get("I", []))
            if result is not None:
                return apply_excluded_q(result)
        return None

    @staticmethod
    def _ai_q_key(q_value) -> str:
        try:
            return f"{float(q_value):.12g}"
        except Exception:
            return str(q_value)

    def _filter_ai_excluded_points_for_display(self, q_arr, *value_arrays):
        excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
        if not excluded:
            return (q_arr, *value_arrays)
        try:
            q_np = np.asarray(q_arr)
            keep = np.array([
                self._ai_q_key(q_val) not in excluded and self._ai_q_key(abs(float(q_val))) not in excluded
                for q_val in q_np
            ], dtype=bool)
            if int(np.sum(keep)) == 0:
                return (q_arr, *value_arrays)
            filtered = [q_np[keep]]
            for arr in value_arrays:
                if arr is None:
                    filtered.append(None)
                    continue
                arr_np = np.asarray(arr)
                if arr_np.shape[0] == q_np.shape[0]:
                    filtered.append(arr_np[keep])
                else:
                    filtered.append(arr)
            return tuple(filtered)
        except Exception:
            return (q_arr, *value_arrays)

    def _show_ai_input_data_dialog(self) -> None:
        arrays = self._current_ai_curve_arrays(apply_exclusions=False)
        if arrays is None:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Input Data",
                "No valid AI input curve is loaded. Load or cut a 1D curve first.",
            )
            return

        existing = getattr(self, "_ai_input_data_dialog", None)
        if existing is not None and existing.isVisible():
            self._ai_input_dialog_arrays = arrays
            self._refresh_ai_input_data_dialog()
            existing.raise_()
            existing.activateWindow()
            return

        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("AI Input Data")
        dialog.resize(820, 560)
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)

        summary = QLabel(dialog)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableWidget(0, 4, dialog)
        table.setHorizontalHeaderLabels(["Use", "q", "I", "sigma"])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, 4):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
        layout.addWidget(table, 1)

        def selected_rows() -> list[int]:
            selection = table.selectionModel()
            if selection is None:
                return []
            return sorted({index.row() for index in selection.selectedRows()})

        def selected_q_values() -> list[float]:
            dialog_arrays = getattr(self, "_ai_input_dialog_arrays", None)
            if dialog_arrays is None:
                return []
            q_arr, _, _ = dialog_arrays
            values = []
            for row in selected_rows():
                if 0 <= row < len(q_arr):
                    values.append(float(q_arr[row]))
            return values

        def delete_selected() -> None:
            values = selected_q_values()
            if not values:
                return
            self._exclude_ai_input_points(values, source="table")

        def restore_selected() -> None:
            values = selected_q_values()
            if not values:
                return
            self._restore_ai_input_points(values)

        def restore_all() -> None:
            self._restore_all_ai_input_points()

        button_row = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected", dialog)
        restore_btn = QPushButton("Restore Selected", dialog)
        restore_all_btn = QPushButton("Restore All", dialog)
        close_btn = QPushButton("Close", dialog)
        delete_btn.clicked.connect(delete_selected)
        restore_btn.clicked.connect(restore_selected)
        restore_all_btn.clicked.connect(restore_all)
        close_btn.clicked.connect(dialog.close)
        button_row.addWidget(delete_btn)
        button_row.addWidget(restore_btn)
        button_row.addWidget(restore_all_btn)
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        def on_finished(_result):
            self._ai_input_data_dialog = None
            self._ai_input_data_table = None
            self._ai_input_data_summary = None
            self._ai_input_dialog_arrays = None

        dialog.finished.connect(on_finished)
        self._ai_input_data_dialog = dialog
        self._ai_input_data_table = table
        self._ai_input_data_summary = summary
        self._ai_input_dialog_arrays = arrays
        self._refresh_ai_input_data_dialog()
        dialog.show()

    def _refresh_ai_input_data_dialog(self) -> None:
        dialog = getattr(self, "_ai_input_data_dialog", None)
        table = getattr(self, "_ai_input_data_table", None)
        summary = getattr(self, "_ai_input_data_summary", None)
        arrays = getattr(self, "_ai_input_dialog_arrays", None)
        if dialog is None or table is None or summary is None or arrays is None:
            return
        try:
            q_arr, i_arr, sigma_arr = arrays
            excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
            table.setRowCount(len(q_arr))
            for row, (q_val, i_val, sigma_val) in enumerate(zip(q_arr, i_arr, sigma_arr)):
                enabled = self._ai_q_key(q_val) not in excluded
                table.setItem(row, 0, QTableWidgetItem("Yes" if enabled else "No"))
                table.setItem(row, 1, QTableWidgetItem(f"{float(q_val):.8g}"))
                table.setItem(row, 2, QTableWidgetItem(f"{float(i_val):.8g}"))
                table.setItem(row, 3, QTableWidgetItem(f"{float(sigma_val):.8g}"))
            kept = sum(1 for q_val in q_arr if self._ai_q_key(q_val) not in excluded)
            removed = len(q_arr) - kept
            summary.setText(
                "Input points: "
                f"{len(q_arr)} | used: {kept} | excluded: {removed}. "
                "In Independent Fit Window, enable Delete Points and click a curve point to exclude it."
            )
        except Exception:
            pass

    def _exclude_ai_input_point_from_plot(self, q_value: float) -> None:
        self._exclude_ai_input_points([q_value], source="plot")

    def _exclude_ai_input_points(self, q_values, source: str = "table") -> None:
        excluded = set(getattr(self, "_ai_excluded_input_q", set()) or set())
        before = len(excluded)
        for q_value in q_values:
            excluded.add(self._ai_q_key(q_value))
            if source == "plot":
                try:
                    excluded.add(self._ai_q_key(abs(float(q_value))))
                except Exception:
                    pass
        self._ai_excluded_input_q = excluded
        added = max(0, len(excluded) - before)
        self._refresh_ai_input_data_dialog()
        self._refresh_ai_input_outlier_views()
        if added:
            label = "from Independent Fit Window" if source == "plot" else "from table"
            self._set_ai_workspace_status(f"Excluded {added} input point(s) {label}.", None)

    def _restore_ai_input_points(self, q_values) -> None:
        excluded = set(getattr(self, "_ai_excluded_input_q", set()) or set())
        before = len(excluded)
        for q_value in q_values:
            excluded.discard(self._ai_q_key(q_value))
            try:
                abs_q = abs(float(q_value))
                excluded.discard(self._ai_q_key(abs_q))
                excluded.discard(self._ai_q_key(-abs_q))
            except Exception:
                pass
        self._ai_excluded_input_q = excluded
        restored = max(0, before - len(excluded))
        self._refresh_ai_input_data_dialog()
        self._refresh_ai_input_outlier_views()
        if restored:
            self._set_ai_workspace_status(f"Restored {restored} input point(s).", None)

    def _restore_all_ai_input_points(self) -> None:
        self._ai_excluded_input_q = set()
        self._refresh_ai_input_data_dialog()
        self._refresh_ai_input_outlier_views()
        self._set_ai_workspace_status("All AI input points restored.", None)

    def _refresh_ai_input_outlier_views(self) -> None:
        try:
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
        except Exception:
            pass

    def _ai_prediction_output_root(self) -> Path:
        return Path.cwd() / "AI_Fitting_Output"

    def _ai_current_prediction_dir(self) -> Path:
        return self._ai_prediction_output_root() / "current_prediction"

    def _clear_ai_current_prediction_dir(self, out_dir: Path) -> None:
        root = self._ai_prediction_output_root().resolve()
        target = Path(out_dir).resolve()
        expected = (root / "current_prediction").resolve()
        if target != expected:
            raise RuntimeError(f"Refusing to clear unexpected AI output directory: {target}")
        target.mkdir(parents=True, exist_ok=True)
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _prepare_ai_prediction_io(self) -> tuple[Path, Path] | None:
        arrays = self._current_ai_curve_arrays()
        if arrays is None:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting",
                "No valid AI input curve is loaded. Load or cut a 1D curve before prediction.",
            )
            return None
        q_arr, i_arr, sigma_arr = arrays
        out_dir = self._ai_current_prediction_dir()
        try:
            self._clear_ai_current_prediction_dir(out_dir)
        except Exception as exc:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting",
                f"Failed to prepare the reusable AI output folder:\n{exc}",
            )
            return None
        input_csv = out_dir / "input_curve.csv"
        table = np.column_stack([q_arr, i_arr, sigma_arr])
        np.savetxt(input_csv, table, delimiter=",", header="q,I,sigma", comments="")
        self._ai_output_dir = out_dir
        self._ai_input_csv = input_csv
        return input_csv, out_dir

    def _ai_exact_nonempty_arg(self):
        constraints = self.build_ai_constraints_json_from_ui()
        exact = constraints.get("exact_nonempty")
        try:
            exact = int(exact)
        except Exception:
            exact = None
        return exact if exact and exact > 0 else None

    def _start_ai_prediction(self, run_mode: str = "fast") -> None:
        process = getattr(self, "_ai_process", None)
        if process is not None and process.state() != QProcess.NotRunning:
            self._set_ai_workspace_status("AI prediction is already running.", None)
            return

        model_path = self._selected_ai_model_path()
        if model_path is None or not model_path.exists():
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting",
                "Please import or select a valid AI fitting model first.",
            )
            return
        io_paths = self._prepare_ai_prediction_io()
        if io_paths is None:
            return
        input_csv, out_dir = io_paths

        script = Path.cwd() / "utils" / "predict_topK.py"
        if not script.is_file():
            QMessageBox.warning(self.main_window or self.ui, "AI Fitting", f"Prediction script not found:\n{script}")
            return

        exact = self._ai_exact_nonempty_arg()
        run_settings = self._ai_run_settings()
        constraints_path = self._write_ai_constraints_json(out_dir)
        args = [
            str(script),
            "--model_dir", str(model_path),
            "--input_csv", str(input_csv),
            "--output_dir", str(out_dir),
            "--score_mode", "unweighted_log",
            "--sampling_std", str(run_settings["full_sampling_std"] if run_mode == "full" else 0.005),
            "--include_mean_candidate",
            "--allow_unsafe_lambda",
        ]
        if run_mode == "full":
            args.extend([
                "--num_samples", str(int(run_settings["full_num_samples"])),
                "--top_k", str(int(run_settings["full_top_k"])),
                "--refine_top_n", str(int(run_settings["full_refine_top_n"])),
                "--refine_max_nfev", str(int(run_settings["full_refine_max_nfev"])),
                "--refine_progress_interval", str(int(run_settings["full_refine_progress_interval"])),
                "--refine_ftol", str(float(run_settings["full_refine_ftol"])),
                "--refine_xtol", str(float(run_settings["full_refine_xtol"])),
                "--refine_gtol", str(float(run_settings["full_refine_gtol"])),
                "--refine_stall_patience", str(int(run_settings["full_refine_stall_patience"])),
                "--refine_stall_tol", str(float(run_settings["full_refine_stall_tol"])),
                "--refine_target_logrmse", str(float(run_settings["full_refine_target_logrmse"])),
                "--progress_interval", "100",
            ])
        else:
            args.extend([
                "--num_samples", str(int(run_settings["fast_num_samples"])),
                "--top_k", str(int(run_settings["fast_top_k"])),
                "--refine_top_n", "0",
                "--progress_interval", str(int(run_settings["fast_progress_interval"])),
            ])
        if exact is not None:
            args.extend(["--exact_nonempty", str(exact)])
        if constraints_path is not None:
            args.extend(["--constraints_json", str(constraints_path)])

        process = QProcess(self.main_window or self.ui)
        process.setWorkingDirectory(str(Path.cwd()))
        process.readyReadStandardOutput.connect(self._on_ai_process_stdout)
        process.readyReadStandardError.connect(self._on_ai_process_stderr)
        process.finished.connect(self._on_ai_process_finished)
        process.errorOccurred.connect(self._on_ai_process_error)
        self._ai_process = process
        self._ai_candidate_rows = []
        self._set_ai_running_state(True)
        self._set_ai_workspace_status(f"Starting {run_mode} AI fitting run...", 0)
        self._append_ai_log(f"Command: {sys.executable} {' '.join(args)}")
        process.start(sys.executable, args)

    def _set_ai_running_state(self, running: bool) -> None:
        for button in getattr(self, "_ai_action_buttons", []) or []:
            text = button.text()
            if text in ("Fast Predict", "Full Auto Fit"):
                button.setEnabled(not running)
        for name in ("aiFittingFastPredictButton", "aiFittingFullAutoFitButton"):
            button = getattr(self.ui, name, None)
            if button is not None:
                button.setEnabled(not running)
        main_stop = getattr(self.ui, "aiFittingStopButton", None)
        if main_stop is not None:
            main_stop.setEnabled(running)
        if self._ai_stop_button is not None:
            self._ai_stop_button.setEnabled(running)
        can_export = bool(getattr(self, "_ai_output_dir", None)) and not running
        if self._ai_open_output_button is not None:
            self._ai_open_output_button.setEnabled(bool(getattr(self, "_ai_output_dir", None)))
        if self._ai_export_output_button is not None:
            self._ai_export_output_button.setEnabled(can_export)
        main_export = getattr(self.ui, "aiFittingExportOutputButton", None)
        if main_export is not None:
            main_export.setEnabled(can_export)

    def _append_ai_log(self, text: str) -> None:
        text = str(text).rstrip()
        if not text:
            return
        if not isinstance(getattr(self, "_ai_log_lines", None), list):
            self._ai_log_lines = []
        self._ai_log_lines.append(text)
        if len(self._ai_log_lines) > 2000:
            self._ai_log_lines = self._ai_log_lines[-2000:]
        browser = getattr(self, "_ai_log_browser", None)
        if browser is not None:
            browser.append(text)
        out_dir = getattr(self, "_ai_output_dir", None)
        if out_dir:
            try:
                with (Path(out_dir) / "gui_run.log").open("a", encoding="utf-8") as fh:
                    fh.write(text + "\n")
            except Exception:
                pass

    def _on_ai_process_stdout(self) -> None:
        process = getattr(self, "_ai_process", None)
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._handle_ai_process_text(text)

    def _on_ai_process_stderr(self) -> None:
        process = getattr(self, "_ai_process", None)
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        self._handle_ai_process_text(text)

    def _handle_ai_process_text(self, text: str) -> None:
        for line in str(text).splitlines():
            self._append_ai_log(line)
            match = re.search(r"Progress\s+(\d+)/(\d+)", line)
            if match:
                current = int(match.group(1))
                total = max(1, int(match.group(2)))
                self._set_ai_workspace_status(f"Sampling progress {current}/{total}", int(current * 100 / total))
            elif re.search(r"refine\s+#(\d+)/(\d+)\s+nfev~(\d+)/(\d+)", line):
                refine_match = re.search(r"refine\s+#(\d+)/(\d+)\s+nfev~(\d+)/(\d+)", line)
                idx = int(refine_match.group(1))
                total = max(1, int(refine_match.group(2)))
                nfev = int(refine_match.group(3))
                max_nfev = max(1, int(refine_match.group(4)))
                refine_fraction = ((idx - 1) + min(1.0, nfev / max_nfev)) / total
                self._set_ai_workspace_status(line[:180], int(100 * refine_fraction))
            elif "Refine #" in line:
                self._set_ai_workspace_status(line[:180], None)
            elif line.startswith("Wrote "):
                self._set_ai_workspace_status(line, 100)

    def _on_ai_process_finished(self, exit_code: int, exit_status) -> None:
        self._set_ai_running_state(False)
        if exit_code == 0:
            self._set_ai_workspace_status(f"AI fitting finished. Output: {self._ai_output_dir}", 100)
            self._show_ai_candidate_table(self._ai_output_dir)
        else:
            self._set_ai_workspace_status(f"AI fitting failed with exit code {exit_code}. See log for details.", 0)

    def _on_ai_process_error(self, error) -> None:
        self._set_ai_running_state(False)
        self._set_ai_workspace_status(f"AI process error: {error}", 0)

    def _stop_ai_fitting_process(self) -> None:
        process = getattr(self, "_ai_process", None)
        if process is None or process.state() == QProcess.NotRunning:
            return
        self._append_ai_log("Stopping AI fitting process...")
        process.terminate()
        QTimer.singleShot(2500, lambda: process.kill() if process.state() != QProcess.NotRunning else None)

    def _open_ai_output_folder(self) -> None:
        out_dir = getattr(self, "_ai_output_dir", None)
        if out_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(out_dir))))

    def _export_ai_prediction_output(self) -> None:
        out_dir = Path(getattr(self, "_ai_output_dir", "") or self._ai_current_prediction_dir())
        if not out_dir.is_dir() or not any(out_dir.iterdir()):
            QMessageBox.information(
                self.main_window or self.ui,
                "Export AI Output",
                "No AI prediction output is available yet. Run a prediction first.",
            )
            return

        settings = self._ai_fitting_settings()
        start_dir = str(settings.get("last_export_parent") or Path.cwd())
        parent = QFileDialog.getExistingDirectory(
            self.main_window or self.ui,
            "Choose Folder for Exported AI Output",
            start_dir,
        )
        if not parent:
            return

        parent_path = Path(parent)
        try:
            source_dir = out_dir.resolve()
            parent_resolved = parent_path.resolve()
            if parent_resolved == source_dir or source_dir in parent_resolved.parents:
                QMessageBox.warning(
                    self.main_window or self.ui,
                    "Export AI Output",
                    "Choose a folder outside the reusable AI output directory.",
                )
                return
        except Exception:
            pass
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = parent_path / f"ai_prediction_{timestamp}"
        suffix = 1
        while dest.exists():
            dest = parent_path / f"ai_prediction_{timestamp}_{suffix}"
            suffix += 1
        try:
            shutil.copytree(out_dir, dest)
        except Exception as exc:
            QMessageBox.warning(
                self.main_window or self.ui,
                "Export AI Output",
                f"Failed to export AI prediction output:\n{exc}",
            )
            return

        self._save_ai_fitting_settings(last_export_parent=str(parent_path))
        self._set_ai_workspace_status(f"Exported AI output to: {dest}", None)
        self._append_ai_log(f"Exported AI output to: {dest}")
        QMessageBox.information(
            self.main_window or self.ui,
            "Export AI Output",
            f"AI prediction output exported to:\n{dest}",
        )

    def _show_ai_candidate_table(self, output_dir: Path | None = None) -> None:
        output_dir = Path(output_dir or getattr(self, "_ai_output_dir", "") or "")
        results_path = output_dir / "top20_candidates.json"
        if not results_path.is_file():
            self._set_ai_workspace_status("No AI candidate results found yet.", None)
            return
        try:
            with results_path.open("r", encoding="utf-8") as fh:
                rows = json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self.main_window or self.ui, "AI Fitting Results", f"Failed to read results:\n{exc}")
            return
        if not isinstance(rows, list) or not rows:
            self._set_ai_workspace_status("AI fitting produced no candidates.", None)
            return
        self._ai_candidate_rows = rows

        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("AI Fitting Candidates")
        dialog.resize(900, 520)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(rows), 7, dialog)
        table.setHorizontalHeaderLabels(["Rank", "Combination", "Score Prob.", "Posterior", "logRMSE", "Chi2", "Source"])
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row_idx, row in enumerate(rows):
            values = [
                row.get("rank", row_idx + 1),
                row.get("combination", ""),
                f"{float(row.get('score_weighted_probability', 0.0)) * 100:.2f}%",
                f"{float(row.get('posterior_frequency', 0.0)) * 100:.2f}%",
                f"{float(row.get('best_log_rmse', np.nan)):.5g}",
                f"{float(row.get('best_chi2_weighted', np.nan)):.5g}",
                row.get("best_source", ""),
            ]
            for col, value in enumerate(values):
                table.setItem(row_idx, col, QTableWidgetItem(str(value)))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, 7):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.selectRow(0)
        layout.addWidget(table, 1)

        button_row = QHBoxLayout()
        load_btn = QPushButton("Load Selected Params", dialog)
        open_btn = QPushButton("Open Output Folder", dialog)
        close_btn = QPushButton("Close", dialog)
        load_btn.clicked.connect(lambda: self._load_selected_ai_candidate_from_table(table, rows, dialog))
        table.doubleClicked.connect(lambda _index: self._load_selected_ai_candidate_from_table(table, rows, dialog))
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir))))
        close_btn.clicked.connect(dialog.close)
        button_row.addWidget(load_btn)
        button_row.addWidget(open_btn)
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        self._ai_results_dialog = dialog
        dialog.show()

    def _load_selected_ai_candidate_from_table(self, table: QTableWidget, rows: list, dialog: QDialog | None = None) -> None:
        selected = table.currentRow()
        if selected < 0 or selected >= len(rows):
            return
        if self._load_ai_candidate_params(rows[selected]):
            if dialog is not None:
                dialog.accept()

    def _load_ai_candidate_params(self, row: dict) -> bool:
        components = row.get("components") or []
        if not isinstance(components, list) or not components:
            QMessageBox.warning(self.main_window or self.ui, "AI Fitting", "Selected candidate has no component parameters.")
            return False
        try:
            while len(self._iter_particle_widget_ids()) < len(components):
                self._on_add_particle_clicked()
            for widget_id in self._iter_particle_widget_ids():
                self.set_particle_shape(widget_id, "None")

            shape_map = {
                "sphere": "Sphere",
                "cylinder": "Cylinder",
                "vertical_cylinder": "Vertical Cylinder",
                "vertical cylinder": "Vertical Cylinder",
                "verticalcylinder": "Vertical Cylinder",
            }
            param_map = {
                "R": "radius",
                "sigma_R": "sigma_radius",
                "h": "height",
                "sigma_h": "sigma_height",
                "D": "diameter",
                "sigma_D": "sigma_diameter",
            }
            widget_ids = self._iter_particle_widget_ids()
            for idx, component in enumerate(components):
                widget_id = widget_ids[idx]
                raw_type = str(component.get("type", "")).strip()
                shape = shape_map.get(raw_type.lower().replace("-", "_"), raw_type)
                if shape not in COMPONENT_PARAMETER_SCHEMAS:
                    shape = "Sphere"
                particle_id = f"particle_{widget_id}"
                self.model_params_manager.set_particle_shape("fitting", particle_id, shape)
                self.model_params_manager.set_particle_enabled("fitting", particle_id, True)
                self.model_params_manager.set_particle_parameter("fitting", particle_id, shape, "intensity", float(component.get("weight", 1.0)))
                params = component.get("params") or {}
                for source_key, target_key in param_map.items():
                    if source_key in params:
                        self.model_params_manager.set_particle_parameter(
                            "fitting", particle_id, shape, target_key, float(params[source_key])
                        )

            global_map = {
                "background": "background",
                "BG": "background",
                "sigma_res": "sigma_res",
                "sigma_Res": "sigma_res",
                "nu_res": "nu_res",
                "nu_Res": "nu_res",
                "int_res": "int_res",
                "int_Res": "int_res",
                "k": "k_value",
                "k_value": "k_value",
            }
            for key, value in (row.get("global_params") or {}).items():
                target = global_map.get(str(key), global_map.get(str(key).lower()))
                if target is not None:
                    self.model_params_manager.set_global_parameter("fitting", target, float(value))
            self.model_params_manager.save_parameters()
            self.reload_particle_parameters()
            self._set_ai_workspace_status(f"Loaded AI candidate #{row.get('rank', '')}: {row.get('combination', '')}", None)
            return True
        except Exception as exc:
            QMessageBox.warning(self.main_window or self.ui, "AI Fitting", f"Failed to load candidate parameters:\n{exc}")
            return False

    def build_ai_constraints_json_from_ui(self) -> dict:
        mode = self._ai_fitting_settings().get("last_constraint_mode", "Free")
        workspace_combo = getattr(self, "_ai_constraint_combo", None)
        if workspace_combo is not None:
            mode = workspace_combo.currentText()
        main_combo = getattr(self.ui, "aiFittingConstraintComboBox", None)
        if main_combo is not None and workspace_combo is None:
            mode = main_combo.currentText().replace(" Prediction", "")
        settings_constraints = self._ai_run_settings().get("parameter_constraints", {})
        payload = {"mode": mode, "constraints": {}}
        if isinstance(settings_constraints, dict):
            for key in ("type_parameter_ranges", "global_ranges", "parameter_ranges"):
                value = settings_constraints.get(key)
                if value:
                    payload[key] = value
        if mode == "Fixed K":
            try:
                payload["exact_nonempty"] = int(self._ai_fitting_settings().get("fixed_k", 1))
            except Exception:
                payload["exact_nonempty"] = 1
        elif mode == "Current Manual Model":
            shapes = []
            try:
                for widget_id in self._iter_particle_widget_ids():
                    shape = self.get_particle_shape(widget_id)
                    if shape and shape != "None":
                        shapes.append(shape.lower().replace(" ", "_"))
            except Exception:
                shapes = []
            payload["components"] = shapes
            payload["exact_nonempty"] = len(shapes) if shapes else None
        elif mode == "Fixed Combination":
            components = settings_constraints.get("components") if isinstance(settings_constraints, dict) else None
            payload["components"] = components if isinstance(components, list) else []
        return payload

    def _write_ai_constraints_json(self, out_dir: Path) -> Path | None:
        payload = self.build_ai_constraints_json_from_ui()
        has_constraints = any(payload.get(key) for key in ("components", "type_parameter_ranges", "global_ranges", "parameter_ranges"))
        if not has_constraints and payload.get("mode") in ("Free", "Free Prediction"):
            return None
        path = Path(out_dir) / "constraints.json"
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            self._append_ai_log(f"Constraints: {path}")
            return path
        except Exception as exc:
            self._append_ai_log(f"Failed to write constraints JSON: {exc}")
            return None

    def _show_advanced_constraints_dialog(self) -> None:
        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("Advanced Constraints")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Enable ranges to constrain posterior sampling and final least-squares refinement.",
            dialog,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        rows = [
            ("type", "R", 1.0, 100.0),
            ("type", "sigma_R", 0.02, 90.0),
            ("type", "h", 2.0, 500.0),
            ("type", "sigma_h", 0.04, 400.0),
            ("type", "D", 3.0, 500.0),
            ("type", "sigma_D", 0.06, 400.0),
            ("global", "BG", 1e-18, 1e8),
            ("global", "sigma_Res", 0.002, 0.3),
            ("global", "nu_Res", 1.0, 10.0),
            ("global", "int_Res", 1e-18, 1e8),
            ("global", "k", 1e-2, 1e6),
        ]
        stored = self._ai_run_settings().get("parameter_constraints", {})
        type_ranges = stored.get("type_parameter_ranges", {}) if isinstance(stored, dict) else {}
        global_ranges = stored.get("global_ranges", {}) if isinstance(stored, dict) else {}

        table = QTableWidget(len(rows), 5, dialog)
        table.setHorizontalHeaderLabels(["Apply", "Scope", "Parameter", "Min", "Max"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

        row_widgets = []
        for row_idx, (scope, name, default_lo, default_hi) in enumerate(rows):
            if scope == "type":
                existing = None
                for type_name in ("sphere", "cylinder", "vertical_cylinder"):
                    ranges = type_ranges.get(type_name, {}) if isinstance(type_ranges, dict) else {}
                    if name in ranges:
                        existing = ranges[name]
                        break
            else:
                existing = global_ranges.get(name) if isinstance(global_ranges, dict) else None
            enabled = isinstance(existing, (list, tuple)) and len(existing) == 2
            lo = float(existing[0]) if enabled else float(default_lo)
            hi = float(existing[1]) if enabled else float(default_hi)

            check = QCheckBox(table)
            check.setChecked(enabled)
            min_box = QDoubleSpinBox(table)
            max_box = QDoubleSpinBox(table)
            for spin in (min_box, max_box):
                spin.setDecimals(8)
                spin.setRange(0.0, 1e12)
                spin.setSingleStep(max(abs(default_hi - default_lo) / 100.0, 1e-6))
                spin.setMinimumWidth(120)
            min_box.setValue(lo)
            max_box.setValue(hi)
            table.setCellWidget(row_idx, 0, check)
            table.setItem(row_idx, 1, QTableWidgetItem("Component" if scope == "type" else "Global"))
            table.setItem(row_idx, 2, QTableWidgetItem(name))
            table.setCellWidget(row_idx, 3, min_box)
            table.setCellWidget(row_idx, 4, max_box)
            row_widgets.append((scope, name, check, min_box, max_box))

        layout.addWidget(table, 1)

        preview = QTextBrowser(dialog)
        preview.setMaximumHeight(120)
        preview.setPlainText(json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False))
        layout.addWidget(preview)

        def collect_constraints() -> dict:
            type_constraints = {}
            global_constraints = {}
            for scope, name, check, min_box, max_box in row_widgets:
                if not check.isChecked():
                    continue
                lo = float(min_box.value())
                hi = float(max_box.value())
                if hi < lo:
                    lo, hi = hi, lo
                if scope == "type":
                    for type_name in ("sphere", "cylinder", "vertical_cylinder"):
                        type_constraints.setdefault(type_name, {})[name] = [lo, hi]
                else:
                    global_constraints[name] = [lo, hi]
            payload = {}
            if type_constraints:
                payload["type_parameter_ranges"] = type_constraints
            if global_constraints:
                payload["global_ranges"] = global_constraints
            return payload

        def refresh_preview() -> None:
            settings = self._ai_fitting_settings()
            old_constraints = settings.get("parameter_constraints")
            self._save_ai_fitting_settings(parameter_constraints=collect_constraints())
            preview.setPlainText(json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False))
            self._save_ai_fitting_settings(parameter_constraints=old_constraints if isinstance(old_constraints, dict) else {})

        for _scope, _name, check, min_box, max_box in row_widgets:
            check.toggled.connect(lambda _=False: refresh_preview())
            min_box.valueChanged.connect(lambda _=0.0: refresh_preview())
            max_box.valueChanged.connect(lambda _=0.0: refresh_preview())

        save = QPushButton("Save Constraints", dialog)
        clear = QPushButton("Clear All", dialog)
        close = QPushButton("Close", dialog)

        def save_constraints() -> None:
            constraints_payload = collect_constraints()
            self._save_ai_fitting_settings(parameter_constraints=constraints_payload)
            preview.setPlainText(json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False))
            self._set_ai_workspace_status("Advanced parameter constraints saved.", None)
            dialog.accept()

        def clear_constraints() -> None:
            for _scope, _name, check, _min_box, _max_box in row_widgets:
                check.setChecked(False)
            self._save_ai_fitting_settings(parameter_constraints={})
            preview.setPlainText(json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False))

        save.clicked.connect(save_constraints)
        clear.clicked.connect(clear_constraints)
        close.clicked.connect(dialog.reject)
        row = QHBoxLayout()
        row.addWidget(save)
        row.addWidget(clear)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        dialog.exec_()

    def get_global_parameter(self, param: str) -> float:
        """No description."""
        return self.model_params_manager.get_global_parameter('fitting', param, 0.0)

    def set_global_parameter(self, param: str, value: float) -> bool:
        """No description."""
        success = self.model_params_manager.set_global_parameter('fitting', param, value)
        if success:
            # ???UI??????????
            if param == 'background' and hasattr(self.ui, 'fitBGValue'):
                self.ui.fitBGValue.blockSignals(True)
                self.ui.fitBGValue.setValue(value)
                self.ui.fitBGValue.blockSignals(False)
            elif param == 'sigma_res' and hasattr(self.ui, 'fitSigmaResValue'):
                self.ui.fitSigmaResValue.blockSignals(True)
                self.ui.fitSigmaResValue.setValue(value)
                self.ui.fitSigmaResValue.blockSignals(False)
            elif param == 'k_value' and hasattr(self.ui, 'fitKValue'):
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(value)
                self.ui.fitKValue.blockSignals(False)

            # ??????
            self.model_params_manager.save_parameters()
        return success

    def get_all_global_parameters(self) -> dict:
        """No description."""
        return self.model_params_manager.get_all_global_parameters('fitting')

    def reset_global_parameters(self):
        """No description."""
        self.set_global_parameter('background', 0.0)
        self.set_global_parameter('sigma_res', 0.1)
        self.set_global_parameter('nu_res', 5.0)
        self.set_global_parameter('int_res', 0.0)
        self.set_global_parameter('k_value', 1.0)
        self._add_fitting_success("Global parameters reset to default values")

    # ================================
    # ?????????
    # ================================

    def _save_auto_k_enabled(self):
        """auto-K"""
        try:
            from core.user_settings import user_settings
            user_settings.set('_auto_k_enabled', self._auto_k_enabled)
            user_settings.save_settings()
        except Exception as e:
            print(f"Failed to save auto-K setting: {e}")

    def _load_auto_k_enabled(self):
        """No description."""
        try:
            from core.user_settings import user_settings
            self._auto_k_enabled = user_settings.get('_auto_k_enabled', False)
            self._update_auto_k_button_style()
        except Exception as e:
            print(f"Failed to load auto-K setting: {e}")
            self._auto_k_enabled = False

    def _update_auto_k_button_style(self):
        """No description."""
        if hasattr(self.ui, 'FittingAutoKButton'):
            if self._auto_k_enabled:
                self.ui.FittingAutoKButton.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
                )
                self.ui.FittingAutoKButton.setText("Auto-K: ON")
            else:
                self.ui.FittingAutoKButton.setStyleSheet("")
                self.ui.FittingAutoKButton.setText("Auto-K: OFF")
            self._sync_global_secondary_button_widths()

    def _sync_global_secondary_button_widths(self):
        """Keep Auto-K and step-reset buttons visually aligned after runtime style changes."""
        auto_k_button = getattr(self.ui, 'FittingAutoKButton', None)
        if auto_k_button is None:
            return

        parent = auto_k_button.parentWidget()
        if parent is None:
            return

        buttons = [auto_k_button]
        for button in parent.findChildren(QPushButton):
            name = button.objectName() or ''
            if name.endswith('ResetButton') and button not in buttons:
                buttons.append(button)

        if len(buttons) <= 1:
            return

        for button in buttons:
            button.ensurePolished()

        target_width = max(button.sizeHint().width() for button in buttons)
        for button in buttons:
            button.setMinimumWidth(target_width)
            button.setMaximumWidth(target_width)
            button.updateGeometry()

    def _on_auto_k_button_clicked(self):
        """auto-K"""
        # ???????
        self._auto_k_enabled = not self._auto_k_enabled

        # ???????
        self._save_auto_k_enabled()

        # ?????????
        self._update_auto_k_button_style()

        # ??????????
        status = "enabled" if self._auto_k_enabled else "disabled"
        self._add_fitting_message(f"Auto K-value optimization {status}")

        # ???????uto-K??????????????????????????
        if self._auto_k_enabled and hasattr(self, 'I') and hasattr(self, 'I_fitting'):
            if self.I is not None and self.I_fitting is not None:
                self._optimize_k_value()

    def _optimize_k_value(self):
        """K"""
        try:
            if not hasattr(self, 'I') or not hasattr(self, 'I_fitting') or \
               self.I is None or self.I_fitting is None:
                self._add_fitting_error("No fitting data available for K-value optimization")
                return

            # ????????????????
            if self.I.size == 0 or self.I_fitting.size == 0:
                self._add_fitting_error("Empty data arrays for K-value optimization")
                return

            if self.I.shape != self.I_fitting.shape:
                self._add_fitting_error(f"Data shape mismatch: I{self.I.shape} vs I_fitting{self.I_fitting.shape}")
                return

            # ???????????aN???????
            if np.any(~np.isfinite(self.I)) or np.any(~np.isfinite(self.I_fitting)):
                self._add_fitting_error("Data contains NaN or infinite values")
                return

            # ??????K??
            current_k = self.get_global_parameter('k_value')
            self._add_fitting_message(f"Starting K-value optimization from {current_k:.6f}...")

            # ??????ROI?????OI???????????
            try:
                q_arr = np.asarray(self.q) if hasattr(self, 'q') and self.q is not None else None
            except Exception:
                q_arr = None
            roi_min = getattr(self, '_roi_min', None)
            roi_max = getattr(self, '_roi_max', None)

            I_exp_full = self.I
            I_fit_full = self.I_fitting

            if (
                q_arr is not None and q_arr.size == I_exp_full.size == I_fit_full.size and
                roi_min is not None and roi_max is not None and
                np.isfinite(roi_min) and np.isfinite(roi_max) and roi_min < roi_max
            ):
                mask = (
                    np.isfinite(q_arr) & np.isfinite(I_exp_full) & np.isfinite(I_fit_full) &
                    (q_arr >= float(roi_min)) & (q_arr <= float(roi_max))
                )
                if np.any(mask):
                    I_exp_used = I_exp_full[mask]
                    I_fit_used = I_fit_full[mask]
                else:
                    I_exp_used = I_exp_full
                    I_fit_used = I_fit_full
            else:
                I_exp_used = I_exp_full
                I_fit_used = I_fit_full

            # ??1????????????????????????????ROI??? I_exp_used/I_fit_used??
            k_safe = max(abs(current_k), 1e-12)  # ??????
            I_base = I_fit_used / k_safe

            # ??2???????????????
            # ????? ||k * I_base - I_exp||^2 ???? = 0
            # ???: k_opt = (I_base ? I_exp) / (I_base ? I_base)
            I_exp_flat = I_exp_used.flatten()
            I_base_flat = I_base.flatten()

            # ??????????
            valid_mask = np.isfinite(I_exp_flat) & np.isfinite(I_base_flat) & (I_base_flat != 0)

            if not np.any(valid_mask):
                self._add_fitting_error("No valid data points for K optimization")
                return

            I_exp_valid = I_exp_flat[valid_mask]
            I_base_valid = I_base_flat[valid_mask]

            # ????????
            numerator = np.dot(I_base_valid, I_exp_valid)
            denominator = np.dot(I_base_valid, I_base_valid)

            if denominator <= 1e-12:
                self._add_fitting_error("Base function has zero norm, cannot optimize K")
                return

            k_opt = numerator / denominator

            # ??K?????
            if k_opt <= 0:
                # ?????????????
                try:
                    from scipy.optimize import nnls
                    # ???????????? A*k = b?????A = I_base_valid.reshape(-1,1), b = I_exp_valid
                    A = I_base_valid.reshape(-1, 1)
                    b = I_exp_valid
                    k_nnls, residual = nnls(A, b)
                    k_opt = k_nnls[0] if len(k_nnls) > 0 else 1.0
                    optimization_method = "NNLS"
                except ImportError:
                    self._add_fitting_warning("scipy.optimize.nnls not available, using absolute value of analytical solution")
                    k_opt = abs(k_opt)
                    optimization_method = "Analytical (abs)"
            else:
                optimization_method = "Analytical"

            # ?????????
            if not np.isfinite(k_opt) or k_opt <= 0:
                self._add_fitting_error(f"Invalid optimized K-value: {k_opt}")
                return

            # ??????????????
            residual_before = np.sum((current_k * I_base_valid - I_exp_valid) ** 2)
            residual_after = np.sum((k_opt * I_base_valid - I_exp_valid) ** 2)

            # ??3???????????
            if I_base.size == self.I_fitting.size:
                # ?????????????????
                self.I_fitting = k_opt * (self.I_fitting / k_safe)
            else:
                # ?????OI????????OI???????????????????????????
                try:
                    if 'mask' in locals() and mask is not None and mask.size == self.I_fitting.size:
                        I_base_full = self.I_fitting / k_safe
                        I_base_full = np.asarray(I_base_full)
                        I_base_full[mask] = I_base  # ROI?????????????????
                        self.I_fitting = k_opt * I_base_full
                    else:
                        # ?????????mask?????????????
                        self.I_fitting = k_opt * (self.I_fitting / k_safe)
                except Exception:
                    self.I_fitting = k_opt * (self.I_fitting / k_safe)

            # ??4??????????????
            success = self.set_global_parameter('k_value', k_opt)
            if not success:
                self._add_fitting_error("Failed to set optimized K-value")
                return

            # ??5????????I?????K?????
            if hasattr(self.ui, 'fitKValue'):
                # ??????????????????????
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(k_opt)
                self.ui.fitKValue.blockSignals(False)
                self._add_fitting_message(f"UI K-value updated to {k_opt:.6f}")

            # ??????????????????k ??????????????????? k
            try:
                if isinstance(getattr(self, 'fitting', None), dict):
                    meta = self.fitting.get('meta') or {}
                    params_meta = meta.get('params') or {}
                    # params_template ??????? 'k'
                    params_meta['k'] = float(k_opt)
                    meta['params'] = params_meta
                    self.fitting['meta'] = meta
            except Exception:
                pass

            # ??6????????
            self._update_GUI_image('fitting')
            self._update_outside_window('fitting')

            # ?????????
            improvement = ((residual_before - residual_after) / max(residual_before, 1e-12)) * 100

            self._add_fitting_success(
                f"K-value optimized ({optimization_method}): {current_k:.6f} ??{k_opt:.6f}"
            )
            self._add_fitting_success(
                f"Residual improvement: {improvement:.2f}% "
                f"({residual_before:.6e} ??{residual_after:.6e})"
            )

            # ?????????????
            data_info = f"Data range - I_exp: [{np.min(I_exp_valid):.3e}, {np.max(I_exp_valid):.3e}], "
            data_info += f"I_base: [{np.min(I_base_valid):.3e}, {np.max(I_base_valid):.3e}]"
            self._add_fitting_message(data_info)

        except ImportError:
            self._add_fitting_error("scipy.optimize.nnls not available, using analytical solution only")
            # ???????????????????????
        except Exception as e:
            self._add_fitting_error(f"Error during K-value optimization: {e}")
            # ??K????
            if 'current_k' in locals():
                self.set_global_parameter('k_value', current_k)

    # ================================
    # ?????????
    # ================================




    def _on_parameter_editing_finished(self, widget_id: int, shape: str, param: str):
        """No description."""
        try:
            # ?????????????????????????
            if self._loading_parameters or self._initializing:
                return

            # ???????????
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape)
            widget_name = param_mapping.get(param)

            if widget_name and hasattr(self.ui, widget_name):
                widget = getattr(self.ui, widget_name)
                current_value = widget.value()

                # ????????SON
                particle_id = f"particle_{widget_id}"
                success = self.model_params_manager.set_particle_parameter('fitting', particle_id, shape.lower(), param, current_value)

                if success:
                    self.model_params_manager.save_parameters()
                    self._add_particle_message(f"?? Saved to JSON: {particle_id}.{shape.lower()}.{param} = {current_value}")
                else:
                    self._add_fitting_error(f"Failed to save parameter: {particle_id}.{shape.lower()}.{param} = {current_value}")

            # ??????????????????
            is_fitting_mode = self._is_in_fitting_mode()

            if is_fitting_mode:
                self._add_particle_message(f"?? Fitting mode: Auto-updating after {shape}.{param} edit finished")
                # ????????????
                self._perform_manual_fitting()
            else:
                self._add_particle_message(f"?? Normal mode: Parameter {shape}.{param} edit finished (saved only)")

        except Exception as e:
            self._add_fitting_error(f"Failed to handle parameter editing finished: {e}")

    def _on_global_parameter_editing_finished(self, param_name: str):
        """No description."""
        try:
            # ?????????????????????????
            if self._loading_parameters or self._initializing:
                return

            # ???????????????
            current_value = None
            if param_name == 'background' and hasattr(self.ui, 'fitBGValue'):
                current_value = self.ui.fitBGValue.value()
            elif param_name == 'sigma_res' and hasattr(self.ui, 'fitSigmaResValue'):
                current_value = self.ui.fitSigmaResValue.value()
            elif param_name == 'nu_res' and hasattr(self.ui, 'fitNuResValue'):
                current_value = self.ui.fitNuResValue.value()
            elif param_name == 'int_res' and hasattr(self.ui, 'fitIntResValue'):
                current_value = self.ui.fitIntResValue.value()
            elif param_name == 'k_value' and hasattr(self.ui, 'fitKValue'):
                current_value = self.ui.fitKValue.value()

            if current_value is not None:
                # ???????????SON
                success = self.model_params_manager.set_global_parameter('fitting', param_name, current_value)

                if success:
                    self.model_params_manager.save_parameters()
                    self._add_particle_message(f"?? Saved global parameter to JSON: {param_name} = {current_value}")
                else:
                    self._add_fitting_error(f"Failed to save global parameter: {param_name} = {current_value}")

            # ??????????????????
            is_fitting_mode = self._is_in_fitting_mode()

            if is_fitting_mode:
                self._add_particle_message(f"?? Fitting mode: Auto-updating after global {param_name} edit finished")
                # ????????????
                self._perform_manual_fitting()
            else:
                self._add_particle_message(f"?? Normal mode: Global parameter {param_name} edit finished (saved only)")

        except Exception as e:
            self._add_fitting_error(f"Failed to handle global parameter editing finished: {e}")

    def _is_in_fitting_mode(self) -> bool:
        """No description."""
        return hasattr(self, '_fitting_mode_active') and self._fitting_mode_active





    # ================================
    # ?????????
    # ================================

    def _get_fitting_parameter_comment_lines(self):
        """No description."""
        lines = ["# Fitting Parameters Begin"]
        try:
            import re
            from utils.fitting import params_template

            shapes = []
            param_dict = None
            param_source = "current_ui_snapshot"
            widget_ids = list(getattr(self, '_last_active_particle_ids', []) or [])

            if isinstance(getattr(self, 'fitting', None), dict):
                meta = self.fitting.get('meta', {})
                fit_shapes = meta.get('shapes')
                fit_params = meta.get('params')
                if fit_shapes and fit_params:
                    shapes = [str(shape).lower() for shape in fit_shapes]
                    param_dict = {str(k): float(v) for k, v in dict(fit_params).items()}
                    param_source = "last_fitting_result"

            if not shapes:
                shapes, widget_ids = self._collect_active_particles()

            if not param_dict and shapes:
                shape_list, params_list = self._get_last_fitting_spec_and_params(fallback_shapes=shapes)
                if shape_list and params_list:
                    shapes = list(shape_list)
                    param_dict = {
                        str(name): float(value)
                        for name, value in zip(params_template(shapes), params_list)
                    }

            if not shapes or not param_dict:
                lines.append("# Parameter Source: unavailable")
                lines.append("# No fitting parameter snapshot available")
                lines.append("# Fitting Parameters End")
                return lines

            template = params_template(shapes)
            lines.append(f"# Parameter Source: {param_source}")
            lines.append(f"# Active Shapes: {', '.join(shapes)}")

            grouped_particle_params = {}
            global_params = []
            for template_name in template:
                match = re.match(r'^(.*?)(\d+)$', str(template_name))
                if match:
                    param_base = match.group(1)
                    particle_index = int(match.group(2))
                    grouped_particle_params.setdefault(particle_index, []).append((template_name, param_base))
                else:
                    global_params.append(template_name)

            for particle_index in sorted(grouped_particle_params.keys()):
                shape = shapes[particle_index - 1] if particle_index - 1 < len(shapes) else 'unknown'
                widget_id = widget_ids[particle_index - 1] if particle_index - 1 < len(widget_ids) else particle_index
                lines.append(f"# Particle {particle_index}: widget_id={widget_id}, shape={shape}")
                for template_name, _param_base in grouped_particle_params[particle_index]:
                    if template_name in param_dict:
                        lines.append(f"#   {template_name} = {float(param_dict[template_name]):.10g}")

            if global_params:
                lines.append("# Global Parameters:")
                for template_name in global_params:
                    if template_name in param_dict:
                        lines.append(f"#   {template_name} = {float(param_dict[template_name]):.10g}")

        except Exception as e:
            lines.append(f"# Fitting parameter export error: {e}")

        lines.append("# Fitting Parameters End")
        return lines

    def _build_export_header_lines(self, choice: str, data_name: str):
        """No description."""
        lines = []
        try:
            from datetime import datetime

            q_source_kind = None
            if choice == 'Cut Data':
                q_source_kind = 'cut'
            elif choice == '1D File Data':
                q_source_kind = '1d'
            elif choice == 'Fitting Data' and isinstance(getattr(self, 'fitting', None), dict):
                q_source_kind = self.fitting.get('meta', {}).get('data_source', getattr(self, 'data_source', None))

            lines.append("# GIMaP Export")
            lines.append(f"# Export Time: {datetime.now().isoformat(timespec='seconds')}")
            lines.append(f"# Data Type: {choice}")
            lines.append(f"# Export Name: {data_name}")
            lines.append(f"# Display Mode: {getattr(self, 'display_mode', 'normal')}")
            lines.append(f"# Log X: {self._is_fit_log_x_enabled()}")
            lines.append(f"# Log Y: {self._is_fit_log_y_enabled()}")
            lines.append(f"# Normalize: {self._is_fit_norm_enabled()}")
            lines.append(f"# Axis Filter: {self._get_independent_axis_filter_mode()}")
            lines.append(f"# Raw q Source Unit: {self._get_q_source_unit(q_source_kind)}")
            lines.append("# Internal Model q Unit: nm^-1")
            lines.append(f"# q Unit: {self._get_q_unit_label(mathtext=False)}")
            lines.append(f"# X Column: {self._build_q_axis_label(filter_mode='all', mathtext=False)}")
            lines.append("# Y Column: Intensity (a.u.)")

            if self._roi_min is not None and self._roi_max is not None:
                lines.append(f"# ROI Range: {float(self._roi_min):.10g} -> {float(self._roi_max):.10g}")

            if choice == '1D File Data' and getattr(self, 'current_1d_data', None) is not None:
                file_path = self.current_1d_data.get('file_path')
                if file_path:
                    lines.append(f"# 1D File: {file_path}")
            elif choice == 'Cut Data' and getattr(self, 'cut', None) is not None:
                cut_meta = self.cut.get('meta', {}) if isinstance(self.cut, dict) else {}
                title = cut_meta.get('title')
                if title:
                    lines.append(f"# Cut Title: {title}")

        except Exception:
            pass

        lines.extend(self._get_fitting_parameter_comment_lines())
        return lines

    def _export_fitting_data(self):
        """Fitting"""
        try:
            import numpy as np

            # ???itGraphicsView?????
            if not hasattr(self.ui, 'fitGraphicsView') or self.ui.fitGraphicsView is None:
                self._add_fitting_error("fitGraphicsView is not available")
                return

            # ????????????????
            options = []
            if getattr(self, 'cut', None) is not None:
                options.append('Cut Data')
            if getattr(self, 'fitting', None) is not None:
                options.append('Fitting Data')
            if getattr(self, 'current_1d_data', None) is not None:
                options.append('1D File Data')
            if not options:
                self._add_fitting_error("No available data to export (need Cut, Fitting, or 1D data)")
                return

            # ??????????ut????itting?????D
            default_index = 0
            # ???????????????????
            choice, ok = QInputDialog.getItem(
                None,
                "Select Data to Export",
                "Data source:",
                options,
                default_index,
                False
            )
            if not ok:
                return

            # ????????x/y
            x_data = None
            y_data = None
            data_name = ""
            q_source_kind = None
            if choice == 'Cut Data' and self.cut is not None:
                x_data = np.array(self.cut.get('q', []))
                y_data = np.array(self.cut.get('I', []))
                data_name = 'Cut_Data'
                q_source_kind = 'cut'
            elif choice == 'Fitting Data' and self.fitting is not None:
                x_data = np.array(self.fitting.get('q', []))
                y_data = np.array(self.fitting.get('I', []))
                data_name = 'Fitting_Data'
                q_source_kind = self.fitting.get('meta', {}).get('data_source', getattr(self, 'data_source', None))
            elif choice == '1D File Data' and self.current_1d_data is not None:
                x_data = np.array(self.current_1d_data.get('q', []))
                y_data = np.array(self.current_1d_data.get('I', []))
                data_name = '1D_File_Data'
                q_source_kind = '1d'
            else:
                self._add_fitting_error("Selected data is not available to export")
                return

            # ??????????????
            filename, _ = QFileDialog.getSaveFileName(
                None,
                f"Export {data_name}",
                f"{data_name}.txt",
                "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
            )

            if not filename:
                return  # ???????????

            # ????????????
            min_length = min(len(x_data), len(y_data))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]

            # ?????X???????????????????
            x_data = self._convert_q_values_for_display(x_data, source=q_source_kind)
            x_column_name = self._build_q_axis_label(filter_mode='all', mathtext=False)
            y_column_name = 'Intensity (a.u.)'

            # ??????????????
            combined_data = np.column_stack([x_data, y_data])
            header_lines = self._build_export_header_lines(choice, data_name)

            # ???????????????????????????????? # ???????????
            delimiter = ',' if filename.lower().endswith('.csv') else '\t'
            column_header = f'{x_column_name},{y_column_name}' if delimiter == ',' else f'{x_column_name}\t{y_column_name}'

            with open(filename, 'w', encoding='utf-8', newline='\n') as f:
                if header_lines:
                    f.write('\n'.join(header_lines) + '\n')
                f.write(column_header + '\n')
                np.savetxt(f, combined_data, delimiter=delimiter, fmt='%.6e')

            self._add_fitting_success(f"{data_name} exported successfully to: {filename}")

        except Exception as e:
            self._add_fitting_error(f"Export failed: {str(e)}")

    def _show_manual_auto_refine_dialog(self):
        """Open a local least-squares refine dialog based on current manual fitting parameters."""
        try:
            if not SCIPY_AVAILABLE:
                QMessageBox.warning(
                    self.main_window or self.ui,
                    "Auto Refine",
                    "SciPy is required for Auto Refine. Please install scipy first.",
                )
                return

            setup = self._build_manual_refine_setup()
            if setup is None:
                return

            dialog = QDialog(self.main_window or self.ui)
            dialog.setWindowTitle("Auto Refine Manual Fit")
            dialog.resize(980, 640)
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            layout = QVBoxLayout(dialog)

            info = QLabel(
                "Choose parameters to refine. Current manual parameters are used as initial values; "
                "refined values will be written back to the fitting controls.",
                dialog,
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            run_settings = self._ai_run_settings()
            controls = QGridLayout()
            controls.addWidget(QLabel("Max eval:", dialog), 0, 0)
            max_eval = QSpinBox(dialog)
            max_eval.setRange(1, 100000)
            max_eval.setValue(int(run_settings.get("full_refine_max_nfev", 120)))
            controls.addWidget(max_eval, 0, 1)

            controls.addWidget(QLabel("Target logRMSE:", dialog), 0, 2)
            target = QDoubleSpinBox(dialog)
            target.setDecimals(8)
            target.setRange(0.0, 10.0)
            target.setSingleStep(0.00000001)
            target.setValue(float(run_settings.get("full_refine_target_logrmse", 0.0)))
            controls.addWidget(target, 0, 3)

            controls.addWidget(QLabel("ftol:", dialog), 1, 0)
            ftol = QDoubleSpinBox(dialog)
            ftol.setDecimals(10)
            ftol.setRange(0.0, 1.0)
            ftol.setSingleStep(0.00000001)
            ftol.setValue(float(run_settings.get("full_refine_ftol", 1e-8)))
            controls.addWidget(ftol, 1, 1)

            controls.addWidget(QLabel("xtol:", dialog), 1, 2)
            xtol = QDoubleSpinBox(dialog)
            xtol.setDecimals(10)
            xtol.setRange(0.0, 1.0)
            xtol.setSingleStep(0.00000001)
            xtol.setValue(float(run_settings.get("full_refine_xtol", 1e-8)))
            controls.addWidget(xtol, 1, 3)

            controls.addWidget(QLabel("gtol:", dialog), 1, 4)
            gtol = QDoubleSpinBox(dialog)
            gtol.setDecimals(10)
            gtol.setRange(0.0, 1.0)
            gtol.setSingleStep(0.00000001)
            gtol.setValue(float(run_settings.get("full_refine_gtol", 1e-8)))
            controls.addWidget(gtol, 1, 5)

            controls.addWidget(QLabel("Progress every nfev:", dialog), 2, 0)
            progress_every = QSpinBox(dialog)
            progress_every.setRange(1, 10000)
            progress_every.setValue(max(1, int(run_settings.get("full_refine_progress_interval", 5) or 5)))
            progress_every.setToolTip("Update progress every N estimated SciPy least_squares function evaluations.")
            controls.addWidget(progress_every, 2, 1)
            controls.addWidget(QLabel("Show every:", dialog), 2, 2)
            show_every = QSpinBox(dialog)
            show_every.setRange(0, 10000)
            show_every.setValue(10)
            show_every.setToolTip("Update the Fitting Plot every N estimated SciPy least_squares function evaluations; 0 disables live plot updates.")
            controls.addWidget(show_every, 2, 3)
            controls.setColumnStretch(6, 1)
            layout.addLayout(controls)

            table = QTableWidget(len(setup["params"]), 5, dialog)
            table.setHorizontalHeaderLabels(["Refine", "Parameter", "Current", "Min", "Max"])
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            for col in range(2, 5):
                table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
            layout.addWidget(table, 1)

            row_widgets = []
            cached_rows = self._manual_refine_dialog_state()
            initializing_rows = True
            for row, desc in enumerate(setup["params"]):
                value = float(desc["value"])
                default_selected = self._manual_refine_default_selected(desc["name"])
                lower, upper = self._default_manual_refine_bounds(desc["name"], value)
                cached = cached_rows.get(str(desc["name"]), {}) if isinstance(cached_rows, dict) else {}
                if isinstance(cached, dict):
                    default_selected = bool(cached.get("checked", default_selected))
                    try:
                        lower = float(cached.get("min", lower))
                        upper = float(cached.get("max", upper))
                    except Exception:
                        pass

                check = QCheckBox(table)
                check.setChecked(default_selected)
                table.setCellWidget(row, 0, check)
                table.setItem(row, 1, QTableWidgetItem(str(desc["label"])))
                table.setItem(row, 2, QTableWidgetItem(f"{value:.10g}"))

                min_box = QDoubleSpinBox(table)
                max_box = QDoubleSpinBox(table)
                for spin in (min_box, max_box):
                    spin.setDecimals(8)
                    spin.setRange(-1e12, 1e12)
                    spin.setSingleStep(max(abs(value) * 0.01, 1e-8))
                min_box.setValue(float(lower))
                max_box.setValue(float(upper))
                table.setCellWidget(row, 3, min_box)
                table.setCellWidget(row, 4, max_box)
                row_widgets.append((desc, check, min_box, max_box))
            initializing_rows = False

            def persist_row_state():
                if initializing_rows:
                    return
                rows = {}
                for desc, check, min_box, max_box in row_widgets:
                    rows[str(desc["name"])] = {
                        "checked": bool(check.isChecked()),
                        "min": float(min_box.value()),
                        "max": float(max_box.value()),
                    }
                self._save_manual_refine_dialog_state(rows)

            for _desc, check, min_box, max_box in row_widgets:
                check.toggled.connect(lambda _checked=False: persist_row_state())
                min_box.valueChanged.connect(lambda _value: persist_row_state())
                max_box.valueChanged.connect(lambda _value: persist_row_state())

            result_label = QLabel("Ready.", dialog)
            result_label.setWordWrap(True)
            layout.addWidget(result_label)
            progress_bar = QProgressBar(dialog)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            layout.addWidget(progress_bar)

            button_row = QHBoxLayout()
            select_all = QPushButton("Select All", dialog)
            clear = QPushButton("Clear", dialog)
            run = QPushButton("Run Refine", dialog)
            stop = QPushButton("Stop", dialog)
            stop.setEnabled(False)
            apply_current = QPushButton("Apply Current", dialog)
            apply_current.setEnabled(False)
            close = QPushButton("Close", dialog)
            button_row.addWidget(select_all)
            button_row.addWidget(clear)
            button_row.addStretch(1)
            button_row.addWidget(run)
            button_row.addWidget(stop)
            button_row.addWidget(apply_current)
            button_row.addWidget(close)
            layout.addLayout(button_row)

            def set_selected(predicate):
                for desc, check, _min_box, _max_box in row_widgets:
                    check.setChecked(bool(predicate(desc)))
                persist_row_state()

            select_all.clicked.connect(lambda: set_selected(lambda _desc: True))
            clear.clicked.connect(lambda: set_selected(lambda _desc: False))
            close.clicked.connect(dialog.close)
            refine_state = {
                "thread": None,
                "worker": None,
                "latest_result": None,
                "running": False,
            }

            def set_running_state(running: bool):
                refine_state["running"] = bool(running)
                run.setEnabled(not running)
                stop.setEnabled(running)
                apply_current.setEnabled(refine_state["latest_result"] is not None and not running)
                close.setEnabled(not running)
                for widget in (select_all, clear, table, max_eval, target, ftol, xtol, gtol, progress_every, show_every):
                    widget.setEnabled(not running)

            def apply_result(result):
                if not result:
                    return
                self._apply_manual_refine_result(setup, result["params"])
                self._perform_manual_fitting()
                for row, value in enumerate(result["params"]):
                    table.setItem(row, 2, QTableWidgetItem(f"{float(value):.10g}"))
                self._add_fitting_success(
                    f"Applied Auto Refine parameters: logRMSE={float(result.get('final_log_rmse', np.nan)):.6g}"
                )

            def on_progress(payload):
                refine_state["latest_result"] = payload
                max_nfev = max(1, int(payload.get("max_nfev", max_eval.value())))
                nfev = int(payload.get("nfev_est", payload.get("nfev", payload.get("calls", 0))))
                calls = int(payload.get("calls", 0))
                progress_bar.setValue(max(0, min(99, int(100 * nfev / max_nfev))))
                result_label.setText(
                    f"Running: nfev~{nfev}/{max_nfev}, residual calls={calls}, "
                    f"current logRMSE={float(payload.get('current_log_rmse', np.nan)):.6g}, "
                    f"best={float(payload.get('final_log_rmse', payload.get('best_log_rmse', np.nan))):.6g}"
                )
                show_interval = int(payload.get("show_interval", show_every.value()) or 0)
                if show_interval > 0 and nfev > 0 and (nfev == 1 or nfev % show_interval == 0):
                    self._preview_manual_refine_curve(setup, payload.get("params"))

            def finish_worker():
                thread = refine_state.get("thread")
                worker = refine_state.get("worker")
                if thread is not None:
                    thread.quit()
                    thread.wait(3000)
                    thread.deleteLater()
                if worker is not None:
                    worker.deleteLater()
                refine_state["thread"] = None
                refine_state["worker"] = None
                set_running_state(False)

            def on_finished(result):
                refine_state["latest_result"] = result
                progress_bar.setValue(100 if not result.get("stopped") else progress_bar.value())
                if result.get("stopped"):
                    result_label.setText(
                        f"Stopped: best logRMSE {result['initial_log_rmse']:.6g} -> {result['final_log_rmse']:.6g}; "
                        "click Apply Current to save the current best parameters."
                    )
                    self._add_fitting_warning("Auto Refine stopped. Current best parameters are available to apply.")
                else:
                    apply_result(result)
                    result_label.setText(
                        f"Done: logRMSE {result['initial_log_rmse']:.6g} -> {result['final_log_rmse']:.6g}; "
                        f"nfev={result['nfev']}; {result['message']}"
                    )
                    self._add_fitting_success(result_label.text())
                finish_worker()

            def on_failed(message):
                result_label.setText(f"Auto Refine failed: {message}")
                self._add_fitting_error(f"Auto Refine failed: {message}")
                finish_worker()

            def stop_refine():
                worker = refine_state.get("worker")
                if worker is not None:
                    worker.request_stop()
                    result_label.setText("Stopping Auto Refine after the current residual evaluation...")
                    stop.setEnabled(False)

            stop.clicked.connect(stop_refine)
            apply_current.clicked.connect(lambda: apply_result(refine_state.get("latest_result")))

            def on_dialog_finished(_result):
                worker = refine_state.get("worker")
                if worker is not None:
                    worker.request_stop()

            dialog.finished.connect(on_dialog_finished)

            def run_refine():
                try:
                    options = {
                        "max_nfev": int(max_eval.value()),
                        "target_logrmse": float(target.value()),
                        "ftol": float(ftol.value()) if ftol.value() > 0 else None,
                        "xtol": float(xtol.value()) if xtol.value() > 0 else None,
                        "gtol": float(gtol.value()) if gtol.value() > 0 else None,
                        "progress_interval": int(progress_every.value()),
                        "show_interval": int(show_every.value()),
                    }
                    persist_row_state()
                    self._save_ai_fitting_settings(
                        full_refine_max_nfev=int(max_eval.value()),
                        full_refine_target_logrmse=float(target.value()),
                        full_refine_ftol=float(ftol.value()),
                        full_refine_xtol=float(xtol.value()),
                        full_refine_gtol=float(gtol.value()),
                        full_refine_progress_interval=int(progress_every.value()),
                    )
                    selected = []
                    for desc, check, min_box, max_box in row_widgets:
                        if not check.isChecked():
                            continue
                        lo = float(min_box.value())
                        hi = float(max_box.value())
                        if hi <= lo:
                            raise ValueError(f"{desc['label']} max must be greater than min.")
                        selected.append((desc, lo, hi))
                    if not selected:
                        QMessageBox.information(dialog, "Auto Refine", "Select at least one parameter to refine.")
                        return
                    refine_state["latest_result"] = None
                    progress_bar.setValue(0)
                    result_label.setText("Refining...")
                    thread = QThread(dialog)
                    worker = ManualAutoRefineWorker(self, setup, selected, options)
                    worker.moveToThread(thread)
                    thread.started.connect(worker.run)
                    worker.progress.connect(on_progress)
                    worker.finished.connect(on_finished)
                    worker.failed.connect(on_failed)
                    refine_state["thread"] = thread
                    refine_state["worker"] = worker
                    set_running_state(True)
                    thread.start()
                except Exception as exc:
                    result_label.setText(f"Auto Refine failed: {exc}")
                    self._add_fitting_error(f"Auto Refine failed: {exc}")

            run.clicked.connect(run_refine)
            dialog.finished.connect(lambda _result: setattr(self, "_manual_auto_refine_dialog", None))
            self._manual_auto_refine_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

        except Exception as e:
            self._add_fitting_error(f"Failed to open Auto Refine: {e}")

    def _build_manual_refine_setup(self):
        try:
            from utils.fitting import make_mixed_model, params_template

            active_shapes, shape_configs = self._collect_active_particles()
            if not active_shapes:
                self._add_fitting_error("No active particle shapes selected for Auto Refine")
                return None

            q_data = None
            y_data = None
            q_source_kind = None
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                if getattr(self, 'current_cut_data', None) is not None:
                    q_data = np.asarray(self.current_cut_data.get('x_coords'), dtype=float)
                    y_data = np.asarray(self.current_cut_data.get('y_intensity'), dtype=float)
                    q_source_kind = 'cut'
            else:
                if getattr(self, 'current_1d_data', None) is not None:
                    q_data = np.asarray(self.current_1d_data.get('q'), dtype=float)
                    y_data = np.asarray(self.current_1d_data.get('I'), dtype=float)
                    q_source_kind = '1d'
            if q_data is None or y_data is None:
                self._add_fitting_error("No input curve available for Auto Refine")
                return None

            n = min(q_data.size, y_data.size)
            q_data, y_data = q_data[:n], y_data[:n]
            q_data, y_data = self._filter_ai_excluded_points_for_display(q_data, y_data)
            mask = np.isfinite(q_data) & np.isfinite(y_data) & (y_data > 0)
            if self._roi_active():
                lo = min(float(self._roi_min), float(self._roi_max))
                hi = max(float(self._roi_min), float(self._roi_max))
                mask &= (q_data >= lo) & (q_data <= hi)
            q_data, y_data = q_data[mask], y_data[mask]
            if q_data.size < 8:
                self._add_fitting_error("Auto Refine needs at least 8 valid positive-intensity points")
                return None

            q_model = self._convert_q_values_for_model(q_data, source=q_source_kind)
            model_func = make_mixed_model(active_shapes)
            param_names = params_template(active_shapes)
            params = self._get_current_manual_param_values(active_shapes, shape_configs)
            if not params or len(params) != len(param_names):
                self._add_fitting_error("Could not read current manual fitting parameters")
                return None

            descriptors = self._build_manual_refine_param_descriptors(active_shapes, shape_configs, param_names, params)
            self._last_active_particle_ids = shape_configs.copy()
            return {
                "shapes": active_shapes,
                "shape_configs": shape_configs,
                "q_raw": q_data,
                "q_model": q_model,
                "y": y_data,
                "q_source_kind": q_source_kind,
                "model_func": model_func,
                "param_names": param_names,
                "params": descriptors,
            }
        except Exception as exc:
            self._add_fitting_error(f"Auto Refine setup failed: {exc}")
            return None

    def _get_current_manual_param_values(self, active_shapes, shape_configs):
        param_aliases = {
            "intensity": "Int",
            "radius": "R",
            "sigma_radius": "sigma_R",
            "height": "h",
            "sigma_height": "sigma_h",
            "diameter": "D",
            "sigma_diameter": "sigma_D",
        }
        params = []
        for shape, widget_id in zip(active_shapes, shape_configs):
            shape_display = self._shape_display_name(shape)
            schema = COMPONENT_PARAMETER_SCHEMAS.get(shape_display, [])
            for param_key, _suffix, _label, default_value, _decimals, _step in schema:
                alias = param_aliases[param_key]
                params.append(float(self._get_particle_parameter(widget_id, alias, default_value)))

        global_defaults = [
            ("fitBGValue", "background", 0.0),
            ("fitSigmaResValue", "sigma_res", 0.1),
            ("fitNuResValue", "nu_res", 5.0),
            ("fitIntResValue", "int_res", 0.0),
            ("fitKValue", "k_value", 1.0),
        ]
        for widget_name, global_key, default in global_defaults:
            if hasattr(self.ui, widget_name):
                params.append(float(getattr(self.ui, widget_name).value()))
            elif hasattr(self, "get_global_parameter"):
                params.append(float(self.get_global_parameter(global_key)))
            else:
                params.append(float(default))
        return params

    def _build_manual_refine_param_descriptors(self, active_shapes, shape_configs, param_names, params):
        descriptors = []
        global_map = {
            "BG": ("fitBGValue", "background", "Global BG"),
            "sigma_Res": ("fitSigmaResValue", "sigma_res", "Global sigma_Res"),
            "nu_Res": ("fitNuResValue", "nu_res", "Global nu_Res"),
            "int_Res": ("fitIntResValue", "int_res", "Global int_Res"),
            "k": ("fitKValue", "k_value", "Global k"),
        }
        for idx, (name, value) in enumerate(zip(param_names, params)):
            match = re.match(r'^(.*?)(\d+)$', str(name))
            desc = {
                "index": idx,
                "name": str(name),
                "value": float(value),
                "scope": "global",
                "label": str(name),
                "widget_name": None,
                "global_key": None,
                "widget_id": None,
                "shape": None,
                "alias": None,
            }
            if match:
                alias = match.group(1)
                seq_index = int(match.group(2))
                widget_id = shape_configs[seq_index - 1] if 1 <= seq_index <= len(shape_configs) else None
                shape = active_shapes[seq_index - 1] if 1 <= seq_index <= len(active_shapes) else None
                widget_name = self._get_ui_control_name(widget_id, shape, alias) if widget_id and shape else None
                desc.update({
                    "scope": "particle",
                    "label": f"Particle {seq_index} ({self._shape_display_name(shape)} {widget_id}) {alias}",
                    "widget_name": widget_name,
                    "widget_id": widget_id,
                    "shape": shape,
                    "alias": alias,
                })
            else:
                widget_name, global_key, label = global_map.get(str(name), (None, None, str(name)))
                desc.update({
                    "scope": "global",
                    "label": label,
                    "widget_name": widget_name,
                    "global_key": global_key,
                })
            descriptors.append(desc)
        return descriptors

    def _manual_refine_default_selected(self, name: str) -> bool:
        base = re.sub(r'\d+$', '', str(name))
        return base in {"Int", "BG", "int_Res", "k"}

    def _manual_refine_dialog_state(self) -> dict:
        try:
            from core.user_settings import user_settings
            state = user_settings.get("manual_auto_refine", {})
            return state if isinstance(state, dict) else {}
        except Exception:
            return getattr(self, "_manual_auto_refine_state", {}) if isinstance(getattr(self, "_manual_auto_refine_state", None), dict) else {}

    def _save_manual_refine_dialog_state(self, rows: dict) -> None:
        rows = rows if isinstance(rows, dict) else {}
        self._manual_auto_refine_state = rows
        try:
            from core.user_settings import user_settings
            user_settings.set("manual_auto_refine", rows)
            user_settings.save_settings()
        except Exception:
            pass

    def _default_manual_refine_bounds(self, name: str, value: float):
        base = re.sub(r'\d+$', '', str(name))
        value = float(value)
        if base == "BG":
            return 0.0, max(abs(value) * 10.0, 1.0)
        if base in {"sigma_R", "sigma_h", "sigma_D"}:
            return 0.0, max(abs(value) * 5.0, 1.0)
        if base in {"nu_Res"}:
            return 0.1, max(abs(value) * 4.0, 50.0)
        return 0.0, max(abs(value) * 10.0, 1.0)

    def _run_manual_auto_refine(self, setup, selected, options, progress_callback=None, stop_callback=None):
        model_func = setup["model_func"]
        q_model = np.asarray(setup["q_model"], dtype=float)
        y = np.asarray(setup["y"], dtype=float)
        params0 = np.array([float(desc["value"]) for desc in setup["params"]], dtype=float)
        variable_indices = [int(desc["index"]) for desc, _lo, _hi in selected]
        lower = np.array([float(lo) for _desc, lo, _hi in selected], dtype=float)
        upper = np.array([float(hi) for _desc, _lo, hi in selected], dtype=float)
        x0 = params0[variable_indices].copy()
        x0 = np.minimum(np.maximum(x0, lower + 1e-15), upper - 1e-15)

        n_variables = max(1, int(x0.size))
        calls_per_nfev_estimate = n_variables + 1
        progress = {
            "best": np.inf,
            "best_x": x0.copy(),
            "calls": 0,
            "current": np.inf,
            "last_report_nfev": -1,
        }
        target_logrmse = float(options.get("target_logrmse", 0.0) or 0.0)
        progress_interval = max(1, int(options.get("progress_interval", 5) or 5))
        show_interval = int(options.get("show_interval", 0) or 0)
        max_nfev = int(options.get("max_nfev", 120))

        def build_params(x):
            params = params0.copy()
            params[variable_indices] = x
            return params

        def log_rmse_for_params(params):
            y_model = np.asarray(model_func(q_model, *params), dtype=float)
            if y_model.shape != y.shape:
                y_model = y_model[:y.shape[0]]
            if not np.all(np.isfinite(y_model)):
                return np.inf, y_model
            eps = 1e-30
            residual = np.log10(np.maximum(y_model, eps)) - np.log10(np.maximum(y, eps))
            return float(np.sqrt(np.mean(residual * residual))), y_model

        initial_log_rmse, _ = log_rmse_for_params(params0)
        if progress_callback:
            progress_callback({
                "params": params0.copy(),
                "initial_log_rmse": initial_log_rmse,
                "final_log_rmse": initial_log_rmse,
                "best_log_rmse": initial_log_rmse,
                "current_log_rmse": initial_log_rmse,
                "nfev": 0,
                "nfev_est": 0,
                "calls": 0,
                "max_nfev": max_nfev,
                "show_interval": show_interval,
                "message": "started",
                "stopped": False,
            })

        def residuals(x):
            if stop_callback and stop_callback():
                raise RuntimeError("__AUTO_REFINE_STOP_REQUESTED__")
            params = build_params(x)
            y_model = np.asarray(model_func(q_model, *params), dtype=float)
            eps = 1e-30
            if y_model.shape != y.shape or not np.all(np.isfinite(y_model)):
                return np.full_like(y, 1e6, dtype=float)
            residual = np.log10(np.maximum(y_model, eps)) - np.log10(np.maximum(y, eps))
            current = float(np.sqrt(np.mean(residual * residual)))
            progress["calls"] += 1
            progress["current"] = current
            nfev_est = max(1, int(np.ceil(progress["calls"] / calls_per_nfev_estimate)))
            if current < progress["best"]:
                progress["best"] = current
                progress["best_x"] = np.array(x, dtype=float, copy=True)
            if progress_callback and (
                progress["calls"] == 1
                or (
                    nfev_est != progress["last_report_nfev"]
                    and nfev_est % progress_interval == 0
                )
            ):
                progress["last_report_nfev"] = nfev_est
                best_params = build_params(progress["best_x"])
                progress_callback({
                    "params": best_params,
                    "initial_log_rmse": initial_log_rmse,
                    "final_log_rmse": float(progress["best"]),
                    "best_log_rmse": float(progress["best"]),
                    "current_log_rmse": current,
                    "nfev": int(nfev_est),
                    "nfev_est": int(nfev_est),
                    "calls": int(progress["calls"]),
                    "max_nfev": max_nfev,
                    "show_interval": show_interval,
                    "message": "running",
                    "stopped": False,
                })
            if target_logrmse > 0 and current <= target_logrmse:
                raise RuntimeError("__AUTO_REFINE_TARGET_REACHED__")
            return residual

        stopped = False
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                max_nfev=max_nfev,
                ftol=options.get("ftol"),
                xtol=options.get("xtol"),
                gtol=options.get("gtol"),
            )
            x_final = result.x
            message = str(result.message)
            nfev = int(result.nfev)
        except RuntimeError as exc:
            text = str(exc)
            if "__AUTO_REFINE_TARGET_REACHED__" not in text and "__AUTO_REFINE_STOP_REQUESTED__" not in text:
                raise
            x_final = np.array(progress.get("best_x", x0), dtype=float, copy=True)
            stopped = "__AUTO_REFINE_STOP_REQUESTED__" in text
            message = "Stopped by user." if stopped else "Stopped after reaching target logRMSE."
            nfev = max(1, int(np.ceil(int(progress.get("calls", 0)) / calls_per_nfev_estimate)))

        final_params = build_params(x_final)
        final_log_rmse, _ = log_rmse_for_params(final_params)
        result_payload = {
            "params": final_params,
            "initial_log_rmse": initial_log_rmse,
            "final_log_rmse": final_log_rmse,
            "nfev": nfev,
            "nfev_est": nfev,
            "calls": int(progress.get("calls", nfev)),
            "max_nfev": max_nfev,
            "show_interval": show_interval,
            "message": message,
            "stopped": stopped,
        }
        if progress_callback:
            progress_callback(result_payload)
        return result_payload

    def _apply_manual_refine_result(self, setup, refined_params):
        old_loading = getattr(self, "_loading_parameters", False)
        self._loading_parameters = True
        try:
            for desc, value in zip(setup["params"], refined_params):
                value = float(value)
                widget_name = desc.get("widget_name")
                if widget_name and hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    if hasattr(widget, "blockSignals"):
                        widget.blockSignals(True)
                    try:
                        widget.setValue(value)
                    finally:
                        if hasattr(widget, "blockSignals"):
                            widget.blockSignals(False)

                if desc.get("scope") == "particle":
                    widget_id = desc.get("widget_id")
                    shape = desc.get("shape")
                    alias = desc.get("alias")
                    if widget_id and shape and alias and hasattr(self, "model_params_manager"):
                        particle_id = f"particle_{widget_id}"
                        param_key = self._parameter_key_from_alias(shape, alias)
                        self.model_params_manager.set_particle_parameter(
                            "fitting",
                            particle_id,
                            self._shape_key(shape),
                            param_key,
                            value,
                        )
                elif desc.get("global_key") and hasattr(self, "model_params_manager"):
                    self.model_params_manager.set_global_parameter("fitting", desc["global_key"], value)
            try:
                self.model_params_manager.save_parameters()
            except Exception:
                pass
        finally:
            self._loading_parameters = old_loading

    def _preview_manual_refine_curve(self, setup, params):
        if params is None:
            return
        try:
            params = np.asarray(params, dtype=float)
            q_raw = None
            if getattr(self, "q", None) is not None:
                q_raw = np.asarray(self.q, dtype=float)
                q_raw = q_raw[np.isfinite(q_raw)]
            if q_raw is None or q_raw.size == 0:
                q_raw = np.asarray(setup.get("q_raw", setup["q_model"]), dtype=float)
            q_model = self._convert_q_values_for_model(q_raw, source=setup.get("q_source_kind"))
            y_fit = np.asarray(setup["model_func"](q_model, *params), dtype=float)
            if y_fit.size == 0:
                return
            param_dict = {
                str(name): float(value)
                for name, value in zip(setup.get("param_names", []), params)
            }
            self.I_fitting = y_fit
            self.has_fitting_data = True
            self._has_fitting_data = True
            self.fitting = {
                "q": np.array(q_raw[: y_fit.size], copy=True),
                "I": np.array(y_fit, copy=True),
                "meta": {
                    "shapes": list(setup.get("shapes", [])),
                    "params": param_dict,
                    "source": "auto_refine_preview",
                    "data_source": setup.get("q_source_kind"),
                    "q_source_unit": self._get_q_source_unit(setup.get("q_source_kind")),
                    "q_model_unit": "nm",
                    "preview": True,
                },
            }
            self.display_mode = "fitting"
            self._display_mode = "fitting"
            self._fitting_mode_active = True
            self._update_GUI_image("fitting")
            self._update_outside_window("fitting")
        except Exception as exc:
            self._add_fitting_error(f"Auto Refine preview update failed: {exc}")


    def _perform_manual_fitting(self):
        """No description."""
        try:
            from utils.fitting import make_mixed_model, params_template, mixed_model_components

            # 1. ?????????ComboBox?????
            active_shapes, shape_configs = self._collect_active_particles()

            if not active_shapes:
                self._add_fitting_error("No active particle shapes selected for fitting")
                return

            self._add_fitting_success(f"Active shapes: {active_shapes}")
            self._last_active_particle_ids = shape_configs.copy()

            # 2. ???q?????
            q_data = None
            q_source_kind = None
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # ???Cut???????????Q?????nm^-1??
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    q_data = np.array(self.current_cut_data['x_coords'])
                    q_source_kind = 'cut'
                else:
                    self._add_fitting_error("No Cut data available for fitting")
                    return
            else:
                # ???1D????????????????^-1 ???????????nm^-1 ?????
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    q_data = np.array(self.current_1d_data['q'])
                    q_source_kind = '1d'
                else:
                    self._add_fitting_error("No 1D file data available for fitting")
                    return

            q_model = self._convert_q_values_for_model(q_data, source=q_source_kind)
            self._add_fitting_success(
                f"q converted to internal model unit nm^-1 (source={self._get_q_source_unit(q_source_kind)}, display={self._get_q_display_unit()})"
            )

            # 3. ?????????
            model_func = make_mixed_model(active_shapes)
            param_names = params_template(active_shapes)

            self._add_fitting_success(f"Created model with parameters: {param_names}")

            # 4. Read current component parameter values from the dynamic schema.
            params = []
            param_aliases = {
                "intensity": "Int",
                "radius": "R",
                "sigma_radius": "sigma_R",
                "height": "h",
                "sigma_height": "sigma_h",
                "diameter": "D",
                "sigma_diameter": "sigma_D",
            }
            for i, shape in enumerate(active_shapes, 1):
                shape_idx = shape_configs[i - 1]
                shape_display = self._shape_display_name(shape)
                schema = COMPONENT_PARAMETER_SCHEMAS.get(shape_display, [])
                shape_values = {}
                for param_key, _suffix, _label, default_value, _decimals, _step in schema:
                    alias = param_aliases[param_key]
                    value = self._get_particle_parameter(shape_idx, alias, default_value)
                    params.append(value)
                    shape_values[alias] = value

                d_param = shape_values.get("D", 0.0)
                sigma_d_param = shape_values.get("sigma_D", 0.0)
                if d_param == 0 or sigma_d_param == 0:
                    self._add_fitting_success(
                        f"Shape {i} ({shape_display}): Structure factor disabled (D={d_param}, sigma_D={sigma_d_param})"
                    )
                else:
                    self._add_fitting_success(
                        f"Shape {i} ({shape_display}): Structure factor enabled (D={d_param}, sigma_D={sigma_d_param})"
                    )

            # 5. ????????????????????igma_Res??u_Res??nt_Res??
            # ?????I?????????????BG??????????????????????

            if hasattr(self.ui, 'fitBGValue'):
                bg_param = self.ui.fitBGValue.value()
            else:
                bg_param = self.get_global_parameter('background') if hasattr(self, 'get_global_parameter') else 0.0

            # ??? sigma_Res (Br)
            if hasattr(self.ui, 'fitSigmaResValue'):
                sigma_res_param = self.ui.fitSigmaResValue.value()
            else:
                sigma_res_param = self.get_global_parameter('sigma_res') if hasattr(self, 'get_global_parameter') else 0.1

            # ??? nu_Res (Lorentzian???)
            if hasattr(self.ui, 'fitNuResValue'):
                nu_res_param = self.ui.fitNuResValue.value()
            else:
                nu_res_param = self.get_global_parameter('nu_res') if hasattr(self, 'get_global_parameter') else 5.0

            # ??? int_Res (Lorentzian???)
            if hasattr(self.ui, 'fitIntResValue'):
                int_res_param = self.ui.fitIntResValue.value()
            else:
                int_res_param = self.get_global_parameter('int_res') if hasattr(self, 'get_global_parameter') else 0.0

            # ??? k (??????)
            if hasattr(self.ui, 'fitKValue'):
                k_param = self.ui.fitKValue.value()
            else:
                k_param = self.get_global_parameter('k_value') if hasattr(self, 'get_global_parameter') else 1.0

            # ?????????
            if sigma_res_param == 0 or int_res_param == 0:
                self._add_fitting_success(
                    f"Lorentzian resolution component disabled (sigma_res={sigma_res_param}, int_res={int_res_param})"
                )
            else:
                self._add_fitting_success(
                    f"Lorentzian resolution active: sigma_res={sigma_res_param}, nu_res={nu_res_param}, int_res={int_res_param}"
                )

            params.extend([bg_param, sigma_res_param, nu_res_param, int_res_param, k_param])

            # ??????????????
            param_dict = dict(zip(param_names, params))
            self._add_fitting_success(f"Using parameters: {param_dict}")

            # ?????????
            self._validate_parameter_retrieval(active_shapes, shape_configs)

            # 6. ????????
            try:
                # ??????????????????????? q[nm^-1]??? R/D/h[nm] ?????
                fitting_result = model_func(q_model, *params)
                self._add_fitting_success(f"Fitting calculation completed successfully")

                # ???????????????????
                result_stats = {
                    'min': float(np.min(fitting_result)),
                    'max': float(np.max(fitting_result)),
                    'mean': float(np.mean(fitting_result)),
                    'sum': float(np.sum(fitting_result))
                }
                self._add_fitting_success(f"Result stats: {result_stats}")

                # ???????????I_fitting
                self.I_fitting = fitting_result
                self.has_fitting_data = True
                # ????????????????????
                try:
                    self._has_fitting_data = True
                except Exception:
                    pass
                # ????????itting ???????????????????
                try:
                    import time
                    self.fitting = {
                        'q': np.array(q_data, copy=True),
                        'I': np.array(fitting_result, copy=True),
                        'meta': {
                            'shapes': active_shapes,
                            'params': param_dict,
                            'timestamp': time.time(),
                            'source': 'fitting',
                            'data_source': q_source_kind,
                            'q_source_unit': self._get_q_source_unit(q_source_kind),
                            'q_model_unit': 'nm'
                        }
                    }
                except Exception:
                    self.fitting = {'q': q_data, 'I': fitting_result, 'meta': {'source': 'fitting'}}

                # ???????????Fitting with data
                self.display_mode = 'fitting'
                self._fitting_mode_active = True  # ????????????

                # ??????
                self._update_GUI_image('fitting')
                self._update_outside_window('fitting')

                # ???????uto-K??????????????????????
                if self._auto_k_enabled and hasattr(self, 'I') and self.I is not None:
                    try:
                        self._optimize_k_value()
                    except Exception as opt_error:
                        self._add_fitting_error(f"Auto K-value optimization failed: {opt_error}")

            except Exception as calc_error:
                self._add_fitting_error(f"Fitting calculation failed: {str(calc_error)}")

        except Exception as e:
            self._add_fitting_error(f"Manual fitting failed: {str(e)}")

    def _store_fitting_data(self, q_data, intensity_data, active_shapes):
        """No description."""
        try:
            self._fitting_q_data = np.array(q_data)
            self._fitting_intensity_data = np.array(intensity_data)
            self._fitting_shapes = active_shapes.copy() if active_shapes else []
            self._has_fitting_data = True

            # ???GUI???????????
            self._update_gui_fitting_display()

        except Exception as e:
            pass

    def _switch_to_fitting_display_mode(self):
        """No description."""
        try:
            # ????????????????????
            self._display_mode = 'fitting'
            self.display_mode = 'fitting'
            self._fitting_mode_active = True

            # ???GUI???????????
            self._refresh_all_displays_for_fitting_mode()

        except Exception as e:
            pass

    def _switch_to_normal_display_mode(self):
        """No description."""
        try:
            # ????????????????????
            self._display_mode = 'normal'
            self.display_mode = 'normal'
            self._fitting_mode_active = False

            # ??????????????????????
            self._fitting_q_data = None
            self._fitting_intensity_data = None
            self._fitting_shapes = []
            self._has_fitting_data = False
            # ??????????????????????????????????
            try:
                self.has_fitting_data = False
                self.I_fitting = None
            except Exception:
                pass

        except Exception as e:
            pass

    def _update_gui_fitting_display(self):
        """GUI??????????itGraphicsView"""
        try:
            if not hasattr(self, '_fitting_q_data') or self._fitting_q_data is None:
                return
            # ?????????????????????????????????????fitting??????fitting???????????
            try:
                self.display_mode = 'fitting'
                self._display_mode = 'fitting'
                self._fitting_mode_active = True
            except Exception:
                pass

            # ??UI???????????????????????
            self._plot_fitting_result(self._fitting_q_data, self._fitting_intensity_data, self._fitting_shapes)

        except Exception as e:
            pass

    def _refresh_all_displays_for_fitting_mode(self):
        """No description."""
        try:
            if not self._has_fitting_data:
                return

            # 1. ???GUI???
            self._update_gui_fitting_display()

            # 2. ???????????????????????
            if (hasattr(self, 'independent_fit_window') and
                self.independent_fit_window is not None and
                self.independent_fit_window.isVisible()):

                self._refresh_external_window_fitting_display()

        except Exception as e:
            pass

    def _refresh_external_window_fitting_display(self):
        """No description."""
        try:
            if not self._has_fitting_data:
                return

            # ????????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            # ????????
            original_x_data = None
            original_y_data = None
            data_label = ""

            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data['x_coords'])
                    original_y_data = np.array(self.current_cut_data['y_intensity'])
                    data_label = "Cut Data"
            else:
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data['q'])
                    original_y_data = np.array(self.current_1d_data['I'])
                    data_label = "1D File Data"

            # ????????????
            x_label = self._build_q_axis_label()
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Manual Fitting Result - {", ".join(self._fitting_shapes)}'

            self._update_independent_window_with_fitting(
                original_x_data, original_y_data, data_label,
                self._fitting_q_data, self._fitting_intensity_data, self._fitting_shapes,
                x_label, y_label, title,
                log_x, log_y, normalize
            )

        except Exception as e:
            pass

    def _get_particle_parameter(self, shape_idx, param_name, default_value):
        """No description."""
        try:
            # ??????????????
            current_shape = self.get_particle_shape(shape_idx)
            if current_shape == 'None':
                return default_value

            # ?????I????????????UI??????????
            control_name = self._get_ui_control_name(shape_idx, current_shape, param_name)
            if control_name and hasattr(self.ui, control_name):
                control = getattr(self.ui, control_name)
                if hasattr(control, 'value'):
                    value = control.value()
                    return value
                elif hasattr(control, 'text'):
                    try:
                        value = float(control.text())
                        return value
                    except ValueError:
                        pass

            # ???UI???????????????model_params_manager???
            if hasattr(self, 'model_params_manager'):
                particle_id = f"particle_{shape_idx}"
                shape_key = self._shape_key(current_shape)
                param_key = self._parameter_key_from_alias(current_shape, param_name)
                value = self.model_params_manager.get_particle_parameter(
                    'fitting', particle_id, shape_key, param_key
                )
                if value is not None:
                    return value

            # ???????
            return default_value

        except Exception as e:
            return default_value

    def _get_ui_control_name(self, shape_idx, shape_name, param_name):
        """No description."""
        try:
            shape_display = self._shape_display_name(shape_name)
            token = self._shape_object_token(shape_display)
            param_key = self._parameter_key_from_alias(shape_display, param_name)
            suffix = None
            for schema_shape, schema in COMPONENT_PARAMETER_SCHEMAS.items():
                if self._shape_key(schema_shape) != self._shape_key(shape_display):
                    continue
                for candidate_key, candidate_suffix, _label, _default, _decimals, _step in schema:
                    if candidate_key == param_key:
                        suffix = candidate_suffix
                        break
                if suffix:
                    break
            if not suffix:
                return None

            candidate = f'fitParticle{token}{suffix}Value_{shape_idx}'
            return candidate if hasattr(self.ui, candidate) else None

        except Exception:
            return None

    def _plot_fitting_result(self, q_data, intensity_data, active_shapes):
        """No description."""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return

            if not is_matplotlib_available():
                self._add_fitting_error("Matplotlib not available for plotting")
                return

            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            # ?????????????catter???
            original_x_data = None
            original_y_data = None
            data_label = ""

            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # ???Cut???
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data['x_coords'])
                    original_y_data = np.array(self.current_cut_data['y_intensity'])
                    data_label = "Cut Data"
            else:
                # ???1D??????
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data['q'])
                    original_y_data = np.array(self.current_1d_data['I'])
                    data_label = "1D File Data"

            # ??????
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            # ???????????????
            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return

            # ???matplotlib???
            fig = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # ??????????
            fitting_y_data = np.array(intensity_data)  # ????????????????
            plot_original_y = original_y_data.copy() if original_y_data is not None else None
            norm_divisor = None

            if normalize and original_y_data is not None:
                # ?????????????????????????
                max_original = np.max(original_y_data)
                if max_original > 0:
                    norm_divisor = max_original
                    plot_original_y = original_y_data / max_original
                    # ?????????????????????????????
                    fitting_y_data = fitting_y_data / max_original

            original_x_plot = self._convert_q_values_for_display(original_x_data) if original_x_data is not None else None
            fitting_x_plot = self._convert_q_values_for_display(q_data)

            # ???????????scatter??
            if original_x_plot is not None and plot_original_y is not None:
                ax.scatter(original_x_plot, plot_original_y,
                          s=20, alpha=0.7, color='blue',
                          label=data_label, zorder=2)

            # ????????
            ax.plot(fitting_x_plot, fitting_y_data,
                   color='red', linewidth=2,
                   label=f'Fitting ({", ".join(active_shapes)})',
                   zorder=3)

            # ??????????
            x_label = self._build_q_axis_label() if "q" in str(original_x_data).lower() or len(q_data) > 0 else "Position"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Manual Fitting Result - {", ".join(active_shapes)}'

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # ???????????
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)
            ax.tick_params(axis='both', which='both', width=1.6, labelsize=12)

            # ????????????
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=plot_original_y,
                fitting_y=fitting_y_data,
                log_y=log_y,
            )

            # ROI ?????
            self._draw_roi_guides_if_active(ax)

            # ??????
            fig.tight_layout()

            # ????????
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)

            # ????????figure ??canvas ????????????????????
            self._current_fit_figure = fig
            self._current_fit_canvas = canvas

            # ????????????
            if not hasattr(self, 'current_fitting_data'):
                self.current_fitting_data = {}

            self.current_fitting_data = {
                'q': q_data.copy(),
                'I_fitted': intensity_data.copy(),
                'shapes': active_shapes.copy(),
                'title': title,
                'original_x': original_x_data.copy() if original_x_data is not None else None,
                'original_y': original_y_data.copy() if original_y_data is not None else None,
                'data_label': data_label
            }

            self._add_fitting_success(f"Fitting result plotted for shapes: {active_shapes}")

        except Exception as e:
            self._add_fitting_error(f"Failed to plot fitting result: {str(e)}")

    def _get_last_fitting_spec_and_params(self, fallback_shapes=None):
        """ (shapes:list[str], params_in_order:list[float])??
        ?????? self.fitting['meta'] ?? shapes ??params??????????????????????
        ???????????? UI ??????????????????????
        """
        try:
            import re
            from utils.fitting import params_template

            shapes = None
            param_dict = None
            if isinstance(getattr(self, 'fitting', None), dict):
                meta = self.fitting.get('meta', {})
                shapes = meta.get('shapes')
                param_dict = meta.get('params')
            if shapes and param_dict:
                tmpl = params_template(shapes)
                params_list = []
                ok = True
                for name in tmpl:
                    if name in param_dict:
                        params_list.append(float(param_dict[name]))
                    else:
                        ok = False
                        break
                if ok:
                    return shapes, params_list

            act_shapes, act_idx = self._collect_active_particles()
            if not act_shapes:
                return (fallback_shapes, None) if fallback_shapes else (None, None)

            self._last_active_particle_ids = act_idx.copy()
            tmpl = params_template(act_shapes)
            params_list = []

            default_map = {
                'Int': 1.0,
                'R': 10.0,
                'sigma_R': 0.1,
                'D': 100.0,
                'sigma_D': 0.1,
                'h': 20.0,
                'sigma_h': 0.1,
                'BG': 0.0,
                'sigma_Res': 0.1,
                'nu_Res': 5.0,
                'int_Res': 0.0,
                'k': 1.0,
            }
            global_widget_map = {
                'BG': ('fitBGValue', 'background'),
                'sigma_Res': ('fitSigmaResValue', 'sigma_res'),
                'nu_Res': ('fitNuResValue', 'nu_res'),
                'int_Res': ('fitIntResValue', 'int_res'),
                'k': ('fitKValue', 'k_value'),
            }

            for template_name in tmpl:
                match = re.match(r'^(.*?)(\d+)$', str(template_name))
                if match:
                    base_name = match.group(1)
                    seq_index = int(match.group(2))
                    widget_id = act_idx[seq_index - 1] if 1 <= seq_index <= len(act_idx) else None
                    default_value = default_map.get(base_name, 0.0)
                    if widget_id is None:
                        params_list.append(float(default_value))
                    else:
                        params_list.append(float(self._get_particle_parameter(widget_id, base_name, default_value)))
                else:
                    widget_name, global_key = global_widget_map.get(str(template_name), (None, None))
                    default_value = default_map.get(str(template_name), 0.0)
                    if widget_name and hasattr(self.ui, widget_name):
                        params_list.append(float(getattr(self.ui, widget_name).value()))
                    elif global_key and hasattr(self, 'get_global_parameter'):
                        params_list.append(float(self.get_global_parameter(global_key)))
                    else:
                        params_list.append(float(default_value))

            return act_shapes, [float(x) for x in params_list]
        except Exception:
            return (fallback_shapes, None) if fallback_shapes else (None, None)

    def _on_component_checkbox_changed(self, *_):
        """

        ?????Normal ????????????????????Fitting ???????????????????
        """
        try:
            # ??????????????????????
            if not self._is_in_fitting_mode():
                return
            # ?????????????????????????????????UI??????
            self._update_GUI_image('fitting')
            self._update_outside_window('fitting')
        except Exception:
            pass

    def _show_fitting_in_external_window(self, q_data, intensity_data, active_shapes):
        """No description."""
        try:
            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            # ??????????
            original_x_data = None
            original_y_data = None
            data_label = ""

            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # ???Cut???
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data['x_coords'])
                    original_y_data = np.array(self.current_cut_data['y_intensity'])
                    data_label = "Cut Data"
            else:
                # ???1D??????
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data['q'])
                    original_y_data = np.array(self.current_1d_data['I'])
                    data_label = "1D File Data"

            # ?????????????????
            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)

                # ??????
                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
                self.independent_fit_window.show_positive_cb.toggled.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'show_negative_cb'):
                    self.independent_fit_window.show_negative_cb.toggled.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'q_unit_combo'):
                    self.independent_fit_window.q_unit_combo.currentTextChanged.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'y_range_combo'):
                    self.independent_fit_window.y_range_combo.currentTextChanged.connect(self._on_positive_only_changed)
                if hasattr(self.independent_fit_window, 'input_point_delete_requested'):
                    self.independent_fit_window.input_point_delete_requested.connect(self._exclude_ai_input_point_from_plot)
                try:
                    self._sync_axis_filter_controls()
                except Exception:
                    pass

            # ????????????
            x_label = self._build_q_axis_label() if "q" in str(original_x_data).lower() or len(q_data) > 0 else "Position"
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Manual Fitting Result - {", ".join(active_shapes)}'

            # ?????????????- ?????????????????
            self._update_independent_window_with_fitting(
                original_x_data, original_y_data, data_label,
                q_data, intensity_data, active_shapes,
                x_label, y_label, title,
                log_x, log_y, normalize
            )

            # ??????
            if not self.independent_fit_window.isVisible():
                move_window_to_cursor_screen(self.independent_fit_window)
            self.independent_fit_window.show()
            self.independent_fit_window.raise_()
            self.independent_fit_window.activateWindow()

            # ??????
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.setFocus()

            self._add_fitting_success(f"Fitting result displayed in external window")
            return True  # ????????????

        except Exception as e:
            self._add_fitting_error(f"Failed to show fitting in external window: {str(e)}")
            return False  # ????????????

    def _update_independent_window_with_fitting(self, original_x, original_y, data_label,
                                               fitting_x, fitting_y, shapes,
                                               x_label, y_label, title,
                                               log_x, log_y, normalize):
        """No description."""
        try:
            if not hasattr(self.independent_fit_window, 'ax'):
                return

            ax = self.independent_fit_window.ax
            ax.clear()

            # ??????????
            plot_fitting_y = np.array(fitting_y)  # ????????????????
            plot_original_y = np.array(original_y, copy=True) if original_y is not None else None

            if normalize and original_y is not None:
                # ???????????????????????????????
                max_original = np.max(original_y)
                if max_original > 0:
                    plot_original_y = original_y / max_original
                    # ?????????????????????????????
                    plot_fitting_y = fitting_y / max_original

            filter_mode = self._get_independent_axis_filter_mode()
            original_x_plot = original_x
            fitting_x_plot = fitting_x
            original_x_raw_for_delete = None
            if original_x is not None and plot_original_y is not None:
                original_x, plot_original_y = self._filter_ai_excluded_points_for_display(original_x, plot_original_y)
                original_x_raw_for_delete, original_x_plot, plot_original_y, filter_mode = self._filter_q_data_for_independent_display(original_x, plot_original_y)
            if fitting_x is not None and plot_fitting_y is not None:
                _, fitting_x_plot, plot_fitting_y, _ = self._filter_q_data_for_independent_display(fitting_x, plot_fitting_y)

            original_x_plot = self._convert_q_values_for_display(original_x_plot)
            fitting_x_plot = self._convert_q_values_for_display(fitting_x_plot)

            # ???????????scatter??
            if original_x is not None and plot_original_y is not None and len(original_x_plot) > 0:
                ax.scatter(original_x_plot, plot_original_y,
                          s=30, alpha=0.7, color='blue',
                          label=data_label, zorder=2)

            # ????????
            if fitting_x is not None and plot_fitting_y is not None and len(fitting_x_plot) > 0:
                ax.plot(fitting_x_plot, plot_fitting_y,
                       color='red', linewidth=2.5,
                       label=f'Fitting ({", ".join(shapes)})',
                       zorder=3)

            # ??????????
            plot_x_label = x_label
            if isinstance(x_label, str) and 'q' in x_label.lower():
                plot_x_label = self._build_q_axis_label(filter_mode=filter_mode)
            ax.set_xlabel(plot_x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # ???????????
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)

            # ????????????
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=plot_original_y,
                fitting_y=plot_fitting_y,
                log_y=log_y,
            )

            try:
                if (original_x_raw_for_delete is not None and plot_original_y is not None and
                        hasattr(self.independent_fit_window, 'set_deletable_points')):
                    self.independent_fit_window.set_deletable_points(
                        original_x_raw_for_delete,
                        original_x_plot,
                        plot_original_y,
                    )
            except Exception:
                pass

            # ??????
            if hasattr(self.independent_fit_window, 'canvas'):
                self.independent_fit_window.canvas.draw()

        except Exception as e:
            self._add_fitting_error(f"Failed to update independent window with fitting: {str(e)}")

    def _validate_parameter_retrieval(self, active_shapes, shape_configs):
        """No description."""
        try:
            self._add_fitting_success("=== Parameter Retrieval Validation ===")

            for i, shape in enumerate(active_shapes, 1):
                shape_idx = shape_configs[i-1]
                current_shape = self.get_particle_shape(shape_idx)

                self._add_fitting_success(f"Shape {i}: {shape} (widget {shape_idx}, actual: {current_shape})")

                shape_display = self._shape_display_name(shape)
                if self._shape_key(shape_display) == "none":
                    continue
                schema = COMPONENT_PARAMETER_SCHEMAS.get(shape_display, [])
                token = self._shape_object_token(shape_display)

                for param_key, suffix, _label, _default, _decimals, _step in schema:
                    control_name = f"fitParticle{token}{suffix}Value_{shape_idx}"

                    if hasattr(self.ui, control_name):
                        control = getattr(self.ui, control_name)
                        if hasattr(control, 'value'):
                            value = control.value()
                            self._add_fitting_success(f"  {param_key}: {control_name} = {value}")
                        else:
                            self._add_fitting_error(f"  {param_key}: {control_name} has no 'value' method")
                    else:
                        self._add_fitting_error(f"  {param_key}: {control_name} not found in UI")

            # ??????????
            self._add_fitting_success("Global Parameters:")
            if hasattr(self.ui, 'fitBGValue'):
                bg_value = self.ui.fitBGValue.value()
                self._add_fitting_success(f"  background: fitBGValue = {bg_value}")
            else:
                self._add_fitting_error("  fitBGValue not found")

            if hasattr(self.ui, 'fitSigmaResValue'):
                sigma_res = self.ui.fitSigmaResValue.value()
                self._add_fitting_success(f"  sigma_res: fitSigmaResValue = {sigma_res}")
            else:
                self._add_fitting_error("  fitSigmaResValue not found")

            if hasattr(self.ui, 'fitNuResValue'):
                nu_res = self.ui.fitNuResValue.value()
                self._add_fitting_success(f"  nu_res: fitNuResValue = {nu_res}")
            else:
                self._add_fitting_error("  fitNuResValue not found")

            if hasattr(self.ui, 'fitIntResValue'):
                int_res = self.ui.fitIntResValue.value()
                self._add_fitting_success(f"  int_res: fitIntResValue = {int_res}")
            else:
                self._add_fitting_error("  fitIntResValue not found")

            if hasattr(self.ui, 'fitKValue'):
                k_value = self.ui.fitKValue.value()
                self._add_fitting_success(f"  k_value: fitKValue = {k_value}")
            else:
                self._add_fitting_error("  fitKValue not found")

            self._add_fitting_success("=== Validation Complete ===")

        except Exception as e:
            self._add_fitting_error(f"Parameter validation failed: {str(e)}")

    def _clear_fitting_data(self):
        """fitting"""
        try:
            # ????????I_fitting???
            if not hasattr(self, 'I_fitting') or self.I_fitting is None:
                self.status_updated.emit("No fitting data to clear")
                return

            # ???I_fitting???
            self.I_fitting = None
            self.has_fitting_data = False

            # ??????????
            self.display_mode = 'normal'
            self._fitting_mode_active = False  # ????????????

            # ????????????????????
            self._update_GUI_image('normal')
            self._update_outside_window('normal')

            self.status_updated.emit("Fitting data cleared")

        except Exception as e:
            self.status_updated.emit(f"Error clearing fitting data: {str(e)}")

    def _force_update_gui_points_only(self):
        """GUI"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return

            # ????????figure??anvas
            if not hasattr(self, '_current_fit_figure') or self._current_fit_figure is None:
                return

            if not hasattr(self, '_current_fit_canvas') or self._current_fit_canvas is None:
                return

            # ?????????
            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return

            # ?????????????????
            self._current_fit_figure.clear()
            ax = self._current_fit_figure.add_subplot(111)

            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            # ??????
            plot_y = y_data.copy()
            if normalize:
                max_val = np.max(y_data)
                if max_val > 0:
                    plot_y = y_data / max_val

            # ????????
            x_plot = self._convert_q_values_for_display(x_data)
            ax.scatter(x_plot, plot_y, s=30, alpha=0.7, color='blue',
                      label=data_label, zorder=2)

            # ??????????
            x_label = self._build_q_axis_label()
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Data Points Only - {data_label}'

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # ?????????
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            # ?????????
            self._current_fit_canvas.draw()

        except Exception as e:
            # ???????????????????????????????
            pass


    def _update_fitting_plot_points_only(self):
        """No description."""
        try:
            if not hasattr(self, 'current_cut_data') or self.current_cut_data is None:
                return

            # ???????????
            if hasattr(self.ui, 'fitGraphicsView') and hasattr(self, '_current_fit_figure') and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                # ????????og???
                log_x = self._get_checkbox_state('fitLogXCheckBox', False)
                log_y = self._get_checkbox_state('fitLogYCheckBox', False)

                # ????????
                cut_data = self.current_cut_data
                # ???????????
                x_data = None
                y_data = None
                if 'x_coords' in cut_data and 'y_intensity' in cut_data:
                    x_data = cut_data['x_coords']
                    y_data = cut_data['y_intensity']
                elif 'x' in cut_data and 'y' in cut_data:
                    x_data = cut_data['x']
                    y_data = cut_data['y']

                if x_data is not None and y_data is not None:
                    x_plot = self._convert_q_values_for_display(x_data)
                    ax.scatter(x_plot, y_data, c='blue', s=20, alpha=0.7, label='Data')

                    # ????????
                    if log_x:
                        ax.set_xscale('log')
                    if log_y:
                        ax.set_yscale('log')

                    ax.set_xlabel(self._build_q_axis_label())
                    ax.set_ylabel('Intensity')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # ??????
                if hasattr(self, '_current_fit_canvas') and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception as e:
            pass

    def _on_fit_log_changed(self):
        """Log-x/Log-y"""
        try:
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            try:
                self._sync_roi_controls_to_current_display(reset_to_domain=True)
                self._apply_roi_to_data_and_refresh()
            except Exception:
                pass
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Display log scale updated")
            # ??????ROI??????log-x????????????????????????????
            try:
                QTimer.singleShot(0, self._adjust_roi_bounds_for_log_x)
            except Exception:
                self._adjust_roi_bounds_for_log_x()
        except Exception as e:
            self.status_updated.emit(f"Error updating log scale: {str(e)}")

    def _on_normalize_changed(self):
        """Normalize"""
        try:
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Normalize setting updated")
        except Exception as e:
            self.status_updated.emit(f"Error updating normalize setting: {str(e)}")

    def _on_positive_only_changed(self):
        """No description."""
        try:
            if getattr(self, '_syncing_axis_filter', False):
                return

            previous_mode = getattr(self, '_last_axis_filter_mode', 'all')
            self._sync_axis_filter_controls()
            current_filter_mode = self._get_independent_axis_filter_mode()
            self._last_axis_filter_mode = current_filter_mode
            try:
                self._sync_roi_controls_to_current_display(reset_to_domain=(previous_mode != current_filter_mode))
                self._apply_roi_to_data_and_refresh()
            except Exception:
                pass
            mode = self.display_mode if hasattr(self, 'display_mode') else 'normal'
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Display settings synced across main and independent views")
        except Exception as e:
            self.status_updated.emit(f"Error updating display sync: {str(e)}")





    def _update_fitting_plot(self):
        """No description."""
        try:
            if not hasattr(self, 'fitting_data') or self.fitting_data is None:
                return

            if hasattr(self.ui, 'fitGraphicsView') and hasattr(self, '_current_fit_figure') and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                # ????????og???
                log_x = self._get_checkbox_state('fitLogXCheckBox', False)
                log_y = self._get_checkbox_state('fitLogYCheckBox', False)

                # ??????????????????
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    cut_data = self.current_cut_data
                    if 'x_coords' in cut_data and 'y_intensity' in cut_data:
                        ax.scatter(self._convert_q_values_for_display(cut_data['x_coords']), cut_data['y_intensity'], c='blue', s=20, alpha=0.7, label='Data')
                    elif 'x' in cut_data and 'y' in cut_data:
                        ax.scatter(self._convert_q_values_for_display(cut_data['x']), cut_data['y'], c='blue', s=20, alpha=0.7, label='Data')

                # ???????????? self.fitting_data??
                fitting_data = self.fitting_data
                if isinstance(fitting_data, dict) and 'x' in fitting_data and 'y' in fitting_data:
                    ax.plot(self._convert_q_values_for_display(fitting_data['x']), fitting_data['y'], 'r-', linewidth=2, label='Fit')

                # ????????
                if log_x:
                    ax.set_xscale('log')
                if log_y:
                    ax.set_yscale('log')

                ax.set_xlabel(self._build_q_axis_label())
                ax.set_ylabel('Intensity')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # ??????
                if hasattr(self, '_current_fit_canvas') and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception:
            pass

    def _update_fitting_mode_displays_without_line(self):
        """No description."""
        try:
            # 1. ???GUI??? - ???????
            self._update_gui_points_only()

            # 2. ???????????? - ???????
            if (hasattr(self, 'independent_fit_window') and
                self.independent_fit_window is not None and
                self.independent_fit_window.isVisible()):

                self._update_external_window_points_only()

        except Exception as e:
            pass

    def _update_gui_points_only(self):
        """No description."""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return

            # ?????????
            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return

            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            # ??UI???????
            self._plot_data_points_only(x_data, y_data, data_label, log_x, log_y, normalize)

        except Exception as e:
            pass

    def _update_external_window_points_only(self):
        """No description."""
        try:
            if not hasattr(self.independent_fit_window, 'ax'):
                return

            # ?????????
            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return

            # ?????????
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            ax = self.independent_fit_window.ax
            ax.clear()

            # ??????
            plot_y = y_data.copy()
            if normalize:
                max_val = np.max(y_data)
                if max_val > 0:
                    plot_y = y_data / max_val

            x_raw, x_plot, plot_y, filter_mode = self._filter_q_data_for_independent_display(x_data, plot_y)
            x_raw, x_plot, plot_y = self._filter_ai_excluded_points_for_display(x_raw, x_plot, plot_y)
            x_plot = self._convert_q_values_for_display(x_plot)
            if x_plot.size == 0 or plot_y is None or plot_y.size == 0:
                return

            # ???????
            ax.scatter(x_plot, plot_y, s=30, alpha=0.7, color='blue',
                      label=data_label, zorder=2)

            # ??????????????????????
            x_label = self._build_q_axis_label(filter_mode=filter_mode)
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f'Fitting Display Mode - {data_label}'

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # ????????????????????????
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.8)

            # ?????????
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            # ??????
            if hasattr(self.independent_fit_window, 'canvas'):
                try:
                    if hasattr(self.independent_fit_window, 'set_deletable_points'):
                        self.independent_fit_window.set_deletable_points(x_raw, x_plot, plot_y)
                except Exception:
                    pass
                self.independent_fit_window.canvas.draw()

        except Exception as e:
            pass

    def _get_current_data_for_display(self):
        """No description."""
        try:
            if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
                # ???Cut???
                if hasattr(self, 'current_cut_data') and self.current_cut_data is not None:
                    return (np.array(self.current_cut_data['x_coords']),
                           np.array(self.current_cut_data['y_intensity']),
                           "Cut Data")
            else:
                # ???1D??????
                if hasattr(self, 'current_1d_data') and self.current_1d_data is not None:
                    return (np.array(self.current_1d_data['q']),
                           np.array(self.current_1d_data['I']),
                           "1D File Data")

            return None, None, ""

        except Exception as e:
            return None, None, ""

    def _plot_data_points_only(self, x_data, y_data, data_label, log_x, log_y, normalize):
        """UI"""
        try:
            if not hasattr(self.ui, 'fitGraphicsView'):
                return

            # Use the existing fitting GUI figure and canvas
            if hasattr(self, '_current_fit_figure') and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                # Processing data
                plot_y = y_data.copy()
                if normalize:
                    max_val = np.max(y_data)
                    if max_val > 0:
                        plot_y = y_data / max_val

                # Plotting data points
                x_plot = self._convert_q_values_for_display(x_data)
                ax.scatter(x_plot, plot_y, s=30, alpha=0.7, color='blue',
                          label=data_label, zorder=2)

                # Setting up labels and styles
                x_label = self._build_q_axis_label()
                y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
                title = f'Fitting Display Mode - {data_label}'

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Setting logarithmic coordinates
                if log_x:
                    ax.set_xscale('log')
                if log_y:
                    ax.set_yscale('log')

                # Refresh Canvas
                if hasattr(self, '_current_fit_canvas') and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception as e:
            pass
