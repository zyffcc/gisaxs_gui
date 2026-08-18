"""Independent fitting image window composed from focused behaviors."""

from __future__ import annotations


from PyQt5.QtCore import pyqtSignal, Qt

from PyQt5.QtWidgets import (
    QMainWindow,
)

from src.gimap.app.presentation.responsive_layout import (
    install_adaptive_window_profile,
)

from .views import IndependentImageWindowView

from src.gimap.app.presentation.responsive_layout import (
    install_adaptive_window_profile,
)

from .views import IndependentImageWindowView

from .scientific_commands import (
    _create_default_fitting_view_model,
    is_matplotlib_available,
)

from .independent_image_display import IndependentImageDisplayMixin
from .independent_image_interaction import IndependentImageInteractionMixin
from .independent_image_rendering import IndependentImageRenderingMixin
from .independent_image_selection import IndependentImageSelectionMixin


class IndependentMatplotlibWindow(
    IndependentImageDisplayMixin,
    IndependentImageInteractionMixin,
    IndependentImageRenderingMixin,
    IndependentImageSelectionMixin,
    QMainWindow,
    IndependentImageWindowView,
):
    """No description."""

    DEFAULT_TITLE = "GIMaP Image Viewer - Independent Window (right-click to select)"

    SELECTION_TITLE = "GIMaP Image Viewer - Selection Mode (drag to select, Esc to exit)"

    region_selected = pyqtSignal(dict)

    center_picked = pyqtSignal(dict)

    status_updated = pyqtSignal(str)

    display_options_changed = pyqtSignal(dict)

    def __init__(self, parent=None, *, fitting_view_model=None):
        super().__init__(parent)
        self.fitting_view_model = fitting_view_model or _create_default_fitting_view_model()
        self.setupUi(self)
        self.setWindowTitle(self.DEFAULT_TITLE)

        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.pick_center_action = None
        self.ax = None
        self.show_cut_region = True
        self.show_center = True
        self.colormap = "viridis"
        self._dragging_overlay = None
        self._drag_start = None
        self._drag_original_info = None
        self._overlay_press_tolerance_px = 8.0
        self._load_display_options()
        try:
            if is_matplotlib_available():
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import (
                    NavigationToolbar2QT as NavigationToolbar,
                )

                self.figure = Figure(figsize=(10, 8), dpi=100)
                self.canvas = FigureCanvas(self.figure)
                self.toolbar = NavigationToolbar(self.canvas, self)
                self._setup_pick_center_action()
                self._setup_display_option_widgets()
                self.viewerContentLayout.addWidget(self.toolbar)
                self.viewerContentLayout.addWidget(self.canvas)
                self.ax = self.figure.add_subplot(111)
        except Exception:
            pass

        self.current_image = None
        self.colorbar = None

        self.current_xlim = None
        self.current_ylim = None
        self.last_image_shape = None
        self._last_use_log = None
        self._last_show_q_axis = None

        self._q_detector = None
        self._q_cache_key = None
        self._qy_mesh = None
        self._qz_mesh = None

        self.selection_mode = False
        self.pick_center_mode = False
        self.selection_start = None
        self.selection_rect = None
        self.current_selection = None
        self.parameter_selection = None
        self.parameter_selection_center = None
        self.parameter_selection_info = None

        self.setFocusPolicy(Qt.StrongFocus)
        if self.canvas is not None:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
        self.centralwidget.setFocusPolicy(Qt.StrongFocus)

        if self.ax is not None:
            self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
            self.ax.callbacks.connect("ylim_changed", self._on_ylim_changed)

        if self.canvas is not None:
            self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
            self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
            self.canvas.mpl_connect("key_press_event", self._on_key_press)
        install_adaptive_window_profile(
            self, self._apply_screen_profile, apply_window_minimum=False
        )
