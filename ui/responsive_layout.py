"""Screen-aware layout profiles for top-level PyQt windows."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QObject, QEvent, QRect, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QScreen
from PyQt5.QtWidgets import QApplication, QAbstractButton, QCheckBox, QSizePolicy, QWidget

from core.user_settings import user_settings


@dataclass(frozen=True)
class ResponsiveProfile:
    key: str
    label: str
    min_window: QSize
    preferred_window_ratio: float
    sidebar_min: int
    sidebar_max: int
    sidebar_default: int
    content_min: int
    workspace_min: int
    preview_min: int
    page_sizes: tuple[int, int]
    work_sizes: tuple[int, int]
    preview_sizes: tuple[int, int, int]
    font_adjustment: int
    density_scale: float


PROFILES = {
    "compact": ResponsiveProfile(
        key="compact",
        label="Compact",
        min_window=QSize(1040, 700),
        preferred_window_ratio=0.94,
        sidebar_min=170,
        sidebar_max=200,
        sidebar_default=180,
        content_min=820,
        workspace_min=600,
        preview_min=360,
        page_sizes=(620, 380),
        work_sizes=(760, 520),
        preview_sizes=(280, 700, 140),
        font_adjustment=-1,
        density_scale=0.82,
    ),
    "normal": ResponsiveProfile(
        key="normal",
        label="Normal",
        min_window=QSize(1200, 760),
        preferred_window_ratio=0.92,
        sidebar_min=180,
        sidebar_max=220,
        sidebar_default=190,
        content_min=980,
        workspace_min=640,
        preview_min=420,
        page_sizes=(760, 500),
        work_sizes=(760, 680),
        preview_sizes=(300, 860, 160),
        font_adjustment=0,
        density_scale=1.0,
    ),
    "wide": ResponsiveProfile(
        key="wide",
        label="Wide",
        min_window=QSize(1500, 900),
        preferred_window_ratio=0.88,
        sidebar_min=200,
        sidebar_max=240,
        sidebar_default=210,
        content_min=1260,
        workspace_min=760,
        preview_min=500,
        page_sizes=(980, 640),
        work_sizes=(840, 840),
        preview_sizes=(380, 980, 200),
        font_adjustment=1,
        density_scale=1.05,
    ),
}

PROFILE_ALIASES = {
    "standard": "normal",
    "spacious": "wide",
    "manual": "normal",
}


def normalized_profile_key(key: str | None) -> str:
    key = key or "auto"
    return PROFILE_ALIASES.get(key, key)


def scale_value(value: int, profile: ResponsiveProfile, minimum: int | None = None) -> int:
    ui_scale = max(0.8, min(user_settings.get_visual_font_scale() / 100.0, 1.4))
    scaled = int(round(value * profile.density_scale * ui_scale))
    return max(minimum, scaled) if minimum is not None else scaled


def apply_density_profile(root: QWidget, profile: ResponsiveProfile) -> None:
    """Scale wrapper-owned control heights for the active screen profile."""
    from ui.layout_utils import INPUT_WIDGET_TYPES, SMALL_BUTTON_WIDTH

    button_min = scale_value(32, profile, 28)
    button_max = scale_value(36, profile, 30)
    input_min = scale_value(28, profile, 24)
    compact_width = scale_value(SMALL_BUTTON_WIDTH, profile, 30)

    for button in root.findChildren(QAbstractButton):
        if button.maximumWidth() <= SMALL_BUTTON_WIDTH + 4:
            button.setMinimumSize(compact_width, button_min)
            button.setMaximumSize(compact_width, button_max)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            button.setMinimumHeight(button_min)
            button.setMaximumHeight(16777215)
            button.setSizePolicy(button.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)

    for widget in root.findChildren(INPUT_WIDGET_TYPES):
        widget.setMinimumHeight(input_min)
        widget.setMaximumHeight(16777215)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    for checkbox in root.findChildren(QCheckBox):
        checkbox.setMinimumHeight(input_min)
        checkbox.setMaximumHeight(16777215)
        checkbox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    for name, base_min, min_floor, base_max in (
        ("GisaxsInputCard", 260, 210, None),
        ("CutLineCard", 230, 185, None),
        ("FittingControlsCard", 760, 660, None),
        ("ModelParameterCard", 260, 210, None),
        ("DetectorPreviewCard", 260, 210, None),
        ("plotCanvasContainer", 260, 200, None),
        ("PlotPreviewCard", 760, 620, None),
        ("FittingStatusCard", 120, 96, None),
        ("predictModelLibraryCard", 118, 96, 136),
        ("fitMethodWidget", 120, 98, 120),
        ("fitMethodWidget_2", 120, 98, 120),
        ("widget_8", 48, 40, 48),
    ):
        widget = root.findChild(QWidget, name)
        if widget is None:
            continue
        widget.setMinimumHeight(scale_value(base_min, profile, min_floor))
        if base_max is not None:
            widget.setMaximumHeight(scale_value(base_max, profile, min_floor))
        widget.updateGeometry()


def screen_for_window(window: QWidget | None = None) -> QScreen | None:
    app = QApplication.instance()
    if app is None:
        return None

    screen = None
    if window is not None and window.windowHandle() is not None:
        screen = window.windowHandle().screen()
    if screen is None and window is not None:
        screen = app.screenAt(window.frameGeometry().center())
    if screen is None:
        screen = app.screenAt(QCursor.pos())
    if screen is None:
        screen = app.primaryScreen()
    return screen


def available_screen_geometry(window: QWidget | None = None) -> QRect:
    screen = screen_for_window(window)
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1366, 768)


def screen_dpi_scale(screen: QScreen | None) -> float:
    if screen is None or not user_settings.get("auto_detect_monitor_dpi", True):
        return 1.0
    dpi = screen.logicalDotsPerInch()
    return max(1.0, dpi / 96.0) if dpi > 0 else 1.0


def profile_key_for_geometry(geometry: QRect) -> str:
    width = geometry.width()
    if width < 1400:
        return "compact"
    if width < 2200:
        return "normal"
    return "wide"


def current_profile(window: QWidget | None = None) -> ResponsiveProfile:
    mode = normalized_profile_key(user_settings.get("responsive_layout_mode", "auto"))
    if mode in PROFILES:
        return PROFILES[mode]
    if not user_settings.get("adaptive_layout_enabled", True):
        return PROFILES["normal"]
    return PROFILES[profile_key_for_geometry(available_screen_geometry(window))]


def clamp_size_to_screen(size: QSize, geometry: QRect, ratio: float) -> QSize:
    return QSize(
        max(720, min(size.width(), int(geometry.width() * ratio))),
        max(520, min(size.height(), int(geometry.height() * ratio))),
    )


def apply_window_profile(
    window: QWidget,
    profile: ResponsiveProfile | None = None,
    *,
    resize_window: bool = False,
) -> ResponsiveProfile:
    profile = profile or current_profile(window)
    geometry = available_screen_geometry(window)
    min_size = clamp_size_to_screen(profile.min_window, geometry, 0.98)
    window.setMinimumSize(min_size)

    if resize_window:
        target = QSize(
            max(min_size.width(), int(geometry.width() * profile.preferred_window_ratio)),
            max(min_size.height(), int(geometry.height() * profile.preferred_window_ratio)),
        )
        target = clamp_size_to_screen(target, geometry, profile.preferred_window_ratio)
        window.resize(target)
    return profile


def profile_summary(profile: ResponsiveProfile, geometry: QRect) -> str:
    effective = clamp_size_to_screen(profile.min_window, geometry, 0.98)
    return (
        f"{profile.label} ({profile.key}) - screen {geometry.width()} x {geometry.height()}, "
        f"profile minimum {profile.min_window.width()} x {profile.min_window.height()}, "
        f"applied minimum {effective.width()} x {effective.height()}"
    )


def screen_summary(window: QWidget | None = None) -> str:
    screen = screen_for_window(window)
    geometry = screen.availableGeometry() if screen is not None else QRect(0, 0, 1366, 768)
    scale = screen_dpi_scale(screen)
    return f"{geometry.width()} x {geometry.height()} @ {int(round(scale * 100))}%"


class AdaptiveWindowProfileController(QObject):
    """Debounced per-window monitor/profile watcher."""

    profileChanged = pyqtSignal(object, object)

    def __init__(self, window: QWidget, callback=None, debounce_ms: int = 200, apply_window_minimum: bool = True):
        super().__init__(window)
        self.window = window
        self.callback = callback
        self.apply_window_minimum = apply_window_minimum
        self._screen = None
        self._profile_key = None
        self._connected_screens = set()
        self._screen_signal_handlers = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self.refresh)
        window.installEventFilter(self)
        QTimer.singleShot(0, self._connect_window_handle)
        QTimer.singleShot(0, self.refresh)

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() in (QEvent.Move, QEvent.Resize, QEvent.Show):
            self.schedule()
        return super().eventFilter(watched, event)

    def schedule(self) -> None:
        self._timer.start()

    def _connect_window_handle(self) -> None:
        handle = self.window.windowHandle()
        if handle is None:
            QTimer.singleShot(50, self._connect_window_handle)
            return
        try:
            handle.screenChanged.connect(self._on_screen_changed)
        except TypeError:
            pass
        self._connect_screen(handle.screen())

    def _connect_screen(self, screen: QScreen | None) -> None:
        if screen is None or id(screen) in self._connected_screens:
            return
        self._connected_screens.add(id(screen))
        for signal in (
            screen.geometryChanged,
            screen.availableGeometryChanged,
            screen.logicalDotsPerInchChanged,
        ):
            try:
                handler = lambda *args: self.schedule()
                self._screen_signal_handlers.append(handler)
                signal.connect(handler)
            except TypeError:
                pass

    def _on_screen_changed(self, screen: QScreen) -> None:
        self._connect_screen(screen)
        self.schedule()

    def refresh(self) -> None:
        screen = screen_for_window(self.window)
        self._connect_screen(screen)
        profile = current_profile(self.window)
        screen_changed = screen is not self._screen
        profile_changed = profile.key != self._profile_key
        if not screen_changed and not profile_changed:
            return
        self._screen = screen
        self._profile_key = profile.key
        if self.apply_window_minimum:
            apply_window_profile(self.window, profile, resize_window=False)
        if self.callback is not None:
            self.callback(profile, screen)
        self.profileChanged.emit(profile, screen)


def install_adaptive_window_profile(
    window: QWidget,
    callback=None,
    debounce_ms: int = 200,
    apply_window_minimum: bool = True,
):
    controller = AdaptiveWindowProfileController(window, callback, debounce_ms, apply_window_minimum)
    window._adaptive_profile_controller = controller
    return controller
