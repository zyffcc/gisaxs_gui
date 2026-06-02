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


@dataclass(frozen=True)
class ScreenMetrics:
    name: str
    logical_geometry: QRect
    device_pixel_ratio: float
    dpi_scale: float
    estimated_physical_size: QSize


@dataclass(frozen=True)
class LayoutTarget:
    key: str
    label: str
    resolution: QSize | None


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

LAYOUT_TARGETS = {
    "auto": LayoutTarget("auto", "Auto", None),
    "compact": LayoutTarget("compact", "Compact / 720p", QSize(1280, 720)),
    "normal": LayoutTarget("normal", "Normal / 1080p", QSize(1920, 1080)),
    "wide": LayoutTarget("wide", "Wide / 1440p", QSize(2560, 1440)),
    "custom": LayoutTarget("custom", "Custom...", None),
}


def normalized_profile_key(key: str | None) -> str:
    key = key or "auto"
    return PROFILE_ALIASES.get(key, key)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def effective_ui_scale(window: QWidget | None = None, target: QSize | None = None) -> int:
    user_scale = float(user_settings.get_visual_font_scale())
    if not user_settings.get("auto_fit_layout_target", True):
        return int(round(clamp(user_scale, 40.0, 140.0)))

    target_size = target or layout_target_resolution()
    if target_size is None:
        return int(round(clamp(user_scale, 40.0, 140.0)))

    metrics = screen_metrics(window)
    actual = metrics.logical_geometry.size()
    fit_scale = min(
        actual.width() / max(1, target_size.width()),
        actual.height() / max(1, target_size.height()),
    )
    auto_fit_scale = clamp(fit_scale, 0.75, 1.0)
    fitted_scale = user_scale * auto_fit_scale
    return int(round(clamp(fitted_scale, 75.0, 140.0)))


def scale_value(value: int, profile: ResponsiveProfile, minimum: int | None = None) -> int:
    ui_scale = effective_ui_scale() / 100.0
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
    if screen is None and app.activeWindow() is not None:
        active = app.activeWindow()
        if active.windowHandle() is not None:
            screen = active.windowHandle().screen()
        if screen is None:
            screen = app.screenAt(active.frameGeometry().center())
    if screen is None:
        screen = app.screenAt(QCursor.pos())
    if screen is None:
        screen = app.primaryScreen()
    return screen


def parse_resolution(value: str | None) -> QSize | None:
    if not value:
        return None
    text = str(value).strip().lower().replace(" ", "")
    if text in {"auto", "automatic", "default"}:
        return None
    separator = "x" if "x" in text else "*" if "*" in text else None
    if separator is None:
        return None
    parts = text.split(separator, 1)
    try:
        width = int(parts[0])
        height = int(parts[1])
    except (TypeError, ValueError):
        return None
    if width < 640 or height < 480:
        return None
    return QSize(width, height)


def manual_screen_resolution() -> QSize | None:
    return parse_resolution(user_settings.get("manual_screen_resolution", "auto"))


def layout_target_resolution() -> QSize | None:
    mode = str(user_settings.get("layout_target_mode", "") or "").strip().lower()
    if not mode:
        legacy = manual_screen_resolution()
        return legacy
    if mode == "auto":
        return None
    if mode == "custom":
        return parse_resolution(user_settings.get("layout_target_custom", ""))
    target = LAYOUT_TARGETS.get(mode)
    return target.resolution if target is not None else None


def layout_target_label() -> str:
    mode = str(user_settings.get("layout_target_mode", "") or "").strip().lower()
    if not mode:
        manual = manual_screen_resolution()
        return f"{manual.width()} x {manual.height()}" if manual is not None else "Auto"
    if mode == "custom":
        custom = parse_resolution(user_settings.get("layout_target_custom", ""))
        return f"{custom.width()} x {custom.height()}" if custom is not None else "Custom"
    target = LAYOUT_TARGETS.get(mode, LAYOUT_TARGETS["auto"])
    if target.resolution is None:
        return target.label
    return f"{target.label} ({target.resolution.width()} x {target.resolution.height()})"


def available_screen_geometry(window: QWidget | None = None) -> QRect:
    screen = screen_for_window(window)
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1366, 768)


def _device_pixel_ratio(screen: QScreen | None) -> float:
    if screen is None:
        return 1.0
    ratios = []
    try:
        ratios.append(float(screen.devicePixelRatio()))
    except Exception:
        pass
    try:
        dpi = float(screen.logicalDotsPerInch())
        if dpi > 0:
            ratios.append(dpi / 96.0)
    except Exception:
        pass
    return max(1.0, *ratios) if ratios else 1.0


def _raw_device_pixel_ratio(screen: QScreen | None) -> float:
    if screen is None:
        return 1.0
    try:
        return max(1.0, float(screen.devicePixelRatio()))
    except Exception:
        return 1.0


def physical_geometry_for_screen(screen: QScreen | None) -> QRect:
    if screen is None:
        return QRect(0, 0, 1366, 768)
    geometry = screen.geometry()
    ratio = _device_pixel_ratio(screen)
    return QRect(
        int(round(geometry.x() * ratio)),
        int(round(geometry.y() * ratio)),
        int(round(geometry.width() * ratio)),
        int(round(geometry.height() * ratio)),
    )


def physical_screen_geometry(window: QWidget | None = None) -> QRect:
    return physical_geometry_for_screen(screen_for_window(window))


def screen_metrics(window: QWidget | None = None, screen: QScreen | None = None) -> ScreenMetrics:
    screen = screen or screen_for_window(window)
    logical = screen.availableGeometry() if screen is not None else QRect(0, 0, 1366, 768)
    dpr = _raw_device_pixel_ratio(screen)
    dpi_scale = screen_dpi_scale(screen)
    estimate_scale = max(dpr, dpi_scale, 1.0)
    physical = QSize(
        int(round(logical.width() * estimate_scale)),
        int(round(logical.height() * estimate_scale)),
    )
    name = screen.name() if screen is not None else "Unknown"
    return ScreenMetrics(name, logical, dpr, dpi_scale, physical)


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


def auto_profile_key_for_metrics(metrics: ScreenMetrics) -> str:
    logical_width = metrics.logical_geometry.width()
    dpr = max(metrics.device_pixel_ratio, metrics.dpi_scale)
    physical_width = metrics.estimated_physical_size.width()

    if logical_width < 1200:
        if dpr >= 1.5 and physical_width >= 2000:
            return "normal"
        return "compact"
    if logical_width >= 2200:
        return "wide"
    if physical_width >= 2500 and logical_width >= 1200:
        return "normal"
    return "normal"


def current_profile(window: QWidget | None = None) -> ResponsiveProfile:
    return profile_for_screen(screen_for_window(window))


def profile_for_screen(screen: QScreen | None) -> ResponsiveProfile:
    mode = normalized_profile_key(user_settings.get("responsive_layout_mode", "auto"))
    if mode in PROFILES:
        return PROFILES[mode]
    if not user_settings.get("adaptive_layout_enabled", True):
        return PROFILES["normal"]
    target = layout_target_resolution()
    if target is not None:
        return PROFILES[profile_key_for_geometry(QRect(0, 0, target.width(), target.height()))]
    return PROFILES[auto_profile_key_for_metrics(screen_metrics(screen=screen))]


def clamp_size_to_screen(size: QSize, geometry: QRect, ratio: float) -> QSize:
    return QSize(
        max(720, min(size.width(), int(geometry.width() * ratio))),
        max(520, min(size.height(), int(geometry.height() * ratio))),
    )


def window_resize_geometry_for_screen(screen: QScreen | None) -> QRect:
    if screen is None:
        return QRect(0, 0, 1366, 768)
    geometry = screen.availableGeometry()
    scale = screen_dpi_scale(screen)
    if scale > 1.05 and (geometry.width() >= 2200 or geometry.height() >= 1400):
        return QRect(
            geometry.x(),
            geometry.y(),
            max(720, int(geometry.width() / scale)),
            max(520, int(geometry.height() / scale)),
        )
    return geometry


def apply_window_profile(
    window: QWidget,
    profile: ResponsiveProfile | None = None,
    *,
    resize_window: bool = False,
    screen: QScreen | None = None,
) -> ResponsiveProfile:
    profile = profile or current_profile(window)
    screen = screen or screen_for_window(window)
    geometry = window_resize_geometry_for_screen(screen)
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
    metrics = screen_metrics(window)
    logical = metrics.logical_geometry
    physical = metrics.estimated_physical_size
    return (
        f"{logical.width()} x {logical.height()} logical, "
        f"{physical.width()} x {physical.height()} estimated physical @ "
        f"{int(round(max(metrics.device_pixel_ratio, metrics.dpi_scale) * 100))}%"
    )


def layout_target_warning(window: QWidget | None = None) -> str:
    target = layout_target_resolution()
    if target is None:
        return ""
    actual = screen_metrics(window).logical_geometry.size()
    if target.width() > actual.width() or target.height() > actual.height():
        return (
            "The selected layout target is larger than the current screen. "
            "GIMaP will use a smaller UI scale and scrolling to keep the interface usable."
        )
    return ""


class AdaptiveWindowProfileController(QObject):
    """Debounced per-window monitor/profile watcher."""

    profileChanged = pyqtSignal(object, object)

    def __init__(self, window: QWidget, callback=None, debounce_ms: int = 200, apply_window_minimum: bool = True):
        super().__init__(window)
        self.window = window
        self.callback = callback
        self.apply_window_minimum = apply_window_minimum
        self._screen = None
        self._profile_signature = None
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
        if watched is self.window and event.type() in (
            QEvent.Move,
            QEvent.Resize,
            QEvent.Show,
            QEvent.MouseButtonRelease,
            QEvent.WindowStateChange,
        ):
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
        self.refresh(screen)
        QTimer.singleShot(0, self._force_layout_refresh)
        self.schedule()

    def _force_layout_refresh(self) -> None:
        widget = self.window
        while widget is not None:
            layout = widget.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
            widget.updateGeometry()
            widget = widget.parentWidget()

    def refresh(self, forced_screen: QScreen | None = None) -> None:
        screen = forced_screen or screen_for_window(self.window)
        self._connect_screen(screen)
        profile = profile_for_screen(screen)
        screen_changed = screen is not self._screen
        signature = (
            profile.key,
            profile.min_window.width(),
            profile.min_window.height(),
            profile.content_min,
            profile.workspace_min,
            profile.preview_min,
            round(profile.density_scale, 3),
            effective_ui_scale(self.window),
        )
        profile_changed = signature != self._profile_signature
        if not screen_changed and not profile_changed:
            return
        self._screen = screen
        self._profile_signature = signature
        if self.apply_window_minimum:
            should_resize = screen_changed and not self.window.isMaximized() and not self.window.isFullScreen()
            apply_window_profile(self.window, profile, resize_window=should_resize, screen=screen)
        if self.callback is not None:
            self.callback(profile, screen)
        self.profileChanged.emit(profile, screen)
        QTimer.singleShot(0, self._force_layout_refresh)


def install_adaptive_window_profile(
    window: QWidget,
    callback=None,
    debounce_ms: int = 200,
    apply_window_minimum: bool = True,
):
    controller = AdaptiveWindowProfileController(window, callback, debounce_ms, apply_window_minimum)
    window._adaptive_profile_controller = controller
    return controller
