"""Screen metrics, profile selection and window sizing operations."""

from __future__ import annotations

import os

from PyQt5.QtCore import QPoint, QRect, QSize
from PyQt5.QtGui import QCursor, QScreen
from PyQt5.QtWidgets import QApplication, QAbstractButton, QCheckBox, QSizePolicy, QWidget

from src.gimap.app.ports import UserPreferencesRepository

from .responsive_profiles import (
    LAYOUT_TARGETS,
    PROFILES,
    ResponsiveProfile,
    ScreenMetrics,
    _preference,
    _preferences_for,
    clamp,
    layout_target_resolution,
    normalized_profile_key,
)
def effective_ui_scale(
    window: QWidget | None = None,
    target: QSize | None = None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> int:
    user_scale = float(
        _preference("visual_font_scale", 100, window=window, preferences=preferences)
    )
    if not _preference(
        "auto_fit_layout_target", True, window=window, preferences=preferences
    ):
        return int(round(clamp(user_scale, 40.0, 140.0)))

    target_size = target or layout_target_resolution(preferences=preferences)
    if target_size is None:
        return int(round(clamp(user_scale, 40.0, 140.0)))

    metrics = screen_metrics(window, preferences=preferences)
    actual = metrics.logical_geometry.size()
    fit_scale = min(
        actual.width() / max(1, target_size.width()),
        actual.height() / max(1, target_size.height()),
    )
    auto_fit_scale = clamp(fit_scale, 0.75, 1.0)
    fitted_scale = user_scale * auto_fit_scale
    return int(round(clamp(fitted_scale, 75.0, 140.0)))

def scale_value(
    value: int,
    profile: ResponsiveProfile,
    minimum: int | None = None,
    *,
    window: QWidget | None = None,
    preferences: UserPreferencesRepository | None = None,
) -> int:
    ui_scale = effective_ui_scale(window, preferences=preferences) / 100.0
    scaled = int(round(value * profile.density_scale * ui_scale))
    return max(minimum, scaled) if minimum is not None else scaled

def apply_density_profile(
    root: QWidget,
    profile: ResponsiveProfile,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> None:
    """Scale wrapper-owned control heights for the active screen profile."""
    from .layout_primitives import INPUT_WIDGET_TYPES, SMALL_BUTTON_WIDTH

    button_min = scale_value(32, profile, 28, window=root, preferences=preferences)
    button_max = scale_value(36, profile, 30, window=root, preferences=preferences)
    input_min = scale_value(28, profile, 24, window=root, preferences=preferences)
    compact_width = scale_value(
        SMALL_BUTTON_WIDTH,
        profile,
        30,
        window=root,
        preferences=preferences,
    )

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
        ("FittingControlsCard", 440, 400, None),
        ("ModelParameterCard", 260, 210, None),
        ("DetectorPreviewCard", 260, 210, None),
        ("plotCanvasContainer", 260, 200, None),
        ("PlotPreviewCard", 360, 280, None),
        ("FittingPlotControlsCard", 360, 300, None),
        ("FittingStatusCard", 230, 176, None),
        ("predictModelLibraryCard", 118, 96, 136),
        ("fitMethodWidget", 120, 98, 120),
        ("fitMethodWidget_2", 120, 98, 120),
        ("widget_8", 48, 40, 48),
    ):
        widget = root.findChild(QWidget, name)
        if widget is None:
            continue
        toggle = widget.findChild(QAbstractButton, f"{name}ToggleButton")
        if toggle is not None and hasattr(toggle, "isChecked") and not toggle.isChecked():
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

def screen_at_cursor() -> QScreen | None:
    app = QApplication.instance()
    if app is None:
        return None
    return app.screenAt(QCursor.pos()) or app.primaryScreen()

def move_window_to_cursor_screen(window: QWidget, margin: int = 24) -> QScreen | None:
    """Center a top-level window on the monitor that currently contains the mouse."""
    screen = screen_at_cursor()
    if screen is None:
        return None

    available = screen.availableGeometry()
    size = window.size()
    if not size.isValid() or size.isEmpty():
        size = window.sizeHint()
    if not size.isValid() or size.isEmpty():
        size = window.minimumSizeHint()
    if not size.isValid() or size.isEmpty():
        size = QSize(900, 600)

    max_width = max(320, available.width() - margin * 2)
    max_height = max(240, available.height() - margin * 2)
    if size.width() > max_width or size.height() > max_height:
        size = QSize(min(size.width(), max_width), min(size.height(), max_height))
        window.resize(size)

    x = available.x() + max(0, (available.width() - size.width()) // 2)
    y = available.y() + max(0, (available.height() - size.height()) // 2)
    window.move(QPoint(x, y))
    return screen

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

def screen_metrics(
    window: QWidget | None = None,
    screen: QScreen | None = None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> ScreenMetrics:
    screen = screen or screen_for_window(window)
    logical = screen.availableGeometry() if screen is not None else QRect(0, 0, 1366, 768)
    dpr = _raw_device_pixel_ratio(screen)
    dpi_scale = screen_dpi_scale(screen, window=window, preferences=preferences)
    estimate_scale = max(dpr, dpi_scale, 1.0)
    physical = QSize(
        int(round(logical.width() * estimate_scale)),
        int(round(logical.height() * estimate_scale)),
    )
    name = screen.name() if screen is not None else "Unknown"
    return ScreenMetrics(name, logical, dpr, dpi_scale, physical)

def screen_dpi_scale(
    screen: QScreen | None,
    *,
    window: QWidget | None = None,
    preferences: UserPreferencesRepository | None = None,
) -> float:
    if screen is None or not _preference(
        "auto_detect_monitor_dpi", True, window=window, preferences=preferences
    ):
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

def current_profile(
    window: QWidget | None = None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> ResponsiveProfile:
    repository = _preferences_for(window, preferences)
    return profile_for_screen(screen_for_window(window), preferences=repository)

def profile_for_screen(
    screen: QScreen | None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> ResponsiveProfile:
    environment_profile = normalized_profile_key(
        os.environ.get("GIMAP_LAYOUT_PROFILE", "")
    )
    if environment_profile in PROFILES:
        return PROFILES[environment_profile]
    mode = normalized_profile_key(
        _preference("responsive_layout_mode", "auto", preferences=preferences)
    )
    if mode in PROFILES:
        return PROFILES[mode]
    if not _preference("adaptive_layout_enabled", True, preferences=preferences):
        return PROFILES["normal"]
    target = layout_target_resolution(preferences=preferences)
    if target is not None:
        return PROFILES[profile_key_for_geometry(QRect(0, 0, target.width(), target.height()))]
    return PROFILES[
        auto_profile_key_for_metrics(
            screen_metrics(screen=screen, preferences=preferences)
        )
    ]

def clamp_size_to_screen(size: QSize, geometry: QRect, ratio: float) -> QSize:
    return QSize(
        max(720, min(size.width(), int(geometry.width() * ratio))),
        max(520, min(size.height(), int(geometry.height() * ratio))),
    )

def window_resize_geometry_for_screen(
    screen: QScreen | None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> QRect:
    if screen is None:
        return QRect(0, 0, 1366, 768)
    geometry = screen.availableGeometry()
    scale = screen_dpi_scale(screen, preferences=preferences)
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
    preferences: UserPreferencesRepository | None = None,
) -> ResponsiveProfile:
    preferences = _preferences_for(window, preferences)
    profile = profile or current_profile(window, preferences=preferences)
    screen = screen or screen_for_window(window)
    geometry = window_resize_geometry_for_screen(screen, preferences=preferences)
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

def screen_summary(
    window: QWidget | None = None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> str:
    metrics = screen_metrics(window, preferences=preferences)
    logical = metrics.logical_geometry
    physical = metrics.estimated_physical_size
    return (
        f"{logical.width()} x {logical.height()} logical, "
        f"{physical.width()} x {physical.height()} estimated physical @ "
        f"{int(round(max(metrics.device_pixel_ratio, metrics.dpi_scale) * 100))}%"
    )

def layout_target_warning(
    window: QWidget | None = None,
    *,
    preferences: UserPreferencesRepository | None = None,
) -> str:
    target = layout_target_resolution(preferences=preferences)
    if target is None:
        return ""
    actual = screen_metrics(window).logical_geometry.size()
    if target.width() > actual.width() or target.height() > actual.height():
        return (
            "The selected layout target is larger than the current screen. "
            "GIMaP will use a smaller UI scale and scrolling to keep the interface usable."
        )
    return ""
