"""Screen-aware layout breakpoints for the PyQt main window."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QRect, QSize
from PyQt5.QtGui import QCursor
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
    "standard": ResponsiveProfile(
        key="standard",
        label="Standard",
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
    "spacious": ResponsiveProfile(
        key="spacious",
        label="Spacious",
        min_window=QSize(1360, 820),
        preferred_window_ratio=0.90,
        sidebar_min=190,
        sidebar_max=230,
        sidebar_default=200,
        content_min=1120,
        workspace_min=700,
        preview_min=460,
        page_sizes=(860, 560),
        work_sizes=(800, 760),
        preview_sizes=(340, 920, 180),
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


def scale_value(value: int, profile: ResponsiveProfile, minimum: int | None = None) -> int:
    scaled = int(round(value * profile.density_scale))
    return max(minimum, scaled) if minimum is not None else scaled


def apply_density_profile(root: QWidget, profile: ResponsiveProfile) -> None:
    """Scale wrapper-owned control heights for the active screen profile."""
    from ui.layout_utils import INPUT_WIDGET_TYPES, SMALL_BUTTON_WIDTH

    button_min = scale_value(32, profile, 28)
    button_max = scale_value(36, profile, 30)
    input_min = scale_value(28, profile, 24)
    input_max = scale_value(32, profile, 28)
    checkbox_max = scale_value(34, profile, 28)
    compact_width = scale_value(SMALL_BUTTON_WIDTH, profile, 30)

    for button in root.findChildren(QAbstractButton):
        if button.maximumWidth() <= SMALL_BUTTON_WIDTH + 4:
            button.setMinimumSize(compact_width, button_min)
            button.setMaximumSize(compact_width, button_max)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            button.setMinimumHeight(button_min)
            button.setMaximumHeight(button_max)
            button.setSizePolicy(button.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)

    for widget in root.findChildren(INPUT_WIDGET_TYPES):
        widget.setMinimumHeight(input_min)
        widget.setMaximumHeight(input_max)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    for checkbox in root.findChildren(QCheckBox):
        checkbox.setMinimumHeight(input_min)
        checkbox.setMaximumHeight(checkbox_max)
        checkbox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    for name, base_min, min_floor, base_max in (
        ("GisaxsInputCard", 260, 210, None),
        ("CutLineCard", 230, 185, None),
        ("FittingControlsCard", 330, 270, None),
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


def available_screen_geometry(window: QWidget | None = None) -> QRect:
    app = QApplication.instance()
    if app is None:
        return QRect(0, 0, 1366, 768)

    screen = None
    if window is not None and window.windowHandle() is not None:
        screen = window.windowHandle().screen()
    if screen is None:
        screen = app.screenAt(QCursor.pos())
    if screen is None:
        screen = app.primaryScreen()
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1366, 768)


def profile_key_for_geometry(geometry: QRect) -> str:
    width = geometry.width()
    height = geometry.height()
    if width < 1400 or height < 820:
        return "compact"
    if width < 1700 or height < 950:
        return "standard"
    if width < 2300:
        return "spacious"
    return "wide"


def current_profile(window: QWidget | None = None) -> ResponsiveProfile:
    mode = user_settings.get("responsive_layout_mode", "auto")
    if mode in PROFILES:
        return PROFILES[mode]
    return PROFILES[profile_key_for_geometry(available_screen_geometry(window))]


def clamp_size_to_screen(size: QSize, geometry: QRect, ratio: float) -> QSize:
    return QSize(
        max(720, min(size.width(), int(geometry.width() * ratio))),
        max(520, min(size.height(), int(geometry.height() * ratio))),
    )


def apply_window_profile(window: QWidget, profile: ResponsiveProfile | None = None) -> ResponsiveProfile:
    profile = profile or current_profile(window)
    geometry = available_screen_geometry(window)
    min_size = clamp_size_to_screen(profile.min_window, geometry, 0.98)
    window.setMinimumSize(min_size)

    if user_settings.get("responsive_resize_on_start", True):
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
