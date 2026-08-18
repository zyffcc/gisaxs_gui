"""Responsive layout value objects, profile catalogue and user targets."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QRect, QSize
from PyQt5.QtWidgets import QApplication, QWidget

from src.gimap.app.ports import UserPreferencesRepository
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

def _preferences_for(
    window: QWidget | None = None,
    preferences: UserPreferencesRepository | None = None,
) -> UserPreferencesRepository | None:
    if preferences is not None:
        return preferences
    candidates = []
    if window is not None:
        candidates.extend((window, window.window()))
    app = QApplication.instance()
    if app is not None:
        active = app.activeWindow()
        if active is not None:
            candidates.append(active)
        candidates.extend(app.topLevelWidgets())
    for candidate in candidates:
        context = getattr(candidate, "app_context", None)
        repository = getattr(context, "preferences", None)
        if repository is not None:
            return repository
    return None

def _preference(
    key: str,
    default,
    *,
    window: QWidget | None = None,
    preferences: UserPreferencesRepository | None = None,
):
    repository = _preferences_for(window, preferences)
    return repository.get(key, default) if repository is not None else default

def normalized_profile_key(key: str | None) -> str:
    key = key or "auto"
    return PROFILE_ALIASES.get(key, key)

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))

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

def manual_screen_resolution(
    *, preferences: UserPreferencesRepository | None = None
) -> QSize | None:
    return parse_resolution(
        _preference("manual_screen_resolution", "auto", preferences=preferences)
    )

def layout_target_resolution(
    *, preferences: UserPreferencesRepository | None = None
) -> QSize | None:
    mode = str(
        _preference("layout_target_mode", "", preferences=preferences) or ""
    ).strip().lower()
    if not mode:
        legacy = manual_screen_resolution(preferences=preferences)
        return legacy
    if mode == "auto":
        return None
    if mode == "custom":
        return parse_resolution(
            _preference("layout_target_custom", "", preferences=preferences)
        )
    target = LAYOUT_TARGETS.get(mode)
    return target.resolution if target is not None else None

def layout_target_label(
    *, preferences: UserPreferencesRepository | None = None
) -> str:
    mode = str(
        _preference("layout_target_mode", "", preferences=preferences) or ""
    ).strip().lower()
    if not mode:
        manual = manual_screen_resolution(preferences=preferences)
        return f"{manual.width()} x {manual.height()}" if manual is not None else "Auto"
    if mode == "custom":
        custom = parse_resolution(
            _preference("layout_target_custom", "", preferences=preferences)
        )
        return f"{custom.width()} x {custom.height()}" if custom is not None else "Custom"
    target = LAYOUT_TARGETS.get(mode, LAYOUT_TARGETS["auto"])
    if target.resolution is None:
        return target.label
    return f"{target.label} ({target.resolution.width()} x {target.resolution.height()})"
