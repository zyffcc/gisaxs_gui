"""Stable public API for responsive application layouts.

Implementation is split by profile definitions, screen geometry and adaptive
window lifecycle so callers keep a compact import seam.
"""

from .adaptive_window import AdaptiveWindowProfileController, install_adaptive_window_profile
from .responsive_profiles import (
    LAYOUT_TARGETS,
    PROFILE_ALIASES,
    PROFILES,
    LayoutTarget,
    ResponsiveProfile,
    ScreenMetrics,
    clamp,
    layout_target_label,
    layout_target_resolution,
    manual_screen_resolution,
    normalized_profile_key,
    parse_resolution,
)
from .screen_geometry import (
    apply_density_profile,
    apply_window_profile,
    auto_profile_key_for_metrics,
    available_screen_geometry,
    clamp_size_to_screen,
    current_profile,
    effective_ui_scale,
    layout_target_warning,
    move_window_to_cursor_screen,
    physical_geometry_for_screen,
    physical_screen_geometry,
    profile_key_for_geometry,
    profile_for_screen,
    profile_summary,
    scale_value,
    screen_at_cursor,
    screen_dpi_scale,
    screen_for_window,
    screen_metrics,
    screen_summary,
    window_resize_geometry_for_screen,
)

__all__ = [name for name in globals() if not name.startswith("_")]
