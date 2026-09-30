"""Semantic colour and type tokens of the light and dark themes.

Style sheets never contain literal colours: they reference these names as
``@token@`` markers (see ``engine.render``).  Both palettes define exactly the
same names, so any rule written against the light theme also works in dark.
"""

from __future__ import annotations

DEFAULT_MODE = "light"
DEFAULT_FONT_PT = 9.0
FONT_PT_RANGE = (7.0, 14.0)

LIGHT: dict[str, str] = {
    # Surfaces, from the window background up to raised cards and inputs.
    "bg": "#eef1f5",
    "surface": "#ffffff",
    "surface_alt": "#f6f8fb",
    "surface_hover": "#eef3f9",
    "surface_sunken": "#e6ebf1",
    "border": "#d8dee6",
    "border_strong": "#bcc6d2",
    "divider": "#e3e8ee",
    # Text.
    "text": "#1a2230",
    "text_muted": "#566377",
    "text_faint": "#8b97a8",
    "text_disabled": "#a3adba",
    # Accent (selection, primary actions, focus).
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "accent_pressed": "#1e40af",
    "accent_soft": "#e8f0fe",
    "accent_border": "#9dbcf5",
    "on_accent": "#ffffff",
    "focus": "#3b82f6",
    "selection": "#cfe0fc",
    "selection_text": "#1a2230",
    # Status colours: text/icon colour, tinted background, border.
    "success": "#15803d",
    "success_soft": "#ecfdf3",
    "success_border": "#8bd9a8",
    "warning": "#b45309",
    "warning_soft": "#fff8eb",
    "warning_border": "#f5c56b",
    "danger": "#b91c1c",
    "danger_soft": "#fff1f1",
    "danger_border": "#f3a3a3",
    "info": "#1d4ed8",
    "info_soft": "#eef4ff",
    "info_border": "#a9c5f7",
    # Small neutral badges.
    "chip": "#e7ecf2",
    "chip_text": "#334155",
    # Navigation rail (dark in both themes).
    "rail": "#172033",
    "rail_border": "#25324a",
    "rail_hover": "#25324a",
    "rail_text": "#c3cddb",
    "rail_active": "#2563eb",
    "rail_active_text": "#ffffff",
    # Plots (pyqtgraph curves; detector images keep their black canvas).
    "plot_bg": "#ffffff",
    "plot_fg": "#334155",
    "plot_grid": "#e2e8f0",
    # Chrome.
    "scrollbar": "#c3ccd7",
    "scrollbar_hover": "#9aa7b8",
    "tooltip_bg": "#1f2937",
    "tooltip_text": "#f8fafc",
    "disabled_bg": "#eef1f5",
}

DARK: dict[str, str] = {
    "bg": "#13171d",
    "surface": "#1b2028",
    "surface_alt": "#20262f",
    "surface_hover": "#262d38",
    "surface_sunken": "#161a20",
    "border": "#2d3541",
    "border_strong": "#3e4858",
    "divider": "#262d38",
    "text": "#e3e8ef",
    "text_muted": "#a0abba",
    "text_faint": "#6f7b8c",
    "text_disabled": "#5b6573",
    "accent": "#3b82f6",
    "accent_hover": "#5b95f8",
    "accent_pressed": "#2563eb",
    "accent_soft": "#1c2d4a",
    "accent_border": "#34568f",
    "on_accent": "#ffffff",
    "focus": "#60a5fa",
    "selection": "#284672",
    "selection_text": "#f1f5f9",
    "success": "#4ade80",
    "success_soft": "#132a1d",
    "success_border": "#276b43",
    "warning": "#fbbf24",
    "warning_soft": "#2d2410",
    "warning_border": "#7a5a17",
    "danger": "#f87171",
    "danger_soft": "#33191b",
    "danger_border": "#7f2d2f",
    "info": "#7cb1ff",
    "info_soft": "#15253d",
    "info_border": "#2f5189",
    "chip": "#262d38",
    "chip_text": "#c3ccd8",
    "rail": "#0e1116",
    "rail_border": "#1d232c",
    "rail_hover": "#1f2630",
    "rail_text": "#a9b4c3",
    "rail_active": "#3b82f6",
    "rail_active_text": "#ffffff",
    "plot_bg": "#161a20",
    "plot_fg": "#c3ccd8",
    "plot_grid": "#2a313c",
    "scrollbar": "#3a4452",
    "scrollbar_hover": "#546072",
    "tooltip_bg": "#2a313c",
    "tooltip_text": "#f1f5f9",
    "disabled_bg": "#1a1e25",
}

THEMES: dict[str, dict[str, str]] = {"light": LIGHT, "dark": DARK}


def normalized_mode(mode: object) -> str:
    text = str(mode or "").strip().lower()
    return text if text in THEMES else DEFAULT_MODE


def normalized_font_pt(value: object) -> float:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FONT_PT
    low, high = FONT_PT_RANGE
    return max(low, min(high, round(size * 2) / 2))


def font_tokens(font_pt: float) -> dict[str, str]:
    """Type scale derived from the base size the user picked."""
    base = normalized_font_pt(font_pt)
    return {
        "font_pt": f"{base:g}pt",
        "font_small_pt": f"{max(6.5, base - 1):g}pt",
        "font_title_pt": f"{base + 1:g}pt",
        "font_heading_pt": f"{base + 3:g}pt",
        "font_display_pt": f"{base + 7:g}pt",
    }


__all__ = [
    "DARK",
    "DEFAULT_FONT_PT",
    "DEFAULT_MODE",
    "FONT_PT_RANGE",
    "LIGHT",
    "THEMES",
    "font_tokens",
    "normalized_font_pt",
    "normalized_mode",
]
