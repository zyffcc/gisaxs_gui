"""Geometry values repeated in the selected solution, candidates table and Apply confirmation."""

from src.gimap.shared.numbers import format_number

from .pixel_coordinates import shown_center


def geometry_text(candidate) -> tuple[str, str, str]:
    x, y = shown_center(candidate)
    return (
        format_number(x, candidate.center_x_sigma_px, value_only=True, digits=7),
        format_number(y, candidate.center_y_sigma_px, value_only=True, digits=7),
        format_number(candidate.distance_mm, candidate.distance_sigma_mm, value_only=True, digits=6),
    )
