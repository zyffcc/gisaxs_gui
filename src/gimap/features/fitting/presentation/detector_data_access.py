"""Presentation access to the single current detector analysis image."""

from __future__ import annotations


def analysis_image_for(owner):
    """Return the explicit analysis image, with one legacy alias fallback."""

    state = getattr(owner, "current_detector_image", None)
    if state is not None:
        return state.analysis_image
    analysis = getattr(owner, "current_analysis_image", None)
    if analysis is not None:
        return analysis
    return getattr(owner, "current_stack_data", None)


def analysis_revision_for(owner) -> int | None:
    state = getattr(owner, "current_detector_image", None)
    return None if state is None else int(state.revision)


__all__ = ["analysis_image_for", "analysis_revision_for"]
