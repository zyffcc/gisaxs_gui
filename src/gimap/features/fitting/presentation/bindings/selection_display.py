"""Compose focused selection display bindings."""

from .selection_interactions import SelectionInteractionsMixin
from .selection_preview import SelectionPreviewMixin
from .color_display_controls import ColorDisplayControlsMixin
from .main_preview_tools import MainPreviewToolsMixin


class SelectionDisplayMixin(
    SelectionInteractionsMixin,
    SelectionPreviewMixin,
    ColorDisplayControlsMixin,
    MainPreviewToolsMixin,
):
    """Compatibility composition for focused selection display bindings."""

    pass
