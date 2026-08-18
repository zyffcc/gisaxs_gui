"""Compose focused selection display bindings."""

from .selection_interactions import SelectionInteractionsMixin
from .selection_preview import SelectionPreviewMixin
from .color_display_controls import ColorDisplayControlsMixin


class SelectionDisplayMixin(
    SelectionInteractionsMixin, SelectionPreviewMixin, ColorDisplayControlsMixin
):
    """Compatibility composition for focused selection display bindings."""

    pass
