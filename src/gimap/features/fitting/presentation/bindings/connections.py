"""Compose focused connections bindings."""

from .signal_connections import SignalConnectionsMixin
from .image_display_options import ImageDisplayOptionsMixin


class ConnectionsMixin(SignalConnectionsMixin, ImageDisplayOptionsMixin):
    """Compatibility composition for focused connections bindings."""

    pass
