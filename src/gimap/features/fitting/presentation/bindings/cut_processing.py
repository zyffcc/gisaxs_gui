"""Compose focused cut processing bindings."""

from .cut_extraction import CutExtractionMixin
from .cut_display import CutDisplayMixin


class CutProcessingMixin(CutExtractionMixin, CutDisplayMixin):
    """Compatibility composition for focused cut processing bindings."""

    pass
