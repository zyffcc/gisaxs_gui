"""Compose focused roi controls bindings."""

from .roi_range_controls import RoiRangeControlsMixin
from .roi_curve_processing import RoiCurveProcessingMixin


class RoiControlsMixin(RoiRangeControlsMixin, RoiCurveProcessingMixin):
    """Compatibility composition for focused roi controls bindings."""

    pass
