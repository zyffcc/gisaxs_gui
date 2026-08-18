"""Compose focused manual fitting bindings."""

from .manual_fit_execution import ManualFitExecutionMixin
from .fitting_result_display import FittingResultDisplayMixin


class ManualFittingMixin(ManualFitExecutionMixin, FittingResultDisplayMixin):
    """Compatibility composition for focused manual fitting bindings."""

    pass
