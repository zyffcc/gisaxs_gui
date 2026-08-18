"""Compose focused fit export bindings."""

from .fit_result_export import FitResultExportMixin
from .manual_refine_dialog import ManualRefineDialogMixin
from .manual_refine_setup import ManualRefineSetupMixin


class FitExportMixin(FitResultExportMixin, ManualRefineDialogMixin, ManualRefineSetupMixin):
    """Compatibility composition for focused fit export bindings."""

    pass
