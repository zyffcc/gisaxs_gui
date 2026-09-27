"""Compose focused detector cut setup bindings."""

from .fit_graphics_events import FitGraphicsEventsMixin
from .detector_configuration import DetectorConfigurationMixin
from .detector_display import DetectorDisplayMixin
from .center_symmetry import CenterSymmetryMixin


class DetectorCutSetupMixin(
    CenterSymmetryMixin, FitGraphicsEventsMixin, DetectorConfigurationMixin, DetectorDisplayMixin
):
    """Compatibility composition for focused detector cut setup bindings."""

    pass
