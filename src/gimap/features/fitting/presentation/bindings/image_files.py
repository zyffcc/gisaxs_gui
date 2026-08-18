"""Compose focused image files bindings."""

from .scattering_files import ScatteringFilesMixin
from .image_display_loading import ImageDisplayLoadingMixin
from .curve_files import CurveFilesMixin


class ImageFilesMixin(ScatteringFilesMixin, ImageDisplayLoadingMixin, CurveFilesMixin):
    """Compatibility composition for focused image files bindings."""

    pass
