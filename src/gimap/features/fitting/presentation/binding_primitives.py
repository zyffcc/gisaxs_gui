"""Stable exports for fitting presentation primitives.

Concrete implementations live in focused modules; this module preserves the
public import surface used by existing bindings and compatibility callers.
"""

from .scientific_commands import (
    GISAXS_IMAGE_COLORMAPS,
    MATPLOTLIB_AVAILABLE,
    _create_default_fitting_view_model,
    _scientific_commands,
    _ai_catalog,
    COMPONENT_FORMULA_TOOLTIPS,
    COMPONENT_PARAMETER_SCHEMAS,
    COMPONENT_ORDER,
    is_matplotlib_available,
)

from .refinement_workers import (
    ManualAutoRefineWorker,
    RefineUiBridge,
)

from .insitu_workers import (
    InsituBatchImageLoader,
    InsituCutWorker,
)

from .independent_image_window import (
    IndependentMatplotlibWindow,
)

from .independent_fit_window import (
    IndependentFitWindow,
)

from .display_manager import (
    UnifiedDisplayManager,
    _qobject_is_alive,
)

from .image_loading_workers import (
    FolderImageScanWorker,
    AsyncImageLoader,
)

__all__ = [
    "GISAXS_IMAGE_COLORMAPS",
    "MATPLOTLIB_AVAILABLE",
    "_create_default_fitting_view_model",
    "_scientific_commands",
    "_ai_catalog",
    "COMPONENT_FORMULA_TOOLTIPS",
    "COMPONENT_PARAMETER_SCHEMAS",
    "COMPONENT_ORDER",
    "is_matplotlib_available",
    "ManualAutoRefineWorker",
    "RefineUiBridge",
    "InsituBatchImageLoader",
    "InsituCutWorker",
    "IndependentMatplotlibWindow",
    "IndependentFitWindow",
    "UnifiedDisplayManager",
    "_qobject_is_alive",
    "FolderImageScanWorker",
    "AsyncImageLoader",
]
