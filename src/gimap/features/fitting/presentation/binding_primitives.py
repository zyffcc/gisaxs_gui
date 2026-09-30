"""Shared primitives of the Fitting presentation bindings."""

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

from .independent_fit_window import (
    IndependentFitWindow,
)


def _qobject_is_alive(obj) -> bool:
    """``False`` for ``None`` and for Qt objects whose C++ side was deleted."""
    if obj is None:
        return False
    try:
        import sip

        if sip.isdeleted(obj):
            return False
    except Exception:
        pass
    try:
        obj.objectName()
    except RuntimeError:
        return False
    except Exception:
        pass
    return True


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
    "IndependentFitWindow",
    "_qobject_is_alive",
]
