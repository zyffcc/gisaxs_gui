"""Scientific Commands primitives for fitting presentation."""

from __future__ import annotations


from src.gimap.features.fitting.application import (
    ComputeInSituCut,
    FittingAiCalculations,
    FittingCurveCalculations,
    FittingCutCalculations,
    FittingImageCalculations,
    ManualRefinementCalculations,
)


from src.gimap.features.fitting.presentation.scientific_view_model import (
    FittingScientificViewModel,
)


from src.gimap.features.fitting.application import (
    ComputeInSituCut,
    FittingAiCalculations,
    FittingCurveCalculations,
    FittingCutCalculations,
    FittingImageCalculations,
    ManualRefinementCalculations,
)


from src.gimap.features.fitting.presentation.scientific_view_model import (
    FittingScientificViewModel,
)


GISAXS_IMAGE_COLORMAPS = (
    "viridis",
    "cividis",
    "plasma",
    "magma",
    "inferno",
    "turbo",
    "jet",
    "coolwarm",
    "gray",
)

MATPLOTLIB_AVAILABLE = None


def _create_default_fitting_view_model():
    """Build the legacy standalone composition only when no dependency was injected."""

    from src.gimap.app.bootstrap import create_standalone_legacy_context
    from src.gimap.features.fitting.bootstrap import create_fitting_view_model

    return create_fitting_view_model(create_standalone_legacy_context())


def _scientific_commands(owner):
    """Resolve injected commands, with a no-singleton legacy test fallback."""

    view_model = getattr(owner, "fitting_view_model", None)
    commands = getattr(view_model, "science", None)
    if commands is not None:
        return commands
    return FittingScientificViewModel(
        image=FittingImageCalculations(),
        cut=FittingCutCalculations(),
        curve=FittingCurveCalculations(),
        ai=FittingAiCalculations(),
        refinement=ManualRefinementCalculations(),
        insitu_cut=ComputeInSituCut(),
    )


def _ai_catalog(owner):
    catalog = owner.fitting_view_model.storage.ai_catalog
    if catalog is None:
        raise RuntimeError("AI fitting catalog is unavailable")
    return catalog


COMPONENT_FORMULA_TOOLTIPS = {
    "None": ("Component: None\n\nNo component is used."),
    "Sphere": (
        "Component: Sphere\n\n"
        "Formula:\n"
        "F(q,R) = 3 * [sin(qR) - qR cos(qR)] / (qR)^3\n"
        "I(q) = Int * <F(q,R)^2> * S(q; D, sigma_D)\n\n"
        "Parameters:\n"
        "R = radius in nm\n"
        "sigma_R = radius distribution width\n"
        "D = structure spacing in nm\n"
        "sigma_D = structure disorder"
    ),
    "Cylinder": (
        "Component: Cylinder\n\n"
        "Formula:\n"
        "F(q,R,h,alpha) = [2 J1(qR sin(alpha)) / (qR sin(alpha))] "
        "* sinc(qh cos(alpha)/2)\n"
        "I(q) = Int * <F(q,R,h,alpha)^2>_{R,h,alpha} * S(q; D, sigma_D)\n\n"
        "This is the existing isotropic/random-orientation cylinder."
    ),
    "Vertical Cylinder": (
        "Component: Vertical Cylinder\n\n"
        "Formula from gisaxs_fit_v3.1_4structures.py:\n"
        "I(q) = Int * <(R * J1(qR) / q)^2>_R * S(q; D, sigma_D)\n\n"
        "Parameters:\n"
        "R = cylinder radius in nm\n"
        "sigma_R = fractional radius distribution width\n"
        "D = structure spacing in nm\n"
        "sigma_D = structure disorder"
    ),
}

COMPONENT_PARAMETER_SCHEMAS = {
    "Sphere": [
        ("intensity", "Int", "Intensity", 1.0, 6, 0.1),
        ("radius", "R", "Radius (nm)", 10.0, 3, 0.1),
        ("sigma_radius", "SigmaR", "sigma Radius", 0.1, 4, 0.01),
        ("diameter", "D", "D spacing (nm)", 20.0, 3, 0.1),
        ("sigma_diameter", "SigmaD", "sigma D", 0.1, 4, 0.01),
    ],
    "Cylinder": [
        ("intensity", "Int", "Intensity", 1.0, 6, 0.1),
        ("radius", "R", "Radius (nm)", 10.0, 3, 0.1),
        ("sigma_radius", "SigmaR", "sigma Radius", 0.1, 4, 0.01),
        ("height", "h", "Height (nm)", 20.0, 3, 0.1),
        ("sigma_height", "Sigmah", "sigma Height", 0.1, 4, 0.01),
        ("diameter", "D", "D spacing (nm)", 20.0, 3, 0.1),
        ("sigma_diameter", "SigmaD", "sigma D", 0.1, 4, 0.01),
    ],
    "Vertical Cylinder": [
        ("intensity", "Int", "Intensity", 1.0, 6, 0.1),
        ("radius", "R", "Radius (nm)", 10.0, 3, 0.1),
        ("sigma_radius", "SigmaR", "sigma Radius / R", 0.3, 4, 0.01),
        ("diameter", "D", "D spacing (nm)", 20.0, 3, 0.1),
        ("sigma_diameter", "SigmaD", "sigma D", 0.1, 4, 0.01),
    ],
}

COMPONENT_ORDER = ("None", "Sphere", "Cylinder", "Vertical Cylinder")


def is_matplotlib_available():
    """Lazy-check matplotlib availability and cache the result.

    Returns True if matplotlib can be imported, else False. This avoids importing it at
    process start; the first call may pay the cost, which is acceptable when needed.
    """
    global MATPLOTLIB_AVAILABLE
    if MATPLOTLIB_AVAILABLE is None:
        try:
            # Import minimal submodules needed later; do not configure backend here
            import matplotlib  # noqa: F401

            MATPLOTLIB_AVAILABLE = True
        except Exception:
            MATPLOTLIB_AVAILABLE = False
    return MATPLOTLIB_AVAILABLE
