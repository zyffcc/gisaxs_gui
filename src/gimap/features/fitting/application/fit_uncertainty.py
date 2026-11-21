"""How well the values of a manual fit (Refine, Search) are determined: the correlation matrix from the covariance,
the strongly correlated pairs, the values the curve does not determine, the χ² profile of one parameter, and links
and constraints between parameters (the API the page uses; the work is in the domain).

* ``correlation``: the free parameters and their correlation matrix r_ij = C_ij / √(C_ii C_jj) (NaN for a direction
  the curve does not see);
* ``strong_pairs``: |r| above ``SHOWN_CORRELATION`` (0.9);
* ``uncertainty_record``: all of it for the JSON record of a save;
* ``chi2_profile``: ``domain.fit_profile`` (Δχ² = 1 in units of the best fit's χ²ᵣ).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..domain import beam_tail, fit_links
from ..domain.fit_engine import FitData
from ..domain.fit_model import FAMILIES, INFO, FitModel, INTERFERENCES
from ..domain.fit_profile import POINTS, Profile, grid, profile

SHOWN_CORRELATION = 0.9
"""Pairs listed as strongly correlated in the uncertainty view and the report (the fit's own warning uses 0.95)."""

Expression = fit_links.Expression
Inequality = fit_links.Inequality
parameter_name = fit_links.parameter_name
describe_links = fit_links.describe
link_names = fit_links.names
linked = fit_links.linked
prune_links = fit_links.prune
renumbered = fit_links.renumbered
with_inequalities = fit_links.with_inequalities
undetermined = beam_tail.undetermined
tail_flags = beam_tail.tail_flags
flagged = beam_tail.flagged


def parameter_label(model: FitModel, path: tuple) -> str:
    """“1·Sphere R (nm)”, “Background”: a parameter as the report names it."""
    owner, key = path
    info = INFO[key]
    unit = f" ({info.unit})" if info.unit else ""
    if owner == "globals":
        return info.label + unit
    return f"{owner + 1}·{FAMILIES[model.components[owner].family][0]} {info.label}{unit}"


def correlation(result) -> tuple[tuple, np.ndarray]:
    """(paths, r): the correlation matrix of the free parameters of ``result`` (empty without a covariance)."""
    paths = tuple(getattr(result, "covariance_paths", ()) or ())
    covariance = np.asarray(getattr(result, "covariance", ()) or (), float)
    if not paths or covariance.shape != (len(paths), len(paths)):
        return (), np.zeros((0, 0))
    diagonal = np.diag(covariance)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.sqrt(np.where(diagonal > 0, diagonal, np.nan))
        matrix = covariance / np.outer(scale, scale)
    np.fill_diagonal(matrix, np.where(np.isfinite(scale), 1.0, np.nan))
    return paths, np.clip(matrix, -1.0, 1.0)


def strong_pairs(result, threshold: float = SHOWN_CORRELATION) -> list[tuple[tuple, tuple, float]]:
    """Pairs of free parameters with |r| > ``threshold``, strongest first."""
    paths, matrix = correlation(result)
    pairs = [(paths[i], paths[j], float(matrix[i, j])) for i in range(len(paths)) for j in range(i + 1, len(paths))
             if math.isfinite(matrix[i, j]) and abs(matrix[i, j]) > threshold]
    return sorted(pairs, key=lambda pair: -abs(pair[2]))


def _name(path) -> str:
    return parameter_name(path)


def uncertainty_record(result) -> dict:
    """For the ``fit`` block of the JSON record: the correlation matrix with its parameter order, the pairs above
    ``SHOWN_CORRELATION``, the values the curve does not determine and the beam tail's flags."""
    paths, matrix = correlation(result)
    return {
        "correlation": {"parameters": [_name(path) for path in paths],
                        "matrix": [[None if not math.isfinite(v) else round(float(v), 6) for v in row] for row in matrix]},
        "covariance": [[None if not math.isfinite(v) else float(v) for v in row]
                       for row in np.asarray(getattr(result, "covariance", ()) or (), float).reshape(len(paths), -1)]
        if paths else [],
        f"correlated_above_{SHOWN_CORRELATION:g}": [[_name(a), _name(b), round(r, 6)] for a, b, r in strong_pairs(result)],
        "undetermined": [_name(path) for path in undetermined(result)],
        "beam_tail_flags": [[_name(path), why] for path, why in tail_flags(result)],
    }


def profile_values(model: FitModel, path: tuple, error: Optional[float], points: int = POINTS) -> np.ndarray:
    """The default grid of a profile (``fit_profile.grid``): ±3 errors around the value, within its range."""
    return grid(model, path, error, points)


def chi2_profile(model: FitModel, data: FitData, path: tuple, values, *, progress=None, stop=None) -> Profile:
    """The χ² profile of ``path`` (``domain.fit_profile.profile``)."""
    return profile(model, data, path, values, progress=progress, stop=stop)


def interference_label(component) -> str:
    return INTERFERENCES[component.interference][0] if component.structure else "none"


__all__ = ["SHOWN_CORRELATION", "Expression", "Inequality", "Profile", "chi2_profile", "correlation", "describe_links",
           "flagged", "interference_label", "link_names", "linked", "parameter_label", "parameter_name",
           "profile_values", "prune_links", "renumbered", "strong_pairs", "tail_flags", "uncertainty_record",
           "undetermined", "with_inequalities"]
