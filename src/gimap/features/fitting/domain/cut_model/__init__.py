"""GIMaP's one forward model of a 1D GISAXS cut (Physics and Methods, chapter 9).

Used by Fitting, the proposal network's training simulator (``tools/ml1d``) and refinement. Pure numpy and scipy.
"""

from .geometry import MODES, Geometry, from_angles
from .model import (
    COUPLINGS,
    Population,
    Settings,
    couple,
    evaluate,
    population_term,
    population_terms,
    size_averages,
    size_moments,
    tail,
)
from .shapes import FAMILIES
from .sizes import LAWS

__all__ = ["COUPLINGS", "FAMILIES", "LAWS", "MODES", "Geometry", "Population", "Settings", "couple", "evaluate",
           "from_angles", "population_term", "population_terms", "size_averages", "size_moments", "tail"]
