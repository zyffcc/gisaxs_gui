"""Chinese tables of the features added after release 0.2.0, merged into ``ZH`` by ``zh.py`` in one line.

A new feature adds its table here, not in ``zh.py`` (which is near the size limit of a runtime module)."""

from .zh_fit_data import FIT_DATA_ZH
from .zh_flat_field import FLAT_FIELD_ZH
from .zh_gisaxs_cuts import GISAXS_CUTS_ZH
from .zh_normalisation import NORMALISATION_ZH
from .zh_nxcansas import NXCANSAS_ZH

FEATURES_ZH = {**FLAT_FIELD_ZH, **NXCANSAS_ZH, **GISAXS_CUTS_ZH, **FIT_DATA_ZH, **NORMALISATION_ZH}

__all__ = ["FEATURES_ZH"]
