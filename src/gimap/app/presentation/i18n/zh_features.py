"""Chinese tables of the features added after release 0.2.0, merged into ``ZH`` by ``zh.py`` in one line.

A new feature adds its table here, not in ``zh.py`` (which is near the size limit of a runtime module)."""

from .zh_app_session import APP_SESSION_ZH
from .zh_fit_data import FIT_DATA_ZH
from .zh_flat_field import FLAT_FIELD_ZH
from .zh_geometry import GEOMETRY_ZH
from .zh_geometry_cuts import ZH_GEOMETRY_CUTS
from .zh_geometry_orientation import GEOMETRY_ORIENTATION_ZH
from .zh_geometry_tilt import ZH_GEOMETRY_TILT
from .zh_gisaxs_cuts import GISAXS_CUTS_ZH
from .zh_indexing import INDEXING_ZH
from .zh_insitu import INSITU_ZH
from .zh_joint_fit import JOINT_FIT_ZH
from .zh_normalisation import NORMALISATION_ZH
from .zh_nxcansas import NXCANSAS_ZH
from .zh_roll_check import ROLL_CHECK_ZH

FEATURES_ZH = {
    **FLAT_FIELD_ZH, **NXCANSAS_ZH, **GISAXS_CUTS_ZH, **FIT_DATA_ZH, **NORMALISATION_ZH, **GEOMETRY_ZH,
    **APP_SESSION_ZH, **INDEXING_ZH, **INSITU_ZH, **JOINT_FIT_ZH, **ROLL_CHECK_ZH,
    **ZH_GEOMETRY_CUTS, **GEOMETRY_ORIENTATION_ZH, **ZH_GEOMETRY_TILT,
}

__all__ = ["FEATURES_ZH"]
