"""Chinese tables of the features added after release 0.2.0, merged into ``ZH`` by ``zh.py`` in one line.

A new feature adds its table here, not in ``zh.py`` (which is near the size limit of a runtime module)."""

from .zh_absolute_scale import ABSOLUTE_SCALE_ZH
from .zh_app_session import APP_SESSION_ZH
from .zh_assistant_service import ASSISTANT_SERVICE_ZH
from .zh_calibration_residuals import CALIBRATION_RESIDUALS_ZH
from .zh_calibration_uncertainty import CALIBRATION_UNCERTAINTY_ZH
from .zh_fit_data import FIT_DATA_ZH
from .zh_flat_field import FLAT_FIELD_ZH
from .zh_geometry import GEOMETRY_ZH
from .zh_geometry_cuts import ZH_GEOMETRY_CUTS
from .zh_geometry_orientation import GEOMETRY_ORIENTATION_ZH
from .zh_geometry_tilt import ZH_GEOMETRY_TILT
from .zh_gisaxs_cuts import GISAXS_CUTS_ZH
from .zh_gisaxs_posterior import GISAXS_POSTERIOR_ZH
from .zh_horizon_selection import HORIZON_SELECTION_ZH
from .zh_indexing import INDEXING_ZH
from .zh_insitu import INSITU_ZH
from .zh_joint_fit import JOINT_FIT_ZH
from .zh_manual_fit import MANUAL_FIT_ZH
from .zh_mask_tools import MASK_TOOLS_ZH
from .zh_normalisation import NORMALISATION_ZH
from .zh_nxcansas import NXCANSAS_ZH
from .zh_orientation_reduction import ORIENTATION_REDUCTION_ZH
from .zh_orientation_report import ORIENTATION_REPORT_ZH
from .zh_posterior_mcmc import POSTERIOR_MCMC_ZH
from .zh_predict_report import PREDICT_REPORT_ZH
from .zh_reduction_errors import REDUCTION_ERRORS_ZH
from .zh_reproduction import REPRODUCTION_ZH
from .zh_review_phases import REVIEW_PHASES_ZH
from .zh_ring_assignment import RING_ASSIGNMENT_ZH
from .zh_roll_check import ROLL_CHECK_ZH
from .zh_series_chain import SERIES_CHAIN_ZH
from .zh_series_selection import SERIES_SELECTION_ZH

FEATURES_ZH = {
    **ASSISTANT_SERVICE_ZH,
    **CALIBRATION_RESIDUALS_ZH,
    **CALIBRATION_UNCERTAINTY_ZH,
    **PREDICT_REPORT_ZH, **POSTERIOR_MCMC_ZH, **GISAXS_POSTERIOR_ZH, **FLAT_FIELD_ZH, **NXCANSAS_ZH,
    **GISAXS_CUTS_ZH, **FIT_DATA_ZH, **NORMALISATION_ZH, **GEOMETRY_ZH, **APP_SESSION_ZH, **INDEXING_ZH,
    **INSITU_ZH, **JOINT_FIT_ZH, **ABSOLUTE_SCALE_ZH, **MASK_TOOLS_ZH, **MANUAL_FIT_ZH, **ROLL_CHECK_ZH,
    **ZH_GEOMETRY_CUTS, **GEOMETRY_ORIENTATION_ZH, **ZH_GEOMETRY_TILT, **SERIES_CHAIN_ZH, **REVIEW_PHASES_ZH,
    **RING_ASSIGNMENT_ZH, **ORIENTATION_REDUCTION_ZH, **ORIENTATION_REPORT_ZH, **SERIES_SELECTION_ZH,
    **REDUCTION_ERRORS_ZH, **HORIZON_SELECTION_ZH, **REPRODUCTION_ZH,
}

__all__ = ["FEATURES_ZH"]
