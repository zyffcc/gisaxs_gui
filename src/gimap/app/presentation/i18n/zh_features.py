"""Chinese tables of the features added after release 0.2.0, merged into ``ZH`` by ``zh.py`` in one line.

A new feature adds its table here, not in ``zh.py`` (which is near the size limit of a runtime module)."""

from .zh_absolute_scale import ABSOLUTE_SCALE_ZH
from .zh_app_session import APP_SESSION_ZH
from .zh_assistant_service import ASSISTANT_SERVICE_ZH
from .zh_calibration_residuals import CALIBRATION_RESIDUALS_ZH
from .zh_calibration_rings import CALIBRATION_RINGS_ZH
from .zh_calibration_uncertainty import CALIBRATION_UNCERTAINTY_ZH
from .zh_cell_refinement import CELL_REFINEMENT_ZH
from .zh_cell_series import CELL_SERIES_ZH
from .zh_cell_orientation import CELL_ORIENTATION_ZH
from .zh_coverage_report import COVERAGE_REPORT_ZH
from .zh_crystal_orientation import CRYSTAL_ORIENTATION_ZH
from .zh_crystal_film import STRINGS as CRYSTAL_FILM_ZH
from .zh_refracted_film import REFRACTED_FILM_ZH
from .zh_fit_data import FIT_DATA_ZH
from .zh_flat_field import FLAT_FIELD_ZH
from .zh_geometry import GEOMETRY_ZH
from .zh_geometry_cuts import ZH_GEOMETRY_CUTS
from .zh_geometry_orientation import GEOMETRY_ORIENTATION_ZH
from .zh_geometry_tilt import ZH_GEOMETRY_TILT
from .zh_gisaxs_cuts import GISAXS_CUTS_ZH
from .zh_gisaxs_fit_level import GISAXS_FIT_LEVEL_ZH
from .zh_gisaxs_posterior import GISAXS_POSTERIOR_ZH
from .zh_horizon_selection import HORIZON_SELECTION_ZH
from .zh_indexing import INDEXING_ZH
from .zh_insitu import INSITU_ZH
from .zh_joint_fit import JOINT_FIT_ZH
from .zh_joint_height import JOINT_HEIGHT_ZH
from .zh_manual_fit import MANUAL_FIT_ZH
from .zh_mask_tools import MASK_TOOLS_ZH
from .zh_normalisation import NORMALISATION_ZH
from .zh_nxcansas import NXCANSAS_ZH
from .zh_orientation_reduction import ORIENTATION_REDUCTION_ZH
from .zh_orientation_report import ORIENTATION_REPORT_ZH
from .zh_pole_reference import POLE_REFERENCE_ZH
from .zh_posterior_mcmc import POSTERIOR_MCMC_ZH
from .zh_predict_report import PREDICT_REPORT_ZH
from .zh_predict_sampling import PREDICT_SAMPLING_ZH
from .zh_reduction_errors import REDUCTION_ERRORS_ZH
from .zh_reproduction import REPRODUCTION_ZH
from .zh_review_phases import REVIEW_PHASES_ZH
from .zh_ring_assignment import RING_ASSIGNMENT_ZH
from .zh_roll_check import ROLL_CHECK_ZH
from .zh_round2_analyze import ROUND2_ANALYZE_ZH
from .zh_round2_joint import ROUND2_JOINT_ZH
from .zh_round2_resume import ROUND2_RESUME_ZH
from .zh_round2_series import ROUND2_SERIES_ZH
from .zh_sample_roll import SAMPLE_ROLL_ZH
from .zh_frame_incidence import FRAME_INCIDENCE_ZH
from .zh_series_chain import SERIES_CHAIN_ZH
from .zh_series_selection import SERIES_SELECTION_ZH
from .zh_texture_populations import TEXTURE_POPULATIONS_ZH
from .zh_usability_analyze import USABILITY_ANALYZE_ZH
from .zh_usability_calibration import USABILITY_CALIBRATION_ZH
from .zh_usability_gisaxs import USABILITY_GISAXS_ZH
from .zh_usability_giwaxs import USABILITY_GIWAXS_ZH
from .zh_vertical_consistency import VERTICAL_CONSISTENCY_ZH

FEATURES_ZH = {
    **CELL_REFINEMENT_ZH, **CELL_ORIENTATION_ZH, **CELL_SERIES_ZH,
    **SAMPLE_ROLL_ZH,
    **FRAME_INCIDENCE_ZH,
    **ROUND2_ANALYZE_ZH,
    **ROUND2_JOINT_ZH,
    **ROUND2_RESUME_ZH,
    **ROUND2_SERIES_ZH,
    **USABILITY_ANALYZE_ZH,
    **USABILITY_CALIBRATION_ZH,
    **USABILITY_GISAXS_ZH,
    **USABILITY_GIWAXS_ZH,
    **PREDICT_SAMPLING_ZH,
    **VERTICAL_CONSISTENCY_ZH,
    **GISAXS_FIT_LEVEL_ZH,
    **ASSISTANT_SERVICE_ZH,
    **CALIBRATION_RESIDUALS_ZH,
    **CALIBRATION_RINGS_ZH,
    **CALIBRATION_UNCERTAINTY_ZH,
    **PREDICT_REPORT_ZH, **POSTERIOR_MCMC_ZH, **GISAXS_POSTERIOR_ZH, **FLAT_FIELD_ZH, **NXCANSAS_ZH,
    **GISAXS_CUTS_ZH, **FIT_DATA_ZH, **NORMALISATION_ZH, **GEOMETRY_ZH, **APP_SESSION_ZH, **INDEXING_ZH,
    **INSITU_ZH, **JOINT_FIT_ZH, **ABSOLUTE_SCALE_ZH, **MASK_TOOLS_ZH, **MANUAL_FIT_ZH, **ROLL_CHECK_ZH,
    **ZH_GEOMETRY_CUTS, **GEOMETRY_ORIENTATION_ZH, **ZH_GEOMETRY_TILT, **SERIES_CHAIN_ZH, **REVIEW_PHASES_ZH,
    **RING_ASSIGNMENT_ZH, **ORIENTATION_REDUCTION_ZH, **ORIENTATION_REPORT_ZH, **SERIES_SELECTION_ZH,
    **REDUCTION_ERRORS_ZH, **HORIZON_SELECTION_ZH, **REPRODUCTION_ZH, **JOINT_HEIGHT_ZH,
    **COVERAGE_REPORT_ZH, **POLE_REFERENCE_ZH, **CRYSTAL_ORIENTATION_ZH, **TEXTURE_POPULATIONS_ZH,
    **CRYSTAL_FILM_ZH, **REFRACTED_FILM_ZH,
}

__all__ = ["FEATURES_ZH"]
