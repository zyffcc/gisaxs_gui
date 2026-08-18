"""Backward-compatible public symbols for the historical prediction CLI module."""

from Training import prediction_candidates as _candidate_api
from Training import prediction_curve_io as _curve_api
from Training import prediction_preprocessing as _preprocessing_api
from Training import prediction_refinement as _refinement_api
from Training import prediction_scoring as _scoring_api

_COMPATIBILITY_APIS = (
    _candidate_api,
    _curve_api,
    _preprocessing_api,
    _refinement_api,
    _scoring_api,
)
PUBLIC_NAMES = """candidate_refine_setup enforce_size_distribution_constraints load_curve
score_from_metrics apply_q_range downsample_curve drop_log_outliers short_true_runs
fit_metrics robust_log_score sample_candidate cluster_parameter_modes quantile_summary
score_weight unpack_refined_candidate enforce_d_constraints load_model
_validate_model_contract""".split()


def resolve_legacy_symbol(name: str, module_name: str):
    for module in _COMPATIBILITY_APIS:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
