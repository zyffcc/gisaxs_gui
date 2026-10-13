"""Application-facing commands for Fitting domain calculations."""

from __future__ import annotations

from ..domain import (
    default_global_search_bounds,
    default_global_search_selected,
    ai_q_key,
    chi_square,
    default_refine_bounds,
    default_refine_selected,
    filter_axis,
    filter_for_display,
    interpolate_series,
    normalize_geometry,
    normalize_intensity,
    optimize_scale_factor,
    prepare_ai_curve,
    prepare_signed_q_curve,
    q_values_for_display,
    q_values_for_model,
    run_manual_refinement,
    sort_filter_pairs,
    valid_y_values_for_limits,
)
from .ports import FittingModelPort


class FittingCurveCalculations:
    def prepare_signed(self, q_values, intensity, **options):
        return prepare_signed_q_curve(q_values, intensity, **options)

    def filter_for_display(self, q_values, intensity=None, mode="all"):
        return filter_for_display(q_values, intensity, mode)

    def q_for_model(self, q_values, source_unit):
        return q_values_for_model(q_values, source_unit)

    def q_for_display(self, q_values, source_unit, display_unit):
        return q_values_for_display(q_values, source_unit, display_unit)

    def valid_y_for_limits(self, y_values, log_y=False):
        return valid_y_values_for_limits(y_values, log_y)

    def normalize_intensity(self, intensity):
        return normalize_intensity(intensity)

    def sort_filter(self, x_values, intensity, **options):
        return sort_filter_pairs(x_values, intensity, **options)

    def filter_axis(self, q_values, intensity, mode="all", **options):
        return filter_axis(q_values, intensity, mode, **options)

    def interpolate(self, x, y, x_new, method):
        return interpolate_series(x, y, x_new, method)


class FittingAiCalculations:
    def prepare_curve(self, *args, **kwargs):
        return prepare_ai_curve(*args, **kwargs)

    def q_key(self, value) -> str:
        return ai_q_key(value)

    def normalize_geometry(self, value: str) -> str:
        return normalize_geometry(value)

    def chi_square(self, observed, predicted) -> float:
        return chi_square(observed, predicted)

    def optimize_scale(self, observed, fitted, current_scale):
        return optimize_scale_factor(observed, fitted, current_scale)


class ManualRefinementCalculations:
    def default_selected(self, parameter_name):
        return default_refine_selected(parameter_name)

    def default_bounds(self, parameter_name, current_value):
        return default_refine_bounds(parameter_name, current_value)

    def default_global_selected(self, parameter_name):
        return default_global_search_selected(parameter_name)

    def default_global_bounds(
        self,
        parameter_name,
        current_value,
        observed=None,
        q_values=None,
    ):
        return default_global_search_bounds(
            parameter_name,
            current_value,
            observed,
            q_values,
        )

    def execute(self, setup, selected, options, **callbacks):
        return run_manual_refinement(
            setup,
            selected,
            options,
            **callbacks,
        )


class FittingModelCalculations:
    def __init__(self, model: FittingModelPort):
        self._model = model

    def parameter_names(self, shapes):
        return self._model.parameter_names(tuple(shapes))

    def components(self, shapes, q_model, parameters):
        return self._model.components(tuple(shapes), q_model, tuple(parameters))

    def build_function(self, shapes):
        return self._model.build_function(tuple(shapes))


__all__ = [
    "FittingAiCalculations",
    "FittingCurveCalculations",
    "FittingModelCalculations",
    "ManualRefinementCalculations",
]
