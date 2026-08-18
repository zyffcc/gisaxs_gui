"""Application-facing commands for Fitting domain calculations."""

from __future__ import annotations

from ..domain import (
    ai_q_key,
    apply_input_image_options,
    chi_square,
    compute_insitu_cut,
    default_refine_bounds,
    default_refine_selected,
    extract_pixel_profile,
    extract_q_profile,
    filter_axis,
    filter_for_display,
    finite_log_profiles,
    interpolate_series,
    mirror_fill_detector_gaps,
    normalize_geometry,
    normalize_intensity,
    optimize_scale_factor,
    prepare_ai_curve,
    q_values_for_display,
    q_values_for_model,
    run_manual_refinement,
    sample_q_mesh_line,
    sort_filter_pairs,
    valid_y_values_for_limits,
)
from .ports import FittingModelPort, QSpacePort


class FittingImageCalculations:
    def transform(self, image, **options):
        return apply_input_image_options(image, **options)

    def mirror_gaps(self, image, **options):
        return mirror_fill_detector_gaps(image, **options)

    def center_profiles(self, image):
        return finite_log_profiles(image)


class FittingCutCalculations:
    def extract_pixel(self, image, selection):
        return extract_pixel_profile(image, selection)

    def extract_q(self, image, qy_mesh, qz_mesh, selection):
        return extract_q_profile(image, qy_mesh, qz_mesh, selection)

    def sample_mesh_line(self, mesh, pixel_coords, **options):
        return sample_q_mesh_line(mesh, pixel_coords, **options)

    def sort_filter(self, x_values, intensity, **options):
        return sort_filter_pairs(x_values, intensity, **options)

    def filter_axis(self, q_values, intensity, mode="all", **options):
        return filter_axis(q_values, intensity, mode, **options)

    def interpolate(self, x, y, x_new, method):
        return interpolate_series(x, y, x_new, method)


class FittingCurveCalculations:
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
    def default_selected(self, parameters):
        return default_refine_selected(parameters)

    def default_bounds(self, item):
        return default_refine_bounds(item)

    def execute(self, setup, selected, options, **callbacks):
        return run_manual_refinement(
            setup,
            selected,
            options,
            **callbacks,
        )


class ComputeInSituCut:
    def execute(self, payload: dict) -> dict:
        return compute_insitu_cut(dict(payload))


class FittingModelCalculations:
    def __init__(self, model: FittingModelPort):
        self._model = model

    def parameter_names(self, shapes):
        return self._model.parameter_names(tuple(shapes))

    def components(self, shapes, q_model, parameters):
        return self._model.components(
            tuple(shapes), q_model, tuple(parameters)
        )

    def build_function(self, shapes):
        return self._model.build_function(tuple(shapes))


class FittingQSpaceCalculations:
    def __init__(self, q_space: QSpacePort):
        self._q_space = q_space

    def create_detector(self, **geometry):
        return self._q_space.create_detector(**geometry)

    def axis_labels_and_extent(self, detector):
        return self._q_space.axis_labels_and_extent(detector)
