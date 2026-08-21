from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.gimap.features.fitting.presentation.bindings import detector_display
from src.gimap.features.fitting.presentation.bindings.detector_display import (
    DetectorDisplayMixin,
)
from src.gimap.features.fitting.presentation.curve_rendering import (
    NEGATIVE_Q_COLOR,
    POSITIVE_Q_COLOR,
    CurvePlotSpec,
    experimental_curve_series,
    render_curve_plot,
)


class _CurveDisplayHarness(DetectorDisplayMixin):
    data_source = "cut"
    I_fitting = np.array([900.0, 100.0, 800.0, 200.0])

    def _current_curve_view_state(self, *, sync_window=False):
        return SimpleNamespace(
            normalize=False,
            layer_mode="compare",
            q_mode="fold",
            log_y=True,
        )

    def _get_roi_active_prepared_arrays(self):
        return (
            np.array([1.0, 1.0, 2.0, 2.0]),
            np.array([10.0, 12.0, 20.0, 22.0]),
            np.array([-1, 1, -1, 1], dtype=np.int8),
        )

    def _filter_ai_excluded_points_for_display(self, q, intensity, source_sign):
        return q, intensity, source_sign

    def _convert_q_values_for_display(self, q):
        return np.asarray(q, dtype=float)

    def _convert_q_values_for_model(self, q, *, source):
        return np.asarray(q, dtype=float)

    def _get_independent_axis_filter_mode(self):
        return "all"

    def _build_q_axis_label(self, *, filter_mode):
        return "|q| (nm$^{-1}$)"

    def _get_x_axis_scale(self):
        return "log"

    def _roi_active(self):
        return False

    def _get_checkbox_state(self, _name, default):
        return default

    def _get_particle_sequence_flags(self):
        return {1: False}

    def _get_last_fitting_spec_and_params(self):
        return ["sphere"], [1.0]


class _FakeModelCalculations:
    def components(self, _shapes, q_model, _parameters):
        q = np.asarray(q_model, dtype=float)
        particle = 3.0 * q
        background = np.full_like(q, 2.0)
        resolution = 0.5 * q
        return {
            "particles": [{"shape": "sphere", "index": 1, "I": particle}],
            "BG_total": background,
            "resolution": resolution,
            "total": background + particle + resolution,
        }


def test_overlay_curve_builds_distinct_positive_and_negative_layers():
    series = experimental_curve_series(
        [1.0, 1.0, 2.0, 2.0],
        [10.0, 12.0, 20.0, 22.0],
        source_sign=[-1, 1, -1, 1],
        q_mode="fold",
        label="Cut data",
    )

    assert [item.color for item in series] == [POSITIVE_Q_COLOR, NEGATIVE_Q_COLOR]
    assert [item.label for item in series] == [
        "Cut data · +q",
        "Cut data · −q mirrored",
    ]
    np.testing.assert_allclose(series[0].y, [12.0, 22.0])
    np.testing.assert_allclose(series[1].y, [10.0, 20.0])


def test_embedded_and_independent_axes_can_render_the_same_spec():
    from matplotlib.figure import Figure

    series = experimental_curve_series(
        [1.0, 1.0, 2.0, 2.0],
        [10.0, 12.0, 20.0, 22.0],
        source_sign=[-1, 1, -1, 1],
        q_mode="fold",
        label="Cut data",
    )
    spec = CurvePlotSpec(
        series=series,
        x_label="|q| (nm$^{-1}$)",
        y_label="Intensity",
        title="Experimental Curve",
        x_scale="log",
    )
    embedded = Figure().add_subplot(111)
    independent = Figure().add_subplot(111)

    render_curve_plot(embedded, spec)
    render_curve_plot(independent, spec)

    assert [collection.get_label() for collection in embedded.collections] == [
        collection.get_label() for collection in independent.collections
    ]
    assert embedded.get_xscale() == independent.get_xscale() == "log"


def test_fitting_total_is_evaluated_on_the_same_prepared_q_as_the_plot(monkeypatch):
    monkeypatch.setattr(
        detector_display,
        "_scientific_commands",
        lambda _binding: SimpleNamespace(model=_FakeModelCalculations()),
    )
    harness = _CurveDisplayHarness()

    spec = harness._build_curve_plot_spec("fitting")

    assert spec is not None
    model_series = next(item for item in spec.series if item.role == "model")
    np.testing.assert_allclose(model_series.x, [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(model_series.y, [5.5, 5.5, 9.0, 9.0])
    assert not np.array_equal(model_series.y, harness.I_fitting)
