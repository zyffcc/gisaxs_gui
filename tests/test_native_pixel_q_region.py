"""Native cut q must describe exactly the pixels contributing intensity."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.features.fitting.application.scientific import FittingCutCalculations
from src.gimap.features.fitting.domain.cbf_observations import column_observations
from src.gimap.features.fitting.domain.cut_math import (
    extract_native_pixel_profile, pixel_region_bounds, sample_q_mesh_line,
)
from src.gimap.features.fitting.domain.detector_image import DetectorPreprocessing, prepare_detector_image
from src.gimap.features.fitting.domain.insitu_cut import compute_insitu_cut
from src.gimap.features.fitting.domain.models import CutSelection
from src.gimap.features.fitting.presentation.bindings.cut_display import CutDisplayMixin
from src.gimap.features.fitting.presentation.bindings.cut_extraction import CutExtractionMixin
from src.gimap.features.fitting.presentation.bindings.detector_configuration import DetectorConfigurationMixin
from src.gimap.features.fitting.presentation.bindings.insitu_cut_processing import InsituCutProcessingMixin
from src.gimap.features.fitting.presentation.bindings.workflow_feedback import WorkflowFeedbackMixin
from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import WorkflowV5BindingMixin


def _fixture():
    raw = np.arange(160, dtype=float).reshape(8, 20) + 1
    raw[1, 3] = raw[2, 4] = raw[1:3, 5] = -1
    raw[1:3, 6] = 0
    state = prepare_detector_image(raw, DetectorPreprocessing(mask_negative_pixels=True), revision=1)
    row, col = np.indices(raw.shape)
    q = col*.1 + row*.01
    qz = row*.2 + col*.01
    selection = CutSelection(9.5, 5.5, 1., 19., "horizontal")
    return state, q, qz, selection


def _payload(state, q, qz, selection, **options):
    return dict(image_data=state.analysis_image, center_x=selection.center_x,
                center_y=selection.center_y, vertical=selection.height, parallel=selection.width,
                cut_type=selection.orientation, show_q_axis=False, preserve_native=True,
                native_pixel_q=True, qy_mesh=q, qz_mesh=qz, n_points=500,
                analysis_revision=1, **options)


def _unexpected(*_args, **_kwargs):
    raise AssertionError("Native q must not pass through the legacy midpoint conversion")


class _Owner(CutExtractionMixin, CutDisplayMixin, InsituCutProcessingMixin,
             WorkflowV5BindingMixin, DetectorConfigurationMixin, WorkflowFeedbackMixin):
    pass


def _owner(state, q, qz, selection):
    owner = _Owner()
    owner.current_detector_image = state
    owner.current_parameters = {"imported_gisaxs_file": "synthetic.cbf"}
    owner.current_cut_data = {"analysis_revision": 1}
    owner.cut = {}
    owner.fitting_view_model = SimpleNamespace(science=SimpleNamespace(cut=FittingCutCalculations()),
                                              state=SimpleNamespace(cut_result_analysis_revision=1))
    owner._get_cached_q_meshgrids = lambda: (q, qz)
    owner._detector_q_grid = lambda: SimpleNamespace(horizontal=lambda _axis: q, qz=qz)
    owner._should_show_q_axis = lambda: False
    owner._horizontal_q_axis = lambda: "qy"
    owner._horizontal_q_label = lambda: r"$q_y$ (nm$^{-1}$)"
    owner._get_cut_center_coordinates = lambda: (selection.center_x, selection.center_y)
    owner._sort_filter_cut_pairs = lambda x, y, **_: (np.asarray(x), np.asarray(y), None)
    owner._filter_cut_pairs_for_active_axis = lambda x, y, **_: (x, y)
    owner._convert_pixel_to_qy = owner._convert_pixel_to_qz = _unexpected
    owner._log_cut_debug = lambda _: None
    owner.status_updated = SimpleNamespace(emit=lambda _: None)
    owner._suppress_workflow_plot_updates = True
    owner.ui = SimpleNamespace(**{key: SimpleNamespace(value=lambda value=value: value) for key, value in (
        ("gisaxsInputCenterParallelValue", selection.center_x),
        ("gisaxsInputCenterVerticalValue", selection.center_y),
        ("gisaxsInputCutLineVerticalValue", selection.height),
        ("gisaxsInputCutLineParallelValue", selection.width))})
    return owner


def test_native_q_uses_identical_valid_pixels_including_true_zero_counts():
    state, q, _, selection = _fixture()
    y, native_q, columns = extract_native_pixel_profile(state.analysis_image, q, selection)
    expected_columns = np.delete(np.arange(20), 5)
    np.testing.assert_array_equal(columns, expected_columns)
    expected = expected_columns*.1 + .015
    expected[expected_columns == 3] = .32
    expected[expected_columns == 4] = .41
    np.testing.assert_allclose(native_q, expected, atol=1e-15, rtol=0)
    x0, x1, r0, r1 = pixel_region_bounds(q.shape, selection)
    region = (r0, r1, x0, x1)
    before = column_observations(state.analysis_image, np.broadcast_to(q[4:5], q.shape), region)
    after = column_observations(state.analysis_image, q, region)
    np.testing.assert_array_equal(after[0], native_q)
    np.testing.assert_array_equal(after[1], y)
    for index in (1, 2):
        np.testing.assert_array_equal(before[index], after[index])
    assert before[3] == after[3]
    assert after[3]["valid_pixel_counts"][list(columns).index(6)] == 2
    assert y[columns == 6] == 0
    assert not np.array_equal(before[0], after[0])


@pytest.mark.parametrize("axis", ["qy", "qr"])
def test_single_batch_and_prediction_share_region_mean_q(axis):
    state, q, qz, selection = _fixture()
    if axis == "qr":
        q = np.sqrt(q*q + .01)
    owner = _owner(state, q, qz, selection)
    owner._horizontal_q_axis = lambda: axis
    owner._perform_cut_operation(selection.height, selection.width, "horizontal")
    single_q = owner.current_cut_data["x_coords"].copy()
    single_i = owner.current_cut_data["y_intensity"].copy()
    assert owner.current_cut_data["q_source"] == "region_mean_native"
    assert owner.cut["meta"]["q_source"] == "region_mean_native"
    batch = compute_insitu_cut(_payload(state, q, qz, selection, horizontal_q_axis=axis))
    assert batch["source"] == "q" and batch["method"] == "native_masked"
    assert batch["q_source"] == "region_mean_native"
    np.testing.assert_array_equal(batch["x_coords"], single_q)
    np.testing.assert_array_equal(batch["y_intensity"], single_i)
    predicted = owner._native_cbf_input(dict(relative_noise=0., absolute_noise=0.))
    np.testing.assert_array_equal(predicted["x_coords"], single_q)
    np.testing.assert_array_equal(predicted["y_intensity"], single_i)
    assert owner._workflow_observation_metadata["q_source"] == "region_mean_native"
    assert owner._workflow_observation_metadata["valid_pixel_counts"] == [2, 2, 2, 1, 1] + [2]*14


def test_batch_result_is_not_converted_from_q_back_through_pixel_mapping():
    state, q, qz, selection = _fixture()
    result = compute_insitu_cut(_payload(state, q, qz, selection))
    owner = _owner(state, q, qz, selection)
    owner._insitu_workflow_state = "Processing"
    owner._apply_deleted_point_mask_to_current_cut = lambda: 0
    owner._append_insitu_heatmap_cut = lambda *_: None
    owner._log_insitu_workflow = lambda *_: None
    finished = []
    owner._finalize_insitu_workflow_file = lambda **kwargs: finished.append(kwargs)
    owner._on_insitu_cut_failed = _unexpected
    record = {}
    owner._on_insitu_cut_finished(result, record, "synthetic.cbf", {}, False)
    np.testing.assert_array_equal(owner.current_cut_data["x_coords"], result["x_coords"])
    assert owner.current_cut_data["q_source"] == owner.cut["meta"]["q_source"] == "region_mean_native"
    assert record["cut_status"] == "ok" and len(finished) == 1


def test_batch_worker_receives_actual_meshes_for_native_pixel_mode(monkeypatch):
    from src.gimap.features.fitting.presentation.bindings import insitu_cut_processing

    state, q, qz, selection = _fixture()
    owner = _owner(state, q, qz, selection)
    owner._validate_current_cut_settings = lambda: (True, "")
    owner._insitu_cut_geometry = lambda: dict(center_parallel_px=selection.center_x,
                                            center_vertical_px=selection.center_y,
                                            cut_vertical_px=selection.height,
                                            cut_parallel_px=selection.width)
    owner._interp_method_default = "Linear"
    owner._resolve_cut_points = lambda _: 500
    owner._log_insitu_workflow = lambda *_: None
    owner._refresh_insitu_workflow_status = lambda: None
    owner._finalize_insitu_workflow_file = _unexpected
    owner.fitting_view_model.science.insitu_cut = compute_insitu_cut
    captured = []

    class Worker:
        def __init__(self, payload, _command):
            captured.append(payload)
            self.cut_finished = self.error_occurred = self.finished = SimpleNamespace(connect=lambda _: None)

        def start(self):
            pass

    monkeypatch.setattr(insitu_cut_processing, "InsituCutWorker", Worker)
    record = {}
    owner._start_insitu_cut_worker(state.analysis_image, "synthetic.cbf", record, {}, False)
    assert record["cut_status"] == "cutting"
    assert captured[0]["native_pixel_q"] and captured[0]["preserve_native"]
    assert captured[0]["qy_mesh"] is q and captured[0]["qz_mesh"] is qz
    assert captured[0]["show_q_axis"] is False
    assert captured[0]["center_x"] == selection.center_x


def test_vertical_native_cut_uses_actual_valid_columns_for_qz():
    state, q, qz, _ = _fixture()
    selection = CutSelection(3.5, 5., 4., 1., "vertical")
    y, actual, indices = extract_native_pixel_profile(state.analysis_image, qz, selection)
    for value, intensity, row in zip(actual, y, indices):
        cols = np.array([3, 4])
        cols = cols[np.isfinite(state.analysis_image[row, cols])]
        assert value == np.mean(qz[row, cols])
        assert intensity == np.mean(state.analysis_image[row, cols])
    batch = compute_insitu_cut(_payload(state, q, qz, selection))
    order = np.argsort(actual)
    np.testing.assert_array_equal(batch["x_coords"], actual[order])


def test_legacy_interpolation_ignores_native_mapping_flag_and_mesh():
    state, q, qz, selection = _fixture()
    payload = _payload(state, q, qz, selection)
    payload.update(preserve_native=False, n_points=10)
    actual = compute_insitu_cut(payload)
    expected = compute_insitu_cut({**payload, "native_pixel_q": False, "qy_mesh": None, "qz_mesh": None})
    assert actual["source"] == "pixel" and actual["q_source"] is None
    np.testing.assert_array_equal(actual["x_coords"], expected["x_coords"])
    np.testing.assert_array_equal(actual["y_intensity"], expected["y_intensity"])
    # The existing explicit fractional-pixel midpoint mapper is unchanged.
    np.testing.assert_allclose(sample_q_mesh_line(q, [1.25, 3.5], orientation="horizontal", image_shape=q.shape), [.165, .39])


def test_geometry_change_refreshes_native_cut_at_the_same_pixel_region():
    state, q, qz, selection = _fixture()
    owner = _owner(state, q, qz, selection)
    owner._perform_cut_operation(selection.height, selection.width, "horizontal")
    old_q = owner.current_cut_data["x_coords"].copy()
    old_i = owner.current_cut_data["y_intensity"].copy()
    owner.current_parameter_selection = dict(bounds=dict(x_min=0, x_max=19, y_min=5, y_max=6))
    owner._update_cutline_labels_units = owner._update_cutline_step_sizes = lambda: None
    changed = q + np.arange(8)[:, None]*.003
    owner._compute_q_meshgrids_and_store = lambda: setattr(owner, "_get_cached_q_meshgrids", lambda: (changed, qz))
    owner._seed_independent_q_cache = owner._refresh_image_display = lambda: None
    regions = []
    owner._apply_pixel_region_to_active_coordinates = lambda region, **options: regions.append((region, options))
    owner._complete_fitting_step = lambda *_: None
    owner._fail_fitting_step = _unexpected
    reveals = []

    def perform_cut(*, reveal_result):
        reveals.append(reveal_result)
        owner._perform_cut_operation(selection.height, selection.width, "horizontal")

    owner._perform_cut = perform_cut
    owner._on_detector_parameters_changed()
    assert regions == [((1, 2, 0, 19), {"q_mode": False, "horizontal_axis": "qy"})]
    assert reveals == [False]
    assert not np.array_equal(old_q, owner.current_cut_data["x_coords"])
    np.testing.assert_array_equal(old_i, owner.current_cut_data["y_intensity"])
    expected = extract_native_pixel_profile(state.analysis_image, changed, selection)[1]
    np.testing.assert_array_equal(owner.current_cut_data["x_coords"], expected)


def test_requested_native_q_fails_closed_without_matching_mesh():
    state, q, qz, selection = _fixture()
    payload = _payload(state, q, qz, selection)
    with pytest.raises(ValueError, match="same 2D shape"):
        compute_insitu_cut({**payload, "qy_mesh": None})
