from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cut_fitting_stack_test_module",
    ROOT / "controllers" / "fitting_controller.py",
)
FITTING_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FITTING_MODULE)


class _StackEdit:
    def __init__(self, text: str):
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


class CutFittingStackTests(unittest.TestCase):
    def test_standalone_nxs_uses_file_navigation_even_when_dataset_is_3d(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"image_{index:03d}.nxs" for index in (1, 2)]
            for path in paths:
                with h5py.File(path, "w") as handle:
                    handle.create_dataset(
                        "/entry/instrument/detector/data",
                        data=np.zeros((3, 4, 6), dtype=np.float32),
                    )

            selected = []
            controller = SimpleNamespace(
                current_parameters={"imported_gisaxs_file": str(paths[0])},
                fitting_view_model=SimpleNamespace(
                    storage=SimpleNamespace(
                        inspect_scattering_sequence=lambda _path: SimpleNamespace(
                            frame_count=3
                        )
                    )
                ),
                _nxs_frame_index=0,
                _nxs_frame_count=1,
                _folder_image_files=[str(path) for path in paths],
                _folder_image_index=0,
                _nxs_uses_internal_frames=lambda _path: False,
                _update_folder_navigation_buttons=lambda: None,
                _scan_folder_images_for_file=lambda _path: None,
                _select_folder_image=lambda path, frame_index=0: selected.append((path, frame_index)),
                status_updated=SimpleNamespace(emit=lambda _message: None),
            )
            FITTING_MODULE.FittingController._set_nxs_frame_state(
                controller,
                str(paths[0]),
                frame_index=2,
            )
            self.assertEqual(controller._nxs_frame_count, 3)
            self.assertEqual(controller._nxs_frame_index, 2)

            FITTING_MODULE.FittingController._show_folder_image_at_offset(controller, 1)
            self.assertEqual(selected, [(str(paths[1]), 0)])

    def test_mosaic_nxs_next_uses_the_next_internal_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"scan_m0{module}.nxs" for module in (1, 2)]
            for path in paths:
                with h5py.File(path, "w") as handle:
                    handle.create_dataset(
                        "/entry/instrument/detector/data",
                        data=np.zeros((3, 4, 6), dtype=np.float32),
                    )

            shown = []
            selected = []
            controller = SimpleNamespace(
                current_parameters={"imported_gisaxs_file": str(paths[0])},
                _nxs_frame_index=0,
                _nxs_frame_count=3,
                _folder_image_files=[str(paths[0])],
                _folder_image_index=0,
                _nxs_uses_internal_frames=lambda _path: True,
                _update_stack_display=lambda: None,
                _update_folder_navigation_buttons=lambda: None,
                _select_folder_image=lambda path, frame_index=0: selected.append((path, frame_index)),
                _show_image=lambda: shown.append(True),
                parameters_changed=SimpleNamespace(emit=lambda _parameters: None),
                status_updated=SimpleNamespace(emit=lambda _message: None),
            )
            FITTING_MODULE.FittingController._show_folder_image_at_offset(controller, 1)
            self.assertEqual(controller._nxs_frame_index, 1)
            self.assertEqual(controller.current_parameters["nxs_frame_index"], 1)
            self.assertEqual(shown, [True])
            self.assertEqual(selected, [])

    def test_standalone_nxs_stack_starts_at_current_frame_and_clamps_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.nxs"
            source = np.stack([
                np.full((4, 6), value, dtype=np.float32)
                for value in (1, 2, 3, 4)
            ])
            with h5py.File(path, "w") as handle:
                handle.create_dataset("/entry/instrument/detector/data", data=source)

            loader = FITTING_MODULE.AsyncImageLoader()
            summed = loader._load_multiple_nxs_frames(str(path), frame_index=1, stack_count=99)
            np.testing.assert_array_equal(summed, np.full((6, 4), 9, dtype=np.float32))

    def test_stack_count_updates_input_when_requested_value_is_too_large(self):
        edit = _StackEdit("99")
        controller = SimpleNamespace(
            load_mode="Stack",
            current_parameters={"stack_count": 99},
            ui=SimpleNamespace(gisaxsInputStackValue=edit),
            _maximum_stack_count=lambda: 3,
        )
        clamped = FITTING_MODULE.FittingController._clamp_stack_count(
            controller,
            requested=99,
            notify=False,
        )
        self.assertEqual(clamped, 3)
        self.assertEqual(controller.current_parameters["stack_count"], 3)
        self.assertEqual(edit.text(), "3")

    def test_ordinary_stack_sequence_counts_from_selected_file(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("frame1.cbf", "frame2.cbf", "frame10.cbf")]
            for path in paths:
                path.touch()
            controller = SimpleNamespace(
                _folder_image_scan_cache={},
                _folder_image_cache_key=lambda _path: folder,
            )
            controller._natural_sort_key = (
                FITTING_MODULE.FittingController._natural_sort_key.__get__(controller)
            )
            sequence = FITTING_MODULE.FittingController._ordinary_stack_sequence(
                controller,
                str(paths[1]),
            )
            self.assertEqual([Path(path).name for path in sequence], ["frame2.cbf", "frame10.cbf"])

    def test_cbf_loader_clamps_oversized_stack_to_available_files(self):
        try:
            from fabio.cbfimage import CbfImage
        except ImportError:
            self.skipTest("fabio CBF writer is unavailable")
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"frame{index}.cbf" for index in (1, 2, 3)]
            for value, path in enumerate(paths, start=1):
                CbfImage(data=np.full((8, 10), value, dtype=np.int32)).write(str(path))
            loader = FITTING_MODULE.AsyncImageLoader()
            summed = loader._load_multiple_cbf_files(str(paths[1]), stack_count=99)
            np.testing.assert_array_equal(summed, np.full((8, 10), 5, dtype=np.float32))

    def test_preview_fit_resets_transform_before_fitting_new_image(self):
        calls: list[str] = []

        class Scene:
            def setSceneRect(self, _rect):
                calls.append("scene")

        class View:
            def scene(self):
                return Scene()

            def resetTransform(self):
                calls.append("reset")

            def fitInView(self, _item, _mode=None):
                calls.append("fit")

            def update(self):
                calls.append("update")

        class Item:
            def sceneBoundingRect(self):
                return object()

        FITTING_MODULE.FittingController._fit_view_to_item(
            SimpleNamespace(),
            View(),
            Item(),
            keep_aspect=True,
        )
        self.assertEqual(calls, ["scene", "reset", "fit", "update"])


class CutFittingImageInputTests(unittest.TestCase):
    def test_threshold_mask_replaces_nan_and_out_of_range_values(self):
        source = np.array([[np.nan, -2.0, 1.0], [3.0, 7.0, np.inf]], dtype=np.float32)
        masked = FITTING_MODULE.apply_threshold_mask(source, True, lower=0.0, upper=5.0)
        np.testing.assert_array_equal(np.isfinite(masked), [[False, False, True], [True, False, False]])
        self.assertEqual(masked[0, 2], 1.0)
        self.assertEqual(masked[1, 0], 3.0)

    def test_masked_pixels_have_zero_weight_in_integration_mean(self):
        source = np.array([[1.0, np.nan, 3.0], [5.0, 7.0, np.nan]])
        horizontal = FITTING_MODULE.finite_mean_axis(source, axis=0)
        vertical = FITTING_MODULE.finite_mean_axis(source, axis=1)
        np.testing.assert_allclose(horizontal, [3.0, 7.0, 3.0])
        np.testing.assert_allclose(vertical, [2.0, 6.0])

    def test_masked_pixels_have_zero_weight_in_center_finding_profiles(self):
        source = np.array([[10.0, np.nan], [100.0, np.nan]], dtype=np.float32)
        vertical, horizontal = FITTING_MODULE.finite_log_profiles(source)
        np.testing.assert_allclose(vertical, [1.0, 2.0])
        np.testing.assert_allclose(horizontal, [3.0, 0.0])

    def test_flip_ud_is_applied_to_input_without_a_second_display_flip(self):
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        transformed = FITTING_MODULE.apply_input_image_options(input_data, flip_ud=True)
        np.testing.assert_array_equal(transformed, np.flipud(input_data))
        controller = SimpleNamespace(
            current_stack_data=transformed,
            current_raw_image=input_data,
            _mirror_fill_detector_gaps=False,
            _flip_ud=True,
            _last_mirror_fill_count=0,
            _last_mirror_fill_status="",
        )
        displayed = FITTING_MODULE.FittingController._get_current_display_image(controller)
        np.testing.assert_array_equal(displayed, transformed)
        np.testing.assert_array_equal(controller.current_stack_data, np.flipud(input_data))

    def test_controller_rebuilds_current_input_when_flip_changes(self):
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        controller = SimpleNamespace(
            current_raw_image=input_data,
            current_stack_data=input_data,
            data=input_data,
            summed_data=None,
            current_parameters={"stack_count": 1},
            _flip_ud=True,
            _threshold_mask_enabled=False,
            _threshold_mask_min=-1e12,
            _threshold_mask_max=1e12,
            _image_display_cache={},
        )
        FITTING_MODULE.FittingController._reapply_input_image_options(
            controller,
            refresh=False,
        )
        np.testing.assert_array_equal(controller.current_stack_data, np.flipud(input_data))
        np.testing.assert_array_equal(controller.data, np.flipud(input_data))


class CutFittingSamplingTests(unittest.TestCase):
    def test_fractional_pixel_positions_map_to_distinct_q_values(self):
        qy = np.tile(np.array([-2.0, -1.0, 0.0, 1.0]), (3, 1))
        qz = np.zeros_like(qy)
        controller = SimpleNamespace(
            current_stack_data=np.zeros((3, 4), dtype=np.float32),
            _get_cached_q_meshgrids=lambda: (qy, qz),
        )
        q = FITTING_MODULE.FittingController._convert_pixel_coords_to_q(
            controller,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            "qy",
        )
        np.testing.assert_allclose(q, [-2.0, -1.75, -1.5, -1.25, -1.0])
        self.assertEqual(len(np.unique(q)), len(q))

    def test_duplicate_q_coordinates_are_merged_before_interpolation(self):
        controller = SimpleNamespace(_log_cut_debug=lambda _message: None)
        q, intensity, _ = FITTING_MODULE.FittingController._sort_filter_cut_pairs(
            controller,
            [1.0, 1.0, 2.0],
            [2.0, 4.0, 8.0],
            context="test cut",
        )
        np.testing.assert_allclose(q, [1.0, 2.0])
        np.testing.assert_allclose(intensity, [3.0, 8.0])

    def test_negative_only_filter_is_applied_before_resampling(self):
        controller = SimpleNamespace(
            _get_independent_axis_filter_mode=lambda: "negative",
            _log_cut_debug=lambda _message: None,
        )
        q, intensity = FITTING_MODULE.FittingController._filter_cut_pairs_for_active_axis(
            controller,
            [-2.0, -1.0, 1.0, 2.0],
            [20.0, 10.0, 11.0, 22.0],
            context="test cut",
        )
        np.testing.assert_allclose(q, [-2.0, -1.0])
        np.testing.assert_allclose(intensity, [20.0, 10.0])


if __name__ == "__main__":
    unittest.main()
