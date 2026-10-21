"""Which dataset of a NeXus / HDF5 file is the detector image (``shared/detector_io/nexus_datasets.py``).

Every file is synthetic (h5py, in ``tmp_path``): legacy files with the numeric ``signal = 1`` mark,
corrected images whose names say what was subtracted, files that hold only a mask or a dark, model
weights, GIMaP's own multi-frame container, and datasets a caller names by their path.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.shared.detector_io import (
    _dataset_candidates,
    detect_nxs_frame_count,
    forget_layout,
    group_sources,
    inspect_nexus,
    load_detector_image,
    open_nexus,
    select_nxs_dataset,
)
from src.gimap.shared.detector_io import nexus_datasets, nexus_layout

DET = "/entry/instrument/detector"
H, W = 64, 96


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _stack(frames: int = 3, base: int = 0, dtype=np.int32) -> np.ndarray:
    return (np.full((frames, H, W), base) + np.arange(frames)[:, None, None]).astype(dtype)


def _one(path: Path, dataset: str, data: np.ndarray | None = None) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=np.full((3, H, W), 2.5, dtype=np.float32) if data is None else data)
    return path


def _paths(path: Path) -> list[str]:
    with open_nexus(path) as handle:
        return [name for _score, name in _dataset_candidates(handle)]


# -- the mark NeXus used before 2014: signal = 1 on the dataset itself -----------------------------------------------------------


@pytest.mark.parametrize("signal", [1, np.int32(1), np.int64(1), np.array([1], dtype=np.int32), "1", b"1", 1.0],
                         ids=["int", "int32", "int64", "array", "text", "bytes", "float"])
def test_the_old_signal_mark_on_the_dataset_names_the_image(tmp_path: Path, signal) -> None:
    path = tmp_path / "legacy.nxs"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/scan")
        group.create_dataset("monitor_image", data=_stack(2, base=900))
        group.create_dataset("counts", data=_stack(3, base=7)).attrs["signal"] = signal
    layout = inspect_nexus(path)
    assert (layout.dataset, layout.frames, layout.warnings) == ("/entry/scan/counts", 3, ())
    assert "signal" in layout.decided_by and layout.alternatives == ("/entry/scan/monitor_image",)
    assert detect_nxs_frame_count(path) == 3
    np.testing.assert_array_equal(load_detector_image(path, frame_idx=1).data, _stack(3, base=7)[1])


@pytest.mark.parametrize("signal", [2, 0, "x", np.array([], dtype=np.int32), np.float32(2.5)],
                         ids=["two", "zero", "word", "empty", "fraction"])
def test_another_value_of_the_signal_attribute_marks_nothing(tmp_path: Path, signal) -> None:
    path = tmp_path / "legacy.nxs"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/scan")
        group.create_dataset("a_image", data=_stack(2))
        group.create_dataset("counts", data=_stack(3)).attrs["signal"] = signal
    layout = inspect_nexus(path)  # two unnamed images: the file does not say which
    assert layout.dataset == "/entry/scan/a_image" and layout.warnings[0][0] == nexus_layout.SEVERAL_DATASETS


def test_the_classic_detector_data_with_a_numeric_signal_and_its_nxdata_link(tmp_path: Path) -> None:
    path = tmp_path / "classic.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_group("entry").attrs["NX_class"] = "NXentry"
        detector = handle.create_group(DET)
        detector.attrs["NX_class"] = "NXdetector"
        data = detector.create_dataset("data", data=_stack(3, base=7))
        data.attrs["signal"] = np.int32(1)
        plot = handle.create_group("entry/data")
        plot.attrs["NX_class"] = "NXdata"
        plot["data"] = data
    layout = inspect_nexus(path)
    assert (layout.dataset, layout.frames, layout.alternatives, layout.warnings) == (f"{DET}/data", 3, (), ())
    assert load_detector_image(path, frame_idx=2).data[0, 0] == 9


@pytest.mark.parametrize("name, value", [
    ("type", np.int32(3)),
    ("description", np.float32(2.5)),
    ("local_name", np.array([4, 5], dtype=np.int64)),
    ("layout", np.array([1556, 516], dtype=np.int32)),
    ("layout", np.int64(7)),
    ("description", np.array([], dtype="S1")),
])
def test_numbers_where_the_detector_is_described_are_no_sign_of_a_lambda_file(tmp_path: Path, name: str, value) -> None:
    path = tmp_path / "numbers.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{DET}/data", data=_stack(3, base=7))
        handle.create_dataset(f"{DET}/{name}", data=value)
    layout = inspect_nexus(path)
    assert (layout.kind, layout.frames, layout.orientation, layout.warnings) == ("stack", 3, "as_stored", ())
    np.testing.assert_array_equal(load_detector_image(path, frame_idx=1).data, _stack(3, base=7)[1])


def test_text_is_read_from_every_form_hdf5_gives_it_in() -> None:
    text = nexus_datasets._text
    assert [text(value) for value in (b"data", "data", np.bytes_(b"data"), np.str_("data"), np.array([b"data"]),
                                      np.array(["data", "x"], dtype=object))] == ["data"] * 6
    assert [text(value) for value in (1, np.int64(1), np.array([1]), np.array(1), np.float32(2.5))] == [
        "1", "1", "1", "1", "2.5"]
    assert [text(value) for value in (None, np.array([]), h5py.Empty("S1"))] == ["", "", ""]


# -- names: what was done to an image is not what a correction is called ---------------------------------------------------------


@pytest.mark.parametrize("name", [
    "/entry/data/dark_subtracted", "/entry/dark_subtracted/data", "/entry/data/normalized",
    "/entry/data/normalized_image", "/entry/data/data_normalised", "/entry/data/image_dark_corrected",
    "/entry/data/flatfield_corrected", "/entry/instrument/detector/dark_corrected", "/entry/data/error_corrected",
    "/entry/process/sigma_clipped", "/entry/result/weighted_sum", "/entry/normalized/data",
    "/entry/data/background_subtracted", "/entry/data/masked_image", "/entry/data/unmasked",
    "/dark_run_0003/instrument/detector/data", "/entry/data/dark_current_removed", "/entry/data/sigma_filtered",
])
def test_a_corrected_image_is_an_image(tmp_path: Path, name: str) -> None:
    path = _one(tmp_path / "processed.nxs", name)
    layout = inspect_nexus(path)
    assert (layout.kind, layout.dataset, layout.frames, layout.warnings) == ("stack", name, 3, ())
    assert group_sources([path]) == [path.resolve()]
    assert load_detector_image(path, frame_idx=1).data.shape == (H, W) and _paths(path) == [name]
    with open_nexus(path) as handle:
        assert select_nxs_dataset(handle) == name


@pytest.mark.parametrize("name", [
    "/entry/data/pixel_mask", "/entry/data/errors", "/entry/weights/data", "/entry/darkfield/image",
    "/entry/dark/data", "/flatfield", "/entry/data/flat_field", "/entry/data/variance", "/entry/data/normalisation",
    "/entry/data/dark_averaged", "/entry/data/mask_combined", "/entry/data/errors_propagated",
])
def test_a_file_that_holds_only_a_correction_dataset_is_read_and_says_what_it_is(tmp_path: Path, name: str) -> None:
    path = _one(tmp_path / "only.h5", name)
    layout = inspect_nexus(path)
    assert (layout.kind, layout.dataset, layout.frames) == ("stack", name, 3)
    assert layout.warnings == ((nexus_layout.ONLY_AUXILIARY, {"dataset": name}),)
    assert layout.warning_texts() == (
        f"{name} is named like a mask or a correction, not like a detector image: "
        "it is read because this file holds no other image",)
    assert group_sources([path]) == [path.resolve()] and _paths(path) == [name]
    image = load_detector_image(path, frame_idx=2)
    assert float(image.data[0, 0]) == 2.5 and image.metadata["layout_warnings"] == list(layout.warning_texts())


def test_corrections_never_win_over_an_image_and_are_offered_last(tmp_path: Path) -> None:
    path = tmp_path / "reduced.nxs"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/result")
        group.create_dataset("pixel_mask", data=np.ones((H, W), dtype=np.uint8))
        group.create_dataset("dark/data", data=np.ones((H, W), dtype=np.float32))
        group.create_dataset("a_errors", data=np.ones((4, H, W), dtype=np.float32))
        group.create_dataset("pil300k", data=_stack(4))
    layout = inspect_nexus(path)
    assert (layout.dataset, layout.alternatives, layout.warnings) == ("/entry/result/pil300k", (), ())
    assert _paths(path)[0] == "/entry/result/pil300k" and sorted(_paths(path)[1:]) == [
        "/entry/result/a_errors", "/entry/result/dark/data", "/entry/result/pixel_mask"]
    with open_nexus(path) as handle:
        assert select_nxs_dataset(handle) == "/entry/result/pil300k"  # no question: the others are corrections
        scores = [score for score, _name in _dataset_candidates(handle)]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[1] == scores[-1]
    # A caller may still ask for one of them, and then it is read without a remark.
    mask = inspect_nexus(path, dataset="/entry/result/pixel_mask")
    assert (mask.dataset, mask.frames, mask.warnings) == ("/entry/result/pixel_mask", 1, ())


# -- files that are not detector data ---------------------------------------------------------------------------------------------


def _keras_model(path: Path) -> Path:
    """``model.save("x.h5")`` of Keras 2: the configuration in root attributes, the kernels under model_weights."""
    with h5py.File(path, "w") as handle:
        handle.attrs["keras_version"], handle.attrs["backend"], handle.attrs["model_config"] = "2.11.0", "tensorflow", "{}"
        handle.create_dataset("model_weights/conv2d/conv2d/kernel:0", data=np.zeros((3, 3, 32, 64), dtype=np.float32))
        handle.create_dataset("model_weights/dense/dense/kernel:0", data=np.zeros((1024, 256), dtype=np.float32))
        handle.create_dataset("optimizer_weights/Adam/dense/kernel/m:0", data=np.zeros((1024, 256), dtype=np.float32))
    return path


def _keras_weights(path: Path) -> Path:
    """``model.save_weights("x.h5")`` of Keras 2: one group per layer, variables named ``kernel:0``."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dense/dense/kernel:0", data=np.zeros((1024, 256), dtype=np.float32))
        handle.create_dataset("dense/dense/bias:0", data=np.zeros(256, dtype=np.float32))
    return path


def _keras3_weights(path: Path) -> Path:
    """``model.save_weights("x.weights.h5")`` of Keras 3: ``layers/<name>/vars/<n>``."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("layers/dense/vars/0", data=np.zeros((1024, 256), dtype=np.float32))
        handle.create_dataset("layers/dense/vars/1", data=np.zeros(256, dtype=np.float32))
        handle.create_dataset("optimizer/vars/3", data=np.zeros((1024, 256), dtype=np.float32))
        handle.create_group("vars")
    return path


def _only_groups(path: Path) -> Path:
    """A model file whose marks are its two top-level groups (attributes stripped by a copy tool)."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("model_weights/dense/kernel", data=np.zeros((256, 128), dtype=np.float32))
        handle.create_group("optimizer_weights")
    return path


@pytest.mark.parametrize("build", [_keras_model, _keras_weights, _keras3_weights, _only_groups])
def test_model_weights_are_not_detector_data(tmp_path: Path, build) -> None:
    model = build(tmp_path / "best_model.h5")
    frame = _one(tmp_path / "frame.h5", "/entry/data/data", np.zeros((H, W), dtype=np.float32))
    layout = inspect_nexus(model)
    assert (layout.kind, layout.frames, layout.dataset, layout.alternatives) == ("no_image", 0, "", ())
    assert "model" in layout.decided_by and layout.summary_text() == "no detector image"
    assert group_sources([model, frame]) == [frame.resolve()]
    assert group_sources([model], keep_without_image=True) == [model.resolve()]
    assert _paths(model) == []
    with pytest.raises(ValueError, match="best_model.h5 holds no detector image"):
        load_detector_image(model)
    with open_nexus(model) as handle:
        with pytest.raises(ValueError, match="No usable detector image dataset"):
            select_nxs_dataset(handle)


def test_a_kernel_asked_for_by_its_path_is_still_read(tmp_path: Path) -> None:
    model = _keras_weights(tmp_path / "weights.h5")
    image = load_detector_image(model, dataset_path="/dense/dense/kernel:0")
    assert image.data.shape == (1024, 256) and image.metadata["dataset_path"] == "/dense/dense/kernel:0"


def test_one_dataset_named_like_a_variable_does_not_make_a_detector_file_a_model(tmp_path: Path) -> None:
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/measurement/roi:0", data=np.zeros((H, W), dtype=np.float32))
        handle.create_dataset("/entry/measurement/pilatus", data=_stack(4))
    layout = inspect_nexus(path)
    assert (layout.kind, layout.dataset, layout.frames, layout.alternatives) == ("stack", "/entry/measurement/pilatus", 4, ())


def test_tables_of_results_are_not_images(tmp_path: Path) -> None:
    """What a GIMaP training job writes next to its model: one row per sample, one column per parameter."""
    path = tmp_path / "predictions.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("predictions_normalized", data=np.zeros((500, 16)))
        handle.create_dataset("targets_normalized", data=np.zeros((500, 16)))
        handle.create_dataset("history", data=np.zeros((4000, 24)))
    assert inspect_nexus(path).kind == "no_image" and group_sources([path]) == []
    # The same numbers where the file says "this is the detector" are read: the place decides, not the shape.
    strip = _one(tmp_path / "strip.nxs", f"{DET}/data", np.zeros((500, 16), dtype=np.int32))
    assert (inspect_nexus(strip).kind, inspect_nexus(strip).frame_shape) == ("single_frame", (500, 16))


# -- one frame per dataset: GIMaP's own container ---------------------------------------------------------------------------------


def _container(path: Path, frames: dict[int, np.ndarray]) -> Path:
    """Format Converter's "one HDF5 file" output, written by its own writer."""
    from src.gimap.features.format_converter.domain.models import ConversionOptions
    from src.gimap.features.format_converter.infrastructure.adapters.local_files import LocalConversionExecutor

    options = ConversionOptions(output_format="HDF5", destination=str(path.parent), container=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["NX_class"] = "NXroot"
        handle.create_group("entry").attrs["NX_class"] = "NXentry"
        for index, frame in frames.items():
            LocalConversionExecutor._write_container_frame(handle, frame, {"source_file": "x.cbf"}, index, options)
    return path


def test_the_format_converters_container_reads_back_every_frame(tmp_path: Path) -> None:
    written = {index: np.full((40, 60), 10.0 * index, dtype=np.float32) for index in (1, 2, 4, 5, 12)}  # 3 failed
    path = _container(tmp_path / "converted_images.h5", written)
    layout = inspect_nexus(path)
    assert (layout.kind, layout.frames, layout.frame_shape, layout.warnings) == ("stack", 5, (40, 60), ())
    assert layout.dataset == "/entry/data/image_000001" and layout.alternatives == () and layout.complete
    assert layout.chunks == tuple((f"/entry/data/image_{index:06d}", 1) for index in written)
    assert layout.summary_text() == "1 file · 5 frames" and "image_000001" in layout.decided_by
    assert detect_nxs_frame_count(path) == 5 and group_sources([path]) == [path.resolve()]
    for position, (index, frame) in enumerate(written.items()):
        image = load_detector_image(path, frame_idx=position)
        np.testing.assert_array_equal(image.data, frame)
        assert image.metadata["dataset_path"] == f"/entry/data/image_{index:06d}"
        assert (image.metadata["frame_index"], image.metadata["frame_count"]) == (position, 5)
    with pytest.raises(IndexError, match="frame 6 of 5"):
        load_detector_image(path, frame_idx=5)
    with open_nexus(path) as handle:
        assert select_nxs_dataset(handle) == "/entry/data/image_000001"  # one series: nothing to choose between
        offered = _dataset_candidates(handle)
    # A chooser offers the series first, then each of its other datasets on its own.
    assert offered == [(20, "/entry/data/image_000001")] + [(-5, f"/entry/data/image_{index:06d}") for index in (2, 4, 5, 12)]
    alone = load_detector_image(path, dataset_path="/entry/data/image_000004")
    np.testing.assert_array_equal(alone.data, written[4])
    assert alone.metadata["frame_count"] == 1


def test_stacks_stored_one_after_the_other_are_one_series(tmp_path: Path) -> None:
    path = tmp_path / "parts.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/data/image_000001", data=_stack(2, base=0))
        handle.create_dataset("/entry/data/image_000002", data=_stack(3, base=100))
        handle.create_dataset("/entry/data/image_000003", data=np.full((2, H + 2, W), 7, dtype=np.int32))  # another detector
        handle.create_dataset("/entry/data/image_000004", data=np.full((H + 2, W), 8, dtype=np.int32))
    layout = inspect_nexus(path)
    assert (layout.frames, layout.member_frames, layout.frame_shape) == (5, (2, 3), (H, W))
    assert [int(load_detector_image(path, frame_idx=index).data[0, 0]) for index in range(5)] == [0, 1, 100, 101, 102]
    # Where the frames change size another series begins: it is named, not dropped, and can be asked for.
    assert layout.alternatives == ("/entry/data/image_000003",)
    assert layout.warnings == ((nexus_layout.SEVERAL_DATASETS, {
        "count": 2, "dataset": "/entry/data/image_000001", "others": "/entry/data/image_000003"}),)
    other = inspect_nexus(path, dataset="/entry/data/image_000003")
    assert (other.frames, other.member_frames, other.frame_shape, other.warnings) == (3, (2, 1), (H + 2, W), ())
    assert [int(load_detector_image(path, dataset_path="/entry/data/image_000003", frame_idx=index).data[0, 0])
            for index in range(3)] == [7, 7, 8]


# -- a dataset the caller names -------------------------------------------------------------------------------------------------


def _eiger_like(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{DET}/data", data=np.full((4, H, W), 100, dtype=np.uint32))
        handle.create_dataset(f"{DET}/detectorSpecific/flatfield", data=np.full((H, W), 0.5, dtype=np.float32))
        mask = np.zeros((H, W), dtype=np.uint32)
        mask[:2] = 1  # two rows of gap pixels
        handle.create_dataset(f"{DET}/detectorSpecific/pixel_mask", data=mask)
        handle.create_dataset("/entry/monitor/curve", data=np.arange(50.0))
        handle.create_dataset("/entry/small/thumb", data=np.full((3, 5), 9.0))
        handle.create_dataset("/entry/title", data="a scan")
    return path


def test_the_dataset_asked_for_is_the_one_read_wherever_it_is(tmp_path: Path) -> None:
    path = _eiger_like(tmp_path / "eiger.nxs")
    assert inspect_nexus(path).dataset == f"{DET}/data"
    for asked, value, frames in ((f"{DET}/detectorSpecific/flatfield", 0.5, 1), ("/entry/small/thumb", 9.0, 1),
                                 (f"{DET}/data", 100.0, 4), ("entry/instrument/detector/detectorSpecific/flatfield", 0.5, 1)):
        image = load_detector_image(path, dataset_path=asked)
        assert set(np.unique(image.data[np.isfinite(image.data)]).tolist()) == {value}, asked
        assert image.metadata["dataset_path"] == asked and image.metadata["layout_warnings"] == []
        assert detect_nxs_frame_count(path, asked) == frames
        layout = inspect_nexus(path, dataset=asked)
        assert layout.decided_by == "the dataset asked for" and layout.detector_group == DET
    assert int(load_detector_image(path).mask.sum()) == 2 * W  # the mask of the file is applied to its image …
    mask = load_detector_image(path, dataset_path=f"{DET}/detectorSpecific/pixel_mask")
    assert not mask.mask.any() and float(mask.data.sum()) == 2 * W  # … and read as it is when it is what was asked for


@pytest.mark.parametrize("asked, sentence", [
    ("/entry/monitor/curve",
     "Dataset /entry/monitor/curve cannot be read as detector frames: its shape is (50,) and its type float64."),
    ("/entry/title", "Dataset /entry/title cannot be read as detector frames: its shape is () and its type object."),
    ("/entry/instrument", "/entry/instrument is a group of the file, not a dataset."),
])
def test_a_dataset_asked_for_that_is_no_image_is_refused_by_name(tmp_path: Path, asked: str, sentence: str) -> None:
    path = _eiger_like(tmp_path / "eiger.nxs")
    for read in (lambda: load_detector_image(path, dataset_path=asked), lambda: detect_nxs_frame_count(path, asked),
                 lambda: inspect_nexus(path, dataset=asked)):
        with pytest.raises(ValueError) as caught:
            read()
        assert str(caught.value) == sentence
    assert inspect_nexus(path).frames == 4  # the file itself reads as before


def test_the_sentences_a_read_fails_with_are_the_templates_presentation_translates(tmp_path: Path) -> None:
    path = _eiger_like(tmp_path / "eiger.nxs")
    curve = _one(tmp_path / "curve.h5", "/entry/result/intensity", np.arange(100.0))
    said = []
    for error, read in ((IndexError, lambda: load_detector_image(path, frame_idx=4)),
                        (ValueError, lambda: load_detector_image(curve)),
                        (ValueError, lambda: detect_nxs_frame_count(curve)),
                        (ValueError, lambda: load_detector_image(path, dataset_path="/entry/monitor/curve")),
                        (ValueError, lambda: load_detector_image(path, dataset_path="/entry/monitor"))):
        with pytest.raises(error) as caught:
            read()
        said.append(str(caught.value))
    assert said == [
        nexus_layout.FRAME_BEYOND.format(frame=5, frames=4),
        nexus_layout.NO_IMAGE.format(name="curve.h5"),
        nexus_layout.NO_IMAGE.format(name="curve.h5"),
        nexus_datasets.DATASET_NOT_FRAMES.format(dataset="/entry/monitor/curve", shape="(50,)", dtype="float64"),
        nexus_datasets.DATASET_IS_GROUP.format(dataset="/entry/monitor"),
    ]
    assert set(nexus_layout.ERROR_TEMPLATES) == {
        nexus_layout.BEING_WRITTEN, nexus_layout.LEFT_UNFINISHED, nexus_layout.NO_IMAGE, nexus_layout.FRAME_BEYOND,
        nexus_datasets.DATASET_NOT_FRAMES, nexus_datasets.DATASET_IS_GROUP}
    assert said[0] == "frame 5 of 4" and said[1] == "curve.h5 holds no detector image."


def test_a_dataset_that_is_not_in_the_file_is_said_and_the_image_is_read(tmp_path: Path) -> None:
    path = _eiger_like(tmp_path / "eiger.nxs")
    image = load_detector_image(path, dataset_path="/nope")
    assert image.metadata["dataset_path"] == f"{DET}/data" and image.metadata["layout_warnings"] == [
        f"Dataset /nope is not in this file: {DET}/data is read instead"]


def test_one_chunk_asked_for_by_its_path_is_read_on_its_own(tmp_path: Path) -> None:
    path = tmp_path / "chunks.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/data/data_000001", data=_stack(3, base=0))
        handle.create_dataset("/entry/data/data_000002", data=_stack(2, base=100))
    assert inspect_nexus(path).frames == 5
    second = "/entry/data/data_000002"
    image = load_detector_image(path, dataset_path=second, frame_idx=0)
    assert int(image.data[0, 0]) == 100 and image.metadata["dataset_path"] == second
    assert image.metadata["frame_count"] == 2 and detect_nxs_frame_count(path, second) == 2
    with pytest.raises(IndexError, match="frame 3 of 2"):
        load_detector_image(path, dataset_path=second, frame_idx=2)
    layout = inspect_nexus(path, dataset=second)
    assert (layout.dataset, layout.chunks, layout.alternatives) == (second, (), ("/entry/data/data_000001",))
    # The first chunk is the name the series goes by (select_nxs_dataset, NexusLayout.dataset): it gives the series.
    first = "/entry/data/data_000001"
    assert detect_nxs_frame_count(path, first) == 5
    assert [int(load_detector_image(path, dataset_path=first, frame_idx=index).data[0, 0]) for index in (0, 2, 3, 4)] == [
        0, 2, 100, 101]
    with open_nexus(path) as handle:
        assert select_nxs_dataset(handle) == first
