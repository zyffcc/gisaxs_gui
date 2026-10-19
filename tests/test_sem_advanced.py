"""Advanced SEM boundaries use synthetic instance masks, never real model weights.

These tests validate coordinate/measurement provenance and service orchestration.
They do not establish the accuracy of any trained segmentation model on SEM.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.gimap.features.sem.application import SemAnalysisService
from src.gimap.features.sem.domain import (
    SemExtractionSettings,
    SemImage,
    SemInstanceSegmentation,
    measure_labels,
)


def _labelled_image():
    # The image deliberately contains no threshold contrast. Advanced labels
    # must be measured directly rather than passed through thresholding again.
    image = SemImage(np.zeros((31, 37)), "synthetic_sem", {"sha256": "source-fixture"})
    labels = np.zeros(image.image.shape, dtype=np.int32)
    labels[9:13, 12:17] = 7
    labels[9:13, 17:22] = 29  # Adjacent instances, each with its own original ID.
    labels[7:9, 8:11] = 101  # Touches the ROI's left edge, not the image edge.
    labels[22, 23] = 205  # Retained in the review table, rejected by minimum area.
    settings = SemExtractionSettings(
        nm_per_pixel=0.4, roi_xyxy=(8, 5, 30, 25), min_area_px=2,
        method="stardist", ai_scale=2.0,
    )
    return image, labels, settings


def _metadata():
    return {
        "backend": "synthetic_local_instances",
        "weights_sha256": "a" * 64,
        "runtime_versions": {"synthetic_runtime": "1.2.3"},
        "normalization": {"low_percentile": 1.0, "high_percentile": 99.8},
        "polarity": "bright",
        "prediction_scale": 2.0,
        "labels_coordinate_space": "original_image_pixels",
    }


class _InstancePort:
    def __init__(self, labels, metadata=None):
        self.result = SemInstanceSegmentation(labels, metadata or _metadata())
        self.calls = []

    def segment(self, image, settings, *, cancelled=None):
        self.calls.append((image, settings, cancelled))
        if cancelled is not None and cancelled():
            raise CancelledError("Synthetic advanced segmentation cancelled")
        return self.result


def _service(port=None):
    return SemAnalysisService(None, None, None, instance_segmenter=port)


def test_instance_result_copies_labels_and_nested_provenance():
    labels = np.array([[0, 7], [29, 29]], dtype=np.int64)
    metadata = _metadata()
    result = SemInstanceSegmentation(labels, metadata)
    labels[:] = 0
    metadata["runtime_versions"]["synthetic_runtime"] = "changed"
    np.testing.assert_array_equal(result.labels, [[0, 7], [29, 29]])
    assert result.labels.dtype == np.int32
    assert not result.labels.flags.writeable
    assert result.metadata["runtime_versions"]["synthetic_runtime"] == "1.2.3"


@pytest.mark.parametrize(
    "labels",
    [
        np.array([[0, -1]], dtype=np.int32),
        np.array([[0, 1.5]]),
        np.array([[0, np.nan]]),
        np.array([[0, np.inf]]),
        np.array([[0, 2**31]], dtype=np.uint64),
        np.zeros((1, 2, 3), dtype=np.int32),
        np.zeros((0, 2), dtype=np.int32),
    ],
    ids=["negative", "fractional", "nan", "infinite", "int32-overflow", "three-dimensional", "empty"],
)
def test_invalid_instance_masks_are_rejected_before_conversion(labels):
    with pytest.raises(ValueError):
        SemInstanceSegmentation(labels, {})


def test_measure_labels_preserves_touching_instance_ids_and_original_coordinates():
    image, labels, settings = _labelled_image()
    result = measure_labels(image, labels, settings, metadata=_metadata())
    by_id = {particle.label: particle for particle in result.particles}
    assert set(by_id) == {7, 29, 101, 205}
    assert {particle.label for particle in result.particles if particle.accepted} == {7, 29}
    assert by_id[7].area_px == 20 and by_id[29].area_px == 20
    assert by_id[7].area_nm2 == pytest.approx(20 * 0.4**2)
    assert by_id[7].radius_nm == pytest.approx(math.sqrt(20 / math.pi) * 0.4)
    assert by_id[7].diameter_nm == 2 * by_id[7].radius_nm
    assert (by_id[7].centroid_x_px, by_id[7].centroid_y_px) == (14.5, 11.0)
    assert (by_id[29].centroid_x_px, by_id[29].centroid_y_px) == (19.5, 11.0)
    assert "touches_roi_border" in by_id[101].rejection_reason
    assert "area_below_minimum" in by_id[205].rejection_reason
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_array_equal(result.image, image.image)
    assert not result.labels.flags.writeable
    assert result.summary["review_required"] is True


def test_restored_original_pixel_labels_are_not_scaled_again_for_physical_measurements():
    image, labels, settings = _labelled_image()
    original = measure_labels(image, labels, replace(settings, ai_scale=1.0))
    scaled_model = measure_labels(image, labels, replace(settings, ai_scale=2.0))
    np.testing.assert_array_equal(original.radius_nm, scaled_model.radius_nm)
    assert [p.area_px for p in original.particles] == [p.area_px for p in scaled_model.particles]
    assert scaled_model.settings["ai_scale"] == 2.0

    calibrated = measure_labels(image, labels, replace(settings, nm_per_pixel=0.8))
    np.testing.assert_allclose(calibrated.radius_nm, 2 * original.radius_nm)
    np.testing.assert_allclose(
        [p.area_nm2 for p in calibrated.particles],
        4 * np.array([p.area_nm2 for p in original.particles]),
    )


@pytest.mark.parametrize("calibration", [None, 0, -1, np.nan])
def test_advanced_labels_still_require_explicit_physical_calibration(calibration):
    image, labels, settings = _labelled_image()
    with pytest.raises(ValueError, match="(?i)calibration"):
        measure_labels(image, labels, replace(settings, nm_per_pixel=calibration))


def test_labels_outside_roi_and_labels_of_a_different_shape_are_errors():
    image, labels, settings = _labelled_image()
    with pytest.raises(ValueError, match="(?i)shape|resolution"):
        measure_labels(image, labels[:-1], settings)
    outside = labels.copy()
    outside[0, 0] = 7
    with pytest.raises(ValueError, match="(?i)ROI|outside"):
        measure_labels(image, outside, settings)


def test_all_rejected_advanced_instances_keep_their_labels_for_review():
    image, labels, settings = _labelled_image()
    result = measure_labels(image, labels, replace(settings, min_area_px=1000))
    assert result.radius_nm.size == 0
    assert len(result.particles) == 4
    assert all(not particle.accepted for particle in result.particles)
    np.testing.assert_array_equal(result.labels, labels)
    assert result.summary["review_required"] is True


def test_sparse_large_instance_id_is_preserved_without_relabelling():
    image = SemImage(np.zeros((7, 9)), "synthetic")
    labels = np.zeros(image.image.shape, dtype=np.int32)
    labels[2:4, 3:6] = 1_000_007
    result = measure_labels(image, labels, SemExtractionSettings(0.5, method="labels", min_area_px=1))
    particle, = result.particles
    assert particle.label == 1_000_007
    assert particle.area_px == 6
    np.testing.assert_array_equal(result.labels, labels)


def test_empty_advanced_detection_is_a_reviewable_result():
    image = SemImage(np.zeros((7, 9)), "synthetic")
    labels = np.zeros(image.image.shape, dtype=np.int32)
    result = measure_labels(image, labels, SemExtractionSettings(0.5, method="labels"))
    assert not result.particles
    assert not result.radius_nm.size
    assert result.summary["detected_count"] == 0
    assert result.summary["review_required"] is True
    np.testing.assert_array_equal(result.labels, labels)


@pytest.mark.parametrize("method", ["stardist", "labels"])
def test_service_routes_advanced_instances_through_port_and_shared_measurement(method):
    image, labels, settings = _labelled_image()
    settings = replace(settings, method=method, threshold=1e9, bright_particles=False, split_touching=True)
    port = _InstancePort(labels)
    cancelled = lambda: False
    result = _service(port).extract(image, settings, cancelled=cancelled)
    assert port.calls == [(image, settings, cancelled)]
    assert {p.label for p in result.particles if p.accepted} == {7, 29}
    np.testing.assert_array_equal(result.labels, labels)
    assert result.source["sha256"] == "source-fixture"
    # This data must survive to the exporter, wherever the service nests it.
    recorded = json.dumps({"source": result.source, "settings": result.settings, "summary": result.summary})
    assert "a" * 64 in recorded
    assert "synthetic_local_instances" in recorded
    assert "1.2.3" in recorded
    assert "original_image_pixels" in recorded
    assert result.settings["method"] == method


@pytest.mark.parametrize("method", ["stardist", "labels"])
def test_missing_advanced_backend_has_an_explicit_error_without_classical_fallback(method):
    image, _labels, settings = _labelled_image()
    with pytest.raises((ValueError, RuntimeError), match="(?i)backend|segment|configured|available|model|instance"):
        _service().extract(image, replace(settings, method=method))


def test_classical_threshold_still_runs_without_using_the_instance_backend():
    image = np.zeros((20, 24))
    image[5:8, 10:14] = 100

    class DoNotCall:
        def segment(self, *_args, **_kwargs):
            raise AssertionError("Threshold extraction must not start an advanced runtime")

    settings = SemExtractionSettings(0.5, threshold=50, min_area_px=1)
    result = _service(DoNotCall()).extract(SemImage(image, "synthetic"), settings)
    assert result.summary["accepted_count"] == 1
    assert result.particles[0].area_px == 12
    assert result.radius_nm[0] == pytest.approx(math.sqrt(12 / math.pi) * 0.5)


def test_advanced_backend_failure_is_not_replaced_with_successful_threshold_result():
    class FailingBackend:
        def segment(self, *_args, **_kwargs):
            raise RuntimeError("Synthetic local segmentation failed")

    image, _labels, settings = _labelled_image()
    with pytest.raises(RuntimeError, match="Synthetic local segmentation failed"):
        _service(FailingBackend()).extract(image, settings)


def test_cancelled_request_never_starts_the_instance_backend():
    image, labels, settings = _labelled_image()
    port = _InstancePort(labels)
    with pytest.raises(CancelledError):
        _service(port).extract(image, settings, cancelled=lambda: True)
    assert not port.calls


def test_cancellation_after_backend_return_does_not_publish_old_labels():
    image, labels, settings = _labelled_image()
    cancellation = {"requested": False}

    class LateBackend:
        def segment(self, *_args, **_kwargs):
            cancellation["requested"] = True
            return SemInstanceSegmentation(labels, _metadata())

    with pytest.raises(CancelledError):
        _service(LateBackend()).extract(image, settings, cancelled=lambda: cancellation["requested"])


def test_measurement_checks_cancellation_during_instance_processing():
    image, labels, settings = _labelled_image()
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) >= 3

    with pytest.raises(CancelledError):
        measure_labels(image, labels, settings, cancelled=cancelled)
    assert len(checks) >= 3


def test_existing_roi_labels_need_no_model_and_preserve_palette_ids(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    image, labels, settings = _labelled_image()
    roi_labels = labels[5:25, 8:30].astype(np.uint8)
    path = tmp_path / "synthetic_instance_ids.png"
    palette_image = Image.fromarray(roi_labels).convert("P")
    # Display colors have no relation to instance identity.
    palette_image.putpalette([value for i in range(256) for value in (255 - i, i, i // 2)])
    palette_image.save(path)
    original_bytes = path.read_bytes()
    settings = replace(settings, method="labels", label_path=str(path), python_executable="", model_directory="")
    adapter = LocalSemInstanceSegmenter()
    monkeypatch.setattr(adapter, "_run_process", lambda *_a, **_k: pytest.fail("Existing labels must not start a model"))

    assert adapter.status(settings)["available"]
    result = _service(adapter).extract(image, settings)
    np.testing.assert_array_equal(result.labels, labels)
    assert {p.label for p in result.particles} == {7, 29, 101, 205}
    source = result.settings["instance_segmentation"]["label_source"]
    assert source["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert source["path"] == str(path.resolve())
    assert path.read_bytes() == original_bytes


def test_existing_full_image_labels_apply_explicit_roi_without_losing_instance_ids(tmp_path):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    image, labels, settings = _labelled_image()
    path = tmp_path / "synthetic_labels.tif"
    Image.fromarray(labels).save(path)
    settings = replace(settings, method="labels", label_path=str(path))
    result = _service(LocalSemInstanceSegmenter()).extract(image, settings)
    np.testing.assert_array_equal(result.labels, labels)
    source_labels = labels.copy()
    source_labels[7:9, 6:8] = 101  # Same instance continues beyond the ROI's left edge.
    source_labels[1:3, 2:4] = 901  # An independent exterior instance is excluded completely.
    Image.fromarray(source_labels).save(path)
    result = _service(LocalSemInstanceSegmenter()).extract(image, settings)
    np.testing.assert_array_equal(result.labels, labels)
    assert 901 not in {particle.label for particle in result.particles}
    partial = next(particle for particle in result.particles if particle.label == 101)
    assert "touches_roi_border" in partial.rejection_reason
    provenance = result.settings["instance_segmentation"]
    assert provenance["roi_clipped_instance_ids"] == [101]
    assert "ROI" in provenance["label_placement"]


@pytest.mark.parametrize("kind", ["wrong_shape", "rgb", "multi_page"])
def test_existing_labels_cannot_be_silently_resized_or_converted(tmp_path, kind):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    image, _labels, settings = _labelled_image()
    path = tmp_path / "invalid_labels.tif"
    if kind == "wrong_shape":
        Image.fromarray(np.zeros((8, 9), np.int32)).save(path)
    elif kind == "rgb":
        Image.fromarray(np.zeros((*image.image.shape, 3), np.uint8)).save(path)
    else:
        first = Image.fromarray(np.zeros(image.image.shape, np.int32))
        first.save(path, save_all=True, append_images=[first])
    settings = replace(settings, method="labels", label_path=str(path))
    with pytest.raises(ValueError):
        _service(LocalSemInstanceSegmenter()).extract(image, settings)


def test_worker_normalizes_selected_polarity_without_importing_model_libraries():
    from src.gimap.features.sem.infrastructure.stardist_worker import normalize_roi

    raw = np.arange(80, dtype=float).reshape(8, 10)
    raw[2, 3] = np.nan
    before = raw.copy()
    bright, valid, bright_record = normalize_roi(raw, bright_particles=True, low=5, high=95)
    dark, dark_valid, dark_record = normalize_roi(-raw, bright_particles=False, low=5, high=95)
    np.testing.assert_array_equal(bright, dark)
    np.testing.assert_array_equal(valid, dark_valid)
    np.testing.assert_array_equal(raw, before)
    assert np.isfinite(bright).all()
    assert not valid[2, 3]
    assert bright_record["non_finite_count"] == 1
    assert bright_record["channel_strategy"] == "single_channel_YX"
    assert bright_record["polarity"] != dark_record["polarity"]
    assert bright_record["clip"] is False


def _fake_stardist(monkeypatch, labels, *, channels=1):
    calls = []

    class FakeModel:
        config = SimpleNamespace(n_channel_in=channels, n_rays=32, grid=(2, 2))
        thresholds = SimpleNamespace(prob=0.5, nms=0.4)

        def __init__(self, *_args, **_kwargs):
            pass

        def predict_instances(self, values, **kwargs):
            calls.append((values.copy(), kwargs))
            return labels.copy(), {"prob": np.array([0.8, 0.9])}

    tensorflow = ModuleType("tensorflow")
    tensorflow.__version__ = "synthetic-imported-tensorflow"
    tensorflow.config = SimpleNamespace(set_visible_devices=lambda *_a, **_k: None)
    stardist = ModuleType("stardist")
    models = ModuleType("stardist.models")
    models.StarDist2D = FakeModel
    stardist.models = models
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setitem(sys.modules, "stardist", stardist)
    monkeypatch.setitem(sys.modules, "stardist.models", models)
    from src.gimap.features.sem.infrastructure import stardist_worker

    def distribution_version(name):
        if name == "tensorflow":
            raise stardist_worker.importlib.metadata.PackageNotFoundError(name)
        return "synthetic-test-runtime"

    monkeypatch.setattr(stardist_worker.importlib.metadata, "version", distribution_version)
    return stardist_worker, calls


def test_worker_maps_scale_to_original_roi_and_excludes_invalid_source_pixels(tmp_path, monkeypatch):
    raw = np.arange(80, dtype=float).reshape(8, 10)
    raw[2, 3] = np.nan
    input_path = tmp_path / "image.npy"
    np.save(input_path, raw)
    labels = np.zeros(raw.shape, np.int32)
    labels[1:4, 2:6] = 7
    worker, calls = _fake_stardist(monkeypatch, labels)
    settings = replace(SemExtractionSettings(0.5, method="stardist"), ai_scale=2,
                       probability_threshold=0.7, overlap_threshold=0.2)
    from dataclasses import asdict

    request = {"input": str(input_path), "output": str(tmp_path / "labels.npy"),
               "record": str(tmp_path / "record.json"), "model_directory": str(tmp_path / "local_model"),
               "settings": asdict(settings)}
    worker.predict(request)
    result = np.load(request["output"], allow_pickle=False)
    expected = labels.copy()
    expected[2, 3] = 0
    np.testing.assert_array_equal(result, expected)
    assert calls[0][0].shape == raw.shape
    assert calls[0][1]["axes"] == "YX"
    assert calls[0][1]["scale"] == (2, 2)
    assert calls[0][1]["prob_thresh"] == 0.7
    assert calls[0][1]["nms_thresh"] == 0.2
    metadata = json.loads(Path(request["record"]).read_text())
    assert metadata["input_roi_shape"] == metadata["output_roi_shape"] == [8, 10]
    assert metadata["prediction_scale"] == [2, 2]
    assert metadata["normalization"]["non_finite_count"] == 1
    assert metadata["versions"]["stardist"] == "synthetic-test-runtime"
    assert metadata["versions"]["tensorflow"] == "synthetic-imported-tensorflow"


def test_rgb_model_is_rejected_instead_of_silently_reinterpreting_sem_channels(tmp_path, monkeypatch):
    worker, calls = _fake_stardist(monkeypatch, np.zeros((8, 10), np.int32), channels=3)
    source = tmp_path / "image.npy"
    np.save(source, np.arange(80).reshape(8, 10))
    from dataclasses import asdict

    request = {"input": str(source), "output": str(tmp_path / "labels.npy"),
               "record": str(tmp_path / "record.json"), "model_directory": str(tmp_path / "local_model"),
               "settings": asdict(SemExtractionSettings(1, method="stardist"))}
    with pytest.raises(ValueError, match="single-channel"):
        worker.predict(request)
    assert not calls
    assert not Path(request["output"]).exists()


def _local_runtime_fixture(tmp_path, worker_source):
    folder = tmp_path / "local_model"
    folder.mkdir()
    (folder / "config.json").write_text('{"n_channel_in": 1}', encoding="utf-8")
    (folder / "weights_best.h5").write_bytes(b"synthetic fixture bytes; not trained weights")
    worker_path = tmp_path / "local_worker.py"
    worker_path.write_text(worker_source, encoding="utf-8")
    image, _labels, settings = _labelled_image()
    return image, replace(settings, python_executable=sys.executable, model_directory=str(folder)), worker_path


def _track_child_processes(monkeypatch):
    from src.gimap.features.sem.infrastructure import instance_runtime

    original = instance_runtime.subprocess.Popen
    processes = []

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append((process, Path(args[0][-1]).parent))
        return process

    monkeypatch.setattr(instance_runtime.subprocess, "Popen", start)
    return processes


def test_local_process_receives_only_roi_and_records_actual_weight_hash(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    worker = """import json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[1]).read_text())
image = np.load(request['input'], allow_pickle=False)
labels = np.zeros(image.shape, np.int32)
labels[2:4, 3:6] = 41
np.save(request['output'], labels, allow_pickle=False)
Path(request['record']).write_text(json.dumps({'seen_input_shape': list(image.shape)}))
"""
    image, settings, worker_path = _local_runtime_fixture(tmp_path, worker)
    children = _track_child_processes(monkeypatch)
    adapter = LocalSemInstanceSegmenter(worker_path=worker_path, poll_interval=0.01)
    result = adapter.segment(image, settings)
    expected = np.zeros(image.image.shape, np.int32)
    expected[7:9, 11:14] = 41
    np.testing.assert_array_equal(result.labels, expected)
    assert result.metadata["seen_input_shape"] == [20, 22]
    assert result.metadata["roi_xyxy"] == [8, 5, 30, 25]
    weights = Path(settings.model_directory) / "weights_best.h5"
    assert result.metadata["model_files"][weights.name]["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert result.metadata["label_coordinate_grid"] == "original_image_pixels"
    assert result.metadata["review_required"] is True
    assert children and all(process.poll() is not None and not work.exists() for process, work in children)


def test_cancel_reaps_real_local_child_and_removes_its_temporary_input(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    image, settings, worker_path = _local_runtime_fixture(tmp_path, "import time\nwhile True: time.sleep(0.05)\n")
    children = _track_child_processes(monkeypatch)
    cancelled = threading.Event()
    errors = []
    adapter = LocalSemInstanceSegmenter(worker_path=worker_path, poll_interval=0.01)

    def run():
        try:
            adapter.segment(image, settings, cancelled=cancelled.is_set)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not children and time.monotonic() < deadline:
            time.sleep(0.005)
        assert children, "The cancellable test worker did not start"
        cancelled.set()
        thread.join(5)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], CancelledError)
        assert all(process.poll() is not None and not work.exists() for process, work in children)
    finally:
        cancelled.set()
        thread.join(5)


def test_timeout_reaps_local_child_without_a_prediction_result(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    image, settings, worker_path = _local_runtime_fixture(tmp_path, "import time\nwhile True: time.sleep(0.05)\n")
    children = _track_child_processes(monkeypatch)
    adapter = LocalSemInstanceSegmenter(worker_path=worker_path, timeout_s=0.05, poll_interval=0.01)
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.segment(image, settings)
    assert children and all(process.poll() is not None and not work.exists() for process, work in children)


def test_missing_local_resources_are_reported_without_launch_or_download(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure import instance_runtime

    private_root = tmp_path / "uninstalled_local_runtime"
    monkeypatch.setenv("GIMAP_SEM_STARDIST_HOME", str(private_root))
    monkeypatch.setattr(instance_runtime.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Status must stay lightweight"))
    adapter = instance_runtime.LocalSemInstanceSegmenter()
    defaults = adapter.defaults()
    assert not defaults["available"]
    assert not private_root.exists(), "Inspecting a missing model must not install/download anything"
    settings = SemExtractionSettings(0.5, method="stardist")
    missing_interpreter = adapter.status(settings)
    assert not missing_interpreter["available"]
    assert "interpreter" in missing_interpreter["message"].lower()
    missing_model = adapter.status(replace(settings, python_executable=sys.executable))
    assert not missing_model["available"]
    assert "model" in missing_model["message"].lower()
    assert adapter.status(replace(settings, method="threshold"))["available"]


def test_model_changes_during_process_inference_are_rejected(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    worker = """import json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[1]).read_text())
image = np.load(request['input'], allow_pickle=False)
(Path(request['model_directory']) / 'weights_best.h5').write_bytes(b'changed synthetic weights')
np.save(request['output'], np.zeros(image.shape, np.int32), allow_pickle=False)
Path(request['record']).write_text('{}')
"""
    image, settings, worker_path = _local_runtime_fixture(tmp_path, worker)
    children = _track_child_processes(monkeypatch)
    adapter = LocalSemInstanceSegmenter(worker_path=worker_path, poll_interval=0.01)
    with pytest.raises(ValueError, match="model files changed"):
        adapter.segment(image, settings)
    assert children and all(process.poll() is not None and not work.exists() for process, work in children)


def test_waiting_replacement_can_cancel_without_launching_a_second_model(tmp_path, monkeypatch):
    from src.gimap.features.sem.infrastructure.instance_runtime import LocalSemInstanceSegmenter

    worker = """import json, sys, time
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
release = Path(request['model_directory']) / 'release_test_worker'
while not release.exists(): time.sleep(0.01)
import numpy as np
image = np.load(request['input'], allow_pickle=False)
np.save(request['output'], np.zeros(image.shape, np.int32), allow_pickle=False)
Path(request['record']).write_text('{}')
"""
    image, settings, worker_path = _local_runtime_fixture(tmp_path, worker)
    children = _track_child_processes(monkeypatch)
    adapter = LocalSemInstanceSegmenter(worker_path=worker_path, poll_interval=0.01)
    first_cancel, second_cancel = threading.Event(), threading.Event()
    results, errors, second_checks = {}, {}, []

    def second_token():
        second_checks.append(True)
        return second_cancel.is_set()

    def run(key, token):
        try:
            results[key] = adapter.segment(image, settings, cancelled=token)
        except Exception as error:
            errors[key] = error

    first = threading.Thread(target=run, args=("first", first_cancel.is_set))
    second = threading.Thread(target=run, args=("second", second_token))
    first.start()
    try:
        deadline = time.monotonic() + 5
        while not children and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(children) == 1
        second.start()
        while len(second_checks) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(second_checks) >= 2, "Replacement should be waiting for the first local model"
        second_cancel.set()
        second.join(5)
        assert not second.is_alive()
        assert isinstance(errors.get("second"), CancelledError)
        assert len(children) == 1, "A cancelled replacement must not load another heavy model"
        (Path(settings.model_directory) / "release_test_worker").touch()
        first.join(5)
        assert not first.is_alive() and "first" in results
        assert "first" not in errors
        assert all(process.poll() is not None and not work.exists() for process, work in children)
    finally:
        first_cancel.set()
        second_cancel.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)


def test_gui_missing_ai_resources_disable_ai_without_blocking_classical_analysis(qt_application, tmp_path, monkeypatch):
    from src.gimap.features.sem.bootstrap import create_sem_window

    monkeypatch.setenv("GIMAP_SEM_STARDIST_HOME", str(tmp_path / "missing_runtime"))
    image = np.zeros((20, 24), np.uint8)
    image[5:9, 10:15] = 200
    source = tmp_path / "synthetic_sem.png"
    Image.fromarray(image).save(source)
    window = create_sem_window()
    try:
        window.show()
        window._open_source(source, "image")
        assert window._runner.wait(5)
        window.nm_per_pixel_spin.setValue(0.5)
        window.run_button.click()
        assert window._runner.wait(5)
        assert window.view_model.state.measurements.radius_nm.size == 1
        assert window.export_button.isEnabled()

        window.detection_method_combo.setCurrentIndex(window.detection_method_combo.findData("stardist"))
        assert window.view_model.state.measurements is None
        assert not window.export_button.isEnabled()
        assert not window.run_button.isEnabled()
        assert window.instance_status_label.text()
        assert window.ai_resources_widget.isVisibleTo(window)
        assert window.browse_python_button.isEnabled()
        assert window.browse_instance_model_button.isEnabled()
        assert not window.threshold_mode_combo.isEnabled()
        assert window.polarity_combo.isEnabled()

        window.python_executable_edit.setText(sys.executable)
        window.model_directory_edit.setText(str(tmp_path / "missing_model"))
        assert not window.run_button.isEnabled()
        assert not window._runner.is_busy(), "Missing resources must not launch a download or substitute prediction"
        window.detection_method_combo.setCurrentIndex(window.detection_method_combo.findData("threshold"))
        assert window.run_button.isEnabled()
        window.run_button.click()
        assert window._runner.wait(5)
        assert window.view_model.state.measurements.radius_nm.size == 1
        assert window.view_model.state.measurements.settings["method"] == "threshold"
    finally:
        window.dispose()
        window.close()


def test_gui_method_change_discards_late_advanced_instances(qt_application, tmp_path, monkeypatch):
    from src.gimap.features.sem.bootstrap import create_sem_window

    monkeypatch.setenv("GIMAP_SEM_STARDIST_HOME", str(tmp_path / "missing_runtime"))
    source = tmp_path / "synthetic_sem.png"
    image = np.zeros((31, 37), np.uint8)
    image[9:13, 12:17] = 200
    Image.fromarray(image).save(source)
    labels = np.zeros(image.shape, np.int32)
    labels[9:13, 12:17] = 77
    started, release = threading.Event(), threading.Event()

    class DelayedBackend:
        def segment(self, _image, _settings, *, cancelled=None):
            started.set()
            assert release.wait(5)
            return SemInstanceSegmentation(labels, _metadata())

    window = create_sem_window()
    monkeypatch.setattr(window.view_model.service, "_instance_segmenter", DelayedBackend())
    monkeypatch.setattr(window.view_model.service, "instance_status",
                        lambda _settings: {"available": True, "message": "Synthetic local backend"})
    try:
        window._open_source(source, "image")
        assert window._runner.wait(5)
        window.nm_per_pixel_spin.setValue(0.5)
        window.detection_method_combo.setCurrentIndex(window.detection_method_combo.findData("stardist"))
        assert window.run_button.isEnabled()
        window.run_button.click()
        assert started.wait(5)
        window.detection_method_combo.setCurrentIndex(window.detection_method_combo.findData("threshold"))
        release.set()
        assert window._runner.wait(5)
        assert window.view_model.state.measurements is None
        assert window.view_model.state.comparison is None
        assert not window.export_button.isEnabled()
        assert window.run_button.isEnabled()
        window.run_button.click()
        assert window._runner.wait(5)
        assert window.view_model.state.measurements.radius_nm.size == 1
        assert window.view_model.state.measurements.settings["method"] == "threshold"
    finally:
        release.set()
        window._runner.wait(5)
        window.dispose()
        window.close()


def test_advanced_export_retains_instance_provenance_and_original_labels(tmp_path):
    from src.gimap.features.sem.infrastructure import LocalSemExporter, LocalSemInstanceSegmenter

    image, labels, settings = _labelled_image()
    service = SemAnalysisService(None, None, LocalSemExporter(), instance_segmenter=_InstancePort(labels))
    measurements = service.extract(image, settings)
    comparison = service.compare(measurements)
    output = tmp_path / "advanced_export"
    service.export(measurements, comparison, output)
    record = json.loads((output / "sem_analysis.json").read_text())
    recorded = record["measurement_settings"]
    assert recorded["method"] == "stardist"
    assert recorded["roi_xyxy"] == [8, 5, 30, 25]
    assert recorded["nm_per_pixel"] == 0.4
    assert recorded["ai_scale"] == 2.0
    assert recorded["instance_segmentation"]["weights_sha256"] == "a" * 64
    assert recorded["instance_segmentation"]["runtime_versions"]["synthetic_runtime"] == "1.2.3"
    assert recorded["instance_segmentation"]["labels_coordinate_space"] == "original_image_pixels"
    assert not record["comparison"]["prediction_included"]
    with np.load(output / "sem_segmentation.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["labels"], labels)
        np.testing.assert_array_equal(stored["image"], image.image)

    label_path = output / "sem_instance_labels.tif"
    assert label_path.name in record["files"]
    original_tiff_bytes = label_path.read_bytes()
    with Image.open(label_path) as stored:
        assert stored.info["compression"] == "tiff_adobe_deflate"
        exported_labels = np.asarray(stored)
        assert exported_labels.dtype == np.int32
        np.testing.assert_array_equal(exported_labels, labels)
    # Import against the original image and the same ROI, without a model.
    # Accepted and rejected instances must retain their original identities.
    restored = _service(LocalSemInstanceSegmenter()).extract(
        image, replace(settings, method="labels", label_path=str(label_path)),
    )
    np.testing.assert_array_equal(restored.labels, measurements.labels)
    np.testing.assert_array_equal(restored.radius_nm, measurements.radius_nm)
    assert restored.particles == measurements.particles
    assert {p.label for p in restored.particles if not p.accepted} == {101, 205}
    assert label_path.read_bytes() == original_tiff_bytes

    protected = tmp_path / "protected_export"
    protected.mkdir()
    protected_labels = protected / label_path.name
    protected_labels.write_bytes(original_tiff_bytes)
    with pytest.raises(FileExistsError, match="already exist"):
        service.export(measurements, comparison, protected)
    assert protected_labels.read_bytes() == original_tiff_bytes
    assert list(protected.iterdir()) == [protected_labels], "TIFF conflict must prevent partial exports"
    assert not list(output.glob("*.h5")), "Result exports must not copy local model weights"
