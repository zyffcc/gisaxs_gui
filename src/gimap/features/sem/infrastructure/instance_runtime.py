"""Local optional instance segmentation; heavy libraries live in a child process."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import numpy as np
from PIL import Image

from ..application import SemExtractionSettings, SemInstanceSegmentation
from .readers import source_record


def _cancelled(token):
    if token is not None and token():
        raise CancelledError("SEM instance segmentation cancelled")


def _roi(image, settings):
    height, width = image.image.shape
    values = settings.roi_xyxy or (0, 0, width, height)
    if (len(values) != 4 or any(int(v) != v for v in values)
            or not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height)):
        raise ValueError("ROI must have integer pixel bounds inside the SEM image")
    if not np.isfinite(settings.nm_per_pixel) or settings.nm_per_pixel <= 0:
        raise ValueError("A positive explicit calibration in nm/pixel is required")
    return tuple(int(v) for v in values)


def _model_files(folder):
    folder = Path(folder)
    files = [folder / "config.json"]
    thresholds = folder / "thresholds.json"
    if thresholds.is_file():
        files.append(thresholds)
    weights = folder / "weights_best.h5"
    if not weights.is_file():
        weights = folder / "weights_last.h5"
    files.append(weights)
    return files


def _file_records(paths, cancelled=None):
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                _cancelled(cancelled)
                digest.update(block)
        result[path.name] = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                             "sha256": digest.hexdigest()}
    return result


def default_stardist_paths():
    root = Path(os.environ.get("GIMAP_SEM_STARDIST_HOME") or
                Path.home() / ".cache" / "gimap" / "sem" / "stardist")
    executable = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return {"python_executable": str(executable),
            "model_directory": str(root / "models" / "2D_versatile_fluo")}


class LocalSemInstanceSegmenter:
    def __init__(self, *, worker_path=None, timeout_s=900.0, poll_interval=0.05):
        self.worker_path = Path(worker_path or Path(__file__).with_name("stardist_worker.py"))
        self.timeout_s = float(timeout_s)
        self.poll_interval = float(poll_interval)
        self._lock = threading.Lock()

    def defaults(self):
        paths = default_stardist_paths()
        settings = SemExtractionSettings(1.0, method="stardist", **paths)
        return {**paths, **self.status(settings)}

    def status(self, settings):
        if settings.method == "labels":
            if not settings.label_path or not Path(settings.label_path).expanduser().is_file():
                return {"available": False, "message": "Choose an existing instance-label image first."}
            return {"available": True, "message": "Existing labels will be measured on the original image pixel grid."}
        if settings.method != "stardist":
            return {"available": True, "message": "Threshold segmentation is available without an AI runtime."}
        executable = Path(settings.python_executable).expanduser() if settings.python_executable else None
        if executable is None or not executable.is_file():
            return {"available": False, "message": "Choose a local Python interpreter with StarDist and TensorFlow installed."}
        folder = Path(settings.model_directory).expanduser() if settings.model_directory else None
        if folder is None or not folder.is_dir() or not all(p.is_file() for p in _model_files(folder)):
            return {"available": False, "message": "Choose a local StarDist model folder containing config.json and model weights."}
        return {"available": True, "message": "Local model files found. Runtime packages are checked when analysis starts. Review every outline."}

    def segment(self, image, settings, *, cancelled=None):
        _cancelled(cancelled)
        roi = _roi(image, settings)
        status = self.status(settings)
        if not status["available"]:
            raise ValueError(status["message"])
        if settings.method == "labels":
            return self._read_labels(image, settings, roi, cancelled)
        if settings.method != "stardist":
            raise ValueError("Unknown local instance segmentation method")
        self._validate_ai_options(settings)
        # Cancelled tasks remain in TaskRunner until their process exits. This
        # lock prevents a replacement request from loading a second heavy model.
        while not self._lock.acquire(timeout=self.poll_interval):
            _cancelled(cancelled)
        try:
            _cancelled(cancelled)
            return self._predict(image, settings, roi, cancelled)
        finally:
            self._lock.release()

    @staticmethod
    def _validate_ai_options(settings):
        if not np.isfinite(settings.ai_scale) or not 0.1 <= settings.ai_scale <= 8:
            raise ValueError("StarDist scale must be finite and between 0.1 and 8")
        if not (np.isfinite(settings.normalize_low) and np.isfinite(settings.normalize_high)
                and 0 <= settings.normalize_low < settings.normalize_high <= 100):
            raise ValueError("Normalization percentiles must be increasing and between 0 and 100")
        for threshold in (settings.probability_threshold, settings.overlap_threshold):
            if threshold is not None and (not np.isfinite(threshold) or not 0 < threshold < 1):
                raise ValueError("StarDist probability and overlap thresholds must be between 0 and 1")

    @staticmethod
    def _read_labels(image, settings, roi, cancelled):
        path = Path(settings.label_path).expanduser()
        _cancelled(cancelled)
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Choose a single 2D instance-label image, not a multi-page file")
            labels = np.array(source)
            mode = source.mode
        # Validate the stored IDs before any cast or placement. Palette indices
        # are labels here, so unlike RGB SEM image reading they remain indices.
        result = SemInstanceSegmentation(labels, {})
        labels = result.labels
        x0, y0, x1, y1 = roi
        clipped_ids = []
        if labels.shape == image.image.shape:
            inside = np.unique(labels[y0:y1, x0:x1])
            outside = np.ones(labels.shape, bool)
            outside[y0:y1, x0:x1] = False
            clipped_ids = np.intersect1d(inside[inside > 0], np.unique(labels[outside])).tolist()
            full = np.zeros(labels.shape, np.int32)
            full[y0:y1, x0:x1] = labels[y0:y1, x0:x1]
            labels = full
            placement = "full_image; explicit ROI applied; boundary objects use the shared border filter"
        elif labels.shape == (y1 - y0, x1 - x0):
            full = np.zeros(image.image.shape, np.int32)
            full[y0:y1, x0:x1] = labels
            labels = full
            placement = "ROI-sized labels placed at the explicitly selected ROI origin"
        else:
            raise ValueError("Instance labels must match the SEM image or its selected ROI")
        _cancelled(cancelled)
        return SemInstanceSegmentation(labels, {
            "backend": "existing_instance_labels", "label_source": source_record(path),
            "label_image_mode": mode, "stored_label_shape": list(result.labels.shape),
            "label_placement": placement, "roi_xyxy": list(roi),
            "roi_clipped_instance_ids": clipped_ids,
            "label_coordinate_grid": "original_image_pixels", "resampling": "none",
            "original_instance_ids_preserved": True,
        })

    def _predict(self, image, settings, roi, cancelled):
        x0, y0, x1, y1 = roi
        folder = Path(settings.model_directory).expanduser().resolve()
        model_files = _model_files(folder)
        provenance_path = folder / "model_provenance.json"
        if provenance_path.is_file():
            model_files.append(provenance_path)
        before = _file_records(model_files, cancelled)
        with tempfile.TemporaryDirectory(prefix="gimap_sem_stardist_") as temporary:
            work = Path(temporary)
            np.save(work / "image.npy", image.image[y0:y1, x0:x1], allow_pickle=False)
            request = {"input": str(work / "image.npy"), "output": str(work / "labels.npy"),
                       "record": str(work / "record.json"), "error": str(work / "error.json"),
                       "model_directory": str(folder), "settings": asdict(settings)}
            request_path = work / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            self._run_process(settings.python_executable, request_path, work, cancelled)
            _cancelled(cancelled)
            after = _file_records(model_files, cancelled)
            if after != before:
                raise ValueError("Local model files changed during analysis. Use a stable model folder and run again.")
            labels = np.load(work / "labels.npy", allow_pickle=False)
            labels = SemInstanceSegmentation(labels, {}).labels
            if labels.shape != (y1 - y0, x1 - x0):
                raise ValueError("StarDist returned labels outside the original SEM pixel grid")
            metadata = json.loads((work / "record.json").read_text(encoding="utf-8"))
            if provenance_path.is_file():
                metadata["model_origin"] = json.loads(provenance_path.read_text(encoding="utf-8"))
        full = np.zeros(image.image.shape, np.int32)
        full[y0:y1, x0:x1] = labels
        metadata.update({"backend": "stardist_local_process", "model_directory": str(folder),
                         "model_files": before, "python_executable": str(Path(settings.python_executable).expanduser()),
                         "roi_xyxy": list(roi), "label_coordinate_grid": "original_image_pixels",
                         "full_label_shape": list(full.shape), "roi_label_shape": list(labels.shape),
                         "execution": "local subprocess; no image upload or automatic model download",
                         "review_required": True})
        return SemInstanceSegmentation(full, metadata)

    def _run_process(self, executable, request_path, work, cancelled):
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2",
                            "TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "1",
                            "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"})
        command = [str(Path(executable).expanduser()), str(self.worker_path), str(request_path)]
        with (work / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    _cancelled(cancelled)
                    if time.monotonic() - started > self.timeout_s:
                        raise TimeoutError("StarDist timed out. Try a smaller ROI or check the local runtime.")
                    time.sleep(self.poll_interval)
                _cancelled(cancelled)
                if process.returncode:
                    error_path = work / "error.json"
                    if error_path.is_file():
                        error = json.loads(error_path.read_text(encoding="utf-8"))
                        raise RuntimeError(error.get("message", "Local StarDist analysis failed"))
                    raise RuntimeError("Local StarDist could not start. Check that the selected interpreter has StarDist and TensorFlow installed.")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)


__all__ = ["LocalSemInstanceSegmenter", "default_stardist_paths"]
