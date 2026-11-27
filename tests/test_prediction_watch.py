"""Watch for new frames in a folder batch of 2D Prediction: frames written into the folder while the batch runs are
predicted too, once completely written; Keep up with the detector predicts the newest when the model is free and
counts the frames left out. Synthetic files and a fake model call (no TensorFlow)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from src.gimap.features.prediction.application import FolderEntry, FrameWatch, KeepUp
from src.gimap.features.prediction.infrastructure.adapters.fabio_images import LocalPredictionFileCatalog
from src.gimap.features.prediction.presentation.batch_manager import MultiFilePredictManager
from src.gimap.features.prediction.presentation.bindings import live_canvas
from src.gimap.features.prediction.presentation.result_types import PredictStatus
from tests.test_labs_prediction import _app, _wait, _window


def _entry(name, size=10, stamp=1, complete=None):
    return FolderEntry(name, (size, stamp), complete)


def _number(path: str):
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    return int(digits) if digits else None


def test_a_new_frame_is_taken_once_it_is_completely_written():
    watch = FrameWatch(["run_00001.cbf"], index_of=_number)
    assert watch.take([_entry("run_00001.cbf"), _entry("run_00002.cbf", 5, 1)]) == []  # first seen
    assert watch.unsettled == 1
    assert watch.take([_entry("run_00002.cbf", 9, 2)]) == []  # still growing
    assert watch.take([_entry("run_00002.cbf", 9, 2)]) == [["run_00002.cbf"]]  # the same at two looks
    assert watch.take([_entry("run_00002.cbf", 9, 2)]) == []  # taken once
    # an EDF that says it is whole is taken at once; one that says it is not, never on its stamp alone
    assert watch.take([_entry("run_00003.edf", complete=True)]) == [["run_00003.edf"]]
    assert watch.take([_entry("run_00004.edf", complete=False)], now=100.0) == []
    assert watch.take([_entry("run_00004.edf", complete=False)], now=101.0) == []  # unchanged, but says incomplete
    assert watch.take([_entry("run_00004.edf", complete=False)], now=105.5) == [["run_00004.edf"]]  # a damaged file
    assert watch.take([_entry("run_00005.edf", complete=False)]) == []
    assert watch.take([]) == [] and watch.unsettled == 0  # gone before it was whole: forgotten


def test_new_frames_make_stacks_of_every_in_the_order_of_their_numbers():
    watch = FrameWatch([], 2, waiting=["run_00005.cbf"], index_of=_number)  # the trailing frame of the listed ones
    whole = [_entry(f"run_{n:05d}.cbf", complete=True) for n in (8, 6, 7)]
    assert watch.take(whole) == [["run_00005.cbf", "run_00006.cbf"], ["run_00007.cbf", "run_00008.cbf"]]
    assert watch.take([_entry("run_00009.cbf", complete=True)]) == [] and watch.waiting == ("run_00009.cbf",)
    ranged = FrameWatch([], indices=range(1, 4), index_of=_number)
    assert ranged.take([_entry("run_00002.cbf", complete=True), _entry("run_00007.cbf", complete=True)]) == [
        ["run_00002.cbf"]]
    assert ranged.outside == 1


def test_keeping_up_lets_two_stacks_wait_and_leaves_out_the_older_ones():
    assert KeepUp(True).depth == 1  # one here and one in the model's queue: two wait, besides the one predicted
    keep_up = KeepUp(True, depth=2)
    assert keep_up.offer([["a"]], free=True) == [["a"]]  # to the model's queue at once
    assert keep_up.offer([["b"]], free=False) == [] and list(keep_up.pending) == [["b"]]
    assert keep_up.offer([["c"], ["d"]], free=False) == [] and list(keep_up.pending) == [["c"], ["d"]]
    assert keep_up.left_out == 1  # b: older than the two that may wait
    assert keep_up.release(free=False) == [] and keep_up.release(free=True) == [["c"]]
    assert keep_up.offer([["e", "f"], ["g", "h"], ["i", "j"]], free=False) == [] and keep_up.left_out == 4
    assert list(keep_up.pending) == [["g", "h"], ["i", "j"]] and keep_up.release_all() == [["g", "h"], ["i", "j"]]
    fast = KeepUp(True)  # a model as fast as the detector: one new stack at a time, the queue free each time
    assert [fast.offer([[name]], free=True) for name in "xyz"] == [[["x"]], [["y"]], [["z"]]] and fast.left_out == 0
    every_one = KeepUp(False)
    assert every_one.offer([["a"], ["b"]], free=False) == [["a"], ["b"]] and every_one.left_out == 0


def test_the_folder_look_lists_the_frames_with_their_size_and_whether_an_edf_is_whole(tmp_path):
    header = b"{\nHeaderID = EH:000001:000000:000000 ;\nSize = 8 ;\nDataType = SignedInteger ;\n}\n"
    header = header[:-2] + b" " * (512 - len(header)) + b"}\n"
    (tmp_path / "whole.edf").write_bytes(header + b"\0" * 8)
    (tmp_path / "half.edf").write_bytes(header + b"\0" * 3)
    (tmp_path / "frame.cbf").write_bytes(b"cbf")
    (tmp_path / "notes.txt").write_text("a side file")
    entries = {Path(e.path).name: e for e in LocalPredictionFileCatalog().look(tmp_path, (".cbf", ".edf"))}
    assert set(entries) == {"whole.edf", "half.edf", "frame.cbf"}
    assert entries["whole.edf"].complete is True and entries["half.edf"].complete is False
    assert entries["frame.cbf"].complete is None and entries["frame.cbf"].stamp[0] == 3
    listed = LocalPredictionFileCatalog().compatible_files(tmp_path, (".cbf", ".edf"))
    assert {str(path) for path in listed} == {e.path for e in entries.values()}  # the same paths as the batch
    # The frames a watch has taken are not reported again, nor opened (a long run's folder holds thousands).
    import src.gimap.features.prediction.infrastructure.adapters.fabio_images as adapter
    opened = []
    real = adapter.edf_written
    adapter.edf_written = lambda path: opened.append(Path(path).name) or real(path)
    try:
        again = LocalPredictionFileCatalog().look(tmp_path, (".cbf", ".edf"), frozenset({entries["whole.edf"].path}))
    finally:
        adapter.edf_written = real
    assert {Path(e.path).name for e in again} == {"half.edf", "frame.cbf"} and opened == ["half.edf"]


def test_a_followed_run_takes_frames_given_while_it_runs_until_it_is_told_to_end():
    _app()
    manager = MultiFilePredictManager()
    predicted, updates, ended = [], [], threading.Event()
    manager.result_updated.connect(lambda row, data: updates.append((row, data["status"])))
    manager.prediction_completed.connect(ended.set)
    manager.start_batch_prediction(["a"], lambda path: predicted.append(path) or {"file": path}, follow=True)
    _wait(manager.idle, 5)
    assert predicted == ["a"] and manager.is_running  # waits for more
    assert manager.append([(1, "b"), (2, "c")])
    _wait(lambda: predicted == ["a", "b", "c"], 5)
    manager.stop_following()
    _wait(lambda: not manager.is_running, 5)
    _wait(ended.is_set, 2)
    assert ended.is_set() and not manager.append([(3, "d")])  # an ended run takes nothing
    assert (2, PredictStatus.COMPLETED) in updates


def test_stop_ends_a_followed_run_and_cancels_the_frames_not_started():
    _app()
    manager = MultiFilePredictManager()
    updates = []
    manager.result_updated.connect(lambda row, data: updates.append((row, data["status"])))
    manager.start_batch_prediction(["a", "b", "c"], lambda path: time.sleep(0.3) or {"file": path}, follow=True)
    _wait(lambda: (0, PredictStatus.RUNNING) in updates, 5)
    manager.cancel_prediction()
    _wait(lambda: not manager.is_running, 5)
    _wait(lambda: (2, PredictStatus.CANCELLED) in updates, 2)
    assert (1, PredictStatus.CANCELLED) in updates and (2, PredictStatus.CANCELLED) in updates


def _watched_batch(tmp_path, keep_up: bool, delay: float = 0.0):
    window = _window()
    binding = window.runtime.prediction
    predicted = []

    def fake(path):  # the model call: the rest of the batch is the page's own
        time.sleep(delay)
        predicted.append(Path(path).name)
        return {"file": path, "stack_count": 1, "stack_files": [path], "prediction_data": {"parameters": [1.0]}}

    binding._predict_single_file_for_batch = fake
    window.gisaxsPredictMultiFilesRadioButton.setChecked(True)
    window.gisaxsPredictWatchCheck.setChecked(True)
    window.gisaxsPredictKeepUpCheck.setChecked(keep_up)
    binding.current_parameters["mode"] = "multi_files"
    binding.current_parameters["input_folder"] = str(tmp_path)
    binding.current_parameters["range_value"] = ""
    return window, binding, predicted


def test_a_watched_folder_batch_predicts_the_frames_written_after_it_started(tmp_path):
    for n in (1, 2):
        (tmp_path / f"run_{n:05d}.cbf").write_bytes(b"frame" * n)
    window, binding, predicted = _watched_batch(tmp_path, keep_up=False)
    try:
        assert window.gisaxsPredictKeepUpCheck.isEnabled()
        assert binding._predict_multi_files()["processing_started"]
        _wait(lambda: predicted == ["run_00001.cbf", "run_00002.cbf"], 10)
        assert binding._multifile_manager.is_running and binding._watching_frames()
        for n in (3, 4):
            (tmp_path / f"run_{n:05d}.cbf").write_bytes(b"frame" * n)
        _wait(lambda: len(predicted) == 4, 15)
        assert predicted[2:] == ["run_00003.cbf", "run_00004.cbf"]
        rows = binding._multifile_results_widget.get_all_results()
        assert [row.file_name for row in rows] == [f"run_{n:05d}.cbf" for n in (1, 2, 3, 4)]
        _wait(lambda: all(row.status == PredictStatus.COMPLETED for row in rows), 5)  # the signals of the last
        assert all(row.status == PredictStatus.COMPLETED for row in rows)
        window.gisaxsPredictWatchCheck.setChecked(False)  # unticked: the run ends with the frames taken
        _wait(lambda: not binding._multifile_manager.is_running, 5)
        _wait(lambda: not binding._watching_frames(), 5)
        assert not binding._watching_frames()
    finally:
        binding.stop_predictions()
        _wait(lambda: not binding._multifile_manager.is_running, 5)
        window.close()


def test_keeping_up_leaves_out_the_frames_written_while_the_model_was_busy(tmp_path):
    window, binding, predicted = _watched_batch(tmp_path, keep_up=True, delay=1.5)
    said = []
    binding.status_updated.connect(said.append)
    try:
        assert binding._predict_multi_files()["processing_started"]  # an empty folder is watched all the same
        (tmp_path / "run_00001.cbf").write_bytes(b"first")
        _wait(lambda: predicted == ["run_00001.cbf"] or binding._multifile_manager._given >= 1, 10)
        for n in range(2, 6):  # while the model is busy with the first
            (tmp_path / f"run_{n:05d}.cbf").write_bytes(b"frame" * n)
        _wait(lambda: "run_00005.cbf" in predicted, 20)
        assert predicted[0] == "run_00001.cbf" and predicted[-1] == "run_00005.cbf"
        assert len(predicted) < 5 and binding._keep_up.left_out == 5 - len(predicted)
        assert any("left out to keep up" in text for text in said)
    finally:
        binding.stop_predictions()
        _wait(lambda: not binding._multifile_manager.is_running, 5)
        window.close()


def test_an_empty_folder_can_be_watched_and_the_plan_says_what_the_batch_does(tmp_path):
    window, binding, predicted = _watched_batch(tmp_path, keep_up=True)
    try:
        window.gisaxsPredictWatchCheck.setChecked(False)
        binding._refresh_batch_plan_summary()
        assert not binding._input_ready()  # an empty folder, not watched: nothing to predict
        window.gisaxsPredictWatchCheck.setChecked(True)
        assert binding._input_ready()
        assert window.predictionBatchPlanSummary.text() == (
            "No frames in the folder yet: with Watch for new frames, the batch waits for them.")
        assert binding._predict_multi_files()["processing_started"]
        assert window.predictionBatchPlanSummary.text() == "Watching for new frames · 0 predicted"
        (tmp_path / "run_00001.cbf").write_bytes(b"x")  # a CBF: taken at the second look that sees it unchanged
        _wait(lambda: predicted == ["run_00001.cbf"], 10)
        _wait(lambda: window.predictionBatchPlanSummary.text() == "Watching for new frames · 1 predicted", 5)
        assert window.predictionBatchPlanSummary.text() == "Watching for new frames · 1 predicted"
        binding.stop_predictions()
        _wait(lambda: not binding._multifile_manager.is_running, 5)
        _wait(lambda: not binding._watching_frames(), 5)
        assert window.predictionBatchPlanSummary.text().startswith("1 files selected") or "1 file" in (
            window.predictionBatchPlanSummary.text())
    finally:
        binding.stop_predictions()
        _wait(lambda: not binding._multifile_manager.is_running, 5)
        window.close()


def test_the_canvas_follows_the_newest_result_of_a_running_batch_until_a_row_is_chosen(tmp_path):
    """Show the newest result: drawn at once, then at most once every ``SHOW_MS``; a row chosen during the run stays."""
    for n in (1, 2, 3):
        (tmp_path / f"run_{n:05d}.cbf").write_bytes(b"frame" * n)
    window, binding, predicted = _watched_batch(tmp_path, keep_up=False)

    def fake(path):
        predicted.append(Path(path).name)
        value = float(_number(path))
        return {"file": path, "stack_count": 1, "stack_files": [path],
                "prediction_data": {"parameters": np.array([value]), "parameter_names": ["R_nm"]}}

    binding._predict_single_file_for_batch = fake
    shown = []
    draw = binding._display_prediction

    def recorded(outputs, *, live=False):
        shown.append((float(outputs["parameters"][0]), live))
        draw(outputs, live=live)

    binding._display_prediction = recorded
    said = []
    binding.status_updated.connect(said.append)
    follow = window.predictionBatchFollowNewestCheck
    window.show()  # (nothing is drawn while the page is not in view)
    window.runtime.navigate("predict")
    try:
        assert follow.isChecked() and follow.text() == "Show the newest result"
        assert binding._predict_multi_files()["processing_started"]
        _wait(lambda: len(predicted) == 3, 10)
        _wait(lambda: shown and shown[-1] == (3.0, True), live_canvas.SHOW_MS / 1000 + 5)
        assert shown[-1] == (3.0, True) and all(live for _value, live in shown)
        assert len(shown) <= 2  # three results within a moment: drawn once or twice, the newest last
        assert window.predictionBatchCurrentFileLabel.text() == "run_00003.cbf (1 file)"
        tabs = window.gisaxsPredictImageShowTabWidget
        assert tabs.currentWidget() is window.predict2dImageTab  # the first result brought the canvas forward
        assert tabs.isVisible() and not window.predictionPlotPanel.empty_state.isVisible()  # no input preview: still shown
        assert not any("Predicted parameters" in text for text in said)  # the newest is not written in the log
        chosen = binding._multifile_results_widget.table_model.getResult(0)
        binding._on_multifile_result_selected(chosen)
        assert not follow.isChecked()  # the chosen row stays: the box says so
        assert shown[-1] == (1.0, False)
        assert any("Showing the chosen row" in text for text in said)
        (tmp_path / "run_00004.cbf").write_bytes(b"frame" * 4)
        _wait(lambda: len(predicted) == 4, 10)
        _wait(lambda: False, live_canvas.SHOW_MS / 1000 + 0.5)
        assert shown[-1] == (1.0, False)  # not followed any more
    finally:
        binding.stop_predictions()
        _wait(lambda: not binding._multifile_manager.is_running, 5)
        window.close()


def test_the_batch_table_shows_its_headers_and_states_in_the_interface_language():
    from PyQt5.QtCore import Qt

    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.prediction.presentation.result_models import PredictResultsTableModel
    from src.gimap.features.prediction.presentation.result_types import PredictResult

    _app()
    model = PredictResultsTableModel()
    model.addResult(PredictResult("/data/run_00001.cbf", "run_00001.cbf", PredictStatus.COMPLETED))
    headers = []
    model.headerDataChanged.connect(lambda *_args: headers.append(True))
    try:
        apply_language("zh")
        assert headers  # the view draws the headers again
        assert model.headerData(0, Qt.Horizontal, Qt.DisplayRole) == "输入帧"
        assert model.data(model.index(0, 2), Qt.DisplayRole) == "已完成"
        assert model.data(model.index(0, 0), Qt.DisplayRole) == "run_00001.cbf"  # a file name as it is
    finally:
        apply_language(DEFAULT_LANGUAGE)
    assert model.headerData(0, Qt.Horizontal, Qt.DisplayRole) == "Input Stack"
    assert model.data(model.index(0, 2), Qt.DisplayRole) == "Completed"
