"""A batch record replays frame sums, the common-x export and each warm-started peak fit."""

import csv
import json

import numpy as np

from src.gimap.app.reproduce import reproduce_record
from src.gimap.features.analyze.application import BatchChoices, CutRegion
from tests.test_batch_export import batch_page  # noqa: F401 - synthetic three-frame fixture
from tests.test_giwaxs_workspace import _settle


def _numbers(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(line for line in stream if not line.startswith("#"))
        headers = next(rows)
        return headers, np.array([[float(cell) if cell else np.nan for cell in row] for row in rows])


def test_batch_replays_sums_peak_fits_and_common_x_table(batch_page, tmp_path):
    page = batch_page
    page.view_model.add_region(CutRegion("Synthetic ring", (1.04, 1.16)))
    page.sum_spin.setValue(2)
    page.run_analysis()
    _settle(page, lambda: page.view_model.state.analysis.reduction.curve("region1") is not None)
    original = tmp_path / "batch"
    choices = BatchChoices(curves=("region1",), tables=True, fit="peaks", fit_profile="gaussian",
                           fit_start="previous", fit_curves=True, speed="gentle")
    page.run_batch(original, choices, stem="rings")
    record_path = original / "rings_batch.json"
    _settle(page, lambda: not page.batch_running() and record_path.exists())
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert [row["frames_summed"] for row in record["sums"]] == [2, 1]

    destination = tmp_path / "replayed"
    comparison = reproduce_record(record_path, destination)
    assert comparison["status"] == "identical"
    table = comparison["tables"]["region1"]
    headers, values = _numbers(destination / table["file"])
    assert headers == ["x", "frame_0001", "frame_0002"]
    _, original_values = _numbers(original / "rings_region1_frames.csv")
    assert values.shape == original_values.shape
    assert values.tobytes() == original_values.tobytes()

    for result in comparison["results"]:
        params = result["tables"]["peak_region1_parameters"]["columns"]
        assert params and all(column["status"] == "identical" for column in params.values())
        assert result["tables"]["peak_region1_curve"]["file"]


def test_built_series_replays_map_and_selected_profile(batch_page, tmp_path):
    page = batch_page
    page.build_series_map()
    _settle(page, lambda: not page.batch_running() and page._series_map is not None)
    page._series_row = 1
    for name, write in (("series_curve", page.export_series_frame_table), ("series_map", page.export_series_csv)):
        original = write(tmp_path / (name + ".csv"))
        record_path = original.with_suffix(".json")
        destination = tmp_path / name
        report = reproduce_record(record_path, destination)
        assert report["status"] == "identical"
        columns, reproduced = _numbers(destination / report["tables"][name]["file"])
        original_header, measured = _numbers(original)
        if name == "series_map":
            _, axis = _numbers(destination / report["tables"]["series_axis"]["file"])
            np.testing.assert_array_equal(axis[:, 0], np.array(original_header[1:], float))
        else:
            assert columns == ["x", "intensity"]
        assert measured.shape == reproduced.shape
        assert measured.tobytes() == reproduced.tobytes()
