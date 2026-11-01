"""Every export record names the release that wrote it: ``software_version`` = ``shared.version.SOFTWARE``
(Physics and Methods, section 14.5; manual, chapter 12). The keys a record had before are kept: Analyze's
``software`` stays the name of its record format, ``GIMaP Analyze 1``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gimap.shared.version import SOFTWARE, __version__


def test_the_release_is_named_once_and_as_the_documents_say() -> None:
    assert SOFTWARE == f"GIMaP v{__version__}"


def test_analyze_records_keep_the_format_name_and_add_the_release() -> None:
    from src.gimap.features.analyze.application.settings_file import (
        AnalyzeSettings,
        settings_from_record,
        settings_record,
    )

    record = settings_record(AnalyzeSettings())
    assert record["software"] == "GIMaP Analyze 1" and record["software_version"] == SOFTWARE
    assert settings_from_record(record) == AnalyzeSettings()  # read back as before
    older = {key: value for key, value in record.items() if key != "software_version"}
    assert settings_from_record(older) == AnalyzeSettings()  # a file written before the key existed


def test_a_series_map_built_with_the_same_settings_by_another_release_is_not_out_of_date() -> None:
    from src.gimap.features.analyze.application.settings_file import AnalyzeSettings, settings_record
    from src.gimap.features.analyze.presentation.series_model import settings_key

    now = settings_record(AnalyzeSettings())
    assert settings_key({**now, "software_version": "GIMaP v0.1.0"}) == settings_key(now)
    assert "software_version" not in settings_key(now)


def test_the_batch_record_and_readme_name_the_release(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application.batch_export import BatchChoices
    from src.gimap.features.analyze.application.batch_report import BatchRun, batch_record, readme_text

    run = BatchRun(choices=BatchChoices(), destination=tmp_path, stem="run")
    record = batch_record(run, [])
    assert record["software_version"] == SOFTWARE and record["title"].startswith("GIMaP Analyze batch export")
    text = readme_text(BatchChoices(), batch="run", frame="f", frames=3, curves=[])
    assert text.splitlines()[:3] == ["GIMaP Analyze batch export - run", "3 frames.", f"Written by {SOFTWARE}."]


def test_xrr_and_2d_prediction_records_name_the_release(tmp_path: Path) -> None:
    from src.gimap.features.prediction.application.export_record import ExportPredictionRecord, PredictionRecordRequest
    from src.gimap.features.xrr.application.export_record import RECORD_FORMAT, xrr_export_record

    request = SimpleNamespace(result=SimpleNamespace(points=()), settings=None, geometry_sources={}, calibration=None)
    record = xrr_export_record(request, "series_xrr.csv", created="2026-10-13T00:00:00+00:00")
    assert record["format"] == RECORD_FORMAT and record["format_version"] == 1 and record["software_version"] == SOFTWARE
    prediction = ExportPredictionRecord.record(PredictionRecordRequest(path=tmp_path / "frame_prediction.json"))
    assert prediction["schema"] == "gimap.prediction-export/1" and prediction["software_version"] == SOFTWARE


def test_the_2d_prediction_batch_exports_name_the_release(tmp_path: Path) -> None:
    from src.gimap.features.prediction.application import ExportPredictionAscii, ExportPredictionJsonl
    from src.gimap.features.prediction.application.models import PredictionExportItem

    class Repository:
        def __init__(self):
            self.writes = []

        def write_text(self, path, content):
            self.writes.append((Path(path), content))
            return Path(path)

    item = PredictionExportItem("one.cbf", "C:/invented/one.cbf", 1, "12:00", 0.1, None,
                                {"prediction_data": {"h": np.array([1.0, 2.0])}})
    repository = Repository()
    ExportPredictionJsonl(repository).execute((item,), tmp_path, "20261101_000000")
    assert json.loads(repository.writes[0][1])["software_version"] == SOFTWARE
    ExportPredictionAscii(repository).execute((item,), tmp_path, "20261101_000000")
    assert f"# Software: {SOFTWARE}" in repository.writes[1][1].splitlines()[:4]


def test_fitting_records_name_the_release() -> None:
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, series_record
    from src.gimap.features.fitting.application.single_fit import FitData, FitModel, fit_record, new_component

    model = FitModel((new_component("sphere"),))
    data = FitData.prepare(np.linspace(0.1, 1.0, 20), np.linspace(10.0, 1.0, 20))
    record = fit_record(model, data, None, side="both", q_range=None, result=None)
    assert record["software_version"] == SOFTWARE and "model" in record
    series = series_record(model, SeriesSettings(), "C:/invented", "*_fit_input.dat", [])
    assert series["software"] == series["software_version"] == SOFTWARE  # ``software`` kept for older readers


def test_the_format_converter_names_the_release_in_every_record(tmp_path: Path) -> None:
    from src.gimap.features.format_converter.infrastructure.adapters.local_files import LocalSourceRepository
    from tests.test_format_converter_headers import _convert, _header_record, _stack

    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "stack.nxs", 2))
    report = _convert([source], tmp_path / "out", "CBF")
    assert report.failed == [] and len(report.succeeded) == 2
    side = json.loads((tmp_path / "out" / "conversion_metadata.json").read_text(encoding="utf-8"))
    assert [frame["software_version"] for frame in side] == [SOFTWARE, SOFTWARE]
    assert _header_record(tmp_path / "out" / "stack_000001.cbf")["software_version"] == SOFTWARE
    written = json.loads((tmp_path / "out" / "conversion_report.json").read_text(encoding="utf-8"))
    assert written["software_version"] == SOFTWARE and len(written["succeeded"]) == 2
