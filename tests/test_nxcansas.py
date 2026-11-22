"""NXcanSAS (HDF5) curves: written by Analyze, Batch Export, the command line and Fitting's Save Data and Fit, read
by Fitting (also files of other programs). The structure checked here is the one of the NeXus application definition
NXcanSAS 1.1 (https://manual.nexusformat.org/classes/applications/NXcanSAS.html) that SasView's reader takes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gimap.features.analyze.application.nxcansas_export import against_q, nxcansas_name, sas_curves
from src.gimap.features.analyze.domain import Curve
from src.gimap.shared.nxcansas import (
    Instrument,
    SasCurve,
    is_nxcansas,
    q_unit,
    read_curve,
    read_nxcansas,
    write_nxcansas,
)

ROOT = Path(__file__).resolve().parents[1]
CBF = ROOT / "TestSAXSdata" / "jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00033.cbf"
requires_cbf = pytest.mark.skipif(not CBF.is_file(), reason="TestSAXSdata frames are not available")


def _text(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.ravel()[0]
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def structure_problems(path) -> list[str]:
    """What a file lacks of the NXcanSAS definition (the required groups, fields and attributes, and the units
    allowed); empty for a valid file."""
    problems = []
    with h5py.File(path, "r") as root:
        entries = [group for group in root.values()
                   if isinstance(group, h5py.Group) and _text(group.attrs.get("canSAS_class", "")) == "SASentry"]
        if not entries:
            return ["no SASentry"]
        if _text(root.attrs.get("default", "")) not in root:
            problems.append("root @default names no entry")
        for entry in entries:
            name = entry.name
            for key, value in (("NX_class", "NXentry"), ("version", "1.1")):
                if _text(entry.attrs.get(key, "")) != value:
                    problems.append(f"{name}@{key}")
            if _text(entry["definition"][()]) != "NXcanSAS":
                problems.append(f"{name}/definition")
            for field in ("title", "run"):
                if field not in entry:
                    problems.append(f"{name}/{field}")
            data = [group for group in entry.values()
                    if isinstance(group, h5py.Group) and _text(group.attrs.get("canSAS_class", "")) == "SASdata"]
            if not data:
                problems.append(f"{name}: no SASdata")
            for group in data:
                attrs = group.attrs
                if _text(attrs.get("NX_class", "")) != "NXdata" or _text(attrs.get("signal", "")) != "I":
                    problems.append(f"{group.name}@NX_class/signal")
                if _text(attrs.get("I_axes", "")) != "Q" or list(np.atleast_1d(attrs.get("Q_indices"))) != [0]:
                    problems.append(f"{group.name}@I_axes/Q_indices")
                mask = _text(attrs.get("mask", ""))
                if mask not in group or group[mask].shape != group["I"].shape:
                    problems.append(f"{group.name}@mask")
                if _text(group["Q"].attrs.get("units", "")) not in ("1/m", "1/nm", "1/angstrom"):
                    problems.append(f"{group.name}/Q@units")
                allowed = ("1/m", "1/cm", "m2/g", "cm2/g", "arbitrary")
                if _text(group["I"].attrs.get("units", "")) not in allowed:
                    problems.append(f"{group.name}/I@units")
                uncertainty = _text(group["I"].attrs.get("uncertainties", ""))
                if uncertainty and (uncertainty not in group or _text(group[uncertainty].attrs.get("units", ""))
                                    != _text(group["I"].attrs["units"])):
                    problems.append(f"{group.name}/I@uncertainties")
            for child in entry.values():
                if isinstance(child, h5py.Group) and _text(child.attrs.get("canSAS_class", "")) == "SASsample" \
                        and "name" not in child:
                    problems.append(f"{child.name}/name")
    return problems


def _curve(title="frame: Horizontal cut", n=40, sigma=True, **kwargs) -> SasCurve:
    q = np.linspace(-0.2, 0.2, n)
    intensity = 100.0 / (1.0 + (q / 0.02) ** 2)
    return SasCurve(title, q, intensity, np.sqrt(intensity) if sigma else None, **kwargs)


def test_a_written_file_reads_back_exactly_and_has_the_structure_of_the_definition(tmp_path) -> None:
    curves = [
        _curve(run="frame", sample="frame", instrument=Instrument(1.0332, 4.2, "PILATUS 2M", (172e-6, 172e-6), 0.4),
               process="GIMaP Analyze", description="test", record=json.dumps({"software": "x"})),
        _curve("frame: Vertical cut", n=25, sigma=False, q_unit="1/nm"),
    ]
    path = write_nxcansas(tmp_path / "frame_nxcansas.h5", curves)
    assert structure_problems(path) == [] and is_nxcansas(path)
    assert not list(tmp_path.glob(".*.part"))  # written under another name, renamed when complete
    back = read_nxcansas(path)
    assert [curve.title for curve in back] == ["frame: Horizontal cut", "frame: Vertical cut"]
    for written, read in zip(curves, back):
        assert np.array_equal(read.q, written.q) and np.array_equal(read.intensity, written.intensity)
        assert (read.sigma is None) == (written.sigma is None)
        if written.sigma is not None:
            assert np.array_equal(read.sigma, written.sigma)
        assert read.q_unit == written.q_unit and read.intensity_unit == "arbitrary"
    first = back[0]
    assert first.instrument == Instrument(1.0332, 4.2, "PILATUS 2M", (172e-6, 172e-6), 0.4)
    assert json.loads(first.record) == {"software": "x"} and first.process == "GIMaP Analyze"
    assert first.group == "/sasentry01/sasdata01" and not first.mask.any()
    with h5py.File(path, "r") as root:  # one SASentry per curve; αi as the sample's pitch, in degrees
        assert _text(root["sasentry01/sassample/pitch"].attrs["units"]) == "degree"
        assert _text(root["sasentry01/sasprocess01/gimap_record"].attrs["NX_class"]) == "NXnote"
        assert "sasinstrument" not in root["sasentry02"]  # nothing known, nothing written
    with pytest.raises(ValueError, match="no curve 3"):
        read_curve(path, 2)


def test_points_without_a_value_are_masked_and_the_units_of_q_are_read_in_every_spelling(tmp_path) -> None:
    curve = _curve()
    intensity = curve.intensity.copy()
    intensity[3] = np.nan
    path = write_nxcansas(tmp_path / "a.h5", [SasCurve("a", curve.q, intensity, curve.sigma)])
    read = read_curve(path)
    assert read.mask.tolist() == [index == 3 for index in range(curve.q.size)] and np.isnan(read.intensity[3])
    for text, unit in (("1/angstrom", "angstrom"), ("1/A", "angstrom"), ("A^-1", "angstrom"), ("1/Å", "angstrom"),
                       ("1/nm", "nm"), ("nm^-1", "nm"), ("1/m", "m"), ("", None), ("deg", None)):
        assert q_unit(text) == unit, text


def test_a_file_of_another_program_is_read_by_the_attributes_that_name_its_datasets(tmp_path) -> None:
    """As SasView writes it: no NX_class, fixed-length strings, I@uncertainty, a Q resolution, other names."""
    path = tmp_path / "sasview.h5"
    q = np.linspace(0.01, 0.3, 30)
    with h5py.File(path, "w") as root:
        entry = root.create_group("sasentry01")
        entry.attrs["canSAS_class"] = np.bytes_("SASentry")
        entry.create_dataset("title", data=np.array([np.bytes_("other program")]))
        data = entry.create_group("sasdata01")
        data.attrs.update({"canSAS_class": np.bytes_("SASdata"), "signal": np.bytes_("Intensity"),
                           "I_axes": np.bytes_("Qvalue"), "Q_indices": np.array([0])})
        data.create_dataset("Qvalue", data=q * 1e10).attrs["units"] = "1/m"
        signal = data.create_dataset("Intensity", data=1.0 / q)
        signal.attrs.update({"units": "1/cm", "uncertainty": "dI"})
        data.create_dataset("dI", data=0.1 / q).attrs["units"] = "1/cm"
        data.create_dataset("dQ", data=q * 0.0)
        root.create_group("not_an_entry").create_dataset("x", data=[1, 2])
    read = read_curve(path)
    assert read.title == "other program" and read.q_unit == "1/m" and np.allclose(read.q, q * 1e10)
    assert np.allclose(read.sigma, 0.1 / q) and read.mask is None and read.intensity_unit == "1/cm"
    # Fitting reads 1/m as 1/nm: × 1e-9
    from src.gimap.features.fitting.application import LoadCurve, LoadCurveRequest
    from src.gimap.features.fitting.application.curve_units import FROM_FILE
    from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

    loaded = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(path, "angstrom")).value
    assert loaded.q_source_unit == "nm" and loaded.q_unit_from == FROM_FILE and np.allclose(loaded.q, q * 10.0)
    assert loaded.observation["curve_source"]["q_converted"].startswith("Q in 1/m")


def test_analyze_writes_only_the_curves_against_q_and_says_which_q_each_is() -> None:
    x = np.linspace(-0.1, 0.1, 9)

    def curve(key, label):
        return Curve(key, f"{key} title", x, np.ones(9), np.ones(9), np.full(9, 3), label)

    metadata = {"geometry": {"wavelength_angstrom": 1.0, "distance_m": 2.0, "pixel_size_x_m": 1e-4,
                             "pixel_size_y_m": 1e-4, "incidence_deg": 0.2}, "detector_name": "D",
                "error_model": "poisson", "created": "2026-01-01T00:00:00+00:00"}
    curves = [curve("horizontal", "qy (Å⁻¹)"), curve("vertical", "qz (Å⁻¹)"), curve("sector_chi", "χ (°)"),
              curve("radial", "2θ (°)"), curve("box_qpar", "q∥ (Å⁻¹)")]
    assert [against_q(item) for item in curves] == [True, True, False, False, True]
    sas = sas_curves(curves, metadata, run="film_frame0003")
    assert [item.title for item in sas] == [
        "film_frame0003: horizontal title; Q is q_y (signed: both halves of the cut, not |q|)",
        "film_frame0003: vertical title; Q is q_z", "film_frame0003: box_qpar title; Q is q_par (in the sample plane)"]
    assert sas[0].instrument == Instrument(1.0, 2.0, "D", (1e-4, 1e-4), 0.2) and sas[0].q_unit == "1/angstrom"
    assert json.loads(sas[0].record)["curve"]["key"] == "horizontal" and nxcansas_name("s") == "s_nxcansas.h5"
    for model, options, phrase in (
        ("poisson", {"poisson_variance": "smoothed"}, "neighbouring bins"),
        ("pixel_scatter", {"pooled_scatter": True}, "fewer than ten pixels"),
        ("sigma_frame", {}, "supplied pixel standard deviations"),
    ):
        selected = dict(metadata, error_model=model, corrections=options)
        item = sas_curves(curves[:1], selected, run="frame")[0]
        assert phrase in item.description
        assert json.loads(item.record)["error_model"] == model


@pytest.fixture(scope="module")
def gisaxs_analysis():
    if not CBF.is_file():
        pytest.skip("TestSAXSdata frames are not available")
    from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame
    from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    profile = InstrumentProfile("P03 Pilatus", DetectorGeometry(172e-6, 172e-6, 4.2, 798.0, 1308.0, 1.0332, 0.4),
                                "PILATUS 2M", (1679, 1475))
    return AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))(AnalysisRequest(CBF))


@requires_cbf
def test_analyze_export_round_trips_through_fitting_with_identical_q_intensity_and_sigma(gisaxs_analysis, tmp_path):
    from src.gimap.features.analyze.application import ExportAnalysis, export_stem
    from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
    from src.gimap.features.fitting.application import InSituSourceFrame, LoadCurve, LoadCurveRequest
    from src.gimap.features.fitting.application.curve_units import FROM_FILE
    from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

    analysis = gisaxs_analysis
    export = ExportAnalysis(CsvCurveWriter())
    assert not any(path.suffix == ".h5" for path in export(analysis, tmp_path / "plain"))  # off by default
    written = export(analysis, tmp_path, nxcansas=True)
    path = tmp_path / f"{export_stem(analysis)}_nxcansas.h5"
    assert path in written and structure_problems(path) == []
    record = json.loads((tmp_path / f"{export_stem(analysis)}_analysis.json").read_text(encoding="utf-8"))
    assert path.name in record["files"]
    curves = read_nxcansas(path)
    assert [curve.title.split(": ", 1)[1].split(";")[0] for curve in curves] == ["Horizontal cut I(qy)", "Vertical cut I(qz)"]
    assert "Q is q_y (signed" in curves[0].title
    assert curves[0].instrument.wavelength_angstrom == pytest.approx(1.0332)
    assert curves[0].instrument.distance_m == pytest.approx(4.2) and curves[0].instrument.incidence_deg == pytest.approx(0.4)
    assert json.loads(curves[0].record)["input_sha256"][0]["file"] == CBF.name
    load = LoadCurve(LocalCurveRepository())
    for index, key in enumerate(("horizontal", "vertical")):
        source = analysis.reduction.curve(key)
        stored = curves[index]
        assert np.array_equal(stored.q, source.x) and np.array_equal(stored.intensity, source.intensity)
        assert np.array_equal(stored.sigma, source.sigma) and stored.q_unit == "1/angstrom"
        loaded = load.execute(LoadCurveRequest(Path(InSituSourceFrame(path, frame_index=index).token), "nm")).value
        assert loaded.q_source_unit == "angstrom" and loaded.q_unit_from == FROM_FILE  # the file's unit, not the setting
        keep = np.isfinite(source.x) & np.isfinite(source.intensity)
        assert np.array_equal(loaded.q, source.x[keep]) and np.array_equal(loaded.intensity, source.intensity[keep])
        assert np.array_equal(loaded.error, source.sigma[keep])
        assert loaded.observation["curve_source"]["format"] == "NXcanSAS"
    only = export.curves(analysis, ["vertical"], tmp_path / "plot", nxcansas=True)  # the curves of one plot
    assert [curve.title.endswith("Q is q_z") for curve in read_nxcansas(only[-1])] == [True]


@requires_cbf
def test_batch_export_writes_one_nxcansas_file_per_frame_in_its_own_folder(gisaxs_analysis, tmp_path) -> None:
    from src.gimap.features.analyze.application import BatchChoices, ExportAnalysis, export_frame, export_stem, output_names
    from src.gimap.features.analyze.application.batch_folder import chosen_kinds, file_kind
    from src.gimap.features.analyze.application.batch_job import batch_cost
    from src.gimap.features.analyze.application.batch_report import readme_text, written_kinds
    from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter

    choices = BatchChoices(curves=("horizontal",), tables=False, nxcansas=True)
    assert BatchChoices.from_dict(choices.to_dict()) == choices and choices.writes_anything
    written = export_frame(ExportAnalysis(CsvCurveWriter()), gisaxs_analysis, tmp_path, choices)
    stem = export_stem(gisaxs_analysis)
    names = output_names(choices, batch="run", frame=stem)
    assert written == [tmp_path / names["nxcansas"]] and names["nxcansas"] == f"nxcansas/{stem}_nxcansas.h5"
    assert [curve.title.split(": ", 1)[1].split(";")[0] for curve in read_nxcansas(written[0])] == ["Horizontal cut I(qy)"]
    assert "nxcansas" in chosen_kinds(choices) and file_kind(names["nxcansas"], "run") == "nxcansas"
    assert written_kinds(written, tmp_path, output_names(choices, batch="run", frame="<frame>")) == {"nxcansas"}
    assert "nxcansas/<frame>_nxcansas.h5" in readme_text(choices, batch="run", frame="f", frames=1,
                                                          curves=[("horizontal", "Horizontal cut I(qy)")])
    with_it = batch_cost(choices, frames=10, frame_pixels=1000, curve_points=500)[0]
    assert with_it > batch_cost(BatchChoices(tables=False), frames=10, frame_pixels=1000, curve_points=500)[0]


@requires_cbf
def test_the_command_line_writes_the_curves_as_nxcansas_when_asked(gisaxs_analysis, tmp_path) -> None:
    from src.gimap.app.headless_record import NXCANSAS_FILE, write_outputs
    from tools.gimap_agent import parser

    analysis = gisaxs_analysis

    def curve(key):
        found = analysis.reduction.curve(key)
        if found is None:
            return None
        return {"key": found.key, "title": found.title, "x": found.x, "y": found.intensity, "sigma": found.sigma,
                "pixels": found.pixels, "x_label": found.x_label, "region": dict(found.region)}

    session = SimpleNamespace(page=SimpleNamespace(view_model=SimpleNamespace(state=SimpleNamespace(analysis=analysis))),
                              curve=curve, preview_png=lambda *args, **kwargs: None, mask=None)
    for curve_format, csv_written in (("nxcansas", False), ("both", True), ("csv", False)):
        folder = tmp_path / curve_format
        names = {path.name for path in write_outputs(session, {"ok": True}, folder, curve_format=curve_format)}
        assert (NXCANSAS_FILE in names) == (curve_format != "csv")
        assert ("horizontal.csv" in names) == (csv_written or curve_format == "csv")
    assert structure_problems(tmp_path / "nxcansas" / NXCANSAS_FILE) == []
    assert read_nxcansas(tmp_path / "nxcansas" / NXCANSAS_FILE)[0].process == "GIMaP Analyze (command line)"
    arguments = parser().parse_args(["auto", "frame.tif"])
    assert arguments.curve_format == "csv"
    assert parser().parse_args(["call", "f.tif", "s.json", "--curve-format", "both"]).curve_format == "both"
    commands = next(action for action in parser()._actions if isinstance(action, argparse._SubParsersAction))
    assert "--curve-format" in commands.choices["auto"].format_help() and "nxcansas" in commands.choices["auto"].format_help()


def test_fitting_takes_every_curve_of_an_nxcansas_file_and_tells_it_from_detector_frames(tmp_path) -> None:
    from src.gimap.features.fitting.application import CURVE_SUFFIXES, UNSUPPORTED_CURVE, is_curve_file
    from src.gimap.features.fitting.application.curve_sources import source_stem, sources_from_table
    from src.gimap.features.fitting.application.series_frames import frame_name, is_frame
    from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

    path = write_nxcansas(tmp_path / "two.h5", [_curve("first"), _curve("second", n=12, q_unit="1/nm")])
    frames = tmp_path / "frames.h5"
    with h5py.File(frames, "w") as root:
        root.create_dataset("entry/data/data", data=np.zeros((2, 4, 4)))
    assert ".h5" in CURVE_SUFFIXES and "NXcanSAS .h5" in UNSUPPORTED_CURVE
    assert is_curve_file(path) and not is_curve_file(frames) and is_curve_file(tmp_path / "x.dat")
    assert not is_frame(str(path)) and is_frame(str(frames))
    preview = LocalCurveRepository().inspect_table(path)
    assert preview.curve_titles == ("first", "second") and preview.unit == "angstrom" and not preview.allow_shared_q
    sources = sources_from_table(preview, "single")
    assert [source.frame_index for source in sources] == [0, 1]
    assert frame_name(sources[1].token) == "two.h5 · curve 2" and source_stem(sources[1].token) == "two_curve2"
    assert LocalCurveRepository().stated_q_unit(sources[1].token) == "nm"
    with pytest.raises(ValueError, match="no NXcanSAS curve"):
        LocalCurveRepository().inspect_table(frames)


def test_the_fit_page_asks_which_curve_and_saves_data_and_fit_as_nxcansas(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QInputDialog

    from tests.test_fit_page import _page

    path = write_nxcansas(tmp_path / "two.h5", [_curve("first"), _curve("second", n=30)])
    asked = []

    def choose(_parent, _title, label, items, *_args):
        asked.append((label, list(items)))
        return items[1], True

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(choose))
    page = _page()
    try:
        assert page.open_curve_file(path)
        assert asked and asked[0][0] == "two.h5 holds 2 curves. Which one is to be fitted?"
        assert asked[0][1] == ["1: first", "2: second"]
        assert page.session.curve.name == "two.h5 · curve 2" and page.session.curve.q.size == 30
        assert "stated in the file" in page.curve_info.text() and "σ from the file" in page.curve_info.text()
        saved = page.export_data_dialog(str(tmp_path / "fit.h5"))
        assert saved and structure_problems(saved) == []
        data, model = read_nxcansas(saved)
        assert data.q_unit == model.q_unit == "1/nm" and data.sigma is not None and model.sigma is None
        assert data.title == "two_curve2: the points fitted" and np.array_equal(data.q, model.q)
        assert json.loads(data.record) == json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    finally:
        page.dispose()


def test_the_export_step_option_is_remembered_and_every_new_text_has_its_chinese() -> None:
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.analyze.presentation.preferences import (
        AnalyzePreferences,
        load_preferences,
        save_preferences,
    )
    from src.gimap.features.analyze.presentation.views.analyze_steps_view import NXCANSAS_TIP
    from src.gimap.features.analyze.presentation.views.batch_export_view import DATA_OUTPUTS
    from src.gimap.features.fitting.application import UNSUPPORTED_CURVE
    from src.gimap.features.fitting.presentation.curve_import import NXCANSAS_NOTE
    from src.gimap.features.fitting.presentation.single.page import CHOOSE_CURVE

    class Settings:
        def __init__(self):
            self.sections = {}

        def get_section(self, name):
            return self.sections.get(name, {})

        def update_section(self, name, values):
            self.sections.setdefault(name, {}).update(values)

        def save(self):
            pass

    settings = Settings()
    save_preferences(settings, AnalyzePreferences(nxcansas=True))
    assert load_preferences(settings, []).nxcansas is True and AnalyzePreferences().nxcansas is False
    row = next(item for item in DATA_OUTPUTS if item[0] == "nxcansas")
    for text in ("Also as NXcanSAS (.h5)", NXCANSAS_TIP, *row[1:], "NXcanSAS files", NXCANSAS_NOTE, CHOOSE_CURVE,
                 "Choose a Curve", UNSUPPORTED_CURVE,
                 "For the tables of all frames, the per-frame curves and the NXcanSAS files"):
        assert text in ZH, text


def test_sasview_reads_what_gimap_writes(tmp_path) -> None:
    """Only where SasView's loader (the ``sasdata`` package) is installed; GIMaP does not depend on it."""
    loader = pytest.importorskip("sasdata.dataloader.loader")
    path = write_nxcansas(tmp_path / "frame_nxcansas.h5", [
        _curve(run="frame", instrument=Instrument(1.0, 2.5, "D", None, 0.3)), _curve("frame: Vertical cut", n=10)])
    data = loader.Loader().load(str(path))
    assert [item.title for item in data] == ["frame: Horizontal cut", "frame: Vertical cut"]
    assert data[0].dy is not None and data[0].source.wavelength == pytest.approx(1.0)
