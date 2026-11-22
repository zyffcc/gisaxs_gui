"""What the files of a run and Fitting's intake say after a user test: CSV headers any code page reads, a report
that defines what it prints, the unit of q a file did not state, and the rows a curve file lost."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.gimap.app.headless_record import csv_text
from src.gimap.features.assistant.application import PipelineOptions
from src.gimap.features.assistant.application.gisaxs_report import ASYMMETRY_DEFINITION, _parameters
from src.gimap.features.fitting.application import LoadCurve, LoadCurveRequest, curve_intake
from src.gimap.features.fitting.application.curve_units import FROM_FILE, FROM_SETTING
from src.gimap.features.fitting.application.single_fit import Curve
from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

GALAXI = Path(__file__).parent / "data" / "external" / "gisaxs_galaxi"


@pytest.fixture(scope="module")
def galaxi(tmp_path_factory):
    """The public GALAXI frame with its documented geometry and αi, without the model fit."""
    from src.gimap.app.headless_assistant import analyse_frames

    out = tmp_path_factory.mktemp("round2") / "out"
    options = PipelineOptions(calibration=str(GALAXI / "galaxi_bornagain.poni"), incidence_deg=0.463,
                              technique="gisaxs", fit=False)
    [report] = analyse_frames([str(GALAXI / "galaxi_data.tif")], options, out, saved_profiles=False)
    return report, out / "galaxi_data"


def test_csv_headers_are_ascii_in_one_spelling() -> None:
    assert csv_text("qy (Å⁻¹), I, sigma") == "qy (A^-1), I, sigma"
    assert csv_text("q (nm⁻¹)") == "q (nm^-1)" and csv_text("χ (°), I, sigma") == "chi (deg), I, sigma"
    assert csv_text("q∥ (Å⁻¹)").isascii()


def test_the_cut_tables_of_a_run_are_read_in_any_code_page(galaxi) -> None:
    _report, folder = galaxi
    for name, axis in (("horizontal.csv", "qy"), ("vertical.csv", "qz")):
        raw = (folder / name).read_bytes()
        assert raw.isascii(), name
        header = [line for line in raw.decode("ascii").splitlines() if line.startswith("#")]
        assert header[-1] == f"# {axis} (A^-1), I, sigma"
        table = np.genfromtxt(folder / name, delimiter=",", encoding="gbk")  # a Chinese Windows' default
        assert table.ndim == 2 and table.shape[1] == 3 and np.isfinite(table[:, 0]).all()


def test_the_report_states_pixel_size_energy_and_what_the_asymmetry_is(galaxi) -> None:
    report, folder = galaxi
    text = (folder / "report.md").read_text(encoding="utf-8")
    geometry = text.split("## Geometry")[1].split("## ")[0]
    assert "- pixel size 172 µm, energy 9.25 keV" in geometry
    assert f"  - {ASYMMETRY_DEFINITION}" in text and ASYMMETRY_DEFINITION.startswith("asymmetry: ")
    record = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert record["geometry"]["critical_angle_deg"] is None  # none was given: not 0.0
    # ``geometry`` is what q was computed with (the refined symmetry axis); ``calibration`` is the file as read.
    symmetry = record["gisaxs"]["symmetry"]
    assert record["geometry"]["beam_center_px"][0] == pytest.approx(symmetry["x_px"])
    assert record["calibration"]["beam_center_px"][0] == pytest.approx(symmetry["initial_x_px"])


def test_the_fit_table_names_the_disorder_as_the_model_table_does() -> None:
    new = {"R": 9.4, "sigma_R": 0.3, "D": 53.4, "sigma_nn": 0.25, "sigma_D": 0.0997}
    assert _parameters({"components": [new]}, 0) == "R 9.4 nm, σR/R 0.3, D 53.4 nm, σnn/D 0.25"
    old = {"R": 9.4, "D": 53.4, "sigma_D": 0.4}
    assert _parameters({"components": [old]}, 0) == "R 9.4 nm, D 53.4 nm, σD/D 0.4"


def test_the_language_option_says_what_it_translates() -> None:
    from tools import gimap_agent

    text = " ".join(gimap_agent.parser()._subparsers._group_actions[0].choices["auto"].format_help().split())
    assert "section headings" in text and "nothing else is translated" in text


def test_a_unit_the_file_did_not_state_is_said_also_in_a_plausible_range() -> None:
    q_nm = np.linspace(0.05, 2.0, 50)  # 0.005–0.2 Å⁻¹ read in Å⁻¹: nothing in the numbers tells the units apart
    note = curve_intake.unit_note(q_nm, "angstrom", FROM_SETTING)
    assert note.kind == "q_unit" and note.template == curve_intake.UNIT_ASSUMED
    assert note.text.startswith("Unit of q assumed: Å⁻¹ (not stated in the file).") and "nm⁻¹" in note.text
    assert curve_intake.note_from_dict(note.as_dict()) == note
    assert curve_intake.unit_note(q_nm, "angstrom", FROM_FILE) is None
    assert curve_intake.unit_note(q_nm, "nm", FROM_SETTING).text.startswith("Unit of q assumed: nm⁻¹")


def test_rows_with_nan_are_counted_and_said(tmp_path) -> None:
    path = tmp_path / "curve.txt"
    rows = [f"{q:.4f} {100.0 / q:.3f}" for q in np.linspace(0.01, 0.2, 30)]
    rows[4], rows[9], rows[20] = "0.0500 nan", "nan 12.0", "0.1400 abc"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    outcome = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(path))
    assert outcome.succeeded and outcome.value.q.size == 27 and outcome.value.rows_left_out == 3
    curve = Curve.from_arrays(outcome.value.q, outcome.value.intensity, None, name="curve.txt",
                              rows_left_out=outcome.value.rows_left_out)
    notes = curve_intake.intake_notes(curve.q, unreadable=curve.rows_left_out)
    assert [note.kind for note in notes] == ["unreadable_rows"] and notes[0].values["count"] == 3
    clean = tmp_path / "clean.txt"
    clean.write_text("\n".join(f"{q:.4f} {1.0 / q:.3f}" for q in np.linspace(0.01, 0.2, 30)), encoding="utf-8")
    assert LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(clean)).value.rows_left_out == 0
