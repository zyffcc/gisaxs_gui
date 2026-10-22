"""1D Predict reads each curve file in the unit of q its header states (release 3).

Analyze writes its curve for Fitting, ``<stem>_fit_input.dat``, with q in Å⁻¹ and says so in its ``#`` header; the
window's setting (“q unit of files without one in their header”, nm⁻¹ by default) used to apply to every file, so
such a curve was read with q ten times too small. Now a file's own statement wins, the setting is for files that
state none, the window lists the unit of each file before the run and in the table, and the records of the run name
it. All data here is invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QLabel

from src.gimap.features.analyze.domain.models import Curve
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter
from src.gimap.features.fitting.application.workflow_v5 import FROM_FILE, FROM_SETTING
from src.gimap.features.fitting.infrastructure.adapters import workflow_v5
from src.gimap.features.fitting.infrastructure.adapters.curve_units import BOTH_UNITS, stated_q_unit
from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

Q_A = np.linspace(0.005, 0.05, 12)
"""q of the invented curves, in Å⁻¹ (one half: the job is asked for q > 0)."""


def _analyze_curve(folder: Path, name: str = "film_B7_fit_input.dat") -> Path:
    """A curve for Fitting written by Analyze's own writer (its header: ``(q in 1/A)``, ``# columns: qy (1/A) …``)."""
    curve = Curve("fit_input", "Horizontal cut I(qy)", Q_A, 100 * np.exp(-Q_A / 0.02), np.full(Q_A.size, 0.5),
                  np.full(Q_A.size, 5, dtype=np.int64), "qy (Å⁻¹)")
    return CsvCurveWriter().write_xy(curve, folder / name, {"source_file": "invented_frame.cbf"})


def _plain_curve(folder: Path, name: str = "other_program.dat") -> Path:
    """A curve with no word in it: q, I, σ."""
    path = folder / name
    np.savetxt(path, np.column_stack([Q_A, 100 * np.exp(-Q_A / 0.02), np.full(Q_A.size, 0.5)]))
    return path


def _run(tmp_path, monkeypatch, files, q_unit="nm^-1"):
    """The job with the fit replaced by a recorder of what it was given (no fitting, no network)."""
    from src.gimap.features.fitting.infrastructure.adapters import experimental_fit

    given = []

    def fit(_output, item, options, _report, _cancelled):
        given.append((item, options))
        return [dict(side=item["side"], best_log_rmse=0.1)]

    monkeypatch.setattr(experimental_fit, "fit_and_write", fit)
    output = tmp_path / "AI_Fitting_Output" / "v5_test"
    result = workflow_v5.run_workflow_job(
        dict(output_dir=str(output), files=[str(path) for path in files],
             options=dict(method="experimental", q_unit=q_unit, side="positive")),
        lambda *_args: None, lambda: False)
    return result, given, output


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# -- the header ---------------------------------------------------------------------------------------


def test_analyze_states_angstrom_in_the_header_of_its_curve_for_fitting(tmp_path) -> None:
    path = _analyze_curve(tmp_path)
    assert path.read_text(encoding="utf-8").startswith("# GIMaP Analyze fit input (q in 1/A)")
    assert LocalCurveRepository().stated_q_unit(path) == "angstrom"
    assert LocalCurveRepository().stated_q_unit(_plain_curve(tmp_path)) is None


@pytest.mark.parametrize("lines, unit", [
    (["q (1/A),I,sigma,pixels", "0.01,5,1,3"], "angstrom"),  # Analyze's curve tables
    (["q_(1/A) I sigma pixels", "0.01 5 1 3"], "angstrom"),  # … written as .dat
    (["q (Å⁻¹)\tI", "0.1\t2"], "angstrom"),
    (["Q(A-1) I(a.u.)", "0.1 2"], "angstrom"),
    (["q (1/Angstrom) I", "0.1 2"], "angstrom"),
    (["q_nm^-1,q_A^-1,I,sigma", "0.1,0.01,5,1"], "nm"),  # Fitting's table: the first column counts
    (["# columns: q (nm^-1)  I  sigma", "0.1 2 1"], "nm"),
    (["# q [1/nm]  I", "0.1 2"], "nm"),  # names in the last comment line
    (["# a note", "q I sigma", "0.1 2 1"], None),
    (["X\tY", ".1\t-2"], None),
    (["# source: C:/invented/sample A-1/frame.cbf", "0.1 2"], None),  # a name is no unit
    (["# GIMaP Analyze export (q in 1/A, angles in deg)", "chi (deg),I,sigma,pixels", "10,5,1,3"], None),  # no q
])
def test_the_unit_is_read_where_files_state_it(lines, unit) -> None:
    assert stated_q_unit(line + "\n" for line in lines) == unit


def test_a_header_that_states_both_units_is_refused() -> None:
    with pytest.raises(ValueError, match="both in 1/A and in 1/nm"):
        stated_q_unit(["# q in 1/A\n", "q (nm^-1) I\n", "0.1 2\n"])


# -- the run --------------------------------------------------------------------------------------------


def test_an_analyze_curve_is_read_in_angstrom_with_the_setting_on_nm(tmp_path, monkeypatch) -> None:
    result, given, output = _run(tmp_path, monkeypatch, [_analyze_curve(tmp_path)], q_unit="nm^-1")
    ((item, options),) = given
    np.testing.assert_allclose(item["q"], Q_A * 10)  # nm⁻¹ for the model: ten times the Å⁻¹ of the file
    assert options["q_unit"] == "A^-1"  # what solutions.json of this curve records
    read = dict(q_unit_read="A^-1", q_unit_from=FROM_FILE)
    assert result["records"][0]["status"] == "complete" and result["records"][0].items() >= read.items()
    assert result["candidates"][0].items() >= read.items()


def test_a_file_without_a_unit_follows_the_setting(tmp_path, monkeypatch) -> None:
    plain = _plain_curve(tmp_path)
    for setting, factor in (("nm^-1", 1.0), ("A^-1", 10.0)):
        result, given, _output = _run(tmp_path, monkeypatch, [plain], q_unit=setting)
        np.testing.assert_allclose(given[0][0]["q"], Q_A * factor)
        assert result["records"][0]["q_unit_read"] == setting and result["records"][0]["q_unit_from"] == FROM_SETTING


def test_the_records_of_a_mixed_run_name_the_unit_of_each_curve(tmp_path, monkeypatch) -> None:
    clash = tmp_path / "contradicting.dat"
    clash.write_text("# q in 1/A\nq (nm^-1) I\n0.1 2\n0.2 1\n", encoding="utf-8")
    files = [_analyze_curve(tmp_path), _plain_curve(tmp_path), clash]
    result, given, output = _run(tmp_path, monkeypatch, files, q_unit="nm^-1")
    np.testing.assert_allclose(given[0][0]["q"], Q_A * 10)
    np.testing.assert_allclose(given[1][0]["q"], Q_A)
    assert len(given) == 2  # the file that contradicts itself is not fitted
    expected = [("film_B7_fit_input.dat", "A^-1", FROM_FILE), ("other_program.dat", "nm^-1", FROM_SETTING),
                ("contradicting.dat", None, None)]
    for record in (_json(output / "request.json")["q_units"], _json(output / "fitting_run_summary.json")["q_units"]):
        assert [(entry["file"], entry["q_unit_read"], entry["q_unit_from"]) for entry in record] == expected
        assert record[2]["error"] == BOTH_UNITS
    request = _json(output / "request.json")
    assert request["options"]["q_unit"] == "nm^-1" and request["files"] == [str(path) for path in files]  # as sent
    batch = _json(output / "batch_summary.json")
    assert [(entry["status"], entry["q_unit_read"]) for entry in batch] == [
        ("complete", "A^-1"), ("complete", "nm^-1"), ("failed", None)]
    assert batch[2]["error"] == BOTH_UNITS
    rows = _json(output / "top20_candidates.json")
    assert [(row["file"], row["q_unit_read"], row["q_unit_from"]) for row in rows] == expected[:2]


# -- the window ------------------------------------------------------------------------------------------


def _dialog(**kwargs):
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    QApplication.instance() or QApplication([])
    return WorkflowV5Dialog(**kwargs)


def _close(dialog) -> None:
    from PyQt5.QtCore import QCoreApplication, QEvent

    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _add(dialog, monkeypatch, files) -> None:
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *_args: ([str(f) for f in files], "")))
    dialog._choose_files()


def _row(file: str, unit: str, origin: str) -> dict:
    return dict(file=file, side="positive", rank=1, combination="sphere", best_log_rmse=.05, signed_weighted_rms=1.0,
                best_source="experimental_physical", native_q=[1, 2], observed=[1, 1], sigma=[.1, .1],
                display_q=[1, 2], display_fit=[1, 1], q_unit_read=unit, q_unit_from=origin)


def test_the_window_says_the_unit_of_each_file_before_and_after_the_run(tmp_path, monkeypatch) -> None:
    dialog = _dialog(options=dict(q_unit="nm^-1"), stated_q_unit=LocalCurveRepository().stated_q_unit)
    try:
        _add(dialog, monkeypatch, [_analyze_curve(tmp_path), _plain_curve(tmp_path)])
        assert dialog.input_label.text().splitlines() == [
            "2 file(s): film_B7_fit_input.dat, other_program.dat",
            "q in Å⁻¹, as their header states: film_B7_fit_input.dat",
            "q in nm⁻¹, no unit in their header (Parameters…): other_program.dat",
        ]
        dialog.unit.setCurrentIndex(dialog.unit.findData("A^-1"))  # only the file without a unit follows
        assert dialog.input_label.text().splitlines()[1:] == [
            "q in Å⁻¹, as their header states: film_B7_fit_input.dat",
            "q in Å⁻¹, no unit in their header (Parameters…): other_program.dat",
        ]
        dialog.set_results([_row("film_B7_fit_input.dat", "A^-1", FROM_FILE)], str(tmp_path))
        assert dialog.table.item(0, 0).text() == "film_B7_fit_input.dat / + · Å⁻¹"
        assert dialog.table.item(0, 0).toolTip().endswith("q read in Å⁻¹: stated in the file")
    finally:
        _close(dialog)


def test_a_file_whose_unit_cannot_be_told_is_named(tmp_path, monkeypatch) -> None:
    clash = tmp_path / "contradicting.dat"
    clash.write_text("# q in 1/A\nq (nm^-1) I\n0.1 2\n", encoding="utf-8")
    dialog = _dialog(stated_q_unit=LocalCurveRepository().stated_q_unit)
    try:
        _add(dialog, monkeypatch, [clash])
        assert dialog.input_label.text().splitlines()[1] == (
            f"Not fitted, the unit of q cannot be read: contradicting.dat — {BOTH_UNITS}")
    finally:
        _close(dialog)


def test_the_setting_says_it_is_for_files_without_a_unit_and_the_window_is_chinese(tmp_path, monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    dialog = _dialog(options=dict(q_unit="nm^-1"), stated_q_unit=LocalCurveRepository().stated_q_unit)
    labels = lambda: [label.text() for label in dialog.settings_panel.findChildren(QLabel)]  # noqa: E731
    assert "q unit of files without one in their header" in labels()
    assert dialog.unit.toolTip().startswith("A file whose header states the unit of q is read in that unit")
    _add(dialog, monkeypatch, [_analyze_curve(tmp_path), _plain_curve(tmp_path)])
    dialog.set_results([_row("film_B7_fit_input.dat", "A^-1", FROM_FILE)], str(tmp_path))
    i18n.apply_language("zh")
    try:
        i18n.apply_to(dialog, "zh")
        assert "文件头未注明时的 q 单位" in labels()
        assert dialog.unit.toolTip().startswith("文件头注明了 q 单位的文件按注明的单位读取")
        assert dialog.input_label.text().splitlines() == [
            "2 个文件：film_B7_fit_input.dat, other_program.dat",
            "q 按文件头注明的 Å⁻¹：film_B7_fit_input.dat",
            "q 按 nm⁻¹，文件头未注明单位（参数…）：other_program.dat",
        ]
        assert dialog.table.item(0, 0).toolTip().endswith("q 按 Å⁻¹ 读取：文件中注明")
        i18n.apply_language("en")  # back while the window is open
        assert dialog.input_label.text().splitlines()[1] == "q in Å⁻¹, as their header states: film_B7_fit_input.dat"
        assert dialog.table.item(0, 0).toolTip().endswith("q read in Å⁻¹: stated in the file")
    finally:
        i18n.apply_language("en")
        i18n.apply_to(dialog, "en")
        _close(dialog)


def test_without_a_reader_the_window_says_the_rule() -> None:
    dialog = _dialog(options=dict(q_unit="nm^-1"))
    try:
        dialog.files = ["somewhere/curve.dat"]
        dialog._read_units()
        dialog._show_files()
        assert dialog.input_label.text().splitlines()[1] == (
            "q in the unit a file's header states, else in nm⁻¹ (Parameters…).")
    finally:
        _close(dialog)


def test_fitting_hands_the_header_reader_to_the_window() -> None:
    from src.gimap.features.fitting.application import LoadCurve
    from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import WorkflowV5BindingMixin

    QApplication.instance() or QApplication([])
    reader = LoadCurve(LocalCurveRepository()).stated_q_unit

    class Owner(WorkflowV5BindingMixin):
        main_window = None
        fitting_view_model = SimpleNamespace(stated_q_unit=reader, context=SimpleNamespace(jobs=None))

        def _ai_fitting_settings(self):
            return {}

        def _save_ai_fitting_settings(self, **_values):
            pass

    owner = Owner()
    owner.open_ai_fitting_workspace()
    try:
        assert owner._workflow_v5_dialog.stated_q_unit is reader
    finally:
        _close(owner._workflow_v5_dialog)
