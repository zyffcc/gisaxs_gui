"""Manual fit controls, their plotted data and the separate inputs of Predict parameters."""

import numpy as np
from PyQt5.QtWidgets import QComboBox

from src.gimap.features.fitting.application.single_fit import evaluate
from tests.test_fit_page import _curve_file, _page, _predicting, _wait


def choose(editor, name, value):
    combo = editor.findChild(QComboBox, name)
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_manual_choices_drive_parameters_plot_and_undo(tmp_path):
    page = _page()
    try:
        path = tmp_path / "example.csv"
        q = np.linspace(0.03, 0.8, 25)
        np.savetxt(path, np.column_stack((q, 2 + np.exp(-q), q / 20)), delimiter=",",
                   header="q_nm^-1,I,dQ", comments="")
        assert page.open_curve(path)
        editor = page.model_editor
        editor.add_component("sphere")
        choose(editor, "fitSizes_0", "lognormal_median")
        choose(editor, "fitSizes_1", "schulz")
        assert editor.row((0, "R")).name.text() == "Median R (nm)"
        assert editor.row((0, "sigma_R")).name.text() == "σ(ln R)"
        assert editor.row((1, "sigma_R")).name.text() == "σR/R"
        choose(editor, "fitWeighting", "poisson")
        choose(editor, "fitResolutionMode", "file")
        assert editor.row(("globals", "q_resolution")) is None
        data = page.session.data()
        np.testing.assert_allclose(data.sigma, np.sqrt(data.intensity))
        request = page.figure_request(tmp_path / "example.svg")
        plotted = next(s for s in request.series if s.label == "model")
        np.testing.assert_array_equal(plotted.x, data.q)
        np.testing.assert_allclose(plotted.y, evaluate(page.session.model, data.q, dq=data.dq))
        page.undo()
        assert editor.findChild(QComboBox, "fitResolutionMode").currentData() == "constant"
        assert editor.row(("globals", "q_resolution")) is not None
    finally:
        page.dispose()


def test_missing_file_resolution_explained_and_predict_keeps_file_uncertainties(tmp_path):
    page = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        reference = page.session.data()
        jobs = _predicting(page, [5.0])
        choose(page.model_editor, "fitSizes_0", "schulz")
        choose(page.model_editor, "fitWeighting", "relative")
        choose(page.model_editor, "fitResolutionMode", "file")
        assert "dQ" in page.status_label.text()
        page.set_method("local")
        assert "dQ" in page.weighting_note.text()
        page.run_fit("local")
        assert not page.fit_running()
        page.run_fit("predict")
        _wait(page)
        np.testing.assert_array_equal(jobs.payloads[0]["q"], reference.q)
        np.testing.assert_array_equal(jobs.payloads[0]["sigma"], reference.sigma)
        np.testing.assert_array_equal(jobs.payloads[0]["intensity"], reference.intensity)
        assert jobs.payloads[0]["law"] == "number_gauss"
    finally:
        page.dispose()
