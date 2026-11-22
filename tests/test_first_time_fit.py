"""First-time users can see what Predict will fit and save the curve and its parameters together."""

import json

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog

from tests.test_fit_page import _curve_file, _page, _predicting, _wait


def test_predict_describes_the_halves_before_running_and_results_expose_both_exports(tmp_path, monkeypatch):
    page = _page()
    try:
        _predicting(page, [6.0])
        page.open_curve(_curve_file(tmp_path, signed=True), "both_abs")
        assert page.step_rail.detail("fit") == "Ready to fit"
        assert page.fit_button.text() == page.fit_step_button.text() == "Predict parameters"
        assert "Predict uses the mean of both halves." in page.method_hint.text()
        assert page.session.side == "both"  # looking at the recommendation keeps the measured halves

        QTest.mouseClick(page.fit_button, Qt.LeftButton)
        _wait(page)
        assert page.session.side == "mean" and page.fit_button.text() == "Refine"
        assert "mean of both halves" not in page.method_hint.text()
        scroll = page.step_pages["results"]
        assert page.step_stack.currentWidget() == scroll
        QApplication.processEvents()
        for button in (page.export_data_step_button, page.export_plot_step_button):
            centre = button.mapTo(scroll.viewport(), button.rect().center())
            assert scroll.viewport().rect().contains(centre)

        target = tmp_path / "selected.csv"
        assert page.export_data_step_button.text() == "Save Curve and Parameters…"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
        QTest.mouseClick(page.export_data_step_button, Qt.LeftButton)
        record = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        assert record["points"]["side"] == "mean" and record["model"]["components"]
        assert target.read_text(encoding="utf-8").startswith("q")
    finally:
        page.dispose()
