"""A colleague can refit a saved curve and check copied inputs against the original record."""

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialogButtonBox, QTableWidget

from src.gimap.app.reproduce import reproduce_record
from src.gimap.features.fitting.application.single_fit import parameter_text
from src.gimap.shared.reproduction import verified_path
from tests.test_fit_page import _curve_file, _page, _sphere, _wait


def test_refit_a_saved_curve_and_show_recorded_values_after_changing_range(tmp_path):
    source_dir = tmp_path / "acquisition"
    source_dir.mkdir()
    source = _curve_file(source_dir, signed=True)
    page = _page()
    try:
        assert page.open_curve(source, "positive")
        page.set_model(_sphere(5.2))
        page.set_range((0.15, 1.5))
        page.session.excluded = {0.4}
        page.run_fit("local")
        _wait(page)
        csv = tmp_path / "previous.csv"
        assert page.export_data_dialog(csv)
        record = csv.with_suffix(".json")
        moved = source.rename(tmp_path / source.name)

        page.session.clear_curve()
        assert page.open_curve_file(record)
        assert Path(page.session.curve.path) == moved
        assert page.session.side == "positive"
        assert page.session.q_range == (0.15, 1.5)
        assert page.session.excluded == {0.4}
        assert page.compare_saved_fit_button.isVisible() and page.method() == "local"
        saved_model = page.session.saved_fit.model
        page.set_range((0.5, 1.5))
        page.run_fit()
        _wait(page)
        assert page.session.saved_fit.model is saved_model and page.session.result_is_current()
        radii = []

        def read_comparison():
            dialog = QApplication.activeModalWidget()
            table = dialog.findChild(QTableWidget)
            for row in range(table.rowCount()):
                if table.item(row, 0).text().endswith(" R"):
                    radii.append(tuple(table.item(row, column).text() for column in (1, 2)))
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Close).click()

        QTimer.singleShot(0, read_comparison)
        page.compare_saved_fit_button.click()
        path = (0, "R")
        saved = page.session.saved_fit
        old_radius = parameter_text("R", saved_model.get(path).value, saved.errors.get("0:R"))
        new_radius = parameter_text("R", page.session.model.get(path).value, page.session.result.errors.get(path))
        assert len(radii) == 1 and radii[0][0].startswith(old_radius) and radii[0][1].startswith(new_radius)
        new = tmp_path / "current.csv"
        page.export_data_dialog(new)
        assert json.loads(new.with_suffix(".json").read_bytes())["parent_fit_record"] == {
            "path": str(record), "sha256": hashlib.sha256(record.read_bytes()).hexdigest()}
        assert page.open_curve_file(moved)
        assert page.session.saved_fit is None and page.compare_saved_fit_button.isHidden()
    finally:
        page.dispose()


def test_replayed_copy_requires_the_recorded_digest_and_reports_actual_path(tmp_path):
    page = _page()
    try:
        source = _curve_file(tmp_path)
        assert page.open_curve(source)
        exported = tmp_path / "curve.csv"
        page.export_data_dialog(exported)
        folder = tmp_path / "colleague"
        folder.mkdir()
        copied_record = Path(shutil.copyfile(exported.with_suffix(".json"), folder / "record.json"))
        copied_source = source.rename(folder / source.name)
        report = reproduce_record(copied_record, folder / "comparison")
        assert report["status"] == "identical"
        assert report["relocated_inputs"] == {str(source): {
            "path": str(copied_source), "sha256": hashlib.sha256(copied_source.read_bytes()).hexdigest()}}
        assert page.open_curve_file(copied_record)
        previous_curve, previous_fit = page.session.curve, page.session.saved_fit
        copied_source.write_bytes(b"different source")
        assert not page.open_curve_file(copied_record)
        assert "SHA-256" in page.status_label.text()
        assert page.session.curve is previous_curve and page.session.saved_fit is previous_fit
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            reproduce_record(copied_record, folder / "wrong")
        assert list((folder / "wrong").iterdir()) == []
    finally:
        page.dispose()


def test_copy_beside_record_resolves_foreign_path_and_does_not_hide_changed_original(tmp_path):
    sibling = tmp_path / "curve.dat"
    sibling.write_bytes(b"q I")
    reference = {"path": r"Q:\archived\curve.dat", "sha256": hashlib.sha256(sibling.read_bytes()).hexdigest()}
    assert verified_path(reference, tmp_path) == sibling
    original = tmp_path / "original" / sibling.name
    original.parent.mkdir()
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verified_path({**reference, "path": str(original)}, tmp_path)


def test_cancelling_record_verification_preserves_the_open_curve(qt_application, tmp_path, monkeypatch):
    from src.gimap.features.fitting.presentation.single import saved_fit

    page = _page()
    release = threading.Event()
    entered = threading.Event()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        record = tmp_path / "previous.json"
        page.export_data_dialog(record.with_suffix(".csv"))
        before, state = page.session.curve, page.view_model.state
        parse = saved_fit.read_saved_fit

        def reading(path):
            assert threading.current_thread() is not threading.main_thread()
            entered.set()
            assert release.wait(5)
            return parse(path)

        def cancel():
            if not entered.is_set():
                QTimer.singleShot(5, cancel)
                return
            progress = QApplication.activeModalWidget()
            progress.status.cancel_button.click()
            release.set()

        monkeypatch.setattr(saved_fit, "read_saved_fit", reading)
        QTimer.singleShot(0, cancel)
        assert not page.open_curve_file(record)
        assert page.session.curve is before and page.view_model.state is state
        assert page.session.saved_fit is None and "cancelled" in page.status_label.text()
    finally:
        release.set()
        page.dispose()
