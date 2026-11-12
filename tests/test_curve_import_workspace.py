"""Imported table columns keep their identity in the single page, project, and app entry."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog

from src.gimap.app.project import missing
from src.gimap.features.fitting.application.curve_sources import sources_from_table
from src.gimap.features.fitting.presentation import workspace_imports
from tests.test_fit_page import Q_NM, _page, _truth
from src.gimap.features.fitting.application.single_fit import evaluate


def _table(tmp_path):
    path = tmp_path / "shared sample.txt"
    intensity = evaluate(_truth(), Q_NM)
    np.savetxt(path, np.column_stack((Q_NM, intensity, intensity * 2)),
               header="q[nm^-1] I1 I2")
    return path


def test_selected_column_reopens_and_exports_its_physical_source(tmp_path):
    page = _page()
    again = _page()
    try:
        path = _table(tmp_path)
        sources = sources_from_table(page.view_model.inspect_curve_table(path), "shared_q")
        source = sources[1]
        assert page.open_curve(source.token)
        np.testing.assert_allclose(page.session.curve.q, Q_NM)
        np.testing.assert_allclose(page.session.curve.intensity, evaluate(_truth(), Q_NM) * 2)
        assert page.session.curve.sigma is None
        assert page.session.curve.name == source.display_name
        assert "::gimap-curve=" not in page.curve_chip.text()
        state = page.project_state()
        assert not again.apply_project_state(state)
        np.testing.assert_array_equal(again.session.curve.intensity, page.session.curve.intensity)
        assert again.session.curve.path == source.token
        assert again._stem() == "shared sample_I3"
        saved = tmp_path / "selected_fit.csv"
        assert again.export_data_dialog(saved)
        record = json.loads(saved.with_suffix(".json").read_text(encoding="utf-8"))
        origin = record["curve"]["source"]
        assert origin["path"] == str(path)
        assert origin["columns"]["intensity"] == 2 and origin["columns"]["sigma"] is None
        assert record["points"]["weighting"] == again._data().weighting
        assert not missing({"fitting": {"single": state}})
        path.unlink()
        assert missing({"fitting": {"single": state}}) == [str(path)]
    finally:
        page.dispose()
        again.dispose()


def test_changed_column_header_is_reported_when_reopening(tmp_path):
    page = _page()
    again = _page()
    try:
        path = _table(tmp_path)
        source = sources_from_table(page.view_model.inspect_curve_table(path), "shared_q")[1]
        assert page.open_curve(source.token)
        state = page.project_state()
        path.write_text(path.read_text(encoding="utf-8").replace("I1 I2", "I2 I1"), encoding="utf-8")
        notes = again.apply_project_state(state)
        assert notes and again.session.curve is None
        assert "column" in notes[0].lower()
    finally:
        page.dispose()
        again.dispose()


def test_workspace_entry_preserves_selection_and_unit_and_cleans_up(monkeypatch, tmp_path):
    selected = [str(tmp_path / "second.txt"), str(tmp_path / "first.dat")]
    sources = (object(), object())
    calls = []

    class Dialog:
        import_plan = SimpleNamespace(sources=sources, independent=True, q_unit="nm^-1")

        def __init__(self, paths, inspector, parent, **kwargs):
            calls.append((paths, inspector, parent, kwargs))

        def exec_(self):
            return QDialog.Accepted

        def dispose(self):
            calls.append("disposed")

        def deleteLater(self):
            calls.append("deleted")

    series = SimpleNamespace(running=False, open_imported_sources=lambda *args, **kwargs:
                             calls.append((args, kwargs)) or True)
    fit = SimpleNamespace(unit_setting=lambda: "nm", _remembered=lambda _key: "", _remember=lambda **kw: calls.append(kw))
    workspace = SimpleNamespace(series_page=series, fit_page=fit, context_container=object(),
                                view_model=SimpleNamespace(inspect_curve_table=object()),
                                show_context=lambda context: calls.append(context))
    monkeypatch.setattr(workspace_imports, "CurveImportDialog", Dialog)
    workspace_imports.CurveImportMixin.import_curves(workspace, selected)
    assert calls[0][0] == selected and calls[0][3] == {"q_unit": "nm"}
    assert calls[1] == ((sources,), {"independent": True, "q_unit": "nm"})
    assert calls[-3:] == ["insitu", "disposed", "deleted"]


def test_fixed_selection_missing_paths_are_real_and_deduplicated(tmp_path):
    path = tmp_path / "not_here.txt"
    project = {"fitting": {"series": {"source_mode": "selection", "folder": str(tmp_path / "unused"),
                                      "sources": [{"path": str(path)}, {"path": str(path)}]}}}
    assert missing(project) == [str(path)]
    project["fitting"]["series"]["sources"] = 42
    assert missing(project) == []  # restore reports the malformed selection; this is no missing-file list


def test_open_imported_curve_in_single_keeps_the_import_unit(tmp_path):
    from src.gimap.features.fitting.presentation.workspace import FittingWorkspace
    from tests.test_fit_series import _pages, _loaded

    single, series = _pages()
    try:
        path = tmp_path / "headerless.txt"
        np.savetxt(path, np.column_stack((Q_NM, evaluate(_truth(), Q_NM), evaluate(_truth(), Q_NM) * 2)))
        sources = sources_from_table(single.view_model.inspect_curve_table(path), "shared_q")
        assert single.unit_setting() == "angstrom"
        assert series.open_imported_sources(sources, independent=True, q_unit="nm")
        _loaded(series)
        single.set_model(_truth())
        single.session.q_range = (0.3, 1.2)
        single.session.excluded = {0.5}
        workspace = SimpleNamespace(series_page=series, fit_page=single, show_context=lambda _name: None)
        FittingWorkspace._frame_in_single(workspace, sources[1].token, None)
        assert single.session.curve.source_unit == "nm"
        np.testing.assert_allclose(single.session.curve.q, Q_NM)
        assert single.session.model == _truth()
        assert single.session.q_range == (0.3, 1.2) and single.session.excluded == {0.5}
    finally:
        series.dispose()
        single.dispose()


def test_damaged_mapping_does_not_crash_session_or_project_restore(tmp_path):
    from src.gimap.features.fitting.presentation.single.page import PREFERENCES_KEY

    page = _page()
    token = str(tmp_path / "sample.txt") + "::gimap-curve=broken"
    try:
        assert not page.open_curve(token)
        assert "Invalid curve source token" in page.status_label.text()
        page.preferences.set(PREFERENCES_KEY, {"curve": {"path": token}})
        page._restore()
        assert page.session.curve is None
        assert page.apply_project_state({"curve": {"path": token}})
        assert missing({"fitting": {"single": {"curve": {"path": token}}}}) == []
    finally:
        page.dispose()


def test_real_workspace_import_preview_then_model_setup(monkeypatch, tmp_path):
    from tests.test_project import _window
    from tests.test_fit_series import _loaded

    path = _table(tmp_path)
    window = _window()
    workspace = window.components.fitting_workspace

    def confirm(dialog):
        until = time.monotonic() + 20
        while not dialog._complete and time.monotonic() < until:
            QApplication.processEvents()
            time.sleep(0.005)
        assert dialog._complete and not dialog._errors
        assert dialog.layout_combo.currentData() == "shared_q"
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(workspace_imports.CurveImportDialog, "exec_", confirm)
    monkeypatch.setattr(workspace_imports.QFileDialog, "getOpenFileNames", lambda *_a, **_kw: ([str(path)], ""))
    try:
        workspace.fit_page.import_curves_button.click()
        series = workspace.series_page
        _loaded(series)
        assert workspace.context_stack.currentWidget() is series
        assert len(series.paths) == 2 and series.independent_samples()
        series._select(1)
        _loaded(series)
        assert series.to_single_button.isEnabled()
        series.to_single_button.click()
        assert workspace.context_stack.currentWidget() is workspace.fit_page
        curve = workspace.fit_page.session.curve
        assert curve.path == series.paths[1]
        np.testing.assert_allclose(curve.intensity, evaluate(_truth(), Q_NM) * 2)
        assert curve.source_unit == "nm" and curve.sigma is None
    finally:
        window.close()
