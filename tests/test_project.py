"""A GIMaP project: Analyze's frames and set-up, Single analysis and In-situ series, saved and reopened."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from tests.test_assistant_gui import _app
from tests.test_fit_series import _series

DATA = Path(__file__).parent / "data" / "external"
GALAXI = DATA / "gisaxs_galaxi" / "galaxi_data.tif"


def _window():
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    _app()
    geometry = DetectorGeometry(172e-6, 172e-6, 1.73, 597.1, 719.6, 1.34, 0.463)
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("GALAXI", geometry, None, (1043, 981))])
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
                         instrument_profiles=profiles)
    window = MainWindow(context)
    window.resize(1500, 950)
    window.show()
    end = time.monotonic() + 40
    while not window._initialization_completed and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    return window


@pytest.mark.skipif(not GALAXI.exists(), reason="GALAXI example not present")
def test_a_project_reopens_analyze_and_fitting_as_they_were(tmp_path) -> None:
    from src.gimap.features.analyze.domain import MaskShape

    window = _window()
    components = window.components
    analyze = components.analyze_page
    analyze.set_mode_choice("gisaxs")
    analyze.add_paths([str(GALAXI)])
    analyze.tasks.wait(120)
    analyze.view_model.add_mask_shape(MaskShape("rectangle", ((100, 100), (200, 180))))
    analyze.run_analysis()
    analyze.tasks.wait(120)
    folder = _series(tmp_path)
    workspace = components.fitting_workspace
    single = workspace.fit_page
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(FitModel((new_component("cylinder", radius=4.2),)))
    single.set_range((0.2, 1.5))
    first = single.session.all_points().q[30]
    single.session.excluded.add(float(f"{first:.9g}"))
    workspace.series_page.open_series(folder)
    workspace.series_page.every_spin.setValue(2)
    workspace.series_page.start_same.setChecked(True)
    from types import SimpleNamespace

    compare = components.compare_page
    compare.add_map(SimpleNamespace(x=np.linspace(0.05, 0.2, 40), image=np.random.default_rng(0).uniform(1, 2, (6, 40)),
                                    labels=tuple(f"f{i}" for i in range(6)), x_label="q (Å⁻¹)"), "map run")
    compare.add_folder(folder)
    path = window.menus._write_project(tmp_path / "sample.gimap") and Path(window.menus.project_path)
    assert (tmp_path / "sample.compare.npz").exists()  # the map from Analyze, next to the project
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["format"] == "gimap-project" and record["analyze"]["files"] == [str(GALAXI)]
    assert window.windowTitle() == "GIMaP — sample · galaxi_data.tif"  # the project and the frame shown
    before = analyze.project_state()["settings"]
    window.close()

    again = _window()
    assert again.menus.open_project(path)
    again.components.analyze_page.tasks.wait(120)
    assert again.menus.project_path == str(path)  # clearing Analyze while the project opens keeps it
    assert again.windowTitle() == "GIMaP — sample · galaxi_data.tif"
    reopened = again.components
    assert [str(p) for p in reopened.analyze_page.view_model.state.files] == [str(GALAXI)]
    after = reopened.analyze_page.project_state()["settings"]
    for record in (before, after):
        record["profile"].pop("updated_at", None)  # the profile is written to this computer's store again
    assert after == before, [(key, before[key], after.get(key)) for key in before if before[key] != after.get(key)]
    assert len(reopened.analyze_page.view_model.state.corrections.mask_shapes) == 1
    single = reopened.fitting_workspace.fit_page
    assert single.session.curve is not None and single.session.curve.name == "run_00001_fit_input.dat"
    assert single.session.model.components[0].family == "cylinder"
    assert single.session.model.get((0, "R")).value == pytest.approx(4.2)
    assert single.session.q_range == pytest.approx((0.2, 1.5)) and len(single.session.excluded) == 1
    series = reopened.fitting_workspace.series_page
    assert series.folder == str(folder) and series.every_spin.value() == 2 and series.start_same.isChecked()
    assert np.isfinite(single.session.all_points().q).all()
    assert [item.name for item in reopened.compare_page.series] == ["map run", folder.name]
    assert reopened.compare_page.series[0].rows == 6 and reopened.compare_page.series[1].paths
    again.close()


@pytest.mark.skipif(not GALAXI.exists(), reason="GALAXI example not present")
def test_save_project_after_clear_asks_for_a_file_and_keeps_the_project(tmp_path, monkeypatch) -> None:
    from src.gimap.app import menus as menus_module

    window = _window()
    menus, analyze = window.menus, window.components.analyze_page
    assert window.windowTitle() == "GIMaP"
    analyze.add_paths([str(GALAXI)])
    analyze.tasks.wait(120)
    assert window.windowTitle() == "GIMaP — galaxi_data.tif"
    project_a = tmp_path / "A.gimap"
    assert menus._write_project(project_a)

    asked = []
    monkeypatch.setattr(menus_module, "ask_save_json", lambda *args, **kwargs: asked.append(args) or "")
    assert menus.open_project(project_a)
    analyze.tasks.wait(120)
    QApplication.processEvents()
    assert menus.project_path == str(project_a) and window.windowTitle() == "GIMaP — A · galaxi_data.tif"
    assert menus.save_project() and asked == []  # the open project is saved without asking

    saved = project_a.read_text(encoding="utf-8")
    analyze.clear_files()
    if not hasattr(type(analyze), "filesCleared"):  # the Analyze package adds the signal
        menus._files_cleared()
    assert menus.project_path == "" and window.windowTitle() == "GIMaP"
    analyze.add_paths([str(GALAXI)])
    analyze.tasks.wait(120)
    QApplication.processEvents()
    assert window.windowTitle() == "GIMaP — galaxi_data.tif"
    assert menus.save_project() is False and len(asked) == 1  # Save As…, cancelled
    assert project_a.read_text(encoding="utf-8") == saved
    window.close()


def test_a_project_does_not_open_while_a_series_runs(tmp_path, monkeypatch) -> None:
    """Opening a project under a running In-situ series would leave the series going on under it, and Save
    would write its frames into that project: refused before anything is asked or applied."""
    from src.gimap.app import menus as menus_module
    from src.gimap.app import project

    window = _window()
    menus, workspace = window.menus, window.components.fitting_workspace
    other = menus._write_project(tmp_path / "other.gimap") and Path(menus.project_path)
    menus.project_path = str(tmp_path / "current.gimap")
    menus._sync_title()
    recent = list(window.app_context.preferences.get("recent_projects", []))
    folder = _series(tmp_path)
    single, series = workspace.fit_page, workspace.series_page
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(FitModel((new_component("cylinder", radius=4.2),)))
    series.open_series(folder)
    series.start()
    assert series.running
    before = series.project_state()
    warned, asked, applied = [], [], []
    monkeypatch.setattr(menus_module, "warn", lambda _parent, title, text: warned.append((title, text)))
    monkeypatch.setattr(menus_module, "ask_open_json", lambda *args, **kwargs: asked.append(args) or str(other))
    monkeypatch.setattr(project, "apply", lambda *args, **kwargs: applied.append(args) or [])
    try:
        assert menus.open_project(other) is False
        assert menus.open_project() is False and asked == []  # not even a file is asked for
        assert warned == [("Open Project", "Stop In-situ series first, then open the project.")] * 2
        assert applied == [] and series.running and series.project_state() == before
        assert menus.project_path == str(tmp_path / "current.gimap") and window.windowTitle() == "GIMaP — current"
        assert window.app_context.preferences.get("recent_projects", []) == recent
    finally:
        series.stop()
        end = time.monotonic() + 60
        while series.running and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.02)
    assert not series.running
    monkeypatch.undo()
    assert menus.open_project(other) and menus.project_path == str(other)  # with the series done it opens
    window.close()


def test_a_file_that_is_not_a_project_is_refused(tmp_path) -> None:
    from src.gimap.app import project

    path = tmp_path / "other.gimap"
    path.write_text(json.dumps({"format": "something else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a GIMaP project"):
        project.read(path)
