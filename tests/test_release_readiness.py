"""Release contracts: valid counts, coordinates, portable outputs and project isolation."""
from pathlib import Path
import h5py
import numpy as np
import pytest

from src.gimap.features.xrr.application import XrrFrameRef, XrrSeriesSpec
from src.gimap.features.xrr.infrastructure.adapters import LocalXrrSeriesRepository
from src.gimap.features.xrr.domain.reflectometry import extract_circular_roi


def _frame(path, dtype):
    data = np.full((1, 32, 32), 100, dtype=dtype)
    data[0, 16, 16] = -1
    with h5py.File(path, 'w') as f:
        f.create_dataset('/entry/instrument/detector/data', data=data)
    return path


@pytest.mark.parametrize('dtype, expected, pixels', [('int16', 102300, 1023), ('float32', 102299, 1024)])
def test_xrr_count_gaps_are_excluded_but_negative_float_data_are_kept(tmp_path, dtype, expected, pixels):
    path = _frame(tmp_path / 'counts.nxs', dtype)
    frame = LocalXrrSeriesRepository().load_frame(XrrFrameRef(path, 0, 0))
    result = extract_circular_roi(frame.data, frame.invalid_mask, 15.5, 15.5, 32, 'sum')
    assert (result.intensity, result.valid_pixels) == (expected, pixels)


@pytest.mark.parametrize('units, scale', [('urad', 1e6), ('µrad', 1e6), ('μrad', 1e6), ('mrad', 1e3), ('rad', 1)])
def test_xrr_motor_units_are_converted(tmp_path, units, scale):
    path = _frame(tmp_path / 'angles.nxs', 'int16')
    with h5py.File(path, 'a') as f:
        angle = f.create_dataset('/entry/sample/omega', data=[np.radians(0.5) * scale])
        angle.attrs['units'] = units
    refs = LocalXrrSeriesRepository().discover(XrrSeriesSpec(path, angle_mode='nxs_dataset',
                                                           angle_dataset_path='/entry/sample/omega'))
    assert refs[0].theta_deg == pytest.approx(0.5)


def test_unknown_motor_units_are_not_silently_read_as_degrees(tmp_path):
    path = _frame(tmp_path / 'angles.nxs', 'int16')
    with h5py.File(path, 'a') as f:
        angle = f.create_dataset('/entry/sample/omega', data=[1.0])
        angle.attrs['units'] = 'encoder ticks'
    with pytest.raises(ValueError, match='Unsupported angle unit'):
        LocalXrrSeriesRepository().discover(XrrSeriesSpec(path, angle_mode='nxs_dataset',
                                                         angle_dataset_path='/entry/sample/omega'))


def test_prediction_outputs_do_not_depend_on_the_launch_directory(tmp_path, monkeypatch):
    from src.gimap.integrations.state.run_paths import fitting_run_directory
    monkeypatch.setenv('GIMAP_HOME', str(tmp_path / 'user'))
    monkeypatch.chdir(tmp_path)
    output = fitting_run_directory()
    assert output.parent == tmp_path / 'user' / 'AI_Fitting_Output'
    assert output.parent.is_dir()
    assert not (tmp_path / 'AI_Fitting_Output').exists()


def test_project_restores_sputter_and_clears_it_for_an_older_project(tmp_path):
    from tests.test_project import _window
    import json
    window = _window()
    try:
        page = window.components.sputter_page
        window.components.show_page('sputter')
        page.metal.setCurrentIndex(page.metal.findData('Ag'))
        page.alpha.setText('0.23')
        page.energy_value.setText('11.7')
        project = tmp_path / 'sample.gimap'
        assert window.menus._write_project(project)
        page.metal.setCurrentIndex(page.metal.findData('Au'))
        page.alpha.setText('0.9')
        assert window.menus.open_project(project)
        assert page.metal.currentData() == 'Ag'
        assert page.alpha.text() == '0.23'
        assert page.energy_value.text() == '11.7'
        saved = json.loads(project.read_text())
        del saved['sputter']
        project.write_text(json.dumps(saved))
        assert window.menus.open_project(project)
        assert page.alpha.text() == page._empty_project_state['fields']['alpha']
        assert page.result is None
        assert 'trainset' not in window.components.pages
        assert not hasattr(window.runtime, 'trainset')
    finally:
        window.close()


def test_model_manifest_fingerprint_changes_with_the_selected_bundle(tmp_path):
    from src.gimap.features.fitting.infrastructure.adapters.run_provenance import model_provenance
    manifest = tmp_path / 'conditional_fast_manifest_v2.json'
    manifest.write_text('{"files": {"model": "a"}}')
    first = model_provenance('v5', tmp_path)
    manifest.write_text('{"files": {"model": "b"}}')
    assert model_provenance('v5', tmp_path)['manifest_sha256'] != first['manifest_sha256']
    assert first['folder'] == str(tmp_path)


def test_old_scan_inspection_cannot_overwrite_a_new_project(tmp_path, monkeypatch):
    from threading import Event
    from tests.test_project import _window
    from src.gimap.features.sputter.application import ScanInfo
    window = _window()
    page = window.components.sputter_page
    finished = Event()

    def inspect(folder):
        assert finished.wait(5)
        return ScanInfo(folder, frames=4, energy_kev_hint=9.5)

    monkeypatch.setattr(page.service, 'inspect', inspect)
    try:
        page.set_scan_folder(tmp_path)
        page.apply_project_state({})
        finished.set()
        assert page.tasks.wait(5)
        assert page.scan_picker.path() == ''
        assert page.scan_info is None
        assert not page._inspecting
    finally:
        finished.set()
        page.tasks.wait(5)
        window.close()
