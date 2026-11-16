"""Sputter workspace without its window: the geometry memory, the scan's results folder, the process tree and the
scan header.

Everything runs offline with invented data: no real scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import (ClientRequest, ScanInfo, SputterService, add_preset, make_preset,
                                                    matching_preset, prefill, remember, scan_label,
                                                    scan_results_folder)
from src.gimap.features.sputter.infrastructure import CbfHeaderInspector, run_streaming
from tests.sputter_fixtures import FakeInspector, cbf_header, scan_folder


# -- the scan's results folder ---------------------------------------------------------------------------------


def _request(scan: Path, **changes) -> ClientRequest:
    values = dict(scan_folder=scan, metal="Ag", substrate="Si", rate_nm_s=0.25, alpha_i_deg=0.3, sdd_mm=1000.0,
                  energy_ev=9000.0)
    values.update(changes)
    return ClientRequest(**values)


def test_a_scan_keeps_its_results_folder_and_another_geometry_gets_a_new_one(tmp_path):
    scan = scan_folder(tmp_path, "zz_invented_scan_00004")
    root = tmp_path / "results"
    first = scan_results_folder(_request(scan), root)
    assert first.parent == root and first.name.startswith("zz_invented_scan-") and len(first.name.split("-")[-1]) == 8
    assert scan_label(scan) == "zz_invented_scan"
    # the rate, the substrate and the mode are the request's own folder inside, not the scan's folder
    assert scan_results_folder(_request(scan, rate_nm_s=0.4, substrate="PS", mode="refine"), root) == first
    assert scan_results_folder(_request(scan, alpha_i_deg=0.31), root) != first
    service = SputterService(FakeInspector(), lambda: root)
    assert service.results_folder(_request(scan)) == first
    assert service.results_folder(_request(scan), tmp_path / "mine").parent == tmp_path / "mine"


def test_the_results_folder_is_the_one_scans_analysed_before_have(tmp_path):
    """The hash is the one the page computed before the earlier method was removed (its polymer and time window at
    their defaults, the version of its model): a scan analysed then finds its results again."""
    scan = scan_folder(tmp_path)
    request = _request(scan, beam_col=100.5, beam_row=200.5, sample_roll_deg=0.1)
    earlier = {"scan_folder": str(scan.resolve()), "sdd_mm": 1000.0, "alpha_i_deg": 0.3, "metal": "Ag",
               "energy_ev": 9000.0, "wavelength_nm": None, "polymer_type": "PSBMA", "beam_col": 100.5,
               "beam_row": 200.5, "sample_roll_deg": 0.1, "mask_file": None, "bin_seconds": 0.5,
               "model": "14199672853be149ebd9ef89875d02254e4356185ca0de97d4de3d5116731a90"}
    digest = hashlib.sha1(json.dumps(earlier, sort_keys=True, default=str).casefold().encode("utf-8")).hexdigest()[:8]
    assert scan_results_folder(request, tmp_path).name == f"zz_invented_scan-{digest}"


# -- where the values come from ------------------------------------------------------------------------------

INFO = ScanInfo(Path("C:/invented/zz_scan_a"), frames=3, serial="99-0001", flat_field="FF_x99_E9000_T4500.tif",
                timestamp="2030-01-05T10:00:00", energy_kev_hint=9.0, energy_hint_source="header")
GEOMETRY = dict(energy_kind="energy", energy=9.5, sdd=1234.0, alpha=0.3, beam_col=100.5, beam_row=200.5, roll=0.1,
                metal="Au", mask="")
OLDER = dict(GEOMETRY, polymer="PS", window=0.5, rate=0.2, t0=1.0, t_end=8.5)
"""Values an older page kept for a folder (its earlier method's polymer, time window and deposition fields)."""


def test_header_energy_and_last_used_values_say_where_they_come_from():
    filled = prefill(INFO, remember({}, "C:/invented/zz_other", GEOMETRY))
    assert filled.values["energy"] == 9.0 and filled.sources["xray"].kind == "header"
    assert filled.values["sdd"] == 1234.0 and filled.sources["sdd"].kind == "last"
    assert filled.values["metal"] == "Au"
    empty = prefill(INFO, {})
    assert empty.sources["sdd"].kind == "missing" and empty.sources["beam"].kind == "auto"


def test_a_folder_keeps_its_own_values_and_the_header_hint_is_shown_beside_them():
    memory = remember({}, INFO.folder, GEOMETRY)
    filled = prefill(INFO, memory)
    assert filled.values["energy"] == 9.5 and filled.sources["xray"].kind == "folder"
    assert filled.sources["xray"].hint_kev == 9.0  # differs by more than 0.5 %


def test_the_values_of_an_older_page_are_left_unread():
    """A memory written by the page with the earlier method: its geometry counts, its other values are not read."""
    for memory in (remember({}, INFO.folder, OLDER), remember({}, "C:/invented/zz_other", OLDER)):
        filled = prefill(INFO, memory)
        assert filled.values["sdd"] == 1234.0 and filled.values["metal"] == "Au"
        assert not {"polymer", "window", "rate", "t0", "t_end"} & set(filled.values)
    assert "rate" not in remember({}, INFO.folder, OLDER)["last"]  # a rate was never carried to another folder


def test_a_preset_matches_by_serial_flat_field_and_date():
    preset = make_preset("invented beamtime", INFO, dict(GEOMETRY, sdd=2222.0), "2030-01-01", "2030-01-10")
    memory = add_preset(remember({}, "C:/invented/zz_other", GEOMETRY), preset)
    filled = prefill(INFO, memory)
    assert filled.values["sdd"] == 2222.0 and filled.sources["sdd"].kind == "preset"
    assert filled.sources["sdd"].name == "invented beamtime"
    assert filled.sources["xray"].kind == "preset"  # the user's preset before a hint of a file name
    later = ScanInfo(INFO.folder, serial="99-0001", flat_field=INFO.flat_field, timestamp="2030-02-01T00:00:00")
    other = ScanInfo(INFO.folder, serial="99-0002", flat_field=INFO.flat_field, timestamp=INFO.timestamp)
    assert matching_preset(later, memory["presets"]) is None and matching_preset(other, memory["presets"]) is None
    narrow = make_preset("one day", INFO, GEOMETRY, "2030-01-05", "2030-01-05")
    assert matching_preset(INFO, add_preset(memory, narrow)["presets"])["name"] == "one day"


def test_a_flat_field_energy_is_only_shown_beside_the_field():
    """The E<eV> of a flat-field file name is the detector's calibration energy: it never fills the field."""
    flat = ScanInfo(INFO.folder, serial="99-0001", timestamp=INFO.timestamp, energy_kev_hint=9.0,
                    energy_hint_source="flat_field")
    empty = prefill(flat, {})
    assert empty.values.get("energy") is None and empty.sources["xray"].kind == "missing"
    assert empty.sources["xray"].hint_kev == 9.0 and empty.sources["xray"].hint_source == "flat_field"
    last = prefill(flat, remember({}, "C:/invented/zz_other", GEOMETRY))
    assert last.values["energy"] == 9.5 and last.sources["xray"].kind == "last"
    assert last.sources["xray"].hint_source == "flat_field"
    nexus = prefill(ScanInfo(INFO.folder, energy_kev_hint=9.0, energy_hint_source="nexus"), {})
    assert nexus.values["energy"] == 9.0 and nexus.sources["xray"].kind == "nexus"


def test_the_sample_roll_and_the_mask_are_not_carried_to_another_folder():
    used = dict(GEOMETRY, roll=0.45, mask="C:/invented/zz_mask.npy")
    other = prefill(INFO, remember({}, "C:/invented/zz_other", used))
    assert other.values["roll"] == 0.0 and other.sources["roll"].kind == "assumed"
    assert other.values["mask"] == ""
    same = prefill(INFO, remember({}, INFO.folder, used))
    assert same.values["roll"] == 0.45 and same.sources["roll"].kind == "folder"
    assert same.values["mask"] == "C:/invented/zz_mask.npy"
    preset = make_preset("invented beamtime", INFO, used, "2030-01-01", "2030-01-10")
    from_preset = prefill(INFO, add_preset(remember({}, "C:/invented/zz_other", GEOMETRY), preset))
    assert from_preset.values["roll"] == 0.45 and from_preset.sources["roll"].kind == "preset"


# -- the process tree ----------------------------------------------------------------------------------------

CHILD = ("import subprocess, sys, time\n"
         "grand = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
         "open(sys.argv[1], 'w').write(str(grand.pid))\n"
         "print('started', flush=True)\n"
         "time.sleep(60)\n")


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_stop_ends_the_whole_process_tree_without_waiting_for_a_line(tmp_path):
    script, pid_file = tmp_path / "child.py", tmp_path / "grandchild.pid"
    script.write_text(CHILD, encoding="utf-8")
    lines, stop_at = [], {}

    def should_stop():
        return bool(lines) and time.monotonic() >= stop_at.setdefault("t", time.monotonic() + .3)

    started = time.monotonic()
    outcome = run_streaming([sys.executable, str(script), str(pid_file)], on_line=lines.append, should_stop=should_stop)
    assert outcome.stopped and lines == ["started"] and time.monotonic() - started < 15
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(.1)
    assert not _alive(grandchild)


def test_stop_waits_while_it_may_not_stop(tmp_path):
    script = tmp_path / "quick.py"
    script.write_text("import time\nprint('one', flush=True)\ntime.sleep(.6)\nprint('two', flush=True)\n",
                      encoding="utf-8")
    lines = []
    outcome = run_streaming([sys.executable, str(script)], on_line=lines.append, should_stop=lambda: True,
                            can_stop=lambda: False)
    assert not outcome.stopped and outcome.code == 0 and lines == ["one", "two"]


def test_the_model_runs_in_the_console_python_of_this_environment():
    from src.gimap.features.sputter.infrastructure import console_python

    chosen = Path(console_python())
    assert chosen.is_file() and chosen.name.lower() != "pythonw.exe" and chosen.parent == Path(sys.executable).parent


# -- the scan header -----------------------------------------------------------------------------------------


def test_the_header_of_the_first_frame_fills_what_it_can(tmp_path):
    folder = scan_folder(tmp_path)
    info = CbfHeaderInspector().inspect(folder)
    assert info.frames == 3 and info.first_frame.endswith("_00001.cbf") and info.detector == "PILATUS 1M"
    assert info.serial == "99-0001" and info.pixel_um == pytest.approx(172.0) and info.exposure_s == pytest.approx(.03)
    assert info.period_s == pytest.approx(.04) and info.date == "2030-01-02"
    assert info.flat_field == "FF_x99_E9000_T4500.tif" and info.energy_kev_hint == 9.0
    assert info.energy_hint_source == "flat_field"  # the detector's calibration energy, not a beam energy
    # A scan folder whose frames are one level down (``<scan>/p2m``) is taken as that folder.
    parent = tmp_path / "zz_scan_root"
    scan_folder(parent, "p2m", frames=2)
    assert CbfHeaderInspector().inspect(parent).folder == parent / "p2m"
    assert CbfHeaderInspector().inspect(tmp_path / "zz_scan_root" / "nothing").problem


def test_a_wavelength_line_of_the_header_wins(tmp_path):
    folder = tmp_path / "zz_scan_w"
    folder.mkdir()
    (folder / "zz_scan_w_00001.cbf").write_bytes(cbf_header(extra="# Wavelength 1.0000 A\n").encode("latin-1"))
    info = CbfHeaderInspector().inspect(folder)
    assert info.energy_kev_hint == pytest.approx(12.398, abs=1e-3) and info.energy_hint_source == "header"


# -- the start script ----------------------------------------------------------------------------------------


def test_start_script_sets_up_a_source_folder_without_python_knowledge():
    root = Path(__file__).resolve().parents[1]
    bat = (root / "Start_GIMaP.bat").read_text(encoding="ascii")
    source = bat.index("setup_source_env.ps1"); portable_error = bat.index("Portable Python was not found")
    assert source < portable_error and 'if not exist "%PYTHON_EXE%" if exist "%~dp0main.py"' in bat
    setup = (root / "tools" / "setup_source_env.ps1").read_bytes()
    assert all(byte < 128 for byte in setup)
    text = setup.decode("ascii")
    assert '$UvVersion = "' in text and "Get-FileHash -Algorithm SHA256" in text and ".sha256" in text
    # Sputter needs no PyTorch any more: no list of its own packages.
    assert not (root / "requirements-sputter.txt").exists() and "requirements-sputter" not in text
    assert ".gimap/" in (root / ".gitignore").read_text(encoding="utf-8")
    update = (root / "Update_GIMaP.bat").read_text(encoding="ascii")
    assert "update_source.ps1" in update


def test_the_frame_readers_fit_into_the_free_memory(monkeypatch):
    import types
    from src.gimap.features.sputter.infrastructure import process as process_module

    monkeypatch.setattr(process_module.os, "cpu_count", lambda: 20)
    def memory(available):
        fake = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=available))
        monkeypatch.setitem(sys.modules, "psutil", fake)
    memory(64e9)
    assert process_module.default_workers() == 16  # cores decide: at most 16
    memory(2.5e9)
    assert process_module.default_workers() == 3  # (2.5 - 1.5) GB / 0.3 GB
    memory(1.0e9)
    assert process_module.default_workers() == 1  # always at least one
    monkeypatch.setitem(sys.modules, "psutil", None)  # not installed: cores only
    assert process_module.default_workers() == 16
    monkeypatch.setattr(process_module.os, "cpu_count", lambda: 6)
    assert process_module.default_workers() == 4  # two cores left free
    monkeypatch.setattr(process_module.os, "cpu_count", lambda: 2)
    assert process_module.default_workers() == 1
