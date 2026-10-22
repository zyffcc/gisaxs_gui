"""Sputter workspace without its window: results, progress, requests and their folders, the geometry memory,
PyTorch's installer (pip's output and failures), the process tree, the bundled toolkit and the scan header.

Everything runs offline with invented data: no pip, no PyTorch, no real scan.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import (AnalyzeRequest, InstallError, ProgressTracker, ScanInfo,
                                                    StructureResult, ToolkitFailed, add_preset, make_preset,
                                                    matching_preset, overall_fraction, parse_progress, prefill,
                                                    reliability_report, remember, results_folder, scan_label)
from src.gimap.features.sputter.infrastructure import (BundledToolkit, CbfHeaderInspector, PipInstaller, PipReader,
                                                       StructureFolderReader, classify_failure, run_streaming)
from tests.sputter_fixtures import (PHYSICAL_ROWS, cbf_header, result_folder, scan_folder)


@pytest.fixture(autouse=True)
def _no_pip(monkeypatch):
    monkeypatch.setenv("GIMAP_SPUTTER_NO_INSTALL", "1")


# -- results -------------------------------------------------------------------------------------------------


def test_reader_parses_numbers_and_reasons(tmp_path):
    result = StructureFolderReader().read(result_folder(tmp_path), "window")
    assert [r["time_s"] for r in result.rows] == [0.05, 0.6, 1.1]
    assert result.rows[0]["D_nm"] is None and "no resolved peak" in result.rows[0]["D_reason"]
    assert result.counts("R") == {"A": 1, "B": 0, "C": 1, "S": 0, "T": 0, "L": 0, "P": 0, "missing": 1}
    assert result.series("D", "A").value == [9.0] and result.ts_length().value == [12.0]
    html = reliability_report(result)
    assert "±16%" in html and "±31%" in html and "flat-based hemisphere" in html and "Ag model" not in html


def test_reader_refuses_a_folder_without_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="structure_physical.csv"):
        StructureFolderReader().read(tmp_path)


def test_the_physical_path_is_read_first_with_its_flags_and_thickness(tmp_path):
    reader = StructureFolderReader()
    folder = result_folder(tmp_path, physical=True)
    assert reader.available(folder) == ["physical", "window"]
    result = reader.read(folder, "physical")
    assert [r["film"] for r in result.rows] == [False, False, False, True]
    assert result.rows[2]["path_changed"] is True and result.counts("R")["P"] == 3
    assert result.column("t_eq_nm", low="t_eq_lo_nm", high="t_eq_hi_nm").low == [0.12, 0.15]
    assert result.column("nominal_thickness_nm").value == [0.1, 0.2, 0.3, 0.4]
    assert result.film_spans() == [(2.0, 2.0)] and result.column("R_trend_nm").value == [1.9, 2.3, 2.5]
    html = reliability_report(result)
    for words in ("Physical path (default)", "1 of 4 windows differ", "Film from 2.0 s on this path (onset between "
                  "1.5 and 2.5 s)", "about 7 s", "Stage steps back", "island-volume equivalent",
                  "rate 0.2 nm/s", "not determined", "ratio 0.9", "an invented note of the model"):
        assert words in html, words


def test_a_cu_result_says_it_is_read_with_the_ag_model(tmp_path):
    folder = result_folder(tmp_path)
    result = StructureFolderReader().read(folder, "window")
    cu = StructureResult(result.folder, result.rows, dict(result.summary, metal="Cu"), "window")
    assert "Cu: approximated with the Ag model, for reference only." in reliability_report(cu)


def test_the_report_is_written_from_templates_the_interface_can_translate(tmp_path):
    from src.gimap.app.presentation.i18n.zh import ZH

    used = []

    def record(template, **values):
        used.append(template)
        return template.format(**values)

    reader = StructureFolderReader()
    result = reader.read(result_folder(tmp_path, physical=True), "physical")
    reliability_report(result, record)
    reliability_report(reader.read(tmp_path, "window"), record)
    words = [t for t in set(used) if any(c.isalpha() for c in re.sub(r"\{\w+\}", "", t).replace(" s", ""))]
    assert len(words) > 30 and sorted(t for t in words if t not in ZH) == []


# -- progress ------------------------------------------------------------------------------------------------


def test_progress_lines_are_parsed_and_weighted():
    p = parse_progress("SPUTTER_PROGRESS\tfeatures\t0.500\tdirect features")
    assert p is not None and p.stage == "features" and p.message == "direct features"
    assert 0.65 < overall_fraction(p) < 0.76
    assert parse_progress("some other output") is None
    assert overall_fraction(parse_progress("SPUTTER_PROGRESS\tdone\t1\tok")) == 1.0


def test_the_bar_moves_while_the_frames_are_read():
    tracker = ProgressTracker()
    assert tracker.feed("SPUTTER_PROGRESS\tintake\t0.000\treading") and tracker.fraction == 0
    assert tracker.feed("INTAKE_PLAN 4 1000") and tracker.frames == 1000
    tracker.feed("INTAKE_CHUNK intake 2 250 3.1")
    tracker.feed("INTAKE_CHUNK intake 2 250 3.1")  # the same chunk twice counts once
    assert abs(tracker.fraction - .55 * .25) < 1e-9
    assert tracker.message()[1] == {"stage": "Reading the frames", "done": 250, "total": 1000}
    assert tracker.feed("INTAKE_EXCLUDED frame_00007.cbf truncated file") and tracker.excluded[0].frame == "frame_00007.cbf"
    assert tracker.excluded[0].reason == "truncated file"
    assert not tracker.feed("a line of the model") and not tracker.feed("INTAKE_CHUNK broken")
    tracker.feed("SPUTTER_PROGRESS\tbinning\t1.000\tdone")
    assert tracker.fraction == pytest.approx(.65)


# -- the request and its folder ------------------------------------------------------------------------------


def test_request_problems_are_explained(tmp_path):
    bad = AnalyzeRequest(scan_folder=tmp_path / "missing", sdd_mm=None, alpha_i_deg=.4, metal="Fe")
    problems = " ".join(bad.problems())
    for words in ("scan folder", "energy", "distance", "Ag, Au or Cu"):
        assert words in problems
    assert AnalyzeRequest(scan_folder=Path(""), sdd_mm=1., alpha_i_deg=1., metal="Ag", energy_ev=1.).problems()
    ok = AnalyzeRequest(scan_folder=tmp_path, sdd_mm=1000., alpha_i_deg=.3, metal="Cu", energy_ev=9000.)
    assert ok.problems() == []  # Cu is allowed (read with the Ag model)
    assert AnalyzeRequest(scan_folder=tmp_path, sdd_mm=1000., alpha_i_deg=.3, metal="Ag", energy_ev=9000.,
                          rate_nm_s=-1.).problems()


def test_the_same_request_reuses_its_folder_and_another_geometry_gets_a_new_one(tmp_path):
    scan = scan_folder(tmp_path, "zz_invented_scan_00004")
    request = AnalyzeRequest(scan_folder=scan, sdd_mm=1000., alpha_i_deg=.3, metal="Ag", energy_ev=9000.)
    root = tmp_path / "results"
    first = results_folder(request, root)
    assert first.parent == root and first.name.startswith("zz_invented_scan-") and len(first.name.split("-")[-1]) == 8
    assert scan_label(scan) == "zz_invented_scan"
    same = AnalyzeRequest(scan_folder=scan, sdd_mm=1000., alpha_i_deg=.3, metal="Ag", energy_ev=9000., rate_nm_s=.2)
    assert results_folder(same, root) == first  # a deposition rate does not change the analysis
    moved = AnalyzeRequest(scan_folder=scan, sdd_mm=1000., alpha_i_deg=.31, metal="Ag", energy_ev=9000.)
    assert results_folder(moved, root) != first
    elsewhere = AnalyzeRequest(scan_folder=scan, sdd_mm=1000., alpha_i_deg=.3, metal="Ag", energy_ev=9000.,
                               results_root=tmp_path / "mine")
    assert results_folder(elsewhere, root).parent == tmp_path / "mine"
    # another model (a GIMaP with new weights) never reuses the networks' outputs of the old one
    assert results_folder(request, root, "invented-model-1") == results_folder(request, root, "invented-model-1")
    assert results_folder(request, root, "invented-model-2") != results_folder(request, root, "invented-model-1")


def test_a_new_model_gets_a_new_results_folder(tmp_path):
    from src.gimap.features.sputter.bootstrap import create_sputter_service

    home = _toolkit(tmp_path)
    kit = BundledToolkit(home, sys.executable)
    service = create_sputter_service(runner=kit, results_root=lambda: tmp_path / "results")
    request = AnalyzeRequest(scan_folder=scan_folder(tmp_path), sdd_mm=1000., alpha_i_deg=.3, metal="Ag",
                             energy_ev=9000.)
    first = service.results_folder(request)
    assert service.results_folder(request) == first
    (home / "MANIFEST.json").write_text('{"invented": "new weights"}', encoding="utf-8")
    assert service.results_folder(request) != first
    (home / "MANIFEST.json").unlink()  # a development copy: its files tell the version
    unversioned = service.results_folder(request)
    assert unversioned != first and kit.model_id().startswith("unversioned-")


# -- where the values come from ------------------------------------------------------------------------------

INFO = ScanInfo(Path("C:/invented/zz_scan_a"), frames=3, serial="99-0001", flat_field="FF_x99_E9000_T4500.tif",
                timestamp="2030-01-05T10:00:00", energy_kev_hint=9.0, energy_hint_source="header")
GEOMETRY = dict(energy_kind="energy", energy=9.5, sdd=1234.0, alpha=0.3, beam_col=100.5, beam_row=200.5, roll=0.1,
                metal="Au", polymer="PS", mask="", window=0.5, rate=0.2, t0=1.0)


def test_header_energy_and_last_used_values_say_where_they_come_from():
    filled = prefill(INFO, remember({}, "C:/invented/zz_other", GEOMETRY))
    assert filled.values["energy"] == 9.0 and filled.sources["xray"].kind == "header"
    assert filled.values["sdd"] == 1234.0 and filled.sources["sdd"].kind == "last"
    assert filled.values["metal"] == "Au" and filled.values["rate"] is None  # a rate belongs to its own folder
    empty = prefill(INFO, {})
    assert empty.sources["sdd"].kind == "missing" and empty.sources["beam"].kind == "auto"


def test_a_folder_keeps_its_own_values_and_the_header_hint_is_shown_beside_them():
    memory = remember({}, INFO.folder, GEOMETRY)
    filled = prefill(INFO, memory)
    assert filled.values["energy"] == 9.5 and filled.sources["xray"].kind == "folder"
    assert filled.sources["xray"].hint_kev == 9.0  # differs by more than 0.5 %
    assert filled.values["rate"] == 0.2 and filled.values["t0"] == 1.0


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
    assert other.values["mask"] == "" and other.values["polymer"] == "PS"  # the polymer is the last used
    same = prefill(INFO, remember({}, INFO.folder, used))
    assert same.values["roll"] == 0.45 and same.sources["roll"].kind == "folder"
    assert same.values["mask"] == "C:/invented/zz_mask.npy"
    preset = make_preset("invented beamtime", INFO, used, "2030-01-01", "2030-01-10")
    from_preset = prefill(INFO, add_preset(remember({}, "C:/invented/zz_other", GEOMETRY), preset))
    assert from_preset.values["roll"] == 0.45 and from_preset.sources["roll"].kind == "preset"


# -- PyTorch's installer -------------------------------------------------------------------------------------


def test_pip_output_is_read_into_stages():
    reader = PipReader()
    assert reader.read("Collecting torch<2.7,>=2.2 (from -r requirements-sputter.txt (line 4))").values == \
        (("package", "torch"),)
    assert reader.read("  Downloading torch-2.6.0-cp310-cp310-win_amd64.whl.metadata (28 kB)") is None
    step = reader.read("  Downloading torch-2.6.0-cp310-cp310-win_amd64.whl (204.1 MB)")
    assert step.stage == "download" and step.fields() == {"package": "torch", "size": "204.1 MB"}
    step = reader.read("Progress 102050000 of 204100000")
    assert step.stage == "download_progress" and step.fields()["percent"] == "50" and .44 < step.fraction < .46
    assert reader.read("  Using cached sympy-1.13.1-py3-none-any.whl (6.2 MB)").stage == "cached"
    step = reader.read("Installing collected packages: mpmath, sympy, torch")
    assert step.stage == "install" and not step.cancellable and reader.installing
    assert reader.read("Successfully installed torch-2.6.0").stage == "done"
    assert reader.read("Requirement already satisfied: numpy") is None


@pytest.mark.parametrize("lines, kind", [
    (["WARNING: Retrying (Retry(total=4)) after connection broken by 'NewConnectionError(...)'",
      "ERROR: Could not find a version that satisfies the requirement torch"], "network"),
    (["ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied"], "permission"),
    (["ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device"], "disk"),
    (["ERROR: No matching distribution found for torch<2.7,>=2.2"], "no_match"),
    (["D:\\invented\\python.exe: No module named pip"], "no_pip"),
    (["ERROR: something else"], "other"),
])
def test_pip_failures_are_classified(lines, kind):
    assert classify_failure(lines) == kind


class _Call:
    """A fake ``subprocess.run`` for the installer: pip's version, ensurepip and the import check."""

    def __init__(self, *, pip="24.0", torch=None, error=None):
        self.pip, self.torch, self.error = pip, torch, error
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append((command, kwargs.get("env")))
        if command[1:4] == ["-m", "pip", "--version"]:
            ok = self.pip is not None
            return SimpleNamespace(returncode=0 if ok else 1, stdout=f"pip {self.pip} from x" if ok else "",
                                   stderr="" if ok else "No module named pip")
        if command[1:3] == ["-m", "ensurepip"]:
            self.pip = "23.0"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        answer = {"numpy": "2.2.6"}
        if self.torch:
            answer["torch"] = self.torch
        elif self.error:
            answer["error"] = self.error
        else:
            answer["missing"] = "No module named 'torch'"
        import json
        return SimpleNamespace(returncode=0, stdout="GIMAP_VERIFY " + json.dumps(answer), stderr="")


def _installer(tmp_path, call, outcome, python=None):
    seen = {}

    def run(command, **kwargs):
        seen["command"], seen["env"] = command, kwargs["env"]
        seen["constraints"] = Path(command[command.index("-c") + 1]).read_text(encoding="utf-8")
        for line in outcome.get("lines", []):
            kwargs["on_line"](line)
        seen["can_stop"] = kwargs["can_stop"]()
        return SimpleNamespace(code=outcome.get("code", 0), tail=outcome.get("lines", []),
                               stopped=outcome.get("stopped", False))

    requirements = tmp_path / "requirements-sputter.txt"
    requirements.write_text("torch\n", encoding="utf-8")
    return PipInstaller(python or str(tmp_path / "python.exe"), requirements, run=run, call=call), seen


def test_the_installer_pins_numpy_never_installs_for_the_user_and_reports_progress(tmp_path, monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_NO_INSTALL")
    lines = ["Collecting torch", "  Downloading torch-2.6.0-cp310-cp310-win_amd64.whl (204.1 MB)",
             "Installing collected packages: torch", "Successfully installed torch-2.6.0"]
    python = tmp_path / ".gimap" / "env" / "Scripts" / "python.exe"
    installer, seen = _installer(tmp_path, _Call(pip="24.1.2"), {"lines": lines}, python=str(python))
    progress = []
    installer.install(progress.append, lambda: False)
    command = seen["command"]
    assert command[1:4] == ["-m", "pip", "install"] and "--no-user" in command and "--user" not in command
    assert command[command.index("-r") + 1].endswith("requirements-sputter.txt")
    assert seen["constraints"] == "numpy==2.2.6\n" and command[-2:] == ["--progress-bar", "raw"]
    assert seen["env"]["PYTHONNOUSERSITE"] == "1" and "PYTHONPATH" not in seen["env"]
    assert Path(seen["env"]["PIP_CACHE_DIR"]) == tmp_path / ".gimap" / "cache" / "pip"
    assert [p.stage for p in progress] == ["pip", "resolve", "download", "install", "done", "done"]
    assert seen["can_stop"] is False  # pip was writing the packages: Stop waits for its end


def test_a_python_without_pip_gets_it_from_ensurepip(tmp_path, monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_NO_INSTALL")
    call = _Call(pip=None)
    installer, seen = _installer(tmp_path, call, {"lines": ["Successfully installed torch"]})
    installer.install(lambda _p: None, lambda: False)
    assert any(c[1:3] == ["-m", "ensurepip"] for c, _env in call.commands) and "--progress-bar" not in seen["command"]


def test_installer_failures_say_why_and_give_the_command(tmp_path, monkeypatch):
    with pytest.raises(InstallError) as off:
        _installer(tmp_path, _Call(), {})[0].install(lambda _p: None, lambda: False)
    assert off.value.kind == "disabled" and "pip install -r" in str(off.value)
    monkeypatch.delenv("GIMAP_SPUTTER_NO_INSTALL")
    failing = {"code": 1, "lines": ["ERROR: Could not install packages due to an OSError: [Errno 28] No space"]}
    with pytest.raises(InstallError) as full:
        _installer(tmp_path, _Call(), failing)[0].install(lambda _p: None, lambda: False)
    assert full.value.kind == "disk" and "1.5 GB" in str(full.value) and "Errno 28" in full.value.details
    with pytest.raises(InstallError) as stopped:
        _installer(tmp_path, _Call(), {"stopped": True})[0].install(lambda _p: None, lambda: False)
    assert stopped.value.kind == "cancelled"
    with pytest.raises(InstallError) as other:
        _installer(tmp_path, _Call(), {"code": 3, "lines": ["ERROR: invented"]})[0].install(lambda _p: None, lambda: False)
    assert other.value.kind == "other" and "exit code 3" in str(other.value)


def test_verify_tells_missing_from_broken_pytorch(tmp_path):
    make = lambda call: PipInstaller(str(tmp_path / "python.exe"), tmp_path / "r.txt", call=call)  # noqa: E731
    assert make(_Call(torch="2.6.0+cpu")).verify().ready
    missing = make(_Call()).verify()
    assert not missing.ready and missing.problem == "missing" and missing.numpy_version == "2.2.6"
    broken = make(_Call(error="OSError: [WinError 126] invented.dll")).verify()
    assert broken.problem == "import_failed" and "WinError 126" in broken.detail


def test_an_installation_cut_off_while_writing_is_repaired_on_the_next_predict(tmp_path, monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_NO_INSTALL")
    writing = ["Collecting torch", "Installing collected packages: torch"]
    installer, seen = _installer(tmp_path, _Call(), {"lines": writing, "stopped": True})
    asked = []
    with pytest.raises(InstallError):  # closed once pip runs (``now``: even while it writes)
        installer.install(lambda _p: None, lambda: "now" if asked.append(1) or len(asked) > 1 else False)
    assert installer.interrupted()  # the window was closed while pip wrote the packages
    installer, seen = _installer(tmp_path, _Call(), {"lines": ["Successfully installed torch"]})
    installer.install(lambda _p: None, lambda: False, repair=True)
    assert "--force-reinstall" in seen["command"] and "--no-deps" in seen["command"]
    assert not installer.interrupted()
    stopped_early = _installer(tmp_path, _Call(), {"lines": ["Collecting torch"], "stopped": True})[0]
    asked = []
    with pytest.raises(InstallError):
        stopped_early.install(lambda _p: None, lambda: asked.append(1) or len(asked) > 1)
    assert not stopped_early.interrupted()  # nothing was written yet
    assert "--force-reinstall --no-deps" in installer.manual_command(repair=True)


def test_a_failing_ensurepip_says_there_is_no_pip(tmp_path, monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_NO_INSTALL")

    class Hanging(_Call):
        def __call__(self, command, **kwargs):
            if command[1:3] == ["-m", "ensurepip"]:
                raise subprocess.TimeoutExpired(command, 600)
            return super().__call__(command, **kwargs)

    with pytest.raises(InstallError) as failed:
        _installer(tmp_path, Hanging(pip=None), {})[0].install(lambda _p: None, lambda: False)
    assert failed.value.kind == "no_pip" and "ensurepip" in failed.value.details


def test_predict_repairs_a_pytorch_that_was_cut_off_and_explains_one_that_was_not(tmp_path):
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from tests.sputter_fixtures import FakeInspector, FakeInstaller, FakeRunner

    request = AnalyzeRequest(scan_folder=scan_folder(tmp_path), sdd_mm=1000., alpha_i_deg=.3, metal="Ag",
                             energy_ev=9000.)
    installer = FakeInstaller()
    installer.broken, installer.cut_off = True, True
    service = create_sputter_service(runner=FakeRunner(), installer=installer, inspector=FakeInspector(),
                                     results_root=lambda: tmp_path / "results")
    assert service.predict(request, lambda _e: None).variant == "physical" and installer.repairs == 1
    installer.broken = True  # broken without a cut-off installation: say what to do
    with pytest.raises(InstallError) as failed:
        service.predict(request, lambda _e: None)
    assert failed.value.kind == "import_failed" and "--force-reinstall --no-deps" in str(failed.value)


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


# -- the bundled toolkit -------------------------------------------------------------------------------------

ECHO = ("import json, sys, pathlib\n"
        "print('SPUTTER_PROGRESS\\tintake\\t0.5\\thalf', flush=True)\n"
        "print('a FutureWarning line', flush=True)\n"
        "print('plain line', flush=True)\n"
        "pathlib.Path(sys.argv[sys.argv.index('--out' if '--out' in sys.argv else '--folder') + 1]).mkdir("
        "parents=True, exist_ok=True)\n"
        "open(pathlib.Path(__file__).with_name('args.json'), 'w').write(json.dumps(sys.argv[1:]))\n"
        "sys.exit(0 if '--metal' in sys.argv or sys.argv[1] == 'physical' else 3)\n")


def _toolkit(tmp_path: Path) -> Path:
    home = tmp_path / "kit"
    for part in ("scripts", "sputtergisaxs", "experts/polymer", "experts/TS"):
        (home / part).mkdir(parents=True)
    (home / "scripts" / "sputter_struct_v1.py").write_text(ECHO, encoding="utf-8")
    (home / "MANIFEST.json").write_text("{}", encoding="utf-8")
    return home


def test_the_toolkit_gets_the_request_converts_the_beam_centre_and_reports_errors(tmp_path):
    import json

    home = _toolkit(tmp_path)
    kit = BundledToolkit(home, sys.executable)
    assert kit.problem() == ""
    request = AnalyzeRequest(scan_folder=tmp_path, sdd_mm=1000., alpha_i_deg=.3, metal="Cu", energy_ev=9000.,
                             beam_col=100.5, beam_row=200.5, rate_nm_s=.2, t0_s=3.)
    lines = []
    out = tmp_path / "out"
    kit.analyze(request, out, lines.append, lambda: False)
    args = json.loads((home / "scripts" / "args.json").read_text())
    value = lambda name: args[args.index(name) + 1]  # noqa: E731
    assert args[0] == "analyze" and float(value("--beam-col")) == 100.0 and float(value("--beam-row")) == 200.0
    assert value("--device") == "cpu" and 1 <= int(value("--workers")) <= 8 and value("--metal") == "Cu"
    assert float(value("--rate-nm-s")) == .2 and float(value("--t0-s")) == 3. and value("--out") == str(out)
    assert lines[0].startswith("SPUTTER_PROGRESS") and "plain line" in lines and not any("FutureWarning" in l for l in lines)
    record = json.loads((out / "gimap_request.json").read_text(encoding="utf-8"))
    assert record["request"]["alpha_i_deg"] == .3 and record["device"] == "cpu"
    assert record["digest"] == request.digest(kit.model_id()) and record["model_id"] == kit.model_id()
    assert record["scan_folder"] == str(tmp_path)
    kit.physical(out, None, None, lines.append)
    assert json.loads((home / "scripts" / "args.json").read_text()) == ["physical", "--folder", str(out)]
    kit.physical(out, .5, 2., lines.append)
    assert json.loads((home / "scripts" / "args.json").read_text())[-4:] == ["--rate-nm-s", "0.5", "--t0-s", "2.0"]
    (home / "scripts" / "sputter_struct_v1.py").write_text("import sys\nprint('boom', flush=True)\nsys.exit(2)\n",
                                                          encoding="utf-8")
    with pytest.raises(ToolkitFailed) as failed:
        kit.analyze(request, out, lines.append, lambda: False)
    assert "exit code 2" in str(failed.value) and "boom" in failed.value.details


def test_a_missing_toolkit_is_said_with_its_folder(tmp_path):
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.infrastructure import LOCAL_TOOLKIT

    assert LOCAL_TOOLKIT == Path(__file__).resolve().parents[1] / "utils" / "sputter_toolkit"
    service = create_sputter_service(runner=BundledToolkit(tmp_path / "nothing"))
    assert str(tmp_path / "nothing") in service.toolkit_problem() and "Update GIMaP" in service.toolkit_problem()


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


# -- the sketch ----------------------------------------------------------------------------------------------


def test_sketch_spacing_follows_the_reliability_rules():
    import math
    from src.gimap.features.sputter.domain import Sketcher, series_extent, storyboard_indices, window_sketch

    base = dict(time_s=5., module="polymer_embed", R_nm=2.0, R_grade="A", sigma_log=0.0)
    ok = window_sketch(dict(base, D_nm=20., D_grade="A", D_direct_check="consistent", D_corr_nm=21.))
    assert ok.spacing == 20. and ok.spacing_source.startswith("model D")
    assert math.isclose(ok.coverage, math.pi * 4 * 2 / (math.sqrt(3) * 400))
    bad = window_sketch(dict(base, D_nm=20., D_direct_check="outside simulated 90% range", D_corr_nm=40.))
    assert bad.spacing == 40. and bad.spacing_source == "direct 2π/q*" and bad.notes
    none = window_sketch(dict(base))
    assert none.spacing_source == "not measured" and none.coverage is None and none.spacing == 8.
    film = window_sketch(dict(time_s=50., module="TS_layer", L_TS_nm=70., TS_metal_fraction_conditional=.6))
    assert film.kind == "film" and film.spacing == 70.
    sk = Sketcher()
    assert sk.top_view(ok, 150.) and len(sk.top_view(window_sketch(dict(base, D_corr_nm=4.)), 150.)) > 1000
    assert sum(sum(r) for r in sk.film_field(film, 150., n=40)) > 0
    width, height = series_extent([ok, bad, none, film])
    assert 30 <= width <= 150 and height >= 3
    times = [window_sketch(dict(base, time_s=t, D_corr_nm=10. if t != 10. else None)) for t in (1., 2., 5., 10., 11., 100.)]
    picks = storyboard_indices(times, 3)
    assert picks[0] == 0 and picks[-1] == 5 and 3 not in picks  # 10 s has no measured spacing; 11 s is preferred


def test_a_film_of_the_physical_path_has_no_particles():
    from src.gimap.features.sputter.domain import Sketcher, parse_row, window_sketch

    film = window_sketch(parse_row(dict(PHYSICAL_ROWS[3], L_TS_nm="")))
    assert film.kind == "film" and film.L is None and "not determined" in film.notes[0]
    assert Sketcher().top_view(film, 100.) == [] and Sketcher().side_row(film, 100.) == []
    island = window_sketch(parse_row(PHYSICAL_ROWS[2]))  # the rule said film, the path says islands
    assert island.kind == "islands" and island.stage == "islands"


def test_median_smoothing_keeps_models_apart_and_flags_overlap():
    from src.gimap.features.sputter.domain import smoothed_rows, window_sketch

    rows = [dict(time_s=float(i), module="polymer_embed", R_nm=r, D_corr_nm=10.) for i, r in enumerate([2., 2., 9., 2., 2.])]
    rows.append(dict(time_s=5., module="TS_layer", L_TS_nm=60.))
    smooth = smoothed_rows(rows, half=2)
    assert smooth[2]["R_nm"] == 2.0 and smooth[2]["smoothed_windows"] == 5   # the single-window jump is gone
    assert smooth[5]["L_TS_nm"] == 60. and smooth[5]["smoothed_windows"] == 1  # TS window not mixed with islands
    tight = window_sketch(dict(time_s=1., module="polymer_embed", R_nm=4., D_corr_nm=6.))
    assert tight.coverage > .9 and any("overlap" in n for n in tight.notes)


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
    # The Sputter page installs PyTorch itself on the first Predict, from this file.
    assert (root / "requirements-sputter.txt").is_file()
    assert ".gimap/" in (root / ".gitignore").read_text(encoding="utf-8")
    update = (root / "Update_GIMaP.bat").read_text(encoding="ascii")
    assert "update_source.ps1" in update


def test_the_frame_readers_fit_into_the_free_memory(monkeypatch):
    import types
    from src.gimap.features.sputter.infrastructure import toolkit as toolkit_module

    monkeypatch.setattr(toolkit_module.os, "cpu_count", lambda: 20)
    def memory(available):
        fake = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=available))
        monkeypatch.setitem(sys.modules, "psutil", fake)
    memory(64e9)
    assert toolkit_module.default_workers() == 8  # cores decide
    memory(2.3e9)
    assert toolkit_module.default_workers() == 3  # (2.3 - 1.0) GB / 0.4 GB
    memory(0.5e9)
    assert toolkit_module.default_workers() == 1  # always at least one
    monkeypatch.setitem(sys.modules, "psutil", None)  # not installed: cores only
    assert toolkit_module.default_workers() == 8
