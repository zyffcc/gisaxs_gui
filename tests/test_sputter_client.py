"""Analyze of the Sputter workspace: the reader of the client package's output, its command line, its progress
lines and the service — for an in-situ scan, a porous substrate and a series of static samples (invented results
and a fake process: ``tests/sputter_client_fixtures.py``)."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.sputter.application import (CLIENT_INCOMPLETE, CLIENT_MISSING, CLIENT_NO_MAIN, CLIENT_PROBLEMS,
                                                    STATIC_PROBLEMS, STATIC_UNSUPPORTED, ClientFailed, ClientMissing,
                                                    ClientProgress, ClientRequest, ClientService, ClientStarted,
                                                    StaticDefaults, StaticRequest, StaticService, Stopped,
                                                    split_frames)
from src.gimap.features.sputter.infrastructure import (CHOICES_FILE, CLIENT_OVERRIDE, CLIENT_REQUEST_FILE, SERIES_FILE,
                                                       ClientFolderReader, ClientPackageRunner, StaticPackageRunner,
                                                       archive_folders, locate_package, static_defaults)
from src.gimap.features.sputter.infrastructure.client import IMPORT_MARKER, PACKAGE_MASK, STORAGE_FOLDER
from src.gimap.features.sputter.infrastructure.process import Outcome
from tests.sputter_client_fixtures import (AF, POROUS_WANTED, PROGRESS_LINES, Q, STATIC_LINES, FakeClientRunner,
                                           FakeStaticRunner, entry, package_folder, porous_windows, static_package,
                                           static_sample, static_samples, window, write_result, write_static)


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv(CLIENT_OVERRIDE, raising=False)


def _request(scan: Path, **changes) -> ClientRequest:
    scan.mkdir(parents=True, exist_ok=True)
    values = dict(scan_folder=scan, metal="Ag", substrate="Si", rate_nm_s=0.25, alpha_i_deg=0.4, sdd_mm=2500.0,
                  energy_ev=11000.0, beam_col=500.5, beam_row=900.5, sample_roll_deg=0.1)
    values.update(changes)
    return ClientRequest(**values)


# -- the reader ----------------------------------------------------------------------------------------------------


def test_the_reader_gives_windows_entries_and_images(tmp_path):
    reader = ClientFolderReader()
    result = reader.read(write_result(tmp_path / "out"))
    assert (result.status, result.scan, result.metal, result.family, result.mode) == \
        ("ok", "zz_invented_scan", "Ag", "zz_family", "quick")
    assert [w.time_s for w in result.windows] == [2.0, 4.0, 6.0, 8.0]
    assert [quantity.key for quantity in result.wanted][:3] == ["R_nm", "H_nm", "D_nm"]
    first, second, error, last = result.windows
    assert first.labels() == ["candidate 1", "candidate 2", "candidate 3", "refined"]
    assert (first.since_onset_s, first.nominal_nm, first.key) == (1.5, pytest.approx(0.4), "2")
    refined = first.entry("refined")
    assert (refined.kind, refined.rank, refined.weight, refined.fit_deviation) == ("refined", None, None, 2.1)
    assert refined.value("R_nm") == 1.2 and refined.value("gap_nm") == pytest.approx(2.4)
    assert first.entry("candidate 1").weight == 0.7 and first.entry("candidate 1").rank == 1
    # the entry shown first: the smooth evolution's, else the lowest fit deviation (pick)
    assert first.default_label() == "candidate 2" and second.default_label() == "refined"
    assert last.default_label() == "candidate 1"
    # a missing quantity is absent, not zero
    assert second.entry("candidate 2").value("H_nm") is None and "H_nm" not in second.entry("candidate 2").values
    assert last.entry("candidate 2").value("coverage") is None
    # a window the package could not use
    assert error.error == "no frames in this time window" and error.entries == () and error.default_label() == ""
    assert result.file("report_md") == tmp_path / "out" / "REPORT.md" and result.file("report_png") is None


def test_the_series_follow_the_choices_and_leave_gaps(tmp_path):
    result = ClientFolderReader().read(write_result(tmp_path / "out"))
    times, radii = result.series("R_nm")
    assert times == [2.0, 4.0, 6.0, 8.0] and radii[:2] == [1.5, 2.2] and math.isnan(radii[2]) and radii[3] == 3.0
    _times, heights = result.series("H_nm", {"4": "candidate 2"})
    assert heights[0] == pytest.approx(2.25) and math.isnan(heights[1]) and math.isnan(heights[3])
    _times, chosen = result.series("R_nm", {"2": "refined", "4": "not an entry", "99": "refined"})
    assert chosen[:2] == [1.2, 2.2]  # a choice that names no entry of its window is the default again
    assert result.nearest(4.9) == 1 and result.nearest(100.0) == 3


def test_the_images_of_a_window_are_read_as_they_are(tmp_path):
    reader = ClientFolderReader()
    result = reader.read(write_result(tmp_path / "out"))
    images = reader.patterns(result, result.windows[0])
    assert images.measured.shape == (AF.size, Q.size) and images.models.shape == (4, AF.size, Q.size)
    assert images.labels == ("candidate 1", "candidate 2", "candidate 3", "refined")
    assert np.allclose(images.q, Q) and np.allclose(images.af, AF)
    assert np.isnan(images.measured[5]).all() and images.measured[0, 0] == 0.0  # no data stays NaN, zero stays zero
    assert not np.allclose(images.model("refined"), images.model("candidate 1"), equal_nan=True)
    assert images.model("not an entry") is None
    with pytest.raises(FileNotFoundError, match="window 6"):
        reader.patterns(result, result.windows[2])


def test_a_file_name_that_is_a_path_is_never_followed(tmp_path):
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"x")
    item = window(2.0, [entry("candidate 1", 1.0, 1.0)], pick="candidate 1")
    item["arrays"] = "../../outside.npz"
    reader = ClientFolderReader()
    result = reader.read(write_result(tmp_path / "out", windows=[dict(item, arrays="w_2.npz")]))
    main = tmp_path / "out" / "main" / "main.json"
    record = json.loads(main.read_text(encoding="utf-8"))
    record["windows"][0]["arrays"] = "../../outside.npz"
    main.write_text(json.dumps(record), encoding="utf-8")
    result = reader.read(tmp_path / "out")
    assert result.windows[0].arrays == ""
    with pytest.raises(FileNotFoundError):
        reader.patterns(result, result.windows[0])


def test_arrays_that_do_not_fit_together_are_a_problem_said(tmp_path):
    reader = ClientFolderReader()
    result = reader.read(write_result(tmp_path / "out"))
    np.savez(tmp_path / "out" / "main" / "w_2.npz", q=Q, af=AF, measured=np.ones((3, 3)), models=np.ones((4, 3, 3)),
             labels=np.array(["a", "b", "c", "d"]))
    with pytest.raises(ValueError, match="window 2"):
        reader.patterns(result, result.windows[0])


def test_the_folder_chosen_may_be_the_output_its_main_folder_or_the_scans_results(tmp_path):
    reader = ClientFolderReader()
    scan_results = tmp_path / "zz_scan-12345678"
    older = write_result(scan_results / "iampred-aaaaaaaa")
    newer = write_result(scan_results / "iampred-bbbbbbbb")
    import os

    os.utime(older / "main" / "main.json", (1_000_000_000, 1_000_000_000))
    assert reader.locate(newer) == newer and reader.locate(newer / "main") == newer
    assert reader.locate(scan_results) == newer  # the newest result of the scan
    assert reader.locate(tmp_path) is None
    service = ClientService(FakeClientRunner(), reader)
    assert service.open(scan_results).folder == newer
    with pytest.raises(FileNotFoundError, match="main/main.json"):
        service.open(tmp_path)
    only_report = write_result(tmp_path / "old_build", with_main=False)  # a package that does not write main/
    with pytest.raises(FileNotFoundError, match="main/main.json"):
        service.open(only_report)
    (newer / "main" / "main.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        service.open(newer)


def test_choices_are_kept_next_to_the_result_and_only_where_they_differ_from_the_default(tmp_path):
    reader = ClientFolderReader(lambda: tmp_path / "user")
    service = ClientService(FakeClientRunner(), reader)
    result = service.open(write_result(tmp_path / "out"))
    first = result.windows[0]
    assert service.choices(result) == {}
    choices = service.with_choice(first, "refined", {})
    assert choices == {"2": "refined"}
    saved = service.save_choices(result, choices)
    assert saved == tmp_path / "out" / "main" / CHOICES_FILE
    record = json.loads(saved.read_text(encoding="utf-8"))
    assert record["choices"] == {"2": "refined"} and record["schema"] == 1 and "software_version" in record
    assert service.choices(service.open(tmp_path / "out")) == {"2": "refined"}
    assert service.with_choice(first, "candidate 2", choices) == {}  # the default is not a choice to keep
    assert service.with_choice(first, "not an entry", choices) == {}
    # a choice of an entry the window no longer has is dropped when the result is read
    saved.write_text(json.dumps({"choices": {"2": "gone", "4": "candidate 1", "77": "refined"}}), encoding="utf-8")
    assert service.choices(result) == {"4": "candidate 1"}


def test_choices_of_a_result_that_cannot_be_written_to_go_to_the_user_folder(tmp_path):
    reader = ClientFolderReader(lambda: tmp_path / "user")
    result = reader.read(write_result(tmp_path / "out"))
    (tmp_path / "out" / "main" / CHOICES_FILE).mkdir()  # a file of that name cannot be written
    saved = reader.save_choices(result, {"2": "refined"})
    assert saved.parent == tmp_path / "user" / STORAGE_FOLDER / "choices" and saved.is_file()
    assert reader.choices(result) == {"2": "refined"}
    with pytest.raises(OSError):
        ClientFolderReader().save_choices(result, {"2": "refined"})  # nowhere else to write to


# -- the package and its command -----------------------------------------------------------------------------------


def test_the_package_is_the_environments_then_the_chosen_then_the_program_folders(tmp_path, monkeypatch):
    chosen, bundled, named = (package_folder(tmp_path / name) for name in ("chosen", "bundled", "named"))
    nowhere = tmp_path / "nowhere"
    assert locate_package(None, nowhere).problem == CLIENT_MISSING
    found = locate_package(None, bundled)
    assert (found.home, found.source, found.problem) == (bundled, "bundled", "") and len(found.identity) == 64
    assert locate_package(chosen, bundled).source == "chosen" and locate_package(chosen, bundled).home == chosen
    monkeypatch.setenv(CLIENT_OVERRIDE, str(named))
    assert locate_package(chosen, bundled).source == "environment"
    monkeypatch.setenv(CLIENT_OVERRIDE, str(nowhere))  # a named folder that is no package is said, not skipped
    broken = locate_package(chosen, bundled)
    assert broken.problem == CLIENT_INCOMPLETE.format(folder=nowhere, missing="code/iam/iampred/analyze.py")
    monkeypatch.delenv(CLIENT_OVERRIDE)
    (chosen / "MANIFEST.json").write_text('{"invented": "another version"}', encoding="utf-8")
    assert locate_package(chosen, bundled).identity != found.identity
    unversioned = package_folder(tmp_path / "unversioned", manifest=False)
    assert locate_package(unversioned, nowhere).identity.startswith("unversioned-")


class _Process:
    """Stands for ``run_streaming``: keeps what it was asked and says the lines given."""

    def __init__(self, lines=PROGRESS_LINES, code=0, stopped=False):
        self.lines, self.code, self.stopped = lines, code, stopped
        self.calls: list = []

    def __call__(self, command, *, on_line, should_stop, env=None, cwd=None, **_more):
        self.calls.append((list(command), dict(env or {}), cwd))
        for line in self.lines:
            on_line(line)
        return Outcome(self.code, list(self.lines), self.stopped)


def _runner(tmp_path, process, package=None) -> ClientPackageRunner:
    return ClientPackageRunner(lambda: tmp_path / "user", python="zz_python", run=process,
                               chosen=package or package_folder(tmp_path / "package"), local=tmp_path / "nowhere",
                               workers=lambda: 3)


def test_the_command_and_the_environment_are_those_of_the_packages_launcher(tmp_path):
    process = _Process()
    runner = _runner(tmp_path, process)
    package = runner.package()
    request = _request(tmp_path / "zz invented scan")
    out = tmp_path / "results" / "iampred-12345678"
    lines = []
    run = runner.run(request, out, lines.append, lambda: False)
    assert (run.code, run.stopped) == (0, False) and lines == list(PROGRESS_LINES)
    command, env, cwd = process.calls[0]
    label = request.label()
    data = tmp_path / "user" / STORAGE_FOLDER / package.identity[:12] / "data" / PACKAGE_MASK
    assert label.startswith("gui_zz_invented_scan_") and len(label) == len("gui_zz_invented_scan_") + 8
    assert command == [
        "zz_python", "-u", "-m", "iampred.analyze", "--frames", str(tmp_path / "zz invented scan"), "--label", label,
        "--metal", "Ag", "--substrate", "Si", "--rate-nm-s", "0.25", "--alpha-i", "0.4", "--out", str(out),
        "--mode", "auto", "--workers", "3", "--data", str(data), "--", "--sdd-mm", "2500.0", "--energy-ev", "11000.0",
        "--beam-col", "500.0", "--beam-row", "900.0", "--roll", "0.1"]  # GIMaP's 500.5 is pixel-centre index 500
    home = tmp_path / "package"
    assert cwd == str(home / "code" / "iam")
    wanted = {"IAM_GIMAP": str(home / "code" / "gimap"), "IAM_DIST": str(home / "code" / "toolkit"),
              "IAM_DATA": str(data), "IAM_WORK": str(data.parent.parent / "work"), "IAM_RAW": str(home / "no_raw"),
              "IAMPRED_MODELS": str(home / "models"), "PYTHONPATH": str(home / "code" / "iam"), "TZ": "Europe/Berlin",
              "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1"}
    assert {key: env.get(key) for key in wanted} == wanted
    assert data.is_dir() and Path(env["IAM_WORK"]).is_dir()
    record = json.loads((out / CLIENT_REQUEST_FILE).read_text(encoding="utf-8"))
    assert record["label"] == label and record["package_id"] == package.identity and record["command"] == command
    assert record["scan_folder"] == str(tmp_path / "zz invented scan") and record["request"]["rate_nm_s"] == 0.25


def test_the_options_of_the_import_follow_the_request(tmp_path):
    process = _Process()
    runner = _runner(tmp_path, process)
    mask = tmp_path / "zz_mask.npy"
    np.save(mask, np.ones((4, 4), dtype=bool))
    request = _request(tmp_path / "scan", metal="Cu", substrate="PEO", rate_is_qcm=True, energy_ev=None,
                       wavelength_nm=0.11, beam_col=None, beam_row=None, sample_roll_deg=0.0, mask_file=mask,
                       mode="refine", onset_s=1.5, duration_s=30.0, film_nm=40.0)
    runner.run(request, tmp_path / "out", lambda _line: None, lambda: False)
    command, env, _cwd = process.calls[0]
    assert command[command.index("--metal") + 1] == "Cu" and command[command.index("--substrate") + 1] == "PEO"
    assert "--qcm-nm-s" in command and "--rate-nm-s" not in command and command[command.index("--mode") + 1] == "refine"
    tail = command[command.index("--") + 1:]
    assert tail == ["--sdd-mm", "2500.0", "--wavelength-nm", "0.11", "--auto-centre", "--roll", "0.0", "--mask", str(mask),
                    "--t-dep", "1.5", "--t-sput", "30.0", "--poly-t", "40.0"]
    assert Path(env["IAM_DATA"]).name.startswith("mask-") and len(Path(env["IAM_DATA"]).name) == 5 + 12
    # the film thickness belongs to a polymer film: Si has none
    runner.run(_request(tmp_path / "scan", film_nm=40.0), tmp_path / "out2", lambda _line: None, lambda: False)
    assert "--poly-t" not in process.calls[1][0]
    with pytest.raises(OSError, match="mask file cannot be read"):
        runner.run(_request(tmp_path / "scan", mask_file=tmp_path / "gone.npy"), tmp_path / "out3", lambda _l: None,
                   lambda: False)


def test_the_label_names_the_import_and_the_folder_the_whole_request(tmp_path):
    request = _request(tmp_path / "scan")
    same = _request(tmp_path / "scan")
    assert request.label() == same.label() and request.digest("p") == same.digest("p")
    for change in (dict(rate_nm_s=0.3), dict(rate_is_qcm=True), dict(substrate="TiO2"), dict(metal="Au"),
                   dict(beam_col=None, beam_row=None), dict(onset_s=2.0), dict(duration_s=20.0), dict(sdd_mm=2600.0)):
        assert _request(tmp_path / "scan", **change).label() != request.label(), change
    # the mode is not part of the import; it and the package's version are part of the output folder's name
    assert _request(tmp_path / "scan", mode="full").label() == request.label()
    assert _request(tmp_path / "scan", mode="full").digest("p") != request.digest("p")
    assert request.digest("p") != request.digest("another package")
    # a film thickness changes nothing on a substrate without a polymer film
    assert _request(tmp_path / "scan", film_nm=30.0).label() == request.label()
    assert _request(tmp_path / "scan", substrate="PS", film_nm=30.0).label() != \
        _request(tmp_path / "scan", substrate="PS").label()
    service = ClientService(FakeClientRunner(), ClientFolderReader())
    assert service.results_folder(request, tmp_path / "res").parent == tmp_path / "res"
    assert service.results_folder(request, tmp_path / "res").name == "iampred-" + request.digest("0" * 64)


def test_a_request_says_what_is_missing(tmp_path):
    assert _request(tmp_path / "scan").problems() == []
    rate, substrate, mode, duration, film = CLIENT_PROBLEMS
    assert _request(tmp_path / "scan", rate_nm_s=None).problems() == [rate]
    assert _request(tmp_path / "scan", rate_nm_s=0.0).problems() == [rate]
    assert _request(tmp_path / "scan", substrate="glass").problems() == [substrate]
    assert _request(tmp_path / "scan", mode="fast").problems() == [mode]
    assert _request(tmp_path / "scan", duration_s=0.0, film_nm=-1.0).problems() == [duration, film]
    several = _request(tmp_path / "scan", energy_ev=None, sdd_mm=None, beam_row=None).problems()
    assert len(several) == 3 and all("Give" in problem for problem in several)
    assert "scan folder" in ClientRequest(tmp_path / "gone", "Ag", "Si", 0.2, alpha_i_deg=0.4, sdd_mm=1.0,
                                          energy_ev=9000.0).problems()[0]


def test_an_import_that_was_cut_off_is_removed_and_a_finished_one_is_kept(tmp_path):
    cut_off = _Process(lines=("[1/4] intake ...",), code=None, stopped=True)
    runner = _runner(tmp_path, cut_off)
    request = _request(tmp_path / "scan")
    package = runner.package()
    data, _work = runner.roots(package, request)
    label = request.label()
    assert runner.run(request, tmp_path / "out", lambda _line: None, lambda: True).stopped
    assert not (data / (label + IMPORT_MARKER)).exists()
    (data / label / "intake").mkdir(parents=True)  # what the cut-off import left: the package would refuse the label
    (data / label / "intake" / "half.json").write_text("{}", encoding="utf-8")
    finished = _Process()
    runner = _runner(tmp_path, finished, package.home)
    seen = []
    runner._run_process = lambda command, **kw: (seen.append((data / label).exists()), finished(command, **kw))[1]
    runner.run(request, tmp_path / "out", lambda _line: None, lambda: False)
    assert seen == [False]  # removed before the package ran
    assert (data / (label + IMPORT_MARKER)).is_file()  # the import step said "done"
    (data / label).mkdir(parents=True)
    (data / label / "scan.json").write_text("{}", encoding="utf-8")
    runner.run(request, tmp_path / "out", lambda _line: None, lambda: False)
    assert (data / label / "scan.json").is_file()  # a finished import is reused, never removed


# -- progress --------------------------------------------------------------------------------------------------------


def test_the_steps_of_the_package_move_the_bar():
    progress = ClientProgress()
    assert progress.fraction == 0.0 and progress.message() == ("", {})
    assert progress.feed("[1/4] intake ...") and (progress.index, progress.total, progress.step) == (1, 4, "intake")
    assert progress.fraction == 0.0 and not progress.finished
    template, values = progress.message()
    assert template == "Step {index} of {total}: {stage}…" and values == {"index": 1, "total": 4,
                                                                           "stage": "Reading the frames"}
    assert not progress.feed("a plain line of the package") and not progress.feed("")
    assert progress.feed("      done in 12 s") and progress.finished and progress.fraction == pytest.approx(0.35)
    assert progress.feed("[2/4] predict ...") and progress.fraction == pytest.approx(0.35)
    assert not progress.feed("      alternative zz_family: done in 4 s")  # not the end of a step
    assert progress.feed("      FAILED (see zz.log) in 2 s") and progress.failed == ["predict"]
    assert progress.feed("status: failed at predict; report: None") and progress.status == "failed at predict"
    fresh = ClientProgress()
    assert fresh.feed("[1/4] intake ... reused the imported scan") and fresh.reused and fresh.done == ["intake"]
    assert fresh.message()[0].endswith("already done for this scan") and fresh.fraction == pytest.approx(0.35)
    for line in ("[2/4] predict ...", "      done in 5 s", "[3/4] report ...", "      done in 1 s"):
        fresh.feed(line)
    assert fresh.fraction == pytest.approx(0.85)
    fresh.feed("[4/4] present (main) ...")
    assert fresh.step == "present" and fresh.message()[1]["stage"] == "Drawing the measured and the model images"
    fresh.feed("status: ok; report: zz")
    assert fresh.fraction == 1.0
    other = ClientProgress()  # a package with other steps: the share is the number of steps done
    other.feed("[2/3] predict ...")
    assert other.fraction == pytest.approx(1 / 3)
    other.feed("      done in 1 s")
    assert other.fraction == pytest.approx(2 / 3)


# -- the service -----------------------------------------------------------------------------------------------------


def test_analyze_runs_the_package_once_and_shows_the_result_it_has_afterwards(tmp_path):
    runner = FakeClientRunner()
    service = ClientService(runner, ClientFolderReader())
    request = _request(tmp_path / "scan")
    events = []
    result = service.analyze(request, tmp_path / "res", events.append)
    assert result.folder == service.results_folder(request, tmp_path / "res") == runner.folders[0]
    assert isinstance(events[0], ClientStarted) and not events[0].reused and events[0].label == request.label()
    assert events[1:len(PROGRESS_LINES) + 1] == list(PROGRESS_LINES)
    again = []
    assert service.analyze(request, tmp_path / "res", again.append).folder == result.folder
    assert len(runner.requests) == 1 and len(again) == 1 and again[0].reused  # the same request: its result, no run
    service.analyze(_request(tmp_path / "scan", mode="refine"), tmp_path / "res", events.append)
    assert len(runner.requests) == 2 and runner.folders[1] != runner.folders[0]


def test_analyze_says_why_it_has_no_result(tmp_path):
    reader = ClientFolderReader()
    request = _request(tmp_path / "scan")
    with pytest.raises(ValueError, match="deposition rate"):
        ClientService(FakeClientRunner(), reader).analyze(_request(tmp_path / "scan", rate_nm_s=None), tmp_path, print)
    with pytest.raises(ClientMissing, match="not found"):
        ClientService(FakeClientRunner(missing=CLIENT_MISSING), reader).analyze(request, tmp_path, lambda _e: None)
    with pytest.raises(ClientFailed, match=r"\(failed at predict\)") as failed:
        ClientService(FakeClientRunner(status="failed at predict"), reader).analyze(request, tmp_path / "a",
                                                                                   lambda _e: None)
    assert "the invented reason of the failure" in failed.value.details  # the last lines of the failed step's log
    with pytest.raises(Stopped, match="reused by the next run"):
        ClientService(FakeClientRunner(wait_for_stop=True), reader).analyze(request, tmp_path / "b", lambda _e: None,
                                                                             lambda: True)
    old_build = ClientService(FakeClientRunner(with_main=False), reader)
    with pytest.raises(ClientFailed) as no_main:
        old_build.analyze(request, tmp_path / "c", lambda _e: None)
    assert str(no_main.value) == CLIENT_NO_MAIN.format(folder=old_build.results_folder(request, tmp_path / "c"))

    class NoAnalysis(FakeClientRunner):
        def run(self, request, out_folder, on_line, should_stop):
            from src.gimap.features.sputter.application import ClientRun

            return ClientRun(2, False, ("no network for Cu on Si. Supported: an invented list",))

    with pytest.raises(ClientFailed, match=r"\(exit code 2\)") as refused:
        ClientService(NoAnalysis(), reader).analyze(request, tmp_path / "d", lambda _e: None)
    assert "no network for Cu on Si" in refused.value.details  # the package's own last lines


# -- a porous substrate and a series of static samples ---------------------------------------------------------------


def test_the_reader_keeps_the_base_layer_of_a_solution_for_its_picture(tmp_path):
    """``base_nm`` is no shown quantity (no tile, not in ``wanted``) but every entry keeps it for the real-space picture."""
    from src.gimap.features.sputter.infrastructure import ClientFolderReader

    windows = [window(2.0, [entry("candidate 1", 3.0, 2.0, base_nm=1.5), entry("candidate 2", 4.0, 2.5)], smooth="candidate 1")]
    result = ClientFolderReader().read(write_result(tmp_path / "out", windows=windows))
    first, second = result.windows[0].entries
    assert first.value("base_nm") == 1.5 and second.value("base_nm") is None
    assert "base_nm" not in [quantity.key for quantity in result.wanted]


def test_the_real_space_picture_puts_the_grains_of_a_solution_at_their_size():
    from src.gimap.features.sputter.domain import (FIELD_DISTANCES, SIDE_DISTANCES, drawn_coverage, real_space,
                                                   side_grains, top_grains)

    values = {"R_nm": 4.0, "D_nm": 12.0, "width": 0.2, "H_nm": 7.0, "base_nm": 1.0}
    space = real_space(values)
    assert (space.radius, space.distance, space.base, space.grain_height) == (4.0, 12.0, 1.0, 6.0)
    assert space.field == FIELD_DISTANCES * 12.0 and space.side == SIDE_DISTANCES * 12.0
    grains = top_grains(space)
    inside = [grain for grain in grains if 0 <= grain.x <= space.field and 0 <= grain.y <= space.field]
    # about one grain per hexagonal cell of the distance D, the radii spread around R
    assert abs(len(inside) - space.field ** 2 / (math.sqrt(3) / 2 * 12.0 ** 2)) <= 8
    radii = sorted(grain.radius for grain in grains)
    assert radii[len(radii) // 2] == pytest.approx(4.0, rel=0.15) and radii[0] < 4.0 < radii[-1]
    assert all(grain.height == pytest.approx(6.0 * grain.radius / 4.0) for grain in grains)
    # the drawn coverage is that of grains of radius R at the distance D, about (2π/√3)(R/D)²
    assert drawn_coverage(space, grains) == pytest.approx(2 * math.pi / math.sqrt(3) * (4.0 / 12.0) ** 2, abs=0.08)
    row = side_grains(space)
    assert len(row) >= SIDE_DISTANCES and all(grain.y == 0 for grain in row)
    # the dice stay: another solution moves no grain relative to its distance
    other = top_grains(real_space(dict(values, R_nm=5.0, D_nm=15.0)))
    assert [round(grain.x / 15.0, 9) for grain in other] == [round(grain.x / 12.0, 9) for grain in grains]
    # no height: hemispheres on the base; a narrow distribution: all grains alike
    assert real_space({"R_nm": 3.0, "D_nm": 9.0, "base_nm": 2.0}).height == 5.0
    assert {round(grain.radius, 9) for grain in top_grains(real_space({"R_nm": 3.0, "D_nm": 9.0}))} == {3.0}


def test_touching_grains_form_clusters_and_a_cluster_across_the_view_has_percolated():
    from src.gimap.features.sputter.domain import Grain, arrangement, clusters, real_space, spanning, top_grains

    row = [Grain(x, 5.0, 2.0, 1.0) for x in (2.0, 6.0, 10.0)]  # touching neighbours: one cluster ...
    apart = [Grain(x, 5.0, 1.0, 1.0) for x in (2.0, 6.0, 10.0)]  # ... and three separate grains
    assert len(set(clusters(row))) == 1 and len(set(clusters(apart))) == 3
    assert spanning(row, clusters(row), 12.0) == set(clusters(row))  # it reaches both side edges of a 12 nm view
    assert spanning(row, clusters(row), 20.0) == set()
    # the same arrangement of a solution at three sizes: islands, grains that touch, a network across the view
    stages = [arrangement(space, top_grains(space)) for space in
              (real_space({"R_nm": r, "D_nm": 10.0, "width": 0.2}) for r in (2.0, 4.0, 5.2))]
    assert stages == ["islands", "coalescing", "network"]


def test_a_porous_solution_is_drawn_from_where_its_metal_sits():
    from src.gimap.features.sputter.domain import layer_particles, pore_density, porous_space, real_space

    values = {"R_nm": 0.9, "width": 0.2, "amount_nm": 8.0, "particle_nm": 75.0, "pore_depth_nm": 400.0,
              "pore_width_nm": 60.0, "cap_cover": 0.3, "cap_thickness_nm": 10.0, "island_amount_nm": 1.5,
              "cap_amount_nm": 0.5, "pore_amount_nm": 2.0, "cluster_height_nm": 1.8, "cluster_distance_nm": 2.0}
    space = porous_space(values)
    assert real_space(values) is None and space.particle == 75.0 and space.deposited == 8.0
    assert space.depth_shown >= 400.0 + 3 * 60.0 and space.depth_shown % 150.0 == 0 and space.side == 900.0
    assert pore_density(space, 400.0) == 1.0 and pore_density(space, 460.0) == pytest.approx(math.exp(-0.5))
    particles = layer_particles(space)
    top_row = [particle for particle, _capped in particles if particle.y == -75.0 and 0 <= particle.x <= space.side]
    assert sum(capped for _particle, capped in particles) == round(0.3 * len(top_row))
    assert min(particle.y for particle, _capped in particles) <= -space.depth_shown + 75.0
    # without a cap, nothing is capped; without a pore depth, no pore metal is drawn
    plain = porous_space(dict(values, cap_cover=0.0, pore_depth_nm=None))
    assert not any(capped for _particle, capped in layer_particles(plain)) and pore_density(plain, 0.0) == 0.0
    assert porous_space({"R_nm": 1.0, "particle_nm": 75.0}) is None  # the package gave no metal amounts


def test_a_solution_without_a_grain_arrangement_has_no_picture():
    from src.gimap.features.sputter.domain import real_space

    assert real_space({"R_nm": 1.2, "width": 0.2, "f_islands": 0.4, "xi_nm": 5.0}) is None  # porous: no distance
    assert real_space({"D_nm": 10.0}) is None and real_space({"R_nm": 0.0, "D_nm": 10.0}) is None
    assert real_space({"R_nm": float("nan"), "D_nm": 10.0}) is None


def test_a_result_on_a_porous_substrate_has_its_own_quantities(tmp_path):
    out = write_result(tmp_path / "out", windows=porous_windows(), wanted=POROUS_WANTED)
    result = ClientFolderReader().read(out)
    assert result.porous and not result.static
    values = result.windows[0].entries[0].values
    assert sorted(values) == ["R_nm", "amount_nm", "cap_share", "depth_nm", "f_islands", "film_share", "width", "xi_nm"]
    assert values["f_islands"] == 2.0  # above 1: kept as it is
    assert result.series("xi_nm")[1] == [4.8, 8.0, 12.0] and result.series("depth_nm")[0] == [3.0, 6.0, 9.0]
    assert all(math.isnan(value) for key in ("H_nm", "D_nm", "gap_nm", "coverage") for value in result.series(key)[1])
    flat = ClientFolderReader().read(write_result(tmp_path / "flat"))
    assert not flat.porous and not flat.static and flat.x_label == ""


def test_a_static_series_is_read_with_its_samples_its_axis_and_its_ranges(tmp_path):
    reader = ClientFolderReader()
    out = write_static(tmp_path / "zz_series")
    assert reader.locate(out) == out and reader.complete(out)
    result = reader.read(out)
    assert result.static and not result.porous and (result.x_label, result.x_unit) == ("nominal thickness", "nm")
    assert (result.scan, result.metal, result.family) == ("zz_series", "Au", "zz_static_family")
    assert [window.key for window in result.windows] == ["zz_a", "zz_b", "zz_c", "zz_d"]  # the label, not the place
    assert [result.position(window) for window in result.windows] == [4.0, 4.0, 7.0, 10.0]
    last = result.windows[3]
    assert (last.label, last.sample, last.substrate, last.reference, last.frames) == \
        ("zz_d", "the sample zz_d", "bare substrate measured", "zz_d_ref", 3)
    assert last.since_onset_s is None and last.labels() == ["model"] and last.default_label() == "model"
    found = last.entries[0]
    assert found.source == "model" and found.range("R_nm") == (4.5, 6.0) and found.range("gap_nm") is None
    assert result.series("R_nm") == ([4.0, 4.0, 7.0, 10.0], [2.0, 2.5, 3.5, 5.0])
    lows, highs = result.spread("H_nm")
    assert lows == pytest.approx([2.16, 2.7, 3.78, 5.4]) and highs == pytest.approx([2.88, 3.6, 5.04, 7.2])
    assert all(math.isnan(value) for value in result.spread("gap_nm")[0])
    assert result.nearest(6.0) == 2 and result.nearest(100.0) == 3
    images = reader.patterns(result, last)  # main/w_<label>.npz
    assert images.labels == ("model",) and images.measured.shape == (AF.size, Q.size)
    assert result.file("report_md") == out / "REPORT.md"
    # a range that is not one, or of a quantity the result does not list, is left out
    samples = static_samples()[:1]
    samples[0]["entries"][0]["ranges"] = {"R_nm": [3.0, 1.0], "H_nm": ["zz", 2.0], "zz_other": [1.0, 2.0],
                                          "D_nm": [5.0, 7.0]}
    odd = reader.read(write_static(tmp_path / "odd", samples=samples))
    assert odd.windows[0].entries[0].ranges == {"D_nm": (5.0, 7.0)}


def test_the_failed_step_of_a_static_run_gives_its_note_and_its_log(tmp_path):
    reader = ClientFolderReader()
    intake = write_static(tmp_path / "a", status="failed at intake")
    assert reader.failure_details(intake, reader.analysis(intake)) == "the invented reason of the failure"
    predict = write_static(tmp_path / "b", status="failed at predict")
    details = reader.failure_details(predict, reader.analysis(predict)).splitlines()
    assert details[0].startswith("the invented step failed (see ") and details[-1] == "the last line of the invented log"
    assert not reader.complete(predict) and reader.read(predict).windows  # the earlier samples can still be shown


# -- static frames: the package's values, the command and the series ---------------------------------------------------


def _static(tmp_path, process=None, package=None) -> StaticPackageRunner:
    package = package or static_package(tmp_path / "package")
    return StaticPackageRunner(_runner(tmp_path, process or _StaticProcess(), package), lambda: tmp_path / "series")


def _sample(tmp_path, **changes) -> StaticRequest:
    frames = tmp_path / "zz frames"
    frames.mkdir(exist_ok=True)
    values = dict(frames=str(frames), name="zz 8nm", nominal_nm=8.0, series_folder=tmp_path / "series" / "one")
    values.update(changes)
    return StaticRequest(**values)


class _StaticProcess(_Process):
    """The ``static`` command: imports the label (and its bare substrate) into the data folder, writes the series."""

    def __init__(self, lines=(*STATIC_LINES, "status: ok; report: none"), code=0, stopped=False, imports=True):
        super().__init__(lines, code, stopped)
        self.imports = imports

    def __call__(self, command, *, on_line, should_stop, env=None, cwd=None, **_more):
        value = lambda option: command[command.index(option) + 1]  # noqa: E731
        label, root, out = value("--label"), Path(value("--data")), Path(value("--out"))
        names = [label] + ([label + "_ref"] if "--reference-frames" in command else [])
        for name in names if self.imports else ():
            (root / name / "binned").mkdir(parents=True, exist_ok=True)
            (root / name / "binned" / "run.json").write_text("{}", encoding="utf-8")
        if self.code == 0:
            kept = [item for item in (out / "main" / "main.json",) if item.is_file()]
            samples = json.loads(kept[0].read_text(encoding="utf-8"))["windows"] if kept else []
            samples = [item for item in samples if item["label"] != label]
            samples.append(static_sample(len(samples) + 1, label, float(value("--nominal-nm")), 2.0, 3.0))
            write_static(out, samples=samples, label=label, data_root=root)
        return super().__call__(command, on_line=on_line, should_stop=should_stop, env=env, cwd=cwd)


def test_the_package_says_its_own_geometry_of_static_frames_and_its_masks(tmp_path):
    runner = _static(tmp_path)
    found = runner.defaults()
    # the beam centre of the source, (column 300, row 700) as pixel-centre indices, in GIMaP's coordinates
    assert (found.energy_ev, found.sdd_mm, found.beam_col, found.beam_row, found.sample_roll_deg, found.alpha_i_deg,
            found.problem) == (9500.0, 4321.0, 300.5, 700.5, -0.25, 0.35, "")
    assert runner.default_series() == tmp_path / "series" / "series_1"
    assert [mask.name for mask in runner.runner.masks()] == ["zz_first_mask.npy", "zz_second_mask.npy"]
    older = _static(tmp_path, package=package_folder(tmp_path / "older"))  # a package without the static entry
    assert older.defaults().problem == STATIC_UNSUPPORTED and older.runner.masks() == []
    with pytest.raises(FileNotFoundError, match="no entry for static frames"):
        older.run(_sample(tmp_path), lambda _line: None, lambda: False)
    (tmp_path / "odd").mkdir()
    odd = static_package(tmp_path / "odd" / "package")
    (odd / "code/iam/iampred/static.py").write_text("GEOMETRY = make()\n", encoding="utf-8")
    assert static_defaults(locate_package(odd, tmp_path / "nowhere")) == StaticDefaults()  # nothing said: nothing filled


def test_the_static_command_follows_the_request(tmp_path):
    process = _StaticProcess()
    runner = _static(tmp_path, process)
    archive = tmp_path / "zz frames.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        for name in ("zz_8nm/a_00001.cbf", "zz_8nm/a_00002.CBF", "zz_bare/b_00001.cbf", "top_00001.cbf", "notes.txt"):
            opened.writestr(name, b"")
    assert archive_folders(archive) == ["", "zz_8nm/", "zz_bare/"]
    mask = tmp_path / "zz_mask.npy"
    np.save(mask, np.ones((2, 2), dtype=bool))
    request = _sample(tmp_path, frames=f"{archive}::zz_8nm/", reference_frames=f"{archive}::zz_bare/", mode="refine",
                      energy_ev=9000.0, sdd_mm=5000.0, beam_col=400.5, beam_row=800.5, sample_roll_deg=-0.3,
                      mask_file=mask)
    assert request.problems() == [] and request.label() == "zz_8nm"
    lines = []
    run = runner.run(request, lines.append, lambda: False)
    assert (run.code, run.stopped) == (0, False) and lines[:len(STATIC_LINES)] == list(STATIC_LINES)
    command, env, cwd = process.calls[0]
    series = tmp_path / "series" / "one"
    record = json.loads((series / SERIES_FILE).read_text(encoding="utf-8"))
    root = tmp_path / "user" / STORAGE_FOLDER / "static" / record["series_id"]  # apart from the in-situ data folders
    assert command == [
        "zz_python", "-u", "-m", "iampred.static", "--frames", f"{archive}::zz_8nm/", "--label", "zz_8nm",
        "--nominal-nm", "8.0", "--out", str(series), "--sample", "zz 8nm", "--mode", "refine", "--workers", "3",
        "--data", str(root), "--reference-frames", f"{archive}::zz_bare/", "--mask", str(mask), "--sdd-mm", "5000.0",
        "--energy-ev", "9000.0", "--beam-col", "400.0", "--beam-row", "800.0", "--roll", "-0.3"]
    home = tmp_path / "package"
    assert cwd == str(home / "code" / "iam") and env["IAM_STATIC_DATA"] == env["IAM_DATA"] == str(root)
    assert env["IAMPRED_MODELS"] == str(home / "models") and env["PYTHONPATH"] == str(home / "code" / "iam")
    assert root.is_dir() and Path(env["IAM_WORK"]).is_dir() and root not in Path(env["IAM_WORK"]).parents
    assert record["mask"].startswith("mask-") and record["samples"]["zz_8nm"]["mode"] == "refine"
    assert record["samples"]["zz_8nm"]["command"] == command
    marker = json.loads((root / ("zz_8nm" + IMPORT_MARKER)).read_text(encoding="utf-8"))
    assert marker["intake"] == request.intake_digest() and marker["request"]["nominal_nm"] == 8.0
    # nothing but what is needed: the package then uses its own geometry and mask, and Si as the substrate
    plain = _sample(tmp_path, name="zz_12", nominal_nm=12.0, series_folder=tmp_path / "series" / "two")
    runner.run(plain, lambda _line: None, lambda: False)
    command = process.calls[1][0]
    other = json.loads((tmp_path / "series" / "two" / SERIES_FILE).read_text(encoding="utf-8"))
    assert command[4:] == ["--frames", str(tmp_path / "zz frames"), "--label", "zz_12", "--nominal-nm", "12.0",
                           "--out", str(tmp_path / "series" / "two"), "--sample", "zz_12", "--mode", "quick",
                           "--workers", "3", "--data", str(root.parent / other["series_id"])]
    assert other["mask"] == PACKAGE_MASK and other["series_id"] != record["series_id"]  # a data folder per series


def test_a_static_request_says_what_is_missing(tmp_path):
    assert _sample(tmp_path).problems() == []
    one_frame = tmp_path / "zz_frame.cbf"
    one_frame.write_bytes(b"")
    assert _sample(tmp_path, frames=str(one_frame)).problems() == []
    assert split_frames(" zz.zip :: inner/ ") == ("zz.zip", "inner/") and split_frames("zz folder") == ("zz folder", "")
    wrong = StaticRequest(frames=str(tmp_path / "nowhere"), name=" ?! ", nominal_nm=None, series_folder=Path(""),
                          reference_frames=str(tmp_path / "zz frames") + "::inner/", mode="auto", energy_ev=-1.0)
    assert wrong.problems() == list(STATIC_PROBLEMS[:2]) + [STATIC_PROBLEMS[2]] + list(STATIC_PROBLEMS[4:])
    assert _sample(tmp_path, name="zz_ref").problems() == [STATIC_PROBLEMS[3]]  # where a bare substrate is kept
    assert _sample(tmp_path, name="样品 zz-8.5 nm").label() == "zz-8_5_nm"
    # the label names the sample; everything the import depends on is in the digests
    first = _sample(tmp_path)
    assert first.intake_digest() == _sample(tmp_path, mode="full").intake_digest()
    assert first.digest("p") != _sample(tmp_path, mode="full").digest("p") != first.digest("another package")
    assert first.intake_digest() != _sample(tmp_path, nominal_nm=9.0).intake_digest()
    assert first.intake_digest() != _sample(tmp_path, beam_col=1.0).intake_digest()


def test_the_same_sample_again_is_shown_analysed_again_or_replaced(tmp_path):
    process = _StaticProcess()
    runner = _static(tmp_path, process)
    request = _sample(tmp_path, reference_frames=str(tmp_path / "zz frames"))
    assert not runner.analysed(request) and not runner.replaces(request)
    runner.run(request, lambda _line: None, lambda: False)
    series = request.series_folder
    root = Path(process.calls[0][0][process.calls[0][0].index("--data") + 1])
    assert runner.analysed(request) and not runner.replaces(request)
    # another mode: the frames stay imported, the sample is analysed again
    refined = _sample(tmp_path, reference_frames=str(tmp_path / "zz frames"), mode="refine")
    assert not runner.analysed(refined) and not runner.replaces(refined)
    (root / "zz_8nm" / "binned" / "zz_kept").write_text("", encoding="utf-8")
    runner.run(refined, lambda _line: None, lambda: False)
    assert (root / "zz_8nm" / "binned" / "zz_kept").is_file() and runner.analysed(refined)
    # other values under the same name: the import, the bare substrate and the sample in the series go first
    other = _sample(tmp_path, nominal_nm=9.0)
    assert runner.replaces(other) and not runner.analysed(other)
    (series / "zz_8nm" / "zz_old").write_text("", encoding="utf-8")
    runner.run(other, lambda _line: None, lambda: False)
    assert not (root / "zz_8nm" / "binned" / "zz_kept").exists() and not (root / "zz_8nm_ref").exists()
    assert not (series / "zz_8nm" / "zz_old").exists() and runner.analysed(other) and not runner.replaces(other)
    # an import that was cut off (no marker) goes without a question; a run that failed is not kept as analysed
    cut = _sample(tmp_path, name="zz_cut")
    (root / "zz_cut" / "binned").mkdir(parents=True)
    (root / "zz_cut" / "binned" / "zz_half").write_text("", encoding="utf-8")
    assert not runner.replaces(cut)
    failing = _static(tmp_path, _StaticProcess(lines=("[intake] ...", "      FAILED: zz in 1 s"), code=1, imports=False))
    failing.run(cut, lambda _line: None, lambda: False)
    assert not (root / "zz_cut").exists() and not failing.analysed(cut)
    assert not (root / ("zz_cut" + IMPORT_MARKER)).exists() and runner.analysed(other)


def test_a_series_takes_a_sample_only_where_its_data_are(tmp_path):
    runner = _static(tmp_path)
    full = tmp_path / "zz_not_a_series"
    full.mkdir()
    (full / "zz_file.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no series of static frames"):
        runner.run(_sample(tmp_path, series_folder=full), lambda _line: None, lambda: False)
    # a series the package made without GIMaP: added to in its own data folder, never under a name that is there
    data = tmp_path / "zz_data"
    (data / "zz_a").mkdir(parents=True)
    foreign = write_static(tmp_path / "foreign", data_root=data)
    assert runner.series(foreign) == (data, {}, False)
    with pytest.raises(ValueError, match="already hold a sample named zz_a"):
        runner.run(_sample(tmp_path, name="zz_a", series_folder=foreign), lambda _line: None, lambda: False)
    process = _StaticProcess()
    _static(tmp_path, process).run(_sample(tmp_path, series_folder=foreign), lambda _line: None, lambda: False)
    assert process.calls[0][0][process.calls[0][0].index("--data") + 1] == str(data)
    assert not (foreign / SERIES_FILE).exists() and not (data / ("zz_8nm" + IMPORT_MARKER)).exists()
    with pytest.raises(ValueError, match="was analysed elsewhere"):
        runner.run(_sample(tmp_path, series_folder=write_static(tmp_path / "moved")), lambda _l: None, lambda: False)
    # GIMaP's own series: one pixel mask, and every sample's import must still be there
    own = _sample(tmp_path)
    runner.run(own, lambda _line: None, lambda: False)
    mask = tmp_path / "zz_mask.npy"
    np.save(mask, np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="another pixel mask"):
        runner.run(_sample(tmp_path, name="zz_second", mask_file=mask), lambda _line: None, lambda: False)
    runner.run(_sample(tmp_path, mask_file=mask), lambda _line: None, lambda: False)  # the only sample: it may change
    record = json.loads((own.series_folder / SERIES_FILE).read_text(encoding="utf-8"))
    assert record["mask"].startswith("mask-")
    shutil.rmtree(tmp_path / "user" / STORAGE_FOLDER / "static" / record["series_id"] / "zz_8nm")
    with pytest.raises(ValueError, match="not on this computer"):
        runner.run(_sample(tmp_path, name="zz_second", mask_file=mask), lambda _line: None, lambda: False)
    # a series id that is not one is never used as a folder name
    (tmp_path / "forged").mkdir()
    (tmp_path / "forged" / SERIES_FILE).write_text('{"series_id": "../../zz_elsewhere"}', encoding="utf-8")
    with pytest.raises(ValueError, match="holds no series of static frames"):
        runner.series(tmp_path / "forged")


def test_the_steps_of_the_static_command_move_the_bar():
    progress = ClientProgress.static()
    assert progress.feed("[intake] ...") and (progress.index, progress.total, progress.step) == (1, 3, "intake")
    assert progress.message() == ("Step {index} of {total}: {stage}…", {"index": 1, "total": 3,
                                                                         "stage": "Reading the frames"})
    assert not progress.feed("      zz: 4 frames, 4 usable, 1e+06 counts") and progress.fraction == 0.0
    assert progress.feed("      done in 2 s") and progress.done == ["intake"] and progress.fraction == pytest.approx(0.3)
    assert progress.feed("[predict] ...") and progress.message()[1]["stage"] == "Finding the solution of the sample"
    assert progress.feed("      FAILED: the invented step failed (see zz.log) in 3 s") and progress.failed == ["predict"]
    assert progress.feed("status: failed at predict; report: None") and progress.fraction == 1.0
    assert not ClientProgress.static().feed("[zz_other] ...")  # not a step of the command


def test_static_analyze_adds_a_sample_once_and_says_why_it_failed(tmp_path):
    reader = ClientFolderReader()
    runner = FakeStaticRunner(tmp_path / "series")
    service = StaticService(runner, reader)
    assert service.defaults().sdd_mm == 4321.0 and service.default_series() == tmp_path / "series" / "series_1"
    first = _sample(tmp_path)
    events = []
    result = service.analyze(first, events.append)
    assert result.static and [window.label for window in result.windows] == ["zz_8nm"]
    assert isinstance(events[0], ClientStarted) and (events[0].label, events[0].reused) == ("zz_8nm", False)
    assert events[1:len(STATIC_LINES) + 1] == list(STATIC_LINES)
    result = service.analyze(_sample(tmp_path, name="zz_4nm", nominal_nm=4.0), events.append)
    assert [window.label for window in result.windows] == ["zz_4nm", "zz_8nm"]  # along the nominal thickness
    again = []
    assert len(service.analyze(first, again.append).windows) == 2 and len(runner.requests) == 2
    assert len(again) == 1 and again[0].reused  # the same request: the series as it is, no run
    assert service.replaces(_sample(tmp_path, nominal_nm=9.0)) and not service.replaces(first)
    assert not service.replaces(_sample(tmp_path, nominal_nm=None))  # nothing to ask about a request that cannot run
    with pytest.raises(ValueError, match="nominal thickness"):
        service.analyze(_sample(tmp_path, nominal_nm=None), events.append)
    with pytest.raises(ClientMissing, match="no entry for static frames"):
        StaticService(FakeStaticRunner(tmp_path, problem=STATIC_UNSUPPORTED), reader).analyze(first, events.append)
    failing = StaticService(FakeStaticRunner(tmp_path / "other", status="failed at predict"), reader)
    with pytest.raises(ClientFailed, match=r"static frames stopped with an error \(failed at predict\)") as failed:
        failing.analyze(_sample(tmp_path, series_folder=tmp_path / "other" / "one"), events.append)
    assert failed.value.details.splitlines()[-1] == "the last line of the invented log"
    with pytest.raises(Stopped, match="stay in the series"):
        StaticService(FakeStaticRunner(tmp_path / "third", wait_for_stop=True), reader).analyze(
            _sample(tmp_path, series_folder=tmp_path / "third" / "one"), events.append, lambda: True)
    assert ClientService(FakeClientRunner(), reader).static is None  # offered only where it was given
