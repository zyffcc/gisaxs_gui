"""The fringe analysis (``utils/sputter_fringes``): the island height H(t) from the moving q_z fringes.

Synthetic result folders only (``tests/sputter_fringe_data.py``: a growing layer's fringes, Poisson noise, invented
detector gaps and geometry): H recovered within its interval, orders counted correctly, no H from cuts without
fringes, ``below_range`` before the first minimum enters the range, and a command line that writes atomically and never
into its input.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.sputter_fringe_data import GAPS, linear, true_order, write_result
from utils.sputter_fringes.cli import COLUMNS, HEIGHTS, SUMMARY, analyse, main

ROOT = Path(__file__).resolve().parents[1]
FAST = linear(0.25, 3.0)
SLOW = linear(0.08, 3.0)
OFF = (40, 41, 90)
"""Windows flagged as off the sample (their intensity is lower): left out."""


def _quiet(_line: str) -> None:
    pass


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    folder = write_result(tmp_path_factory.mktemp("fast") / "result", FAST, beam_off=OFF)
    rows, summary = analyse(folder, on_line=_quiet)
    return folder, rows, summary


@pytest.fixture(scope="module")
def slow(tmp_path_factory):
    folder = write_result(tmp_path_factory.mktemp("slow") / "result", SLOW, windows=220, level=4.0e4)
    rows, summary = analyse(folder, on_line=_quiet)
    return folder, rows, summary


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    folder = write_result(tmp_path_factory.mktemp("flat") / "result", lambda t: 0.0, windows=200, seed=7)
    rows, summary = analyse(folder, on_line=_quiet)
    return folder, rows, summary


def _measured(rows):
    return [r for r in rows if r["H_nm"]]


def _covered(rows, height) -> float:
    measured = _measured(rows)
    inside = [float(r["H_lo68_nm"]) <= height(float(r["time_s"])) <= float(r["H_hi68_nm"]) for r in measured]
    return sum(inside) / len(inside)


def _orders_right(summary, height) -> None:
    accepted = [t for t in summary["tracks"] if t["status"] == "accepted"]
    assert accepted and all(t["order"] for t in accepted)
    for t in accepted:
        for time, af in ((t["t_start_s"], t["af_start_deg"]), (t["t_end_s"], t["af_end_deg"])):
            n = true_order(af, height(time))
            assert abs(n - t["order"]) < 0.25, (t, n)


# -- a growing layer --------------------------------------------------------------------------------------------


def test_heights_of_a_fast_growing_layer_are_recovered_with_their_orders(fast):
    _folder, rows, summary = fast
    assert summary["status"] == "measured"
    assert len(_measured(rows)) >= 40 and _covered(rows, FAST) >= 0.9
    _orders_right(summary, FAST)
    assert max(t["order"] for t in summary["tracks"] if t["status"] == "accepted") >= 3
    growth = summary["growth"]
    assert abs(growth["v_nm_s"] - 0.25) <= 3 * growth["v_error_nm_s"] + 0.03 * 0.25
    # several periods in range: alpha_c is fitted to the fringes, close to the metal's bulk value used to make them
    alpha_c = summary["alpha_c"]
    assert alpha_c["source"] == "phase-locked fits" and abs(alpha_c["used_deg"] - alpha_c["bulk_deg"]) < 0.01
    assert summary["phase_locked_groups"] >= 10


def test_a_slow_layer_is_counted_from_its_first_minimum_and_is_below_the_range_before(slow):
    _folder, rows, summary = slow
    assert summary["status"] == "measured"
    assert len(_measured(rows)) >= 12 and _covered(rows, SLOW) >= 0.9
    _orders_right(summary, SLOW)
    orders = {t["order"]: t["order_source"] for t in summary["tracks"] if t["status"] == "accepted"}
    assert orders.get(1) == "entered at the top first" and summary["phase_locked_groups"] == 0
    below = [r for r in rows if r["status"] == "below_range"]
    assert len(below) >= 10
    for r in below:
        assert SLOW(float(r["time_s"])) < float(r["H_upper_bound_nm"]) and not r["H_nm"]
    first = min(float(r["time_s"]) for r in _measured(rows))
    assert max(float(r["time_s"]) for r in below) < first
    assert summary["alpha_c"]["source"].startswith("bulk")  # too few periods to fit it: the bulk value, said so


def test_cuts_without_fringes_give_no_height(flat):
    _folder, rows, summary = flat
    assert summary["status"] == "not_detected" and not _measured(rows)
    detected = [r for r in rows if r["status"] not in ("not_detected", "below_range")]
    assert len(detected) <= 0.02 * len(rows)
    assert any("H is not measured" in note for note in summary["notes"])


def test_gaps_off_sample_windows_and_sources_come_from_the_data(fast):
    folder, rows, summary = fast
    gaps = summary["ranges"]["detector_gaps_deg"]["0"]
    for lo, hi in GAPS:  # each invented module gap lies inside a gap found in the data
        assert any(a <= lo + 1e-6 and hi - 1e-6 <= b for a, b in gaps)
    lower = summary["ranges"]["lower_bound_deg"]  # narrow rows without pixels split nothing: one range more than gaps
    assert len(summary["ranges"]["clean_alpha_f_deg"]["0"]) == 1 + sum(lo > lower for lo, _hi in GAPS)
    assert summary["windows"]["beam_off_sample"] == len(OFF)
    assert summary["windows"]["used"] == summary["windows"]["total"] - len(OFF)
    sources = summary["sources"]
    run = (folder / "binned" / "run.json").read_bytes()
    assert sources["run.json"] == hashlib.sha256(run).hexdigest() and len(sources["chunks"]) >= 3
    assert sources["beam_off_frames"] == [f"bin_{i:06d}" for i in OFF]
    assert list(rows[0]) == list(COLUMNS)


# -- the command line -------------------------------------------------------------------------------------------


def _snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_the_command_line_writes_atomically_and_never_into_its_input(flat, tmp_path, capsys):
    folder = flat[0]
    before = _snapshot(folder)
    assert main(["--folder", str(folder), "--out", str(folder)]) == 2
    assert main(["--folder", str(folder), "--out", str(folder / "binned" / "fringes")]) == 2
    assert "FRINGES_ERROR" in capsys.readouterr().out
    assert _snapshot(folder) == before
    out = tmp_path / "out"
    assert main(["--folder", str(folder), "--out", str(out), "--metal", "Ag"]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([HEIGHTS, SUMMARY])  # no temporary file left
    assert _snapshot(folder) == before
    inside = folder / "fringes_v1"  # GIMaP's own place in a result folder: allowed, and nothing else changes
    try:
        assert main(["--folder", str(folder), "--out", str(inside)]) == 0
        after = _snapshot(folder)
        assert {k: v for k, v in after.items() if not k.startswith("fringes_v1/")} == before
    finally:
        for path in inside.glob("*"):
            path.unlink()
        inside.rmdir()


def test_a_changed_chunk_is_refused(flat, tmp_path, capsys):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(flat[0], copy)
    chunk = copy / "binned" / "chunk_00000.npz"
    data = bytearray(chunk.read_bytes())
    data[-1] ^= 0xFF
    chunk.write_bytes(bytes(data))
    assert main(["--folder", str(copy), "--out", str(tmp_path / "out")]) == 2
    assert "SHA-256" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_the_script_runs_as_a_subprocess(flat, tmp_path):
    out = tmp_path / "script"
    done = subprocess.run([sys.executable, str(ROOT / "utils" / "sputter_fringes" / "fringes.py"), "--folder",
                           str(flat[0]), "--out", str(out)], capture_output=True, text=True, timeout=300,
                          cwd=str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FRINGES_DONE\tnot_detected" in done.stdout and (out / SUMMARY).is_file()


def test_the_toolkit_is_not_touched():
    """The tool imports the toolkit's grid read only: its manifest still matches every file."""
    import json

    kit = ROOT / "utils" / "sputter_toolkit"
    manifest = json.loads((kit / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for name, sha in manifest.items():
        assert hashlib.sha256((kit / name).read_bytes()).hexdigest() == sha, name
