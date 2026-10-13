"""Nothing that is committed names the data of a private evaluation case.

A case built on a user's own (often unpublished) data lives in ``docs/agents/eval/local/``, which git
ignores: it is the one place where the names of such data are written. Everything that is committed
stays free of them (AGENTS.md). The names are read from the local case files here, so this test does
not repeat them: the name of the case file, the folder of the data, the sample folders and files of the
frames, and the calibration the case expects. On a clone there is no local case and nothing to look for.

The numbers go the same way: the peak positions a case fixes (those with three or more decimals anywhere;
rounded to two decimals where a line speaks of q or peaks), the length of its series and the numbers of its
samples are read from the case and looked for in every file that can be committed. A failure names the
file and the line, never the name or the number.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "agents" / "eval"
LOCAL = CASES / "local"
NO_LOCAL_CASE = "no private evaluation case on this computer (docs/agents/eval/local/ is empty on a clone)"
FRAMES_WORD = re.compile(r"(?<![A-Za-z])frames?(?![A-Za-z])|帧", re.IGNORECASE)
"""A line that speaks of frames: only there is a bare number taken for the length of a series."""
SUMMED_AT_THE_END = 10
"""The automatic analysis sums the last ten frames of a series: its first frame, N − 9, gives N away."""
Q_WORD = re.compile(r"(?<![A-Za-z])(?:q|peaks?|rings?)(?![A-Za-z])|Å⁻¹|峰|环", re.IGNORECASE)
"""A line that speaks of q, peaks or rings: only there is a number of two decimals taken for a peak position."""
NEAR_A_PEAK = 0.005
"""A two-decimal number this close (relative) to a peak position of a case is that peak, rounded."""
LISTED = r"\s*(?:[,;、/&]|and|or|和|与|及)\s*"
"""What stands between two numbers of a list: “27, 31”, “27 and 31”."""
PERSONAL = ("_handoff/",)
"""Untracked folders of a person's own notes: they are never committed, so they are not read here."""


def _private_cases(folder: Path | None = None) -> list[tuple[Path, dict]]:
    """The local cases that stand on data outside the repository, each with its content.

    A case somebody drafts in the local folder on the public data of ``tests/data`` names files that
    are committed, and documented, on purpose: nothing of it is private.
    """
    found = []
    for case in sorted((LOCAL if folder is None else folder).glob("*.json")):
        data = case.read_bytes()  # read as the tool reads it: Windows adds a byte order mark, PowerShell 5.1 writes UTF-16
        record = json.loads(data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"))
        if not isinstance(record, dict):
            continue  # not a case: the tool says so
        root = Path(str(record.get("data_root") or ""))
        root = (root if root.is_absolute() else ROOT / root).resolve()
        if ROOT.resolve() not in (root, *root.parents):
            found.append((case, record))
    return found


def _private_names(*, data_folder: bool = True, folder: Path | None = None) -> set[str]:
    """Lower-case fragments that occur only in the names of the private evaluation data.

    ``data_folder=False`` leaves out the folder of the data: it is usually a person's name, which
    code may carry for another reason (the id of a model that person trained).
    """
    names: set[str] = set()
    for case, record in _private_cases(folder):
        names.add(case.stem)
        root = str(record.get("data_root") or "").replace("\\", "/").rstrip("/")
        if root and data_folder:
            names.add(root.rsplit("/", 1)[-1])  # the folder of the data, usually a person's name
        for frame in record.get("frames") or ():
            for part in str(frame).replace("\\", "/").split("/"):
                tokens = part.split(".")[0].split("_")
                if len(tokens) < 3:
                    continue  # a detector folder, not a sample name
                names.add("_".join(tokens[:2]) + "_")  # how every sample name begins
                names.update(token for token in tokens  # parts such as a rate or a sample label
                             if re.search(r"[A-Za-z]", token) and re.search(r"\d", token))
        expected = (record.get("baseline") or {}).get("geometry_source_contains")
        if expected:
            names.add(str(expected))
    return {name.lower() for name in names if len(name) >= 4}


def _git_files(*arguments: str) -> list[str]:
    listed = subprocess.run(["git", "ls-files", "-z", *arguments], cwd=ROOT, capture_output=True,
                            check=True, timeout=120).stdout.decode("utf-8")
    return [name for name in listed.split("\0") if name]


def _has_git() -> bool:
    return bool(shutil.which("git")) and (ROOT / ".git").exists()


def _documents() -> list[Path]:
    """Every Markdown file git tracks, plus the release documents (new ones are not tracked yet)."""
    found = {ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "docs" / "User_Manual.md", CASES / "README.md"}
    if _has_git():
        found.update(ROOT / name for name in _git_files("--", "*.md"))
    else:
        found.update((ROOT / "docs").rglob("*.md"))
        found.update((ROOT / "src" / "gimap").rglob("README.md"))
    return sorted(path for path in found if path.is_file() and LOCAL not in path.parents)


def _committable() -> list[str]:
    """What git tracks, and what it would add: untracked files it does not ignore (new sources, tests)."""
    untracked = [name for name in _git_files("--others", "--exclude-standard") if not name.startswith(PERSONAL)]
    return sorted(set(_git_files()) | set(untracked))


def test_the_case_files_give_names_to_look_for() -> None:
    if not _private_cases():
        pytest.skip(NO_LOCAL_CASE)
    assert _private_names()
    assert _private_names(data_folder=False) < _private_names()


def _private_numbers(folder: Path | None = None) -> dict[str, set[str]]:
    """What the private cases fix in numbers, each as it would be written.

    ``peaks``: the peak positions, the spike and the ring of the baseline that have three or more decimals
    (a position with two is written everywhere: 0.40, 1.20). ``near``: every position rounded to two
    decimals, with the two-decimal numbers within ``NEAR_A_PEAK`` of it (peaks of the same line differ that
    much from sample to sample). ``frames``: the length of the series when the case's name states it
    ("… 300 frames"; a hundred frames or more: "50 frames" is written in many places), and the first of its
    last ten frames, counted from 1 and from 0. ``samples``: the numbers in the names of the sample folders,
    not the zero-padded counters of a scan.
    """
    peaks: set[str] = set()
    near: set[str] = set()
    frames: set[str] = set()
    samples: set[str] = set()
    for _case, record in _private_cases(folder):
        baseline = record.get("baseline") if isinstance(record.get("baseline"), dict) else {}
        ring = baseline.get("ring") if isinstance(baseline.get("ring"), dict) else {}
        positions = baseline.get("reliable_peaks") if isinstance(baseline.get("reliable_peaks"), list) else []
        for q in (*positions, baseline.get("spike"), ring.get("q")):
            if isinstance(q, (int, float)) and not isinstance(q, bool):
                written = repr(float(q))
                if "e" not in written and len(written.partition(".")[2]) >= 3:
                    peaks.add(written)
                if 0.1 <= q < 100.0:
                    hundredths = range(int(q * (1 - NEAR_A_PEAK) * 100), int(q * (1 + NEAR_A_PEAK) * 100) + 2)
                    near.update(f"{step / 100:.2f}" for step in hundredths if abs(step / 100 - q) <= NEAR_A_PEAK * q)
        for count in re.findall(r"(?<!\d)(\d{3,})\s*frames", str(record.get("name") or ""), re.IGNORECASE):
            frames.update((count, str(int(count) - SUMMED_AT_THE_END + 1), str(int(count) - SUMMED_AT_THE_END)))
        for frame in record.get("frames") or ():
            sample = str(frame).replace("\\", "/").split("/")[0].split(".")[0]
            samples.update(token for token in sample.split("_") if token.isdigit() and token[0] != "0" and len(token) >= 2)
    return {"peaks": peaks, "near": near, "frames": frames, "samples": samples}


def _number_lines(text: str, numbers: dict[str, set[str]]) -> list[int]:
    """The lines of ``text`` (counted from 1) that write out one of ``numbers``.

    A peak position counts wherever it stands as a number of its own, its two-decimal rounding only on a line
    that speaks of q, peaks or rings; the length of a series only on a line that speaks of frames (the same
    digits are also an HTTP status or a size); a sample number after the word sample, or next to another
    sample number in a list.
    """
    def one_of(values: set[str]) -> str:
        return "(?:" + "|".join(re.escape(value) for value in sorted(values)) + ")"

    checks = []
    if numbers.get("peaks"):
        peak = re.compile(r"(?<![\d.])" + one_of(numbers["peaks"]) + r"(?!\d)")
        checks.append(lambda line: peak.search(line) is not None)
    if numbers.get("near"):
        rounded = re.compile(r"(?<![\d.])" + one_of(numbers["near"]) + r"(?!\d)")
        checks.append(lambda line: rounded.search(line) is not None and Q_WORD.search(line) is not None)
    if numbers.get("frames"):
        count = re.compile(r"(?<![\d.])" + one_of(numbers["frames"]) + r"(?!\d|\.\d)")
        checks.append(lambda line: count.search(line) is not None and FRAMES_WORD.search(line) is not None)
    if numbers.get("samples"):
        sample = re.compile(r"(?:sample|样品)\s*(?:no\.?\s*|#\s*)?" + one_of(numbers["samples"]) + r"(?!\d)", re.IGNORECASE)
        checks.append(lambda line: sample.search(line) is not None)
        own = r"(?<![\w.])(" + one_of(numbers["samples"]) + r")(?![\w]|\.\d)"
        listed = re.compile(own + r"(?=" + LISTED + own + r")", re.IGNORECASE)  # the second one is only looked at
        checks.append(lambda line: any(found.group(1) != found.group(2) for found in listed.finditer(line)))
    return [number for number, line in enumerate(text.splitlines(), start=1) if any(check(line) for check in checks)]


def _files_writing_private_numbers() -> list[str]:
    """``file: line …`` for every file that can be committed and writes out a number of a private case.

    The numbers stay inside this function: a failure report (even one that prints local variables) shows
    places only, and may be pasted anywhere.
    """
    numbers = _private_numbers()
    quick = [value.encode("utf-8") for values in numbers.values() for value in values]
    found = []
    for name in _committable():
        path = ROOT / name
        if not path.is_file():
            continue  # deleted in the working tree
        data = path.read_bytes()
        if b"\0" in data[:8192] or not any(value in data for value in quick):
            continue  # an image, a model; or none of the digits anywhere
        lines = _number_lines(data.decode("utf-8", errors="replace"), numbers)
        if lines:
            shown = ", ".join(str(number) for number in lines[:12]) + (" …" if len(lines) > 12 else "")
            found.append(f"{name}: line {shown}")
    return found


def test_names_are_taken_from_cases_on_data_outside_the_repository(tmp_path: Path) -> None:
    """Invented cases: what is looked for, and that a case on the public test data adds nothing."""
    private = {
        "data_root": str(tmp_path / "abc_beamtime"),
        "frames": ["abc_film_runA_0001/det/abc_film_runA_0001_00001.nxs", "abc_film_slow3x_0002/det/abc_film_slow3x_0002_00001.nxs"],
        "baseline": {"geometry_source_contains": "abc_standard_lab6_take2"},
    }
    public = {"data_root": "tests/data/external/giwaxs_p08_mapi", "frames": ["S121_MAI_A2_00841.tif"],
              "baseline": {"geometry_source_contains": "LaB6_2021_12_DESY_P08.poni"}}
    (tmp_path / "public-frame.json").write_text(json.dumps(public), encoding="utf-8")
    assert _private_cases(tmp_path) == [] and _private_names(folder=tmp_path) == set()
    inside = dict(public, data_root=str(ROOT / "tests" / "data" / "external" / "giwaxs_p08_mapi"))  # the same, written in full
    (tmp_path / "public-frame.json").write_text(json.dumps(inside), encoding="utf-8")
    assert _private_cases(tmp_path) == []

    for number, data in enumerate((json.dumps(private).encode("utf-8"), b"\xef\xbb\xbf" + json.dumps(private).encode("utf-8"),
                                   json.dumps(private).encode("utf-16"))):  # as Windows tools save a file
        (tmp_path / "invented-film.json").write_bytes(data)
        assert [case.name for case, _record in _private_cases(tmp_path)] == ["invented-film.json"], number
        assert _private_names(folder=tmp_path) == {"invented-film", "abc_beamtime", "abc_film_", "slow3x", "abc_standard_lab6_take2"}
    assert _private_names(folder=tmp_path, data_folder=False) == {"invented-film", "abc_film_", "slow3x", "abc_standard_lab6_take2"}
    (tmp_path / "outside.json").write_text(json.dumps(dict(public, data_root="../another_folder")), encoding="utf-8")
    assert "outside.json" in [case.name for case, _record in _private_cases(tmp_path)]  # relative, but not in the repository


def test_numbers_are_taken_from_private_cases_and_found_only_where_they_mean_the_data(tmp_path: Path) -> None:
    """Invented cases and invented lines: what is looked for, and what is left alone."""
    private = {
        "name": "Invented film: 2 samples x 300 frames",
        "data_root": str(tmp_path / "abc_beamtime"),
        "frames": ["abc_film_runA_27_0001/det/abc_film_runA_27_0001_00001.nxs", "abc_film_slow3x_31_0002/det/x_00001.nxs"],
        "baseline": {"reliable_peaks": [1.5, 2.2, 1.2345], "spike": 4.125, "ring": {"q": 2.2, "f_min": 0.3}},
    }
    public = dict(private, data_root="tests/data/external/giwaxs_p08_mapi")
    (tmp_path / "public-frame.json").write_text(json.dumps(public), encoding="utf-8")
    assert _private_numbers(tmp_path) == {"peaks": set(), "near": set(), "frames": set(), "samples": set()}
    short = {"name": "Invented film: 50 frames", "data_root": str(tmp_path / "abc_beamtime"), "baseline": "not filled in yet"}
    (tmp_path / "short-series.json").write_text(json.dumps(short), encoding="utf-8")  # a common number, and half a case
    assert _private_numbers(tmp_path) == {"peaks": set(), "near": set(), "frames": set(), "samples": set()}
    (tmp_path / "invented-film.json").write_text(json.dumps(private), encoding="utf-8")
    numbers = _private_numbers(tmp_path)
    assert numbers == {
        "peaks": {"1.2345", "4.125"},
        "near": {"1.23", "1.24", "1.50", "2.19", "2.20", "2.21", "4.11", "4.12", "4.13", "4.14"},  # within half a percent
        "frames": {"300", "291", "290"}, "samples": {"27", "31"},
    }

    text = "\n".join((
        'start = {"peaks": [{"q": 1.2345}]}',             # 1: a peak position of the case
        "a spike at 4.125 Å⁻¹.",                           # 2: … at the end of a sentence too
        "q = 11.2345, 1.23456 and 4.1250; d = 2.2 Å",     # 3: other numbers, and a position with one decimal
        "assert response.status == 300",                  # 4: the same digits, not frames
        "SeriesWorkbench(frame, frames=300)",             # 5: the length of the series
        "| run.nxs | OK | 291–300 of 300 | frames |",     # 6: its last ten frames
        "panel = QFrame(); panel.resize(300, 291)",       # 7: a widget, not a frame of a series
        "the first 3000 frames, 0.300 s per frame",       # 8: other numbers on a line about frames
        "Sample 27: the line is there from the start",    # 9: a numbered sample
        "样品 31 的峰",                                    # 10
        "27 samples at 31 °C",                            # 11: counts and temperatures
        "Shadow at q 1.23, 61–90° at q 4.13 Å⁻¹.",        # 12: peak positions rounded to two decimals
        "margin = 1.23  # cm; a factor 4.13",             # 13: the same digits where nothing speaks of q
        "the ring near q = 1.25 and a peak at 4.15",      # 14: further away than half a percent
        "analysis = SimpleNamespace(frame_index=290, frame_total=10)",   # 15: the first of the last ten, counted from 0
        '"""“27, 31 alike; 44 different”: the label of the groups."""',  # 16: two sample numbers in a list
        "runs 27 and 31 were measured first",             # 17
        "spin.setRange(27, 27); box.resize(27 * 31, 31)", # 18: one number twice, a product
    ))
    assert _number_lines(text, numbers) == [1, 2, 5, 6, 9, 10, 12, 15, 16, 17]
    assert _number_lines(text, {"peaks": set(), "near": set(), "frames": set(), "samples": set()}) == []


def test_no_document_names_the_evaluation_data() -> None:
    names = _private_names()
    found = []
    for path in _documents():
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        hits = sorted(name for name in names if name in lowered)
        if hits:
            found.append(f"{path.relative_to(ROOT).as_posix()}: {len(hits)} name(s)")
    # The names themselves are not printed: a failure report may be pasted anywhere.
    assert not found, "names of private data in documents (compare with docs/agents/eval/local/*.json): " + "; ".join(found)


def test_no_file_that_can_be_committed_names_the_frames_of_a_private_case() -> None:
    """Code and tests too: the sample names, the case's own name and the calibration it expects."""
    names = {name.encode("utf-8") for name in _private_names(data_folder=False)}
    if not names:
        pytest.skip(NO_LOCAL_CASE)
    if not _has_git():
        pytest.skip("not a git checkout: what is committed is not known here")
    found = []
    for name in _committable():
        path = ROOT / name
        if not path.is_file():
            continue  # deleted in the working tree
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue  # an image, a model: not text
        lowered = data.lower()
        hits = sum(private in lowered for private in names)
        if hits:
            found.append(f"{name}: {hits} name(s)")
    assert not found, "names of private data in files (compare with docs/agents/eval/local/*.json): " + "; ".join(found)


def test_no_file_that_can_be_committed_writes_out_the_numbers_of_a_private_case() -> None:
    """Peak positions (also rounded), the length of the series, the numbered samples: tests and comments use invented ones."""
    if not _private_cases():
        pytest.skip(NO_LOCAL_CASE)
    if not _has_git():
        pytest.skip("not a git checkout: what is committed is not known here")
    found = _files_writing_private_numbers()
    # The numbers themselves are not printed: a failure report may be pasted anywhere.
    assert not found, ("numbers of private data in files: a peak position, the length of the series or a numbered sample "
                       "(compare with docs/agents/eval/local/*.json and use invented values): " + "; ".join(found))


def test_the_private_cases_are_out_of_reach_of_git() -> None:
    assert "docs/agents/eval/local/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if not _has_git():
        pytest.skip("not a git checkout")
    probe = subprocess.run(["git", "check-ignore", "-q", "docs/agents/eval/local/any-case.json"], cwd=ROOT, timeout=120)
    assert probe.returncode == 0, "git does not ignore docs/agents/eval/local/"
    tracked = [name for name in _git_files("--", "docs/agents/eval/local") if (ROOT / name).is_file()]
    assert not tracked, "git tracks files of docs/agents/eval/local/"
