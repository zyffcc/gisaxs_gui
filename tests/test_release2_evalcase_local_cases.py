"""Evaluation cases on private data stay on the computer that holds the data (release 2, EVALCASE).

``tools/eval_giwaxs_agent.py`` reads its cases from ``docs/agents/eval/local/``, a folder git ignores,
and says in a sentence what to do when there is none. Everything here runs on invented cases written
into a temporary folder, so it passes on a clone. One test runs the real local case(s): only on request
(``GIMAP_EVAL_LOCAL=1``) and only where both a case and its data are on this computer.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import tools.eval_giwaxs_agent as tool
from tests.test_giwaxs_eval import INVENTED_CASE, _report

ROOT = Path(__file__).resolve().parents[1]
README = tool.CASES_DIR / "README.md"
PUBLIC = ROOT / "tests" / "data" / "external" / "giwaxs_p08_mapi"
FILM = INVENTED_CASE["name"]
NO_LOCAL_CASE = "no private evaluation case on this computer (docs/agents/eval/local/ is empty on a clone)"
ON_REQUEST = "GIMAP_EVAL_LOCAL"
"""Set to 1 to let the suite run the local case(s) on their data: the suite does not read private data unasked."""
NOT_ASKED = (f"the regression on private data runs on request: set {ON_REQUEST}=1, "
             "or run `python tools/eval_giwaxs_agent.py baseline`")


@pytest.fixture
def local(tmp_path: Path, monkeypatch) -> Path:
    """An empty folder in the place of docs/agents/eval/local/."""
    folder = tmp_path / "eval" / "local"
    folder.mkdir(parents=True)
    monkeypatch.setattr(tool, "LOCAL_CASES", folder)
    return folder


def _write_case(folder: Path, stem: str, **changes) -> Path:
    case = copy.deepcopy(INVENTED_CASE)
    case.update(changes)
    path = folder / f"{stem}.json"
    path.write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _good() -> list[dict]:
    return [_report("runA_0001"), _report("runB_0002")]


def _write_run(folder: Path, reports: list[dict] | None = None) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(_good() if reports is None else reports, ensure_ascii=False), encoding="utf-8")
    return str(folder)


def _powder(folder: Path) -> Path:
    baseline = dict(INVENTED_CASE["baseline"], standard="agbh")
    return _write_case(folder, "invented-powder", name="Invented: a powder", baseline=baseline)


# -- no case ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("arguments", [["baseline"], ["baseline", "--from", "anywhere"], ["report", "report.md"]])
def test_without_a_case_the_tool_says_what_a_case_is_and_where_one_goes(local: Path, capsys, arguments) -> None:
    assert tool.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "Traceback" not in captured.err
    assert captured.err.strip() == tool.NO_CASE
    for words in ("No evaluation case on this computer", "A case is a JSON file", "docs/agents/eval/local/", "--case",
                  "docs/agents/eval/README.md", "git ignores that folder", "private or unpublished", "never committed"):
        assert words in tool.NO_CASE


def test_a_checkout_without_the_folder_is_a_checkout_without_a_case(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(tool, "LOCAL_CASES", tmp_path / "eval" / "local")  # a clone: the folder is not there
    assert tool.local_cases() == []
    assert tool.main(["baseline"]) == 2 and capsys.readouterr().err.strip() == tool.NO_CASE
    with pytest.raises(tool.NothingToEvaluate):
        tool.find_cases()


def test_the_help_and_the_command_line_say_the_same(tmp_path: Path) -> None:
    """As a process: the help names the folder; an unknown case is one sentence and exit code 2."""
    script = str(ROOT / "tools" / "eval_giwaxs_agent.py")

    def run(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, script, *arguments], cwd=tmp_path, capture_output=True, timeout=120,
                              encoding="utf-8", errors="replace")

    shown = run("--help")
    assert shown.returncode == 0
    for words in ("docs/agents/eval/local/", "a folder git ignores", "never committed", "docs/agents/eval/README.md",
                  "2 nothing was evaluated"):
        assert words in shown.stdout
    missing = run("--case", "no-such-case", "report", "report.md")
    assert missing.returncode == 2 and missing.stdout == "" and "Traceback" not in missing.stderr
    assert missing.stderr.strip().startswith("No such case: no-such-case (neither a file, nor the name of one in ")
    assert len(missing.stderr.strip().splitlines()) == 1


# -- one case, several cases -------------------------------------------------------------------------


def test_the_one_local_case_is_used_without_naming_it(local: Path, tmp_path: Path, capsys) -> None:
    _write_case(local, "invented-film")
    assert tool.main(["baseline", "--from", _write_run(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    assert f"case: {FILM}" in out and "12/12 checks pass" in out and "FAIL" not in out

    wrong = [_report("runA_0001"), _report("runB_0002", series_hints=["q = 1.2, 2.4 Å⁻¹ follow a cubic series"])]
    assert tool.main(["baseline", "--from", _write_run(tmp_path / "wrong", wrong)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  no false series: q = 1.2, 2.4 Å⁻¹ follow a cubic series" in out and "11/12 checks pass" in out

    report = tmp_path / "report.md"
    report.write_text("The calibration used LaB6. The 4.2 peak is a sharp Bragg reflection.", encoding="utf-8")
    assert tool.main(["report", str(report)]) == 1
    out = capsys.readouterr().out
    assert "? artefact_called_a_peak — One bin wide, in every sample." in out
    assert "“The 4.2 peak is a sharp Bragg reflection.”" in out and "baseline facts relayed: 1/2" in out
    report.write_text("4.2 is a spike in every sample: an artefact. A shadow hides the high angles.", encoding="utf-8")
    assert tool.main(["report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "none found" in out and "findings beyond the baseline: 4/6 points" in out


def test_with_several_cases_a_saved_run_or_a_report_needs_one_to_be_named(local: Path, tmp_path: Path, capsys) -> None:
    film = _write_case(local, "invented-film")
    _powder(local)
    run = _write_run(tmp_path / "run")
    report = tmp_path / "report.md"
    report.write_text("4.2 is a spike.", encoding="utf-8")
    for arguments, why in ((["baseline", "--from", run], "--from holds the run of one case"),
                           (["report", str(report)], "a report is scored against one case")):
        assert tool.main(arguments) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.strip() == (
            f"2 cases on this computer (invented-film.json, invented-powder.json): {why}, so name it with --case.")

    # A name in the folder (with or without .json) or a path; before or after the command.
    for arguments in (["--case", "invented-film", "baseline", "--from", run],
                      ["baseline", "--case", "invented-film.json", "--from", run],
                      ["baseline", "--from", run, "--case", str(film)]):
        assert tool.main(arguments) == 0
        assert f"case: {FILM}" in capsys.readouterr().out
    assert tool.main(["baseline", "--from", run, "--case", "invented-powder"]) == 1  # it expects another standard
    out = capsys.readouterr().out
    assert "case: Invented: a powder" in out and "FAIL  standard: lab6" in out
    assert tool.main(["report", str(report), "--case", "invented-film"]) == 0

    elsewhere = _write_case(tmp_path, "kept-somewhere-else", name="Invented: kept elsewhere")
    assert tool.main(["--case", str(elsewhere), "baseline", "--from", run]) == 0
    assert "case: Invented: kept elsewhere" in capsys.readouterr().out
    assert tool.main(["--case", "no-such-case", "baseline"]) == 2
    assert capsys.readouterr().err.startswith("No such case: no-such-case ")


def test_the_baseline_runs_every_local_case_into_a_folder_of_its_own(local: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    _write_case(local, "invented-film")
    _powder(local)
    calls: list[tuple[str, str, Path]] = []
    results = {FILM: _good(), "Invented: a powder": [_report("runA_0001", calibration_quality={"standard": "agbh", "mean_line_q_error_percent": 0.08})]}

    def run_baseline(case: dict, out: Path) -> list[dict]:
        calls.append((case["name"], case["notes"], out))
        return results[case["name"]]

    monkeypatch.setattr(tool, "run_baseline", run_baseline)
    out = tmp_path / "out"
    assert tool.main(["baseline", "--out", str(out)]) == 0
    assert calls == [(FILM, "GIWAXS, alpha_i = 0.25 deg", out / "invented-film"),
                     ("Invented: a powder", "GIWAXS, alpha_i = 0.25 deg", out / "invented-powder")]
    printed = capsys.readouterr().out
    assert printed.count("12/12 checks pass") == 2 and f"outputs: {out / 'invented-powder'}" in printed

    calls.clear()
    assert tool.main(["baseline", "--case", "invented-film", "--out", str(out)]) == 0
    assert [call[2] for call in calls] == [out]  # one case: the folder itself

    import tempfile

    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix="": str(tmp_path / f"{prefix}made"))
    calls.clear()
    assert tool.main(["--case", "invented-film", "baseline"]) == 0
    assert [call[2] for call in calls] == [tmp_path / "gimap_eval_made"]  # without --out: a temporary folder

    results[FILM] = [_report("runA_0001", needs_attention=[{"question": "which energy?"}])]
    capsys.readouterr()
    assert tool.main(["baseline", "--out", str(out)]) == 1  # one case fails: the other is still run
    printed = capsys.readouterr().out
    assert "FAIL  open questions: [1]" in printed and printed.count("First level") == 2


def test_data_that_are_elsewhere_are_a_sentence_and_exit_code_two(local: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    _write_case(local, "invented-film", data_root=str(tmp_path / "not_on_this_computer"))
    out = tmp_path / "out"
    assert tool.main(["baseline", "--out", str(out)]) == 2  # the real run_baseline: it stops before any analysis
    captured = capsys.readouterr()
    assert captured.out == "" and "Traceback" not in captured.err and not out.exists()
    assert captured.err.startswith("The case's data are not on this computer: ")
    assert "abc_film_runA_0001_00001.nxs (2 of 2 frames missing)." in captured.err
    import tempfile

    made = tmp_path / "gimap_eval_made"
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix="": made.mkdir() or str(made))
    assert tool.main(["baseline"]) == 2 and not made.exists()  # without --out: no empty temporary folder is left
    assert "The case's data are not on this computer" in capsys.readouterr().err
    _write_case(local, "invented-film", frames=[])
    assert tool.main(["baseline"]) == 2 and capsys.readouterr().err.strip() == 'The case lists no "frames".'

    # Several cases: the one without its data is skipped with a line, the others are evaluated.
    _write_case(local, "invented-film", data_root=str(tmp_path / "not_on_this_computer"))
    _powder(local)
    real = tool.run_baseline
    powder = [_report("runA_0001", calibration_quality={"standard": "agbh", "mean_line_q_error_percent": 0.08})]
    monkeypatch.setattr(tool, "run_baseline", lambda case, folder: real(case, folder) if case["name"] == FILM else powder)
    assert tool.main(["baseline", "--out", str(out)]) == 2  # nothing failed, but not everything was evaluated
    captured = capsys.readouterr()
    assert "case file invented-film.json: not evaluated. The case's data are not on this computer" in captured.err
    assert "case: Invented: a powder" in captured.out and "12/12 checks pass" in captured.out
    powder[0]["series_hints"] = ["a cubic series"]
    assert tool.main(["baseline", "--out", str(out)]) == 1  # a failed check counts more than a skipped case


def test_a_broken_case_or_run_is_a_sentence_too(local: Path, tmp_path: Path, capsys) -> None:
    run = _write_run(tmp_path / "run")
    report = tmp_path / "report.md"
    report.write_text("Nothing to say.", encoding="utf-8")
    broken = local / "broken.json"

    def sentence(arguments: list[str]) -> str:
        assert tool.main(arguments) == 2
        captured = capsys.readouterr()
        assert captured.out == "" and "Traceback" not in captured.err
        return captured.err.strip()

    broken.write_text("{ not json", encoding="utf-8")
    assert sentence(["baseline", "--from", run]).startswith(f"The case is not valid JSON: {broken} (")
    broken.write_text("[1, 2]", encoding="utf-8")
    assert sentence(["report", str(report)]) == f"The case is not a JSON object: {broken}."
    broken.write_text('{"name": "Invented: half a case"}', encoding="utf-8")
    assert sentence(["baseline", "--from", run]) == 'The case broken.json has no "baseline" part (see docs/agents/eval/README.md).'
    assert sentence(["report", str(report)]) == 'The case broken.json has no "report" part (see docs/agents/eval/README.md).'

    broken.unlink()
    _write_case(local, "invented-film")
    assert sentence(["baseline", "--from", str(tmp_path / "no_run")]).startswith("The summary of the earlier run cannot be read: ")
    (tmp_path / "run" / "summary.json").write_text('{"ok": true}', encoding="utf-8")
    assert sentence(["baseline", "--from", run]).startswith("The summary of the earlier run is not a list of reports: ")
    assert sentence(["report", str(tmp_path / "no_report.md")]).startswith("The report cannot be read: ")
    report.write_text(" \n", encoding="utf-8")  # the wrong file, a run that wrote nothing: not "no mistake found"
    assert sentence(["report", str(report)]) == f"The report is empty: {report}."


def test_files_as_windows_tools_save_them_are_read(local: Path, tmp_path: Path, capsys) -> None:
    """A case edited on another computer, a report redirected into a file: Windows adds a byte order mark,
    and Windows PowerShell 5.1 writes UTF-16. Read as plain UTF-8, such a report matched no pattern at all
    and passed for a report without mistakes."""
    case = _write_case(local, "invented-film")
    plain = case.read_bytes()
    run = _write_run(tmp_path / "run")
    report = tmp_path / "report.md"
    english = "The calibration used LaB6. The 4.2 peak is a sharp Bragg reflection."
    chinese = "4.2 处是一个衍射峰。标样是六硼化镧。"

    def scored(data: bytes) -> tuple[int, str]:
        report.write_bytes(data)
        code = tool.main(["report", str(report)])
        return code, capsys.readouterr().out

    for saved in (b"\xef\xbb\xbf" + plain, plain.decode("utf-8").encode("utf-16")):  # the case itself
        case.write_bytes(saved)
        assert tool.main(["baseline", "--from", run]) == 0
        assert f"case: {FILM}" in capsys.readouterr().out
        assert scored(english.encode("utf-8"))[0] == 1
    case.write_bytes(plain)

    for text in (english, chinese):
        for data in (text.encode("utf-8"), b"\xef\xbb\xbf" + text.encode("utf-8"), text.encode("utf-16"),
                     b"\xfe\xff" + text.encode("utf-16-be")):
            code, out = scored(data)
            assert code == 1 and "? artefact_called_a_peak" in out and "baseline facts relayed: 1/2" in out
    assert "“4.2 处是一个衍射峰。”" in scored(b"\xef\xbb\xbf" + chinese.encode("utf-8"))[1]  # the mark is not in the sentence

    # Another encoding is refused: it is not scored as a report that says nothing.
    report.write_bytes(chinese.encode("gbk"))
    assert tool.main(["report", str(report)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.strip() == f"The report is not UTF-8 text: {report} (save it as UTF-8)."
    case.write_bytes(plain.decode("utf-8").encode("gbk"))  # its patterns hold Chinese words
    assert tool.main(["baseline", "--from", run]) == 2
    assert capsys.readouterr().err.strip() == f"The case is not UTF-8 text: {case} (save it as UTF-8)."


def _rule_changes() -> list[tuple[dict, str]]:
    def rubric(group: str, **changes) -> dict:
        made = copy.deepcopy(INVENTED_CASE["report"])
        made[group][0] = {key: value for key, value in {**made[group][0], **changes}.items() if value is not None}
        return made

    one_backslash = json.loads('"(?i)' + chr(92) + "bno" + chr(92) + 'b"')  # JSON makes a backspace of it, not a word boundary
    assert one_backslash == "(?i)" + chr(8) + "no" + chr(8)
    return [
        (rubric("mistakes", unless=one_backslash),
         f'rule "cubic_series" of "mistakes": {one_backslash!r} holds a control character; in JSON a backslash is written twice'),
        (rubric("mistakes", all=["(unclosed"]),
         "rule \"cubic_series\" of \"mistakes\": '(unclosed' is not a regular expression (missing ), unterminated subpattern at position 0)"),
        (rubric("mistakes", unless="[a-"), "rule \"cubic_series\" of \"mistakes\": '[a-' is not a regular expression ("),
        (rubric("mistakes", all="(?i)cubic"), 'rule "cubic_series" of "mistakes": "all" is not a list of patterns'),
        (rubric("baseline_facts", any=[7]), 'rule "calibration_stated" of "baseline_facts": 7 is not a regular expression ('),
        (rubric("baseline_facts", any=None), 'rule "calibration_stated" of "baseline_facts" has no pattern in "any" or "all"'),
        (rubric("findings", id=None), 'rule 1 of "findings" has no "id"'),
        (rubric("findings", points="two"), 'rule "start_and_end_compared" of "findings": "points" is not a whole number'),
        ({"mistakes": {"id": "not a list"}}, '"mistakes" is not a list of rules'),
        ({"findings": ["shadow"]}, 'rule 1 of "findings" has no "id"'),
    ]


@pytest.mark.parametrize(("rubric", "problem"), _rule_changes())
def test_a_rule_the_scoring_would_misread_is_refused(local: Path, tmp_path: Path, capsys, rubric: dict, problem: str) -> None:
    """Rules are regular expressions written by hand inside JSON: a slip is named, not a traceback.

    A string in the place of a list would be matched letter by letter, a rule without a pattern in
    every sentence: both would be scored without a word.
    """
    assert tool.rubric_problem(rubric).startswith(problem)
    _write_case(local, "invented-film", report=rubric)
    report = tmp_path / "report.md"
    report.write_text("There is no cubic series.", encoding="utf-8")
    assert tool.main(["report", str(report)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "Traceback" not in captured.err
    assert captured.err.startswith(f'The "report" part of the case invented-film.json cannot be used: {problem}')
    assert captured.err.strip().endswith("(see docs/agents/eval/README.md).")


def test_the_rules_of_a_good_case_are_accepted() -> None:
    assert tool.rubric_problem(INVENTED_CASE["report"]) is None and tool.rubric_problem(_readme_example()["report"]) is None
    assert tool.rubric_problem({}) is None  # a case may leave a group out
    for path in tool.local_cases():  # where a private case exists: it is still accepted (nothing of it is printed)
        assert tool.rubric_problem(tool.case_part(tool.load_case(path), "report", path)) is None, "a local case is refused"


@pytest.mark.parametrize(("changes", "error"), [
    ({"ring": {"q": 2.2, "f_min": 0.3}}, "KeyError: 'f_max'"),  # half a range
    ({"ring": {"f_min": 0.3, "f_max": 0.6}}, "KeyError: 'q'"),
    ({"reliable_peaks": "1.5"}, "TypeError: "),
    ({"max_line_error_percent": "small"}, "ValueError: "),
])
def test_a_baseline_part_that_cannot_be_checked_is_a_sentence(
        local: Path, tmp_path: Path, monkeypatch, capsys, changes: dict, error: str) -> None:
    _write_case(local, "invented-film", baseline={**INVENTED_CASE["baseline"], **changes})
    assert tool.main(["baseline", "--from", _write_run(tmp_path / "run")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "Traceback" not in captured.err
    assert captured.err.startswith(f"The checks of the case invented-film.json cannot be made ({error}")
    assert captured.err.strip().endswith('its "baseline" part, or the summary of the run, is not what docs/agents/eval/README.md describes.')

    # With other cases next to it, it is skipped with that sentence and the others are checked.
    _powder(local)
    powder = [_report("runA_0001", calibration_quality={"standard": "agbh", "mean_line_q_error_percent": 0.08})]
    monkeypatch.setattr(tool, "run_baseline", lambda case, out: powder)
    assert tool.main(["baseline", "--out", str(tmp_path / "out")]) == 2
    captured = capsys.readouterr()
    assert "case file invented-film.json: not evaluated. The checks of the case invented-film.json cannot be made" in captured.err
    assert "case: Invented: a powder" in captured.out and "12/12 checks pass" in captured.out


def test_an_earlier_run_is_read_not_written(local: Path, tmp_path: Path, capsys) -> None:
    _write_case(local, "invented-film")
    run = Path(_write_run(tmp_path / "run"))
    assert tool.main(["baseline", "--from", str(run / "summary.json")]) == 0  # the file itself is fine too
    assert "12/12 checks pass" in capsys.readouterr().out
    (run / "summary.json").write_text('["report.md", 3]', encoding="utf-8")  # another list
    assert tool.main(["baseline", "--from", str(run)]) == 2
    assert capsys.readouterr().err.startswith("The summary of the earlier run is not a list of reports: ")
    (run / "summary.json").write_text("[]", encoding="utf-8")  # a run of no frame: nothing in it can fail a check
    assert tool.main(["baseline", "--from", str(run)]) == 2
    assert capsys.readouterr().err.startswith("The summary of the earlier run holds no report: ")
    _write_run(run, [dict(_report("runA_0001"), peaks=[{"caveat": ""}])])  # a summary of something else
    assert tool.main(["baseline", "--from", str(run)]) == 2
    assert "cannot be made (KeyError: 'q')" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stopped:  # --out would be ignored: say so
        tool.main(["baseline", "--from", str(run), "--out", str(tmp_path / "out")])
    assert stopped.value.code == 2 and "argument --out: not allowed with argument --from" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    _write_case(local, "invented-film", frames="abc_film_runA_0001.nxs")  # one name, not a list of names
    assert tool.main(["baseline"]) == 2 and capsys.readouterr().err.strip() == 'The case lists no "frames".'


# -- where cases live --------------------------------------------------------------------------------


def test_the_tool_names_no_case_of_its_own() -> None:
    source = Path(tool.__file__).read_text(encoding="utf-8")
    assert set(re.findall(r"[\w-]+\.json", source)) == {"summary.json"}  # no case file is written into the tool
    assert not hasattr(tool, "DEFAULT_CASE")
    assert tool.LOCAL_CASES == ROOT / "docs" / "agents" / "eval" / "local" and tool.CASES_DIR == tool.LOCAL_CASES.parent
    assert "docs/agents/eval/local/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for path in tool.local_cases():  # where a private case exists: its name is not in what is committed
        assert path.stem.lower() not in source.lower()
        assert path.stem.lower() not in README.read_text(encoding="utf-8").lower()


def test_a_relative_data_root_is_taken_from_the_project_folder(tmp_path: Path) -> None:
    public = {"data_root": "tests/data/external/giwaxs_p08_mapi", "frames": ["S121_MAI_A2_00841.tif"]}
    assert tool.case_frames(public) == [PUBLIC / "S121_MAI_A2_00841.tif"]
    private = {"data_root": str(tmp_path / "beamtime"), "frames": ["abc_film_runA_0001/det/x.nxs", "y.nxs"]}
    assert tool.case_frames(private) == [tmp_path / "beamtime" / "abc_film_runA_0001" / "det" / "x.nxs", tmp_path / "beamtime" / "y.nxs"]
    assert tool.case_frames({"data_root": "anywhere"}) == []


def test_a_case_that_is_committed_stands_on_data_inside_the_repository() -> None:
    """Next to the README only cases on public data: a relative ``data_root`` and frames that are here."""
    for path in sorted(tool.CASES_DIR.glob("*.json")):
        case = tool.load_case(path)
        root = Path(str(case.get("data_root") or ""))
        assert not root.is_absolute(), f"{path.name}: a case on data outside the repository belongs in local/"
        assert ROOT.resolve() in (ROOT / root).resolve().parents, f"{path.name}: data_root leaves the repository"
        assert all(frame.is_file() for frame in tool.case_frames(case)), f"{path.name}: frames are missing"


# -- the README --------------------------------------------------------------------------------------


def _readme_example() -> dict:
    [block] = re.findall(r"```json\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)
    return json.loads(block)


def test_the_readme_example_is_a_case_the_tool_accepts() -> None:
    example = _readme_example()
    assert set(example) == {"name", "data_root", "frames", "notes", "baseline", "report"}
    assert tool.case_part(example, "baseline") is example["baseline"] and len(tool.case_frames(example)) == 2
    ring = {"q": 2.2, "herman": 0.45, "notes": []}
    reports = [_report("runA_0001", rings=[ring]), _report("runB_0002", rings=[ring])]
    checks = tool.check_baseline(reports, example["baseline"])
    assert len(checks) == 10 and all(passed for _name, passed, _detail in checks)
    withheld = [_report("runA_0001"), _report("runB_0002")]  # the example asks for f inside a range
    assert [name for name, passed, _detail in tool.check_baseline(withheld, example["baseline"]) if not passed] == ["ring near q = 2.2"]

    score = tool.score_report("The standard was LaB6. The 4.2 peak is a Bragg reflection.", example["report"])
    assert set(score["mistakes"]) == {"artefact_called_a_peak"} and score["facts"] == {"calibration_stated": True}
    score = tool.score_report("4.2 是伪影，不是衍射峰。第 1 帧没有 2.2，之后才出现。", example["report"])
    assert score["mistakes"] == {} and score["points"] == 2 == score["max_points"]


def test_the_readme_documents_every_key_the_tool_reads() -> None:
    source = Path(tool.__file__).read_text(encoding="utf-8")
    keys = set(re.findall(r"""\b(?:case|expected|ring|rule|rubric)(?:\.get\(|\[)["']([a-z_]+)["']""", source))
    keys |= set(re.findall(r'"([a-z_]+)" in (?:expected|ring|rule)\b', source))
    keys |= set(re.findall(r'case_part\(.*?, "([a-z_]+)"', source))
    assert keys >= {"name", "data_root", "frames", "notes", "baseline", "report", "geometry_source_contains", "standard",
                    "max_line_error_percent", "reliable_peaks", "peak_tolerance", "spike", "no_series_hints", "ring", "q",
                    "f_min", "f_max", "note_contains", "sector_note_contains", "open_questions", "mistakes",
                    "baseline_facts", "findings", "id", "any", "all", "unless", "why", "points"}  # the patterns still find them
    text = README.read_text(encoding="utf-8")
    assert not sorted(key for key in keys if f"`{key}`" not in text), "keys the tool reads that the README does not explain"
    for words in ("docs/agents/eval/local/", "被 git 忽略", "不提交", "编造的", "--case", "退出码"):
        assert words in text


# -- real data ---------------------------------------------------------------------------------------


def test_the_baseline_of_a_case_runs_on_a_public_frame(tmp_path: Path, capsys) -> None:
    """The whole way once, on data every clone has: a case, the headless pipeline, the checks."""
    if not (PUBLIC / "S121_MAI_A2_00841.tif").is_file():
        pytest.skip("the public GIWAXS frame is not in this checkout")
    before = sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in PUBLIC.iterdir())
    case = tmp_path / "public-frame.json"
    case.write_text(json.dumps({
        "name": "Public: the P08 frame of tests/data/external",
        "data_root": "tests/data/external/giwaxs_p08_mapi",
        "frames": ["S121_MAI_A2_00841.tif"],
        "notes": "GIWAXS, alpha_i = 0.075 deg, calibration LaB6_2021_12_DESY_P08.poni",
        "baseline": {"geometry_source_contains": "LaB6_2021_12_DESY_P08.poni"},
    }), encoding="utf-8")
    out = tmp_path / "out"
    code = tool.main(["--case", str(case), "baseline", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "case: Public: the P08 frame of tests/data/external" in printed
    assert "PASS  every frame analysed: all OK" in printed and "PASS  calibration file: " in printed
    # A .poni is adopted as it is: no standard image was fitted, so there is no line error to pass.
    assert code == 1 and "FAIL  line error: [None] %" in printed and "3/4 checks pass" in printed
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))[0]["ok"] is True
    assert (out / "S121_MAI_A2_00841" / "report.md").is_file()
    assert sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in PUBLIC.iterdir()) == before


def _local_cases_with_their_data() -> list[Path]:
    """The local cases whose frames are on this computer; skips the test when there is none or nobody asked."""
    if os.environ.get(ON_REQUEST) != "1":
        pytest.skip(NOT_ASKED)
    cases = tool.local_cases()
    if not cases:
        pytest.skip(NO_LOCAL_CASE)
    here = [path for path in cases if (frames := tool.case_frames(tool.load_case(path))) and all(frame.is_file() for frame in frames)]
    if not here:
        pytest.skip("the data of the local evaluation case(s) are not on this computer")
    return here


def test_a_local_case_passes_its_baseline_where_its_data_are(tmp_path: Path, capsys) -> None:
    """The regression on real data, on request: read-only, outputs in a temporary folder (a minute and a half).

    It is not part of an ordinary run of the suite: a case on private data is judged by the person who owns
    the data, with the tool, and the suite reads no private data unasked.
    Nothing of the case is printed when it fails (a failure report may be pasted anywhere):
    ``python tools/eval_giwaxs_agent.py baseline`` shows the checks.
    """
    for number, path in enumerate(_local_cases_with_their_data(), start=1):
        case = tool.load_case(path)
        reports = tool.run_baseline(case, tmp_path / f"case_{number}")
        checks = tool.check_baseline(reports, tool.case_part(case, "baseline", path))
        capsys.readouterr()  # the progress lines name the frames
        failed = sum(not passed for _name, passed, _detail in checks)
        assert failed == 0, (f"{failed} of {len(checks)} checks of local case {number} fail: "
                             "run `python tools/eval_giwaxs_agent.py baseline` to see them")


def test_the_test_on_real_data_is_skipped_with_a_reason_and_runs_when_it_can(local: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(ON_REQUEST, raising=False)
    with pytest.raises(pytest.skip.Exception, match="runs on request: set GIMAP_EVAL_LOCAL=1"):
        _local_cases_with_their_data()
    monkeypatch.setenv(ON_REQUEST, "1")
    with pytest.raises(pytest.skip.Exception, match="no private evaluation case on this computer"):
        _local_cases_with_their_data()
    data = tmp_path / "beamtime"
    path = _write_case(local, "invented-film", data_root=str(data))
    with pytest.raises(pytest.skip.Exception, match="the data of the local evaluation case"):
        _local_cases_with_their_data()
    for frame in INVENTED_CASE["frames"]:
        (data / frame).parent.mkdir(parents=True)
        (data / frame).write_bytes(b"")
    assert _local_cases_with_their_data() == [path]

    # The test itself, with the analysis replaced: it passes on a good run, and a failure names nothing of the case.
    runs: list[Path] = []
    monkeypatch.setattr(tool, "run_baseline", lambda case, out: runs.append(out) or _good())
    test_a_local_case_passes_its_baseline_where_its_data_are(tmp_path, capsys)
    assert runs == [tmp_path / "case_1"]
    wrong = [_report("runA_0001", series_hints=["q = 1.2, 2.4 Å⁻¹ follow a cubic series"]), _report("runB_0002")]
    monkeypatch.setattr(tool, "run_baseline", lambda case, out: wrong)
    with pytest.raises(AssertionError) as failure:
        test_a_local_case_passes_its_baseline_where_its_data_are(tmp_path, capsys)
    message = str(failure.value)
    assert message.startswith("1 of 12 checks of local case 1 fail: run `python tools/eval_giwaxs_agent.py baseline`")
    assert not any(private in message for private in ("abc_", "invented-film", "cubic", "1.2", "Invented"))
