"""Evaluate GIWAXS agents on a case: the baseline must not err, a capable agent should find more.

A case is one JSON file: the frames of a measurement somebody knows well, the facts the baseline
must get right on them, and the patterns that score an agent's written report.  The format is
described in docs/agents/eval/README.md.

    python tools/eval_giwaxs_agent.py baseline [--case CASE] [--from DIR | --out DIR]
        runs `gimap_agent.py auto` on the case's frames (or reads DIR/summary.json) and checks
        the facts the case fixes: calibration, line error, reliable peaks, flagged spike, ring f,
        no false series, no open questions.
    python tools/eval_giwaxs_agent.py report REPORT.md [--case CASE]
        scores an agent's written report: mistakes (must be absent; each hit is printed for a
        person to confirm, negated sentences are skipped), baseline facts relayed, and findings
        beyond the baseline (points).

Cases are read from docs/agents/eval/local/, a folder git ignores.  A case names real files and
what was measured on them, so a case built on private or unpublished data stays on the computer
that holds the data and is never committed.  Without --case every *.json in that folder is used:
`baseline` runs them all, while `--from DIR` and `report` belong to one case, so name it when
there are several.  --case takes a path, or the name of a file in that folder.

Exit code: 0 every check passes (`report`: no mistake found); 1 a check fails (a mistake is
found); 2 nothing was evaluated: no case on this computer, its data are not here, or a case or
a report that cannot be used (then a sentence says why).  1 wins over 2.  Cases and reports are
read as UTF-8; the UTF-16 that Windows PowerShell's `>` writes is read too.  Regular expressions
are a first pass, not a judge: read the report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = PROJECT_ROOT / "docs" / "agents" / "eval"
LOCAL_CASES = CASES_DIR / "local"
"""Where cases are looked for. git ignores it: cases on private data are never committed."""
README = "docs/agents/eval/README.md"
_SENTENCE = re.compile(r"(?<=[。！？!?\n])|(?<=\.)\s+(?=[A-Z0-9*(|])")  # a semicolon stays inside its sentence

NO_CASE = (
    "No evaluation case on this computer: nothing was evaluated.\n"
    "A case is a JSON file that names the frames of one measurement, the facts the baseline must get "
    "right on them, and the patterns that score a written report (format: docs/agents/eval/README.md).\n"
    "Put one in docs/agents/eval/local/, or give its path with --case. git ignores that folder: a case "
    "built on private or unpublished data stays on the computer that holds the data and is never committed."
)


class NothingToEvaluate(Exception):
    """No case, or not its data: the message is a sentence for the person at the command line."""


# -- cases -----------------------------------------------------------------------------------------


def local_cases(folder: Path | None = None) -> list[Path]:
    """The case files of this computer, by name."""
    folder = LOCAL_CASES if folder is None else Path(folder)
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def find_cases(named: str | None = None, folder: Path | None = None) -> list[Path]:
    """The case named (a path, or a file name in the local folder), otherwise every local case."""
    folder = LOCAL_CASES if folder is None else Path(folder)
    if named:
        for candidate in (Path(named), folder / named, folder / f"{named}.json"):
            if candidate.is_file():
                return [candidate]
        raise NothingToEvaluate(f"No such case: {named} (neither a file, nor the name of one in {folder}).")
    found = local_cases(folder)
    if not found:
        raise NothingToEvaluate(NO_CASE)
    return found


def _text(path: Path, what: str) -> str:
    """A file as text: UTF-8, with or without a byte order mark, or the UTF-16 Windows PowerShell's ``>`` writes.

    Anything else is refused: scored as it is, a report in another encoding would match no pattern and
    pass for a report without mistakes.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise NothingToEvaluate(f"{what} cannot be read: {path} ({error.strerror or error}).") from error
    try:
        return data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
    except UnicodeDecodeError as error:
        raise NothingToEvaluate(f"{what} is not UTF-8 text: {path} (save it as UTF-8).") from error


def load_case(path: Path) -> dict:
    try:
        case = json.loads(_text(path, "The case"))
    except ValueError as error:
        raise NothingToEvaluate(f"The case is not valid JSON: {path} ({error}).") from error
    if not isinstance(case, dict):
        raise NothingToEvaluate(f"The case is not a JSON object: {path}.")
    return case


def case_part(case: dict, key: str, path: Path | None = None) -> dict:
    """The "baseline" or "report" part of a case."""
    part = case.get(key)
    if not isinstance(part, dict):
        where = f" {Path(path).name}" if path else ""
        raise NothingToEvaluate(f'The case{where} has no "{key}" part (see {README}).')
    return part


def case_frames(case: dict) -> list[Path]:
    """The case's frames; a relative ``data_root`` is taken from the project folder (public test data)."""
    root = Path(str(case.get("data_root") or ""))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    frames = case.get("frames")
    return [root / str(frame) for frame in frames] if isinstance(frames, list) else []


# -- first level: the baseline --------------------------------------------------------------------


def check_baseline(reports: list[dict], expected: dict) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) for every fact the case fixes."""
    checks: list[tuple[str, bool, str]] = []
    checks.append(("every frame analysed", all(report.get("ok") for report in reports),
                   ", ".join(Path(report.get("frame") or "?").name for report in reports if not report.get("ok")) or "all OK"))
    sources = [str((report.get("geometry") or {}).get("source") or "") for report in reports]
    wanted = expected.get("geometry_source_contains", "")
    checks.append(("calibration file", all(wanted in source for source in sources), "; ".join(sorted(set(sources)))))
    qualities = [report.get("calibration_quality") or {} for report in reports]
    standards = {quality.get("standard") for quality in qualities}
    checks.append(("standard", standards == {expected.get("standard")}, ", ".join(map(str, standards))))
    errors = [quality.get("mean_line_q_error_percent") for quality in qualities]
    limit = float(expected.get("max_line_error_percent", 0.2))
    checks.append(("line error", all(error is not None and error <= limit for error in errors), f"{errors} % (≤ {limit} %)"))
    tolerance = float(expected.get("peak_tolerance", 0.01))
    for q in expected.get("reliable_peaks", []):
        missing = [
            Path(report.get("frame") or "?").name for report in reports
            if not any(abs(peak["q"] - q) <= tolerance * q and not peak.get("caveat") for peak in report.get("peaks") or [])
        ]
        checks.append((f"reliable peak near {q:g}", not missing, "missing in " + ", ".join(missing) if missing else "in every frame"))
    spike = expected.get("spike")
    if spike:
        unflagged = [
            Path(report.get("frame") or "?").name for report in reports
            if not any(abs(peak["q"] - spike) <= 0.001 * spike and "spike" in peak.get("flags", []) for peak in report.get("peaks") or [])
        ]
        checks.append((f"spike at {spike:g} flagged", not unflagged, "not flagged in " + ", ".join(unflagged) if unflagged else "flagged in every frame"))
    if expected.get("no_series_hints"):
        hints = [hint for report in reports for hint in report.get("series_hints") or []]
        checks.append(("no false series", not hints, "; ".join(hints) or "none"))
    ring = expected.get("ring")
    if ring:
        values, good = [], True
        for report in reports:
            near = [item for item in report.get("rings") or [] if abs(item["q"] - ring["q"]) <= 0.02 * ring["q"]]
            item = near[0] if near else None
            value = None if item is None else item.get("herman")
            if item is None:
                ok = False
            elif "f_min" in ring:
                ok = value is not None and ring["f_min"] <= value <= ring["f_max"]
            else:
                ok = value is None  # f must be withheld
            text = "" if item is None else " ".join(item.get("notes") or []) + " " + str(item.get("reason") or "")
            if ring.get("note_contains") and ring["note_contains"] not in text:
                ok = False
            good = good and ok
            values.append(value)
        wanted = f"{ring['f_min']}–{ring['f_max']}" if "f_min" in ring else "withheld"
        detail = f"f = {values} (expected {wanted}" + (f", notes saying '{ring['note_contains']}'" if ring.get("note_contains") else "") + ")"
        checks.append((f"ring near q = {ring['q']:g}", good, detail))
    text = expected.get("sector_note_contains")
    if text:
        silent = [
            Path(report.get("frame") or "?").name for report in reports
            if not any(text in str(peak.get("orientation") or "") for peak in report.get("peaks") or [])
        ]
        checks.append((f"sectors say '{text}'", not silent, "not in " + ", ".join(silent) if silent else "in every frame"))
    if "open_questions" in expected:
        counts = [len(report.get("needs_attention") or []) for report in reports]
        checks.append(("open questions", all(count <= expected["open_questions"] for count in counts), str(counts)))
    return checks


def run_baseline(case: dict, out: Path) -> list[dict]:
    frames = case_frames(case)
    if not frames:
        raise NothingToEvaluate('The case lists no "frames".')
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        raise NothingToEvaluate(
            f"The case's data are not on this computer: {missing[0]} ({len(missing)} of {len(frames)} frames missing)."
        )
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.gimap.app.headless_assistant import analyse_frames
    from src.gimap.features.assistant.application import PipelineOptions

    return analyse_frames([str(frame) for frame in frames], PipelineOptions(notes=case.get("notes", "")), out,
                          saved_profiles=False, progress=lambda text: print(text, file=sys.stderr, flush=True))


# -- both levels: an agent's written report ---------------------------------------------------------


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text) if part and part.strip()]


def _matches(rule: dict, sentence: str) -> bool:
    if "any" in rule and not any(re.search(pattern, sentence) for pattern in rule["any"]):
        return False
    if "all" in rule and not all(re.search(pattern, sentence) for pattern in rule["all"]):
        return False
    return not (rule.get("unless") and re.search(rule["unless"], sentence))


def _hits(rule: dict, parts: list[str]) -> list[str]:
    return [part for part in parts if _matches(rule, part)]


def score_report(text: str, rubric: dict) -> dict:
    """Mistakes (with the sentences to confirm), baseline facts relayed, findings beyond the baseline."""
    parts = sentences(text)
    mistakes = {rule["id"]: _hits(rule, parts) for rule in rubric.get("mistakes", [])}
    facts = {rule["id"]: bool(_hits(rule, parts)) for rule in rubric.get("baseline_facts", [])}
    findings = {rule["id"]: (_hits(rule, parts), int(rule.get("points", 1))) for rule in rubric.get("findings", [])}
    return {
        "mistakes": {key: hits for key, hits in mistakes.items() if hits},
        "facts": facts,
        "findings": {key: bool(hits) for key, (hits, _points) in findings.items()},
        "points": sum(points for hits, points in findings.values() if hits),
        "max_points": sum(points for _hits, points in findings.values()),
        "why": {rule["id"]: rule.get("why", "") for rule in rubric.get("mistakes", [])},
    }


def rubric_problem(rubric: dict) -> str | None:
    """What the scoring would misread in the "report" part of a case, or None. Rules are written by hand."""
    for group in ("mistakes", "baseline_facts", "findings"):
        rules = rubric.get(group, [])
        if not isinstance(rules, list):
            return f'"{group}" is not a list of rules'
        for number, rule in enumerate(rules, start=1):
            if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
                return f'rule {number} of "{group}" has no "id"'
            name = f'rule "{rule["id"]}" of "{group}"'
            patterns: list = []
            for key in ("any", "all"):
                if not isinstance(rule.get(key, []), list):
                    return f'{name}: "{key}" is not a list of patterns'
                patterns += rule.get(key, [])
            if not patterns:
                return f'{name} has no pattern in "any" or "all"'
            for pattern in patterns + ([rule["unless"]] if rule.get("unless") else []):
                try:
                    re.compile(pattern)
                except (re.error, TypeError) as error:
                    return f"{name}: {pattern!r} is not a regular expression ({error})"
                if "\b" in pattern or "\f" in pattern:  # JSON reads "\b" as a backspace: the rule would never match
                    return (f"{name}: {pattern!r} holds a control character; in JSON a backslash is written twice, "
                            'a word boundary as "\\\\b"')
            if not isinstance(rule.get("points", 1), int):
                return f'{name}: "points" is not a whole number'
    return None


# -- command line ----------------------------------------------------------------------------------


def _print_baseline(checks: list[tuple[str, bool, str]]) -> int:
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    failed = sum(not passed for _name, passed, _detail in checks)
    print(f"\nFirst level (the baseline must not err): {len(checks) - failed}/{len(checks)} checks pass")
    return 0 if failed == 0 else 1


def _print_report(score: dict) -> int:
    print("Mistakes (must be none; confirm each sentence yourself):")
    for key, hits in score["mistakes"].items():
        print(f"  ? {key} — {score['why'].get(key, '')}")
        for hit in hits[:3]:
            print(f"      “{hit[:180]}”")
    if not score["mistakes"]:
        print("  none found")
    relayed = sum(score["facts"].values())
    print(f"\nFirst level, baseline facts relayed: {relayed}/{len(score['facts'])} "
          + ", ".join(f"{key} {'✓' if value else '✗'}" for key, value in score["facts"].items()))
    print(f"Second level, findings beyond the baseline: {score['points']}/{score['max_points']} points "
          + ", ".join(f"{key} {'✓' if value else '✗'}" for key, value in score["findings"].items()))
    return 1 if score["mistakes"] else 0


def _one_case(paths: list[Path], why: str) -> Path:
    if len(paths) > 1:
        names = ", ".join(path.name for path in paths)
        raise NothingToEvaluate(f"{len(paths)} cases on this computer ({names}): {why}, so name it with --case.")
    return paths[0]


def _saved_reports(folder: Path) -> list[dict]:
    """The reports of an earlier run: its folder, or the summary.json in it."""
    path = Path(folder)
    path = path if path.is_file() else path / "summary.json"
    try:
        reports = json.loads(_text(path, "The summary of the earlier run"))
    except ValueError as error:
        raise NothingToEvaluate(f"The summary of the earlier run is not valid JSON: {path} ({error}).") from error
    if not isinstance(reports, list) or not all(isinstance(report, dict) for report in reports):
        raise NothingToEvaluate(f"The summary of the earlier run is not a list of reports: {path}.")
    if not reports:  # no report fails no check
        raise NothingToEvaluate(f"The summary of the earlier run holds no report: {path}.")
    return reports


def _run_into(case: dict, out: Path | None) -> tuple[list[dict], Path]:
    """The baseline's reports and their folder: ``out``, or a temporary folder when none was asked for."""
    if out is not None:
        return run_baseline(case, out), out
    import tempfile

    folder = Path(tempfile.mkdtemp(prefix="gimap_eval_"))
    try:
        return run_baseline(case, folder), folder
    except NothingToEvaluate:
        try:
            folder.rmdir()  # nothing was run: leave no empty folder behind
        except OSError:
            pass
        raise


def _score(path: Path, report: Path) -> int:
    rubric = case_part(load_case(path), "report", path)
    problem = rubric_problem(rubric)
    if problem:
        raise NothingToEvaluate(f'The "report" part of the case {path.name} cannot be used: {problem} (see {README}).')
    text = _text(report, "The report")
    if not text.strip():
        raise NothingToEvaluate(f"The report is empty: {report}.")
    return _print_report(score_report(text, rubric))


def _check(path: Path, arguments: argparse.Namespace, several: bool) -> int:
    """Run (or read) the baseline of one case and print its checks."""
    case = load_case(path)
    expected = case_part(case, "baseline", path)
    if arguments.source:
        reports = _saved_reports(Path(arguments.source))
    else:
        out = Path(arguments.out) if arguments.out else None
        if out is not None and several:
            out = out / path.stem  # several cases: a folder for each
        reports, out = _run_into(case, out)
        print(f"outputs: {out}")
    try:
        checks = check_baseline(reports, expected)
    except (AttributeError, KeyError, TypeError, ValueError) as error:  # a hand-written case, a foreign summary
        raise NothingToEvaluate(
            f"The checks of the case {path.name} cannot be made ({type(error).__name__}: {error}): its "
            f'"baseline" part, or the summary of the run, is not what {README} describes.'
        ) from error
    print(f"case: {case.get('name') or path.stem}\n")
    return _print_baseline(checks)


def _evaluate(arguments: argparse.Namespace) -> int:
    paths = find_cases(arguments.case)
    if arguments.command == "report":
        return _score(_one_case(paths, "a report is scored against one case"), Path(arguments.path))
    if arguments.source:
        paths = [_one_case(paths, "--from holds the run of one case")]
    failed = skipped = False
    for path in paths:
        try:
            failed = bool(_check(path, arguments, len(paths) > 1)) or failed
        except NothingToEvaluate as problem:
            if len(paths) == 1:
                raise
            print(f"case file {path.name}: not evaluated. {problem}\n", file=sys.stderr)
            skipped = True
            continue
        if len(paths) > 1:
            print()
    return 1 if failed else 2 if skipped else 0


def _parser() -> argparse.ArgumentParser:
    case_help = "a case file (JSON), or the name of one in docs/agents/eval/local/ (default: every case there)"
    after = argparse.ArgumentParser(add_help=False)  # --case is accepted before and after the command
    after.add_argument("--case", default=argparse.SUPPRESS, metavar="CASE", help=case_help)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=None, metavar="CASE", help=case_help)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", parents=[after], help="run the baseline on the case's data and check it")
    run = baseline.add_mutually_exclusive_group()  # an earlier run is read, not written
    run.add_argument("--from", dest="source", metavar="DIR", help="read DIR/summary.json from an earlier `auto` run instead")
    run.add_argument("--out", metavar="DIR", help="output folder for the run (default: a temporary folder)")
    report = commands.add_parser("report", parents=[after], help="score an agent's written report")
    report.add_argument("path", help="the report (Markdown or text)")
    return parser


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    arguments = _parser().parse_args(argv)
    try:
        return _evaluate(arguments)
    except NothingToEvaluate as problem:
        print(problem, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
