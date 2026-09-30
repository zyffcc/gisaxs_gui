"""Evaluate GIWAXS agents on a real case: the baseline must not err, a capable agent should find more.

    python tools/eval_giwaxs_agent.py baseline [--case CASE.json] [--from DIR | --out DIR]
        runs `gimap_agent.py auto` on the case's frames (or reads DIR/summary.json) and checks
        the facts the case fixes: calibration, line error, reliable peaks, flagged spike, ring f,
        no false series, no open questions.
    python tools/eval_giwaxs_agent.py report REPORT.md [--case CASE.json]
        scores an agent's written report: mistakes (must be absent; each hit is printed for a
        person to confirm, negated sentences are skipped), baseline facts relayed, and findings
        beyond the baseline (points).

The default case is docs/agents/eval/yuxin-p03-2021.json.  Regular expressions are a first
pass, not a judge: read the report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = PROJECT_ROOT / "docs" / "agents" / "eval" / "yuxin-p03-2021.json"
_SENTENCE = re.compile(r"(?<=[。！？!?\n])|(?<=\.)\s+(?=[A-Z0-9*(|])")  # a semicolon stays inside its sentence


def load_case(path: Path = DEFAULT_CASE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.gimap.app.headless_assistant import analyse_frames
    from src.gimap.features.assistant.application import PipelineOptions

    root = Path(case["data_root"])
    frames = [str(root / frame) for frame in case["frames"]]
    missing = [frame for frame in frames if not Path(frame).is_file()]
    if missing:
        raise FileNotFoundError(f"The case's data are not on this machine: {missing[0]}")
    return analyse_frames(frames, PipelineOptions(notes=case.get("notes", "")), out, saved_profiles=False,
                          progress=lambda text: print(text, file=sys.stderr, flush=True))


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


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="case file (JSON)")
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="run the baseline on the case's data and check it")
    baseline.add_argument("--from", dest="source", help="read DIR/summary.json from an earlier `auto` run instead")
    baseline.add_argument("--out", help="output folder for the run (default: a temporary folder)")
    report = commands.add_parser("report", help="score an agent's written report")
    report.add_argument("path", help="the report (Markdown or text)")
    arguments = parser.parse_args(argv)
    case = load_case(Path(arguments.case))
    if arguments.command == "report":
        text = Path(arguments.path).read_text(encoding="utf-8", errors="replace")
        return _print_report(score_report(text, case["report"]))
    if arguments.source:
        reports = json.loads((Path(arguments.source) / "summary.json").read_text(encoding="utf-8"))
    else:
        import tempfile

        out = Path(arguments.out) if arguments.out else Path(tempfile.mkdtemp(prefix="gimap_eval_"))
        reports = run_baseline(case, out)
        print(f"outputs: {out}")
    print(f"case: {case['name']}\n")
    return _print_baseline(check_baseline(reports, case["baseline"]))


if __name__ == "__main__":
    sys.exit(main())
