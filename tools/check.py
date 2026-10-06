"""Run the repository's development checks with the active Python interpreter.

Steps, in this order; each runs even when an earlier one failed, and a summary follows:

1. Main test suite  ``python -m pytest tests`` (gate)
2. Offscreen smoke  ``python tools/offscreen_smoke.py`` (gate): the real window starts and closes
3. Ruff             ``python -m ruff check .`` (gate): syntax errors, invalid control flow,
                    undefined names
4. Research tests   ``python -m pytest utils/ML_Fitting_1D_GISAXS/tests
                    --continue-on-collection-errors``: reported only, never fails the check

The research tests belong to the fitting research code that the GUI runs only as a subprocess.
On Windows they are skipped by default: one of them imports the POSIX-only ``resource`` module,
which stops pytest at collection (``--research run`` tries them anyway).

Qt runs offscreen (``QT_QPA_PLATFORM``), and the steps use a temporary user data folder
(``GIMAP_HOME``) unless one is set, so a check never writes into the real one. Arguments after
``--`` go to the main suite's pytest, e.g. ``python tools/check.py -- -x -k analyze``; when they
name test files or folders, only those run instead of ``tests`` (an option's value, as in
``-k tests`` or ``--deselect tests/test_x.py::t``, does not count).

Exit code: 0 when every gate step that ran passed, 1 when one failed or the run was interrupted
(2 for a wrong command line). The research tests never change it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_TESTS = "utils/ML_Fitting_1D_GISAXS/tests"
STEP_NAMES = ("tests", "smoke", "ruff", "research")
WINDOWS_RESEARCH_REASON = (
    "skipped on Windows: a research test imports the POSIX-only 'resource' module and stops "
    "collection (--research run to try anyway)"
)
_PYTEST_COUNTS = re.compile(
    r"\b\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b.* in [\d.]+s"
)
_RESULT_LINES = ("Offscreen startup OK", "All checks passed", "Found ")
"""The smoke's and ruff's own one-line verdicts."""


@dataclass
class Step:
    key: str
    title: str
    command: list[str]
    gate: bool = True
    skip_reason: str = ""
    status: str = "not run"
    seconds: float = 0.0
    note: str = ""


def _note(lines) -> str:
    """The step's own one-line result: pytest's counts, the smoke's OK line or ruff's verdict, with
    pytest's usage error (a path that is not there …) when it gave one."""
    usage = next((line.strip() for line in lines if line.startswith("ERROR: ")), "")
    found = ""
    for line in reversed(lines):
        text = line.strip().strip("=").strip()
        if _PYTEST_COUNTS.search(text) or text.startswith(_RESULT_LINES):
            found = text
            break
        if text.startswith(("no tests ran", "Interrupted:", "!!!")):
            found = text.strip("! ")
            break
    found = found or next((line.strip() for line in reversed(lines) if line.strip()), "")
    if usage and usage != found:
        return f"{found}; {usage}" if found else usage
    return found


def _run(step: Step, env: dict[str, str]) -> None:
    """Run one step, showing its output as it comes and keeping the end of it for the summary."""
    print(f"\n=== {step.title}{'' if step.gate else ' (reported, not a gate)'}", flush=True)
    print(f"> {' '.join(step.command)}", flush=True)
    tail: deque[str] = deque(maxlen=40)
    started = time.monotonic()
    process = subprocess.Popen(
        step.command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line)
        process.wait()
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        step.seconds = time.monotonic() - started
        step.status, step.note = "interrupted", "stopped with Ctrl+C"
        raise
    step.seconds = time.monotonic() - started
    step.status = "passed" if process.returncode == 0 else f"failed (exit {process.returncode})"
    step.note = _note(list(tail))


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _summary(steps: list[Step]) -> bool:
    """Print one line per step; True when no gate step failed."""
    print("\n=== Summary")
    statuses = [
        step.status + (", not a gate" if not step.gate and step.status.startswith("failed") else "")
        for step in steps
    ]
    title_width = max(len(step.title) for step in steps)
    status_width = max(len(status) for status in statuses)
    for step, status in zip(steps, statuses, strict=True):
        timing = _duration(step.seconds) if step.seconds else ""
        note = step.skip_reason or step.note
        print(f"  {step.title:<{title_width}}  {status:<{status_width}}  {timing:>7}  {note}")
    failed = [
        step.title for step in steps
        if step.gate and step.status not in ("passed", "skipped", "not run")
    ]
    if any(step.status == "interrupted" for step in steps):
        print("Result: INTERRUPTED")
    else:
        print("Result: " + (f"FAILED ({', '.join(failed)})" if failed else "OK"))
    return not failed


_VALUE_OPTIONS = frozenset({
    "-k", "-m", "-p", "-c", "-o", "-W", "-r", "-n", "--deselect", "--ignore", "--ignore-glob",
    "--rootdir", "--basetemp", "--confcutdir", "--maxfail", "--durations", "--durations-min",
    "--tb", "--junitxml", "--junit-xml", "--log-level", "--log-file", "--import-mode",
    "--capture", "--override-ini", "--config-file", "--pythonwarnings",
})
"""pytest options whose value is the next argument; that value is never a test path to run."""


def _names_tests(arguments: list[str]) -> bool:
    """Whether the pytest arguments name test files or folders (``tests/test_x.py::test_y``) to
    run instead of ``tests``. Option values (``-k tests``, ``--deselect tests/test_x.py::t``) do
    not count, so `tests` still runs."""
    previous = ""
    for argument in arguments:
        if (
            not argument.startswith("-") and previous not in _VALUE_OPTIONS
            and (PROJECT_ROOT / argument.split("::")[0]).exists()
        ):
            return True
        previous = argument
    return False


def _steps(arguments) -> list[Step]:
    python = sys.executable
    extra = list(arguments.pytest_args)
    suite = [] if _names_tests(extra) else ["tests"]
    steps = [
        Step("tests", "Main test suite", [python, "-m", "pytest", *suite, *extra]),
        Step("smoke", "Offscreen smoke", [python, "tools/offscreen_smoke.py"]),
        Step("ruff", "Ruff", [python, "-m", "ruff", "check", "."]),
        Step(
            "research", "Research tests",
            [python, "-m", "pytest", RESEARCH_TESTS, "--continue-on-collection-errors"], gate=False,
        ),
    ]
    for step in steps:
        if step.key in arguments.skip:
            step.skip_reason = "skipped (--skip)"
        elif step.key == "research" and arguments.research == "skip":
            step.skip_reason = "skipped (--research skip)"
        elif step.key == "research" and arguments.research == "auto" and os.name == "nt":
            step.skip_reason = WINDOWS_RESEARCH_REASON
        if step.skip_reason:
            step.status = "skipped"  # also when the run is interrupted before reaching it
    return steps


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--skip", action="append", default=[], choices=STEP_NAMES, metavar="STEP",
        help=f"leave out a step ({', '.join(STEP_NAMES)}); repeat for more",
    )
    parser.add_argument(
        "--research", choices=("auto", "run", "skip"), default="auto",
        help="the research tests: auto (default) skips them on Windows and runs them elsewhere",
    )
    parser.add_argument(
        "pytest_args", nargs="*",
        help="after --: arguments for the main suite's pytest (test paths replace tests/)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Redirected to a file or a pipe (a log, an agent): UTF-8, so “Å⁻¹” survives the Windows code
        # page; a console code page without it still shows the rest of the line.
        redirected = not sys.stdout.isatty() and not os.environ.get("PYTHONIOENCODING")
        sys.stdout.reconfigure(**({"encoding": "utf-8"} if redirected else {}), errors="replace")
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONIOENCODING"] = "utf-8"
    steps = _steps(arguments)
    home = tempfile.TemporaryDirectory(prefix="gimap-check-home-", ignore_cleanup_errors=True)
    with home as home_path:
        env.setdefault("GIMAP_HOME", home_path)
        try:
            for step in steps:
                if not step.skip_reason:
                    _run(step, env)
        except KeyboardInterrupt:
            _summary(steps)
            return 1
    return 0 if _summary(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
