"""Run the repository's development checks with the active Python interpreter."""

from __future__ import annotations

import os
import subprocess
import sys


def run(command: list[str], *, env: dict[str, str]) -> None:
    """Run one check and stop immediately if it fails."""
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    """Run tests, the legacy-safe lint baseline, and strict lint for this script."""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    checks = [
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "."],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "tools/check.py",
            "--select",
            "E4,E7,E9,F,I",
        ],
    ]
    for command in checks:
        run(command, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
