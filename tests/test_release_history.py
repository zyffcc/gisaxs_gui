"""The commits removed from main when its history was rewritten must not come back.

On 2026-10-13 the commits of 2026-09-30 … 2026-10-13 were replaced by one clean commit (they held traces of
unpublished data). A clone made before that day still has them; merging such a clone with GitHub's main, or
pushing from it, brings them back. This test fails as soon as one of them is part of the branch checked out,
so the full check stops before such a branch is pushed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REMOVED = {
    "600d2a5": "feat: Analyze workspace, automatic analysis and a new Fitting page",
    "6a0e3c7": "feat: In-situ series page, point exclusion, projects and a button audit",
    "d825b6d": "feat: Compare page and stages of a series; remove Classification",
    "c3253ef": "feat: polish every page overnight; rewrite the agent docs for current models",
    "ea7db6f": "fix: finish 0.1.0 — release audit fixes, changelog, clean packaging",
}


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(not shutil.which("git") or not (ROOT / ".git").exists(), reason="not a git checkout")
def test_no_commit_removed_by_the_history_rewrite_is_on_this_branch() -> None:
    back = []
    for short, subject in REMOVED.items():
        if _git("cat-file", "-e", f"{short}^{{commit}}").returncode != 0:
            continue  # not even in this clone
        if _git("merge-base", "--is-ancestor", short, "HEAD").returncode == 0:
            back.append(f"{short} {subject}")
    assert not back, (
        "This branch contains commits that were removed from main on 2026-10-13 (they hold traces of unpublished "
        "data): " + "; ".join(back) + ". Do not push it. Move the clone to GitHub's main with "
        "tools/update_source.ps1 (README.md, 'Updating a source checkout') and bring your own changes over with "
        "git cherry-pick or as files, without merging the old history."
    )
