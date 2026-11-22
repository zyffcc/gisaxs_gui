"""Identify the checkout or the source revision stamped into a distribution."""

from __future__ import annotations

import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def source_commit() -> str | None:
    folder = Path(__file__).resolve().parent
    stamp = json.loads((folder / "revision.json").read_text(encoding="utf-8"))["commit"]
    if re.fullmatch(r"[0-9a-f]{40,64}", stamp or ""):
        return stamp
    root = folder.parents[2]
    if (root / ".git").exists():
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return None
