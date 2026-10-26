"""Run the fringe analysis bundled with GIMaP (``utils/sputter_fringes/fringes.py``) and read its outputs.

The tool needs numpy and scipy only and reads the Sputter toolkit's grid constants (``--toolkit``: the toolkit GIMaP
uses, ``GIMAP_SPUTTER_TOOLKIT`` included); it runs in GIMaP's own Python like the toolkit (``process.run_streaming``,
Stop ends it). Its version is the SHA-256 over its Python sources, computed as the tool records it in
``fringe_summary.json`` (``tool_sha256``): outputs of another version are stale.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Optional

from ..application import FRINGES_FAILED, FringesFailed, StructureResult
from ..domain import HEIGHTS_FILE, SUMMARY_FILE, FringeHeights, parse_heights
from .process import child_environment, console_python, run_streaming
from .toolkit import toolkit_home

LOCAL_FRINGES = Path(__file__).resolve().parents[5] / "utils" / "sputter_fringes"
FRINGES_OVERRIDE = "GIMAP_SPUTTER_FRINGES"
ENTRY = "fringes.py"
GROUP_SECONDS = 1.0
METALS = ("Ag", "Au", "Cu")


def fringes_home() -> Path:
    override = os.environ.get(FRINGES_OVERRIDE, "").strip()
    return Path(override) if override else LOCAL_FRINGES


def tool_digest(folder: Path) -> str:
    """SHA-256 over the tool's Python sources (names and bytes, sorted), as ``cli.tool_sha256`` computes it."""
    digest = hashlib.sha256()
    for path in sorted(Path(folder).glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class FringeTool:
    def __init__(self, home: Optional[Path] = None, python: Optional[str] = None, *, run=run_streaming):
        self._home, self.python, self._run = home, python or console_python(), run

    @property
    def home(self) -> Path:
        return Path(self._home) if self._home is not None else fringes_home()

    def problem(self) -> str:
        return "" if (self.home / ENTRY).is_file() else str(self.home)

    def tool_digest(self) -> str:
        try:
            return tool_digest(self.home)
        except OSError:
            return ""

    def run(self, result_folder: Path, out_folder: Path, metal: str, on_line: Callable[[str], None],
            should_stop: Callable[[], bool]) -> None:
        args = [self.python, "-u", str(self.home / ENTRY), "--folder", str(result_folder), "--out", str(out_folder),
                "--group-seconds", repr(GROUP_SECONDS), "--toolkit", str(toolkit_home())]
        if metal in METALS:
            args += ["--metal", metal]
        outcome = self._run(args, on_line=on_line, should_stop=should_stop, env=child_environment(),
                            cwd=str(self.home))
        if outcome.stopped:
            return
        if outcome.code != 0:
            said = [line.split("\t", 1)[-1] for line in outcome.tail if line.startswith("FRINGES_ERROR")]
            reason = said[-1] if said else "\n".join(outcome.tail[-5:])
            raise FringesFailed(FRINGES_FAILED.format(code=outcome.code, reason=reason))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FringeFolderStore:
    def has_input(self, result_folder: Path) -> bool:
        return (Path(result_folder) / "binned" / "run.json").is_file()

    def writable(self, folder: Path) -> bool:
        probe = Path(folder) / f".gimap_probe_{uuid.uuid4().hex[:8]}"
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            probe.unlink()
            return True
        except OSError:
            return False

    def read(self, out_folder: Path) -> Optional[FringeHeights]:
        out_folder = Path(out_folder)
        try:
            summary = json.loads((out_folder / SUMMARY_FILE).read_text(encoding="utf-8"))
            with (out_folder / HEIGHTS_FILE).open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except (OSError, ValueError):
            return None
        return parse_heights(out_folder, rows, summary) if isinstance(summary, dict) else None

    def fresh(self, heights: FringeHeights, result: StructureResult, tool_digest_now: str) -> bool:
        summary = heights.summary
        sources = summary.get("sources") if isinstance(summary.get("sources"), dict) else {}
        if not tool_digest_now or summary.get("tool_sha256") != tool_digest_now:
            return False
        try:
            if _sha256(Path(result.folder) / "binned" / "run.json") != sources.get("run.json"):
                return False
        except OSError:
            return False
        if result.metal in METALS and summary.get("metal") != result.metal:
            return False
        if result.has("beam_off_sample") and "beam_off_frames" in sources:
            flagged = {str(r.get("frame")) for r in result.rows if r.get("beam_off_sample") is True}
            if flagged != set(sources.get("beam_off_frames") or []):
                return False
        return True


__all__ = ["ENTRY", "FRINGES_OVERRIDE", "FringeFolderStore", "FringeTool", "GROUP_SECONDS", "LOCAL_FRINGES",
           "fringes_home", "tool_digest"]
