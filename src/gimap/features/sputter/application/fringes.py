"""Island height H from the q_z fringes: measure it for a result in the background, or read what was measured.

The analysis is a separate tool (``utils/sputter_fringes``) that GIMaP runs as a subprocess of its own Python
(``FringeRunner``); it reads the result folder's ``binned/`` and writes ``fringe_heights.csv`` and
``fringe_summary.json``. Where: the result folder's ``fringes_v1/`` when it can be written, else
``<user data>/sputter_fringes/<result folder's name>/``. Outputs are used only when they are fresh
(``FringeStore.fresh``: the same tool, the same ``binned/run.json``, the same metal and the same windows off the
sample); otherwise the page measures again. A failure never stops the page: it is said in the log, and the page goes
on without H.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol

from ..domain import FRINGE_FOLDER, FringeHeights, StructureResult

FRINGES_FAILED = "The island height from the q_z fringes could not be measured (exit code {code}): {reason}"
FRINGES_MISSING = "The fringe analysis is not in this GIMaP folder: {folder} is missing."
FALLBACK_FOLDER = "sputter_fringes"


class FringesFailed(RuntimeError):
    pass


class FringeRunner(Protocol):
    def problem(self) -> str:
        """"" when the tool is there, else the folder that is missing."""
        ...

    def tool_digest(self) -> str:
        """The tool's version (new outputs are needed when it changes)."""
        ...

    def run(self, result_folder: Path, out_folder: Path, metal: str, on_line: Callable[[str], None],
            should_stop: Callable[[], bool]) -> None:
        """Measure; raises ``FringesFailed`` (or ``OSError``) when the tool fails."""
        ...


class FringeStore(Protocol):
    def has_input(self, result_folder: Path) -> bool:
        """The result folder holds the binned cuts the tool reads (``binned/run.json``)."""
        ...

    def writable(self, folder: Path) -> bool:
        """``folder`` (created when missing) can receive files."""
        ...

    def read(self, out_folder: Path) -> Optional[FringeHeights]:
        """The outputs in ``out_folder`` (None when they are missing or cannot be read)."""
        ...

    def fresh(self, heights: FringeHeights, result: StructureResult, tool_digest: str) -> bool:
        """The outputs were made by this tool from the result's current binned cuts, metal and off-sample windows."""
        ...


class FringeService:
    def __init__(self, runner: FringeRunner, store: FringeStore, fallback_root: Callable[[], Path]):
        self.runner, self.store, self.fallback_root = runner, store, fallback_root

    def candidates(self, result_folder: Path) -> list:
        """Where the outputs of a result may be: its own ``fringes_v1/``, then the user data folder's copy."""
        result_folder = Path(result_folder)
        return [result_folder / FRINGE_FOLDER, Path(self.fallback_root()) / FALLBACK_FOLDER / result_folder.name]

    def location(self, result_folder: Path) -> Path:
        own, fallback = self.candidates(result_folder)
        return own if self.store.writable(own) else fallback

    def cached(self, result: StructureResult) -> Optional[FringeHeights]:
        """Fresh outputs of ``result`` (None when there are none)."""
        digest = self.runner.tool_digest()
        for folder in self.candidates(result.folder):
            heights = self.store.read(folder)
            if heights is not None and self.store.fresh(heights, result, digest):
                return heights
        return None

    def can_measure(self, result: StructureResult) -> bool:
        """The result holds binned cuts and the tool is there."""
        return self.store.has_input(result.folder) and not self.runner.problem()

    def measure(self, result: StructureResult, on_line: Callable[[str], None] = lambda _line: None,
                should_stop: Callable[[], bool] = lambda: False) -> Optional[FringeHeights]:
        """Run the tool for ``result`` and read its outputs (None when stopped)."""
        missing = self.runner.problem()
        if missing:
            raise FringesFailed(FRINGES_MISSING.format(folder=missing))
        out = self.location(result.folder)
        self.runner.run(Path(result.folder), out, result.metal, on_line, should_stop)
        if should_stop():
            return None
        heights = self.store.read(out)
        if heights is None:
            raise FringesFailed(FRINGES_FAILED.format(code=0, reason=f"no outputs in {out}"))
        return heights


__all__ = ["FALLBACK_FOLDER", "FRINGES_FAILED", "FRINGES_MISSING", "FringeRunner", "FringeService", "FringeStore",
           "FringesFailed"]
