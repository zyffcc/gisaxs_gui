"""The toolkit's progress lines, and the share of a run that is done.

The toolkit writes on stdout (the contract with ``utils/sputter_toolkit``):

* ``SPUTTER_PROGRESS<TAB>stage<TAB>fraction<TAB>message`` at the start and end of each stage;
* ``INTAKE_PLAN <chunks> <frames>`` once before the frames are read;
* ``INTAKE_CHUNK <name> <index> <frames> <seconds>`` after each chunk of frames (in any order: the chunks are
  read in parallel);
* ``INTAKE_EXCLUDED <frame> <reason>`` for each damaged frame that is left out.

Reading the frames takes most of the time, so the chunks move the bar during that stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

STAGES = ("intake", "binning", "features", "network", "done")
WEIGHTS = {"intake": .55, "binning": .1, "features": .2, "network": .15, "done": 0.}
STAGE_TEXT = {
    "intake": "Reading the frames",
    "binning": "Summing the counts into time windows",
    "features": "Direct features of each window",
    "network": "Running the networks",
    "done": "Done",
}
"""What the page says for each stage (the toolkit's own message goes to the log)."""
FRAMES_READ = "{stage}: {done} of {total} frames"
EXCLUDED_FRAMES = "{count} damaged frames left out"


@dataclass(frozen=True)
class Progress:
    stage: str
    fraction: float
    message: str


def parse_progress(line: str) -> Optional[Progress]:
    """A toolkit progress line: ``SPUTTER_PROGRESS<TAB>stage<TAB>fraction<TAB>message``."""
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 4 or parts[0] != "SPUTTER_PROGRESS":
        return None
    try:
        fraction = float(parts[2])
    except ValueError:
        return None
    return Progress(parts[1], fraction, "\t".join(parts[3:]))


def overall_fraction(p: Progress) -> float:
    """Stage-weighted share of the whole run (reading frames dominates)."""
    done = 0.
    for stage in STAGES:
        if stage == p.stage:
            return min(1., done + WEIGHTS[stage] * max(0., min(1., p.fraction))) if stage != "done" else 1.
        done += WEIGHTS[stage]
    return done


@dataclass(frozen=True)
class Excluded:
    frame: str
    reason: str


@dataclass
class ProgressTracker:
    """Follows one run of the toolkit: the stage, the share done (0–1) and the frames left out."""

    stage: str = ""
    fraction: float = 0.
    chunks: int = 0
    frames: int = 0
    frames_done: int = 0
    chunks_done: set = field(default_factory=set)
    excluded: list = field(default_factory=list)

    def feed(self, line: str) -> bool:
        """Take one line of output; True when it was a progress line (the rest is the toolkit's log)."""
        progress = parse_progress(line)
        if progress is not None:
            if progress.stage in WEIGHTS:
                self.stage = progress.stage
                self.fraction = max(self.fraction, overall_fraction(progress))
            return True
        words = line.split()
        if not words:
            return False
        try:
            if words[0] == "INTAKE_PLAN" and len(words) >= 3:
                self.chunks, self.frames = int(words[1]), int(words[2])
                self.stage = "intake"
                return True
            if words[0] == "INTAKE_CHUNK" and len(words) >= 4:
                index, count = int(words[2]), int(words[3])
                if index not in self.chunks_done:
                    self.chunks_done.add(index)
                    self.frames_done += count
                self.stage = "intake"
                self.fraction = max(self.fraction, WEIGHTS["intake"] * self.intake_share())
                return True
        except ValueError:
            return False
        if words[0] == "INTAKE_EXCLUDED" and len(words) >= 2:
            self.excluded.append(Excluded(words[1], " ".join(words[2:])))
            return True
        return False

    def intake_share(self) -> float:
        if self.frames > 0:
            return min(1., self.frames_done / self.frames)
        if self.chunks > 0:
            return min(1., len(self.chunks_done) / self.chunks)
        return 0.

    def message(self) -> tuple[str, dict]:
        """The template and values of what is happening now (the page translates the template)."""
        stage = STAGE_TEXT.get(self.stage, "")
        if self.stage == "intake" and self.frames:
            return FRAMES_READ, {"stage": stage, "done": min(self.frames_done, self.frames), "total": self.frames}
        return "{stage}", {"stage": stage}


__all__ = ["EXCLUDED_FRAMES", "Excluded", "FRAMES_READ", "Progress", "ProgressTracker", "STAGES", "STAGE_TEXT",
           "WEIGHTS", "overall_fraction", "parse_progress"]
