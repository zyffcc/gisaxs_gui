"""How the import spent its time, in words: the live view (``live_panel.py``) and Details ▸ Report
(``result_tabs.py``) say the same.

A newer analysis package tells, for every chunk of frames it reads, how long its process spent opening and reading the
files from disk against its whole time, and at the end of the import the totals (``application/client_live.py``).
When the disk takes at least 0.15 s per frame and at least half of the time (or the package says so), reading from the
disk is the slow part: a USB or network drive, or an on-access virus scan of every file. The page then says so, with
what helps. An older package tells none of this: nothing is said.
"""

from __future__ import annotations

from typing import Optional

from .client_texts import shown_number
from .texts import text

DISK_PART = "disk {seconds} s per frame ({share} % of the time)"
DISK_PART_ALONE = "disk {seconds} s per frame"
READING_LINE = "Reading {done} of {total} frames"
PROCESSES = "Processes: {count}"
SLOW_DISK_NOTE = ("Reading the frames from this disk is the slow part (a USB or network drive, or an on-access virus "
                  "scan of every file). Copying the scan folder to a local SSD first makes the import faster.")
TEMPLATES = (DISK_PART, DISK_PART_ALONE, READING_LINE, PROCESSES, SLOW_DISK_NOTE)
"""The texts about reading the frames that need Chinese."""


def disk_words(per_frame: Optional[float], share: Optional[float]) -> str:
    """"disk 0.82 s per frame (85 % of the time)" ("" without the seconds per frame)."""
    if per_frame is None:
        return ""
    seconds = shown_number(per_frame, 2)
    if share is None:
        return text(DISK_PART_ALONE, seconds=seconds)
    return text(DISK_PART, seconds=seconds, share=f"{100 * share:.0f}")


def clock_time(seconds: float) -> str:
    """Seconds as m:ss (h:mm:ss from an hour on)."""
    whole = max(0, int(round(float(seconds))))
    hours, rest = divmod(whole, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


__all__ = ["DISK_PART", "DISK_PART_ALONE", "PROCESSES", "READING_LINE", "SLOW_DISK_NOTE", "TEMPLATES", "clock_time",
           "disk_words"]
