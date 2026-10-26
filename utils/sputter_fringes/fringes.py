"""Island or layer height H(t) from the moving q_z fringes of a GIMaP Sputter result.

Usage::

    python utils/sputter_fringes/fringes.py --folder RESULT --out OUTDIR [--metal Ag|Au|Cu] [--alpha-c DEG]
        [--group-seconds 1.0] [--qy-band 0.05 0.30] [--toolkit DIR] [--seed 0]

RESULT is a Sputter result folder (it holds ``binned/run.json``); it is only read. OUTDIR receives
``fringe_heights.csv`` (one row per time group) and ``fringe_summary.json``; it may not be RESULT itself or one of the
toolkit's folders in it. Exit code 0 when the outputs were written (whatever the status), 2 when the folder cannot be
read as it is, 1 on an unexpected error.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sputter_fringes.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    sys.exit(main())
