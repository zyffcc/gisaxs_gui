"""Compatibility launcher for the feature-owned WAXS workspace.

This path remains executable for users and scripts that historically launched
``python WAXS/WAXS.py``.  The implementation now lives under
``src.gimap.features.waxs`` so the main GUI and standalone entry cannot drift.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gimap.features.waxs.infrastructure import (  # noqa: E402,F401
    detect_nxs_frame_count,
    load_image_matrix,
    load_tiff_matrix,
)
from src.gimap.features.waxs.standalone import (  # noqa: E402
    WaxsStandaloneWindow,
    launch_waxs,
)


class MainWindow(WaxsStandaloneWindow):
    """Backward-compatible window class name."""


if __name__ == "__main__":
    raise SystemExit(launch_waxs(MainWindow))
