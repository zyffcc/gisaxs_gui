"""WAXS detector image repository port。"""

from pathlib import Path
from typing import Protocol

import numpy as np


class WaxsImageRepository(Protocol):
    def frame_count(self, path: Path) -> int: ...

    def load_frame(self, path: Path, frame_index: int) -> np.ndarray: ...
