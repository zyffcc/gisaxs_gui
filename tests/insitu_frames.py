"""Small in-situ series made of the public GALAXI GISAXS frame (``tests/data/external``): a crop around the beam and
the Yoneda band, with synthetic changes — an overall scale and a side peak in qy that moves — and counting noise,
written as 32-bit integer TIFF frames (the Pilatus gap code −1 kept), one file per frame."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

ROOT = Path(__file__).resolve().parents[1]
GALAXI = ROOT / "tests" / "data" / "external" / "gisaxs_galaxi" / "galaxi_data.tif"
CROP = (500, 447, 260, 300)
"""First row, first column, rows, columns of the crop (the direct beam of the frame is at (597.1, 719.6) px)."""
SMALL = (560, 497, 100, 200)
"""A smaller crop for many frames (the Yoneda band and the beam column)."""


@lru_cache(maxsize=4)
def galaxi_crop(crop=CROP) -> np.ndarray:
    from src.gimap.shared.detector_io import load_detector_image

    row, column, rows, columns = crop
    data = np.asarray(load_detector_image(str(GALAXI)).data, dtype=np.float64)
    return data[row:row + rows, column:column + columns].copy()


def crop_profile(crop=CROP) -> InstrumentProfile:
    row, column, rows, columns = crop
    geometry = DetectorGeometry(172e-6, 172e-6, 1.73, 597.1 - column, 719.6 - row, 1.34, 0.463)
    return InstrumentProfile("GALAXI crop", geometry, None, (rows, columns))


def frame_data(index: int, *, crop=CROP, jump_at=None, scale_step=0.01, seed=None) -> np.ndarray:
    """Frame ``index``: the crop × (1 + scale_step · index), plus a side peak on the Yoneda band whose distance from
    the beam column grows by half a pixel per frame (and by 25 px at ``jump_at``), with Poisson noise."""
    base = galaxi_crop(crop)
    rows, columns = np.indices(base.shape)
    row, column, _rows, _columns = crop
    beam = 597.1 - column
    yoneda = 605.0 - row
    offset = 30.0 + 0.5 * index + (25.0 if jump_at is not None and index >= jump_at else 0.0)
    peak = sum(400.0 * np.exp(-0.5 * ((columns - (beam + sign * offset)) / 3.0) ** 2) for sign in (-1, 1))
    peak = peak * np.exp(-0.5 * ((rows - yoneda) / 6.0) ** 2)
    level = np.clip(base, 0.0, None) * (1.0 + scale_step * index) + peak
    counts = np.random.default_rng(index if seed is None else seed).poisson(level).astype(np.int32)
    counts[base < 0] = -1  # the module gaps of the Pilatus
    return counts


def write_frame(folder: Path, index: int, **options) -> Path:
    """``folder/galaxi_<index>.tif`` (written under another name and renamed: a reader never sees half a file)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"galaxi_{index:05d}.tif"
    part = folder / f".{path.name}.part"
    Image.fromarray(frame_data(index, **options), mode="I").save(part, format="TIFF")
    part.replace(path)
    return path
