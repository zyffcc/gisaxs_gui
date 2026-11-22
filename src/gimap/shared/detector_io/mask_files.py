"""Two-dimensional detector masks in stored array order. Non-zero pixels are masked."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np


def read_mask_image(path: Path, frames) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            data = np.load(path, allow_pickle=False)
        except (ValueError, EOFError) as exc:
            raise ValueError("Could not decode the NumPy mask.") from exc
        if not isinstance(data, np.ndarray):  # np.load opened a ZIP archive with a misleading suffix.
            data.close()
            raise ValueError("A mask image must be a two-dimensional numeric array.")
    elif suffix == ".msk":
        from fabio.fit2dmaskimage import Fit2dMaskImage

        try:
            with Fit2dMaskImage() as image:
                data = image.read(str(path)).data
        except OSError:
            raise
        except Exception as exc:  # FabIO raises a bare Exception for an invalid Fit2D signature.
            raise ValueError("Could not decode the Fit2D mask.") from exc
    else:
        data = frames.load(path, 0).data
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.dtype.kind not in "buif":
        raise ValueError("A mask image must be a two-dimensional numeric array.")
    return data != 0


def write_mask_image(path: Path, mask: np.ndarray) -> None:
    """Write zero or one at each detector pixel, preserving the row and column order."""
    data = np.asarray(mask, dtype=bool).astype(np.uint8)
    if data.ndim != 2:
        raise ValueError("A mask image must be a two-dimensional numeric array.")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        with path.open("wb") as stream:
            np.save(stream, data, allow_pickle=False)
    elif suffix in (".tif", ".tiff"):
        from PIL import Image

        Image.fromarray(data).save(path, format="TIFF")
    elif suffix == ".edf":
        from fabio.edfimage import EdfImage

        EdfImage(data=data).write(str(path))
    elif suffix == ".msk":
        from fabio.fit2dmaskimage import Fit2dMaskImage

        Fit2dMaskImage(data=data).write(str(path))
    else:
        raise ValueError("Choose a JSON, EDF, TIFF, NumPy or Fit2D mask file.")


def write_mask_export(path: Path, record: dict, mask: np.ndarray | None = None) -> None:
    """Stage a recipe or a bitmap and its recipe together before replacing the destination."""
    sidecar = path if mask is None else path.with_suffix(path.suffix + ".json")
    with TemporaryDirectory(dir=path.parent) as scratch:
        folder = Path(scratch)
        recipe = folder / "record.json"
        recipe.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if mask is None:
            recipe.replace(path)
            return
        bitmap = folder / ("bitmap" + path.suffix)
        write_mask_image(bitmap, mask)
        backup = folder / "previous.json"
        if sidecar.exists():
            copyfile(sidecar, backup)
        recipe.replace(sidecar)
        try:
            bitmap.replace(path)
        except OSError:
            if backup.exists():
                backup.replace(sidecar)
            else:
                sidecar.unlink()
            raise
