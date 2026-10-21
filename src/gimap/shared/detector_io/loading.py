"""CBF / NeXus / HDF5 / TIFF / EDF detector image loading adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .metadata import extract_cbf_metadata, extract_edf_metadata, extract_nxs_metadata
from .models import DetectorImage
from .nexus_datasets import (  # noqa: F401  (re-exported: the dataset choosers import them from here)
    DEFAULT_DATASET_PATH,
    AmbiguousDatasetError,
    _dataset_candidates,
    select_nxs_dataset,
)
from .nexus_files import NO_IMAGE
from .nexus_layout import HDF5_SUFFIXES, cached_layout, is_hdf5, open_nexus
from .nexus_reading import NXS_REJECT_MASK_BITS, nxs_invalid_pixel_mask, read_nexus_frame  # noqa: F401


DETECTOR_SUFFIXES = (".cbf", ".tif", ".tiff", ".edf") + HDF5_SUFFIXES


def detect_nxs_frame_count(file_path: str | Path, dataset_path: Optional[str] = None) -> int:
    """Frames of the NeXus / HDF5 source ``file_path`` belongs to (1 for every other format).

    For a module set that is the number of frames every module holds.
    """
    path = Path(file_path)
    if not is_hdf5(path):
        return 1
    layout = cached_layout(path, dataset_path)
    if layout.kind == "no_image":
        raise ValueError(NO_IMAGE.format(name=path.name))
    return int(layout.frames)


def nxs_series_paths(path: str | Path) -> list[Path]:
    """Return all P03 module files belonging to the selected NXS source.

    The files are a set only when their content says so (``nexus_layout``); a file that cannot be
    read now stands alone.
    """
    path = Path(path)
    if not is_hdf5(path):
        return [path]
    try:
        layout = cached_layout(path)
    except Exception:  # noqa: BLE001 - listing never fails on an unreadable file; reading it says why
        return [path]
    if layout.kind != "module_set":
        return [path]
    return [path.parent / member.name for member in layout.members]  # in the folder as the caller named it


def _read_cbf(path: Path) -> DetectorImage:
    try:
        import fabio
    except ImportError as exc:
        raise ImportError("fabio is required to read CBF calibration images.") from exc
    cbf = fabio.open(str(path))
    if cbf.data is None:
        raise ValueError(f"Empty CBF detector image: {path}")
    data = np.asarray(cbf.data, dtype=np.float32)
    invalid = ~np.isfinite(data) | (data < 0)
    metadata = extract_cbf_metadata(cbf.header, data.shape)
    # Beam_xy is (column, row) in pixels of the stored image, which is also
    # the canonical frame here (row 0 at the top, origin at the pixel corner,
    # as pyFAI reads it).  Headers are often stale; Analyze uses this centre
    # only when the user opts in.
    if metadata.get("header_beam_xy_px") is not None:
        metadata["header_beam_center"] = tuple(float(v) for v in metadata["header_beam_xy_px"])
    # Beamline CBF headers often contain detector settings but not incident
    # energy.  A simultaneous detector normally writes an NXS file under a
    # sibling directory of the same scan; read only its lightweight metadata.
    if metadata.get("energy_kev") is None:
        scan_root = path.parent.parent
        companion_paths = sorted(
            candidate for candidate in scan_root.glob("*/*.nxs")
            if candidate.is_file()
        )
        for companion in companion_paths:
            try:
                with open_nexus(companion) as handle:
                    companion_metadata = extract_nxs_metadata(handle)
                if companion_metadata.get("energy_kev"):
                    metadata["energy_kev"] = companion_metadata["energy_kev"]
                    metadata["wavelength_angstrom"] = companion_metadata["wavelength_angstrom"]
                    metadata["energy_source"] = f"companion NXS: {companion}"
                    break
            except Exception:
                continue
    fields = {key: metadata[key] for key in (
        "detector_name", "pixel_size_x_m", "pixel_size_y_m", "energy_kev", "wavelength_angstrom", "distance_m", "beam_center_x_px", "beam_center_y_px"
    )}
    return DetectorImage(data=data, mask=invalid, source_path=path, metadata=metadata, **fields)


def _read_edf(path: Path) -> DetectorImage:
    import fabio

    image = fabio.open(str(path))
    try:
        if image.data is None:
            raise ValueError(f"Empty EDF detector image: {path}")
        raw = np.array(image.data, copy=True)
        header = dict(image.header)
    finally:
        image.close()
    if raw.ndim == 3:
        raw = raw[0]
    data = np.asarray(raw, dtype=np.float32)
    # Like TIFF, values stay as stored (negative integer sentinels are Analyze's call);
    # only the header's explicit Dummy marks pixels without data here.
    invalid = ~np.isfinite(data)
    metadata = extract_edf_metadata(header, data.shape)
    metadata["stored_dtype"] = str(raw.dtype)
    try:
        dummy = float(header.get("Dummy", 0.0))
        tolerance = float(header.get("DDummy", 0.0))
    except ValueError:
        dummy, tolerance = 0.0, 0.0
    if dummy != 0.0:  # the SAXS-package convention: |value − Dummy| ≤ DDummy marks a pixel without data
        invalid |= np.abs(data - dummy) <= max(tolerance, 0.0)
    metadata["reader"] = "fabio"
    fields = {key: metadata[key] for key in (
        "detector_name", "pixel_size_x_m", "pixel_size_y_m", "energy_kev", "wavelength_angstrom", "distance_m",
        "beam_center_x_px", "beam_center_y_px",
    )}
    return DetectorImage(data=data, mask=invalid, source_path=path, metadata=metadata, **fields)


def _read_tiff(path: Path) -> DetectorImage:
    reader = "pillow"
    try:
        from PIL import Image
        with Image.open(path) as image:
            data = np.asarray(image)
    except (ImportError, OSError, ValueError):
        # Some detector TIFFs store signed 32-bit samples with WhiteIsZero.
        # Matplotlib uses Pillow too, so retrying there cannot decode these.
        # FabIO preserves scientific pixel values, including valid negatives;
        # interpreting detector-specific invalid codes belongs to preprocessing.
        import fabio

        detector = fabio.open(str(path))
        try:
            if detector.data is None:
                raise ValueError(f"Empty TIFF detector image: {path}")
            data = np.array(detector.data, copy=True)
        finally:
            detector.close()
        reader = "fabio"
    if data.ndim == 3:
        data = np.mean(data[..., :3], axis=2)
    stored_dtype = str(np.asarray(data).dtype)
    data = np.asarray(data, dtype=np.float32)
    return DetectorImage(
        data,
        ~np.isfinite(data),
        path,
        metadata={"format": "tiff", "reader": reader, "transformations": [], "stored_dtype": stored_dtype},
    )


def load_detector_image(
    path: str | Path,
    *,
    frame_idx: int = 0,
    dataset_path: Optional[str] = None,
) -> DetectorImage:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Calibration image was not found: {source}")
    suffix = source.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        return read_nexus_frame(cached_layout(source, dataset_path), frame_idx, source_path=source)
    if suffix == ".cbf":
        return _read_cbf(source)
    if suffix in {".tif", ".tiff"}:
        return _read_tiff(source)
    if suffix == ".edf":
        return _read_edf(source)
    raise ValueError(f"Unsupported detector image {source.name}: use CBF, NeXus/HDF5, TIFF or EDF.")


def dump_metadata(image: DetectorImage) -> str:
    return json.dumps(image.metadata, indent=2, default=str)
