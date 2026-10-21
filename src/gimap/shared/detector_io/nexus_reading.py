"""Read one frame of a NeXus / HDF5 source that ``nexus_layout`` described.

Orientation.  A file of the Lambda server (a module set, or a detector group with a module position,
a ``layout`` like ``1556X516`` or a name containing "lambda") gets the P03 transform GIMaP has always
used for it: modules placed by their translation, the canvas transposed and flipped vertically.
Every other NeXus / HDF5 image is returned as stored (row 0 on top), like the CBF or TIFF of the same
detector.  ``metadata["transformations"]`` lists what was applied.

Validity.  ``mask`` is True where a pixel holds no data: the NeXus ``pixel_mask`` bits with a defined
bad meaning, non-finite values, the parts of a stitched canvas no module covers and, in unsigned
integer data, the value the detector writes for "no data" (the largest of the type).  Those pixels are
NaN in ``data``; every other value is as stored (``metadata["stored_dtype"]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .models import DetectorImage
from .nexus_files import FRAME_BEYOND, NO_IMAGE
from .nexus_layout import P03_LAMBDA, NexusLayout, open_nexus
from .nexus_modules import lambda_layout, module_position
from .nexus_metadata import extract_nexus_metadata
from .radiation import energy_to_wavelength

# NeXus NXdetector pixel masks are bit fields, not boolean arrays.  Reject the
# bits with an explicitly defined invalid/unwanted meaning: gap (0), dead (1),
# under/over responding (2/3), noisy (4), problematic cluster (6), and the
# experiment user mask (8).  Undefined lower bits such as bit 10 and upper tag
# bits such as bit 31 (virtual/interpolated pixel) must not erase valid data.
NXS_REJECT_MASK_BITS = np.uint32(0x0000015F)
_NEXUS_SUFFIXES = (".nxs", ".nx", ".nx5")


def nxs_invalid_pixel_mask(raw_mask: Any) -> np.ndarray:
    """Convert a NeXus bit-field pixel mask into the application's invalid mask."""
    values = np.asarray(raw_mask, dtype=np.uint32)
    return (values & NXS_REJECT_MASK_BITS) != 0


def _pixel_mask(group: Any, frame: int, shape: tuple[int, int], image: h5py.Dataset) -> Optional[np.ndarray]:
    """The invalid pixels the file marks next to the data (``pixel_mask``, Dectris: ``detectorSpecific/``)."""
    if group is None:
        return None
    for name in ("pixel_mask", "detectorSpecific/pixel_mask"):
        node = group.get(name)
        if not isinstance(node, h5py.Dataset) or node == image:  # a mask read as the image is not masked by itself
            continue
        if node.ndim == 3 and node.shape[0] > 0 and tuple(node.shape[1:]) == shape:
            return nxs_invalid_pixel_mask(node[min(frame, node.shape[0] - 1)])
        if node.ndim == 2 and tuple(node.shape) == shape:
            return nxs_invalid_pixel_mask(node[()])
    return None


def _stored_frame(dataset: h5py.Dataset, index: int) -> np.ndarray:
    """Frame ``index`` of a 2-D image, a stack or a grid of frames (storage order), as stored."""
    if dataset.ndim == 2:
        return dataset[()]
    return dataset[np.unravel_index(index, dataset.shape[:-2])]


def _module(handle: h5py.File, layout: NexusLayout, dataset_path: str, local: int, frame: int,
            name: str) -> dict[str, Any]:
    """One stored frame with its invalid pixels as NaN, its mask and where the file says it belongs."""
    dataset = handle.get(dataset_path)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Detector dataset {dataset_path} is missing in {name}.")
    raw = _stored_frame(dataset, local)
    if raw.ndim != 2:
        raise ValueError(f"{name}: unexpected frame shape {raw.shape}.")
    group = handle.get(layout.detector_group)
    invalid = _pixel_mask(group, frame, raw.shape, dataset)
    sentinel = raw == np.iinfo(raw.dtype).max if raw.dtype.kind == "u" else None
    image = raw.astype(np.float32)
    if sentinel is not None:
        invalid = sentinel if invalid is None else invalid | sentinel
    if invalid is not None:
        image[invalid] = np.nan
    return {"name": name, "image": image, "mask": invalid, "dtype": str(dataset.dtype),
            "sentinels": 0 if sentinel is None else int(np.count_nonzero(sentinel)),
            "position": module_position(group), "stored": lambda_layout(group)}


def _chunk_of(layout: NexusLayout, index: int) -> tuple[str, int]:
    """Dataset and frame in it for frame ``index`` of a chunked series."""
    for path, count in layout.chunks:
        if index < count:
            return path, index
        index -= count
    raise IndexError(f"frame beyond the {layout.frames} of the data chunks")


def _p03_canvas(modules: list[dict[str, Any]], steps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Modules placed by their translation, then the canvas transposed and flipped (the P03 Lambda view)."""
    for item in modules:
        item["tx"], item["ty"] = item["position"] or (0, 0)
        stored = item["stored"]
        # The server names the module as columns × rows: one stored the other way round is turned first.
        if stored is not None and stored[0] != stored[1] and item["image"].shape == stored:
            item["image"] = item["image"].T
            item["mask"] = item["mask"].T if item["mask"] is not None else None
            if "module transpose" not in steps:
                steps.append("module transpose")
    shape = modules[0]["image"].shape
    for item in modules:
        if item["image"].shape != shape:
            raise ValueError(f"{item['name']}: unexpected module shape {item['image'].shape}.")
    module_x, module_y = shape
    min_tx = min(item["tx"] for item in modules)
    min_ty = min(item["ty"] for item in modules)
    shift_x, shift_y = max(0, -min_tx), max(0, -min_ty)
    width_x = max(module_x + item["tx"] + shift_x for item in modules)
    width_y = max(module_y + item["ty"] + shift_y for item in modules)
    grid = np.full((width_x, width_y), np.nan, dtype=np.float32)
    invalid = np.ones((width_x, width_y), dtype=bool)
    for item in modules:
        x0, y0 = item["tx"] + shift_x, item["ty"] + shift_y
        region = np.s_[x0 : x0 + module_x, y0 : y0 + module_y]
        grid[region] = item["image"]
        invalid[region] = ~np.isfinite(item["image"]) | (item["mask"] if item["mask"] is not None else False)
    if len(modules) > 1:
        steps.append("stitched using P03 translations")
    elif (width_x, width_y) != (module_x, module_y):
        steps.append("placed at its P03 translation")
    steps += ["canvas transpose", "vertical flip"]
    # This is the exact transpose + vertical flip used by the embedded GIWAXS page.
    return np.flipud(grid.T).astype(np.float32, copy=False), np.flipud(invalid.T)


def read_nexus_frame(layout: NexusLayout, frame_idx: int = 0, source_path: Optional[str | Path] = None
                     ) -> DetectorImage:
    """Frame ``frame_idx`` (0-based) of the source ``layout`` describes.

    ``IndexError("frame 6 of 5")`` for a frame the source does not have (never the last one instead);
    ``ValueError`` for a file without a detector image.
    """
    if layout.kind == "no_image":
        raise ValueError(NO_IMAGE.format(name=layout.path.name))
    index = int(frame_idx)
    if not 0 <= index < layout.frames:
        raise IndexError(FRAME_BEYOND.format(frame=index + 1, frames=layout.frames))
    dataset_path, local = _chunk_of(layout, index) if layout.chunks else (layout.dataset, index)
    members = layout.members if layout.kind == "module_set" else layout.members[:1]
    modules: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for member in members:
        with open_nexus(member) as handle:
            modules.append(_module(handle, layout, dataset_path, local, index, member.name))
            if len(modules) == 1:  # the header values of a set are those of its first module
                metadata = extract_nexus_metadata(handle, layout.detector_group, layout.entry, frame=index)

    steps: list[str] = []
    if layout.orientation == P03_LAMBDA:
        data, mask = _p03_canvas(modules, steps)
    else:
        data = modules[0]["image"]
        mask = ~np.isfinite(data)
        if modules[0]["mask"] is not None:
            mask |= modules[0]["mask"]
    # The header beam centre is (along the last stored axis, along the first).  As stored that is
    # (column, row); after the P03 transpose + flip the first stored axis becomes the columns and the
    # last one the rows counted from the bottom.  A stitched set has no single header frame, so none.
    raw_center = metadata.get("header_beam_center_px")
    if raw_center is not None and len(modules) == 1:
        fast, slow = raw_center
        metadata["header_beam_center"] = (
            (float(slow), float(data.shape[0]) - float(fast)) if layout.orientation == P03_LAMBDA
            else (float(fast), float(slow)))
    path = Path(source_path) if source_path is not None else layout.path
    metadata.update({
        "format": "nxs" if layout.path.suffix.lower() in _NEXUS_SUFFIXES else "hdf5",
        "dataset_path": dataset_path if layout.chunks else layout.dataset,
        "frame_index": index,
        "frame_count": layout.frames,
        "stored_dtype": modules[0]["dtype"],
        "layout": layout.kind,
        "layout_summary": layout.summary_text(),
        "layout_warnings": list(layout.warning_texts()),
        "orientation": layout.orientation,
        "transformations": steps,
        "sentinel_pixels": sum(module["sentinels"] for module in modules),
        "mask_semantics": (
            "True is invalid; rejects NeXus-defined bits 0, 1, 2, 3, 4, 6, and 8; "
            "preserves undefined/tag bits including 10 and virtual-pixel bit 31; "
            "unsigned integer pixels at the largest value of their type hold no data"
        ),
        "pixel_mask_reject_bits": "0x0000015F",
    })
    if layout.kind == "module_set":
        metadata["module_files"] = [str(member) for member in layout.members]
    elif layout.kind == "chunked_master":
        metadata["data_files"] = [str(member) for member in layout.members[1:]]
    if metadata["wavelength_angstrom"] is None and metadata["energy_kev"]:
        metadata["wavelength_angstrom"] = energy_to_wavelength(metadata["energy_kev"])
    return DetectorImage(data=data, mask=mask, source_path=path, metadata=metadata, **{key: metadata[key] for key in (
        "detector_name", "pixel_size_x_m", "pixel_size_y_m", "energy_kev", "wavelength_angstrom", "distance_m",
        "beam_center_x_px", "beam_center_y_px"
    )})


__all__ = ["NXS_REJECT_MASK_BITS", "nxs_invalid_pixel_mask", "read_nexus_frame"]
