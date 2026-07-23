from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .geometry_model import energy_to_wavelength
from .detector_metadata import extract_cbf_metadata, extract_nxs_metadata
from .models import DetectorImage


DEFAULT_DATASET_PATH = "/entry/instrument/detector/data"
DEFAULT_MASK_PATH = "/entry/instrument/detector/pixel_mask"
DEFAULT_TRANSLATION_PATH = "/entry/instrument/detector/translation/distance"


class AmbiguousDatasetError(ValueError):
    def __init__(self, paths: list[str]):
        self.paths = paths
        super().__init__("Multiple detector image datasets are plausible: " + ", ".join(paths))


def _as_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.item() if hasattr(value, "item") else value


def _dataset_candidates(handle: h5py.File) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset) or obj.ndim not in (2, 3):
            return
        if not np.issubdtype(obj.dtype, np.number) or min(obj.shape[-2:]) < 32:
            return
        lower = name.lower()
        score = int(np.prod(obj.shape[-2:]) > 100_000)
        score += 20 if "/instrument/detector/data" in lower else 0
        score += 8 if lower.endswith("/data") else 0
        score += 4 if "detector" in lower else 0
        score -= 10 if any(token in lower for token in ("mask", "flatfield", "variance", "error")) else 0
        candidates.append((score, "/" + name.lstrip("/")))

    handle.visititems(visit)
    return sorted(candidates, key=lambda item: (-item[0], item[1]))


def select_nxs_dataset(handle: h5py.File, dataset_path: Optional[str] = None) -> str:
    if dataset_path and dataset_path in handle:
        return dataset_path
    if DEFAULT_DATASET_PATH in handle:
        return DEFAULT_DATASET_PATH
    candidates = _dataset_candidates(handle)
    if not candidates:
        raise ValueError("No usable detector image dataset was found in this NXS file.")
    best_score = candidates[0][0]
    best = [path for score, path in candidates if score == best_score]
    if len(best) > 1:
        raise AmbiguousDatasetError(best)
    return best[0]


def detect_nxs_frame_count(file_path: str | Path, dataset_path: Optional[str] = None) -> int:
    path = Path(file_path)
    if path.suffix.lower() != ".nxs":
        return 1
    with h5py.File(str(path), "r") as handle:
        selected = select_nxs_dataset(handle, dataset_path)
        dataset = handle[selected]
        return int(dataset.shape[0]) if dataset.ndim == 3 else 1


def _series_paths(path: Path) -> list[Path]:
    match = re.search(r"_m(\d+)\.nxs$", path.name, re.IGNORECASE)
    separator = "_"
    if match is None:
        match = re.search(r"m(\d+)\.nxs$", path.name, re.IGNORECASE)
        separator = ""
    if match is None:
        return [path]
    prefix = path.name[: match.start()]
    series: list[tuple[int, Path]] = []
    for sibling in path.parent.glob(f"{prefix}{separator}m*.nxs"):
        number = re.search(r"m(\d+)\.nxs$", sibling.name, re.IGNORECASE)
        if number:
            series.append((int(number.group(1)), sibling))
    return [item[1] for item in sorted(series)] or [path]


def _read_nxs(path: Path, frame_idx: int, dataset_path: Optional[str]) -> DetectorImage:
    paths = _series_paths(path)
    with h5py.File(str(paths[0]), "r") as handle:
        selected = select_nxs_dataset(handle, dataset_path)
        dataset = handle[selected]
        module_x, module_y = map(int, dataset.shape[-2:])
        metadata = extract_nxs_metadata(handle)

    modules: list[dict[str, Any]] = []
    for module_path in paths:
        with h5py.File(str(module_path), "r") as handle:
            if selected not in handle:
                raise ValueError(f"Detector dataset {selected} is missing in {module_path.name}.")
            dataset = handle[selected]
            if dataset.ndim == 3:
                safe_frame = max(0, min(int(frame_idx), int(dataset.shape[0]) - 1))
                image = dataset[safe_frame].astype(np.float32)
            else:
                safe_frame = 0
                image = dataset[()].astype(np.float32)
            mask = None
            if DEFAULT_MASK_PATH in handle:
                mask_data = handle[DEFAULT_MASK_PATH]
                if mask_data.ndim == 3:
                    mask = mask_data[min(safe_frame, mask_data.shape[0] - 1)] != 0
                elif mask_data.ndim == 2:
                    mask = mask_data[()] != 0
            translation = handle[DEFAULT_TRANSLATION_PATH][()] if DEFAULT_TRANSLATION_PATH in handle else (0, 0)
            flat_translation = np.asarray(translation).reshape(-1)
            tx = int(flat_translation[1]) if flat_translation.size > 1 else 0
            ty = int(flat_translation[0]) if flat_translation.size > 0 else 0

        if image.shape == (module_y, module_x):
            image = image.T
            mask = mask.T if mask is not None else None
        elif image.shape != (module_x, module_y):
            raise ValueError(f"{module_path.name}: unexpected module shape {image.shape}.")
        if mask is not None:
            image[mask] = np.nan
        modules.append({"image": image, "mask": mask, "tx": tx, "ty": ty})

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

    # This is the exact transpose + vertical flip used by the embedded GIWAXS page.
    data = np.flipud(grid.T).astype(np.float32, copy=False)
    final_mask = np.flipud(invalid.T)
    metadata.update({
        "format": "nxs",
        "dataset_path": selected,
        "frame_index": int(frame_idx),
        "module_files": [str(item) for item in paths],
        "transformations": ["module transpose when required", "stitched using P03 translations", "canvas transpose", "vertical flip"],
        "mask_semantics": "True is invalid",
    })
    if metadata["wavelength_angstrom"] is None and metadata["energy_kev"]:
        metadata["wavelength_angstrom"] = energy_to_wavelength(metadata["energy_kev"])
    return DetectorImage(data=data, mask=final_mask, source_path=path, metadata=metadata, **{key: metadata[key] for key in (
        "detector_name", "pixel_size_x_m", "pixel_size_y_m", "energy_kev", "wavelength_angstrom", "distance_m", "beam_center_x_px", "beam_center_y_px"
    )})


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
                with h5py.File(str(companion), "r") as handle:
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


def _read_tiff(path: Path) -> DetectorImage:
    try:
        from PIL import Image
        with Image.open(path) as image:
            data = np.asarray(image)
    except Exception:
        import matplotlib.pyplot as plt
        data = plt.imread(str(path))
    if data.ndim == 3:
        data = np.mean(data[..., :3], axis=2)
    data = np.asarray(data, dtype=np.float32)
    return DetectorImage(data, ~np.isfinite(data), path, metadata={"format": "tiff", "transformations": []})


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
    if suffix == ".nxs":
        return _read_nxs(source, frame_idx, dataset_path)
    if suffix == ".cbf":
        return _read_cbf(source)
    if suffix in {".tif", ".tiff"}:
        return _read_tiff(source)
    raise ValueError("Unsupported calibration image. Select an .nxs or .cbf file.")


def dump_metadata(image: DetectorImage) -> str:
    return json.dumps(image.metadata, indent=2, default=str)
