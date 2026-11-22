"""File identities from the particular frames that were used, including multi-file detector sources."""

from pathlib import Path
from typing import Any, Mapping, Optional

from src.gimap.shared.reproduction import file_reference

from ..domain.mask_edits import MaskImage


def source_files(path: str | Path, metadata: Optional[Mapping[str, Any]]) -> list[str]:
    """Additional paths for a module set or a master with its data files."""
    metadata = metadata or {}
    modules = [str(item) for item in metadata.get("module_files") or ()]
    if modules:
        return modules
    data = [str(item) for item in metadata.get("data_files") or ()]
    return [str(path), *data] if data else []


def frame_references(path, metadata, known=None):
    paths = dict.fromkeys([str(path), *source_files(path, metadata)])
    return {key: file_reference(key) for key in paths if known is None or key not in known}


def image_files(path, frames):
    describe = getattr(frames, "describe", None)
    layout = describe(path) if describe else None
    return tuple(dict.fromkeys([str(path), *(str(part) for part in layout.members)])) if layout else (str(path),)


def image_stamp(path, frames):
    """A changed module or linked data file invalidates the cached correction image as well as its digest."""
    stamp = []
    for name in image_files(path, frames):
        stat = Path(name).stat()
        stamp.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(stamp)


def correction_references(corrections, frames):
    images = [corrections.dark_path, corrections.background_path, corrections.flat_field_path, corrections.mask_path]
    if corrections.error_model == "sigma":
        images.append(corrections.sigma_path)
    paths = []
    for edit in corrections.mask_shapes:
        paths.extend(edit.sources)
        if isinstance(edit, MaskImage):
            images.append(edit.path)
    for path in dict.fromkeys(filter(None, images)):
        paths.extend(image_files(path, frames))
    return {key: file_reference(key) for key in dict.fromkeys(paths)}
