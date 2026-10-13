"""Read-only look at the files around a frame, for calibration material.

Searched, in this order: the frame's folder and its subfolders, the parent
folder and every sibling folder (calibration, log and standard folders
first), the folder two levels up with its subfolders, and three levels up
the calibration / log folders.  ``search`` looks through any one folder the
assistant names (up and down from the frame, or a folder the user gave).
Every search stops after a fixed number of directory entries or seconds, so a
network drive or a folder with 10⁵ frames cannot stall a run.  Only files whose
names look like calibration material or logs are returned, unless a name
filter is given.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from src.gimap.shared.detector_io import load_detector_image

from ..domain import READABLE_SUFFIXES, classify, private_name, suffix, tokens

MAX_ENTRIES = 120_000
MAX_SECONDS = 15.0
FOLDER_WORDS = (
    "calib", "kalib", "poni", "standard", "agbh", "agbe", "behenate", "lab6", "ceo2", "log", "meta",
    "setup", "param", "config", "detector", "geometry", "info", "notes", "beamtime", "scan", "fio",
)
FOLDER_TOKENS = ("cal", "std", "logs", "raw", "processed", "shared", "data")


def interesting_folder(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in FOLDER_WORDS) or any(word in FOLDER_TOKENS for word in tokens(name))


class LocalFileExplorer:
    def __init__(self, *, max_entries: int = MAX_ENTRIES, max_seconds: float = MAX_SECONDS):
        self.max_entries = int(max_entries)
        self.max_seconds = float(max_seconds)

    # -- search ------------------------------------------------------------------------

    def _plan(self, frame: Path) -> list[tuple[Path, str, int]]:
        """(folder, how it relates to the frame, how deep to look below it)."""
        folder = frame.parent
        plan = [(folder, "same folder", 3)]
        parent = folder.parent
        if parent == folder:
            return plan
        plan.append((parent, "parent folder", 0))
        siblings = [item for item in self._subfolders(parent) if item != folder]
        for sibling in sorted(siblings, key=lambda item: not interesting_folder(item.name)):
            plan.append((sibling, f"sibling folder {sibling.name}", 3 if interesting_folder(sibling.name) else 2))
        grand = parent.parent
        if grand == parent:
            return plan
        plan.append((grand, "two levels up", 0))
        uncles = [item for item in self._subfolders(grand) if item != parent]
        for uncle in sorted(uncles, key=lambda item: not interesting_folder(item.name)):
            plan.append((uncle, f"folder {uncle.name} two levels up", 3 if interesting_folder(uncle.name) else 1))
        great = grand.parent
        if great != grand:
            plan.append((great, "three levels up", 0))
            for other in self._subfolders(great):
                if other != grand and interesting_folder(other.name):
                    plan.append((other, f"folder {other.name} three levels up", 2))
        return plan

    @staticmethod
    def _subfolders(folder: Path) -> list[Path]:
        try:
            with os.scandir(folder) as entries:
                return sorted(
                    Path(entry.path) for entry in entries
                    if entry.is_dir(follow_symlinks=False) and not private_name(entry.name)
                )
        except OSError:
            return []

    def _walk(self, folder: Path, depth: int, prefix: str, budget: dict) -> Iterator[tuple[os.DirEntry, str]]:
        try:
            with os.scandir(folder) as iterator:
                entries = list(iterator)
        except OSError:
            return
        folders = []
        for entry in entries:
            budget["seen"] += 1
            if budget["seen"] > self.max_entries or time.monotonic() > budget["deadline"]:
                budget["truncated"] = True
                return
            try:
                if private_name(entry.name):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    folders.append(entry)
                elif entry.is_file():
                    yield entry, f"{prefix}/{entry.name}"
            except OSError:
                continue
        if depth <= 0:
            return
        for entry in sorted(folders, key=lambda item: (not interesting_folder(item.name), item.name)):
            yield from self._walk(Path(entry.path), depth - 1, f"{prefix}/{entry.name}", budget)
            if budget["truncated"]:
                return

    def _collect(self, plan, budget: dict, keep) -> tuple[list[dict], list[str]]:
        seen_paths: set[str] = set()
        entries, folders = [], []
        for folder, relation, depth in plan:
            folders.append(f"{relation}: {folder}")
            for entry, relative in self._walk(folder, depth, folder.name, budget):
                key = os.path.normcase(entry.path)
                if key in seen_paths or not keep(entry.name, relative):
                    continue
                seen_paths.add(key)
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                entries.append({
                    "path": entry.path,
                    "relative": relative,
                    "size": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                    "relation": relation if relative.count("/") <= 1 else f"{relation}, subfolder {relative.rsplit('/', 1)[0]}",
                })
            if budget["truncated"]:
                break
        return entries, folders

    def _budget(self) -> dict:
        return {"seen": 0, "truncated": False, "deadline": time.monotonic() + self.max_seconds}

    def related_files(self, frame_path: str) -> dict:
        frame = Path(frame_path).resolve()
        budget = self._budget()
        entries, folders = self._collect(self._plan(frame), budget, lambda _name, relative: classify(relative) is not None)
        try:
            frame_stat = frame.stat()
            frame_info = {"path": str(frame), "mtime": float(frame_stat.st_mtime), "size": int(frame_stat.st_size)}
        except OSError:
            frame_info = {"path": str(frame)}
        return {"frame": frame_info, "entries": entries, "folders": folders, "truncated": budget["truncated"]}

    def search(self, folder: str, name_contains: Optional[str] = None, max_depth: int = 4) -> dict:
        """Files under ``folder``: calibration material and logs, or any readable file whose name contains ``name_contains``."""
        root = Path(folder).resolve()
        needle = (name_contains or "").strip().lower()

        def keep(name: str, relative: str) -> bool:
            if needle:
                return needle in name.lower() and suffix(name) in READABLE_SUFFIXES
            return classify(relative) is not None

        budget = self._budget()
        depth = max(0, min(int(max_depth), 8))
        entries, folders = self._collect([(root, f"in {root.name or root}", depth)], budget, keep)
        return {"entries": entries, "folders": folders, "truncated": budget["truncated"]}

    @staticmethod
    def kind(path: str) -> Optional[str]:
        """``file``, ``folder`` or ``None`` when nothing is there."""
        try:
            target = Path(path)
            if target.is_file():
                return "file"
            if target.is_dir():
                return "folder"
        except (OSError, ValueError):
            return None
        return None

    # -- reading -----------------------------------------------------------------------

    @staticmethod
    def read_text(path: str, max_bytes: int = 65536) -> str:
        with open(path, "rb") as stream:
            data = stream.read(int(max_bytes))
        text = data.decode("utf-8", errors="replace")
        if text.count("�") > max(8, len(text) // 50):
            text = data.decode("latin-1")
        return text

    @staticmethod
    def image_header(path: str) -> dict:
        image = load_detector_image(path)
        metadata = image.metadata or {}
        data = np.asarray(image.data)

        def scaled(value, factor: float) -> Optional[float]:
            return float(value) * factor if isinstance(value, (int, float)) and np.isfinite(value) else None

        values = {
            "shape": [int(value) for value in data.shape[:2]],
            "detector": image.detector_name,
            "pixel_size_um": [scaled(image.pixel_size_x_m, 1e6), scaled(image.pixel_size_y_m, 1e6)],
            "energy_kev": scaled(image.energy_kev, 1.0),
            "wavelength_angstrom": scaled(image.wavelength_angstrom, 1.0) or scaled(metadata.get("header_wavelength_angstrom"), 1.0),
            "distance_mm": scaled(image.distance_m, 1e3) or scaled(metadata.get("header_distance_m"), 1e3),
            "beam_center_px": (
                [float(image.beam_center_x_px), float(image.beam_center_y_px)]
                if image.beam_center_x_px is not None and image.beam_center_y_px is not None else None
            ),
            "exposure_s": scaled(metadata.get("exposure_time_s"), 1.0),
            "timestamp": metadata.get("timestamp"),
            "format": metadata.get("format") or Path(path).suffix.lower().lstrip("."),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))),
        }
        return {key: value for key, value in values.items() if value not in (None, [None, None])}


__all__ = ["LocalFileExplorer", "interesting_folder"]
