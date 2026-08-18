"""Local cache adapter for cloud and network detector files."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from src.gimap.shared.file_paths import normalize_path


REMOTE_PATH_MARKERS = (
    "onedrive",
    "dropbox",
    "google drive",
    "googledrive",
    "iclouddrive",
    "icloud",
    "nextcloud",
    "owncloud",
)


class LocalRemoteFileCacheAdapter:
    def __init__(self, project_root: Path):
        self._project_root = Path(project_root).resolve()

    def default_directory(self) -> str:
        return os.path.join(".gimap_cache", "remote_files")

    def resolve_directory(self, cache_dir: str) -> Path:
        value = str(cache_dir or self.default_directory()).strip()
        if not value:
            value = self.default_directory()
        if os.path.isabs(value):
            return Path(normalize_path(value))
        return Path(normalize_path(str(self._project_root / value)))

    def display_directory(self, cache_dir: str) -> str:
        value = str(cache_dir or "").strip()
        default_relative = self.default_directory()
        default_absolute = self.resolve_directory(default_relative)
        if not value:
            return default_relative
        try:
            if os.path.normcase(os.path.abspath(value)) == os.path.normcase(
                os.path.abspath(default_absolute)
            ):
                return default_relative
        except Exception:
            pass
        return value

    def is_remote(self, path: str) -> bool:
        try:
            if not path:
                return False
            text = normalize_path(str(path))
            if text.startswith("\\\\") or text.startswith("//"):
                return True
            lowered = text.replace("\\", "/").lower()
            return any(marker in lowered for marker in REMOTE_PATH_MARKERS) or (
                self._is_mapped_network_drive(text)
            )
        except Exception:
            return False

    def target_path(self, source_path: str, cache_dir: str) -> Path:
        source = normalize_path(source_path)
        digest = hashlib.sha256(
            source.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        name = os.path.basename(source) or "remote_file"
        return self.resolve_directory(cache_dir) / f"{digest}_{name}"

    def prepare(
        self,
        source_path: str,
        cache_dir: str,
        max_gb: float,
        *,
        on_progress=None,
        is_cancelled=None,
    ) -> Path:
        source = Path(normalize_path(source_path))
        directory = self.resolve_directory(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = self.target_path(str(source), str(directory))
        temporary = Path(str(target) + ".part")
        try:
            source_stat = source.stat()
            if target.exists():
                target_stat = target.stat()
                if (
                    int(target_stat.st_size) == int(source_stat.st_size)
                    and int(target_stat.st_mtime) >= int(source_stat.st_mtime)
                ):
                    if on_progress:
                        on_progress(100, f"Using cached remote file: {source.name}")
                    return target
            total = max(1, int(source_stat.st_size))
            copied = 0
            with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
                while True:
                    if is_cancelled and is_cancelled():
                        raise RuntimeError("Remote file copy cancelled")
                    chunk = source_handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target_handle.write(chunk)
                    copied += len(chunk)
                    if on_progress:
                        percent = int(min(99, max(1, copied * 100 / total)))
                        on_progress(percent, f"Copying remote file... {percent}%")
            try:
                os.utime(temporary, (source_stat.st_atime, source_stat.st_mtime))
            except Exception:
                pass
            temporary.replace(target)
            if on_progress:
                on_progress(100, f"Cached remote file: {source.name}")
            self._enforce_limit(directory, max_gb)
            return target
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except Exception:
                    pass

    def clear(self, cache_dir: str) -> int:
        directory = self.resolve_directory(cache_dir)
        if not directory.is_dir():
            return 0
        removed = 0
        for entry in directory.iterdir():
            if not entry.is_file() or not (
                self._is_cache_name(entry.name) or entry.name.endswith(".part")
            ):
                continue
            try:
                entry.unlink()
                removed += 1
            except Exception:
                pass
        return removed

    @staticmethod
    def _is_cache_name(name: str) -> bool:
        return bool(re.match(r"^[0-9a-f]{16}_.+", name or "", re.IGNORECASE))

    @staticmethod
    def _is_mapped_network_drive(path: str) -> bool:
        try:
            if os.name != "nt":
                return False
            drive, _tail = os.path.splitdrive(os.path.abspath(path))
            if not drive:
                return False
            import ctypes

            return int(ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")) == 4
        except Exception:
            return False

    def _enforce_limit(self, directory: Path, max_gb: float) -> None:
        limit = int(max(0.25, float(max_gb or 3.0)) * 1024**3)
        files = []
        total = 0
        for entry in directory.iterdir():
            if (
                not entry.is_file()
                or entry.name.endswith(".part")
                or not self._is_cache_name(entry.name)
            ):
                continue
            try:
                stat = entry.stat()
            except Exception:
                continue
            files.append((stat.st_mtime, stat.st_size, entry))
            total += int(stat.st_size)
        for _modified, size, entry in sorted(files):
            if total <= limit:
                break
            try:
                entry.unlink()
                total -= int(size)
            except Exception:
                pass
