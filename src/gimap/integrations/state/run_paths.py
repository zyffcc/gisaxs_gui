"""Writable per-user output folders for subprocess analyses on Windows and macOS."""
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from .user_store import user_data_dir


def fitting_output_root() -> Path:
    path = user_data_dir() / "AI_Fitting_Output"
    path.mkdir(parents=True, exist_ok=True)
    # Fail before starting a long job if the user data folder is not writable.
    with NamedTemporaryFile(prefix=".write-check-", dir=path):
        pass
    return path


def fitting_run_directory() -> Path:
    return fitting_output_root() / ("v5_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
