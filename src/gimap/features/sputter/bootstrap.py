"""Compose Sputter: the service with the bundled toolkit, PyTorch's installer, the scan header and the readers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.gimap.integrations.state.user_store import user_data_dir

from .application import SputterService
from .infrastructure import BinnedWindowReader, BundledToolkit, CbfHeaderInspector, PipInstaller, StructureFolderReader


def default_results_root() -> Path:
    """``<user data folder>/sputter``: one folder per scan and request."""
    return user_data_dir() / "sputter"


def create_sputter_service(*, runner=None, installer=None, inspector=None, reader=None, windows=None,
                           results_root: Optional[Callable[[], Path]] = None) -> SputterService:
    """The real adapters unless others are given (tests give fakes)."""
    return SputterService(runner or BundledToolkit(), reader or StructureFolderReader(), installer or PipInstaller(),
                          inspector or CbfHeaderInspector(), results_root or default_results_root,
                          windows or BinnedWindowReader())


__all__ = ["create_sputter_service", "default_results_root"]
