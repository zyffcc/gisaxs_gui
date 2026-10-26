"""Compose Sputter: the service with the bundled toolkit, PyTorch's installer, the scan header, the readers and the
island height from the q_z fringes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.gimap.integrations.state.user_store import user_data_dir

from .application import FringeService, SputterService
from .infrastructure import (BinnedWindowReader, BundledToolkit, CbfHeaderInspector, FringeFolderStore, FringeTool,
                             PipInstaller, StructureFolderReader)


def default_results_root() -> Path:
    """``<user data folder>/sputter``: one folder per scan and request."""
    return user_data_dir() / "sputter"


def create_sputter_service(*, runner=None, installer=None, inspector=None, reader=None, windows=None,
                           results_root: Optional[Callable[[], Path]] = None, fringes=None) -> SputterService:
    """The real adapters unless others are given (tests give fakes). ``fringes``: the island height from the q_z
    fringes (by default the tool bundled in ``utils/sputter_fringes``; outputs the result folder cannot take go to the
    user data folder)."""
    return SputterService(runner or BundledToolkit(), reader or StructureFolderReader(), installer or PipInstaller(),
                          inspector or CbfHeaderInspector(), results_root or default_results_root,
                          windows or BinnedWindowReader(),
                          fringes or FringeService(FringeTool(), FringeFolderStore(), user_data_dir))


__all__ = ["create_sputter_service", "default_results_root"]
