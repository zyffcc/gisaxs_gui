"""Compose Sputter: the service with the bundled toolkit, PyTorch's installer, the scan header, the readers, the
island height from the q_z fringes and Analyze (the client analysis package)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.gimap.integrations.state.user_store import user_data_dir

from .application import ClientService, FringeService, SputterService, StaticService
from .infrastructure import (BinnedWindowReader, BundledToolkit, CbfHeaderInspector, ClientFolderReader,
                             ClientPackageRunner, FringeFolderStore, FringeTool, PipInstaller, StaticPackageRunner,
                             StructureFolderReader)


def default_results_root() -> Path:
    """``<user data folder>/sputter``: one folder per scan and request."""
    return user_data_dir() / "sputter"


def create_sputter_service(*, runner=None, installer=None, inspector=None, reader=None, windows=None,
                           results_root: Optional[Callable[[], Path]] = None, fringes=None,
                           client=None) -> SputterService:
    """The real adapters unless others are given (tests give fakes). ``fringes``: the island height from the q_z
    fringes (by default the tool bundled in ``utils/sputter_fringes``; outputs the result folder cannot take go to the
    user data folder). ``client``: Analyze, the client analysis package (by default the package found by
    ``infrastructure.client.locate_package``, its data and work folders in the user data folder)."""
    return SputterService(runner or BundledToolkit(), reader or StructureFolderReader(), installer or PipInstaller(),
                          inspector or CbfHeaderInspector(), results_root or default_results_root,
                          windows or BinnedWindowReader(),
                          fringes or FringeService(FringeTool(), FringeFolderStore(), user_data_dir),
                          client or create_client_service())


def default_series_root() -> Path:
    """``<user data folder>/sputter/static``: where a series of static frames goes when no folder is named."""
    return default_results_root() / "static"


def create_client_service(*, runner=None, reader=None, static_runner=None) -> ClientService:
    """Analyze: the package's process and the reader of its output folders (tests give fakes). Static frames run
    through the same package: ``static_runner``, by default made from a real ``runner`` (a fake one has none)."""
    runner = runner or ClientPackageRunner(user_data_dir)
    reader = reader or ClientFolderReader(user_data_dir)
    if static_runner is None and isinstance(runner, ClientPackageRunner):
        static_runner = StaticPackageRunner(runner, default_series_root)
    return ClientService(runner, reader, StaticService(static_runner, reader) if static_runner is not None else None)


__all__ = ["create_client_service", "create_sputter_service", "default_results_root", "default_series_root"]
