"""Compose Sputter: the scan header, the results root and Analyze (the client analysis package)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.gimap.integrations.state.user_store import user_data_dir

from .application import ClientService, CompareService, SputterService, StaticService
from .infrastructure import (CbfHeaderInspector, ClientFolderReader, ClientPackageRunner, CompareFolderFiles,
                             StaticPackageRunner, local_data_dir)


def default_results_root() -> Path:
    """``<user data folder>/sputter``: one folder per scan and geometry."""
    return user_data_dir() / "sputter"


def create_sputter_service(*, inspector=None, results_root: Optional[Callable[[], Path]] = None,
                           client=None) -> SputterService:
    """The real adapters unless others are given (tests give fakes). ``client``: Analyze, the client analysis package
    (by default the package found by ``infrastructure.client.locate_package``, its data and work folders in the user
    data folder)."""
    client = client or create_client_service()
    return SputterService(inspector or CbfHeaderInspector(), results_root or default_results_root, client,
                          CompareService(client, CompareFolderFiles()))


def default_series_root() -> Path:
    """``<user data folder>/sputter/static``: where a series of static frames goes when no folder is named."""
    return default_results_root() / "static"


def create_client_service(*, runner=None, reader=None, static_runner=None) -> ClientService:
    """Analyze: the package's process and the reader of its output folders (tests give fakes). Static frames run
    through the same package: ``static_runner``, by default made from a real ``runner`` (a fake one has none). The
    import caches go to the local folder (``local_data_dir``); those made before in the user data folder are used
    where they are."""
    runner = runner or ClientPackageRunner(local_data_dir, legacy_root=user_data_dir)  # caches: local, not roaming
    reader = reader or ClientFolderReader(user_data_dir)
    if static_runner is None and isinstance(runner, ClientPackageRunner):
        static_runner = StaticPackageRunner(runner, default_series_root)
    return ClientService(runner, reader, StaticService(static_runner, reader) if static_runner is not None else None)


__all__ = ["create_client_service", "create_sputter_service", "default_results_root", "default_series_root"]
