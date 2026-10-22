"""PyTorch for the Sputter model: is it there, and installing it on the first Predict.

The model runs in GIMaP's own Python (its console ``python.exe``). When PyTorch is missing there, Predict
installs it with pip into that same environment (about 1.2 GB, once) before it runs the model. The installer
is a port (``EnvironmentInstaller``); its failures are ``InstallError`` with a kind and an English message
made from ``INSTALL_MESSAGES`` (the page translates the template and fills in the same values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

DOWNLOAD_SIZE = "about 1.2 GB"
STOP_NOW = "now"
"""What ``should_stop()`` returns when the window closes: pip is ended even while it writes the packages
(otherwise Stop waits for that step, so no half-installed PyTorch is left)."""

INSTALL_MESSAGES = {
    "network": ("PyTorch could not be downloaded: there is no connection to the Python package index (pypi.org). "
                "Connect to the internet or set the proxy (HTTPS_PROXY), then press Predict again. To install by "
                "hand: {command}"),
    "permission": ("PyTorch could not be installed: there is no permission to write into this Python ({prefix}). "
                   "Start GIMaP from a folder you can write to (Start_GIMaP.bat), or install by hand as an "
                   "administrator: {command}"),
    "disk": ("PyTorch could not be installed: the disk is full. It needs about 1.5 GB of free space ({prefix}); free "
             "some space, then press Predict again."),
    "no_match": ("pip found no PyTorch for this Python ({version}). Start GIMaP with Start_GIMaP.bat, which sets up a "
                 "Python that fits, or install PyTorch by hand: {command}"),
    "no_pip": ("This Python has no pip, and pip could not be added (ensurepip). Install pip for {python}, then press "
               "Predict again, or install by hand: {command}"),
    "import_failed": ("PyTorch is installed but does not load ({detail}). On Windows this usually means that the "
                      "Microsoft Visual C++ Redistributable is missing (install it from Microsoft), or that the "
                      "installation is damaged: reinstall it with {command}"),
    "frozen": "This GIMaP cannot install packages into itself. Install PyTorch by hand: {command}",
    "disabled": ("Installing PyTorch automatically is switched off (GIMAP_SPUTTER_NO_INSTALL=1). Install it by hand: "
                 "{command}"),
    "cancelled": "Stopped before PyTorch was installed. Press Predict to start again.",
    "other": "pip stopped with exit code {code}; its last lines are under Details. To install by hand: {command}",
}
INSTALL_STEP = "Step 1 of 2: installing PyTorch ({size}, the first time only)…"
INSTALL_STAGES = {
    "pip": "Preparing pip…",
    "resolve": "Looking for {package}…",
    "download": "Downloading {package} ({size})…",
    "download_progress": "Downloading {package}: {percent}%",
    "cached": "{package} is already downloaded",
    "install": "Installing the packages (this cannot be stopped)…",
    "done": "PyTorch is installed",
}
"""What the page says during each stage of the installation (templates; the page translates them)."""


@dataclass(frozen=True)
class InstallProgress:
    """One step of the installation: ``stage`` (a key of ``INSTALL_STAGES``), its values, the share done
    (``None``: unknown) and whether Stop still stops it (not while pip writes the packages)."""

    stage: str
    fraction: Optional[float] = None
    values: tuple = ()
    cancellable: bool = True

    def template(self) -> str:
        return INSTALL_STAGES.get(self.stage, "{package}")

    def fields(self) -> dict:
        return dict(self.values)


@dataclass(frozen=True)
class EnvironmentStatus:
    """The Python that runs the model and its PyTorch. ``problem``: "" (ready), "missing", "import_failed"."""

    python: str
    torch_version: Optional[str] = None
    numpy_version: Optional[str] = None
    problem: str = "missing"
    detail: str = ""

    @property
    def ready(self) -> bool:
        return not self.problem and bool(self.torch_version)


class InstallError(RuntimeError):
    """A failed or stopped installation: ``kind`` (a key of ``INSTALL_MESSAGES``), the message and pip's last lines."""

    def __init__(self, kind: str, message: str, details: str = ""):
        super().__init__(message)
        self.kind = kind
        self.details = details


def install_message(kind: str, **values) -> str:
    template = INSTALL_MESSAGES.get(kind, INSTALL_MESSAGES["other"])
    fields = {"command": "", "prefix": "", "version": "", "python": "", "detail": "", "code": ""} | values
    return template.format(**fields)


class EnvironmentInstaller(Protocol):
    python: str

    def status(self) -> EnvironmentStatus:
        """Quick look (no import, no subprocess): for the line under Predict."""
        ...

    def verify(self) -> EnvironmentStatus:
        """Import PyTorch in the model's Python (a subprocess)."""
        ...

    def install(self, on_progress: Callable[[InstallProgress], None], should_stop: Callable[[], bool],
                repair: bool = False) -> None:
        """Install the model's packages into the model's Python; ``InstallError`` when that fails or is stopped.
        ``repair``: install them again over a damaged installation (not their dependencies)."""
        ...

    def interrupted(self) -> bool:
        """An installation was cut off while pip wrote the packages (the window was closed)."""
        ...

    def manual_command(self, repair: bool = False) -> str:
        ...


__all__ = ["DOWNLOAD_SIZE", "STOP_NOW", "EnvironmentInstaller", "EnvironmentStatus", "INSTALL_MESSAGES", "INSTALL_STAGES",
           "INSTALL_STEP", "InstallError", "InstallProgress", "install_message"]
