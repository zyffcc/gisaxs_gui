"""Install the Sputter model's packages (PyTorch) with pip into GIMaP's own Python.

``python -m pip install --disable-pip-version-check --no-input --no-user -c <numpy pinned> -r
requirements-sputter.txt`` in the environment GIMaP runs in:

* the installed NumPy version is pinned (a constraint file), so PyTorch does not replace GIMaP's NumPy;
* never ``--user``: the model runs with ``PYTHONNOUSERSITE=1`` and would not see packages there; a Python
  that cannot be written to fails with a permission message instead;
* a Python without pip (an environment made with ``uv venv``) gets it first with ``ensurepip`` (offline);
* in a source folder set up by ``Start_GIMaP.bat`` (``.gimap``) pip's cache stays inside ``.gimap``;
* progress comes from pip's lines (``pip_output``); Stop ends pip until it starts writing the packages, then
  waits for its end, so no half-installed PyTorch is left (closing the window stops it at once: a marker file
  beside the Python says so, and when PyTorch then does not load, the next Predict installs the packages of
  ``requirements-sputter.txt`` again over them, ``--force-reinstall --no-deps``; NumPy and the other
  dependencies are left alone, GIMaP has them loaded);
* ``GIMAP_SPUTTER_NO_INSTALL=1`` switches the installation off (tests, administrators).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from ..application import STOP_NOW, EnvironmentStatus, InstallError, InstallProgress, install_message
from .pip_output import PipReader, classify_failure
from .process import child_environment, console_python, run_streaming

PROJECT_ROOT = Path(__file__).resolve().parents[5]
REQUIREMENTS = PROJECT_ROOT / "requirements-sputter.txt"
NO_INSTALL = "GIMAP_SPUTTER_NO_INSTALL"
UNFINISHED = "gimap_sputter_install.unfinished"
"""Beside the model's Python while pip writes the packages; left there when that was cut off."""
REPAIR = ["--force-reinstall", "--no-deps"]
VERIFY_CODE = (
    "import json\n"
    "out = {}\n"
    "try:\n    import numpy\n    out['numpy'] = numpy.__version__\nexcept Exception as exc:\n"
    "    out['numpy_error'] = repr(exc)\n"
    "try:\n    import torch\n    out['torch'] = torch.__version__\n"
    "except ModuleNotFoundError as exc:\n    out['missing'] = str(exc)\n"
    "except Exception as exc:\n    out['error'] = type(exc).__name__ + ': ' + str(exc)\n"
    "print('GIMAP_VERIFY ' + json.dumps(out))\n"
)


def _version_tuple(text: str) -> tuple:
    parts = []
    for part in text.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class PipInstaller:
    def __init__(self, python: Optional[str] = None, requirements: Path = REQUIREMENTS, *, run=run_streaming,
                 call=subprocess.run):
        self.python = python or console_python()
        self.requirements = Path(requirements)
        self._run, self._call = run, call

    # -- looking -------------------------------------------------------------------------------------
    def manual_command(self, repair: bool = False) -> str:
        extra = " ".join(REPAIR) + " " if repair else ""
        return f'"{self.python}" -m pip install {extra}-r "{self.requirements}"'

    def _marker(self) -> Path:
        return Path(self.python).parent / UNFINISHED

    def interrupted(self) -> bool:
        return self._marker().is_file()

    def _same_environment(self) -> bool:
        return Path(self.python).resolve().parent == Path(console_python()).resolve().parent

    def status(self) -> EnvironmentStatus:
        """From this process (no import of PyTorch): is it installed in the environment, outside the user's site?"""
        if not self._same_environment():
            return EnvironmentStatus(self.python, problem="unknown")
        try:
            spec = importlib.util.find_spec("torch")
        except (ImportError, ValueError):
            spec = None
        user = site.getusersitepackages() if hasattr(site, "getusersitepackages") else ""
        origin = str(getattr(spec, "origin", "") or "")
        if spec is None or (user and origin.startswith(str(user))):
            return EnvironmentStatus(self.python, problem="missing")
        try:
            version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            version = "?"
        return EnvironmentStatus(self.python, torch_version=version, problem="")

    def _env(self) -> dict:
        extra = {"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}
        parts = Path(self.python).resolve().parts
        if ".gimap" in parts:  # a source folder's own environment: nothing is written outside the folder
            gimap = Path(*parts[:parts.index(".gimap") + 1])
            extra["PIP_CACHE_DIR"] = str(gimap / "cache" / "pip")
        return child_environment(**extra)

    def verify(self) -> EnvironmentStatus:
        """Import PyTorch in the model's Python, as the model will (a subprocess, ``PYTHONNOUSERSITE=1``)."""
        try:
            done = self._call([self.python, "-c", VERIFY_CODE], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=self._env(), timeout=300, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return EnvironmentStatus(self.python, problem="import_failed", detail=str(exc))
        found = {}
        for line in (done.stdout or "").splitlines():
            if line.startswith("GIMAP_VERIFY "):
                found = json.loads(line[len("GIMAP_VERIFY "):])
        if not found:
            tail = ((done.stderr or "") + (done.stdout or "")).strip().splitlines()[-3:]
            return EnvironmentStatus(self.python, problem="import_failed", detail=" ".join(tail) or "no answer")
        if found.get("torch"):
            return EnvironmentStatus(self.python, found["torch"], found.get("numpy"), problem="")
        if found.get("error"):
            return EnvironmentStatus(self.python, None, found.get("numpy"), problem="import_failed",
                                     detail=found["error"])
        return EnvironmentStatus(self.python, None, found.get("numpy"), problem="missing",
                                 detail=found.get("missing", ""))

    # -- installing ----------------------------------------------------------------------------------
    def _fail(self, kind: str, details: str = "", code: str = "") -> InstallError:
        return InstallError(kind, install_message(
            kind, command=self.manual_command(), prefix=str(Path(self.python).parent), python=self.python,
            version=f"Python {sys.version_info.major}.{sys.version_info.minor}, "
                    f"{64 if sys.maxsize > 2 ** 32 else 32}-bit", code=code), details)

    def _pip_version(self, env: dict) -> Optional[str]:
        try:
            done = self._call([self.python, "-m", "pip", "--version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=120, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return None
        words = (done.stdout or "").split()
        return words[1] if done.returncode == 0 and len(words) > 1 and words[0] == "pip" else None

    def install(self, on_progress: Callable[[InstallProgress], None], should_stop: Callable[[], object],
                repair: bool = False) -> None:
        if os.environ.get(NO_INSTALL, "").strip() == "1":
            raise self._fail("disabled")
        if getattr(sys, "frozen", False):
            raise self._fail("frozen")
        env = self._env()
        on_progress(InstallProgress("pip"))
        pip = self._pip_version(env)
        if pip is None:
            try:
                done = self._call([self.python, "-m", "ensurepip", "--upgrade", "--default-pip"], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", env=env, timeout=600,
                                  stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise self._fail("no_pip", f"ensurepip: {exc}") from exc
            pip = self._pip_version(env) if done.returncode == 0 else None
            if pip is None:
                raise self._fail("no_pip", ((done.stdout or "") + (done.stderr or ""))[-2000:])
        if should_stop():
            raise self._fail("cancelled")
        numpy_version = self.verify().numpy_version
        scratch = Path(tempfile.mkdtemp(prefix="gimap_sputter_pip_"))
        try:
            constraints = scratch / "constraints.txt"
            constraints.write_text(f"numpy=={numpy_version}\n" if numpy_version else "", encoding="utf-8")
            command = [self.python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-user",
                       *(REPAIR if repair else []), "-c", str(constraints), "-r", str(self.requirements)]
            if _version_tuple(pip) >= (24, 1):
                command += ["--progress-bar", "raw"]
            reader = PipReader()

            def line(text: str) -> None:
                progress = reader.read(text)
                if progress is not None:
                    on_progress(progress)

            marker = self._marker()
            try:
                marker.write_text("pip install " + " ".join(command[3:]) + "\n", encoding="utf-8")
            except OSError:  # an environment that cannot be written to fails in pip with its own message
                pass
            outcome = self._run(command, on_line=line, should_stop=should_stop, env=env,
                                can_stop=lambda: not reader.installing or should_stop() == STOP_NOW)
            if not (outcome.stopped and reader.installing):  # only a cut-off writing leaves the marker
                marker.unlink(missing_ok=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if outcome.stopped:
            raise self._fail("cancelled")
        if outcome.code != 0:
            raise self._fail(classify_failure(outcome.tail), "\n".join(outcome.tail[-12:]), code=str(outcome.code))
        on_progress(InstallProgress("done", 1.0, cancellable=False))


__all__ = ["NO_INSTALL", "PipInstaller", "REPAIR", "REQUIREMENTS", "UNFINISHED", "VERIFY_CODE"]
