"""Run the external sputter toolkit (its own Python with PyTorch) as a subprocess.

The toolkit folder holds ``scripts/sputter_struct_v1.py``, the ``sputtergisaxs``
package, ``configs/`` and ``experts/polymer`` + ``experts/TS``. Progress comes
back as ``SPUTTER_PROGRESS`` lines on stdout; Cancel terminates the process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from ..application import AnalyzeRequest

ENTRY = Path("scripts") / "sputter_struct_v1.py"


class ToolkitNotConfigured(RuntimeError):
    pass


class SubprocessToolkit:
    """``settings()`` returns (python executable, toolkit folder); both are chosen on the Sputter page."""

    def __init__(self, settings: Callable[[], tuple[str, str]]):
        self.settings = settings

    def _resolve(self) -> tuple[str, Path]:
        python, folder = self.settings()
        python = python or os.environ.get("GIMAP_SPUTTER_PYTHON", "")
        folder = folder or os.environ.get("GIMAP_SPUTTER_TOOLKIT", "")
        if not python or not Path(python).is_file():
            raise ToolkitNotConfigured("Choose the Python executable of the sputter toolkit (an environment with "
                                       "PyTorch, NumPy, SciPy, fabio and Matplotlib).")
        if not folder or not (Path(folder) / ENTRY).is_file():
            raise ToolkitNotConfigured(f"Choose the sputter toolkit folder (it contains {ENTRY.as_posix()}).")
        return python, Path(folder)

    def _run(self, args: list[str], on_line: Callable[[str], None], should_stop: Callable[[], bool]) -> list[str]:
        python, folder = self._resolve()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        process = subprocess.Popen([python, "-u", str(folder / ENTRY), *args], cwd=str(folder), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env,
                                   creationflags=flags)
        tail: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\r\n")
            if not line or "FutureWarning" in line or line.lstrip().startswith("warnings.warn"):
                continue
            tail = (tail + [line])[-40:]
            on_line(line)
            if should_stop():
                process.terminate()
                break
        code = process.wait()
        if should_stop():
            raise RuntimeError("Stopped. Finished steps are kept in the output folder and are reused on the next run.")
        if code != 0:
            raise RuntimeError("The sputter toolkit stopped with an error:\n" + "\n".join(tail[-12:]))
        return tail

    def analyze(self, request: AnalyzeRequest, on_line: Callable[[str], None], should_stop: Callable[[], bool]) -> Path:
        _, folder = self._resolve()
        args = ["analyze", "--input", str(request.scan_folder), "--out", str(request.out_folder),
                "--sdd-mm", repr(float(request.sdd_mm)), "--alpha-i-deg", repr(float(request.alpha_i_deg)),
                "--metal", request.metal, "--polymer-type", request.polymer_type or "polymer",
                "--sample-roll-deg", repr(float(request.sample_roll_deg)), "--bin-seconds", repr(float(request.bin_seconds)),
                "--device", request.device, "--workers", str(int(request.workers)),
                "--polymer-expert", str(folder / "experts" / "polymer"), "--ts-expert", str(folder / "experts" / "TS")]
        args += ["--energy-ev", repr(float(request.energy_ev))] if request.energy_ev is not None else \
            ["--wavelength-nm", repr(float(request.wavelength_nm))]
        if request.beam_col is not None and request.beam_row is not None:
            args += ["--beam-col", repr(float(request.beam_col)), "--beam-row", repr(float(request.beam_row))]
        if request.mask_file:
            args += ["--mask", str(request.mask_file)]
        if request.refined_folder:
            args += ["--refined", str(request.refined_folder)]
        self._run(args, on_line, should_stop)
        return Path(request.out_folder) / "structure.csv"

    def table(self, binned_folder: Path, out_folder: Path, metal: str, refined_folder: Optional[Path],
              on_line: Callable[[str], None]) -> Path:
        args = ["table", "--binned", str(binned_folder), "--out", str(out_folder), "--metal", metal]
        if refined_folder:
            args += ["--refined", str(refined_folder)]
        self._run(args, on_line, lambda: False)
        return Path(out_folder) / "structure.csv"


__all__ = ["SubprocessToolkit", "ToolkitNotConfigured", "ENTRY"]
