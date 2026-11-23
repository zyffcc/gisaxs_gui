"""Build and install the ordinary wheel and replay a public reduction outside the checkout."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="gimap-wheel-") as temporary:
        work = Path(temporary)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(GIMAP_HOME=str(work / "user"), QT_QPA_PLATFORM="offscreen",
                           MPLBACKEND="Agg", PYTHONNOUSERSITE="1")
        if os.name == "nt":
            environment["QT_QPA_FONTDIR"] = "C:/Windows/Fonts"

        def run(*args):
            subprocess.run(args, env=environment, cwd=work, check=True)

        run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", work, ROOT)
        wheel, = work.glob("gimap-*.whl")
        install = work / "venv"
        run(sys.executable, "-m", "venv", install)
        scripts = install / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        run(python, "-m", "pip", "install", wheel)
        run(python, "-m", "pip", "check")
        shutil.copytree(ROOT / "tests/data/external/gisaxs_galaxi", work / "sample")
        smoke = work / "wheel_smoke.py"
        shutil.copyfile(ROOT / "tests/wheel_smoke.py", smoke)
        run(python, "-I", smoke)
        mib = 1024**2
        size = sum(path.stat().st_size for path in install.rglob("*") if path.is_file())
        print(f"Wheel {wheel.stat().st_size / mib:.1f} MiB, fresh venv {size / mib:.1f} MiB")


if __name__ == "__main__":
    main()
