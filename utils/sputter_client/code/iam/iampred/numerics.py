'Deterministic numerics of the package (2026-11-26, results/iam/repro_20261126).'

from __future__ import annotations

import os
import platform
import sys

AVX512 = ("AVX512F", "AVX512CD", "AVX512_KNL", "AVX512_KNM", "AVX512_SKX", "AVX512_CLX", "AVX512_CNL", "AVX512_ICL")
_STATE: dict = {}


def _x86() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64", "x64", "i686", "i386")


def _avx2_fma() -> bool | None:
    'Whether the CPU has AVX2 and FMA3 (None: cannot tell). Read without NumPy (it must not be imported yet).'
    try:
        system = platform.system()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("flags"):
                        flags = set(line.split(":", 1)[1].split())
                        return "avx2" in flags and "fma" in flags
            return None
        if system == "Windows":
            import ctypes

            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        if system == "Darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.leaf7_features", "machdep.cpu.features"],
                                 capture_output=True, text=True, timeout=5).stdout.upper()
            return "AVX2" in out and "FMA" in out
    except Exception:
        return None
    return None


def apply() -> dict:
    'Enter the deterministic mode (once per process); returns the state that describe() reports.'
    if _STATE:
        return _STATE
    mode = os.environ.get("IAMPRED_NUMERICS", "det").strip().lower() or "det"
    late = "numpy" in sys.modules
    set_now = {}
    if mode != "native" and _x86():
        if "NPY_DISABLE_CPU_FEATURES" not in os.environ:
            os.environ["NPY_DISABLE_CPU_FEATURES"] = ",".join(AVX512)
            set_now["NPY_DISABLE_CPU_FEATURES"] = os.environ["NPY_DISABLE_CPU_FEATURES"]
        cpu = _avx2_fma()
        if cpu:
            for k, v in (("OPENBLAS_CORETYPE", "Haswell"), ("MKL_CBWR", "AVX2")):
                if k not in os.environ:
                    os.environ[k] = v
                    set_now[k] = v
        _STATE.update(avx2_fma=cpu)
    _STATE.update(mode=mode, numpy_imported_before=late, set=set_now, machine=platform.machine(),
                  system=platform.system())
    return _STATE


def describe() -> dict:
    'The numerical mode of this process: what apply() set, the variables in effect, the NumPy version.'
    st = dict(apply())
    st["env"] = {k: os.environ.get(k) for k in ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "MKL_CBWR",
                                                 "OMP_NUM_THREADS", "IAMPRED_NUMERICS") if os.environ.get(k) is not None}
    try:
        import numpy as np
        st["numpy"] = np.__version__
    except Exception:
        pass
    det = st["mode"] != "native" and not st["numpy_imported_before"]
    if not _x86():
        st["deterministic"] = None
        st["note"] = ("not x86-64: no CPU-dependent path is fixed here; bit-identical results are expected between "
                      "processors of the same kind with the same software")
    else:
        st["deterministic"] = bool(det and st.get("avx2_fma"))
        st["note"] = ("one NumPy SIMD path and one BLAS kernel on x86-64 CPUs with AVX2 (results/iam/repro_20261126)"
                      if st["deterministic"] else
                      "native numerics: results can differ in the last bit between CPU types and grow in the refinement")
    return st
