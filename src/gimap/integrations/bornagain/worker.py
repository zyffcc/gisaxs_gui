"""仅由 Job worker process 调用的 BornAgain handlers。"""

from __future__ import annotations

from .api_24_1 import simulate_raw_24_1
from .errors import (
    BornAgainBrokenError,
    BornAgainNotInstalledError,
    BornAgainUnsupportedVersionError,
)
from .version import BornAgainVersion
from ..jobs import encode_array


REQUIRED_API = (
    "Beam",
    "Particle",
    "Sample",
    "ScatteringSimulation",
    "SphericalDetector",
)


def _import_supported_bornagain():
    try:
        import bornagain as ba
    except ModuleNotFoundError as exc:
        if exc.name == "bornagain":
            raise BornAgainNotInstalledError(
                "BornAgain is not installed in this Python environment."
            ) from exc
        raise BornAgainBrokenError(f"BornAgain dependency is missing: {exc}") from exc
    except Exception as exc:
        raise BornAgainBrokenError(f"BornAgain import failed: {exc}") from exc

    try:
        version = BornAgainVersion.parse(getattr(ba, "version", ""))
    except Exception as exc:
        raise BornAgainBrokenError(f"BornAgain version cannot be determined: {exc}") from exc
    if not version.supported:
        raise BornAgainUnsupportedVersionError(
            f"BornAgain {version} is unsupported; GIMaP currently supports 24.1.x."
        )
    missing = [name for name in REQUIRED_API if not hasattr(ba, name)]
    if not hasattr(ba, "ParticleLayout") and not hasattr(ba, "Dilute2D"):
        missing.append("ParticleLayout or Dilute2D")
    if missing:
        raise BornAgainBrokenError(
            "BornAgain 24.1 installation is incomplete; missing API: " + ", ".join(missing)
        )
    return ba, version


def probe_bornagain(_payload, _report, _is_cancelled):
    try:
        ba, version = _import_supported_bornagain()
    except BornAgainNotInstalledError as exc:
        return {"state": "not_installed", "message": str(exc)}
    except BornAgainUnsupportedVersionError as exc:
        raw = str(exc).split()[1] if len(str(exc).split()) > 1 else ""
        return {"state": "unsupported", "message": str(exc), "version": raw}
    except BornAgainBrokenError as exc:
        return {"state": "broken", "message": str(exc)}
    return {
        "state": "available",
        "message": f"BornAgain {version} is available.",
        "version": str(version),
        "module_path": str(getattr(ba, "__file__", "")),
    }


def simulate_bornagain(payload, report, is_cancelled):
    if is_cancelled():
        raise RuntimeError("BornAgain simulation cancelled before start.")
    ba, _version = _import_supported_bornagain()
    report(0, 1, "Running BornAgain 24.1 simulation")
    image = simulate_raw_24_1(ba, payload["config"], payload["sampled"])
    report(1, 1, "BornAgain simulation complete")
    return encode_array(image)
