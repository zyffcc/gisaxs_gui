"""Application data in a wheel or beside the sources in an editable or portable install."""

from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = _PACKAGE / "_resources"
if not RESOURCE_ROOT.is_dir():
    RESOURCE_ROOT = _PACKAGE.parents[1]
