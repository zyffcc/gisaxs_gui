"""The version of GIMaP: the one place it is written.

The About box, the error log and every export record that names the software read it from here, so a
file written by one release never carries the number of another.
"""

from __future__ import annotations

APP_NAME = "GIMaP"
__version__ = "0.2.0"
PRE_RELEASE = True
"""The release stage: ``True`` while this version is a pre-release (the About box then says “(Pre-release)”)."""
SOFTWARE = f"{APP_NAME} v{__version__}"
"""How an export record names the program that wrote it (``software_version``)."""

__all__ = ["APP_NAME", "PRE_RELEASE", "SOFTWARE", "__version__"]
